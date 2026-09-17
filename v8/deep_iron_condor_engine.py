from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / 'v8/config.yaml').read_text())
OUT = ROOT / 'v8/research'; OUT.mkdir(parents=True, exist_ok=True)
SEED = int(CFG['seed']); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(2)
Q = np.asarray(CFG['model']['quantiles'], dtype=np.float32)

def clean_dates(df, col='date'):
    x=df.copy(); x[col]=pd.to_datetime(x[col],errors='coerce').dt.tz_localize(None).dt.normalize(); return x.dropna(subset=[col]).sort_values(col)

def load_inputs():
    options=pd.read_parquet(ROOT/'data/cache/nifty_options_long.parquet'); futures=pd.read_parquet(ROOT/'data/cache/nifty_futures_long.parquet'); context=pd.read_parquet(ROOT/'data/cache/v5_market_context.parquet'); news=pd.read_csv(ROOT/'data/cache/v5_news_daily.csv'); index=clean_dates(pd.read_parquet(ROOT/'data/cache/v62_nifty50_index.parquet')); prices=pd.read_parquet(ROOT/'data/cache/v62_constituent_prices.parquet'); weights=pd.read_csv(ROOT/'data/cache/v62_constituent_weights.csv'); membership=pd.read_csv(ROOT/'data/cache/v62_nifty50_membership.csv'); flows=pd.read_parquet(ROOT/'data/cache/v62_fii_dii.parquet')
    for c in ['open','high','low','close','volume']:
        if c in index: index[c]=pd.to_numeric(index[c],errors='coerce')
    index=index.dropna(subset=['close']).drop_duplicates('date',keep='last')
    for c in ['date','expiry']:
        options[c]=pd.to_datetime(options[c],errors='coerce').dt.tz_localize(None).dt.normalize(); futures[c]=pd.to_datetime(futures[c],errors='coerce').dt.tz_localize(None).dt.normalize()
    options['option_type']=options['option_type'].astype(str).str.upper();
    for c in ['strike','open_interest','volume','close']: options[c]=pd.to_numeric(options[c],errors='coerce')
    options=options.dropna(subset=['date','expiry','strike','close']); options=options[options['close']>0].copy(); futures=futures.dropna(subset=['date','expiry']).copy()
    for c in ['close','open_interest','volume']:
        if c in futures: futures[c]=pd.to_numeric(futures[c],errors='coerce')
    context=clean_dates(context); context['series']=context['series'].astype(str).str.strip(); context['close']=pd.to_numeric(context['close'],errors='coerce')
    news.columns=[str(c).strip().lower() for c in news.columns]; news=clean_dates(news)
    for c in news.columns:
        if c!='date': news[c]=pd.to_numeric(news[c],errors='coerce')
    prices.columns=[str(c).strip().lower() for c in prices.columns]; prices['symbol']=prices['symbol'].astype(str).str.upper(); prices['date']=pd.to_datetime(prices['date'],errors='coerce').dt.tz_localize(None).dt.normalize(); prices['close']=pd.to_numeric(prices['close'],errors='coerce'); prices=prices.dropna(subset=['symbol','date','close']).drop_duplicates(['symbol','date'],keep='last')
    weights.columns=[str(c).strip().upper() for c in weights.columns]; weights['DATE']=pd.to_datetime(weights['DATE'],errors='coerce').dt.tz_localize(None).dt.normalize()
    for c in weights.columns:
        if c!='DATE': weights[c]=pd.to_numeric(weights[c],errors='coerce').fillna(0.0)
    membership.columns=[str(c).strip().lower() for c in membership.columns]; membership['symbol']=membership['symbol'].astype(str).str.upper(); membership['valid_from']=pd.to_datetime(membership['valid_from'],errors='coerce').dt.tz_localize(None).dt.normalize(); membership['valid_to']=pd.to_datetime(membership['valid_to'],errors='coerce').dt.tz_localize(None).dt.normalize()
    flows['date']=pd.to_datetime(flows['date'],errors='coerce').dt.tz_localize(None).dt.normalize()
    for c in ['fii_net','dii_net']:
        if c in flows: flows[c]=pd.to_numeric(flows[c],errors='coerce')
    return options,futures,context,news,index,prices,weights,membership,flows

def strict_asof(left_dates, right, date_col='date'):
    """Strict point-in-time backward lookup without pandas merge_asof dtype fragility."""
    left = pd.DatetimeIndex(pd.to_datetime(left_dates, errors='coerce'))
    if left.tz is not None:
        left = left.tz_localize(None)
    left = left.normalize()
    r = right.copy()
    r[date_col] = pd.to_datetime(r[date_col], errors='coerce')
    if hasattr(r[date_col].dt, 'tz') and r[date_col].dt.tz is not None:
        r[date_col] = r[date_col].dt.tz_localize(None)
    r[date_col] = r[date_col].dt.normalize()
    r = r.dropna(subset=[date_col]).sort_values(date_col)
    r = r.drop_duplicates(date_col, keep='last').rename(columns={date_col: '_date'})
    out = pd.DataFrame(index=left)
    out.index.name = 'date'
    if r.empty:
        return out
    right_ns = r['_date'].to_numpy(dtype='datetime64[ns]').astype('int64')
    left_ns = left.to_numpy(dtype='datetime64[ns]').astype('int64')
    pos = np.searchsorted(right_ns, left_ns, side='left') - 1
    valid = pos >= 0
    safe = np.clip(pos, 0, len(r) - 1)
    picked = r.iloc[safe].copy()
    picked.index = left
    if (~valid).any():
        picked.loc[~valid, '_date'] = pd.NaT
        for col in picked.columns:
            if col != '_date':
                picked.loc[~valid, col] = np.nan
    return picked

def make_constituent_features(dates,prices,weights,membership):
    p=prices.pivot_table(index='date',columns='symbol',values='close',aggfunc='last').sort_index(); r=p.pct_change(fill_method=None); symbols=[c for c in weights.columns if c!='DATE' and c in r.columns]
    if not symbols:return pd.DataFrame(index=pd.DatetimeIndex(dates))
    wsrc=weights[['DATE']+symbols].rename(columns={'DATE':'date'}); w=strict_asof(dates,wsrc).drop(columns=['_date'],errors='ignore'); W=w[symbols].fillna(0.0); R=r.reindex(pd.DatetimeIndex(dates))[symbols]
    for sym,g in membership.groupby('symbol'):
        if sym not in symbols:continue
        valid=pd.Series(False,index=W.index)
        for _,row in g.dropna(subset=['valid_from']).iterrows():
            end=row['valid_to'] if pd.notna(row['valid_to']) else W.index.max(); valid|=(W.index>=row['valid_from'])&(W.index<=end)
        W.loc[~valid,sym]=0.0
    wsum=W.sum(axis=1).replace(0,np.nan); cnt=(W>0).sum(axis=1).replace(0,np.nan); out=pd.DataFrame(index=W.index); out['const_wret1']=(R.fillna(0)*W).sum(axis=1)/wsum; out['const_breadth']=(((R>0).astype(float)*(W>0)).sum(axis=1)/cnt); out['const_dispersion']=R.std(axis=1); return out.replace([np.inf,-np.inf],np.nan).fillna(0)

def make_global_features(index,futures,context,flows,news,prices,weights,membership):
    dates=pd.DatetimeIndex(index['date'].unique()).sort_values(); x=index.set_index('date').sort_index(); c=x['close'].astype(float); r=c.pct_change(fill_method=None); g=pd.DataFrame(index=dates); g['ret1']=r; g['ret5']=c.pct_change(5,fill_method=None); g['ret20']=c.pct_change(20,fill_method=None); g['ret60']=c.pct_change(60,fill_method=None); g['vol10']=r.rolling(10).std(); g['vol20']=r.rolling(20).std(); g['ema20_gap']=c/c.ewm(span=20,adjust=False).mean()-1; up=r.clip(lower=0).ewm(alpha=1/14,adjust=False).mean(); dn=(-r.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean(); g['rsi14']=up/(dn+1e-9)
    if {'high','low'}.issubset(x.columns):g['atr14']=(x['high']-x['low']).abs().rolling(14).mean()/c
    p=context.pivot_table(index='date',columns='series',values='close',aggfunc='last').sort_index(); cc=pd.DataFrame(index=p.index)
    for col in p.columns:cc[f'ctx_{col}_ret5']=p[col].pct_change(5,fill_method=None)
    if len(cc):g=g.join(strict_asof(dates,cc.reset_index()).drop(columns=['_date'],errors='ignore'))
    if not futures.empty:
        f=futures.sort_values(['date','expiry']); exps=pd.DatetimeIndex(sorted(f['expiry'].dropna().unique())); pos=np.searchsorted(exps.values.astype('datetime64[ns]'),dates.values.astype('datetime64[ns]'),side='right'); nxt=pd.DataFrame({'date':dates,'expiry':[pd.Timestamp(exps[i]) if i<len(exps) else pd.NaT for i in pos]}); ff=f.merge(nxt,on=['date','expiry'],how='inner')
        if not ff.empty:
            agg=ff.groupby('date').agg(fut_close=('close','last'),fut_oi=('open_interest','last'),fut_volume=('volume','last')); agg['fut_basis']=agg['fut_close']/c.reindex(agg.index)-1; agg['fut_oi_chg5']=agg['fut_oi'].pct_change(5,fill_method=None); agg['fut_volume_z']=(agg['fut_volume']-agg['fut_volume'].rolling(20).mean())/(agg['fut_volume'].rolling(20).std()+1e-9); g=g.join(strict_asof(dates,agg.reset_index()).drop(columns=['_date'],errors='ignore'))
    g=g.join(make_constituent_features(dates,prices,weights,membership)); age_flow=int(CFG['freshness']['flow_max_age_days'])
    if {'fii_net','dii_net'}.issubset(flows.columns):
        f=flows.groupby('date')[['fii_net','dii_net']].sum().sort_index(); ff=pd.DataFrame(index=f.index)
        for col in ['fii_net','dii_net']:
            s=f[col]; ff[f'{col}_z']=(s-s.rolling(60,min_periods=10).mean())/(s.rolling(60,min_periods=10).std()+1e-9); ff[f'{col}_5z']=s.rolling(5).sum()/(s.rolling(60,min_periods=10).std()+1e-9)
        a=strict_asof(dates,ff.reset_index()); d=strict_asof(dates,f.reset_index())[['_date']]; age=(pd.DatetimeIndex(dates)-pd.DatetimeIndex(d['_date'])).days.astype(float); g['flow_age_days']=age
        for col in ff.columns:g[col]=a[col].where(age<=age_flow,0)
    age_news=int(CFG['freshness']['news_max_age_days']); pos=[c for c in news.columns if 'pos' in c]; neg=[c for c in news.columns if 'neg' in c]; cnt=[c for c in news.columns if 'count' in c]
    if pos or neg or cnt:
        n=news.set_index('date').sort_index(); ps=n[pos].sum(axis=1) if pos else pd.Series(0.,index=n.index); ns=n[neg].sum(axis=1) if neg else pd.Series(0.,index=n.index); vc=n[cnt].sum(axis=1) if cnt else pd.Series(0.,index=n.index); nd=pd.DataFrame(index=n.index); nd['news_sent5']=(ps.rolling(5).sum()-ns.rolling(5).sum())/vc.rolling(5).sum().clip(lower=1); nd['news_vol5']=np.log1p(vc.rolling(5).sum().clip(lower=0)); a=strict_asof(dates,nd.reset_index()); d=strict_asof(dates,n.reset_index()[['date']])[['_date']]; age=(pd.DatetimeIndex(dates)-pd.DatetimeIndex(d['_date'])).days.astype(float); g['news_age_days']=age
        for col in nd.columns:g[col]=a[col].where(age<=age_news,0)
    return g.replace([np.inf,-np.inf],np.nan).fillna(0)

def make_option_features(index,options):
    dates=pd.DatetimeIndex(index['date']); spot=index.set_index('date')['close'].astype(float); exps=np.array(sorted(pd.to_datetime(options['expiry']).unique()),dtype='datetime64[ns]'); pos=np.searchsorted(exps,dates.values.astype('datetime64[ns]'),side='right'); target=np.array([pd.Timestamp(exps[i]) if i<len(exps) else pd.NaT for i in pos],dtype='datetime64[ns]'); o=options.merge(pd.DataFrame({'date':dates,'target_expiry':target}),on='date',how='inner'); o=o[o['expiry'].eq(o['target_expiry'])].copy(); out=pd.DataFrame(index=dates)
    if o.empty:return out
    o['spot']=o['date'].map(spot); o['m']=(((o['strike']/o['spot'])-1)*100).round().clip(-8,8).astype(int)
    for typ in ['CE','PE']:
        z=o[o['option_type'].eq(typ)]
        for stat in ['open_interest','volume']:
            h=z.groupby(['date','m'])[stat].sum().unstack(fill_value=0).reindex(dates).fillna(0); h=h.div(h.sum(axis=1).replace(0,1),axis=0)
            for b in range(-8,9):out[f'opt_{typ.lower()}_{stat[:2]}_{b:+d}']=h[b] if b in h.columns else 0.0
    ce=o[o['option_type'].eq('CE')].groupby('date')['open_interest'].sum(); pe=o[o['option_type'].eq('PE')].groupby('date')['open_interest'].sum(); out['opt_pcr_oi']=pe/ce.replace(0,np.nan); return out.replace([np.inf,-np.inf],np.nan).fillna(0)

def build_samples(inp):
    options,futures,context,news,index,prices,weights,membership,flows=inp; dates=pd.DatetimeIndex(index['date'].sort_values().unique()); gf=make_global_features(index,futures,context,flows,news,prices,weights,membership).join(make_option_features(index,options),how='left').fillna(0); idx=index.set_index('date').sort_index(); exps=pd.DatetimeIndex(sorted(pd.to_datetime(options['expiry']).unique())); seq=int(CFG['sequence_length']); rows=[]; X=[]
    for i in range(seq-1,len(dates),int(CFG['sample_stride'])):
        d=pd.Timestamp(dates[i]); p=np.searchsorted(exps.values,np.datetime64(d),side='right')
        if p>=len(exps):continue
        e=pd.Timestamp(exps[p]); path_dates=dates[(dates>d)&(dates<=e)]
        if not len(path_dates):continue
        td=pd.Timestamp(path_dates[-1]); w=idx.loc[path_dates]; spot=float(idx.loc[d,'close']); close_ret=float(idx.loc[td,'close'])/spot-1; hi=float(w['high'].max() if 'high' in w else w['close'].max()); lo=float(w['low'].min() if 'low' in w else w['close'].min()); path_hi=hi/spot-1; path_lo=lo/spot-1
        X.append(gf.iloc[i-seq+1:i+1].to_numpy(np.float32)); rows.append({'signal_date':d,'expiry':e,'target_date':td,'spot':spot,'target_return':close_ret,'path_low_return':path_lo,'path_high_return':path_hi,'target_down_exc':max(0,close_ret-path_lo),'target_up_exc':max(0,path_hi-close_ret)})
    return np.stack(X).astype(np.float32),pd.DataFrame(rows).sort_values('signal_date').reset_index(drop=True),gf.columns.tolist()

class ICTransformer(nn.Module):
    def __init__(self,n_features):
        super().__init__(); c=CFG['model']; d=int(c['d_model']); self.inp=nn.Linear(n_features,d); enc=nn.TransformerEncoderLayer(d_model=d,nhead=int(c['heads']),dim_feedforward=4*d,dropout=float(c['dropout']),batch_first=True,norm_first=True,activation='gelu'); self.enc=nn.TransformerEncoder(enc,num_layers=int(c['layers'])); self.norm=nn.LayerNorm(d); self.head=nn.Linear(d,21)
    def forward(self,x):
        h=self.norm(self.enc(self.inp(x))[:,-1]); raw=self.head(h).view(-1,3,len(Q)); close=torch.sort(raw[:,0],dim=-1).values; down=torch.cumsum(torch.nn.functional.softplus(raw[:,1]),dim=-1); up=torch.cumsum(torch.nn.functional.softplus(raw[:,2]),dim=-1); low=close-down; high=close+up; return torch.stack([low,close,high],dim=1)

def pinball(pred,target):
    out=0.0
    for k,q in enumerate(Q):
        e=target-pred[:,k]; out+=torch.maximum((q-1)*e,q*e).mean()
    return out/len(Q)

def fit_model(X,meta):
    y=meta[['target_return','target_down_exc','target_up_exc']].to_numpy(np.float32); xm=X.reshape(-1,X.shape[-1]).mean(0); xs=X.reshape(-1,X.shape[-1]).std(0); xs=np.where(xs<1e-6,1,xs); Xn=np.clip((X-xm)/xs,-8,8).astype(np.float32); n=len(meta); cut=max(1,int(n*.9)); tr=np.arange(cut); va=np.arange(cut,n) if cut<n else tr; model=ICTransformer(X.shape[-1]); opt=torch.optim.AdamW(model.parameters(),lr=float(CFG['model']['learning_rate']),weight_decay=float(CFG['model']['weight_decay'])); dl=DataLoader(TensorDataset(torch.from_numpy(Xn[tr]),torch.from_numpy(y[tr])),batch_size=int(CFG['model']['batch_size']),shuffle=False); best=1e99; state=None; bad=0
    for _ in range(int(CFG['model']['epochs'])):
        model.train()
        for xb,yb in dl:
            opt.zero_grad(set_to_none=True); pred=model(xb); close=pred[:,1]; down=close-pred[:,0]; up=pred[:,2]-close; loss=pinball(close,yb[:,0])+pinball(down,yb[:,1])+pinball(up,yb[:,2]); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),float(CFG['model']['gradient_clip'])); opt.step()
        model.eval()
        with torch.no_grad():
            pred=model(torch.from_numpy(Xn[va])); close=pred[:,1]; down=close-pred[:,0]; up=pred[:,2]-close; val=float(pinball(close,torch.from_numpy(y[va,0]))+pinball(down,torch.from_numpy(y[va,1]))+pinball(up,torch.from_numpy(y[va,2])))
        if val<best-1e-6:best=val; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; bad=0
        else:
            bad+=1
            if bad>=int(CFG['model']['patience']):break
    if state is not None:model.load_state_dict(state)
    return model,(xm.astype(np.float32),xs.astype(np.float32))

def predict(model,norm,X):
    xm,xs=norm; Xn=np.clip((X-xm)/xs,-8,8).astype(np.float32); model.eval()
    with torch.no_grad(): raw=model(torch.from_numpy(Xn)).numpy()
    low=np.minimum(raw[:,0],raw[:,1]); high=np.maximum(raw[:,2],raw[:,1]); return np.stack([low,raw[:,1],high],axis=1)

def sample_q(q):return np.interp(np.linspace(.0025,.9975,400),Q,q)
def payoff(S,k1,k2,k3,k4,credit):
    S=np.asarray(S,float); return credit-np.maximum(k2-S,0)+np.maximum(k1-S,0)-np.maximum(S-k3,0)+np.maximum(S-k4,0)
def build_book(options):
    book={}
    for (d,e,t),g in options.groupby(['date','expiry','option_type'],sort=False):
        z=g.groupby('strike',as_index=False)['close'].last().sort_values('strike'); book[(pd.Timestamp(d),pd.Timestamp(e),t)]=(z['strike'].to_numpy(float),z['close'].to_numpy(float))
    return book
def quote(book,date,expiry,typ,target):
    key=(pd.Timestamp(date),pd.Timestamp(expiry),typ)
    if key not in book:return np.nan,np.nan
    k,p=book[key]; j=int(np.searchsorted(k,float(target))); cand=[i for i in (j-1,j) if 0<=i<len(k)]
    if not cand:return np.nan,np.nan
    i=min(cand,key=lambda a:abs(k[a]-float(target))); return float(k[i]),float(p[i])
def costs(legs,lot,date):
    slip=float(CFG['costs']['slippage_points_per_leg']); buy=sum(max(legs[k]+slip,0) for k in ['buy_put','buy_call']); sell=sum(max(legs[k]-slip,0) for k in ['sell_put','sell_call']); turnover=(buy+sell)*lot; brokerage=4*float(CFG['costs']['brokerage_per_order']); exchange=turnover*float(CFG['costs']['exchange_txn_pct']); sebi=turnover*float(CFG['costs']['sebi_turnover_pct']); gst=float(CFG['costs']['gst_pct'])*(brokerage+exchange); stamp=buy*lot*float(CFG['costs']['stamp_buy_pct']); rate=float(CFG['costs']['stt_sell_pct_from_2026_04_01']) if pd.Timestamp(date)>=pd.Timestamp('2026-04-01') else float(CFG['costs']['stt_sell_pct_pre_2026_04_01']); return brokerage+exchange+sebi+gst+stamp+sell*lot*rate

def candidate_ladder(date,expiry,spot,book,pred_dist):
    low_q,close_q,high_q=pred_dist; close_s=spot*(1+sample_q(close_q)); low_s=spot*(1+sample_q(low_q)); high_s=spot*(1+sample_q(high_q)); rows=[]; lot=float(CFG['lot_size'])
    for w in CFG['wing_width_points']:
      for po in CFG['short_put_moneyness_points']:
       for co in CFG['short_call_moneyness_points']:
        sp,sprem=quote(book,date,expiry,'PE',spot-po); lp,lprem=quote(book,date,expiry,'PE',sp-w) if np.isfinite(sp) else (np.nan,np.nan); sc,screm=quote(book,date,expiry,'CE',spot+co); lc,lcrem=quote(book,date,expiry,'CE',sc+w) if np.isfinite(sc) else (np.nan,np.nan)
        if not all(np.isfinite(v) for v in [sp,lp,sc,lc,sprem,lprem,screm,lcrem]):continue
        pw=sp-lp; cw=lc-sc
        if pw<=0 or abs(pw-cw)>1e-6:continue
        gross=sprem+screm-lprem-lcrem
        if gross<=0:continue
        entry_cost=costs({'buy_put':lprem,'sell_put':sprem,'sell_call':screm,'buy_call':lcrem},lot,date); credit=gross-entry_cost/lot; width=pw
        if credit<=0 or credit>=width:continue
        rr=credit/(width-credit); prof=payoff(close_s,lp,sp,sc,lc,credit); pop=float(np.mean(prof>0)); loss=float(np.mean(prof<0)); ev=float(np.mean(prof))*lot; p_lower=float(np.mean(low_s<=sp)); p_upper=float(np.mean(high_s>=sc)); touch=1-(1-p_lower)*(1-p_upper); score=rr*pop*max(0,1-touch)
        rows.append({'signal_date':str(pd.Timestamp(date).date()),'expiry':str(pd.Timestamp(expiry).date()),'spot':spot,'short_put':sp,'long_put':lp,'short_call':sc,'long_call':lc,'width':width,'gross_credit':gross,'entry_costs':entry_cost,'net_credit':credit,'max_profit':credit*lot,'max_loss':(width-credit)*lot,'reward_risk':rr,'model_profit_probability':pop,'model_loss_probability':loss,'model_touch_probability':touch,'model_expected_pnl':ev,'selection_score':score})
    rows.sort(key=lambda r:(r['selection_score'],r['reward_risk']),reverse=True); return rows

def realized(c,meta):
    if c is None:return None
    s=float(meta['spot']); close=s*(1+float(meta['target_return'])); pnl=float(payoff(close,c['long_put'],c['short_put'],c['short_call'],c['long_call'],c['net_credit']))*float(CFG['lot_size']); touch=int(float(meta['path_low_return'])<=(c['short_put']-s)/s or float(meta['path_high_return'])>=(c['short_call']-s)/s); z=dict(c); z.update({'expiry_close':close,'realized_pnl':pnl,'realized_win':int(pnl>0),'realized_touch':touch}); return z

def entries(meta):
    out=[]; gap=int(CFG['entry']['days_before_expiry'])
    for e,g in meta.groupby('expiry'):
        desired=pd.Timestamp(e)-pd.Timedelta(days=gap); out.append(int((pd.to_datetime(g['signal_date'])-desired).abs().idxmin()))
    return sorted(set(out))

def trade_metrics(df):
    if df is None or df.empty:return {'trades':0,'wins':0,'win_rate':0,'total_pnl':0,'mean_pnl':0,'profit_factor':0,'max_drawdown':0,'avg_reward_risk':0,'avg_model_pop':0,'avg_touch':0,'avg_max_loss':0}
    p=df['realized_pnl'].astype(float); eq=p.cumsum(); dd=(eq.cummax()-eq).max(); gain=p[p>0].sum(); loss=-p[p<0].sum(); return {'trades':len(df),'wins':int((p>0).sum()),'win_rate':float((p>0).mean()),'total_pnl':float(p.sum()),'mean_pnl':float(p.mean()),'profit_factor':float(gain/loss) if loss>0 else (float('inf') if gain>0 else 0),'max_drawdown':float(dd),'avg_reward_risk':float(df['reward_risk'].mean()),'avg_model_pop':float(df['model_profit_probability'].mean()),'avg_touch':float(df['model_touch_probability'].mean()),'avg_max_loss':float(df['max_loss'].mean())}

def pred_metrics(model,norm,X,meta,train_meta):
    d=predict(model,norm,X); actual=np.column_stack([meta['path_low_return'],meta['target_return'],meta['path_high_return']]).astype(float); q50=d[:,:,3]; mae=np.mean(np.abs(actual-q50),axis=0); naive=np.column_stack([np.full(len(meta),train_meta['path_low_return'].median()),np.full(len(meta),train_meta['target_return'].median()),np.full(len(meta),train_meta['path_high_return'].median())]); nm=np.mean(np.abs(actual-naive),axis=0); return {'model_low_mae':float(mae[0]),'model_close_mae':float(mae[1]),'model_high_mae':float(mae[2]),'naive_low_mae':float(nm[0]),'naive_close_mae':float(nm[1]),'naive_high_mae':float(nm[2]),'close_mae_lift_vs_naive':float(1-mae[1]/nm[1]) if nm[1]>0 else 0}

def splits(meta,kind):
    exps=pd.DatetimeIndex(sorted(meta['expiry'].unique()))
    if kind=='wfo':
        pre=exps[exps<pd.Timestamp(CFG['new_oos_start'])]; nw=int(CFG['walk_forward']['windows']); nt=int(CFG['walk_forward']['test_expiries']); minte=int(CFG['walk_forward']['train_expiries'])
        if len(pre)<minte+nt:return []
        starts=np.unique(np.linspace(minte,len(pre)-nt,nw).round().astype(int)); out=[]
        for i,s in enumerate(starts):
            te=pre[s:s+nt]; test=meta[meta['expiry'].isin(te)].copy()
            if test.empty:continue
            test_start=pd.Timestamp(test['signal_date'].min()); train=meta[(meta['expiry']<te[0]-pd.Timedelta(days=int(CFG['purge_days'])))&(meta['signal_date']<test_start-pd.Timedelta(days=int(CFG['embargo_days'])))].copy()
            if len(train)>=int(CFG['min_train_samples']):out.append((train,test,f'wfo_{i+1}'))
        return out
    if kind=='new_oos':test=meta[meta['signal_date']>=pd.Timestamp(CFG['new_oos_start'])].copy()
    else:
        split_exp=exps[int(len(exps)*.7)]; test=meta[meta['expiry']>=split_exp].copy()
    if test.empty:return []
    test_start=pd.Timestamp(test['signal_date'].min()); train=meta[(meta['signal_date']<test_start-pd.Timedelta(days=int(CFG['embargo_days'])))&(meta['expiry']<pd.Timestamp(test['expiry'].min())-pd.Timedelta(days=int(CFG['purge_days'])))].copy(); return [(train,test,kind)] if len(train)>=int(CFG['min_train_samples']) else []

def run_strategy(model,norm,X,meta,book):
    idxs=entries(meta); d=predict(model,norm,X[idxs]); trades=[]; ladder=[]
    for n,j in enumerate(idxs):
        row=meta.iloc[j]; rows=candidate_ladder(pd.Timestamp(row['signal_date']),pd.Timestamp(row['expiry']),float(row['spot']),book,d[n]); ladder+=rows[:5]; valid=[r for r in rows if r['reward_risk']>=float(CFG['min_ic_reward_risk']) and r['model_profit_probability']>=float(CFG['min_profit_probability']) and r['model_touch_probability']<=float(CFG['max_touch_probability']) and r['model_loss_probability']<=float(CFG['max_expected_loss_probability']) and r['model_expected_pnl']>0]
        if valid:
            z=realized(valid[0],row); z['strategy']='deep_ic'; trades.append(z)
    return pd.DataFrame(trades),pd.DataFrame(ladder)

def backtest_case(case,X,meta,book):
    train,test,name=case; model,norm=fit_model(X[train.index.to_numpy()],train); pm=pred_metrics(model,norm,X[test.index.to_numpy()],test,train); tr,lad=run_strategy(model,norm,X[test.index.to_numpy()],test.reset_index(drop=True),book); tm=trade_metrics(tr); return {'name':name,'train_samples':len(train),'test_samples':len(test),**pm,**{f'strategy_{k}':v for k,v in tm.items()}},tr,lad

def main():
    inp=load_inputs(); options=inp[0]; X,meta,cols=build_samples(inp); keep=pd.to_datetime(meta['target_date'])<=pd.Timestamp(CFG['history_end']); X=X[keep.to_numpy()]; meta=meta.loc[keep].reset_index(drop=True)
    if len(meta)<int(CFG['min_train_samples']):raise RuntimeError(f'need {CFG["min_train_samples"]} samples, got {len(meta)}')
    book=build_book(options); wfo_rows=[]; wfo_trades=[]
    for case in splits(meta,'wfo'):
        row,tr,_=backtest_case(case,X,meta,book); wfo_rows.append(row); wfo_trades.append(tr.assign(fold=case[2]) if not tr.empty else tr)
    wfo_df=pd.DataFrame(wfo_rows); wfo_df.to_csv(OUT/'wfo_metrics.csv',index=False); pd.concat(wfo_trades,ignore_index=True).to_csv(OUT/'wfo_strategy_trades.csv',index=False) if any(not x.empty for x in wfo_trades) else pd.DataFrame().to_csv(OUT/'wfo_strategy_trades.csv',index=False)
    split_rows=[]; split_trades=[]
    for case in splits(meta,'70_30')+splits(meta,'new_oos'):
        row,tr,_=backtest_case(case,X,meta,book); split_rows.append(row); split_trades.append(tr.assign(split=case[2]) if not tr.empty else tr)
    split_df=pd.DataFrame(split_rows); split_df[split_df['name'].eq('70_30')].to_csv(OUT/'split_70_30_metrics.csv',index=False); split_df[split_df['name'].eq('new_oos')].to_csv(OUT/'new_oos_metrics.csv',index=False); alltr=pd.concat(split_trades,ignore_index=True) if split_trades else pd.DataFrame()
    alltr[alltr.get('split',pd.Series(dtype=str)).eq('70_30')].to_csv(OUT/'split_70_30_trades.csv',index=False) if not alltr.empty else pd.DataFrame().to_csv(OUT/'split_70_30_trades.csv',index=False); alltr[alltr.get('split',pd.Series(dtype=str)).eq('new_oos')].to_csv(OUT/'new_oos_trades.csv',index=False) if not alltr.empty else pd.DataFrame().to_csv(OUT/'new_oos_trades.csv',index=False)
    latest_model,latest_norm=fit_model(X,meta); latest_pred=predict(latest_model,latest_norm,X[[-1]])[0]; row=meta.iloc[-1]; latest_rows=candidate_ladder(pd.Timestamp(row['signal_date']),pd.Timestamp(row['expiry']),float(row['spot']),book,latest_pred); valid=[r for r in latest_rows if r['reward_risk']>=float(CFG['min_ic_reward_risk']) and r['model_profit_probability']>=float(CFG['min_profit_probability']) and r['model_touch_probability']<=float(CFG['max_touch_probability']) and r['model_loss_probability']<=float(CFG['max_expected_loss_probability']) and r['model_expected_pnl']>0]
    latest={'signal_date':str(pd.Timestamp(row['signal_date']).date()),'next_expiry':str(pd.Timestamp(row['expiry']).date()),'spot':float(row['spot']),'quantiles':Q.tolist(),'coherent_low_return':latest_pred[0].tolist(),'coherent_close_return':latest_pred[1].tolist(),'coherent_high_return':latest_pred[2].tolist(),'candidate_count':len(valid),'selected_ic':valid[0] if valid else None,'top_candidate_ladder':latest_rows[:15]}; (OUT/'latest_distribution.json').write_text(json.dumps(latest,indent=2)); (OUT/'latest_iron_condor.json').write_text(json.dumps(latest,indent=2)); (pd.DataFrame([valid[0]]) if valid else pd.DataFrame(latest_rows[:15])).to_csv(OUT/('latest_iron_condor.csv' if valid else 'latest_candidate_ladder.csv'),index=False)
    oos_df=split_df[split_df['name'].eq('new_oos')]; gate='RESEARCH_ONLY_PENDING_STABLE_OOS'
    if len(wfo_df)==int(CFG['walk_forward']['windows']) and not oos_df.empty and int(oos_df.iloc[0]['strategy_trades'])>=int(CFG['gates']['minimum_oos_trades']):gate='OOS_VALIDATED_PENDING_MANUAL_REVIEW'
    summary={'version':'V8.1','samples':len(meta),'features':len(cols),'expiry_count':int(meta['expiry'].nunique()),'history_end':CFG['history_end'],'new_oos_start':CFG['new_oos_start'],'wfo_windows_completed':len(wfo_df),'wfo_windows_required':int(CFG['walk_forward']['windows']),'latest':{'signal_date':latest['signal_date'],'next_expiry':latest['next_expiry'],'spot':latest['spot'],'candidate_count':latest['candidate_count'],'selected_ic':latest['selected_ic']},'research_gate':gate,'objectives':{'coherent_range':True,'pop_from_expiry_payoff_distribution':True,'touch_probability_from_path_distribution':True,'post_cost_ic_payoff':True,'six_window_wfo':len(wfo_df)==int(CFG['walk_forward']['windows']),'70_30':not split_df[split_df['name'].eq('70_30')].empty,'new_oos':not oos_df.empty}}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)); (OUT/'status.json').write_text(json.dumps({'status':gate,'summary':summary},indent=2)); print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
