from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / 'v8/config.yaml').read_text())
OUT = ROOT / 'v8/research'
OUT.mkdir(parents=True, exist_ok=True)
SEED = int(CFG['seed'])
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(2)

Q = np.array(CFG['model']['quantiles'], dtype=np.float32)


def clean_dates(df: pd.DataFrame, col: str = 'date') -> pd.DataFrame:
    x = df.copy()
    x[col] = pd.to_datetime(x[col], errors='coerce').dt.normalize()
    return x.dropna(subset=[col]).sort_values(col)


def load_inputs():
    options = pd.read_parquet(ROOT / 'data/cache/nifty_options_long.parquet')
    futures = pd.read_parquet(ROOT / 'data/cache/nifty_futures_long.parquet')
    context = pd.read_parquet(ROOT / 'data/cache/v5_market_context.parquet')
    news = pd.read_csv(ROOT / 'data/cache/v5_news_daily.csv')
    index = clean_dates(pd.read_parquet(ROOT / 'data/cache/v62_nifty50_index.parquet'))
    prices = pd.read_parquet(ROOT / 'data/cache/v62_constituent_prices.parquet')
    weights = pd.read_csv(ROOT / 'data/cache/v62_constituent_weights.csv')
    membership = pd.read_csv(ROOT / 'data/cache/v62_nifty50_membership.csv')
    flows = pd.read_parquet(ROOT / 'data/cache/v62_fii_dii.parquet')

    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c in index.columns:
            index[c] = pd.to_numeric(index[c], errors='coerce')
    index = index.dropna(subset=['close']).drop_duplicates('date', keep='last')

    for c in ['date', 'expiry']:
        options[c] = pd.to_datetime(options[c], errors='coerce').dt.normalize()
        futures[c] = pd.to_datetime(futures[c], errors='coerce').dt.normalize()
    options['option_type'] = options['option_type'].astype(str).str.upper()
    for c in ['strike', 'open_interest', 'volume', 'close']:
        options[c] = pd.to_numeric(options[c], errors='coerce').fillna(0.0)
        if c in futures.columns:
            futures[c] = pd.to_numeric(futures[c], errors='coerce').fillna(0.0)
    options = options.dropna(subset=['date', 'expiry'])
    futures = futures.dropna(subset=['date', 'expiry'])

    context = clean_dates(context)
    context['series'] = context['series'].astype(str).str.strip()
    context['close'] = pd.to_numeric(context['close'], errors='coerce')

    news.columns = [str(c).strip().lower() for c in news.columns]
    news = clean_dates(news)
    for c in news.columns:
        if c != 'date':
            news[c] = pd.to_numeric(news[c], errors='coerce')

    prices.columns = [str(c).strip().lower() for c in prices.columns]
    prices['symbol'] = prices['symbol'].astype(str).str.upper()
    prices['date'] = pd.to_datetime(prices['date'], errors='coerce').dt.normalize()
    prices['close'] = pd.to_numeric(prices['close'], errors='coerce')
    prices = prices.dropna(subset=['symbol', 'date', 'close']).drop_duplicates(['symbol', 'date'], keep='last')

    weights.columns = [str(c).strip().upper() for c in weights.columns]
    weights['DATE'] = pd.to_datetime(weights['DATE'], errors='coerce').dt.normalize()
    for c in weights.columns:
        if c != 'DATE':
            weights[c] = pd.to_numeric(weights[c], errors='coerce').fillna(0.0)

    membership.columns = [str(c).strip().lower() for c in membership.columns]
    membership['symbol'] = membership['symbol'].astype(str).str.upper()
    membership['valid_from'] = pd.to_datetime(membership['valid_from'], errors='coerce').dt.normalize()
    membership['valid_to'] = pd.to_datetime(membership['valid_to'], errors='coerce').dt.normalize()

    flows['date'] = pd.to_datetime(flows['date'], errors='coerce').dt.normalize()
    for c in ['fii_net', 'dii_net']:
        if c in flows.columns:
            flows[c] = pd.to_numeric(flows[c], errors='coerce')
    return options, futures, context, news, index, prices, weights, membership, flows


def strict_asof(left_dates, right, date_col='date'):
    l = pd.DataFrame({'date': pd.DatetimeIndex(left_dates)})
    r = right.copy().sort_values(date_col).rename(columns={date_col: '_date'})
    return pd.merge_asof(l, r, left_on='date', right_on='_date', direction='backward', allow_exact_matches=False).set_index('date')


def make_global_features(index, options, futures, context, flows, news):
    dates = pd.DatetimeIndex(index['date'].unique())
    x = index.set_index('date').sort_index()
    c = x['close'].astype(float)
    r = c.pct_change(fill_method=None)
    g = pd.DataFrame(index=dates)
    g['ret1'] = r
    g['ret5'] = c.pct_change(5, fill_method=None)
    g['ret20'] = c.pct_change(20, fill_method=None)
    g['vol10'] = r.rolling(10).std()
    g['vol20'] = r.rolling(20).std()
    g['ema20_gap'] = c / c.ewm(span=20, adjust=False).mean() - 1
    g['rsi14'] = r.clip(lower=0).ewm(alpha=1/14, adjust=False).mean() / ((-r.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean() + 1e-9)
    g['atr14'] = ((x['high'] - x['low']).abs().rolling(14).mean() / c) if {'high','low'}.issubset(x.columns) else np.nan

    p = context.pivot_table(index='date', columns='series', values='close', aggfunc='last').sort_index()
    cc = pd.DataFrame(index=p.index)
    for col in p.columns:
        cc[f'ctx_{col}_ret5'] = p[col].pct_change(5, fill_method=None)
    if len(cc):
        g = g.join(strict_asof(dates, cc.reset_index()).drop(columns=['_date'], errors='ignore'))

    if {'fii_net', 'dii_net'}.issubset(flows.columns):
        f = flows.groupby('date')[['fii_net','dii_net']].sum().sort_index()
        ff = pd.DataFrame(index=f.index)
        for col in ['fii_net','dii_net']:
            s = f[col]
            ff[f'{col}_z'] = (s - s.rolling(60, min_periods=10).mean()) / (s.rolling(60, min_periods=10).std() + 1e-9)
            ff[f'{col}_5z'] = s.rolling(5).sum() / (s.rolling(60, min_periods=10).std() + 1e-9)
        g = g.join(strict_asof(dates, ff.reset_index()).drop(columns=['_date'], errors='ignore'))

    pos = [c for c in news.columns if 'pos' in c]
    neg = [c for c in news.columns if 'neg' in c]
    cnt = [c for c in news.columns if 'count' in c]
    if pos or neg or cnt:
        n = news.set_index('date').sort_index()
        ps = n[pos].sum(axis=1) if pos else pd.Series(0.0, index=n.index)
        ns = n[neg].sum(axis=1) if neg else pd.Series(0.0, index=n.index)
        vc = n[cnt].sum(axis=1) if cnt else pd.Series(0.0, index=n.index)
        nd = pd.DataFrame(index=n.index)
        nd['news_sent5'] = (ps.rolling(5).sum() - ns.rolling(5).sum()) / vc.rolling(5).sum().clip(lower=1)
        nd['news_vol5'] = np.log1p(vc.rolling(5).sum().clip(lower=0))
        g = g.join(strict_asof(dates, nd.reset_index()).drop(columns=['_date'], errors='ignore'))

    return g.replace([np.inf,-np.inf], np.nan).fillna(0.0)


def make_option_features(index, options):
    dates = pd.DatetimeIndex(index['date'])
    spot = index.set_index('date')['close']
    exps = np.array(sorted(pd.to_datetime(options['expiry']).unique()), dtype='datetime64[ns]')
    pos = np.searchsorted(exps, dates.values.astype('datetime64[ns]'), side='right')
    next_exp = pd.Series([pd.Timestamp(exps[i]) if i < len(exps) else pd.NaT for i in pos], index=dates)
    base = pd.DataFrame({'date':dates, 'expiry':next_exp.values})
    o = options.merge(base, on='date')
    o = o[o['expiry'].eq(o['expiry_y'] if 'expiry_y' in o else o['expiry'])] if False else o
    # merged duplicate-name handling: options expiry remains expiry, base supplies expiry_x/expiry_y in pandas
    if 'expiry_y' in o.columns:
        o = o[o['expiry_x'].eq(o['expiry_y'])].copy().rename(columns={'expiry_x':'expiry'})
    if o.empty:
        return pd.DataFrame(index=dates)
    o['spot'] = o['date'].map(spot)
    o['m'] = ((o['strike'] / o['spot'] - 1.0) * 100).round().clip(-8,8).astype(int)
    out = pd.DataFrame(index=dates)
    for typ in ['CE','PE']:
        z = o[o['option_type'].eq(typ)]
        for stat in ['open_interest','volume']:
            h = z.groupby(['date','m'])[stat].sum().unstack(fill_value=0).reindex(dates).fillna(0)
            h = h.div(h.sum(axis=1).replace(0,1), axis=0)
            for b in range(-8,9):
                out[f'{typ.lower()}_{stat[:2]}_{b:+d}'] = h[b] if b in h.columns else 0.0
    ce = o[o['option_type'].eq('CE')].groupby('date').open_interest.sum()
    pe = o[o['option_type'].eq('PE')].groupby('date').open_interest.sum()
    out['pcr_oi'] = pe / ce.replace(0,np.nan)
    return out.reindex(dates).replace([np.inf,-np.inf],np.nan).fillna(0.0)


def build_samples(inp):
    options, futures, context, news, index, prices, weights, membership, flows = inp
    dates = pd.DatetimeIndex(index['date'].sort_values().unique())
    gf = make_global_features(index, options, futures, context, flows, news)
    of = make_option_features(index, options)
    gf = gf.join(of, how='left').fillna(0.0)
    idx = index.set_index('date').sort_index()
    exps = pd.DatetimeIndex(sorted(pd.to_datetime(options['expiry']).unique()))
    seq = int(CFG['sequence_length'])
    rows = []; x=[]
    for i in range(seq-1, len(dates), int(CFG['sample_stride'])):
        d = pd.Timestamp(dates[i])
        p = np.searchsorted(exps.values, np.datetime64(d), side='right')
        if p >= len(exps):
            continue
        e = pd.Timestamp(exps[p])
        td_candidates = dates[dates <= e]
        if len(td_candidates)==0: continue
        td = pd.Timestamp(td_candidates[-1])
        if td <= d or td not in idx.index: continue
        spot=float(idx.loc[d,'close']); row=idx.loc[td]
        high=float(row['high']) if 'high' in row else float(row['close'])
        low=float(row['low']) if 'low' in row else float(row['close'])
        y=np.array([float(row['close'])/spot-1, high/spot-1, low/spot-1],dtype=np.float32)
        x.append(gf.iloc[i-seq+1:i+1].to_numpy(np.float32))
        rows.append({'signal_date':d,'expiry':e,'target_date':td,'spot':spot,'target_return':y[0],'target_high_return':y[1],'target_low_return':y[2]})
    return np.stack(x).astype(np.float32), pd.DataFrame(rows).sort_values('signal_date').reset_index(drop=True), gf.columns.tolist()


class ICTransformer(nn.Module):
    def __init__(self, n_features:int):
        super().__init__(); c=CFG['model']; d=int(c['d_model'])
        self.inp=nn.Linear(n_features,d)
        enc=nn.TransformerEncoderLayer(d_model=d,nhead=int(c['heads']),dim_feedforward=4*d,dropout=float(c['dropout']),batch_first=True,norm_first=True,activation='gelu')
        self.enc=nn.TransformerEncoder(enc,num_layers=int(c['layers']))
        self.norm=nn.LayerNorm(d)
        self.head=nn.Linear(d,21)
    def forward(self,x):
        h=self.enc(self.inp(x)); h=self.norm(h[:,-1]); return self.head(h).view(-1,3,7)


def pinball(pred,target,quantiles):
    loss=0.0
    for k,q in enumerate(quantiles):
        e=target-pred[:,k]; loss += torch.maximum((q-1)*e,q*e).mean()
    return loss/len(quantiles)


def fit_model(x,y):
    xm=x.reshape(-1,x.shape[-1]).mean(0); xs=x.reshape(-1,x.shape[-1]).std(0); xs=np.where(xs<1e-6,1,xs)
    xn=np.clip((x-xm)/xs,-8,8).astype(np.float32)
    n=len(y); cut=max(1,int(n*0.9)); tr=np.arange(cut); va=np.arange(cut,n) if cut<n else tr
    model=ICTransformer(x.shape[-1]); opt=torch.optim.AdamW(model.parameters(),lr=float(CFG['model']['learning_rate']),weight_decay=float(CFG['model']['weight_decay']))
    ds=TensorDataset(torch.from_numpy(xn[tr]),torch.from_numpy(y[tr])); dl=DataLoader(ds,batch_size=int(CFG['model']['batch_size']),shuffle=False)
    best=1e9; state=None; bad=0
    for _ in range(int(CFG['model']['epochs'])):
        model.train()
        for xb,yb in dl:
            opt.zero_grad(set_to_none=True); q=model(xb)
            loss=pinball(q[:,0,:],yb[:,0],Q)+pinball(q[:,1,:],yb[:,1],Q)+pinball(q[:,2,:],yb[:,2],Q)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),float(CFG['model']['gradient_clip'])); opt.step()
        model.eval();
        with torch.no_grad():
            q=model(torch.from_numpy(xn[va])); val=float(pinball(q[:,0,:],torch.from_numpy(y[va,0]),Q)+pinball(q[:,1,:],torch.from_numpy(y[va,1]),Q)+pinball(q[:,2,:],torch.from_numpy(y[va,2]),Q))
        if val < best-1e-5: best=val; state={k:v.detach().clone() for k,v in model.state_dict().items()}; bad=0
        else:
            bad+=1
            if bad>=int(CFG['model']['patience']): break
    if state is not None: model.load_state_dict(state)
    return model,(xm.astype(np.float32),xs.astype(np.float32))


def predict(model,norm,x):
    xm,xs=norm; xn=np.clip((x-xm)/xs,-8,8).astype(np.float32)
    model.eval();
    with torch.no_grad(): q=model(torch.from_numpy(xn)).numpy()
    return q


def payoff_ic(sp,k1,k2,k3,k4,net_credit,lot):
    # k1<k2 are puts; k3<k4 are calls
    width=max(k2-k1,k4-k3)
    max_profit=net_credit*lot
    max_loss=(width-net_credit)*lot
    return max_profit,max_loss


def reward_risk(net_credit,width):
    risk=max(width-net_credit,1e-9)
    return net_credit/risk


def option_quote(options,date,expiry,typ,target):
    o=options[(options['date']==date)&(options['expiry']==expiry)&(options['option_type']==typ)].copy()
    if o.empty:return np.nan,np.nan
    o['d']=(o['strike']-target).abs(); r=o.sort_values('d').iloc[0]
    return float(r['strike']),float(r['close'])


def candidate_ics(date,expiry,spot,options,model_q):
    # q0 = close, q1 = high, q2 = low; construct a conservative model-based range from q10/q90.
    close,high,low=model_q
    put_floor=spot*(1+low[1]); call_ceiling=spot*(1+high[5])
    candidates=[]
    wings=CFG['wing_width_points']; pouts=CFG['short_put_moneyness_points']; couts=CFG['short_call_moneyness_points']
    for w in wings:
        for po in pouts:
            for co in couts:
                sp_target=spot-po; sc_target=spot+co
                lp,cp=option_quote(options,date,expiry,'PE',sp_target)
                sp,cpc=option_quote(options,date,expiry,'PE',sp_target-w)
                sc,cc=option_quote(options,date,expiry,'CE',sc_target)
                lc,ccw=option_quote(options,date,expiry,'CE',sc_target+w)
                if any(np.isnan(z) for z in [lp,cp,sp,cpc,sc,cc,lc,ccw]): continue
                credit=cp-cpc+cc-ccw
                rr=reward_risk(credit,w)
                if credit<=0 or rr<float(CFG['min_ic_reward_risk']) or rr>float(CFG['max_ic_reward_risk']): continue
                # A simple model range screen; touched-range probability is refined by empirical simulation in evaluation.
                if lp >= put_floor or sc <= call_ceiling: continue
                prof,loss=payoff_ic(spot,lp,sp,sc,lc,credit,float(CFG['lot_size']))
                candidates.append({'short_put':lp,'long_put':sp,'short_call':sc,'long_call':lc,'credit':credit,'width':w,'reward_risk':rr,'max_profit':prof,'max_loss':loss,'put_distance':po,'call_distance':co})
    candidates.sort(key=lambda z:(z['reward_risk'],z['credit']),reverse=True)
    return candidates[:50]


def main():
    inp=load_inputs(); options,_,_,_,index,_,_,_,_=inp
    x,meta,cols=build_samples(inp)
    hist_end=pd.Timestamp(CFG['history_end']); keep=pd.to_datetime(meta['target_date'])<=hist_end
    x=x[keep.to_numpy()]; meta=meta.loc[keep].reset_index(drop=True)
    y=meta[['target_return','target_high_return','target_low_return']].to_numpy(np.float32)
    model,norm=fit_model(x,y)
    q=predict(model,norm,x[-1:])[0]
    latest_date=pd.Timestamp(meta['signal_date'].iloc[-1]); expiry=pd.Timestamp(meta['expiry'].iloc[-1]); spot=float(meta['spot'].iloc[-1])
    candidates=candidate_ics(latest_date,expiry,spot,options,q)
    chosen=candidates[0] if candidates else None
    summary={'version':'V8','samples':len(meta),'features':len(cols),'latest_signal_date':str(latest_date.date()),'next_expiry':str(expiry.date()),'spot':spot,'candidate_count':len(candidates),'selected_ic':chosen,'model':'deep_transformer_quantiles','research_gate':'RESEARCH_ONLY_PENDING_OOS'}
    json.dump(summary,open(OUT/'summary.json','w'),indent=2)
    json.dump({'q_close':q[0].tolist(),'q_high':q[1].tolist(),'q_low':q[2].tolist(),'quantiles':Q.tolist()},open(OUT/'latest_distribution.json','w'),indent=2)
    if chosen: pd.DataFrame([chosen]).to_csv(OUT/'latest_iron_condor.csv',index=False)
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
