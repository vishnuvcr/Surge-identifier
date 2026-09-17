from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / 'v5/config.yaml').read_text())
CAPITAL = float(CFG['capital'])
STRATEGIES = ['LONG_CALL','LONG_PUT','BULL_CALL_SPREAD','BEAR_PUT_SPREAD','BULL_PUT_SPREAD','BEAR_CALL_SPREAD','LONG_STRADDLE','LONG_STRANGLE','IRON_CONDOR','IRON_BUTTERFLY']
FEATURES = ['ret1','ret5','ret10','ret20','ema10_gap','ema20_gap','ema50_gap','ema10_20_gap','ema20_50_gap','rsi14','vol10','vol20','breakout20','breakdown20','range60','slope20','atm_straddle_pct','skew_proxy','dte','india_vix','india_vix_ret5','spx_ret5','usd_inr_ret5','brent_ret5','bank_rel5','midcap_rel5','fii_net_z','dii_net_z','fii_fut_long_short','pcr','flow_sentiment','news_sentiment','news_volume_log','news_neg_share']


def lot_size(expiry):
    e = pd.Timestamp(expiry).normalize()
    if e < pd.Timestamp('2015-10-30'): return 25
    if e < pd.Timestamp('2021-08-01'): return 75
    if e < pd.Timestamp('2024-05-02'): return 50
    if e < pd.Timestamp('2024-11-20'): return 25
    if e < pd.Timestamp('2026-01-06'): return 75
    return 65


class Market:
    def __init__(self, options, futures, context=None, news=None):
        o = options.copy()
        for c in ['date','expiry']: o[c] = pd.to_datetime(o[c]).dt.normalize()
        o['option_type'] = o.option_type.astype(str).str.upper()
        o = o.drop_duplicates(['date','expiry','strike','option_type'], keep='last')
        f = futures.copy()
        for c in ['date','expiry']: f[c] = pd.to_datetime(f[c]).dt.normalize()
        f = f[f.expiry >= f.date].sort_values(['date','expiry']).drop_duplicates(['date','expiry'], keep='last')
        spot = f.groupby('date', as_index=False).first()[['date','close']]
        self.spot = dict(zip(spot.date, spot.close.astype(float)))
        self.dates = tuple(pd.Timestamp(x) for x in sorted(set(o.date) & set(spot.date)))
        self.expiries, self.prices, self.strikes = {}, {}, {}
        for (d,e), g in o.groupby(['date','expiry'], sort=False):
            d,e = pd.Timestamp(d),pd.Timestamp(e); self.expiries.setdefault(d,[]).append(e)
            for typ,gg in g.groupby('option_type'):
                vals=gg.loc[gg.close>0,['strike','close']].dropna(); self.strikes[(d,e,typ)] = np.array(sorted(vals.strike.astype(float).unique()),dtype=float)
                for k,c in vals.itertuples(index=False,name=None): self.prices[(d,e,float(k),typ)] = float(c)
        for d in self.expiries: self.expiries[d]=tuple(sorted(self.expiries[d]))
        self.spot_series=pd.Series(self.spot).sort_index(); self.date_index={d:i for i,d in enumerate(self.dates)}
        self.context=self._prepare_context(context); self.news=self._prepare_news(news)

    def _prepare_context(self, context):
        if context is None or context.empty: return pd.DataFrame(index=pd.DatetimeIndex(self.dates))
        x=context.copy(); x['date']=pd.to_datetime(x.date).dt.normalize()
        p=x.pivot_table(index='date',columns='series',values='close',aggfunc='last').sort_index(); out=pd.DataFrame(index=p.index)
        for c in p.columns: out[c]=p[c]; out[f'{c}_ret1']=p[c].pct_change(); out[f'{c}_ret5']=p[c].pct_change(5)
        for c in ['fii_net','dii_net','pcr','sentiment_score','fii_idx_fut_long','fii_idx_fut_short']:
            if c in x.columns:
                s=x.groupby('date')[c].last().sort_index(); out=out.join(s.rename(c),how='outer')
        for c in ['fii_net','dii_net']:
            if c in out:
                med=out[c].expanding(60).median(); mad=(out[c]-med).abs().expanding(60).median(); out[f'{c}_z']=(out[c]-med)/(1.4826*mad.replace(0,np.nan))
        if 'fii_idx_fut_long' in out and 'fii_idx_fut_short' in out:
            out['fii_fut_long_short']=out.fii_idx_fut_long/out.fii_idx_fut_short.replace(0,np.nan)
        return out

    def _prepare_news(self, news):
        if news is None or news.empty: return pd.DataFrame(index=pd.DatetimeIndex(self.dates))
        n=news.copy(); n['date']=pd.to_datetime(n.date).dt.normalize(); num=n.select_dtypes(include=[np.number]).columns
        if 'Symbol' in n.columns or 'symbol' in n.columns:
            pos=[c for c in num if 'news_pos' in c]; neg=[c for c in num if 'news_neg' in c]; cnt=[c for c in num if 'news_count' in c]
            rows=[]
            for d,z in n.groupby('date'):
                den=float(z[cnt].sum().sum()) if cnt else 0.; ps=float(z[pos].sum().sum()) if pos else np.nan; ns=float(z[neg].sum().sum()) if neg else np.nan
                rows.append((d,(ps-ns)/max(den,1.) if np.isfinite(ps+ns) else np.nan,math.log1p(max(den,0.)),ns/max(den,1.) if np.isfinite(ns) else np.nan))
            return pd.DataFrame(rows,columns=['date','news_sentiment','news_volume_log','news_neg_share']).set_index('date')
        out=n.set_index('date').sort_index(); ren={}
        for c in ['sentiment','news_sentiment','financial_tone']:
            if c in out.columns: ren['news_sentiment']=c; break
        for c in ['volume','news_volume','article_count']:
            if c in out.columns: ren['news_volume_log']=c; break
        for c in ['negative_share','news_neg_share']:
            if c in out.columns: ren['news_neg_share']=c; break
        out=out[[ren[k] for k in ren]].rename(columns={v:k for k,v in ren.items()})
        if 'news_volume_log' in out: out['news_volume_log']=np.log1p(out.news_volume_log.clip(lower=0))
        return out

    def asof_context(self,d):
        cutoff=pd.Timestamp(d)-pd.Timedelta(days=1); frames=[]
        if not self.context.empty: frames.append(self.context.loc[self.context.index<cutoff].tail(1))
        if not self.news.empty: frames.append(self.news.loc[self.news.index<cutoff].tail(1))
        if not frames:return {}
        return pd.concat(frames,axis=1).iloc[-1].to_dict()

    def price(self,d,e,k,typ):
        v=self.prices.get((pd.Timestamp(d),pd.Timestamp(e),float(k),typ)); return None if v is None or not np.isfinite(v) or v<=0 else float(v)
    def nearest(self,d,e,typ,target):
        ks=self.strikes.get((pd.Timestamp(d),pd.Timestamp(e),typ)); return None if ks is None or len(ks)==0 else float(ks[np.argmin(np.abs(ks-target))])
    def expiry_for(self,d,lo,hi):
        for e in self.expiries.get(pd.Timestamp(d),()):
            if lo <= (e-pd.Timestamp(d)).days <= hi:return e
        return None
    def nth_date(self,d,n,end=None):
        i=self.date_index.get(pd.Timestamp(d));
        if i is None:return None
        j=i+n
        if j>=len(self.dates):return None
        x=self.dates[j]
        return None if end is not None and x>pd.Timestamp(end) else x


def rsi(s,n=14):
    d=s.diff();up=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean();dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean();return 100-100/(1+up/(dn+1e-12))


def make_features(m,d):
    x=m.spot_series.loc[:d].astype(float)
    if len(x)<80:return None
    e10=x.ewm(span=10,adjust=False).mean().iloc[-1];e20=x.ewm(span=20,adjust=False).mean().iloc[-1];e50=x.ewm(span=50,adjust=False).mean().iloc[-1];ret=x.pct_change()
    h20=x.shift(1).rolling(20).max().iloc[-1];l20=x.shift(1).rolling(20).min().iloc[-1];h60=x.shift(1).rolling(60).max().iloc[-1];l60=x.shift(1).rolling(60).min().iloc[-1]
    expiry=m.expiry_for(d,int(CFG['selection_dte_min']),int(CFG['selection_dte_max']));straddle=skew=dte=np.nan
    if expiry is not None:
        dte=(expiry-d).days;atm=m.nearest(d,expiry,'CE',m.spot[d])
        if atm is not None:
            c=m.price(d,expiry,atm,'CE');p=m.price(d,expiry,atm,'PE');
            if c and p:straddle=(c+p)/m.spot[d]
        kc=m.nearest(d,expiry,'CE',m.spot[d]+100);kp=m.nearest(d,expiry,'PE',m.spot[d]-100)
        if kc is not None and kp is not None:
            cp=m.price(d,expiry,kc,'CE');pp=m.price(d,expiry,kp,'PE')
            if cp and pp:skew=(pp-cp)/max(pp+cp,1e-9)
    c=m.asof_context(d); spot5=x.pct_change(5).iloc[-1]
    return {'ret1':ret.iloc[-1],'ret5':spot5,'ret10':x.pct_change(10).iloc[-1],'ret20':x.pct_change(20).iloc[-1],'ema10_gap':x.iloc[-1]/e10-1,'ema20_gap':x.iloc[-1]/e20-1,'ema50_gap':x.iloc[-1]/e50-1,'ema10_20_gap':e10/e20-1,'ema20_50_gap':e20/e50-1,'rsi14':rsi(x).iloc[-1]/100,'vol10':ret.rolling(10).std().iloc[-1]*math.sqrt(252),'vol20':ret.rolling(20).std().iloc[-1]*math.sqrt(252),'breakout20':x.iloc[-1]/h20-1,'breakdown20':x.iloc[-1]/l20-1,'range60':(x.iloc[-1]-l60)/max(h60-l60,1e-9),'slope20':np.polyfit(np.arange(20),x.iloc[-20:].values,1)[0]/x.iloc[-1],'atm_straddle_pct':straddle,'skew_proxy':skew,'dte':dte,'india_vix':c.get('india_vix',np.nan),'india_vix_ret5':c.get('india_vix_ret5',np.nan),'spx_ret5':c.get('spx_ret5',np.nan),'usd_inr_ret5':c.get('INR=X_ret5',np.nan),'brent_ret5':c.get('BZ=F_ret5',np.nan),'bank_rel5':c.get('nifty_bank_ret5',np.nan)-spot5,'midcap_rel5':c.get('nifty_midcap_ret5',np.nan)-spot5,'fii_net_z':c.get('fii_net_z',np.nan),'dii_net_z':c.get('dii_net_z',np.nan),'fii_fut_long_short':c.get('fii_fut_long_short',np.nan),'pcr':c.get('pcr',np.nan),'flow_sentiment':c.get('sentiment_score',np.nan),'news_sentiment':c.get('news_sentiment',np.nan),'news_volume_log':c.get('news_volume_log',np.nan),'news_neg_share':c.get('news_neg_share',np.nan)}


def regime_from_features(f):
    trend=float(f.get('ema20_50_gap',0) if pd.notna(f.get('ema20_50_gap',np.nan)) else 0)+2*float(f.get('slope20',0) if pd.notna(f.get('slope20',np.nan)) else 0);vol=float(f.get('vol20',0) if pd.notna(f.get('vol20',np.nan)) else 0);vix=float(f.get('india_vix',0) if pd.notna(f.get('india_vix',np.nan)) else 0);high=vol>=0.22 or vix>=20
    if abs(trend)<0.002:return 'RANGE_HIGH_VOL' if high else 'RANGE_LOW_VOL'
    if trend>0:return 'BULL_HIGH_VOL' if high else 'BULL_LOW_VOL'
    return 'BEAR_HIGH_VOL' if high else 'BEAR_LOW_VOL'


def legs(m,d,e,name):
    s=m.spot[d];w=float(CFG['wing_width_points']);sw=float(CFG['spread_width_points']);otm=float(CFG['otm_points']);spec={'LONG_CALL':[('CE',0,1)],'LONG_PUT':[('PE',0,1)],'BULL_CALL_SPREAD':[('CE',0,1),('CE',sw,-1)],'BEAR_PUT_SPREAD':[('PE',0,1),('PE',-sw,-1)],'BULL_PUT_SPREAD':[('PE',0,-1),('PE',-sw,1)],'BEAR_CALL_SPREAD':[('CE',0,-1),('CE',sw,1)],'LONG_STRADDLE':[('CE',0,1),('PE',0,1)],'LONG_STRANGLE':[('CE',otm,1),('PE',-otm,1)],'IRON_CONDOR':[('PE',-otm,-1),('PE',-otm-w,1),('CE',otm,-1),('CE',otm+w,1)],'IRON_BUTTERFLY':[('PE',0,-1),('PE',-w,1),('CE',0,-1),('CE',w,1)]}[name]
    out=[]
    for typ,off,sign in spec:
        k=m.nearest(d,e,typ,s+off);p=None if k is None else m.price(d,e,k,typ)
        if k is None or p is None:return None
        out.append((typ,k,sign,p))
    return out


def terminal_bounds(ll):
    strikes=sorted({x[1] for x in ll});pts=[strikes[0]-5000,*strikes,strikes[-1]+5000];entry=sum(sign*p for _,_,sign,p in ll);vals=[]
    for s in pts:vals.append(sum(sign*max((s-k) if typ=='CE' else (k-s),0.) for typ,k,sign,_ in ll)-entry)
    return min(vals),max(vals),entry


def costs(entry,exit_,signs,qty):
    buy=sell=0.
    for ep,xp,sign in zip(entry,exit_,signs):
        if sign>0:buy+=ep*qty;sell+=xp*qty
        else:sell+=ep*qty;buy+=xp*qty
    turnover=buy+sell;brokerage=2*len(signs)*float(CFG['brokerage_per_order']);txn=turnover*float(CFG['exchange_txn_pct']);sebi=turnover*float(CFG['sebi_turnover_pct']);stt=sell*float(CFG['stt_sell_pct']);stamp=buy*float(CFG['stamp_buy_pct']);gst=(brokerage+txn+sebi)*float(CFG['gst_pct']);return brokerage+txn+sebi+stt+stamp+gst


def outcome(m,d,strategy,end=None):
    entry=m.nth_date(d,1,end)
    if entry is None:return None
    e=m.expiry_for(entry,int(CFG['selection_dte_min']),int(CFG['selection_dte_max']))
    if e is None:return None
    exit_=m.nth_date(entry,int(CFG['holding_days']),end)
    if exit_ is None or exit_>=e:return None
    ll=legs(m,entry,e,strategy)
    if ll is None:return None
    ee=[];xx=[];signs=[];es=float(CFG['entry_slippage_points']);xs=float(CFG['exit_slippage_points'])
    for typ,k,sign,p in ll:
        xp=m.price(exit_,e,k,typ)
        if xp is None:return None
        ee.append(p+es if sign>0 else max(p-es,.01));xx.append(max(xp-xs,.01) if sign>0 else xp+xs);signs.append(sign)
    lot=lot_size(e);gross=sum(sign*(xp-ep) for xp,ep,sign in zip(xx,ee,signs))*lot;mn,mx,entry_value=terminal_bounds(ll);theo_min=mn*lot;theo_max=mx*lot
    if gross>theo_max+max(5.,abs(theo_max)*.02) or gross<theo_min-max(5.,abs(theo_min)*.02): raise RuntimeError(f'PAYOFF_INVARIANT_FAILED {d.date()} {strategy} gross={gross:.2f} bounds=({theo_min:.2f},{theo_max:.2f})')
    net=gross-costs(ee,xx,signs,lot);risk=max(0.,-theo_min)
    return {'entry':entry,'exit':exit_,'expiry':e,'net_1lot':net,'gross_1lot':gross,'risk_1lot':risk,'risk_return':net/max(risk,1.) ,'theoretical_max_profit':theo_max,'theoretical_max_loss':theo_min,'entry_value_points':entry_value}


def build_dataset(m):
    rows=[]
    for d in m.dates:
        if d<pd.Timestamp('2018-01-01'):continue
        f=make_features(m,d)
        if f is None:continue
        vals={};valid=0
        for s in STRATEGIES:
            r=outcome(m,d,s);vals[s]=np.nan if r is None else r['risk_return'];valid+=r is not None
        if valid>=8:rows.append({'signal_date':d,'regime':regime_from_features(f),**f,**vals})
    return pd.DataFrame(rows)


def fit(train):
    z=train.dropna(subset=STRATEGIES)
    if len(z)<int(CFG['min_training_samples']):raise RuntimeError(f'training rows {len(z)} < minimum {CFG["min_training_samples"]}')
    mc=CFG['model'];model=make_pipeline(SimpleImputer(strategy='median'),RandomForestRegressor(n_estimators=int(mc['n_estimators']),max_depth=int(mc['max_depth']),min_samples_leaf=int(mc['min_samples_leaf']),random_state=int(mc['random_state']),n_jobs=-1));model.fit(z[FEATURES],z[STRATEGIES]);return model,z


def execute(m,d,strategy,end):
    r=outcome(m,d,strategy,end)
    if r is None:return None
    if r['risk_1lot']<=0 or r['risk_1lot']>CAPITAL*float(CFG['max_trade_risk_pct']):return None
    lot=lot_size(r['expiry'])
    return {'signal_date':d,'entry_date':r['entry'],'exit_date':r['exit'],'expiry':r['expiry'],'strategy':strategy,'regime':regime_from_features(make_features(m,d)),'lot_size':lot,'lots':1,'qty':lot,'risk_cash':r['risk_1lot'],'gross_pnl':r['gross_1lot'],'costs':r['gross_1lot']-r['net_1lot'],'net_pnl':r['net_1lot'],'capital_return':r['net_1lot']/CAPITAL,'theoretical_max_profit':r['theoretical_max_profit'],'theoretical_max_loss':r['theoretical_max_loss']}


def metrics(t):
    if t.empty:return {'trades':0,'net_pnl':0.,'return':0.,'win_rate':0.,'max_drawdown':0.,'profit_factor':0.,'months':0,'median_monthly_return':0.,'mean_monthly_return':0.,'months_ge_30pct':0.,'months_nonnegative':0.,'worst_month':0.,'best_month':0.}
    eq=CAPITAL+t.net_pnl.cumsum();dd=eq/eq.cummax()-1;w=t.net_pnl>0;mo=t.groupby(t.exit_date.dt.to_period('M')).net_pnl.sum()/CAPITAL;g=t.loc[w,'net_pnl'].sum();l=-t.loc[~w,'net_pnl'].sum();return {'trades':int(len(t)),'net_pnl':float(t.net_pnl.sum()),'return':float(t.net_pnl.sum()/CAPITAL),'win_rate':float(w.mean()),'median_trade':float(t.net_pnl.median()),'worst_trade':float(t.net_pnl.min()),'best_trade':float(t.net_pnl.max()),'max_drawdown':float(dd.min()),'profit_factor':float(g/l) if l>0 else math.inf,'months':int(len(mo)),'median_monthly_return':float(mo.median()),'mean_monthly_return':float(mo.mean()),'months_ge_30pct':float((mo>=.30).mean()),'months_nonnegative':float((mo>=0).mean()),'worst_month':float(mo.min()),'best_month':float(mo.max())}


def load_context():
    p=ROOT/'data/cache/v5_market_context.parquet';flow_p=ROOT/'data/cache/v5_flow_context.parquet';news_p=ROOT/'data/cache/v5_news_daily.csv';context=pd.read_parquet(p) if p.exists() else pd.DataFrame()
    if flow_p.exists():
        flow=pd.read_parquet(flow_p);flow.date=pd.to_datetime(flow.date).dt.normalize();context=flow if context.empty else context.merge(flow,on='date',how='outer',suffixes=('','_flow'))
    news=pd.read_csv(news_p) if news_p.exists() else pd.DataFrame();return context,news


def main():
    opt=pd.read_parquet(ROOT/'data/cache/nifty_options_long.parquet');fut=pd.read_parquet(ROOT/'data/cache/nifty_futures_long.parquet');opt.columns=opt.columns.str.lower();fut.columns=fut.columns.str.lower();context,news=load_context();m=Market(opt,fut,context,news);out=ROOT/'v5/research';out.mkdir(parents=True,exist_ok=True);dataset=build_dataset(m);dataset.to_parquet(out/'candidate_dataset.parquet',index=False)
    windows=[];trades=[];learning=[]
    for wi,w in enumerate(CFG['walk_forward'],1):
        train_end=pd.Timestamp(w['train_end']);test_start=pd.Timestamp(w['test_start']);test_end=pd.Timestamp(m.dates[-1] if w['test_end']=='latest' else w['test_end']);train_start=train_end-pd.DateOffset(years=int(CFG['rolling_train_years']));train=dataset[(dataset.signal_date>train_start)&(dataset.signal_date<=train_end)].copy();test=dataset[(dataset.signal_date>=test_start)&(dataset.signal_date<=test_end)].copy();model,trainv=fit(train)
        priors={}
        for reg,z in trainv.groupby('regime'):
            if len(z)>=int(CFG['min_regime_samples']):
                med=z[STRATEGIES].median().sort_values(ascending=False);priors[reg]=med.to_dict();learning.append({'window':wi,'regime':reg,'samples':int(len(z)),'best_strategy':med.index[0],'median_risk_return':float(med.iloc[0]),'all_strategy_medians':{k:float(v) for k,v in med.items()}})
        pred=pd.DataFrame(model.predict(test[FEATURES]),index=test.index,columns=STRATEGIES);busy=None;choices={}
        for idx,row in test.sort_values('signal_date').iterrows():
            d=pd.Timestamp(row.signal_date)
            if busy is not None and d<=busy:continue
            p=pred.loc[idx].copy();reg=row.regime
            if reg in priors:p=(1-float(CFG['regime_blend_weight']))*p+float(CFG['regime_blend_weight'])*pd.Series(priors[reg])
            p=p.sort_values(ascending=False);top=p.index[0];edge=float(p.iloc[0]-p.iloc[1]);choices[str(d.date())]=top
            if float(p.iloc[0])<float(CFG['no_trade_threshold']) or edge<float(CFG['min_predicted_edge_vs_second']):continue
            tr=execute(m,d,top,test_end)
            if tr is None:continue
            tr.update({'window':wi,'predicted_risk_return':float(p[top]),'prediction_edge':edge,'train_regime_best_risk_return':float(priors.get(reg,{}).get(top,np.nan))});trades.append(tr);busy=pd.Timestamp(tr['exit_date'])
        windows.append({'window':wi,'train_start':str(train_start.date()),'train_end':str(train_end.date()),'test_start':str(test_start.date()),'test_end':str(test_end.date()),'training_rows':int(len(trainv)),'test_rows':int(len(test)),'choices':pd.Series(list(choices.values())).value_counts().to_dict()})
    ledger=pd.DataFrame(trades)
    if not ledger.empty:
        for c in ['signal_date','entry_date','exit_date','expiry']:ledger[c]=pd.to_datetime(ledger[c])
    summary={'strategy':'V5.1 Regime-Adaptive Multi-Strategy Options Engine','candidate_strategies':STRATEGIES,'model':'RandomForest multi-output regression blended with regime-specific historical median risk-return','capital':CAPITAL,'max_lots_per_trade':1,'overall_oos':metrics(ledger),'walk_forward':windows,'learning_points':learning,'validation':['Exactly one lot per trade.','Defined-risk payoffs are checked analytically before a trade enters the dataset.','Any payoff-bound violation aborts the run.','Context/news values are lagged by at least one session.','No overlapping positions.'],'data_context':['NIFTY option chain + futures','India VIX, NIFTY Bank/Midcap, S&P 500, S&P VIX, USDINR, Brent','FII/DII and participant OI/PCR/sentiment history when available','Daily historical news sentiment file when available; never future-filled']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,default=str));(out/'walk_forward_summary.json').write_text(json.dumps(windows,indent=2,default=str));(out/'regime_learning.json').write_text(json.dumps(learning,indent=2,default=str));ledger.to_csv(out/'oos_trade_ledger.csv',index=False);print(json.dumps(summary['overall_oos'],indent=2))


if __name__=='__main__':main()
