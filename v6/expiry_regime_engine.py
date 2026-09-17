from __future__ import annotations

import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / 'v6/config.yaml').read_text())
CAPITAL=float(CFG['capital']); TH=float(CFG['regime_threshold']); TOPK=int(CFG['top_k']); MINREG=int(CFG['min_regime_samples'])
FEATURES=['ret1','ret5','ret10','ret20','ema10_gap','ema20_gap','ema50_gap','ema10_20_gap','ema20_50_gap','rsi14','vol10','vol20','breakout20','breakdown20','range60','slope20','atm_straddle_pct','skew_proxy','dte','india_vix','india_vix_ret5','spx_ret5','usd_inr_ret5','brent_ret5','bank_rel5','midcap_rel5','news_sentiment','news_volume_log','news_neg_share']
UNSUPPORTED={'fii_net_z','dii_net_z','fii_fut_long_short','pcr','flow_sentiment'}

# Standard strategy set spanning the structures shown in the supplied Sensibull ready-made screens.
STRATEGIES=[
'LONG_CALL','LONG_PUT','SHORT_CALL','SHORT_PUT','BULL_CALL_SPREAD','BEAR_CALL_SPREAD','BULL_PUT_SPREAD','BEAR_PUT_SPREAD',
'LONG_STRADDLE','SHORT_STRADDLE','LONG_STRANGLE','SHORT_STRANGLE','LONG_IRON_CONDOR','SHORT_IRON_CONDOR','LONG_IRON_BUTTERFLY','SHORT_IRON_BUTTERFLY',
'BULL_BUTTERFLY','BEAR_BUTTERFLY','BULL_CONDOR','BEAR_CONDOR','CALL_RATIO_SPREAD','PUT_RATIO_SPREAD','CALL_RATIO_BACKSPREAD','PUT_RATIO_BACKSPREAD',
'STRAP','STRIP','RISK_REVERSAL','REVERSE_RISK_REVERSAL','LONG_SYNTHETIC_FUTURE','SHORT_SYNTHETIC_FUTURE','JADE_LIZARD','REVERSE_JADE_LIZARD',
'RANGE_FORWARD','LONG_CALENDAR_CALL','LONG_CALENDAR_PUT','BATMAN','DOUBLE_PLATEAU']


def load_market():
    from v5.run import Market
    o=pd.read_parquet(ROOT/'data/cache/nifty_options_long.parquet')
    f=pd.read_parquet(ROOT/'data/cache/nifty_futures_long.parquet')
    c=pd.read_parquet(ROOT/'data/cache/v5_market_context.parquet') if (ROOT/'data/cache/v5_market_context.parquet').exists() else None
    q=pd.read_parquet(ROOT/'data/cache/v5_flow_context.parquet') if (ROOT/'data/cache/v5_flow_context.parquet').exists() else None
    n=pd.read_csv(ROOT/'data/cache/v5_news_daily.csv') if (ROOT/'data/cache/v5_news_daily.csv').exists() else None
    if n is not None: n.columns=[str(x).strip().lower() for x in n.columns]
    if c is not None and 'date' not in c.columns and c.index.name=='date': c=c.reset_index()
    if q is not None and 'date' not in q.columns and q.index.name=='date': q=q.reset_index()
    return Market(o,f, pd.concat([x for x in [c,q] if x is not None],ignore_index=True) if c is not None or q is not None else None,n)


def next_expiry(m,d):
    es=sorted(set(e for e in m.expiries.get(pd.Timestamp(d),())) )
    future=[]
    # expiries are repeated on each option date; search the global expiry set instead
    for ed in set(sum((list(v) for v in m.expiries.values()),[])):
        if pd.Timestamp(ed)>pd.Timestamp(d): future.append(pd.Timestamp(ed))
    return min(future) if future else None


def spot_on(m,d):
    d=pd.Timestamp(d)
    if d in m.spot:return float(m.spot[d])
    z=[x for x in m.dates if x<=d]
    return float(m.spot[z[-1]]) if z else np.nan


def cycle_samples(m):
    exps=sorted(set(sum((list(v) for v in m.expiries.values()),[])))
    out=[]
    for i,e in enumerate(exps):
        if i==0: continue
        prev=exps[i-1]
        candidates=[d for d in m.dates if prev<d<e]
        if not candidates: continue
        d=pd.Timestamp(candidates[0]); s=spot_on(m,d); y=spot_on(m,e)
        if not np.isfinite(s) or not np.isfinite(y): continue
        out.append({'entry_date':d,'expiry':pd.Timestamp(e),'spot':s,'target_close':y,'target_return':y/s-1})
    return pd.DataFrame(out)


def feat(m,d):
    from v5 import run as base
    f=base.make_features(m,pd.Timestamp(d))
    if f is None:return None
    c=m.asof_context(pd.Timestamp(d))
    f['usd_inr_ret5']=c.get('usd_inr_ret5',c.get('INR=X_ret5',np.nan)); f['brent_ret5']=c.get('brent_ret5',c.get('BZ=F_ret5',np.nan))
    return {k: (float(f[k]) if pd.notna(f.get(k,np.nan)) else np.nan) for k in FEATURES}


def build_dataset(m,samples):
    rows=[]
    for r in samples.itertuples(index=False):
        f=feat(m,r.entry_date)
        if f is None: continue
        z=dict(f); z.update(entry_date=r.entry_date,expiry=r.expiry,spot=r.spot,target_close=r.target_close,target_return=r.target_return)
        rows.append(z)
    return pd.DataFrame(rows)


def regime(ret):
    if ret>=TH:return 'BULLISH'
    if ret<=-TH:return 'BEARISH'
    return 'RANGE_BOUND'


def fit_predict(train,test):
    model=make_pipeline(SimpleImputer(strategy='median'),RandomForestRegressor(n_estimators=int(CFG['n_estimators']),max_depth=int(CFG['max_depth']),min_samples_leaf=int(CFG['min_samples_leaf']),random_state=int(CFG['random_state']),n_jobs=-1))
    X=train[FEATURES]; y=train.target_return
    model.fit(X,y); pred=model.predict(test[FEATURES]); residual=y-model.predict(X); lo=float(np.nanquantile(residual,0.10)); hi=float(np.nanquantile(residual,0.90))
    out=test[['entry_date','expiry','spot','target_close','target_return']].copy(); out['pred_return']=pred; out['pred_close']=out.spot*(1+pred); out['pred_low']=out.spot*(1+pred+lo); out['pred_high']=out.spot*(1+pred+hi); out['pred_regime']=[regime(x) for x in pred]; out['residual_q10']=lo; out['residual_q90']=hi
    return model,out


def nearest(m,d,e,typ,target):
    return m.nearest(pd.Timestamp(d),pd.Timestamp(e),typ,float(target))


def strike(m,d,e,typ,spot,offset_pct=0.0,offset_points=0.0):
    target=spot*(1+offset_pct)+offset_points
    return nearest(m,d,e,typ,target)


def price(m,d,e,k,typ):
    if k is None:return None
    return m.price(pd.Timestamp(d),pd.Timestamp(e),float(k),typ)


def intrinsic(k,typ,spot): return max(spot-float(k),0.0) if typ=='CE' else max(float(k)-spot,0.0)


def build_legs(m,d,e,name):
    s=spot_on(m,d); step=float(CFG['strike_step']); p=float(CFG['otm_pct']);
    a=nearest(m,d,e,'CE',s); b=nearest(m,d,e,'PE',s)
    cu1=nearest(m,d,e,'CE',s+step); cu2=nearest(m,d,e,'CE',s+2*step); cu3=nearest(m,d,e,'CE',s+3*step); cu4=nearest(m,d,e,'CE',s+4*step)
    cl1=nearest(m,d,e,'CE',s-step); cl2=nearest(m,d,e,'CE',s-2*step); cl3=nearest(m,d,e,'CE',s-3*step); cl4=nearest(m,d,e,'CE',s-4*step)
    pu1=nearest(m,d,e,'PE',s+step); pu2=nearest(m,d,e,'PE',s+2*step); pu3=nearest(m,d,e,'PE',s+3*step); pu4=nearest(m,d,e,'PE',s+4*step)
    pl1=nearest(m,d,e,'PE',s-step); pl2=nearest(m,d,e,'PE',s-2*step); pl3=nearest(m,d,e,'PE',s-3*step); pl4=nearest(m,d,e,'PE',s-4*step)
    def L(typ,k,sgn,exp=e): return (exp,typ,k,sgn)
    specs={
      'LONG_CALL':[L('CE',a,1)], 'LONG_PUT':[L('PE',b,1)], 'SHORT_CALL':[L('CE',a,-1)], 'SHORT_PUT':[L('PE',b,-1)],
      'BULL_CALL_SPREAD':[L('CE',a,1),L('CE',cu1,-1)], 'BEAR_CALL_SPREAD':[L('CE',a,-1),L('CE',cu1,1)],
      'BULL_PUT_SPREAD':[L('PE',b,-1),L('PE',pl1,1)], 'BEAR_PUT_SPREAD':[L('PE',b,1),L('PE',pl1,-1)],
      'LONG_STRADDLE':[L('CE',a,1),L('PE',b,1)], 'SHORT_STRADDLE':[L('CE',a,-1),L('PE',b,-1)],
      'LONG_STRANGLE':[L('CE',cu1,1),L('PE',pl1,1)], 'SHORT_STRANGLE':[L('CE',cu1,-1),L('PE',pl1,-1)],
      'LONG_IRON_CONDOR':[L('PE',pl2,1),L('PE',pl1,-1),L('CE',cu1,-1),L('CE',cu2,1)],
      'SHORT_IRON_CONDOR':[L('PE',pl2,-1),L('PE',pl1,1),L('CE',cu1,1),L('CE',cu2,-1)],
      'LONG_IRON_BUTTERFLY':[L('PE',pl1,1),L('PE',b,-1),L('CE',a,-1),L('CE',cu1,1)],
      'SHORT_IRON_BUTTERFLY':[L('PE',pl1,-1),L('PE',b,1),L('CE',a,1),L('CE',cu1,-1)],
      'BULL_BUTTERFLY':[L('CE',a,1),L('CE',cu1,-2),L('CE',cu2,1)], 'BEAR_BUTTERFLY':[L('PE',b,1),L('PE',pl1,-2),L('PE',pl2,1)],
      'BULL_CONDOR':[L('CE',a,1),L('CE',cu1,-1),L('CE',cu2,-1),L('CE',cu3,1)], 'BEAR_CONDOR':[L('PE',b,1),L('PE',pl1,-1),L('PE',pl2,-1),L('PE',pl3,1)],
      'CALL_RATIO_SPREAD':[L('CE',cu1,1),L('CE',cu2,-2)], 'PUT_RATIO_SPREAD':[L('PE',pl1,1),L('PE',pl2,-2)],
      'CALL_RATIO_BACKSPREAD':[L('CE',cu1,-1),L('CE',cu2,2)], 'PUT_RATIO_BACKSPREAD':[L('PE',pl1,-1),L('PE',pl2,2)],
      'STRAP':[L('CE',a,2),L('PE',b,1)], 'STRIP':[L('CE',a,1),L('PE',b,2)],
      'RISK_REVERSAL':[L('CE',cu1,-1),L('PE',pl1,1)], 'REVERSE_RISK_REVERSAL':[L('CE',cu1,1),L('PE',pl1,-1)],
      'LONG_SYNTHETIC_FUTURE':[L('CE',a,1),L('PE',b,-1)], 'SHORT_SYNTHETIC_FUTURE':[L('CE',a,-1),L('PE',b,1)],
      'JADE_LIZARD':[L('PE',pl1,-1),L('CE',cu2,-1),L('CE',cu3,1)], 'REVERSE_JADE_LIZARD':[L('CE',cu1,-1),L('PE',pl2,-1),L('PE',pl3,1)],
      'RANGE_FORWARD':[L('CE',cu1,1),L('PE',pl1,-1)],
      'BATMAN':[L('CE',cu1,1),L('CE',cu2,-2),L('PE',pl1,1),L('PE',pl2,-2)],
      'DOUBLE_PLATEAU':[L('CE',a,1),L('CE',cu1,-2),L('CE',cu2,1),L('PE',b,1),L('PE',pl1,-2),L('PE',pl2,1)],
    }
    if name in ('LONG_CALENDAR_CALL','LONG_CALENDAR_PUT'):
        all_e=sorted(set(sum((list(v) for v in m.expiries.values()),[]))); nxt=[x for x in all_e if x>e];
        if not nxt:return None
        far=nxt[0]; typ='CE' if name.endswith('CALL') else 'PE'; k=cu1 if typ=='CE' else pl1
        return [L(typ,k,-1,e),L(typ,k,1,far)]
    return specs.get(name)


def transaction_cost(entry,exit_,qty,nlegs):
    buy=sell=0.0
    for ep,xp in zip(entry,exit_):
        if ep is None or xp is None:return np.nan
        # gross turnover approximation: half legs are not assumed; use absolute value of both sides.
        buy += ep*qty; sell += xp*qty
    turnover=buy+sell; brokerage=2*nlegs*float(CFG['brokerage_per_order']); txn=turnover*float(CFG['exchange_txn_pct']); sebi=turnover*float(CFG['sebi_turnover_pct']); stt=sell*float(CFG['stt_sell_pct']); stamp=buy*float(CFG['stamp_buy_pct']); gst=(brokerage+txn+sebi)*float(CFG['gst_pct']); return brokerage+txn+sebi+stt+stamp+gst


def evaluate_strategy(m,row,name):
    d=pd.Timestamp(row.entry_date); e=pd.Timestamp(row.expiry); exitd=e; spot_exit=spot_on(m,exitd); legs=build_legs(m,d,e,name)
    if not legs:return None
    qty=1; entry=[]; exitv=[]; signs=[]
    for ex,typ,k,sgn in legs:
        ep=price(m,d,ex,k,typ)
        if ep is None:return None
        if ex==e: xp=intrinsic(k,typ,spot_exit)
        else:
            xp=0.0 if ex==e else price(m,exitd,ex,k,typ)
            if xp is None: xp=intrinsic(k,typ,spot_exit)
        # Slippage on traded options only. At expiry intrinsic is the settlement value.
        ep=max(0.0,ep+float(CFG['entry_slippage_points'])); xp=max(0.0,xp-float(CFG['exit_slippage_points'])) if sgn>0 else max(0.0,xp+float(CFG['exit_slippage_points']))
        entry.append(ep); exitv.append(xp); signs.append(sgn)
    gross=sum((xp-ep)*sgn*qty for ep,xp,sgn in zip(entry,exitv,signs))
    # Costs based on leg direction.
    buy=sum((ep if sgn>0 else xp)*qty for ep,xp,sgn in zip(entry,exitv,signs)); sell=sum((xp if sgn>0 else ep)*qty for ep,xp,sgn in zip(entry,exitv,signs)); turnover=buy+sell
    brokerage=2*len(legs)*float(CFG['brokerage_per_order']); txn=turnover*float(CFG['exchange_txn_pct']); sebi=turnover*float(CFG['sebi_turnover_pct']); stt=sell*float(CFG['stt_sell_pct']); stamp=buy*float(CFG['stamp_buy_pct']); gst=(brokerage+txn+sebi)*float(CFG['gst_pct']); cost=brokerage+txn+sebi+stt+stamp+gst
    net=gross-cost
    return {'entry_date':d,'expiry':e,'strategy':name,'gross_pnl':gross,'cost':cost,'net_pnl':net,'return_on_capital':net/CAPITAL}


def all_outcomes(m,ds):
    rows=[]
    for r in ds.itertuples(index=False):
        for s in STRATEGIES:
            z=evaluate_strategy(m,r,s)
            if z is not None: rows.append(z|{'pred_regime':getattr(r,'pred_regime',None),'target_return':r.target_return})
    return pd.DataFrame(rows)


def strategy_map(train_outcomes):
    t=train_outcomes.copy(); t['actual_regime']=t.target_return.apply(regime)
    g=t.groupby(['actual_regime','strategy'])['return_on_capital'].agg(['median','mean','count']).reset_index()
    maps={}
    for rg in ['BULLISH','BEARISH','RANGE_BOUND']:
        x=g[g.actual_regime==rg].copy()
        if x.empty: maps[rg]=[]; continue
        x=x[x['count']>=MINREG].sort_values(['median','mean'],ascending=False)
        maps[rg]=x.head(TOPK).to_dict('records')
    return maps,g


def apply_selection(preds, outcomes, mapping):
    out=outcomes.merge(preds[['entry_date','pred_regime','pred_close','pred_low','pred_high']],on='entry_date',how='left')
    selected=[]
    for d,g in out.groupby('entry_date',sort=True):
        rg=str(g.pred_regime.iloc[0]); picks=[x['strategy'] for x in mapping.get(rg,[])]
        if not picks: continue
        q=g[g.strategy.isin(picks)].copy(); q['selected_rank']=q.strategy.map({s:i+1 for i,s in enumerate(picks)}); selected.append(q)
    return pd.concat(selected,ignore_index=True) if selected else pd.DataFrame()


def summarize(df):
    if df.empty:return {'trades':0,'net_pnl':0.0,'return':0.0,'win_rate':np.nan,'max_drawdown':0.0,'profit_factor':np.nan}
    x=df.sort_values(['entry_date','strategy']).copy(); pnl=x.net_pnl.astype(float); eq=CAPITAL+pnl.cumsum(); dd=eq/eq.cummax()-1; gains=pnl[pnl>0].sum(); losses=-pnl[pnl<0].sum();
    return {'trades':len(x),'net_pnl':float(pnl.sum()),'return':float(pnl.sum()/CAPITAL),'win_rate':float((pnl>0).mean()),'max_drawdown':float(dd.min()),'profit_factor':float(gains/losses) if losses>0 else np.inf,'median_trade':float(pnl.median())}


def run_wfo(m,ds):
    windows=[('2021-01-01','2021-12-31'),('2022-01-01','2022-12-31'),('2023-01-01','2023-12-31'),('2024-01-01','2024-12-31'),('2025-01-01','2025-12-31'),('2026-01-01','2026-03-31')]
    all_sel=[]; all_pred=[]; report=[]
    for i,(ts,te) in enumerate(windows,1):
        train=ds[(ds.entry_date>=pd.Timestamp(ts)-pd.DateOffset(years=int(CFG['rolling_train_years'])))&(ds.entry_date<=pd.Timestamp(ts)-pd.Timedelta(days=1))].copy(); test=ds[(ds.entry_date>=pd.Timestamp(ts))&(ds.entry_date<=pd.Timestamp(te))].copy()
        if len(train)<50 or test.empty: continue
        _,pred=fit_predict(train,test); pred.index=test.index; all_pred.append(pred); tr_out=all_outcomes(m,train); mp,_=strategy_map(tr_out); te_out=all_outcomes(m,test); sel=apply_selection(pred,te_out,mp); all_sel.append(sel); report.append({'window':i,'train_rows':len(train),'test_rows':len(test),'strategy_map':mp,'selected_summary':summarize(sel)})
    return pd.concat(all_pred,ignore_index=True),pd.concat(all_sel,ignore_index=True),report


def run_7030(m,ds):
    cut=ds.entry_date.min()+ (ds.entry_date.max()-ds.entry_date.min())*0.7; train=ds[ds.entry_date<=cut].copy(); test=ds[ds.entry_date>cut].copy(); _,pred=fit_predict(train,test); tr_out=all_outcomes(m,train); mp,g=strategy_map(tr_out); te_out=all_outcomes(m,test); sel=apply_selection(pred,te_out,mp); return pred,sel,mp,g


def run_new_oos(m,ds):
    start=pd.Timestamp(CFG['new_oos_start']); pre=ds[ds.entry_date<start].copy(); oos=ds[ds.entry_date>=start].copy(); _,pred=fit_predict(pre,oos); tr_out=all_outcomes(m,pre); mp,g=strategy_map(tr_out); oo=all_outcomes(m,oos); sel=apply_selection(pred,oo,mp); return pred,sel,mp,g


def main():
    outdir=ROOT/'v6/research'; outdir.mkdir(parents=True,exist_ok=True)
    m=load_market(); samples=cycle_samples(m); ds=build_dataset(m,samples); ds=ds[(ds.entry_date>=pd.Timestamp(CFG['pre_oos_start'])) & (ds.entry_date<=pd.Timestamp(CFG['new_oos_end'] if CFG['new_oos_end']!='latest' else m.dates[-1]))].copy()
    wfo_pred,wfo_sel,wfo_report=run_wfo(m,ds); split_pred,split_sel,split_map,split_matrix=run_7030(m,ds[ds.entry_date<pd.Timestamp(CFG['new_oos_start'])]); oos_pred,oos_sel,oos_map,oos_matrix=run_new_oos(m,ds)
    all_hist=all_outcomes(m,ds[ds.entry_date<pd.Timestamp(CFG['new_oos_start'])]); all_hist.to_csv(outdir/'all_strategy_outcomes_pre_oos.csv',index=False)
    wfo_pred.to_csv(outdir/'walk_forward_predictions.csv',index=False); wfo_sel.to_csv(outdir/'walk_forward_selected_trades.csv',index=False); split_pred.to_csv(outdir/'split_70_30_predictions.csv',index=False); split_sel.to_csv(outdir/'split_70_30_selected_trades.csv',index=False); oos_pred.to_csv(outdir/'new_oos_predictions.csv',index=False); oos_sel.to_csv(outdir/'new_oos_selected_trades.csv',index=False); split_matrix.to_csv(outdir/'regime_strategy_matrix_70_30_train.csv',index=False); oos_matrix.to_csv(outdir/'regime_strategy_matrix_new_oos_train.csv',index=False)
    summary={'sample_count':int(len(ds)),'prediction_definition':'Next NIFTY expiry close from the first trading session after the previous expiry; model predicts expiry return then converts it to predicted NIFTY close.','regime_threshold':TH,'strategies':STRATEGIES,'walk_forward':wfo_report,'walk_forward_selected_summary':summarize(wfo_sel),'70_30':{'train_rows':int((ds.entry_date<pd.Timestamp(CFG['new_oos_start'])).sum()),'selected_summary':summarize(split_sel),'mapping':split_map},'new_oos':{'start':str(CFG['new_oos_start']),'selected_summary':summarize(oos_sel),'mapping':oos_map},'selection_policy':f'For each predicted regime, recruit the top {TOPK} strategies by median return_on_capital learned only from the training sample; require {MINREG} regime samples.','oos_policy':'2026-04-01 onward is held out from all V6 model/strategy selection.'}
    (outdir/'summary.json').write_text(json.dumps(summary,indent=2,default=str)); (outdir/'predictions.csv').write_text(pd.concat([wfo_pred.assign(split='walk_forward'),split_pred.assign(split='70_30'),oos_pred.assign(split='new_oos')],ignore_index=True).to_csv(index=False))

if __name__=='__main__': main()
