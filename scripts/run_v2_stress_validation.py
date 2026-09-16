from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OOS_START = pd.Timestamp('2023-03-09')
OOS_END = pd.Timestamp('2026-09-16')
CAPITAL = 100000.0
PARAMS = {'entry_weekday': 3, 'distance': 0.0125, 'width': 250, 'take_profit': 0.40, 'stop_loss': 1.0}
SLIP = 0.50
THRESHOLDS = [0.0, 3.0, 5.0, 7.0, 10.0, 12.0, 15.0, 20.0, 25.0]


def lot_size(expiry):
    e = pd.Timestamp(expiry).normalize()
    if e < pd.Timestamp('2015-10-30'): return 25
    if e < pd.Timestamp('2021-08-01'): return 75
    if e < pd.Timestamp('2024-05-02'): return 50
    if e < pd.Timestamp('2024-11-20'): return 25
    if e < pd.Timestamp('2026-01-06'): return 75
    return 65


def costs(entry, exit_px, lot):
    brokerage=80.0
    sell=(entry[1]+entry[2]+exit_px[0]+exit_px[3])*lot
    buy=(entry[0]+entry[3]+exit_px[1]+exit_px[2])*lot
    turnover=sum(abs(a)+abs(b) for a,b in zip(entry,exit_px))*lot
    txn=turnover*0.00035; stt=max(sell,0)*0.0010; stamp=max(buy,0)*0.00003; sebi=turnover*0.000001; gst=(brokerage+txn+sebi)*0.18
    return brokerage+stt+stamp+sebi+txn+gst


class Market:
    def __init__(self,o,f):
        o=o.copy(); o['date']=pd.to_datetime(o.date).dt.normalize(); o['expiry']=pd.to_datetime(o.expiry).dt.normalize(); o['option_type']=o.option_type.astype(str).str.upper(); o=o.drop_duplicates(['date','expiry','strike','option_type'],keep='last')
        f=f.copy(); f['date']=pd.to_datetime(f.date).dt.normalize(); f['expiry']=pd.to_datetime(f.expiry).dt.normalize(); f=f[f.expiry>=f.date].sort_values(['date','expiry']).drop_duplicates(['date','expiry'],keep='last')
        spot=f.sort_values(['date','expiry']).groupby('date',as_index=False).first()[['date','close']]
        self.spot=dict(zip(spot.date,spot.close.astype(float))); self.dates=tuple(pd.Timestamp(x) for x in sorted(set(o.date)&set(spot.date)))
        self.strikes={}; self.px={}; self.high={}; self.low={}; self.exp={}
        for (d,e),g in o.groupby(['date','expiry'],sort=False):
            d,e=pd.Timestamp(d),pd.Timestamp(e)
            if d in self.spot: self.exp.setdefault(d,[]).append(e); self.strikes[(d,e)]=np.asarray(sorted(g.strike.astype(float).unique()))
        for d in self.exp: self.exp[d]=tuple(sorted(self.exp[d]))
        for d,e,k,t,c,h,l in o[['date','expiry','strike','option_type','close','high','low']].itertuples(index=False,name=None):
            key=(pd.Timestamp(d),pd.Timestamp(e),float(k),str(t).upper())
            self.px[key]=float(c) if pd.notna(c) else np.nan; self.high[key]=float(h) if pd.notna(h) else np.nan; self.low[key]=float(l) if pd.notna(l) else np.nan
    def p(self,d,e,k,t):
        v=self.px.get((pd.Timestamp(d),pd.Timestamp(e),float(k),t)); return None if v is None or pd.isna(v) else float(v)
    def h(self,d,e,k,t):
        v=self.high.get((pd.Timestamp(d),pd.Timestamp(e),float(k),t)); return None if v is None or pd.isna(v) else float(v)
    def l(self,d,e,k,t):
        v=self.low.get((pd.Timestamp(d),pd.Timestamp(e),float(k),t)); return None if v is None or pd.isna(v) else float(v)
    def expiry_for(self,d):
        for e in self.exp.get(pd.Timestamp(d),()):
            if 1 <= (e-pd.Timestamp(d)).days <= 7: return e
        return None
    def choose(self,d,e):
        spot=self.spot.get(pd.Timestamp(d)); strikes=self.strikes.get((pd.Timestamp(d),pd.Timestamp(e)))
        if spot is None or strikes is None: return None
        p=strikes[strikes<=spot*(1-PARAMS['distance'])]; c=strikes[strikes>=spot*(1+PARAMS['distance'])]
        if len(p)==0 or len(c)==0: return None
        ps,cs=float(p[-1]),float(c[0]); pl=strikes[strikes<=ps-PARAMS['width']+1e-9]; cl=strikes[strikes>=cs+PARAMS['width']-1e-9]
        if len(pl)==0 or len(cl)==0: return None
        return float(pl[-1]),ps,cs,float(cl[0])


def run_threshold(m,min_credit):
    rows=[]; active=None
    for d in [d for d in m.dates if d.weekday()==3 and OOS_START<=d<=OOS_END]:
        if active is not None and d<=active: continue
        e=m.expiry_for(d)
        if e is None: continue
        legs=m.choose(d,e)
        if legs is None: continue
        lk=[(legs[0],'PE'),(legs[1],'PE'),(legs[2],'CE'),(legs[3],'CE')]
        entry_date=next((x for x in m.dates if x>d and x<=e),None)
        if entry_date is None or entry_date>OOS_END: continue
        en=[m.p(entry_date,e,k,t) for k,t in lk]
        if any(v is None or v<=0 for v in en): continue
        raw=en[1]+en[2]-en[0]-en[3]; eff=raw-4*SLIP
        if eff<min_credit: continue
        exit_date=reason=None; exit_mark=None; exit_px=None; min_bound=None; bound_day=None; sl_cross=False
        for day in [x for x in m.dates if entry_date<x<=e and x<=OOS_END]:
            vals=[m.p(day,e,k,t) for k,t in lk]
            hi=[m.h(day,e,k,t) for k,t in lk]; lo=[m.l(day,e,k,t) for k,t in lk]
            if not any(v is None for v in vals):
                mark=vals[1]+vals[2]-vals[0]-vals[3]; debit=-mark+4*SLIP
                if debit <= (1-PARAMS['take_profit'])*eff: exit_date,reason,exit_mark,exit_px=day,'take_profit',mark,vals; break
                if debit >= (1+PARAMS['stop_loss'])*eff: exit_date,reason,exit_mark,exit_px=day,'stop_loss',mark,vals; break
            if not any(v is None for v in hi+lo):
                bound_pnl=eff + lo[1] + lo[2] - hi[0] - hi[3] - 4*SLIP
                if min_bound is None or bound_pnl<min_bound: min_bound=float(bound_pnl); bound_day=day
                if (- (lo[1]+lo[2]-hi[0]-hi[3]) + 4*SLIP) >= (1+PARAMS['stop_loss'])*eff: sl_cross=True
        if exit_date is None:
            if e>OOS_END: continue
            vals=[m.p(e,e,k,t) for k,t in lk]
            if any(v is None for v in vals): continue
            exit_date,reason,exit_mark,exit_px=e,'expiry',vals[1]+vals[2]-vals[0]-vals[3],vals
        lot=lot_size(e); gross=(eff+exit_mark-4*SLIP)*lot; charge=costs(en,exit_px,lot); net=gross-charge
        rows.append({'signal_date':d,'entry_date':entry_date,'exit_date':exit_date,'expiry':e,'raw_credit':raw,'effective_credit':eff,'min_credit_filter':min_credit,'lot':lot,'max_loss_points':max(legs[1]-legs[0],legs[3]-legs[2])-raw,'net_pnl':net,'reason':reason,'conservative_bound_points':min_bound,'conservative_bound_date':bound_day,'possible_sl_cross':sl_cross})
        active=pd.Timestamp(exit_date)
    return pd.DataFrame(rows)


def main():
    m=Market(pd.read_parquet('data/cache/nifty_options_long.parquet'),pd.read_parquet('data/cache/nifty_futures_long.parquet'))
    fixed=run_threshold(m,10.0)
    sens=[]
    for th in THRESHOLDS:
        t=run_threshold(m,th)
        sens.append({'minimum_credit':th,'trades':len(t),'win_rate':float((t.net_pnl>0).mean()) if len(t) else 0,'net_pnl':float(t.net_pnl.sum()) if len(t) else 0,'return':float(t.net_pnl.sum()/CAPITAL) if len(t) else 0,'worst_trade':float(t.net_pnl.min()) if len(t) else 0,'max_drawdown':float((lambda e:(e/e.cummax()-1).min())(CAPITAL+t.net_pnl.cumsum())) if len(t) else 0})
    path=fixed.copy(); path['possible_sl_cross']=path['possible_sl_cross'].astype(bool)
    summary={'method':'V2 fixed-parameter OOS path-risk and minimum-credit robustness','period_start':str(OOS_START.date()),'period_end':str(OOS_END.date()),'parameters':PARAMS,'selection_slippage_per_leg':SLIP,'selected_minimum_credit':10.0,'oos_trades':len(path),'negative_conservative_bound_trades':int((path.conservative_bound_points<0).sum()),'negative_bound_rate':float((path.conservative_bound_points<0).mean()) if len(path) else 0,'deep_bound_trades':int((path.conservative_bound_points<=-0.5*path.max_loss_points).sum()) if len(path) else 0,'possible_sl_cross_trades':int(path.possible_sl_cross.sum()),'possible_sl_cross_rate':float(path.possible_sl_cross.mean()) if len(path) else 0,'all_eod_wins':bool((path.net_pnl>0).all()),'threshold_sensitivity':sens,'interpretation':'Conservative OHLC bound treats leg extremes as potentially adverse but does not prove simultaneity or order. A possible stop-loss crossing is a stress flag, not an executed fill.'}
    out=ROOT/'backtest/results/iron-condor-v2-stress'; out.mkdir(parents=True,exist_ok=True)
    (out/'v2_stress_summary.json').write_text(json.dumps(summary,indent=2,default=str)); path.to_csv(out/'v2_path_risk_trades.csv',index=False); pd.DataFrame(sens).to_csv(out/'v2_credit_threshold_sensitivity.csv',index=False)
    print(json.dumps(summary,indent=2,default=str))

if __name__=='__main__': main()
