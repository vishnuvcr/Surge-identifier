from __future__ import annotations

import itertools, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
CFG=yaml.safe_load((ROOT/'v4/config.yaml').read_text())
CAPITAL=float(CFG['capital'])

class Cost:
    def total(self, entry, exit_, qty):
        # One long option opened and closed = 2 orders.
        brokerage=2*float(CFG['brokerage_per_order'])
        buy_value=max(entry,0)*qty; sell_value=max(exit_,0)*qty
        turnover=(abs(entry)+abs(exit_))*qty
        txn=turnover*float(CFG['exchange_txn_pct'])
        sebi=turnover*float(CFG['sebi_turnover_pct'])
        stt=sell_value*float(CFG['stt_sell_pct'])
        stamp=buy_value*float(CFG['stamp_buy_pct'])
        gst=(brokerage+txn+sebi)*float(CFG['gst_pct'])
        return brokerage+txn+sebi+stt+stamp+gst

class Market:
    def __init__(self, options, futures):
        o=options.copy(); o['date']=pd.to_datetime(o.date).dt.normalize(); o['expiry']=pd.to_datetime(o.expiry).dt.normalize(); o['option_type']=o.option_type.astype(str).str.upper()
        o=o.drop_duplicates(['date','expiry','strike','option_type'],keep='last')
        f=futures.copy(); f['date']=pd.to_datetime(f.date).dt.normalize(); f['expiry']=pd.to_datetime(f.expiry).dt.normalize(); f=f[f.expiry>=f.date].sort_values(['date','expiry']).drop_duplicates(['date','expiry'],keep='last')
        spot=f.sort_values(['date','expiry']).groupby('date',as_index=False).first()[['date','close']]
        self.spot=dict(zip(spot.date,spot.close.astype(float))); self.dates=tuple(pd.Timestamp(x) for x in sorted(set(o.date)&set(spot.date)))
        self.exp={}; self.p={}
        for (d,e),g in o.groupby(['date','expiry'],sort=False):
            d,e=pd.Timestamp(d),pd.Timestamp(e); self.exp.setdefault(d,[]).append(e)
        for d in self.exp:self.exp[d]=tuple(sorted(self.exp[d]))
        for d,e,k,t,c in o[['date','expiry','strike','option_type','close']].itertuples(index=False,name=None):
            self.p[(pd.Timestamp(d),pd.Timestamp(e),float(k),str(t).upper())]=float(c) if pd.notna(c) else np.nan
        self.spot_series=pd.Series(self.spot).sort_index()
        self.fast={}

    def price(self,d,e,k,t):
        v=self.p.get((pd.Timestamp(d),pd.Timestamp(e),float(k),str(t).upper())); return None if v is None or pd.isna(v) else float(v)

    def expiry_for(self,d,min_dte,max_dte):
        for e in self.exp.get(pd.Timestamp(d),()):
            dte=(e-pd.Timestamp(d)).days
            if min_dte<=dte<=max_dte:return e
        return None


def indicators(spot, fast, slow, rsi_n, breakout_n, atr_n):
    s=spot.copy();
    emaf=s.ewm(span=fast,adjust=False).mean(); emas=s.ewm(span=slow,adjust=False).mean()
    delta=s.diff(); up=delta.clip(lower=0); down=(-delta.clip(upper=0)); rs=up.ewm(alpha=1/rsi_n,adjust=False).mean()/(down.ewm(alpha=1/rsi_n,adjust=False).mean()+1e-12); rsi=100-100/(1+rs)
    ret=s.pct_change(); atr_pct=ret.rolling(atr_n).std()*np.sqrt(252)*100
    hh=s.shift(1).rolling(breakout_n).max(); ll=s.shift(1).rolling(breakout_n).min()
    return emaf,emas,rsi,atr_pct,hh,ll


def select_option(m,d,e,direction,mstep):
    spot=m.spot[d]; step=float(CFG['strike_step_points']); atm=round(spot/step)*step; strike=atm-mstep*step if direction=='CE' else atm+mstep*step
    # mstep: +1 means one strike ITM, -1 OTM in directional sense.
    return strike


def run_config(m, params, start, end, collect=True):
    emaf,emas,rsi,atr,hh,ll=indicators(m.spot_series,int(params['ema_fast']),int(params['ema_slow']),int(params['rsi_period']),int(params['breakout_lookback']),int(params['atr_period']))
    dates=[d for d in m.dates if start<=d<=end]
    rows=[]; active_until=None
    for i,d in enumerate(dates):
        if active_until is not None and d<=active_until: continue
        if d not in emaf.index or any(pd.isna(x.loc[d]) for x in [emaf,emas,rsi,atr,hh,ll]): continue
        call_signal=(emaf.loc[d]>emas.loc[d]) and (m.spot[d]>hh.loc[d]) and (rsi.loc[d]>=params['rsi_call_min']) and (atr.loc[d]>=params['atr_pct_min'])
        put_signal=(emaf.loc[d]<emas.loc[d]) and (m.spot[d]<ll.loc[d]) and (rsi.loc[d]<=params['rsi_put_max']) and (atr.loc[d]>=params['atr_pct_min'])
        if not (call_signal or put_signal): continue
        direction='CE' if call_signal else 'PE'
        expiry=m.expiry_for(d,int(CFG['min_dte']),int(CFG['max_dte']))
        if expiry is None: continue
        entry_date=next((x for x in m.dates if x>d and x<=expiry),None)
        if entry_date is None or entry_date>end: continue
        strike=select_option(m,d,expiry,direction,int(params['moneyness_step']))
        ep=m.price(entry_date,expiry,strike,direction)
        if ep is None or ep<=0: continue
        # Risk-based quantity, capped by available capital.
        risk_cash=CAPITAL*float(params['position_risk_pct']); stop_loss=float(ep)*float(params['stop_pct']);
        if stop_loss<=0: continue
        qty=max(1,int(math.floor(risk_cash/((stop_loss+0.01)*max(1,m_lot_size(expiry))))))
        qty=min(qty,10)
        lot=m_lot_size(expiry); contracts=qty*lot
        exit_date=None; reason=None; xp=None
        max_hold=int(params['max_hold_days'])
        sim=[x for x in m.dates if entry_date<x<=expiry and (x-entry_date).days<=max_hold and x<=end]
        for day in sim:
            px=m.price(day,expiry,strike,direction)
            if px is None: continue
            if px>=ep*(1+float(params['target_pct'])): exit_date,reason,xp=day,'target',px; break
            if px<=ep*(1-float(params['stop_pct'])): exit_date,reason,xp=day,'stop',px; break
        if exit_date is None:
            last=sim[-1] if sim else None
            if last is None:
                continue
            xp=m.price(last,expiry,strike,direction)
            if xp is None: continue
            exit_date,reason=last,'time'
        cost=Cost().total(ep,xp,contracts); gross=(xp-ep)*contracts; net=gross-cost
        active_until=pd.Timestamp(exit_date)
        if collect: rows.append({'signal_date':d,'entry_date':entry_date,'exit_date':exit_date,'expiry':expiry,'direction':direction,'strike':strike,'entry_premium':ep,'exit_premium':xp,'lot':lot,'contracts':contracts,'gross_pnl':gross,'costs':cost,'net_pnl':net,'reason':reason,'capital_return':net/CAPITAL})
    return pd.DataFrame(rows)


def m_lot_size(expiry):
    e=pd.Timestamp(expiry).normalize()
    if e<pd.Timestamp('2015-10-30'):return 25
    if e<pd.Timestamp('2021-08-01'):return 75
    if e<pd.Timestamp('2024-05-02'):return 50
    if e<pd.Timestamp('2024-11-20'):return 25
    if e<pd.Timestamp('2026-01-06'):return 75
    return 65


def metrics(t):
    if t.empty:return {'trades':0,'net_pnl':0.0,'return':0.0,'win_rate':0.0}
    eq=CAPITAL+t.net_pnl.cumsum(); dd=eq/eq.cummax()-1; wins=(t.net_pnl>0)
    monthly=t.groupby(t.exit_date.dt.to_period('M')).net_pnl.sum()/CAPITAL
    gains=float(t.loc[wins,'net_pnl'].sum()); losses=float(-t.loc[~wins,'net_pnl'].sum())
    return {'trades':int(len(t)),'net_pnl':float(t.net_pnl.sum()),'return':float(t.net_pnl.sum()/CAPITAL),'win_rate':float(wins.mean()),'median_trade':float(t.net_pnl.median()),'worst_trade':float(t.net_pnl.min()),'best_trade':float(t.net_pnl.max()),'max_drawdown':float(dd.min()),'profit_factor':float(gains/losses) if losses else math.inf,'months':int(len(monthly)),'median_monthly_return':float(monthly.median()),'mean_monthly_return':float(monthly.mean()),'months_ge_30pct':float((monthly>=0.30).mean()),'months_nonnegative':float((monthly>=0).mean()),'worst_month':float(monthly.min()),'best_month':float(monthly.max())}


def score(t):
    if t.empty or len(t)<int(CFG['min_training_trades']):return -1e9
    m=metrics(t)
    return 3*m['median_monthly_return']+1*m['mean_monthly_return']+0.8*m['months_ge_30pct']+0.4*m['months_nonnegative']+0.25*m['win_rate']+0.2*m['return']+0.5*m['max_drawdown']


def main():
    opt=pd.read_parquet('data/cache/nifty_options_long.parquet'); fut=pd.read_parquet('data/cache/nifty_futures_long.parquet'); m=Market(opt,fut)
    windows=CFG['walk_forward']; all_rows=[]; wf=[]
    grid=itertools.product(CFG['max_hold_days'],CFG['moneyness_steps'],CFG['rsi_period'],CFG['rsi_call_min'],CFG['rsi_put_max'],CFG['ema_fast'],CFG['ema_slow'],CFG['breakout_lookback'],CFG['atrop_period'],CFG['atr_pct_min'],CFG['target_pct'],CFG['stop_pct'],CFG['position_risk_pct'])
    params_list=[]
    for hold,ms,rp,rc,rpmax,ef,es,bo,ap,amin,tp,sl,risk in grid:
        if ef>=es or rpmax>=rc: continue
        params_list.append({'max_hold_days':hold,'moneyness_step':ms,'rsi_period':rp,'rsi_call_min':rc,'rsi_put_max':rpmax,'ema_fast':ef,'ema_slow':es,'breakout_lookback':bo,'atr_period':ap,'atr_pct_min':amin,'target_pct':tp,'stop_pct':sl,'position_risk_pct':risk})
    for wi,w in enumerate(windows,1):
        te=pd.Timestamp(w['train_end']); ts=pd.Timestamp(w['test_start']); xe=m.dates[-1] if str(w['test_end']).lower()=='latest' else pd.Timestamp(w['test_end'])
        best=None; lb=[]
        for p in params_list:
            tr=run_config(m,p,m.dates[0],te)
            if len(tr)>=int(CFG['min_training_trades']): lb.append((score(tr),p,metrics(tr)))
        lb.sort(key=lambda x:x[0],reverse=True)
        if not lb: raise RuntimeError(f'No V4 configuration met minimum training trades in window {wi}')
        best=lb[0][1]; test=run_config(m,best,ts,xe)
        tm=metrics(test); wf.append({'window':wi,'train_end':str(te.date()),'test_start':str(ts.date()),'test_end':str(xe.date()),'selected':best,'train_score':float(lb[0][0]),'test':tm})
        if not test.empty: all_rows.append(test)
    trades=pd.concat(all_rows,ignore_index=True) if all_rows else pd.DataFrame()
    out=Path('v4/research'); out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(wf).to_json(out/'walk_forward_summary.json',orient='records',indent=2)
    trades.to_csv(out/'oos_trade_ledger.csv',index=False)
    summary={'strategy':'V4 NIFTY directional option buying','accounting':'Long option P&L = exit premium - entry premium - realistic transaction costs; no fabricated intraday fills.','capital':CAPITAL,'target_monthly_return':CFG['monthly_target_return'],'overall_oos':metrics(trades),'windows':wf}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,default=str,allow_nan=True))
    print(json.dumps(summary,indent=2,default=str,allow_nan=True))
if __name__=='__main__':main()
