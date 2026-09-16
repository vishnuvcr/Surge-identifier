from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.iron_condor_v2.yaml").read_text())
CAPITAL = float(CFG["capital"])


def lot_size(expiry):
    e = pd.Timestamp(expiry).normalize()
    if e < pd.Timestamp("2015-10-30"): return 25
    if e < pd.Timestamp("2021-08-01"): return 75
    if e < pd.Timestamp("2024-05-02"): return 50
    if e < pd.Timestamp("2024-11-20"): return 25
    if e < pd.Timestamp("2026-01-06"): return 75
    return 65


def cost_model(entry, exit_px, lot):
    brokerage = 80.0
    sell_value = (entry[1] + entry[2] + exit_px[0] + exit_px[3]) * lot
    buy_value = (entry[0] + entry[3] + exit_px[1] + exit_px[2]) * lot
    turnover = sum(abs(a) + abs(b) for a, b in zip(entry, exit_px)) * lot
    txn = turnover * 0.00035
    stt = max(sell_value, 0.0) * 0.0010
    stamp = max(buy_value, 0.0) * 0.00003
    sebi = turnover * 0.000001
    gst = (brokerage + txn + sebi) * 0.18
    return brokerage + stt + stamp + sebi + txn + gst


class Market:
    def __init__(self, options, futures):
        o = options.copy()
        o["date"] = pd.to_datetime(o.date).dt.normalize()
        o["expiry"] = pd.to_datetime(o.expiry).dt.normalize()
        o["option_type"] = o.option_type.astype(str).str.upper()
        o = o.drop_duplicates(["date", "expiry", "strike", "option_type"], keep="last")
        f = futures.copy()
        f["date"] = pd.to_datetime(f.date).dt.normalize()
        f["expiry"] = pd.to_datetime(f.expiry).dt.normalize()
        f = f[f.expiry >= f.date].sort_values(["date", "expiry"]).drop_duplicates(["date", "expiry"], keep="last")
        spot = f.groupby("date", as_index=False).first()[["date", "close"]]
        self.spot = dict(zip(spot.date, spot.close.astype(float)))
        self.dates = tuple(pd.Timestamp(x) for x in sorted(set(o.date) & set(spot.date)))
        self.exp = {}
        self.strikes = {}
        self.px = {}
        self.high = {}
        self.low = {}
        for (d, e), g in o.groupby(["date", "expiry"], sort=False):
            d, e = pd.Timestamp(d), pd.Timestamp(e)
            if d in self.spot:
                self.exp.setdefault(d, []).append(e)
                self.strikes[(d, e)] = np.asarray(sorted(g.strike.astype(float).unique()))
        for d in self.exp:
            self.exp[d] = tuple(sorted(self.exp[d]))
        for d, e, k, t, close, high, low in o[["date", "expiry", "strike", "option_type", "close", "high", "low"]].itertuples(index=False, name=None):
            key = (pd.Timestamp(d), pd.Timestamp(e), float(k), str(t).upper())
            if pd.notna(close): self.px[key] = float(close)
            if pd.notna(high): self.high[key] = float(high)
            if pd.notna(low): self.low[key] = float(low)

    def p(self, d, e, k, t):
        return self.px.get((pd.Timestamp(d), pd.Timestamp(e), float(k), str(t).upper()))

    def h(self, d, e, k, t):
        return self.high.get((pd.Timestamp(d), pd.Timestamp(e), float(k), str(t).upper()))

    def l(self, d, e, k, t):
        return self.low.get((pd.Timestamp(d), pd.Timestamp(e), float(k), str(t).upper()))

    def expiry(self, d):
        for e in self.exp.get(pd.Timestamp(d), ()):
            dte = (e - pd.Timestamp(d)).days
            if int(CFG["min_days_to_expiry"]) <= dte <= int(CFG["max_days_to_expiry"]):
                return e
        return None

    def choose(self, d, e, distance, width):
        s = self.spot.get(pd.Timestamp(d))
        ks = self.strikes.get((pd.Timestamp(d), pd.Timestamp(e)))
        if s is None or ks is None or len(ks) < 8: return None
        puts = ks[ks <= s * (1 - distance)]
        calls = ks[ks >= s * (1 + distance)]
        if not len(puts) or not len(calls): return None
        ps, cs = float(puts[-1]), float(calls[0])
        pls = ks[ks <= ps - width + 1e-9]
        cls = ks[ks >= cs + width - 1e-9]
        if not len(pls) or not len(cls): return None
        return float(pls[-1]), ps, cs, float(cls[0])


def score(t):
    if t.empty: return -1e9
    t = t.sort_values("exit")
    w = t.groupby(t.exit.dt.to_period("W")).net.sum() / CAPITAL
    eq = CAPITAL + t.net.cumsum()
    dd = eq / eq.cummax() - 1.0
    win = float((t.net > 0).mean())
    return 2*float(w.median()) + float(w.mean()) + 0.10*float((w >= 0.05).mean()) + 0.25*float((w >= 0).mean()) + 0.10*win + 0.50*float(w.min()) + 0.50*float(dd.min())


def run_config(m, p, start, end, slip=0.5):
    rows=[]
    active_until=None
    dates=[d for d in m.dates if d.weekday()==int(p["entry_weekday"]) and start<=d<=end]
    for signal in dates:
        if active_until is not None and signal <= active_until: continue
        exp=m.expiry(signal)
        if exp is None: continue
        legs=m.choose(signal, exp, float(p["distance"]), int(p["width"]))
        if legs is None: continue
        lk=[(legs[0],'PE'),(legs[1],'PE'),(legs[2],'CE'),(legs[3],'CE')]
        entry_date=next((d for d in m.dates if d>signal and d<=exp),None)
        if entry_date is None or entry_date>end: continue
        entry=[m.p(entry_date,exp,k,t) for k,t in lk]
        if any(v is None or v<=0 for v in entry): continue
        raw_credit=entry[1]+entry[2]-entry[0]-entry[3]
        effective_credit=raw_credit-4*slip
        if raw_credit<=0 or effective_credit<float(p["minimum_credit"]): continue
        lot=lot_size(exp)
        max_loss=max(legs[1]-legs[0],legs[3]-legs[2])-raw_credit
        if max_loss<=0: continue
        exit_date=None; reason='expiry'; exit_mark=None
        for day in [d for d in m.dates if entry_date<d<=exp and d<=end]:
            vals=[m.p(day,exp,k,t) for k,t in lk]
            if any(v is None for v in vals): continue
            mark=vals[1]+vals[2]-vals[0]-vals[3]  # current condor debit/liability
            exit_debit=mark+4*slip
            tp_debit=(1-float(p['take_profit']))*effective_credit
            sl_debit=(1+float(p['stop_loss']))*effective_credit
            if exit_debit<=tp_debit:
                exit_date,reason,exit_mark=day,'take_profit',mark; break
            if exit_debit>=sl_debit:
                exit_date,reason,exit_mark=day,'stop_loss',mark; break
        if exit_date is None:
            if exp>end: continue
            exit_date=exp
            vals=[m.p(exp,exp,k,t) for k,t in lk]
            if any(v is None for v in vals): continue
            exit_mark=vals[1]+vals[2]-vals[0]-vals[3]
        exit_px=[m.p(exit_date,exp,k,t) for k,t in lk]
        if any(v is None for v in exit_px): continue
        # Corrected IC accounting: entry credit - exit debit - modeled slippage/costs.
        gross=(effective_credit-exit_mark-4*slip)*lot
        net=gross-cost_model(entry,exit_px,lot)
        rows.append({'signal':signal,'entry':entry_date,'exit':exit_date,'expiry':exp,'weekday':signal.weekday(),'lot':lot,'put_long':legs[0],'put_short':legs[1],'call_short':legs[2],'call_long':legs[3],'raw_credit':raw_credit,'effective_credit':effective_credit,'net':net,'gross':gross,'costs':cost_model(entry,exit_px,lot),'reason':reason,'max_loss':max_loss*lot,'return_max_loss':net/(max_loss*lot)})
        active_until=exit_date
    return pd.DataFrame(rows)


def metrics(t):
    if t.empty: return {'trades':0,'win_rate':0.0,'net_pnl':0.0,'return':0.0,'max_drawdown':0.0}
    t=t.sort_values(['exit','entry']).reset_index(drop=True)
    eq=CAPITAL+t.net.cumsum(); dd=eq/eq.cummax()-1
    gains=float(t.loc[t.net>0,'net'].sum()); losses=float(-t.loc[t.net<0,'net'].sum())
    return {'trades':int(len(t)),'win_rate':float((t.net>0).mean()),'net_pnl':float(t.net.sum()),'return':float(t.net.sum()/CAPITAL),'median_trade':float(t.net.median()),'worst_trade':float(t.net.min()),'best_trade':float(t.net.max()),'profit_factor':float(gains/losses) if losses else None,'max_drawdown':float(dd.min()),'take_profit_trades':int((t.reason=='take_profit').sum()),'stop_loss_trades':int((t.reason=='stop_loss').sum()),'expiry_trades':int((t.reason=='expiry').sum())}


def select(m,start,end):
    base=[]
    for wd,d,w,tp,sl,mc in itertools.product(CFG['entry_weekdays'],CFG['short_distance_pct'],CFG['wing_width_points'],CFG['take_profit_credit_pct'],CFG['stop_loss_credit_multiple'],CFG['minimum_credit_points']):
        p={'entry_weekday':wd,'distance':d,'width':w,'take_profit':tp,'stop_loss':sl,'minimum_credit':mc}
        t=run_config(m,p,start,end,CFG['selection_slippage_per_leg_points'])
        if len(t)>=int(CFG['min_training_trades']): base.append((score(t),p,t))
    if not base: raise RuntimeError('No corrected V2 configuration met minimum training trades')
    base.sort(key=lambda x:x[0],reverse=True)
    return base[0],base[:25]


def main():
    o=pd.read_parquet('data/cache/nifty_options_long.parquet'); f=pd.read_parquet('data/cache/nifty_futures_long.parquet'); m=Market(o,f); dates=list(m.dates)
    out=ROOT/'backtest/results/iron-condor-v2-corrected'; out.mkdir(parents=True,exist_ok=True)
    schedule=CFG['walk_forward_schedule']; windows=[]; all_oos=[]
    for i,s in enumerate(schedule,1):
        train_end=pd.Timestamp(s['train_end']); test_start=pd.Timestamp(s['test_start']); test_end=max(dates) if str(s['test_end']).lower()=='latest' else pd.Timestamp(s['test_end'])
        train_start=dates[0]
        selected,top=select(m,train_start,train_end)
        _,p,train_t=selected
        oos_t=run_config(m,p,test_start,test_end,CFG['selection_slippage_per_leg_points'])
        windows.append({'window':i,'train_start':str(train_start.date()),'train_end':str(train_end.date()),'test_start':str(test_start.date()),'test_end':str(test_end.date()),'selected':p,'selection_score':float(selected[0]),'train':metrics(train_t),'oos':metrics(oos_t)})
        if not oos_t.empty: all_oos.append(oos_t.assign(window=i))
    combined=pd.concat(all_oos,ignore_index=True) if all_oos else pd.DataFrame()
    # Baseline 70/30 holdout, for direct comparison with the earlier V2 experiment.
    split=int(len(dates)*0.70); train_end=dates[split-1]; test_start=dates[split]
    selected,top=select(m,dates[0],train_end); _,p70,train70=selected; oos70=run_config(m,p70,test_start,dates[-1],CFG['selection_slippage_per_leg_points'])
    summary={'strategy':'Iron Condor V2 corrected','accounting':'P&L = entry credit - exit condor debit - entry/exit slippage - modeled costs','data_start':str(dates[0].date()),'data_end':str(dates[-1].date()),'walk_forward_windows':len(windows),'walk_forward_oos':metrics(combined),'baseline_70_30':{'train_start':str(dates[0].date()),'train_end':str(train_end.date()),'oos_start':str(test_start.date()),'oos_end':str(dates[-1].date()),'selected':p70,'train':metrics(train70),'oos':metrics(oos70)},'windows':windows}
    (out/'v2_corrected_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=True)); (out/'walk_forward_results.csv').write_text(pd.DataFrame([{'window':w['window'],'train_end':w['train_end'],'test_start':w['test_start'],'test_end':w['test_end'],'selection_score':w['selection_score'],'oos_trades':w['oos']['trades'],'oos_win_rate':w['oos']['win_rate'],'oos_net_pnl':w['oos']['net_pnl'],'oos_return':w['oos']['return'],'oos_max_drawdown':w['oos']['max_drawdown']} for w in windows]).to_csv(index=False))
    combined.to_csv(out/'walk_forward_trades.csv',index=False); oos70.to_csv(out/'baseline_oos_trades.csv',index=False); train70.to_csv(out/'baseline_train_trades.csv',index=False)
    stress=[]
    for slip in CFG['slippage_per_leg_points']:
        st=run_config(m,p70,test_start,dates[-1],float(slip)); sm=metrics(st); stress.append({'slippage_per_leg':float(slip),**sm})
    pd.DataFrame(stress).to_csv(out/'slippage_sensitivity.csv',index=False)
    (out/'selected_parameters_70_30.json').write_text(json.dumps(p70,indent=2)); (out/'top25_70_30.json').write_text(json.dumps([{'score':float(a),'params':b,'trades':len(c)} for a,b,c in top],indent=2))
    print(json.dumps(summary,indent=2,allow_nan=True))

if __name__=='__main__': main()
