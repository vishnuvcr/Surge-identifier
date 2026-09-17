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
CFG = yaml.safe_load((ROOT / "v5/config.yaml").read_text())
CAPITAL = float(CFG["capital"])
STRATEGIES = [
    "LONG_CALL", "LONG_PUT", "BULL_CALL_SPREAD", "BEAR_PUT_SPREAD",
    "BULL_PUT_SPREAD", "BEAR_CALL_SPREAD", "LONG_STRADDLE", "LONG_STRANGLE",
    "IRON_CONDOR", "IRON_BUTTERFLY",
]
FEATURES = [
    "ret1", "ret5", "ret10", "ret20", "ema10_gap", "ema20_gap", "ema50_gap",
    "ema10_20_gap", "ema20_50_gap", "rsi14", "vol10", "vol20", "breakout20",
    "breakdown20", "range60", "slope20", "atm_straddle_pct", "skew_proxy", "dte",
]


def lot_size(expiry):
    e = pd.Timestamp(expiry).normalize()
    if e < pd.Timestamp("2015-10-30"): return 25
    if e < pd.Timestamp("2021-08-01"): return 75
    if e < pd.Timestamp("2024-05-02"): return 50
    if e < pd.Timestamp("2024-11-20"): return 25
    if e < pd.Timestamp("2026-01-06"): return 75
    return 65


class Market:
    def __init__(self, options, futures):
        o = options.copy()
        for c in ["date", "expiry"]: o[c] = pd.to_datetime(o[c]).dt.normalize()
        o["option_type"] = o.option_type.astype(str).str.upper()
        o = o.drop_duplicates(["date", "expiry", "strike", "option_type"], keep="last")
        f = futures.copy()
        for c in ["date", "expiry"]: f[c] = pd.to_datetime(f[c]).dt.normalize()
        f = f[f.expiry >= f.date].sort_values(["date", "expiry"])
        f = f.drop_duplicates(["date", "expiry"], keep="last")
        spot = f.groupby("date", as_index=False).first()[["date", "close"]]
        self.spot = dict(zip(spot.date, spot.close.astype(float)))
        self.dates = tuple(pd.Timestamp(x) for x in sorted(set(o.date) & set(spot.date)))
        self.expiries = {}
        self.prices = {}
        self.strikes = {}
        for (d, e), g in o.groupby(["date", "expiry"], sort=False):
            d, e = pd.Timestamp(d), pd.Timestamp(e)
            self.expiries.setdefault(d, []).append(e)
            for typ, gg in g.groupby("option_type"):
                vals = gg.loc[gg.close > 0, ["strike", "close"]].dropna()
                self.strikes[(d, e, typ)] = np.array(sorted(vals.strike.astype(float).unique()), dtype=float)
                for k, c in vals.itertuples(index=False, name=None):
                    self.prices[(d, e, float(k), typ)] = float(c)
        for d in self.expiries: self.expiries[d] = tuple(sorted(self.expiries[d]))
        self.spot_series = pd.Series(self.spot).sort_index()
        self.date_index = {d: i for i, d in enumerate(self.dates)}

    def price(self, d, e, k, typ):
        v = self.prices.get((pd.Timestamp(d), pd.Timestamp(e), float(k), typ))
        return None if v is None or not np.isfinite(v) or v <= 0 else float(v)

    def nearest(self, d, e, typ, target):
        ks = self.strikes.get((pd.Timestamp(d), pd.Timestamp(e), typ))
        if ks is None or len(ks) == 0: return None
        return float(ks[np.argmin(np.abs(ks - target))])

    def expiry_for(self, d, lo, hi):
        d = pd.Timestamp(d)
        for e in self.expiries.get(d, ()):
            dte = (e - d).days
            if lo <= dte <= hi: return e
        return None

    def nth_date(self, d, n, end=None):
        i = self.date_index.get(pd.Timestamp(d))
        if i is None: return None
        j = i + n
        if j >= len(self.dates): return None
        x = self.dates[j]
        if end is not None and x > pd.Timestamp(end): return None
        return x


def rsi(s, n=14):
    d = s.diff(); up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100/(1 + up/(dn+1e-12))


def make_features(m, d):
    x = m.spot_series.loc[:d].astype(float)
    if len(x) < 60: return None
    e10 = x.ewm(span=10, adjust=False).mean().iloc[-1]; e20 = x.ewm(span=20, adjust=False).mean().iloc[-1]; e50 = x.ewm(span=50, adjust=False).mean().iloc[-1]
    ret = x.pct_change()
    h20 = x.shift(1).rolling(20).max().iloc[-1]; l20 = x.shift(1).rolling(20).min().iloc[-1]
    h60 = x.shift(1).rolling(60).max().iloc[-1]; l60 = x.shift(1).rolling(60).min().iloc[-1]
    expiry = m.expiry_for(d, int(CFG["selection_dte_min"]), int(CFG["selection_dte_max"]))
    straddle = np.nan; skew = np.nan; dte = np.nan
    if expiry is not None:
        dte = (expiry-d).days; atm = m.nearest(d, expiry, "CE", m.spot[d])
        if atm is not None:
            c = m.price(d, expiry, atm, "CE"); p = m.price(d, expiry, atm, "PE")
            if c and p: straddle = (c+p)/m.spot[d]
        c = m.nearest(d, expiry, "CE", m.spot[d]+100); p = m.nearest(d, expiry, "PE", m.spot[d]-100)
        if c is not None and p is not None:
            cp = m.price(d, expiry, c, "CE"); pp = m.price(d, expiry, p, "PE")
            if cp and pp: skew = (pp-cp)/max(pp+cp,1e-9)
    return {
        "ret1": ret.iloc[-1], "ret5": x.pct_change(5).iloc[-1], "ret10": x.pct_change(10).iloc[-1], "ret20": x.pct_change(20).iloc[-1],
        "ema10_gap": x.iloc[-1]/e10-1, "ema20_gap": x.iloc[-1]/e20-1, "ema50_gap": x.iloc[-1]/e50-1,
        "ema10_20_gap": e10/e20-1, "ema20_50_gap": e20/e50-1, "rsi14": rsi(x).iloc[-1]/100,
        "vol10": ret.rolling(10).std().iloc[-1]*math.sqrt(252), "vol20": ret.rolling(20).std().iloc[-1]*math.sqrt(252),
        "breakout20": x.iloc[-1]/h20-1, "breakdown20": x.iloc[-1]/l20-1, "range60": (x.iloc[-1]-l60)/max(h60-l60,1e-9),
        "slope20": np.polyfit(np.arange(20),x.iloc[-20:].values,1)[0]/x.iloc[-1], "atm_straddle_pct": straddle, "skew_proxy": skew, "dte": dte,
    }


def legs(m, d, e, name):
    s = m.spot[d]; w=float(CFG["wing_width_points"]); sw=float(CFG["spread_width_points"]); otm=float(CFG["otm_points"])
    spec={
      "LONG_CALL":[("CE",0,1)], "LONG_PUT":[("PE",0,1)],
      "BULL_CALL_SPREAD":[("CE",0,1),("CE",sw,-1)], "BEAR_PUT_SPREAD":[("PE",0,1),("PE",-sw,-1)],
      "BULL_PUT_SPREAD":[("PE",0,-1),("PE",-sw,1)], "BEAR_CALL_SPREAD":[("CE",0,-1),("CE",sw,1)],
      "LONG_STRADDLE":[("CE",0,1),("PE",0,1)], "LONG_STRANGLE":[("CE",otm,1),("PE",-otm,1)],
      "IRON_CONDOR":[("PE",-otm,-1),("PE",-otm-w,1),("CE",otm,-1),("CE",otm+w,1)],
      "IRON_BUTTERFLY":[("PE",0,-1),("PE",-w,1),("CE",0,-1),("CE",w,1)],
    }[name]
    out=[]
    for typ,off,sign in spec:
        k=m.nearest(d,e,typ,s+off)
        if k is None: return None
        p=m.price(d,e,k,typ)
        if p is None: return None
        out.append((typ,k,sign,p))
    return out


def max_loss(leglist):
    strikes=[z[1] for z in leglist]; lo=min(strikes)-400; hi=max(strikes)+400; grid=np.linspace(lo,hi,401)
    entry=sum(sign*p for _,_,sign,p in leglist)
    best=0.0
    for s in grid:
        payoff=sum(sign*max((s-k) if typ=="CE" else (k-s),0) for typ,k,sign,_ in leglist)
        best=min(best,payoff-entry)
    return max(-best,0.01)


def costs(entry, exit_, signs, qty):
    buy=sell=0.0
    for ep,xp,sign in zip(entry,exit_,signs):
        if sign>0: buy += ep*qty; sell += xp*qty
        else: sell += ep*qty; buy += xp*qty
    turnover=buy+sell; brokerage=2*len(signs)*float(CFG["brokerage_per_order"])
    txn=turnover*float(CFG["exchange_txn_pct"]); sebi=turnover*float(CFG["sebi_turnover_pct"]); stt=sell*float(CFG["stt_sell_pct"]); stamp=buy*float(CFG["stamp_buy_pct"])
    gst=(brokerage+txn+sebi)*float(CFG["gst_pct"])
    return brokerage+txn+sebi+stt+stamp+gst


def outcome(m, d, strategy, end=None):
    entry=m.nth_date(d,1,end); 
    if entry is None: return None
    e=m.expiry_for(entry,int(CFG["selection_dte_min"]),int(CFG["selection_dte_max"]))
    if e is None: return None
    exit_=m.nth_date(entry,int(CFG["holding_days"]),end)
    if exit_ is None or exit_>=e: return None
    ll=legs(m,entry,e,strategy)
    if ll is None: return None
    entry_exec=[]; exit_exec=[]; signs=[]
    es=float(CFG["entry_slippage_points"]); xs=float(CFG["exit_slippage_points"])
    for typ,k,sign,p in ll:
        xp=m.price(exit_,e,k,typ)
        if xp is None: return None
        entry_exec.append(p+es if sign>0 else max(p-es,0.01)); exit_exec.append(max(xp-xs,0.01) if sign>0 else xp+xs); signs.append(sign)
    lot=lot_size(e); gross=sum(sign*(xp-ep) for xp,ep,sign in zip(exit_exec,entry_exec,signs))*lot
    net=gross-costs(entry_exec,exit_exec,signs,lot); risk=max_loss(ll)*lot
    return {"entry":entry,"exit":exit_,"expiry":e,"net_1lot":net,"risk_1lot":max(risk,1.0),"risk_return":net/max(risk,1.0)}


def build_dataset(m):
    rows=[]; start=pd.Timestamp("2018-01-01")
    for d in m.dates:
        if d<start: continue
        f=make_features(m,d)
        if f is None: continue
        vals={}; valid=0
        for s in STRATEGIES:
            r=outcome(m,d,s)
            vals[s]=np.nan if r is None else r["risk_return"]
            valid += r is not None
        if valid>=8: rows.append({"signal_date":d,**f,**vals})
    return pd.DataFrame(rows)


def fit(train):
    z=train.dropna(subset=STRATEGIES)
    if len(z)<int(CFG["min_training_samples"]): raise RuntimeError(f"training rows {len(z)} < minimum {CFG['min_training_samples']}")
    mc=CFG["model"]
    model=make_pipeline(SimpleImputer(strategy="median"),RandomForestRegressor(n_estimators=int(mc["n_estimators"]),max_depth=int(mc["max_depth"]),min_samples_leaf=int(mc["min_samples_leaf"]),random_state=int(mc["random_state"]),n_jobs=-1))
    model.fit(z[FEATURES],z[STRATEGIES]); return model,z


def execute(m,d,strategy,end):
    r=outcome(m,d,strategy,end)
    if r is None:return None
    e=r["expiry"]; ll=legs(m,r["entry"],e,strategy); lot=lot_size(e); risk_lot=r["risk_1lot"]
    lots=max(1,min(int(CFG["max_lots"]),int(math.floor(CAPITAL*float(CFG["risk_per_trade_pct"])/risk_lot))))
    net=r["net_1lot"]*lots
    return {"signal_date":d,"entry_date":r["entry"],"exit_date":r["exit"],"expiry":e,"strategy":strategy,"lot_size":lot,"lots":lots,"qty":lots*lot,"risk_cash":risk_lot*lots,"net_pnl":net,"capital_return":net/CAPITAL}


def metrics(t):
    if t.empty:return {"trades":0,"net_pnl":0.0,"return":0.0,"win_rate":0.0,"max_drawdown":0.0,"profit_factor":0.0,"months":0,"median_monthly_return":0.0,"mean_monthly_return":0.0,"months_ge_30pct":0.0,"months_nonnegative":0.0,"worst_month":0.0,"best_month":0.0}
    eq=CAPITAL+t.net_pnl.cumsum(); dd=eq/eq.cummax()-1; w=t.net_pnl>0; mo=t.groupby(t.exit_date.dt.to_period("M")).net_pnl.sum()/CAPITAL; g=t.loc[w,"net_pnl"].sum(); l=-t.loc[~w,"net_pnl"].sum()
    return {"trades":int(len(t)),"net_pnl":float(t.net_pnl.sum()),"return":float(t.net_pnl.sum()/CAPITAL),"win_rate":float(w.mean()),"median_trade":float(t.net_pnl.median()),"worst_trade":float(t.net_pnl.min()),"best_trade":float(t.net_pnl.max()),"max_drawdown":float(dd.min()),"profit_factor":float(g/l) if l>0 else math.inf,"months":int(len(mo)),"median_monthly_return":float(mo.median()),"mean_monthly_return":float(mo.mean()),"months_ge_30pct":float((mo>=0.30).mean()),"months_nonnegative":float((mo>=0).mean()),"worst_month":float(mo.min()),"best_month":float(mo.max())}


def main():
    opt=pd.read_parquet(ROOT/"data/cache/nifty_options_long.parquet"); fut=pd.read_parquet(ROOT/"data/cache/nifty_futures_long.parquet")
    opt.columns=opt.columns.str.lower(); fut.columns=fut.columns.str.lower(); m=Market(opt,fut)
    out=ROOT/"v5/research"; out.mkdir(parents=True,exist_ok=True); cache=out/"candidate_dataset.parquet"
    if cache.exists(): dataset=pd.read_parquet(cache); dataset.signal_date=pd.to_datetime(dataset.signal_date).dt.normalize()
    else: dataset=build_dataset(m); dataset.to_parquet(cache,index=False)
    if dataset.empty: raise RuntimeError("candidate dataset empty")
    trades=[]; windows=[]
    for wi,w in enumerate(CFG["walk_forward"],1):
        train_end=pd.Timestamp(w["train_end"]); test_start=pd.Timestamp(w["test_start"]); test_end=m.dates[-1] if str(w["test_end"]).lower()=="latest" else pd.Timestamp(w["test_end"])
        train_start=train_end-pd.DateOffset(years=int(CFG["rolling_train_years"])); train=dataset[(dataset.signal_date>train_start)&(dataset.signal_date<=train_end)]; model,trainv=fit(train)
        test=dataset[(dataset.signal_date>=test_start)&(dataset.signal_date<=test_end)].copy(); pred=model.predict(test[FEATURES]); pred=pd.DataFrame(pred,columns=STRATEGIES,index=test.index)
        busy=None; choices={}
        for idx,row in test.sort_values("signal_date").iterrows():
            d=pd.Timestamp(row.signal_date)
            if busy is not None and d<=busy: continue
            p=pred.loc[idx].sort_values(ascending=False); top=p.index[0]; edge=float(p.iloc[0]-p.iloc[1])
            choice="NO_TRADE" if float(p.iloc[0])<float(CFG["no_trade_threshold"]) or edge<float(CFG["min_predicted_edge_vs_second"]) else top; choices[d]=choice
            if choice=="NO_TRADE": continue
            tr=execute(m,d,choice,test_end)
            if tr is None: continue
            tr["window"]=wi; tr["predicted_risk_return"]=float(p[top]); tr["prediction_edge"]=edge; tr["realized_1lot_risk_return"]=float(row[choice]); trades.append(tr); busy=pd.Timestamp(tr["exit_date"])
        vc=pd.Series(list(choices.values())).value_counts().to_dict(); windows.append({"window":wi,"train_start":str(train_start.date()),"train_end":str(train_end.date()),"test_start":str(test_start.date()),"test_end":str(test_end.date()),"training_rows":int(len(trainv)),"test_rows":int(len(test)),"choices":vc})
    ledger=pd.DataFrame(trades)
    if not ledger.empty:
        ledger["exit_date"]=pd.to_datetime(ledger.exit_date); ledger=ledger.sort_values("entry_date").reset_index(drop=True)
    summary={"strategy":"V5 Adaptive Multi-Strategy Options Engine","candidate_strategies":STRATEGIES,"model":"RandomForest multi-output regression on forward 5-session risk-adjusted return","capital":CAPITAL,"overall_oos":metrics(ledger),"walk_forward":windows,"notes":["Features use only information available on signal date; execution starts next session.","Strikes are selected from the actual historical option chain.","Only defined-risk structures and long options are included in V5.0.","NO_TRADE is allowed when predicted edge/confidence is weak.","V5.1 should add intraday path simulation, richer volatility-surface features, and strategy-specific exits."]}
    (out/"summary.json").write_text(json.dumps(summary,indent=2,default=str,allow_nan=True)); (out/"walk_forward_summary.json").write_text(json.dumps(windows,indent=2,default=str,allow_nan=True)); ledger.to_csv(out/"oos_trade_ledger.csv",index=False)
    print(json.dumps(summary,indent=2,default=str,allow_nan=True))

if __name__=="__main__": main()
