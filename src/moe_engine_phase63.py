from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.moe_engine import EVENT_FEATURES, EXPERT_FEATURES, EXPERT_SEEDS, regime_labels

THRESHOLDS = (0.025, 0.03, 0.04, 0.05)
CPR_EXPERT = "cpr_camarilla"
CPR_FEATURES = [
    "cpr_p", "cpr_bc", "cpr_tc", "cpr_width_pct", "cpr_width_z20", "cpr_narrow",
    "cpr_trend", "cpr_pos", "cpr_dist_p", "cpr_dist_tc", "cpr_dist_bc",
    "cpr_dist_prev_high", "cpr_dist_prev_low", "cpr_camarilla_r3", "cpr_camarilla_r4",
    "cpr_camarilla_s3", "cpr_camarilla_s4", "cpr_dist_r3", "cpr_dist_r4",
    "cpr_dist_s3", "cpr_dist_s4", "cpr_gap", "cpr_breakout_pressure",
    "ret1", "ret5", "atr_pct", "rsi14", "relvol5", "relvol20",
    "market_ret5", "market_breadth", "cross_ret_rank", "cross_vol_rank",
]


def keys(levels: Iterable[float]) -> list[str]:
    return [str(float(x)).replace(".", "p").lstrip("0") for x in levels]


def add_cpr_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build CPR/Camarilla levels only from the completed prior session."""
    out = df.sort_values(["symbol", "date"]).copy()
    g = out.groupby("symbol", sort=False)
    ph = g["high"].shift(1).astype(float)
    pl = g["low"].shift(1).astype(float)
    pc = g["close"].shift(1).astype(float)
    pp = (ph + pl + pc) / 3.0
    bc = (ph + pl) / 2.0
    tc = 2.0 * pp - bc
    width = (tc - bc).abs() / pp.replace(0, np.nan)
    # Camarilla standard levels from prior day's range and close.
    rng = ph - pl
    r3 = pc + rng * 1.1 / 4.0
    r4 = pc + rng * 1.1 / 2.0
    s3 = pc - rng * 1.1 / 4.0
    s4 = pc - rng * 1.1 / 2.0
    out["cpr_p"], out["cpr_bc"], out["cpr_tc"] = pp, bc, tc
    out["cpr_width_pct"] = width
    out["cpr_width_z20"] = (width - width.groupby(out.symbol).transform(lambda x: x.rolling(20, min_periods=5).mean().shift(1))) / width.groupby(out.symbol).transform(lambda x: x.rolling(20, min_periods=5).std().shift(1).replace(0, np.nan))
    out["cpr_narrow"] = (out["cpr_width_z20"] <= -0.5).astype(float)
    out["cpr_trend"] = (pp - g["close"].shift(2).astype(float)) / pc.replace(0, np.nan)
    mid = (bc + tc) / 2.0
    out["cpr_pos"] = (out["close"] - mid) / pp.replace(0, np.nan)
    for name, level in [("p", pp), ("tc", tc), ("bc", bc), ("prev_high", ph), ("prev_low", pl), ("r3", r3), ("r4", r4), ("s3", s3), ("s4", s4)]:
        out[f"cpr_dist_{name}"] = (out["close"] - level) / out["close"].replace(0, np.nan)
    out["cpr_camarilla_r3"], out["cpr_camarilla_r4"] = r3, r4
    out["cpr_camarilla_s3"], out["cpr_camarilla_s4"] = s3, s4
    out["cpr_gap"] = (out["open"] - mid) / out["open"].replace(0, np.nan)
    out["cpr_breakout_pressure"] = np.maximum(out["close"] - tc, 0) / out["close"].replace(0, np.nan) + np.maximum(out["close"] - r3, 0) / out["close"].replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def add_targets(df: pd.DataFrame, stop_loss: float = 0.0125, thresholds: tuple[float, ...] = THRESHOLDS) -> pd.DataFrame:
    out = df.sort_values(["symbol", "date"]).copy()
    g = out.groupby("symbol", sort=False)
    op, hi, lo, cl = [g[c].shift(-1).astype(float) for c in ["open", "high", "low", "close"]]
    out["next_mfe"], out["next_mae"], out["next_close_ret"] = hi / op - 1, lo / op - 1, cl / op - 1
    out["label_date"] = g["date"].shift(-1)
    stop = out["next_mae"] <= -float(stop_loss)
    for level, key in zip(thresholds, keys(thresholds)):
        hit = out["next_mfe"] >= float(level)
        out[f"hit_{key}"] = (hit & ~stop).astype(int)
        out[f"stop_{key}"] = stop.astype(int)
    return out


@dataclass
class Expert63:
    name: str
    features: list[str]
    hit_model: ExtraTreesClassifier
    stop_model: ExtraTreesClassifier
    excursion_model: ExtraTreesRegressor
    hit_keys: list[str]
    training_rows: int
    base_rate: dict[str, float]


@dataclass
class Router63:
    model: LogisticRegression
    scaler: StandardScaler
    feature_names: list[str]


def expert_sets() -> dict[str, list[str]]:
    sets = {k: list(v) for k, v in EXPERT_FEATURES.items()}
    sets["universal"] = list(dict.fromkeys(sum((list(v) for v in EXPERT_FEATURES.values()), []) + CPR_FEATURES + list(EVENT_FEATURES)))
    sets[CPR_EXPERT] = CPR_FEATURES
    return sets


def _frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[[c for c in cols if c in df.columns]].replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)


def train_experts(train_df: pd.DataFrame, thresholds=THRESHOLDS, stop_loss=0.0125, seed=42, min_positive=20) -> dict[str, Expert63]:
    work = add_cpr_features(train_df)
    work["regime_code"] = regime_labels(work)
    hkeys = keys(thresholds)
    hit_cols = [f"hit_{k}" for k in hkeys]
    bundles = {}
    for name, base in expert_sets().items():
        if name == "event" and (not EVENT_FEATURES or float(work[EVENT_FEATURES].abs().sum().sum()) == 0):
            continue
        valid = work.dropna(subset=hit_cols + ["next_mfe", "next_mae"]).copy()
        if len(valid) < 700:
            continue
        cols = [c for c in dict.fromkeys(base + ["regime_code"] + list(EVENT_FEATURES)) if c in valid.columns]
        X = _frame(valid, cols)
        yall = valid[hit_cols].astype(int)
        usable = [c for c in hit_cols if int(yall[c].sum()) >= min_positive and int((yall[c] == 0).sum()) >= min_positive]
        if len(usable) < 2:
            continue
        Y = yall[usable]
        rare = Y.max(axis=1).to_numpy(bool)
        weights = np.where(rare, 3.0, 1.0)
        clf = ExtraTreesClassifier(n_estimators=150, max_depth=9, min_samples_leaf=20, max_features=0.75, class_weight="balanced", random_state=seed + EXPERT_SEEDS.get(name, 991), n_jobs=-1)
        clf.fit(X, Y, sample_weight=weights)
        stop_clf = ExtraTreesClassifier(n_estimators=100, max_depth=8, min_samples_leaf=25, max_features=0.75, class_weight="balanced", random_state=seed + 100 + EXPERT_SEEDS.get(name, 991), n_jobs=-1)
        stop_clf.fit(X, valid[f"stop_{hkeys[0]}"].astype(int))
        exc = ExtraTreesRegressor(n_estimators=100, max_depth=9, min_samples_leaf=25, max_features=0.75, random_state=seed + 200 + EXPERT_SEEDS.get(name, 991), n_jobs=-1)
        exc.fit(X, valid[["next_mfe", "next_mae"]].clip(-0.20, 0.20))
        bundles[name] = Expert63(name, cols, clf, stop_clf, exc, [c.replace("hit_", "") for c in usable], len(valid), {c: float(Y[c].mean()) for c in usable})
    return bundles


def _proba(model, X, hkeys):
    raw = model.predict_proba(X)
    if not isinstance(raw, list): raw = [raw]
    return {f"p_hit_{k}": (a[:, 1] if a.shape[1] > 1 else np.zeros(len(X))) for k, a in zip(hkeys, raw)}


def score_experts(bundles, latest, thresholds=THRESHOLDS):
    if not bundles or latest.empty: return pd.DataFrame()
    work = add_cpr_features(latest)
    work["regime_code"] = regime_labels(work)
    hkeys = keys(thresholds)
    parts = []
    for name, b in bundles.items():
        X = _frame(work, b.features)
        p = _proba(b.hit_model, X, b.hit_keys)
        sr = b.stop_model.predict_proba(X)
        pstop = sr[:, 1] if sr.shape[1] > 1 else np.zeros(len(X))
        exc = np.asarray(b.excursion_model.predict(X), float)
        if exc.ndim == 1: exc = exc.reshape(-1, 2)
        part = pd.DataFrame({"date": work.date.to_numpy(), "symbol": work.symbol.to_numpy(), "expert": name, "regime_code": work.regime_code.to_numpy(), "p_stop": pstop, "pred_mfe": exc[:, 0], "pred_mae": exc[:, 1]})
        for k in hkeys: part[f"p_hit_{k}"] = p.get(f"p_hit_{k}", np.zeros(len(part)))
        part["tail_score"] = 0.35*part[f"p_hit_{hkeys[0]}"] + 0.25*part[f"p_hit_{hkeys[1]}"] + 0.20*part[f"p_hit_{hkeys[2]}"] + 0.15*part[f"p_hit_{hkeys[3]}"] + 0.05*np.clip(part.pred_mfe,0,.15) - .25*part.p_stop
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _wide(scored, experts, hkeys):
    cols = [f"p_hit_{k}" for k in hkeys] + ["p_stop", "pred_mfe", "pred_mae", "tail_score"]
    parts=[]
    for e in experts:
        x=scored[scored.expert==e].set_index(["date","symbol"])
        if not x.empty: parts.append(x[cols].add_prefix(e+"__"))
    return pd.concat(parts,axis=1).sort_index() if parts else pd.DataFrame()


def train_router(train_df, thresholds=THRESHOLDS, stop_loss=.0125, seed=42):
    # OOF gate labels which expert had the best realized tail-aware score.
    dates=np.array(sorted(pd.to_datetime(train_df.date).unique()))
    if len(dates)<120: return None
    hkeys=keys(thresholds); xs=[]; ys=[]
    starts=np.linspace(max(50,len(dates)//3), len(dates)-15, 3, dtype=int)
    for ci in sorted(set(int(x) for x in starts)):
        cut=pd.Timestamp(dates[ci]); end=pd.Timestamp(dates[min(ci+max(8,len(dates)//10),len(dates)-1)])
        tr=train_df[train_df.date<cut]; va=train_df[(train_df.date>=cut)&(train_df.date<=end)]
        if tr.date.nunique()<50 or va.empty: continue
        local=train_experts(tr,thresholds,stop_loss,seed,min_positive=10)
        if len(local)<2: continue
        s=score_experts(local,va,thresholds)
        if s.empty: continue
        target=va.set_index(["date","symbol"])[f"hit_{hkeys[0]}"]
        w=_wide(s,list(local),hkeys).join(target.rename("real_hit"),how="inner")
        for _,r in w.iterrows():
            vals={}
            for e in local:
                b=e+"__"; p25=float(r.get(b+f"p_hit_{hkeys[0]}",0)); p30=float(r.get(b+f"p_hit_{hkeys[1]}",0)); ps=float(r.get(b+"p_stop",0)); tail=float(r.get(b+"tail_score",0))
                vals[e]=2*float(r.real_hit)*(.6*p25+.4*p30)+tail-.4*ps
            xs.append(r.drop(labels=["real_hit"])); ys.append(max(vals,key=vals.get))
    if not xs: return None
    X=pd.DataFrame(xs).replace([np.inf,-np.inf],np.nan).fillna(0); y=pd.Series(ys)
    if y.nunique()<2: return None
    scaler=StandardScaler(); model=LogisticRegression(max_iter=2000,C=.2,class_weight="balanced",random_state=seed)
    model.fit(scaler.fit_transform(X.astype(float)),y.to_numpy())
    return Router63(model,scaler,list(X.columns))


def route_and_rank(scored, router, cfg):
    if scored.empty: return scored
    th=tuple(float(x) for x in cfg.get("target_levels",THRESHOLDS)); hk=keys(th); experts=sorted(scored.expert.unique()); wide=_wide(scored,experts,hk)
    if wide.empty: return scored.iloc[0:0].copy()
    X=wide.reindex(columns=router.feature_names,fill_value=0).fillna(0)
    if router is not None:
        probs=router.model.predict_proba(router.scaler.transform(X.astype(float))); idx=np.argmax(probs,axis=1); chosen=np.asarray(router.model.classes_)[idx]; conf=np.max(probs,axis=1)
    else:
        arr=wide.reindex(columns=[e+f"__p_hit_{hk[0]}" for e in experts],fill_value=0).to_numpy(float); idx=np.argmax(arr,axis=1); chosen=np.asarray(experts)[idx]; conf=np.max(arr,axis=1)
    routing=pd.DataFrame({"date":wide.index.get_level_values(0),"symbol":wide.index.get_level_values(1),"router_expert":chosen,"router_confidence":conf})
    rows=scored.merge(routing,on=["date","symbol"]); rows=rows[rows.expert==rows.router_expert].copy()
    if rows.empty:return rows
    for col,k in zip(["p25","p30","p40","p50"],hk): rows[col]=rows[f"p_hit_{k}"]
    rows["target"]=np.select([rows.p50>=.08,rows.p40>=.12,rows.p30>=.18],[.05,.04,.03],default=.025)
    hurdle=float(cfg.get("cost_hurdle_pct",.0015))+float(cfg.get("risk_penalty_pct",.001))
    rows["risk_adjusted_edge"]=rows.target*rows.p25-hurdle-.25*rows.p_stop
    rows["selection_score"]=rows.risk_adjusted_edge+.20*rows.router_confidence+.10*rows.p30+.05*np.clip(rows.pred_mfe,0,.10)
    # Avoid an arbitrary hard-coded expert score floor. Economic hurdle + calibrated hit probability control participation.
    rows=rows[(rows.p25>=float(cfg.get("min_hit_probability",.08)))&(rows.risk_adjusted_edge>=float(cfg.get("min_risk_adjusted_edge",.001)))]
    return rows.sort_values(["selection_score","p25","pred_mfe"],ascending=False).drop_duplicates("symbol")
