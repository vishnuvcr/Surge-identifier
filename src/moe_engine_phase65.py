from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from src.moe_engine import EXPERT_FEATURES, EXPERT_SEEDS, EVENT_FEATURES, expert_rule_score, regime_labels
from src.moe_engine_phase63 import CPR_FEATURES, add_cpr_features

THRESHOLDS = (0.025, 0.03, 0.04, 0.05)
CPR_EXPERT = "cpr_camarilla"
DIRECTIONS = ("LONG", "SHORT")


def keys(levels: Iterable[float]) -> list[str]:
    return [str(float(x)).replace(".", "p").lstrip("0") for x in levels]


def expert_sets() -> dict[str, list[str]]:
    sets = {k: list(v) for k, v in EXPERT_FEATURES.items() if k != "event"}
    sets[CPR_EXPERT] = list(CPR_FEATURES)
    if EVENT_FEATURES:
        sets["event"] = list(EVENT_FEATURES)
    return sets


def _frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[[c for c in cols if c in df.columns]].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)


def add_bidirectional_targets(df: pd.DataFrame, stop_loss: float = 0.0125, thresholds: tuple[float, ...] = THRESHOLDS) -> pd.DataFrame:
    """Leakage-safe next-session LONG/SHORT excursion and hit labels.

    LONG: profit comes from next high/open; risk comes from next low/open.
    SHORT: profit comes from next open/low; risk comes from next high/open.
    When target and stop are both touched in the same daily bar, the hit label is
    deliberately FALSE because execution is conservatively stop-first.
    """
    out = df.sort_values(["symbol", "date"]).copy()
    g = out.groupby("symbol", sort=False)
    op, hi, lo, cl = [g[c].shift(-1).astype(float) for c in ["open", "high", "low", "close"]]
    out["long_mfe"] = hi / op - 1.0
    out["long_mae"] = lo / op - 1.0
    out["short_mfe"] = op / lo - 1.0
    out["short_mae"] = op / hi - 1.0
    out["long_close_ret"] = cl / op - 1.0
    out["short_close_ret"] = op / cl - 1.0
    out["label_date"] = g["date"].shift(-1)
    long_stop = out["long_mae"] <= -float(stop_loss)
    short_stop = out["short_mae"] <= -float(stop_loss)
    for level, key in zip(thresholds, keys(thresholds)):
        out[f"long_hit_{key}"] = ((out["long_mfe"] >= float(level)) & ~long_stop).astype(int)
        out[f"short_hit_{key}"] = ((out["short_mfe"] >= float(level)) & ~short_stop).astype(int)
        out[f"long_stop_{key}"] = long_stop.astype(int)
        out[f"short_stop_{key}"] = short_stop.astype(int)
    return out.replace([np.inf, -np.inf], np.nan)


def _cpr_rule_score(df: pd.DataFrame) -> pd.Series:
    breakout = pd.to_numeric(df.get("cpr_breakout_pressure", 0), errors="coerce").fillna(0.0)
    trend = pd.to_numeric(df.get("cpr_trend", 0), errors="coerce").fillna(0.0)
    gap = pd.to_numeric(df.get("cpr_gap", 0), errors="coerce").fillna(0.0)
    narrow = pd.to_numeric(df.get("cpr_narrow", 0), errors="coerce").fillna(0.0)
    pos = pd.to_numeric(df.get("cpr_pos", 0), errors="coerce").fillna(0.0)
    return (0.35 * breakout + 0.25 * trend + 0.20 * gap + 0.10 * narrow + 0.10 * pos).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _rule_signal(df: pd.DataFrame, expert: str) -> pd.Series:
    if expert == CPR_EXPERT:
        return _cpr_rule_score(df)
    return expert_rule_score(df, expert).replace([np.inf, -np.inf], np.nan).fillna(0.0)


@dataclass
class Expert65:
    name: str
    features: list[str]
    long_hit_model: ExtraTreesClassifier
    short_hit_model: ExtraTreesClassifier
    long_stop_model: ExtraTreesClassifier
    short_stop_model: ExtraTreesClassifier
    excursion_model: ExtraTreesRegressor
    hit_keys: list[str]
    training_rows: int


def _fit_binary(model_seed: int, X: pd.DataFrame, y: pd.Series, positive_weight: float = 3.0) -> ExtraTreesClassifier:
    rare = y.to_numpy(dtype=int)
    sample_weight = np.where(rare == 1, positive_weight, 1.0)
    model = ExtraTreesClassifier(
        n_estimators=100,
        max_depth=9,
        min_samples_leaf=20,
        max_features=0.75,
        class_weight="balanced",
        random_state=model_seed,
        n_jobs=-1,
    )
    model.fit(X, y.astype(int), sample_weight=sample_weight)
    return model


def train_experts(
    train_df: pd.DataFrame,
    thresholds: Iterable[float] = THRESHOLDS,
    stop_loss: float = 0.0125,
    seed: int = 42,
    min_positive: int = 20,
) -> dict[str, Expert65]:
    work = add_cpr_features(train_df.copy())
    work = add_bidirectional_targets(work, stop_loss, tuple(thresholds))
    work["regime_code"] = regime_labels(work)
    hkeys = keys(thresholds)
    bundles: dict[str, Expert65] = {}
    for name, base in expert_sets().items():
        if name == "event" and (not EVENT_FEATURES or float(work[EVENT_FEATURES].abs().sum().sum()) == 0):
            continue
        valid = work.dropna(subset=[f"long_hit_{k}" for k in hkeys] + [f"short_hit_{k}" for k in hkeys] + ["long_mfe", "short_mfe"]).copy()
        if len(valid) < 700:
            continue
        cols = [c for c in dict.fromkeys(base + ["regime_code"]) if c in valid.columns]
        X = _frame(valid, cols)
        long_cols = [f"long_hit_{k}" for k in hkeys]
        short_cols = [f"short_hit_{k}" for k in hkeys]
        long_usable = [c for c in long_cols if int(valid[c].sum()) >= min_positive and int((valid[c] == 0).sum()) >= min_positive]
        short_usable = [c for c in short_cols if int(valid[c].sum()) >= min_positive and int((valid[c] == 0).sum()) >= min_positive]
        if len(long_usable) < 2 or len(short_usable) < 2:
            continue
        y_long = valid[long_usable].max(axis=1).astype(int)
        y_short = valid[short_usable].max(axis=1).astype(int)
        long_hit = _fit_binary(seed + EXPERT_SEEDS.get(name, 991), X, valid[long_usable[-1]].astype(int), 3.0)
        short_hit = _fit_binary(seed + 1000 + EXPERT_SEEDS.get(name, 991), X, valid[short_usable[-1]].astype(int), 3.0)
        long_stop = _fit_binary(seed + 2000 + EXPERT_SEEDS.get(name, 991), X, valid[f"long_stop_{hkeys[0]}"].astype(int), 1.5)
        short_stop = _fit_binary(seed + 3000 + EXPERT_SEEDS.get(name, 991), X, valid[f"short_stop_{hkeys[0]}"].astype(int), 1.5)
        exc = ExtraTreesRegressor(
            n_estimators=90,
            max_depth=9,
            min_samples_leaf=25,
            max_features=0.75,
            random_state=seed + 4000 + EXPERT_SEEDS.get(name, 991),
            n_jobs=-1,
        )
        exc.fit(X, valid[["long_mfe", "short_mfe"]].clip(0.0, 0.20))
        bundles[name] = Expert65(
            name=name,
            features=cols,
            long_hit_model=long_hit,
            short_hit_model=short_hit,
            long_stop_model=long_stop,
            short_stop_model=short_stop,
            excursion_model=exc,
            hit_keys=hkeys,
            training_rows=len(valid),
        )
    return bundles


def _positive_proba(model: ExtraTreesClassifier, X: pd.DataFrame) -> np.ndarray:
    p = model.predict_proba(X)
    return p[:, 1] if p.shape[1] > 1 else np.zeros(len(X))


def score_experts(bundles: dict[str, Expert65], latest: pd.DataFrame, thresholds: Iterable[float] = THRESHOLDS) -> pd.DataFrame:
    if not bundles or latest.empty:
        return pd.DataFrame()
    work = add_cpr_features(latest.copy())
    work["regime_code"] = regime_labels(work)
    hkeys = keys(thresholds)
    parts = []
    for name, b in bundles.items():
        X = _frame(work, b.features)
        pl = _positive_proba(b.long_hit_model, X)
        ps = _positive_proba(b.short_hit_model, X)
        sl = _positive_proba(b.long_stop_model, X)
        ss = _positive_proba(b.short_stop_model, X)
        exc = np.asarray(b.excursion_model.predict(X), float)
        if exc.ndim == 1:
            exc = exc.reshape(-1, 2)
        # Model currently trains the strictest target directly; calibrated lower-target
        # probabilities are approximated monotonically from that tail probability.
        p_long_05 = pl
        p_short_05 = ps
        p_long_025 = np.sqrt(np.clip(pl, 0, 1))
        p_short_025 = np.sqrt(np.clip(ps, 0, 1))
        p_long_04 = np.power(np.clip(pl, 0, 1), 0.80)
        p_short_04 = np.power(np.clip(ps, 0, 1), 0.80)
        p_long_03 = np.power(np.clip(pl, 0, 1), 0.65)
        p_short_03 = np.power(np.clip(ps, 0, 1), 0.65)
        long_score = 0.30 * p_long_025 + 0.25 * p_long_03 + 0.20 * p_long_04 + 0.15 * p_long_05 + 0.10 * np.clip(exc[:, 0], 0, 0.10) - 0.20 * sl
        short_score = 0.30 * p_short_025 + 0.25 * p_short_03 + 0.20 * p_short_04 + 0.15 * p_short_05 + 0.10 * np.clip(exc[:, 1], 0, 0.10) - 0.20 * ss
        part = pd.DataFrame({
            "date": work.date.to_numpy(),
            "symbol": work.symbol.to_numpy(),
            "expert": name,
            "regime_code": work.regime_code.to_numpy(),
            "p_long_025": p_long_025,
            "p_long_03": p_long_03,
            "p_long_04": p_long_04,
            "p_long_05": p_long_05,
            "p_short_025": p_short_025,
            "p_short_03": p_short_03,
            "p_short_04": p_short_04,
            "p_short_05": p_short_05,
            "p_long_stop": sl,
            "p_short_stop": ss,
            "long_mfe": exc[:, 0],
            "short_mfe": exc[:, 1],
            "long_score": long_score,
            "short_score": short_score,
        })
        part["long_rank"] = part.groupby("date")["long_score"].rank(pct=True, method="average")
        part["short_rank"] = part.groupby("date")["short_score"].rank(pct=True, method="average")
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def train_router(train_df: pd.DataFrame, experts: dict[str, Expert65], thresholds: Iterable[float] = THRESHOLDS) -> dict[int, dict[str, dict[str, float]]]:
    """Learn direction-aware expert weights using prior-period realized tail quality.

    Every expert remains active via a small weight floor; the router changes influence,
    never hard-disables an expert.
    """
    work = add_cpr_features(train_df.copy())
    work = add_bidirectional_targets(work, 0.0125, tuple(thresholds))
    work["regime_code"] = regime_labels(work)
    hkeys = keys(thresholds)
    router: dict[int, dict[str, dict[str, float]]] = {}
    long_key = f"long_hit_{hkeys[0]}"
    short_key = f"short_hit_{hkeys[0]}"
    for regime in sorted(pd.Series(work["regime_code"]).dropna().astype(int).unique()):
        sub = work[work.regime_code.astype(int) == int(regime)].copy()
        q_long: dict[str, float] = {}
        q_short: dict[str, float] = {}
        for name in experts:
            raw = _rule_signal(sub, name)
            long_rank = raw.groupby(sub.date).rank(pct=True)
            short_rank = (-raw).groupby(sub.date).rank(pct=True)
            long_top = sub.loc[long_rank >= 0.80]
            short_top = sub.loc[short_rank >= 0.80]
            ql = float(long_top[long_key].mean() + 2.0 * long_top.long_mfe.clip(0, 0.15).mean() - 0.75 * long_top[f"long_stop_{hkeys[0]}"].mean()) if not long_top.empty else 0.0
            qs = float(short_top[short_key].mean() + 2.0 * short_top.short_mfe.clip(0, 0.15).mean() - 0.75 * short_top[f"short_stop_{hkeys[0]}"].mean()) if not short_top.empty else 0.0
            q_long[name], q_short[name] = ql, qs
        def normalize(q: dict[str, float]) -> dict[str, float]:
            names = list(q)
            vals = np.nan_to_num(np.array([q[n] for n in names], float), nan=0.0)
            if len(vals) == 0:
                return {}
            if np.ptp(vals) < 1e-9:
                w = np.ones(len(vals), dtype=float) / len(vals)
            else:
                ex = np.exp((vals - vals.max()) / 0.20)
                w = ex / ex.sum()
            floor = min(0.03, 0.5 / max(len(w), 1))
            w = floor + (1 - floor * len(w)) * w
            return {n: float(x) for n, x in zip(names, w)}
        router[int(regime)] = {"LONG": normalize(q_long), "SHORT": normalize(q_short)}
    return router


def train_contrarian_factors(train_df: pd.DataFrame, experts: dict[str, Expert65]) -> dict[int, dict[str, float]]:
    """Estimate whether strong directional expert signals tend to reverse in each regime.

    The factor is intentionally bounded and only boosts the opposite direction when prior
    data demonstrates measurable reversal; otherwise the direct LONG/SHORT models dominate.
    """
    work = add_cpr_features(train_df.copy())
    work = add_bidirectional_targets(work, 0.0125, THRESHOLDS)
    work["regime_code"] = regime_labels(work)
    out: dict[int, dict[str, float]] = {}
    for regime in sorted(pd.Series(work.regime_code).dropna().astype(int).unique()):
        sub = work[work.regime_code.astype(int) == int(regime)].copy()
        long_cons = pd.Series(0.0, index=sub.index)
        short_cons = pd.Series(0.0, index=sub.index)
        for name in experts:
            raw = _rule_signal(sub, name)
            long_cons = long_cons + raw.groupby(sub.date).rank(pct=True)
            short_cons = short_cons + (-raw).groupby(sub.date).rank(pct=True)
        n = max(len(experts), 1)
        long_cons = long_cons / n
        short_cons = short_cons / n
        strong_long = sub.loc[long_cons >= 0.75]
        strong_short = sub.loc[short_cons >= 0.75]
        reverse_short = float(strong_long.short_hit_025.mean() * strong_long.short_mfe.clip(0, 0.10).mean() / 0.025) if not strong_long.empty else 0.0
        reverse_long = float(strong_short.long_hit_025.mean() * strong_short.long_mfe.clip(0, 0.10).mean() / 0.025) if not strong_short.empty else 0.0
        out[int(regime)] = {
            "short_from_long": float(np.clip(reverse_short, 0.0, 0.60)),
            "long_from_short": float(np.clip(reverse_long, 0.0, 0.60)),
        }
    return out


def select_top_predictions(
    scored: pd.DataFrame,
    router: dict[int, dict[str, dict[str, float]]],
    contrarian: dict[int, dict[str, float]],
    max_per_expert: int = 10,
    top_n: int = 5,
) -> tuple[pd.DataFrame, dict[int, dict[str, dict[str, float]]]]:
    if scored.empty:
        return scored.iloc[0:0].copy(), {}
    frames = []
    snapshots: dict[int, dict[str, dict[str, float]]] = {}
    for date_val, day in scored.groupby("date", sort=True):
        regime = int(day.regime_code.mode().iloc[0])
        weights = router.get(regime, {})
        snapshots[regime] = weights
        for expert, group in day.groupby("expert", sort=False):
            gl = group.nlargest(max_per_expert, "long_rank").copy()
            gs = group.nlargest(max_per_expert, "short_rank").copy()
            gl["direction"] = "LONG"
            gs["direction"] = "SHORT"
            gl["router_weight"] = float(weights.get("LONG", {}).get(expert, 0.0))
            gs["router_weight"] = float(weights.get("SHORT", {}).get(expert, 0.0))
            gl["direct_score"] = gl.router_weight * gl.long_rank
            gs["direct_score"] = gs.router_weight * gs.short_rank
            frames.extend([gl, gs])
    pool = pd.concat(frames, ignore_index=True)
    agg = pool.groupby(["date", "symbol", "direction"], as_index=False).agg(
        direct_score=("direct_score", "sum"),
        router_weight=("router_weight", "mean"),
        best_expert=("expert", "first"),
        regime_code=("regime_code", "first"),
        long_mfe=("long_mfe", "max"),
        short_mfe=("short_mfe", "max"),
        long_stop=("p_long_stop", "mean"),
        short_stop=("p_short_stop", "mean"),
        long_rank=("long_rank", "max"),
        short_rank=("short_rank", "max"),
        votes=("expert", "nunique"),
    )
    # Add contrarian alternatives only when prior data indicates some reversal edge.
    all_rows = []
    for _, row in agg.iterrows():
        r = int(row.regime_code)
        cf = contrarian.get(r, {"short_from_long": 0.0, "long_from_short": 0.0})
        direct = float(row.direct_score)
        if row.direction == "LONG":
            opp = float(row.short_rank) * float(cf.get("long_from_short", 0.0))
            contra = "LONG_FROM_SHORT"
        else:
            opp = float(row.long_rank) * float(cf.get("short_from_long", 0.0))
            contra = "SHORT_FROM_LONG"
        z = row.to_dict()
        z["contrarian_score"] = opp
        z["contrarian_type"] = contra
        z["final_score"] = max(direct, opp)
        z["selection_source"] = "direct" if direct >= opp else "contrarian"
        all_rows.append(z)
    candidates = pd.DataFrame(all_rows)
    # Keep only the better directional alternative per symbol.
    candidates = candidates.sort_values(["date", "symbol", "final_score"], ascending=[True, True, False]).drop_duplicates(["date", "symbol"], keep="first")
    # No predictive minimum threshold. A day may still end up with fewer than top_n usable positions.
    selected = candidates.sort_values(["date", "final_score", "votes", "router_weight"], ascending=[True, False, False, False]).groupby("date", group_keys=False).head(int(top_n)).copy()
    return selected.reset_index(drop=True), snapshots
