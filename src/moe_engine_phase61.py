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

ALL_FEATURES = [
    "ret1", "ret2", "ret3", "ret5", "ret10", "ret20", "gap", "range_pct",
    "body_pct", "close_pos", "atr_pct", "rsi14", "vol_z20", "turnover_z20",
    "relvol5", "relvol20", "dist_high20", "dist_high60", "dist_low20",
    "trend20", "trend60", "volatility20", "skew20", "market_ret1", "market_ret5",
    "market_breadth", "cross_ret_rank", "cross_vol_rank", "fut_oi_z20",
    "fut_oi_change_z20", "fut_volume_rel20", "pcr_dev20", *EVENT_FEATURES,
]


def _keys(levels: Iterable[float]) -> list[str]:
    return [str(float(x)).replace(".", "p").lstrip("0") for x in levels]


@dataclass
class Expert61:
    name: str
    features: list[str]
    hit_model: ExtraTreesClassifier
    stop_model: ExtraTreesClassifier
    excursion_model: ExtraTreesRegressor
    hit_keys: list[str]
    training_rows: int
    hit_rates: dict[str, float]


@dataclass
class Router61:
    model: LogisticRegression
    scaler: StandardScaler
    feature_names: list[str]
    classes: list[str]


def add_targets(df: pd.DataFrame, stop_loss: float = 0.0125, thresholds: tuple[float, ...] = THRESHOLDS) -> pd.DataFrame:
    out = df.sort_values(["symbol", "date"]).copy()
    g = out.groupby("symbol", sort=False)
    op = g["open"].shift(-1).astype(float)
    hi = g["high"].shift(-1).astype(float)
    lo = g["low"].shift(-1).astype(float)
    cl = g["close"].shift(-1).astype(float)
    out["next_mfe"] = hi / op - 1.0
    out["next_mae"] = lo / op - 1.0
    out["next_close_ret"] = cl / op - 1.0
    out["label_date"] = g["date"].shift(-1)
    stop = out["next_mae"] <= -float(stop_loss)
    for level, key in zip(thresholds, _keys(thresholds)):
        hit = out["next_mfe"] >= float(level)
        both = hit & stop
        out[f"hit_{key}"] = (hit & ~both).astype(int)
        out[f"stop_{key}"] = stop.astype(int)
    return out


def _feature_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[[c for c in cols if c in df.columns]].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)


def _expert_sets() -> dict[str, list[str]]:
    sets = {k: list(v) for k, v in EXPERT_FEATURES.items()}
    sets["universal"] = list(dict.fromkeys(ALL_FEATURES + ["regime_code"]))
    return sets


def train_experts_phase61(train_df: pd.DataFrame, thresholds: tuple[float, ...] = THRESHOLDS, stop_loss: float = 0.0125, seed: int = 42, min_positive: int = 20) -> dict[str, Expert61]:
    work = train_df.copy()
    work["regime_code"] = regime_labels(work)
    hkeys = _keys(thresholds)
    hit_cols = [f"hit_{k}" for k in hkeys]
    sets = _expert_sets()
    bundles: dict[str, Expert61] = {}
    for name, base_cols in sets.items():
        if name == "event" and not EVENT_FEATURES:
            continue
        if name == "event" and float(work[EVENT_FEATURES].abs().sum().sum()) == 0.0:
            continue
        required = hit_cols + ["next_mfe", "next_mae"]
        valid = work.dropna(subset=required).copy()
        if len(valid) < 700:
            continue
        cols = [c for c in dict.fromkeys(base_cols + ["regime_code"] + EVENT_FEATURES) if c in valid.columns]
        X = _feature_frame(valid, cols)
        Yall = valid[hit_cols].astype(int)
        usable = [c for c in hit_cols if int(Yall[c].sum()) >= min_positive and int((Yall[c] == 0).sum()) >= min_positive]
        if len(usable) < 2:
            continue
        Y = Yall[usable]
        rare = Y.max(axis=1).to_numpy(dtype=bool)
        weights = np.where(rare, 3.0, 1.0)
        clf = ExtraTreesClassifier(n_estimators=180, max_depth=9, min_samples_leaf=20, max_features=0.75, class_weight="balanced", random_state=seed + EXPERT_SEEDS.get(name, 991), n_jobs=-1)
        clf.fit(X, Y, sample_weight=weights)
        stop_col = f"stop_{hkeys[0]}"
        stop_clf = ExtraTreesClassifier(n_estimators=140, max_depth=8, min_samples_leaf=25, max_features=0.75, class_weight="balanced", random_state=seed + 100 + EXPERT_SEEDS.get(name, 991), n_jobs=-1)
        stop_clf.fit(X, valid[stop_col].astype(int))
        exc = ExtraTreesRegressor(n_estimators=140, max_depth=9, min_samples_leaf=25, max_features=0.75, random_state=seed + 200 + EXPERT_SEEDS.get(name, 991), n_jobs=-1)
        exc.fit(X, valid[["next_mfe", "next_mae"]].clip(-0.20, 0.20))
        bundles[name] = Expert61(name, cols, clf, stop_clf, exc, [c.replace("hit_", "") for c in usable], len(valid), {c: float(Y[c].mean()) for c in usable})
    return bundles


def _multi_proba(model: ExtraTreesClassifier, X: pd.DataFrame, keys: list[str]) -> dict[str, np.ndarray]:
    raw = model.predict_proba(X)
    if not isinstance(raw, list):
        raw = [raw]
    out: dict[str, np.ndarray] = {}
    for key, arr in zip(keys, raw):
        out[f"p_hit_{key}"] = arr[:, 1] if arr.shape[1] > 1 else np.zeros(len(X))
    return out


def score_experts_phase61(bundles: dict[str, Expert61], latest: pd.DataFrame, thresholds: tuple[float, ...] = THRESHOLDS) -> pd.DataFrame:
    if not bundles or latest.empty:
        return pd.DataFrame()
    work = latest.copy()
    work["regime_code"] = regime_labels(work)
    rows = []
    for name, b in bundles.items():
        X = _feature_frame(work, b.features)
        p = _multi_proba(b.hit_model, X, b.hit_keys)
        stop_raw = b.stop_model.predict_proba(X)
        p_stop = stop_raw[:, 1] if stop_raw.shape[1] > 1 else np.zeros(len(X))
        exc = np.asarray(b.excursion_model.predict(X), dtype=float)
        if exc.ndim == 1:
            exc = exc.reshape(-1, 2)
        part = pd.DataFrame({"date": work["date"].to_numpy(), "symbol": work["symbol"].to_numpy(), "expert": name, "regime_code": work["regime_code"].to_numpy(), "p_stop": p_stop, "pred_mfe": exc[:, 0], "pred_mae": exc[:, 1]})
        for key in _keys(thresholds):
            part[f"p_hit_{key}"] = p.get(f"p_hit_{key}", np.zeros(len(part)))
        p25 = part[f"p_hit_{_keys(thresholds)[0]}"]
        p30 = part[f"p_hit_{_keys(thresholds)[1]}"]
        p40 = part[f"p_hit_{_keys(thresholds)[2]}"]
        p50 = part[f"p_hit_{_keys(thresholds)[3]}"]
        part["tail_score"] = 0.35 * p25 + 0.25 * p30 + 0.20 * p40 + 0.15 * p50 + 0.05 * np.clip(part["pred_mfe"], 0, 0.15) - 0.25 * p_stop
        rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _wide(scored: pd.DataFrame, experts: list[str], keys: list[str]) -> pd.DataFrame:
    cols = [f"p_hit_{k}" for k in keys] + ["p_stop", "pred_mfe", "pred_mae", "tail_score"]
    parts = []
    for expert in experts:
        x = scored[scored.expert == expert].set_index(["date", "symbol"])
        if not x.empty:
            parts.append(x[cols].add_prefix(expert + "__"))
    return pd.concat(parts, axis=1).sort_index() if parts else pd.DataFrame()


def build_oof_router_data(train_df: pd.DataFrame, thresholds: tuple[float, ...], stop_loss: float, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    dates = np.array(sorted(pd.to_datetime(train_df.date).unique()))
    if len(dates) < 140:
        return pd.DataFrame(), pd.Series(dtype=str)
    keys = _keys(thresholds)
    fold_starts = np.linspace(max(60, len(dates) // 4), len(dates) - 25, 4, dtype=int)
    xs, ys = [], []
    for cut_idx in sorted(set(int(x) for x in fold_starts)):
        cut = pd.Timestamp(dates[cut_idx])
        val_end = pd.Timestamp(dates[min(cut_idx + max(8, len(dates) // 12), len(dates) - 1)])
        tr = train_df[train_df.date < cut].copy()
        va = train_df[(train_df.date >= cut) & (train_df.date <= val_end)].copy()
        if tr.date.nunique() < 60 or va.empty:
            continue
        local = train_experts_phase61(tr, thresholds, stop_loss, seed, min_positive=10)
        if len(local) < 2:
            continue
        os = score_experts_phase61(local, va, thresholds)
        if os.empty:
            continue
        target = va.set_index(["date", "symbol"])[f"hit_{keys[0]}"]
        wide = _wide(os, list(local), keys).join(target.rename("real_hit"), how="inner")
        if wide.empty:
            continue
        for idx, row in wide.iterrows():
            expert_scores = {}
            for expert in local:
                base = expert + "__"
                p25 = float(row.get(base + f"p_hit_{keys[0]}", 0.0))
                p30 = float(row.get(base + f"p_hit_{keys[1]}", 0.0))
                stop = float(row.get(base + "p_stop", 0.0))
                tail = float(row.get(base + "tail_score", 0.0))
                # Economic OOF target: reward models that identified the
                # realized tail, while retaining model-confidence information.
                expert_scores[expert] = 2.0 * float(row["real_hit"]) * (0.60 * p25 + 0.40 * p30) + tail - 0.40 * stop
            best = max(expert_scores, key=expert_scores.get)
            xs.append(row.drop(labels=["real_hit"]))
            ys.append(best)
    if not xs:
        return pd.DataFrame(), pd.Series(dtype=str)
    X = pd.DataFrame(xs).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, pd.Series(ys)


def train_router(train_df: pd.DataFrame, thresholds: tuple[float, ...], stop_loss: float, seed: int) -> Router61 | None:
    X, y = build_oof_router_data(train_df, thresholds, stop_loss, seed)
    if X.empty or y.nunique() < 2:
        return None
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.astype(float))
    model = LogisticRegression(max_iter=2500, C=0.20, class_weight="balanced", random_state=seed)
    model.fit(Xs, y.to_numpy())
    return Router61(model, scaler, list(X.columns), [str(x) for x in model.classes_])


def route_and_rank(scored: pd.DataFrame, router: Router61 | None, cfg: dict) -> pd.DataFrame:
    if scored.empty:
        return scored
    thresholds = tuple(float(x) for x in cfg.get("target_levels", THRESHOLDS))
    keys = _keys(thresholds)
    experts = sorted(scored.expert.unique())
    wide = _wide(scored, experts, keys)
    if wide.empty:
        return scored.iloc[0:0].copy()
    if router is not None:
        X = wide.reindex(columns=router.feature_names, fill_value=0.0).fillna(0.0)
        probs = router.model.predict_proba(router.scaler.transform(X.astype(float)))
        idx = np.argmax(probs, axis=1)
        chosen = np.asarray(router.model.classes_)[idx]
        conf = np.max(probs, axis=1)
        routing = pd.DataFrame({"date": wide.index.get_level_values(0), "symbol": wide.index.get_level_values(1), "router_expert": chosen, "router_confidence": conf})
    else:
        p25_cols = [e + f"__p_hit_{keys[0]}" for e in experts]
        arr = wide.reindex(columns=p25_cols, fill_value=0.0).to_numpy(dtype=float)
        idx = np.argmax(arr, axis=1)
        routing = pd.DataFrame({"date": wide.index.get_level_values(0), "symbol": wide.index.get_level_values(1), "router_expert": np.asarray(experts)[idx], "router_confidence": np.max(arr, axis=1)})
    chosen_rows = scored.merge(routing, on=["date", "symbol"], how="inner")
    chosen_rows = chosen_rows[chosen_rows.expert == chosen_rows.router_expert].copy()
    if chosen_rows.empty:
        return chosen_rows
    chosen_rows["p25"] = chosen_rows[f"p_hit_{keys[0]}"]
    chosen_rows["p30"] = chosen_rows[f"p_hit_{keys[1]}"]
    chosen_rows["p40"] = chosen_rows[f"p_hit_{keys[2]}"]
    chosen_rows["p50"] = chosen_rows[f"p_hit_{keys[3]}"]
    # Adaptive target: higher targets require corresponding tail probability.
    chosen_rows["target"] = np.select([chosen_rows.p50 >= 0.08, chosen_rows.p40 >= 0.12, chosen_rows.p30 >= 0.18], [0.05, 0.04, 0.03], default=0.025)
    hurdle = float(cfg.get("cost_hurdle_pct", 0.0015)) + float(cfg.get("risk_penalty_pct", 0.0010))
    chosen_rows["risk_adjusted_edge"] = chosen_rows.target * chosen_rows.p25 - hurdle - 0.25 * chosen_rows.p_stop
    chosen_rows["selection_score"] = chosen_rows.risk_adjusted_edge + 0.20 * chosen_rows.router_confidence + 0.10 * chosen_rows.p30 + 0.05 * np.clip(chosen_rows.pred_mfe, 0, 0.10)
    min_p25 = float(cfg.get("min_hit_probability", 0.08))
    min_edge = float(cfg.get("min_risk_adjusted_edge", 0.0010))
    chosen_rows = chosen_rows[(chosen_rows.p25 >= min_p25) & (chosen_rows.risk_adjusted_edge >= min_edge)]
    return chosen_rows.sort_values(["selection_score", "p25", "pred_mfe"], ascending=False).drop_duplicates("symbol", keep="first")
