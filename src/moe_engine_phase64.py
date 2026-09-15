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


def keys(levels: Iterable[float]) -> list[str]:
    return [str(float(x)).replace('.', 'p').lstrip('0') for x in levels]


def expert_sets() -> dict[str, list[str]]:
    sets = {k: list(v) for k, v in EXPERT_FEATURES.items() if k != 'event'}
    sets[CPR_EXPERT] = list(CPR_FEATURES)
    if EVENT_FEATURES:
        sets['event'] = list(EVENT_FEATURES)
    return sets


def _cpr_rule_score(df: pd.DataFrame) -> pd.Series:
    x = df
    breakout = pd.to_numeric(x.get('cpr_breakout_pressure', 0), errors='coerce').fillna(0.0)
    trend = pd.to_numeric(x.get('cpr_trend', 0), errors='coerce').fillna(0.0)
    gap = pd.to_numeric(x.get('cpr_gap', 0), errors='coerce').fillna(0.0)
    narrow = pd.to_numeric(x.get('cpr_narrow', 0), errors='coerce').fillna(0.0)
    pos = pd.to_numeric(x.get('cpr_pos', 0), errors='coerce').fillna(0.0)
    score = 0.35 * breakout + 0.25 * trend + 0.20 * gap + 0.10 * narrow + 0.10 * pos
    return score.replace([np.inf, -np.inf], np.nan).fillna(0.0)


@dataclass
class Expert64:
    name: str
    features: list[str]
    hit_model: ExtraTreesClassifier
    stop_model: ExtraTreesClassifier
    mfe_model: ExtraTreesRegressor
    hit_keys: list[str]
    training_rows: int


def _frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[[c for c in cols if c in df.columns]].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)


def train_experts(
    train_df: pd.DataFrame,
    thresholds: Iterable[float] = THRESHOLDS,
    stop_loss: float = 0.0125,
    seed: int = 42,
    min_positive: int = 20,
) -> dict[str, Expert64]:
    work = add_cpr_features(train_df)
    work['regime_code'] = regime_labels(work)
    hkeys = keys(thresholds)
    hit_cols = [f'hit_{k}' for k in hkeys]
    bundles: dict[str, Expert64] = {}

    for name, base in expert_sets().items():
        if name == 'event' and (not EVENT_FEATURES or float(work[EVENT_FEATURES].abs().sum().sum()) == 0):
            continue
        valid = work.dropna(subset=hit_cols + ['next_mfe', 'next_mae']).copy()
        if len(valid) < 700:
            continue
        cols = [c for c in dict.fromkeys(base + ['regime_code']) if c in valid.columns]
        X = _frame(valid, cols)
        y_hits = valid[hit_cols].astype(int)
        usable = [c for c in hit_cols if int(y_hits[c].sum()) >= min_positive and int((y_hits[c] == 0).sum()) >= min_positive]
        if len(usable) < 2:
            continue

        rare = y_hits[usable].max(axis=1).to_numpy(bool)
        weights = np.where(rare, 3.0, 1.0)
        clf = ExtraTreesClassifier(
            n_estimators=100, max_depth=9, min_samples_leaf=20, max_features=0.75,
            class_weight='balanced', random_state=seed + EXPERT_SEEDS.get(name, 991), n_jobs=-1,
        )
        clf.fit(X, y_hits[usable], sample_weight=weights)

        stop_clf = ExtraTreesClassifier(
            n_estimators=70, max_depth=8, min_samples_leaf=25, max_features=0.75,
            class_weight='balanced', random_state=seed + 100 + EXPERT_SEEDS.get(name, 991), n_jobs=-1,
        )
        stop_clf.fit(X, valid[f'stop_{hkeys[0]}'].astype(int))

        mfe = ExtraTreesRegressor(
            n_estimators=80, max_depth=9, min_samples_leaf=25, max_features=0.75,
            random_state=seed + 200 + EXPERT_SEEDS.get(name, 991), n_jobs=-1,
        )
        mfe.fit(X, valid['next_mfe'].clip(-0.20, 0.20))

        bundles[name] = Expert64(
            name=name,
            features=cols,
            hit_model=clf,
            stop_model=stop_clf,
            mfe_model=mfe,
            hit_keys=[c.replace('hit_', '') for c in usable],
            training_rows=len(valid),
        )
    return bundles


def _proba(model, X: pd.DataFrame, hkeys: list[str]) -> dict[str, np.ndarray]:
    raw = model.predict_proba(X)
    if not isinstance(raw, list):
        raw = [raw]
    return {
        f'p_hit_{k}': (arr[:, 1] if arr.shape[1] > 1 else np.zeros(len(X)))
        for k, arr in zip(hkeys, raw)
    }


def score_experts(bundles: dict[str, Expert64], latest: pd.DataFrame, thresholds: Iterable[float] = THRESHOLDS) -> pd.DataFrame:
    if not bundles or latest.empty:
        return pd.DataFrame()
    work = add_cpr_features(latest.copy())
    work['regime_code'] = regime_labels(work)
    hkeys = keys(thresholds)
    parts = []

    for name, bundle in bundles.items():
        X = _frame(work, bundle.features)
        hit = _proba(bundle.hit_model, X, bundle.hit_keys)
        stop_raw = bundle.stop_model.predict_proba(X)
        p_stop = stop_raw[:, 1] if stop_raw.shape[1] > 1 else np.zeros(len(X))
        pred_mfe = np.asarray(bundle.mfe_model.predict(X), float)
        p25 = hit.get(f'p_hit_{hkeys[0]}', np.zeros(len(X)))
        p30 = hit.get(f'p_hit_{hkeys[1]}', np.zeros(len(X)))
        p40 = hit.get(f'p_hit_{hkeys[2]}', np.zeros(len(X)))
        p50 = hit.get(f'p_hit_{hkeys[3]}', np.zeros(len(X)))
        tail = 0.35 * p25 + 0.25 * p30 + 0.20 * p40 + 0.15 * p50 + 0.10 * np.clip(pred_mfe, 0.0, 0.10) - 0.15 * p_stop
        part = pd.DataFrame({
            'date': work.date.to_numpy(),
            'symbol': work.symbol.to_numpy(),
            'expert': name,
            'regime_code': work.regime_code.to_numpy(),
            'p_hit_025': p25,
            'p_hit_03': p30,
            'p_hit_04': p40,
            'p_hit_05': p50,
            'p_stop': p_stop,
            'pred_mfe': pred_mfe,
            'tail_score': tail,
        })
        # Cross-sectional percentile is the common scale used by the router.
        part['expert_rank'] = part.groupby('date')['tail_score'].rank(pct=True, method='average')
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def train_regime_router(train_df: pd.DataFrame, experts: dict[str, Expert64], seed: int = 42) -> dict[int, dict[str, float]]:
    """Leakage-safe regime router based on prior-period realized performance of each expert's feature signal.

    This is deliberately a router over expert families, not a gate over individual stocks.
    Every trained expert remains active; weights only change how strongly its predictions
    contribute to the final ensemble.
    """
    work = add_cpr_features(train_df.copy())
    work['regime_code'] = regime_labels(work)
    router: dict[int, dict[str, float]] = {}

    for regime in sorted(pd.Series(work['regime_code']).dropna().astype(int).unique()):
        sub = work[work['regime_code'].astype(int) == int(regime)].copy()
        qualities = {}
        for name in experts:
            if name == CPR_EXPERT:
                raw = _cpr_rule_score(sub)
            else:
                raw = expert_rule_score(sub, name)
            q = pd.Series(raw, index=sub.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            # Historical relationship of the expert signal with next-session tail outcomes.
            top_cut = q.groupby(sub['date']).transform(lambda z: z.quantile(0.8) if len(z) >= 10 else z.max())
            top = sub.loc[q >= top_cut]
            if top.empty:
                quality = 0.0
            else:
                h25 = top['hit_025'].mean() if 'hit_025' in top.columns else 0.0
                mfe = top['next_mfe'].clip(-0.10, 0.15).mean()
                stop = top['stop_025'].mean() if 'stop_025' in top.columns else 0.0
                quality = float(1.0 * h25 + 2.0 * mfe - 0.75 * stop)
            qualities[name] = quality
        vals = np.array(list(qualities.values()), dtype=float)
        if not np.isfinite(vals).any() or np.nanmax(vals) - np.nanmin(vals) < 1e-9:
            weights = {name: 1.0 / max(len(qualities), 1) for name in qualities}
        else:
            vals = np.nan_to_num(vals, nan=0.0)
            temp = 0.20
            expv = np.exp((vals - vals.max()) / temp)
            expv = expv / expv.sum()
            # Small floor prevents the router from hard-disabling an expert.
            floor = min(0.03, 0.5 / max(len(qualities), 1))
            expv = floor + (1.0 - floor * len(expv)) * expv
            weights = {name: float(w) for name, w in zip(qualities, expv)}
        router[int(regime)] = weights
    return router


def select_top_predictions(scored: pd.DataFrame, router: dict[int, dict[str, float]], max_per_expert: int = 10, top_n: int = 5) -> tuple[pd.DataFrame, dict[int, dict[str, float]]]:
    """Parallel expert selection: each expert ranks first, then regime weights combine predictions.

    There is no predictive eligibility threshold. Liquidity/data availability are handled
    outside this function; all trained experts produce predictions for the available universe.
    """
    if scored.empty:
        return scored, {}
    frames = []
    router_snapshots: dict[int, dict[str, float]] = {}
    for date_val, day in scored.groupby('date', sort=True):
        regime = int(day['regime_code'].mode().iloc[0])
        weights = router.get(regime, {e: 1.0 / day.expert.nunique() for e in day.expert.unique()})
        router_snapshots[regime] = weights
        for expert, group in day.groupby('expert', sort=False):
            g = group.nlargest(max_per_expert, 'expert_rank').copy()
            g['router_weight'] = float(weights.get(expert, 0.0))
            g['weighted_score'] = g['router_weight'] * g['expert_rank']
            g['weighted_p30'] = g['router_weight'] * g['p_hit_03']
            g['weighted_p25'] = g['router_weight'] * g['p_hit_025']
            frames.append(g)
    if not frames:
        return scored.iloc[0:0].copy(), router_snapshots
    pool = pd.concat(frames, ignore_index=True)
    agg = pool.groupby(['date', 'symbol'], as_index=False).agg(
        ensemble_score=('weighted_score', 'sum'),
        ensemble_p25=('weighted_p25', 'sum'),
        ensemble_p30=('weighted_p30', 'sum'),
        best_expert=('expert', 'first'),
        regime_code=('regime_code', 'first'),
        pred_mfe=('pred_mfe', 'max'),
        p_stop=('p_stop', 'mean'),
        expert_votes=('expert', 'nunique'),
    )
    agg['target'] = np.select(
        [agg['ensemble_p30'] >= 0.25, agg['ensemble_p25'] >= 0.20],
        [0.03, 0.025],
        default=0.025,
    )
    # No score/edge cutoff. Rank the union of expert top predictions.
    selected = agg.sort_values(['date', 'ensemble_score', 'ensemble_p30', 'pred_mfe'], ascending=[True, False, False, False])
    selected = selected.groupby('date', group_keys=False).head(int(top_n)).copy()
    return selected.reset_index(drop=True), router_snapshots
