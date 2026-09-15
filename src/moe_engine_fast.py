from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from src.moe_engine import (
    EXPERT_FEATURES,
    EXPERT_SEEDS,
    EVENT_FEATURES,
    ExpertBundle,
    expert_rule_score,
    regime_labels,
)


def train_experts(train_df: pd.DataFrame, target_levels=(0.025, 0.03), stop_loss=0.0125, seed=42) -> dict[str, ExpertBundle]:
    """Fast, leakage-equivalent expert training.

    Keeps the same expert feature sets, targets, training rows, monthly refit
    cadence and deterministic seeds. Uses one multi-output ExtraTrees ensemble
    per expert so both target regressions share the same tree-building pass.
    """
    bundles: dict[str, ExpertBundle] = {}
    work = train_df.copy()
    work["regime_code"] = regime_labels(work)
    target_keys = [str(t).replace(".", "p").lstrip("0") for t in target_levels]
    target_cols = [f"expert_ret_{k}" for k in target_keys]

    for name, cols in EXPERT_FEATURES.items():
        xcols = list(dict.fromkeys(cols + ["regime_code"]))
        if name == "event" and float(work[EVENT_FEATURES].abs().sum().sum()) == 0:
            continue
        valid = work.dropna(subset=xcols + target_cols).copy()
        if len(valid) < 500:
            continue

        signal = expert_rule_score(valid, name)
        threshold = float(np.quantile(signal, 0.70))

        # Keep feature names during fit/predict. Besides removing sklearn's
        # feature-name warning, this makes it harder to accidentally score a
        # model with a reordered numpy matrix.
        X = valid[xcols].fillna(0.0).astype(float)
        y = valid[target_cols].astype(float).clip(-0.10, 0.10)

        model = ExtraTreesRegressor(
            n_estimators=80,
            max_depth=8,
            min_samples_leaf=30,
            max_features=0.75,
            random_state=seed + EXPERT_SEEDS[name],
            n_jobs=-1,
        )
        model.fit(X, y)
        bundles[name] = ExpertBundle(name, xcols, model, target_keys, threshold, len(valid))

    return bundles
