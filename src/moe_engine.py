from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

BASE_FEATURES = [
    "ret1", "ret2", "ret3", "ret5", "ret10", "ret20", "gap", "range_pct",
    "body_pct", "close_pos", "atr_pct", "rsi14", "vol_z20", "turnover_z20",
    "relvol5", "relvol20", "dist_high20", "dist_high60", "dist_low20", "trend20",
    "trend60", "volatility20", "skew20", "market_ret1", "market_ret5",
    "market_breadth", "cross_ret_rank", "cross_vol_rank", "fut_oi_z20",
    "fut_oi_change_z20", "fut_volume_rel20", "pcr_dev20",
]
EVENT_FEATURES = [
    "news_sentiment_24h", "news_count_24h", "news_importance_24h",
    "corporate_action_flag", "corporate_action_score",
]
COMMON = ["market_ret5", "market_breadth", "volatility20", "cross_ret_rank", "cross_vol_rank"]
EXPERT_FEATURES = {
    "momentum": COMMON + ["ret3", "ret5", "ret10", "ret20", "trend20", "trend60", "relvol20"],
    "breakout": COMMON + ["dist_high20", "dist_high60", "close_pos", "range_pct", "relvol5", "relvol20", "gap"],
    "fno": COMMON + ["fut_oi_z20", "fut_oi_change_z20", "fut_volume_rel20", "pcr_dev20", "ret1", "ret5", "relvol20"],
    "mean_reversion": COMMON + ["rsi14", "dist_low20", "ret1", "ret2", "ret3", "skew20", "volatility20"],
    "volatility": COMMON + ["range_pct", "atr_pct", "volatility20", "relvol5", "relvol20", "gap", "body_pct"],
    "relative_strength": COMMON + ["ret1", "ret3", "ret5", "ret10", "trend20", "trend60", "market_ret1"],
    "event": COMMON + ["gap", "relvol5", "relvol20"] + EVENT_FEATURES,
}
EXPERTS = tuple(EXPERT_FEATURES)
EXPERT_SEEDS = {name: 1000 + i * 137 for i, name in enumerate(EXPERTS)}
SIGNAL_CUTOFF_HOUR = 15
SIGNAL_CUTOFF_MINUTE = 30


@dataclass
class ExpertBundle:
    name: str
    features: list[str]
    model: RandomForestRegressor
    target_keys: list[str]
    score_threshold: float
    training_rows: int


def _signal_date_from_timestamp(ts: pd.Series) -> pd.Series:
    local = pd.to_datetime(ts, errors="coerce", utc=True).dt.tz_convert("Asia/Kolkata")
    return local.dt.normalize()


def _filter_to_eod_cutoff(ts: pd.Series) -> pd.Series:
    local = pd.to_datetime(ts, errors="coerce", utc=True).dt.tz_convert("Asia/Kolkata")
    cutoff = local.dt.normalize() + pd.Timedelta(hours=SIGNAL_CUTOFF_HOUR, minutes=SIGNAL_CUTOFF_MINUTE)
    return local <= cutoff


def add_point_in_time_events(df: pd.DataFrame, news: pd.DataFrame | None = None, corp: pd.DataFrame | None = None) -> pd.DataFrame:
    out = df.copy()
    for col in EVENT_FEATURES:
        out[col] = 0.0
    if out.empty:
        return out
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()

    if news is not None and not news.empty:
        n = news.copy()
        n["timestamp"] = pd.to_datetime(n["timestamp"], errors="coerce", utc=True)
        n["symbol"] = n["symbol"].astype(str)
        n = n.dropna(subset=["timestamp", "symbol"])
        n = n[_filter_to_eod_cutoff(n["timestamp"])]
        if "sentiment" not in n:
            n["sentiment"] = 0.0
        if "importance" not in n:
            n["importance"] = 1.0
        n["signal_date"] = _signal_date_from_timestamp(n["timestamp"])
        agg = n.groupby(["signal_date", "symbol"], as_index=False).agg(
            news_sentiment_24h=("sentiment", "mean"),
            news_count_24h=("symbol", "size"),
            news_importance_24h=("importance", "sum"),
        )
        out = out.merge(agg, left_on=["date", "symbol"], right_on=["signal_date", "symbol"], how="left")
        for col in ["news_sentiment_24h", "news_count_24h", "news_importance_24h"]:
            out[col] = out[col].fillna(0.0)
        out = out.drop(columns=["signal_date"], errors="ignore")

    if corp is not None and not corp.empty:
        c = corp.copy()
        ts_col = "announcement_timestamp" if "announcement_timestamp" in c.columns else "timestamp"
        c[ts_col] = pd.to_datetime(c[ts_col], errors="coerce", utc=True)
        c["symbol"] = c["symbol"].astype(str)
        c = c.dropna(subset=[ts_col, "symbol"])
        c = c[_filter_to_eod_cutoff(c[ts_col])]
        if "score" not in c:
            c["score"] = 1.0
        c["signal_date"] = _signal_date_from_timestamp(c[ts_col])
        agg = c.groupby(["signal_date", "symbol"], as_index=False).agg(
            corporate_action_flag=("symbol", "size"),
            corporate_action_score=("score", "sum"),
        )
        out = out.merge(agg, left_on=["date", "symbol"], right_on=["signal_date", "symbol"], how="left")
        for col in ["corporate_action_flag", "corporate_action_score"]:
            out[col] = out[col].fillna(0.0)
        out = out.drop(columns=["signal_date"], errors="ignore")

    return out


def add_forward_targets(df: pd.DataFrame, target_levels: tuple[float, ...], stop_loss: float) -> pd.DataFrame:
    out = df.sort_values(["symbol", "date"]).copy()
    g = out.groupby("symbol")
    out["next_open_phase6"] = g["open"].shift(-1)
    out["next_high_phase6"] = g["high"].shift(-1)
    out["next_low_phase6"] = g["low"].shift(-1)
    out["next_close_phase6"] = g["close"].shift(-1)
    for target in target_levels:
        name = str(target).replace(".", "p").lstrip("0")
        op = out["next_open_phase6"].astype(float)
        hi = out["next_high_phase6"].astype(float)
        lo = out["next_low_phase6"].astype(float)
        cl = out["next_close_phase6"].astype(float)
        t = op * (1 + target)
        s = op * (1 - stop_loss)
        both = (hi >= t) & (lo <= s)
        hit_t = hi >= t
        hit_s = lo <= s
        close_ret = cl / op - 1
        out[f"expert_ret_{name}"] = np.select([both, hit_t, hit_s], [-stop_loss, target, -stop_loss], default=close_ret)
    return out


def regime_labels(df: pd.DataFrame) -> pd.Series:
    breadth = df["market_breadth"].fillna(0.5)
    ret5 = df["market_ret5"].fillna(0.0)
    vol = df["volatility20"].fillna(df["volatility20"].median() if df["volatility20"].notna().any() else 0.02)
    score = 0.45 * np.clip((breadth - 0.35) / 0.30, 0, 1)
    score += 0.40 * np.clip((ret5 + 0.03) / 0.08, 0, 1)
    score += 0.15 * (1 - np.clip((vol - 0.01) / 0.05, 0, 1))
    return pd.Series(np.select([score >= 0.65, score >= 0.40], [2, 1], default=0), index=df.index, dtype=float)


def expert_rule_score(df: pd.DataFrame, name: str) -> pd.Series:
    z = lambda s: s.fillna(0.0).clip(-3, 3)
    if name == "momentum":
        score = 0.22*z(df.ret5) + 0.18*z(df.ret20) + 0.16*z(df.trend20) + 0.16*z(df.trend60) + 0.14*z(df.relvol20) + 0.14*z(df.cross_ret_rank - 0.5)
    elif name == "breakout":
        score = 0.24*z(df.dist_high20) + 0.20*z(df.dist_high60) + 0.16*z(df.close_pos - 0.5) + 0.16*z(df.relvol5 - 1) + 0.14*z(df.range_pct) + 0.10*z(df.gap)
    elif name == "fno":
        score = 0.25*z(df.fut_oi_change_z20) + 0.18*z(df.fut_oi_z20) + 0.18*z(df.fut_volume_rel20 - 1) + 0.16*z(-df.pcr_dev20) + 0.13*z(df.ret5) + 0.10*z(df.relvol20)
    elif name == "mean_reversion":
        score = 0.24*z(-df.rsi14 / 50 + 1) + 0.22*z(-df.dist_low20) + 0.18*z(-df.ret3) + 0.14*z(-df.ret1) + 0.12*z(-df.skew20) + 0.10*z(1 - df.volatility20)
    elif name == "volatility":
        score = 0.25*z(df.relvol5 - 1) + 0.22*z(df.range_pct) + 0.18*z(df.atr_pct) + 0.16*z(df.volatility20) + 0.11*z(df.gap.abs()) + 0.08*z(df.body_pct.abs())
    elif name == "relative_strength":
        score = 0.26*z(df.cross_ret_rank - 0.5) + 0.20*z(df.ret5) + 0.18*z(df.ret10) + 0.15*z(df.trend20) + 0.11*z(df.trend60) + 0.10*z(df.market_ret1)
    else:
        event_strength = z(df.news_sentiment_24h) + 0.5*z(df.news_importance_24h) + 0.5*z(df.corporate_action_score)
        score = 0.55*z(event_strength) + 0.18*z(df.gap) + 0.15*z(df.relvol5 - 1) + 0.12*z(df.relvol20 - 1)
    return (1.0 / (1.0 + np.exp(-score))).astype(float)


def train_experts(train_df: pd.DataFrame, target_levels=(0.025, 0.03), stop_loss=0.0125, seed=42) -> dict[str, ExpertBundle]:
    """Train one multi-output forest per expert."""
    bundles = {}
    work = train_df.copy()
    work["regime_code"] = regime_labels(work)
    target_keys = [str(t).replace(".", "p").lstrip("0") for t in target_levels]
    target_cols = [f"expert_ret_{key}" for key in target_keys]

    for name, cols in EXPERT_FEATURES.items():
        xcols = list(dict.fromkeys(cols + ["regime_code"]))
        if name == "event" and float(work[EVENT_FEATURES].abs().sum().sum()) == 0:
            continue
        valid = work.dropna(subset=xcols + target_cols).copy()
        if len(valid) < 500:
            continue
        signal = expert_rule_score(valid, name)
        threshold = float(np.quantile(signal, 0.70))
        X = valid[xcols].fillna(0.0).astype(float)
        y = valid[target_cols].astype(float).clip(-0.10, 0.10)
        model = RandomForestRegressor(
            n_estimators=120,
            max_depth=8,
            min_samples_leaf=30,
            max_features=0.75,
            random_state=seed + EXPERT_SEEDS[name],
            n_jobs=-1,
        )
        model.fit(X, y)
        bundles[name] = ExpertBundle(name, xcols, model, target_keys, threshold, len(valid))
    return bundles


def score_experts(bundles: dict[str, ExpertBundle], latest: pd.DataFrame, target_levels=(0.025, 0.03)) -> pd.DataFrame:
    work = latest.copy()
    work["regime_code"] = regime_labels(work)
    output = []
    target_keys = [str(t).replace(".", "p").lstrip("0") for t in target_levels]
    for name, bundle in bundles.items():
        part = work[["symbol"] + list(dict.fromkeys(bundle.features))].copy()
        part["regime_code"] = work["regime_code"].to_numpy()
        part["expert"] = name
        part["expert_rule_score"] = expert_rule_score(work, name).to_numpy()
        X = part[bundle.features].fillna(0.0).astype(float)
        pred = np.asarray(bundle.model.predict(X), dtype=float)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        for i, key in enumerate(bundle.target_keys):
            if key in target_keys:
                part[f"pred_{key}"] = pred[:, i]
        output.append(part[["symbol", "expert", "expert_rule_score", "regime_code"] + [f"pred_{key}" for key in target_keys]])
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()


def select_trades(scored: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Filter model outputs and optionally diversify across expert families.

    ``selection_mode=expert_diversified`` gives at most one stock to each expert
    for a portfolio. This lets the backtest test whether different market
    situations benefit from different specialists, instead of repeatedly
    selecting several stocks from the same expert.
    """
    if scored.empty:
        return scored
    target_levels = [float(x) for x in cfg["target_levels"]]
    target_keys = [str(t).replace(".", "p").lstrip("0") for t in target_levels]
    preds = scored[[f"pred_{key}" for key in target_keys]].to_numpy(dtype=float)
    hurdle = float(cfg["cost_hurdle_pct"]) + float(cfg["risk_penalty_pct"])
    best_idx = np.argmax(preds, axis=1)
    best_pred = preds[np.arange(len(scored)), best_idx]
    eligible = scored.loc[(scored["expert_rule_score"].to_numpy(dtype=float) >= 0.70) & (best_pred > hurdle)].copy()
    if eligible.empty:
        return eligible

    epreds = eligible[[f"pred_{key}" for key in target_keys]].to_numpy(dtype=float)
    best_idx = np.argmax(epreds, axis=1)
    eligible["target"] = np.asarray(target_levels)[best_idx]
    eligible["predicted_net_return"] = epreds[np.arange(len(eligible)), best_idx]
    eligible["selection_score"] = eligible["predicted_net_return"] + 0.10 * eligible["expert_rule_score"].astype(float)

    mode = str(cfg.get("selection_mode", "ranked"))
    if mode == "expert_diversified":
        # First find each expert's best distinct stock. Then rank those expert
        # representatives. This enforces one stock per expert, which is the
        # diversification experiment requested for Phase 6.
        reps = (
            eligible.sort_values(
                ["expert", "symbol", "selection_score", "predicted_net_return"],
                ascending=[True, True, False, False],
            )
            .drop_duplicates(["expert", "symbol"], keep="first")
            .sort_values(["expert", "selection_score", "predicted_net_return"], ascending=[True, False, False])
            .drop_duplicates("expert", keep="first")
            .sort_values(["selection_score", "predicted_net_return"], ascending=False)
            .reset_index(drop=True)
        )
        return reps

    # Baseline: one expert/target decision per stock.
    eligible = eligible.sort_values(["symbol", "selection_score", "predicted_net_return"], ascending=[True, False, False])
    return eligible.drop_duplicates("symbol", keep="first").sort_values(["selection_score", "predicted_net_return"], ascending=False).reset_index(drop=True)
