from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


FEATURES = [
    "gap", "ret1", "ret2", "ret5", "ret10", "ret20", "range_pct", "body_pct",
    "close_pos", "atr_pct", "rsi14", "vol_z20", "relvol5", "relvol20", "trend20",
    "trend60", "volatility20", "dist_high20", "dist_high60", "dist_low20", "skew20",
    "nifty_ret1", "nifty_ret5", "nifty_gap", "india_vix_ret1", "sector_ret1", "sector_ret5",
    "breadth", "fo_oi_z20", "fo_oi_change_z20", "fo_volume_rel20", "pcr_dev20",
    "cpr_width_pct", "cpr_close_location", "r1_dist", "s1_dist", "prevday_range_pct",
    "dayofweek", "month_turn", "news_score", "news_count", "corporate_action_flag",
]


def build_direction_targets(frame: pd.DataFrame, target: float, stop: float) -> pd.DataFrame:
    x = frame.copy()
    # These are next-session intraday barriers measured from the next open.
    x["long_target_hit"] = x["next_open_to_high"] >= target
    x["long_stop_hit"] = x["next_open_to_low"] <= -stop
    x["short_target_hit"] = x["next_open_to_low"] <= -target
    x["short_stop_hit"] = x["next_open_to_high"] >= stop
    # Direction label is only used inside historical training windows.
    long_edge = x["next_open_to_high"]
    short_edge = -x["next_open_to_low"]
    x["direction_label"] = np.where(
        (long_edge >= target) & (long_edge > short_edge), 1,
        np.where((short_edge >= target) & (short_edge > long_edge), -1, 0)
    )
    return x


def barrier_return(row: pd.Series, side: int, target: float, stop: float) -> tuple[float, str]:
    hi = float(row["next_open_to_high"])
    lo = float(row["next_open_to_low"])
    close = float(row["next_open_to_close"])
    if side == 1:
        if lo <= -stop:
            return -stop, "stop"
        if hi >= target:
            return target, "target"
        return close, "close"
    if hi >= stop:
        return -stop, "stop"
    if lo <= -target:
        return target, "target"
    return -close, "close"


@dataclass
class DirectionOOS:
    predictions: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict


def walk_forward_direction(
    frame: pd.DataFrame,
    train_days: int = 504,
    test_days: int = 21,
    step_days: int = 21,
    target: float = 0.0275,
    stop: float = 0.0125,
    probability_gate: float = 0.60,
) -> DirectionOOS:
    x = build_direction_targets(frame, target, stop)
    x = x.sort_values("date").dropna(subset=FEATURES + ["next_open_to_high", "next_open_to_low", "next_open_to_close"])
    dates = np.array(sorted(x["date"].unique()))
    preds, trades = [], []
    for end in range(train_days, len(dates), step_days):
        tr_dates = dates[end-train_days:end]
        te_dates = dates[end:min(end+test_days, len(dates))]
        train = x[x.date.isin(tr_dates)]
        test = x[x.date.isin(te_dates)].copy()
        if len(test) == 0 or train["direction_label"].nunique() < 2:
            continue
        clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.035, max_leaf_nodes=15, l2_regularization=1.0, random_state=42)
        clf.fit(train[FEATURES], train["direction_label"])
        prob = clf.predict_proba(test[FEATURES])
        classes = list(clf.classes_)
        p_long = prob[:, classes.index(1)] if 1 in classes else np.zeros(len(test))
        p_short = prob[:, classes.index(-1)] if -1 in classes else np.zeros(len(test))
        side = np.where((p_long >= probability_gate) & (p_long > p_short), 1, np.where((p_short >= probability_gate) & (p_short > p_long), -1, 0))
        p = np.maximum(p_long, p_short)
        out = test[["date", "next_open"]].copy()
        out["p_long"] = p_long; out["p_short"] = p_short; out["side"] = side; out["trade"] = side != 0
        preds.append(out)
        for i, (_, r) in enumerate(test.iterrows()):
            if side[i] == 0:
                continue
            ret, reason = barrier_return(r, int(side[i]), target, stop)
            trades.append({"date": r["date"], "side": "LONG" if side[i] == 1 else "SHORT", "entry": r["next_open"], "return": ret, "exit_reason": reason, "probability": float(p[i])})
    predictions = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    td = pd.DataFrame(trades)
    if td.empty:
        return DirectionOOS(predictions, td, {"trade_count": 0})
    r = td["return"].astype(float)
    eq = (1+r).cumprod()
    dd = eq/eq.cummax()-1
    monthly = td.assign(month=pd.to_datetime(td.date).dt.to_period("M")).groupby("month").return.apply(lambda s: (1+s).prod()-1)
    metrics = {
        "trade_count": int(len(td)), "long_trades": int((td.side == "LONG").sum()), "short_trades": int((td.side == "SHORT").sum()),
        "hit_rate": float((td.exit_reason == "target").mean()), "mean_trade_return": float(r.mean()),
        "median_trade_return": float(r.median()), "gross_compounded_return": float(eq.iloc[-1]-1),
        "max_drawdown": float(dd.min()), "monthly_avg_return": float(monthly.mean()), "monthly_median_return": float(monthly.median()),
    }
    return DirectionOOS(predictions, td, metrics)
