from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "gap", "ret1", "ret2", "ret3", "ret5", "ret10", "ret20",
    "range_pct", "body_pct", "close_pos", "atr_pct", "rsi14",
    "vol_z20", "relvol5", "relvol20", "trend20", "trend60", "volatility20",
    "dist_high20", "dist_high60", "dist_low20", "skew20",
    "nifty_ret1", "nifty_ret5", "nifty_gap", "india_vix_ret1",
    "sector_ret1", "sector_ret5", "breadth", "fo_oi_z20", "fo_oi_change_z20",
    "fo_volume_rel20", "pcr_dev20", "cpr_width_pct", "cpr_close_location",
    "r1_dist", "s1_dist", "prevday_range_pct", "dayofweek", "month_turn",
    "news_score", "news_count", "corporate_action_flag",
]


def _safe(a, b):
    return a / b.replace(0, np.nan)


def _rolling_z(s: pd.Series, n: int) -> pd.Series:
    m = s.rolling(n, min_periods=max(5, n // 2)).mean()
    sd = s.rolling(n, min_periods=max(5, n // 2)).std()
    return (s - m) / sd.replace(0, np.nan)


def _stock_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.sort_values("date").copy()
    c, o, h, l, pc = [x[k].astype(float) for k in ["close", "open", "high", "low", "prev_close"]]
    for n in [1, 2, 3, 5, 10, 20]:
        x[f"ret{n}"] = c.pct_change(n)
    x["gap"] = o / pc - 1
    x["range_pct"] = (h - l) / pc.replace(0, np.nan)
    x["body_pct"] = (c - o) / pc.replace(0, np.nan)
    x["close_pos"] = (c - l) / (h - l).replace(0, np.nan)
    tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    x["atr_pct"] = tr.rolling(14, min_periods=8).mean() / c.replace(0, np.nan)
    delta = c.diff()
    up = delta.clip(lower=0).rolling(14, min_periods=8).mean()
    down = (-delta.clip(upper=0)).rolling(14, min_periods=8).mean()
    rs = up / down.replace(0, np.nan)
    x["rsi14"] = 100 - 100 / (1 + rs)
    x["vol_z20"] = _rolling_z(x["volume"].astype(float), 20)
    x["relvol5"] = x["volume"] / x["volume"].rolling(5, min_periods=3).mean()
    x["relvol20"] = x["volume"] / x["volume"].rolling(20, min_periods=10).mean()
    x["trend20"] = c / c.rolling(20, min_periods=10).mean() - 1
    x["trend60"] = c / c.rolling(60, min_periods=20).mean() - 1
    x["volatility20"] = x["ret1"].rolling(20, min_periods=10).std()
    x["dist_high20"] = c / h.rolling(20, min_periods=10).max() - 1
    x["dist_high60"] = c / h.rolling(60, min_periods=20).max() - 1
    x["dist_low20"] = c / l.rolling(20, min_periods=10).min() - 1
    x["skew20"] = x["ret1"].rolling(20, min_periods=15).skew()
    return x


def _cpr_features(x: pd.DataFrame) -> pd.DataFrame:
    p = x["prev_close"]
    ph = x["prev_day_high"] if "prev_day_high" in x else x["high"].shift(1)
    pl = x["prev_day_low"] if "prev_day_low" in x else x["low"].shift(1)
    pp = (ph + pl + p) / 3
    bc = (ph + pl) / 2
    tc = 2 * pp - bc
    x["cpr_width_pct"] = (tc - bc).abs() / p.replace(0, np.nan)
    x["cpr_close_location"] = (x["close"] - pp) / p.replace(0, np.nan)
    x["r1_dist"] = (2 * pp - pl - x["open"]) / p.replace(0, np.nan)
    x["s1_dist"] = (2 * pp - ph - x["open"]) / p.replace(0, np.nan)
    x["prevday_range_pct"] = (ph - pl) / p.replace(0, np.nan)
    return x


def build_single_stock_frame(stock: pd.DataFrame, context: pd.DataFrame | None = None) -> pd.DataFrame:
    x = _stock_features(stock)
    x = _cpr_features(x)
    x["dayofweek"] = x["date"].dt.dayofweek
    x["month_turn"] = x["date"].dt.month
    if context is not None and not context.empty:
        ctx = context.copy()
        ctx["date"] = pd.to_datetime(ctx["date"])
        x = x.merge(ctx, on="date", how="left", suffixes=("", "_ctx"))
    defaults = {
        "nifty_ret1": 0.0, "nifty_ret5": 0.0, "nifty_gap": 0.0,
        "india_vix_ret1": 0.0, "sector_ret1": 0.0, "sector_ret5": 0.0,
        "breadth": 0.5, "fo_oi_z20": 0.0, "fo_oi_change_z20": 0.0,
        "fo_volume_rel20": 1.0, "pcr_dev20": 0.0, "news_score": 0.0,
        "news_count": 0.0, "corporate_action_flag": 0.0,
    }
    for k, v in defaults.items():
        if k not in x:
            x[k] = v
    # IMPORTANT: all context/news/corporate-action values must represent information
    # that was actually published and available before the next-session open.
    x["next_open"] = x["open"].shift(-1)
    x["next_high"] = x["high"].shift(-1)
    x["next_low"] = x["low"].shift(-1)
    x["next_close"] = x["close"].shift(-1)
    x["next_open_to_high"] = x["next_high"] / x["next_open"] - 1
    x["next_open_to_low"] = x["next_low"] / x["next_open"] - 1
    x["next_open_to_close"] = x["next_close"] / x["next_open"] - 1
    return x.replace([np.inf, -np.inf], np.nan)


def _trade_outcome(row: pd.Series, target: float, stop: float) -> tuple[int, float, str]:
    hi = float(row["next_open_to_high"])
    lo = float(row["next_open_to_low"])
    close_ret = float(row["next_open_to_close"])
    # Conservative ambiguity rule when both barriers occur intraday and daily OHLC cannot
    # determine order: stop is assumed to occur first.
    if lo <= -stop:
        return 0, -stop, "stop"
    if hi >= target:
        return 1, target, "target"
    return 0, close_ret, "close"


@dataclass
class OOSReport:
    predictions: pd.DataFrame
    metrics: dict
    trades: pd.DataFrame


def walk_forward_oos(
    frame: pd.DataFrame,
    train_days: int = 504,
    test_days: int = 21,
    step_days: int = 21,
    target: float = 0.0275,
    stop: float = 0.0125,
    probability_gate: float = 0.60,
    min_train_rows: int = 250,
) -> OOSReport:
    data = frame.sort_values("date").dropna(subset=FEATURES + ["next_open_to_high", "next_open_to_low", "next_open_to_close"]).copy()
    unique_dates = np.array(sorted(data["date"].unique()))
    all_preds: list[pd.DataFrame] = []
    all_trades: list[dict] = []
    for end in range(train_days, len(unique_dates), step_days):
        train_dates = unique_dates[max(0, end - train_days):end]
        test_dates = unique_dates[end:min(end + test_days, len(unique_dates))]
        if len(test_dates) == 0:
            break
        train = data[data["date"].isin(train_dates)]
        test = data[data["date"].isin(test_dates)]
        if len(train) < min_train_rows:
            continue
        Xtr, Xte = train[FEATURES].astype(float), test[FEATURES].astype(float)
        ytr = (train["next_open_to_high"] >= target).astype(int)
        model = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.035, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=42,
        )
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
        reg = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.04, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=42,
        )
        reg.fit(Xtr, train["next_open_to_high"])
        expected_high = reg.predict(Xte)
        pred = test[["date", "close", "next_open"]].copy()
        pred["prob_target"] = p
        pred["expected_open_to_high"] = expected_high
        pred["trade"] = pred["prob_target"] >= probability_gate
        all_preds.append(pred)
        for idx, r in test.iterrows():
            prob = float(p[test.index.get_loc(idx)]) if hasattr(test.index, "get_loc") else float(p[0])
            if prob < probability_gate:
                continue
            hit, ret, exit_reason = _trade_outcome(r, target, stop)
            all_trades.append({
                "date": r["date"], "entry": r["next_open"], "return": ret,
                "hit_target": hit, "exit_reason": exit_reason,
                "prob_target": prob, "expected_open_to_high": float(expected_high[test.index.get_loc(idx)]),
            })
    predictions = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    trades = pd.DataFrame(all_trades)
    if predictions.empty:
        raise ValueError("Walk-forward produced no OOS predictions; increase history or reduce train_days")
    actual = data.set_index("date").loc[predictions["date"], "next_open_to_high"].to_numpy()
    predp = predictions["prob_target"].to_numpy()
    metrics = {
        "oos_days": int(predictions["date"].nunique()),
        "oos_rows": int(len(predictions)),
        "auc": float(roc_auc_score((actual >= target).astype(int), predp)) if len(np.unique(actual >= target)) > 1 else float("nan"),
        "average_precision": float(average_precision_score((actual >= target).astype(int), predp)),
        "mae_open_to_high": float(mean_absolute_error(actual, predictions["expected_open_to_high"])),
    }
    if not trades.empty:
        r = trades["return"].astype(float)
        metrics.update({
            "trade_count": int(len(trades)),
            "hit_rate": float(trades["hit_target"].mean()),
            "mean_trade_return": float(r.mean()),
            "median_trade_return": float(r.median()),
            "gross_compounded_return": float((1 + r).prod() - 1),
            "max_drawdown": float(_max_drawdown(r)),
            "monthly_avg_return": float(_monthly_avg_return(trades)),
        })
    else:
        metrics.update({"trade_count": 0})
    return OOSReport(predictions, metrics, trades)


def _max_drawdown(r: pd.Series) -> float:
    eq = (1 + r).cumprod()
    peak = eq.cummax()
    return float((eq / peak - 1).min())


def _monthly_avg_return(trades: pd.DataFrame) -> float:
    if trades.empty:
        return float("nan")
    t = trades.copy()
    t["month"] = pd.to_datetime(t["date"]).dt.to_period("M")
    monthly = t.groupby("month")["return"].apply(lambda s: float((1 + s).prod() - 1))
    return float(monthly.mean()) if len(monthly) else float("nan")


def fit_latest(frame: pd.DataFrame, target: float = 0.0275) -> tuple[HistGradientBoostingClassifier, HistGradientBoostingRegressor]:
    work = frame.dropna(subset=FEATURES + ["next_open_to_high"]).copy()
    y = (work["next_open_to_high"] >= target).astype(int)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_leaf_nodes=15, l2_regularization=1.0, random_state=42)
    clf.fit(work[FEATURES], y)
    reg = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=1.0, random_state=42)
    reg.fit(work[FEATURES], work["next_open_to_high"])
    return clf, reg
