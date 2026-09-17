from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CPRConfig:
    narrow_pct_of_prev_range: float = 0.15
    medium_pct_of_prev_range: float = 0.30
    single_line_pct_of_prev_range: float = 0.03
    candle_break_buffer_bps: float = 2.0
    virgin_max_age_sessions: int = 3


def _levels(ohlc: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    x = ohlc.copy()
    p = prefix
    x[f"{p}pivot"] = (x["high"] + x["low"] + x["close"]) / 3.0
    x[f"{p}bc"] = (x["high"] + x["low"]) / 2.0
    x[f"{p}tc"] = 2.0 * x[f"{p}pivot"] - x[f"{p}bc"]
    x[f"{p}cpr_low"] = x[[f"{p}bc", f"{p}tc"]].min(axis=1)
    x[f"{p}cpr_high"] = x[[f"{p}bc", f"{p}tc"]].max(axis=1)
    x[f"{p}r1"] = 2 * x[f"{p}pivot"] - x["low"]
    x[f"{p}s1"] = 2 * x[f"{p}pivot"] - x["high"]
    x[f"{p}r2"] = x[f"{p}pivot"] + (x["high"] - x["low"])
    x[f"{p}s2"] = x[f"{p}pivot"] - (x["high"] - x["low"])
    x[f"{p}r3"] = x["high"] + 2 * (x[f"{p}pivot"] - x["low"])
    x[f"{p}s3"] = x["low"] - 2 * (x["high"] - x[f"{p}pivot"])
    return x


def _camarilla(ohlc: pd.DataFrame) -> pd.DataFrame:
    x = ohlc.copy()
    r = x["high"] - x["low"]
    x["cam_r3"] = x["close"] + r * 1.1 / 4.0
    x["cam_r4"] = x["close"] + r * 1.1 / 2.0
    x["cam_s3"] = x["close"] - r * 1.1 / 4.0
    x["cam_s4"] = x["close"] - r * 1.1 / 2.0
    return x


def _add_width(x: pd.DataFrame, prefix: str) -> pd.DataFrame:
    x = x.copy()
    x[f"{prefix}_width"] = x[f"{prefix}_cpr_high"] - x[f"{prefix}_cpr_low"]
    x[f"{prefix}_prev_range"] = x["high"] - x["low"]
    x[f"{prefix}_width_ratio"] = x[f"{prefix}_width"] / x[f"{prefix}_prev_range"].replace(0, np.nan)
    return x


def _prior_period_ohlc(daily: pd.DataFrame, period_col: str) -> pd.DataFrame:
    grouped = (
        daily.groupby(["symbol", period_col], as_index=False)
        .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
        .sort_values(["symbol", period_col])
    )
    for c in ["open", "high", "low", "close"]:
        grouped[c] = grouped.groupby("symbol")[c].shift(1)
    return grouped


def period_levels(intraday: pd.DataFrame, cfg: CPRConfig = CPRConfig()) -> pd.DataFrame:
    """Attach point-in-time daily/weekly/monthly CPR levels to intraday bars.

    Required columns: timestamp, symbol, open, high, low, close, volume.
    All levels are derived only from completed prior sessions/periods.
    """
    x = intraday.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"])
    x["session"] = x["timestamp"].dt.normalize()
    x = x.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    daily = (
        x.groupby(["symbol", "session"], as_index=False)
        .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
        .sort_values(["symbol", "session"])
    )
    daily["pd_high"] = daily.groupby("symbol")["high"].shift(1)
    daily["pd_low"] = daily.groupby("symbol")["low"].shift(1)

    dsrc = daily[["symbol", "session", "open", "high", "low", "close"]].copy()
    for c in ["open", "high", "low", "close"]:
        dsrc[c] = dsrc.groupby("symbol")[c].shift(1)
    dlev = _levels(dsrc).rename(columns={c: f"d_{c}" for c in ["pivot", "bc", "tc", "cpr_low", "cpr_high", "r1", "s1", "r2", "s2", "r3", "s3"]})
    cam = _camarilla(dsrc)
    dlev["d_cam_r3"] = cam["cam_r3"]
    dlev["d_cam_s3"] = cam["cam_s3"]
    dlev["d_pd_high"] = daily["pd_high"]
    dlev["d_pd_low"] = daily["pd_low"]
    dlev = _add_width(dlev, "d")
    dlev["d_ascending"] = dlev.groupby("symbol")["d_cpr_low"].diff() > 0
    dlev["d_descending"] = dlev.groupby("symbol")["d_cpr_low"].diff() < 0

    daily_touch = daily[["symbol", "session", "high", "low"]].merge(
        dlev[["symbol", "session", "d_cpr_low", "d_cpr_high"]], on=["symbol", "session"], how="left"
    )
    daily_touch["self_touched"] = (
        (daily_touch["high"] >= daily_touch["d_cpr_low"])
        & (daily_touch["low"] <= daily_touch["d_cpr_high"])
    )
    dlev["virgin"] = ~daily_touch["self_touched"].fillna(True)
    for lag in range(1, cfg.virgin_max_age_sessions + 1):
        virgin_flag = dlev.groupby("symbol")["virgin"].shift(lag).fillna(False)
        dlev[f"virgin_{lag}_low"] = dlev.groupby("symbol")["d_cpr_low"].shift(lag).where(virgin_flag)
        dlev[f"virgin_{lag}_high"] = dlev.groupby("symbol")["d_cpr_high"].shift(lag).where(virgin_flag)

    daily["week_period"] = daily["session"].dt.to_period("W-FRI")
    weekly = _prior_period_ohlc(daily, "week_period")
    wlev = _add_width(_levels(weekly, "w_"), "w")

    daily["month_period"] = daily["session"].dt.to_period("M")
    monthly = _prior_period_ohlc(daily, "month_period")
    mlev = _add_width(_levels(monthly, "m_"), "m")

    x["week_period"] = x["session"].dt.to_period("W-FRI")
    x["month_period"] = x["session"].dt.to_period("M")

    out = x.merge(dlev, on=["symbol", "session"], how="left")
    wcols = ["symbol", "week_period", "w_cpr_low", "w_cpr_high", "w_r1", "w_s1", "w_r2", "w_s2", "w_r3", "w_s3", "w_width_ratio"]
    mcols = ["symbol", "month_period", "m_cpr_low", "m_cpr_high", "m_r1", "m_s1", "m_r2", "m_s2", "m_width_ratio"]
    out = out.merge(wlev[wcols], on=["symbol", "week_period"], how="left")
    out = out.merge(mlev[mcols], on=["symbol", "month_period"], how="left")
    return out


def width_class(row: pd.Series, prefix: str, cfg: CPRConfig) -> str:
    ratio = row.get(f"{prefix}_width_ratio", np.nan)
    if not np.isfinite(ratio):
        return "unknown"
    if ratio <= cfg.single_line_pct_of_prev_range:
        return "single_line"
    if ratio <= cfg.narrow_pct_of_prev_range:
        return "narrow"
    if ratio <= cfg.medium_pct_of_prev_range:
        return "medium"
    return "wide"


def _cross_up(close: pd.Series, level: pd.Series) -> pd.Series:
    return (close > level) & (close.shift(1) <= level.shift(1))


def _cross_down(close: pd.Series, level: pd.Series) -> pd.Series:
    return (close < level) & (close.shift(1) >= level.shift(1))


def generate_signals(df: pd.DataFrame, strategy_id: str, cfg: CPRConfig = CPRConfig()) -> pd.DataFrame:
    x = df.copy().sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    out = pd.DataFrame(index=x.index)
    out["signal"] = 0
    out["strategy_id"] = strategy_id

    dwidth = x.apply(lambda r: width_class(r, "d", cfg), axis=1)
    near = dwidth.eq("narrow")
    wide = dwidth.eq("wide")
    weekly_above = x["close"] > x["w_cpr_high"]
    weekly_below = x["close"] < x["w_cpr_low"]

    if strategy_id == "camarilla_inside_reversal":
        r3_inside = x["d_cam_r3"].between(x["d_cpr_low"], x["d_cpr_high"])
        s3_inside = x["d_cam_s3"].between(x["d_cpr_low"], x["d_cpr_high"])
        out.loc[r3_inside & _cross_down(x["close"], x["d_cpr_low"]), "signal"] = -1
        out.loc[s3_inside & _cross_up(x["close"], x["d_cpr_high"]), "signal"] = 1

    elif strategy_id == "mean_reversion_r1_pdh_s1_pdl":
        upper = x[["d_r1", "d_pd_high"]].max(axis=1)
        lower = x[["d_s1", "d_pd_low"]].min(axis=1)
        out.loc[(x["high"] > upper) & _cross_down(x["close"], upper), "signal"] = -1
        out.loc[(x["low"] < lower) & _cross_up(x["close"], lower), "signal"] = 1

    elif strategy_id == "narrow_cpr_breakout":
        upper = x[["d_r1", "d_pd_high"]].max(axis=1)
        lower = x[["d_s1", "d_pd_low"]].min(axis=1)
        out.loc[near & _cross_up(x["close"], upper), "signal"] = 1
        out.loc[near & _cross_down(x["close"], lower), "signal"] = -1

    elif strategy_id == "virgin_cpr_reversal":
        for lag in range(1, cfg.virgin_max_age_sessions + 1):
            lo = x[f"virgin_{lag}_low"]
            hi = x[f"virgin_{lag}_high"]
            out.loc[(x["high"] >= lo) & (x["low"] <= hi) & _cross_down(x["close"], hi), "signal"] = -1
            out.loc[(x["high"] >= lo) & (x["low"] <= hi) & _cross_up(x["close"], lo), "signal"] = 1

    elif strategy_id == "mtf_cpr_daytrade":
        upper = x[["d_r1", "d_pd_high"]].max(axis=1)
        lower = x[["d_s1", "d_pd_low"]].min(axis=1)
        out.loc[weekly_above & _cross_up(x["close"], upper), "signal"] = 1
        out.loc[weekly_below & _cross_down(x["close"], lower), "signal"] = -1

    elif strategy_id == "inside_ascending_descending":
        ascending = x["d_ascending"].fillna(False)
        descending = x["d_descending"].fillna(False)
        prev_low = x.groupby("symbol")["d_cpr_low"].shift(1)
        prev_high = x.groupby("symbol")["d_cpr_high"].shift(1)
        inside = (x["d_cpr_low"] >= prev_low) & (x["d_cpr_high"] <= prev_high)
        upper = x[["d_r1", "d_pd_high"]].max(axis=1)
        lower = x[["d_s1", "d_pd_low"]].min(axis=1)
        out.loc[(near | inside) & ascending & _cross_up(x["close"], upper), "signal"] = 1
        out.loc[(near | inside) & descending & _cross_down(x["close"], lower), "signal"] = -1

    elif strategy_id == "masterclass_regime_open":
        upper = x["d_r1"]
        lower = x["d_s1"]
        far_up = x["open"] > upper
        far_dn = x["open"] < lower
        out.loc[weekly_above & near & ~far_up & _cross_up(x["close"], upper), "signal"] = 1
        out.loc[weekly_below & near & ~far_dn & _cross_down(x["close"], lower), "signal"] = -1
        out.loc[far_up & _cross_down(x["close"], x["d_cpr_high"]), "signal"] = -1
        out.loc[far_dn & _cross_up(x["close"], x["d_cpr_low"]), "signal"] = 1

    elif strategy_id == "weekly_cpr_masterclass":
        no_trade = wide | x["w_width_ratio"].isna()
        upper = x[["d_r1", "d_pd_high"]].max(axis=1)
        lower = x[["d_s1", "d_pd_low"]].min(axis=1)
        out.loc[(~no_trade) & weekly_above & near & _cross_up(x["close"], upper), "signal"] = 1
        out.loc[(~no_trade) & weekly_below & near & _cross_down(x["close"], lower), "signal"] = -1

    else:
        raise ValueError(f"Unknown CPR strategy_id: {strategy_id}")

    return pd.concat([x, out], axis=1)


STRATEGIES = (
    "camarilla_inside_reversal",
    "mean_reversion_r1_pdh_s1_pdl",
    "narrow_cpr_breakout",
    "virgin_cpr_reversal",
    "mtf_cpr_daytrade",
    "inside_ascending_descending",
    "masterclass_regime_open",
    "weekly_cpr_masterclass",
)
