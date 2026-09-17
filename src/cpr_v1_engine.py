from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
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
    x[f"{p}pivot"] = (x.high + x.low + x.close) / 3.0
    x[f"{p}bc"] = (x.high + x.low) / 2.0
    x[f"{p}tc"] = 2.0 * x[f"{p}pivot"] - x[f"{p}bc"]
    x[f"{p}cpr_low"] = x[[f"{p}bc", f"{p}tc"]].min(axis=1)
    x[f"{p}cpr_high"] = x[[f"{p}bc", f"{p}tc"]].max(axis=1)
    x[f"{p}r1"] = 2 * x[f"{p}pivot"] - x.low
    x[f"{p}s1"] = 2 * x[f"{p}pivot"] - x.high
    x[f"{p}r2"] = x[f"{p}pivot"] + (x.high - x.low)
    x[f"{p}s2"] = x[f"{p}pivot"] - (x.high - x.low)
    x[f"{p}r3"] = x.high + 2 * (x[f"{p}pivot"] - x.low)
    x[f"{p}s3"] = x.low - 2 * (x.high - x[f"p"] if False else x[f"{p}pivot"])
    return x


def _camarilla(ohlc: pd.DataFrame) -> pd.DataFrame:
    x = ohlc.copy()
    r = x.high - x.low
    x["cam_r3"] = x.close + r * 1.1 / 4.0
    x["cam_r4"] = x.close + r * 1.1 / 2.0
    x["cam_s3"] = x.close - r * 1.1 / 4.0
    x["cam_s4"] = x.close - r * 1.1 / 2.0
    return x


def period_levels(intraday: pd.DataFrame) -> pd.DataFrame:
    """Attach point-in-time daily/weekly/monthly CPR levels to 5-min bars.

    Required intraday columns: timestamp, symbol, open, high, low, close, volume.
    All levels for session t are calculated only from the completed prior period.
    """
    x = intraday.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"])
    x["session"] = x["timestamp"].dt.normalize()
    x = x.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    daily = x.groupby(["symbol", "session"], as_index=False).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
    daily = daily.sort_values(["symbol", "session"])
    daily["pd_high"] = daily.groupby("symbol").high.shift(1)
    daily["pd_low"] = daily.groupby("symbol").low.shift(1)
    daily_levels = _levels(daily).rename(columns={c: f"d_{c}" for c in ["pivot", "bc", "tc", "cpr_low", "cpr_high", "r1", "s1", "r2", "s2", "r3", "s3"]})
    daily_levels = _camarilla(daily_levels)
    daily_levels["d_cam_r3"] = daily_levels["cam_r3"]
    daily_levels["d_cam_s3"] = daily_levels["cam_s3"]
    daily_levels["d_width"] = daily_levels.d_cpr_high - daily_levels.d_cpr_low
    daily_levels["d_prev_range"] = daily_levels.high.shift(1) - daily_levels.low.shift(1)
    daily_levels["d_width_ratio"] = daily_levels.d_width / daily_levels.d_prev_range.replace(0, np.nan)
    daily_levels["d_ascending"] = daily_levels.groupby("symbol").d_cpr_low.diff() > 0
    daily_levels["d_descending"] = daily_levels.groupby("symbol").d_cpr_low.diff() < 0

    def attach_period(rule: str, prefix: str) -> pd.DataFrame:
        temp = daily.set_index("session").groupby("symbol")[['open','high','low','close']].resample(rule).agg({'open':'first','high':'max','low':'min','close':'last'}).reset_index()
        temp = temp.sort_values(["symbol", "session"])
        temp = temp.groupby("symbol").shift(1)
        temp["symbol"] = daily.groupby("symbol").symbol.first().values.repeat(temp.groupby("symbol").size().values) if False else temp.get("symbol", np.nan)
        return temp

    # Build weekly/monthly frames explicitly to avoid accidental future leakage.
    daily_idx = daily.set_index("session")
    weekly = daily_idx.groupby("symbol")[['open','high','low','close']].resample('W-FRI').agg({'open':'first','high':'max','low':'min','close':'last'}).reset_index()
    weekly = weekly.sort_values(["symbol", "session"])
    weekly = weekly.groupby("symbol", group_keys=False).apply(lambda g: g.assign(prev_high=g.high.shift(1), prev_low=g.low.shift(1))).reset_index(drop=True)
    wlev = _levels(weekly, "w_")
    wlev["w_width"] = wlev.w_cpr_high - wlev.w_cpr_low
    wlev["w_prev_range"] = wlev.high - wlev.low
    wlev["w_width_ratio"] = wlev.w_width / wlev.w_prev_range.replace(0, np.nan)
    wlev["w_pd_high"] = wlev.prev_high
    wlev["w_pd_low"] = wlev.prev_low

    monthly = daily_idx.groupby("symbol")[['open','high','low','close']].resample('ME').agg({'open':'first','high':'max','low':'min','close':'last'}).reset_index()
    monthly = monthly.sort_values(["symbol", "session"])
    monthly = monthly.groupby("symbol", group_keys=False).apply(lambda g: g.assign(prev_high=g.high.shift(1), prev_low=g.low.shift(1))).reset_index(drop=True)
    mlev = _levels(monthly, "m_")
    mlev["m_width"] = mlev.m_cpr_high - mlev.m_cpr_low
    mlev["m_prev_range"] = mlev.high - mlev.low
    mlev["m_width_ratio"] = mlev.m_width / mlev.m_prev_range.replace(0, np.nan)
    mlev["m_pd_high"] = mlev.prev_high
    mlev["m_pd_low"] = mlev.prev_low

    out = x.merge(daily_levels[["symbol","session","d_cpr_low","d_cpr_high","d_r1","d_s1","d_r2","d_s2","d_r3","d_s3","d_pd_high","d_pd_low","d_cam_r3","d_cam_s3","d_width_ratio","d_ascending","d_descending"]], on=["symbol","session"], how="left")
    wcols = ["symbol","session","w_cpr_low","w_cpr_high","w_r1","w_s1","w_r2","w_s2","w_r3","w_s3","w_pd_high","w_pd_low","w_width_ratio"]
    mcols = ["symbol","session","m_cpr_low","m_cpr_high","m_r1","m_s1","m_r2","m_s2","m_pd_high","m_pd_low","m_width_ratio"]
    # Map period-end level dates to the first session after that period.
    wlev["apply_session"] = wlev.session + pd.Timedelta(days=3)
    wlev["apply_session"] = wlev.apply_session.dt.normalize()
    mlev["apply_session"] = (mlev.session + pd.offsets.MonthEnd(1) + pd.Timedelta(days=1)).dt.normalize()
    out = out.merge(wlev[wcols + ["apply_session"]].rename(columns={"apply_session":"session"}), on=["symbol","session"], how="left")
    out = out.merge(mlev[mcols + ["apply_session"]].rename(columns={"apply_session":"session"}), on=["symbol","session"], how="left")
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
    x = df.copy().sort_values(["symbol","timestamp"]).reset_index(drop=True)
    out = pd.DataFrame(index=x.index)
    out["signal"] = 0
    out["strategy_id"] = strategy_id

    dwidth = x.apply(lambda r: width_class(r, "d", cfg), axis=1)
    near = dwidth.eq("narrow")
    wide = dwidth.eq("wide")
    weekly_above = x.close > x.w_cpr_high
    weekly_below = x.close < x.w_cpr_low
    daily_above = x.close > x.d_cpr_high
    daily_below = x.close < x.d_cpr_low

    if strategy_id == "camarilla_inside_reversal":
        r3_inside = x.d_cam_r3.between(x.d_cpr_low, x.d_cpr_high)
        s3_inside = x.d_cam_s3.between(x.d_cpr_low, x.d_cpr_high)
        short = r3_inside & _cross_down(x.close, x.d_cpr_low)
        long = s3_inside & _cross_up(x.close, x.d_cpr_high)
        out.loc[short, "signal"] = -1
        out.loc[long, "signal"] = 1

    elif strategy_id == "mean_reversion_r1_pdh_s1_pdl":
        upper = x[["d_r1","d_pd_high"]].max(axis=1)
        lower = x[["d_s1","d_pd_low"]].min(axis=1)
        short = (x.high > upper) & _cross_down(x.close, upper)
        long = (x.low < lower) & _cross_up(x.close, lower)
        out.loc[short, "signal"] = -1
        out.loc[long, "signal"] = 1

    elif strategy_id == "narrow_cpr_breakout":
        out.loc[near & _cross_up(x.close, x[["d_r1","d_pd_high"]].max(axis=1)), "signal"] = 1
        out.loc[near & _cross_down(x.close, x[["d_s1","d_pd_low"]].min(axis=1)), "signal"] = -1

    elif strategy_id == "virgin_cpr_reversal":
        # Virgin CPR detection is defined from a prior completed daily session.
        touched = (x.high >= x.d_cpr_low) & (x.low <= x.d_cpr_high)
        day_touch = touched.groupby([x.symbol, x.session]).transform("any")
        # A prior-session virgin zone is used only when it was untouched in its own session.
        daily = x[["symbol","session","d_cpr_low","d_cpr_high"]].drop_duplicates(["symbol","session"]).copy()
        daily["self_touched"] = ((x.groupby([x.symbol, x.session]).high.transform("max") >= x.d_cpr_low) & (x.groupby([x.symbol, x.session]).low.transform("min") <= x.d_cpr_high)).groupby([x.symbol, x.session]).transform("first")
        daily["virgin"] = ~daily.self_touched.fillna(True)
        for lag in range(1, cfg.virgin_max_age_sessions + 1):
            v = daily.groupby("symbol").virgin.shift(lag).fillna(False)
            lo = daily.groupby("symbol").d_cpr_low.shift(lag)
            hi = daily.groupby("symbol").d_cpr_high.shift(lag)
            m = x.merge(pd.DataFrame({"symbol":daily.symbol,"session":daily.session,"v":v.values,"lo":lo.values,"hi":hi.values}), on=["symbol","session"], how="left")
            out.loc[(m.v.fillna(False)) & (x.high >= m.lo) & (x.low <= m.hi) & _cross_down(x.close, m.hi), "signal"] = -1
            out.loc[(m.v.fillna(False)) & (x.high >= m.lo) & (x.low <= m.hi) & _cross_up(x.close, m.lo), "signal"] = 1

    elif strategy_id == "mtf_cpr_daytrade":
        long = weekly_above & daily_above & (_cross_up(x.close, x[["d_r1","d_pd_high"]].max(axis=1)) | _cross_up(x.close, x.d_cpr_high))
        short = weekly_below & daily_below & (_cross_down(x.close, x[["d_s1","d_pd_low"]].min(axis=1)) | _cross_down(x.close, x.d_cpr_low))
        out.loc[long, "signal"] = 1
        out.loc[short, "signal"] = -1

    elif strategy_id == "inside_ascending_descending":
        ascending = x.d_ascending.fillna(False)
        descending = x.d_descending.fillna(False)
        inside = (x.d_cpr_low >= x.groupby("symbol").d_cpr_low.shift(1)) & (x.d_cpr_high <= x.groupby("symbol").d_cpr_high.shift(1))
        long = (near | inside) & ascending & _cross_up(x.close, x[["d_r1","d_pd_high"]].max(axis=1))
        short = (near | inside) & descending & _cross_down(x.close, x[["d_s1","d_pd_low"]].min(axis=1))
        out.loc[long, "signal"] = 1
        out.loc[short, "signal"] = -1

    elif strategy_id == "masterclass_regime_open":
        upper = x.d_r1
        lower = x.d_s1
        far_up = x.open > upper
        far_dn = x.open < lower
        long = weekly_above & near & (~far_up) & _cross_up(x.close, upper)
        short = weekly_below & near & (~far_dn) & _cross_down(x.close, lower)
        out.loc[long, "signal"] = 1
        out.loc[short, "signal"] = -1
        out.loc[far_up & _cross_down(x.close, x.d_cpr_high), "signal"] = -1
        out.loc[far_dn & _cross_up(x.close, x.d_cpr_low), "signal"] = 1

    elif strategy_id == "weekly_cpr_masterclass":
        no_trade = wide & x.w_width_ratio.fillna(np.inf).gt(cfg.medium_pct_of_prev_range)
        long = (~no_trade) & weekly_above & near & _cross_up(x.close, x[["d_r1","d_pd_high"]].max(axis=1))
        short = (~no_trade) & weekly_below & near & _cross_down(x.close, x[["d_s1","d_pd_low"]].min(axis=1))
        out.loc[long, "signal"] = 1
        out.loc[short, "signal"] = -1

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
