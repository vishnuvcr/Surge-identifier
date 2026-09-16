from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostModel:
    """Modeled Paytm Money / India F&O costs for an 8-order four-leg round trip."""
    brokerage_per_order: float = 10.0
    stt_sell_pct: float = 0.0010
    stamp_buy_pct: float = 0.00003
    sebi_turnover_pct: float = 0.000001
    exchange_txn_pct: float = 0.00035
    gst_pct: float = 0.18

    def total(self, legs, lot):
        brokerage = 8 * self.brokerage_per_order
        sell_value = sum(entry if side < 0 else exit for entry, exit, side in legs) * lot
        buy_value = sum(entry if side > 0 else exit for entry, exit, side in legs) * lot
        turnover = sum(abs(entry) + abs(exit) for entry, exit, _ in legs) * lot
        txn = turnover * self.exchange_txn_pct
        stt = max(sell_value, 0.0) * self.stt_sell_pct
        stamp = max(buy_value, 0.0) * self.stamp_buy_pct
        sebi = turnover * self.sebi_turnover_pct
        gst = (brokerage + txn + sebi) * self.gst_pct
        return brokerage + stt + stamp + sebi + txn + gst


def lot_size_for_expiry(expiry: pd.Timestamp) -> int:
    e = pd.Timestamp(expiry).normalize()
    if e < pd.Timestamp("2015-10-30"):
        return 25
    if e < pd.Timestamp("2021-08-01"):
        return 75
    if e < pd.Timestamp("2024-05-02"):
        return 50
    if e < pd.Timestamp("2024-11-20"):
        return 25
    if e < pd.Timestamp("2026-01-06"):
        return 75
    return 65


class PreparedMarket:
    def __init__(self, rows: pd.DataFrame):
        r = rows.copy()
        r["date"] = pd.to_datetime(r["date"]).dt.normalize()
        r["expiry"] = pd.to_datetime(r["expiry"]).dt.normalize()
        r = r.sort_values(["date", "expiry", "strike", "option_type"])
        r = r.drop_duplicates(["date", "expiry", "strike", "option_type"], keep="last")
        r["option_type"] = r["option_type"].astype(str).str.upper()
        self.rows = r.reset_index(drop=True)
        self.dates = tuple(pd.Timestamp(x) for x in sorted(self.rows["date"].unique()))
        # Default is Tuesday (Python weekday=1); the research config can override it.
        self.entry_dates = tuple(d for d in self.dates if d.weekday() == 1)
        self.spot = self.rows.groupby("date")["underlying_close"].last().dropna().astype(float).to_dict()
        self.expiries_by_date = {}
        self.strikes_by_date_expiry = {}
        for (d, e), g in self.rows.groupby(["date", "expiry"], sort=False):
            d, e = pd.Timestamp(d), pd.Timestamp(e)
            self.expiries_by_date.setdefault(d, []).append(e)
            self.strikes_by_date_expiry[(d, e)] = np.asarray(sorted(pd.to_numeric(g.strike, errors="coerce").dropna().unique()), dtype=float)
        for d in self.expiries_by_date:
            self.expiries_by_date[d] = tuple(sorted(self.expiries_by_date[d]))
        self.prices, self.highs, self.lows = {}, {}, {}
        for d, e, k, typ, close, high, low in self.rows[["date","expiry","strike","option_type","close","high","low"]].itertuples(index=False, name=None):
            key = (pd.Timestamp(d), pd.Timestamp(e), float(k), str(typ).upper())
            if pd.notna(close): self.prices[key] = float(close)
            if pd.notna(high): self.highs[key] = float(high)
            if pd.notna(low): self.lows[key] = float(low)

    def price(self, d, e, k, typ):
        return self.prices.get((pd.Timestamp(d), pd.Timestamp(e), float(k), str(typ).upper()))

    def high(self, d, e, k, typ):
        return self.highs.get((pd.Timestamp(d), pd.Timestamp(e), float(k), str(typ).upper()))

    def low(self, d, e, k, typ):
        return self.lows.get((pd.Timestamp(d), pd.Timestamp(e), float(k), str(typ).upper()))


def prepare_market(rows: pd.DataFrame) -> PreparedMarket:
    return PreparedMarket(rows)


def choose(market: PreparedMarket, d, e, distance, width):
    spot = market.spot.get(pd.Timestamp(d))
    strikes = market.strikes_by_date_expiry.get((pd.Timestamp(d), pd.Timestamp(e)))
    if spot is None or strikes is None or len(strikes) < 8:
        return None
    puts = strikes[strikes <= spot * (1 - distance)]
    calls = strikes[strikes >= spot * (1 + distance)]
    if not len(puts) or not len(calls):
        return None
    ps, cs = float(puts[-1]), float(calls[0])
    pls, cls = strikes[strikes <= ps - width + 1e-9], strikes[strikes >= cs + width - 1e-9]
    if not len(pls) or not len(cls):
        return None
    return float(pls[-1]), ps, cs, float(cls[0])


def _mark(market, day, expiry, legs):
    vals = [market.price(day, expiry, k, t) for k, t in legs]
    if any(v is None for v in vals): return None, vals
    return vals[1] + vals[2] - vals[0] - vals[3], vals


def _conservative_intraday_bound(market, day, expiry, legs):
    highs = [market.high(day, expiry, k, t) for k, t in legs]
    lows = [market.low(day, expiry, k, t) for k, t in legs]
    if any(v is None for v in highs + lows): return None
    return lows[1] + lows[2] - highs[0] - highs[3]


def _metrics(t: pd.DataFrame, capital: float):
    t = t.sort_values(["exit_date", "entry_date"]).reset_index(drop=True)
    weekly = t.groupby(t.exit_date.dt.to_period("W")).net_pnl.sum() / capital
    fixed_equity = capital + t.net_pnl.cumsum()
    fixed_dd = fixed_equity / fixed_equity.cummax() - 1.0
    per_trade_return = t.net_pnl / capital
    theoretical_compound = capital * (1.0 + per_trade_return).cumprod()
    compound_dd = theoretical_compound / theoretical_compound.cummax() - 1.0
    gains = float(t.loc[t.net_pnl > 0, "net_pnl"].sum())
    losses = float(-t.loc[t.net_pnl < 0, "net_pnl"].sum())
    duration_days = max((t.exit_date.max() - t.entry_date.min()).days, 1)
    years = duration_days / 365.25
    return {
        "data_ok": True,
        "trades": int(len(t)), "weeks": int(len(weekly)),
        "win_rate": float((t.net_pnl > 0).mean()),
        "avg_weekly_return": float(weekly.mean()), "median_weekly_return": float(weekly.median()),
        "p05_weekly_return": float(weekly.quantile(0.05)), "worst_weekly_return": float(weekly.min()),
        "best_weekly_return": float(weekly.max()),
        "weeks_at_or_above_5pct": float((weekly >= 0.05).mean()),
        "weeks_nonnegative": float((weekly >= 0).mean()),
        "fixed_lot_total_return": float(t.net_pnl.sum() / capital),
        "fixed_lot_ending_equity": float(fixed_equity.iloc[-1]),
        "fixed_lot_max_drawdown": float(fixed_dd.min()),
        "theoretical_compounded_return": float(theoretical_compound.iloc[-1] / capital - 1.0),
        "theoretical_compounded_ending_equity": float(theoretical_compound.iloc[-1]),
        "theoretical_compounded_max_drawdown": float(compound_dd.min()),
        "annualized_fixed_lot_return": float((fixed_equity.iloc[-1] / capital) ** (1 / years) - 1) if fixed_equity.iloc[-1] > 0 else -1.0,
        "annualized_theoretical_compounded_return": float((theoretical_compound.iloc[-1] / capital) ** (1 / years) - 1) if theoretical_compound.iloc[-1] > 0 else -1.0,
        "weekly_sharpe": float(weekly.mean() / weekly.std(ddof=1) * np.sqrt(52)) if len(weekly) > 1 and weekly.std(ddof=1) > 0 else 0.0,
        "profit_factor": float(gains / losses) if losses > 0 else float("inf"),
        "best_trade": float(t.net_pnl.max()), "worst_trade": float(t.net_pnl.min()),
        "avg_trade": float(t.net_pnl.mean()), "median_trade": float(t.net_pnl.median()),
        "positive_trades": int((t.net_pnl > 0).sum()), "negative_trades": int((t.net_pnl <= 0).sum()),
        "take_profit_trades": int((t.reason == "take_profit").sum()),
        "stop_loss_trades": int((t.reason == "stop_loss").sum()),
        "expiry_trades": int((t.reason == "expiry").sum()),
        "negative_intraday_bound_trades": int((t.conservative_intraday_pnl < 0).sum()),
        "deep_intraday_bound_trades": int((t.conservative_intraday_pnl <= -0.5 * t.max_loss).sum()),
    }


def backtest_prepared(market: PreparedMarket, cfg):
    """Run a fixed configuration using the configured entry weekday and DTE window."""
    entry_weekday = int(cfg.get("entry_weekday", 1))
    min_dte = int(cfg.get("min_days_to_expiry", 1))
    max_dte = int(cfg.get("max_days_to_expiry", 7))
    if not 0 <= entry_weekday <= 6:
        raise ValueError("entry_weekday must be 0..6 using Python weekday convention (Mon=0, Tue=1, ...)")
    if min_dte < 0 or max_dte < min_dte:
        raise ValueError("Invalid expiry DTE window")

    entry_dates = tuple(d for d in market.dates if d.weekday() == entry_weekday)
    trades = []
    for d in entry_dates:
        expiry = next((x for x in market.expiries_by_date.get(d, ()) if min_dte <= (x - d).days <= max_dte), None)
        if expiry is None: continue
        strikes = choose(market, d, expiry, cfg["distance"], cfg["width"])
        if strikes is None: continue
        pl, ps, cs, cl = strikes
        legs = [(pl,"PE"),(ps,"PE"),(cs,"CE"),(cl,"CE")]
        entry = [market.price(d, expiry, k, t) for k,t in legs]
        if any(x is None or x <= 0 for x in entry): continue
        credit = entry[1] + entry[2] - entry[0] - entry[3]
        lot = lot_size_for_expiry(expiry)
        max_loss_points = max(ps-pl, cl-cs) - credit
        if credit <= 0 or max_loss_points <= 0: continue
        exit_date, reason, mark = expiry, "expiry", None
        adverse, adverse_day = np.nan, pd.NaT
        for day in [x for x in market.dates if d < x <= expiry]:
            bound = _conservative_intraday_bound(market, day, expiry, legs)
            if bound is not None:
                value = credit + bound
                if np.isnan(adverse) or value < adverse:
                    adverse, adverse_day = float(value), day
            m, _ = _mark(market, day, expiry, legs)
            if m is None: continue
            pnl_pts = credit + m
            if pnl_pts >= cfg["take_profit"] * credit:
                exit_date, reason, mark = day, "take_profit", m; break
            if pnl_pts <= -cfg["stop_loss"] * credit:
                exit_date, reason, mark = day, "stop_loss", m; break
        if mark is None:
            mark, _ = _mark(market, exit_date, expiry, legs)
            if mark is None: continue
        exit_px = [market.price(exit_date, expiry, k, t) for k,t in legs]
        if any(x is None for x in exit_px): continue
        gross = (credit + mark) * lot
        cost = CostModel().total([(entry[0],exit_px[0],1),(entry[1],exit_px[1],-1),(entry[2],exit_px[2],-1),(entry[3],exit_px[3],1)], lot)
        net = gross - cost
        trades.append({
            "entry_date": d, "exit_date": exit_date, "expiry": expiry,
            "put_long": pl, "put_short": ps, "call_short": cs, "call_long": cl,
            "entry_put_long": entry[0], "entry_put_short": entry[1], "entry_call_short": entry[2], "entry_call_long": entry[3],
            "exit_put_long": exit_px[0], "exit_put_short": exit_px[1], "exit_call_short": exit_px[2], "exit_call_long": exit_px[3],
            "credit": credit, "max_loss_points": max_loss_points, "max_loss": max_loss_points * lot,
            "breakeven_lower": ps - credit, "breakeven_upper": cs + credit,
            "gross_pnl": gross, "costs": cost, "net_pnl": net,
            "return_on_capital": net / cfg["capital"], "return_on_max_loss": net / (max_loss_points * lot),
            "reason": reason, "lot": lot,
            "conservative_intraday_pnl_points": adverse,
            "conservative_intraday_bound_date": adverse_day,
            "conservative_intraday_pnl": adverse * lot if not np.isnan(adverse) else np.nan,
        })
    t = pd.DataFrame(trades)
    if t.empty: return t, {"data_ok": False, "trades": 0}
    summary = _metrics(t, cfg["capital"])
    t = t.sort_values(["exit_date","entry_date"]).reset_index(drop=True)
    fixed = cfg["capital"] + t.net_pnl.cumsum()
    comp = cfg["capital"] * (1.0 + t.net_pnl / cfg["capital"]).cumprod()
    t["fixed_lot_equity"] = fixed
    t["fixed_lot_drawdown"] = fixed / fixed.cummax() - 1.0
    t["return_on_equity"] = t.net_pnl / t.fixed_lot_equity.shift(1).fillna(cfg["capital"])
    t["theoretical_compounded_equity"] = comp
    t["theoretical_compounded_drawdown"] = comp / comp.cummax() - 1.0
    return t, summary


def backtest(rows, cfg):
    return backtest_prepared(prepare_market(rows), cfg)
