from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class CostModel:
    """Approximate India equity/derivatives transaction costs for four-leg options."""
    brokerage_per_order: float = 20.0
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
        gst = (brokerage + txn) * self.gst_pct
        return brokerage + stt + stamp + sebi + txn + gst


def lot_size_for_expiry(expiry: pd.Timestamp) -> int:
    """Historical NIFTY weekly/monthly contract lot sizes by expiry vintage.

    Official NSE changes used here:
      < 2015-10-30: 25
      2015-10-30 through Jul/Aug 2021: 75
      Aug 2021 through May 2024: 50
      May 2024 through Nov 2024: 25
      Nov 2024 through Dec 2025: 75
      2026 onward: 65
    The backtest is weekly (1-7 DTE), so these expiry cutovers are the relevant
    contract-vintage boundaries.
    """
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
    """Compact lookup representation shared across hundreds of parameter runs."""

    def __init__(self, rows: pd.DataFrame):
        r = rows.copy()
        r["date"] = pd.to_datetime(r["date"]).dt.normalize()
        r["expiry"] = pd.to_datetime(r["expiry"]).dt.normalize()
        r = r.sort_values(["date", "expiry", "strike", "option_type"])
        r = r.drop_duplicates(["date", "expiry", "strike", "option_type"], keep="last")
        r["option_type"] = r["option_type"].astype(str).str.upper()
        self.rows = r.reset_index(drop=True)
        self.dates = tuple(pd.Timestamp(x) for x in sorted(self.rows["date"].unique()))
        self.entry_dates = tuple(d for d in self.dates if d.weekday() == 2)
        self.spot = (
            self.rows.dropna(subset=["underlying_close"])
            .groupby("date", as_index=True)["underlying_close"]
            .last()
            .astype(float)
            .to_dict()
        )
        self.expiries_by_date = {}
        self.strikes_by_date_expiry = {}
        for (d, e), g in self.rows.groupby(["date", "expiry"], sort=False):
            d = pd.Timestamp(d)
            e = pd.Timestamp(e)
            self.expiries_by_date.setdefault(d, []).append(e)
            self.strikes_by_date_expiry[(d, e)] = np.asarray(sorted(pd.to_numeric(g["strike"], errors="coerce").dropna().unique()), dtype=float)
        for d in self.expiries_by_date:
            self.expiries_by_date[d] = tuple(sorted(self.expiries_by_date[d]))
        price_df = self.rows.set_index(["date", "expiry", "strike", "option_type"])["close"]
        # A plain dict gives substantially faster repeated leg-price lookups than
        # scanning the full DataFrame for every one of 512 configurations.
        self.prices = {
            (pd.Timestamp(d), pd.Timestamp(e), float(k), str(t).upper()): float(v)
            for (d, e, k, t), v in price_df.items()
            if pd.notna(v)
        }

    def price(self, d, e, k, t):
        return self.prices.get((pd.Timestamp(d), pd.Timestamp(e), float(k), str(t).upper()))


def prepare_market(rows: pd.DataFrame) -> PreparedMarket:
    return PreparedMarket(rows)


def choose(market: PreparedMarket, d, e, distance, width):
    s = market.spot.get(pd.Timestamp(d))
    if s is None:
        return None
    strikes = market.strikes_by_date_expiry.get((pd.Timestamp(d), pd.Timestamp(e)))
    if strikes is None or len(strikes) < 8:
        return None
    ps = strikes[strikes <= s * (1 - distance)]
    cs = strikes[strikes >= s * (1 + distance)]
    if len(ps) == 0 or len(cs) == 0:
        return None
    ps, cs = float(ps[-1]), float(cs[0])
    pls = strikes[strikes <= ps - width + 1e-9]
    cls = strikes[strikes >= cs + width - 1e-9]
    if len(pls) == 0 or len(cls) == 0:
        return None
    return float(pls[-1]), ps, cs, float(cls[0])


def _mark(market: PreparedMarket, day, expiry, legs):
    vals = [market.price(day, expiry, k, t) for k, t in legs]
    if any(v is None for v in vals):
        return None, vals
    return vals[1] + vals[2] - vals[0] - vals[3], vals


def backtest_prepared(market: PreparedMarket, cfg):
    trades = []
    for d in market.entry_dates:
        exps = market.expiries_by_date.get(d, ())
        e = next((x for x in exps if 1 <= (x - d).days <= 7), None)
        if e is None:
            continue
        strikes = choose(market, d, e, cfg["distance"], cfg["width"])
        if strikes is None:
            continue
        pl, ps, cs, cl = strikes
        entry_legs = [(pl, "PE"), (ps, "PE"), (cs, "CE"), (cl, "CE")]
        ep = [market.price(d, e, k, t) for k, t in entry_legs]
        if any(x is None or x <= 0 for x in ep):
            continue
        credit = ep[1] + ep[2] - ep[0] - ep[3]
        lot = lot_size_for_expiry(e)
        max_loss = max(ps - pl, cl - cs) - credit
        if credit <= 0 or max_loss <= 0:
            continue

        exit_d, reason, mark = e, "expiry", None
        candidate_days = [x for x in market.dates if d < x <= e]
        for day in candidate_days:
            m, vals = _mark(market, day, e, entry_legs)
            if m is None:
                continue
            pnl_pts = credit + m
            if pnl_pts >= cfg["take_profit"] * credit:
                exit_d, reason, mark = day, "take_profit", m
                break
            if pnl_pts <= -cfg["stop_loss"] * credit:
                exit_d, reason, mark = day, "stop_loss", m
                break

        if mark is None:
            mark, vals = _mark(market, exit_d, e, entry_legs)
            if mark is None:
                continue

        xp = [market.price(exit_d, e, k, t) for k, t in entry_legs]
        if any(x is None for x in xp):
            continue
        gross = (credit + mark) * lot
        legs = [(ep[0], xp[0], 1), (ep[1], xp[1], -1), (ep[2], xp[2], -1), (ep[3], xp[3], 1)]
        fee = CostModel().total(legs, lot)
        net = gross - fee
        trades.append({
            "entry_date": d,
            "exit_date": exit_d,
            "expiry": e,
            "put_long": pl,
            "put_short": ps,
            "call_short": cs,
            "call_long": cl,
            "credit": credit,
            "net_pnl": net,
            "costs": fee,
            "max_loss": max_loss * lot,
            "return_on_capital": net / cfg["capital"],
            "return_on_max_loss": net / (max_loss * lot),
            "reason": reason,
            "lot": lot,
        })

    t = pd.DataFrame(trades)
    if t.empty:
        return t, {"data_ok": False, "trades": 0}

    t = t.sort_values(["exit_date", "entry_date"]).reset_index(drop=True)
    weekly = t.groupby(t.exit_date.dt.to_period("W")).net_pnl.sum() / cfg["capital"]
    simple_total = float(t.net_pnl.sum() / cfg["capital"])
    equity = cfg["capital"] + t.net_pnl.cumsum()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    compounded_return = float(equity.iloc[-1] / cfg["capital"] - 1.0)
    weekly_std = float(weekly.std(ddof=1)) if len(weekly) > 1 else 0.0
    weekly_sharpe = float(weekly.mean() / weekly_std * np.sqrt(52)) if weekly_std > 0 else 0.0
    gains = t.loc[t.net_pnl > 0, "net_pnl"].sum()
    losses = -t.loc[t.net_pnl < 0, "net_pnl"].sum()
    profit_factor = float(gains / losses) if losses > 0 else float("inf")
    duration = (t.exit_date.max() - t.entry_date.min()).days
    years = max(duration / 365.25, 1 / 365.25)
    annualized = float((equity.iloc[-1] / cfg["capital"]) ** (1 / years) - 1.0) if equity.iloc[-1] > 0 else -1.0

    summary = {
        "data_ok": True,
        "trades": int(len(t)),
        "weeks": int(len(weekly)),
        "win_rate": float((t.net_pnl > 0).mean()),
        "avg_weekly_return": float(weekly.mean()),
        "median_weekly_return": float(weekly.median()),
        "p05_weekly_return": float(weekly.quantile(.05)),
        "worst_weekly_return": float(weekly.min()),
        "best_weekly_return": float(weekly.max()),
        "weeks_at_or_above_5pct": float((weekly >= .05).mean()),
        "weeks_nonnegative": float((weekly >= 0).mean()),
        "simple_total_return": simple_total,
        "compounded_return": compounded_return,
        "ending_equity": float(equity.iloc[-1]),
        "max_drawdown": float(drawdown.min()),
        "annualized_return": annualized,
        "weekly_sharpe": weekly_sharpe,
        "profit_factor": profit_factor,
        "best_trade": float(t.net_pnl.max()),
        "worst_trade": float(t.net_pnl.min()),
        "avg_trade": float(t.net_pnl.mean()),
        "median_trade": float(t.net_pnl.median()),
        "positive_trades": int((t.net_pnl > 0).sum()),
        "negative_trades": int((t.net_pnl <= 0).sum()),
    }
    t["equity"] = equity.values
    t["drawdown"] = drawdown.values
    t["return_on_equity"] = t["net_pnl"] / t["equity"].shift(1).fillna(cfg["capital"])
    return t, summary


def backtest(rows, cfg):
    return backtest_prepared(prepare_market(rows), cfg)
