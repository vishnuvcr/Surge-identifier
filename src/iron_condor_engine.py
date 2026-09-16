from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    brokerage_per_order: float = 10.0
    stt_sell_pct: float = 0.0015
    stamp_buy_pct: float = 0.00003
    sebi_turnover_pct: float = 0.000001
    exchange_txn_pct: float = 0.00035
    gst_pct: float = 0.18

    def total(self, legs, lot):
        brokerage = 8 * self.brokerage_per_order
        sell_value = sum((e if s < 0 else x) for e, x, s in legs) * lot
        buy_value = sum((e if s > 0 else x) for e, x, s in legs) * lot
        turnover = sum(abs(e) + abs(x) for e, x, _ in legs) * lot
        txn = turnover * self.exchange_txn_pct
        return brokerage + max(sell_value, 0) * self.stt_sell_pct + max(buy_value, 0) * self.stamp_buy_pct + turnover * self.sebi_turnover_pct + txn + (brokerage + txn) * self.gst_pct


def _spot(r, d):
    x = r[(r.date == d) & r.underlying_close.notna()]
    return float(x.underlying_close.iloc[0]) if not x.empty else None


def _p(r, d, e, k, t):
    x = r[(r.date == d) & (r.expiry == e) & (r.strike == k) & (r.option_type == t)]
    if x.empty:
        return None
    z = pd.to_numeric(x.close, errors="coerce").dropna()
    return float(z.iloc[-1]) if not z.empty else None


def choose(r, d, e, distance, width):
    s = _spot(r, d)
    if s is None:
        return None
    strikes = np.array(sorted(r.loc[(r.date == d) & (r.expiry == e), "strike"].dropna().unique()), dtype=float)
    if len(strikes) < 8:
        return None
    ps = strikes[strikes <= s * (1 - distance)]
    cs = strikes[strikes >= s * (1 + distance)]
    if len(ps) == 0 or len(cs) == 0:
        return None
    ps, cs = float(ps[-1]), float(cs[0])
    pls, cls = strikes[strikes <= ps - width + 1e-9], strikes[strikes >= cs + width - 1e-9]
    if len(pls) == 0 or len(cls) == 0:
        return None
    return float(pls[-1]), ps, cs, float(cls[0])


def backtest(rows, cfg):
    r = rows.copy()
    for c in ["date", "expiry"]:
        r[c] = pd.to_datetime(r[c]).dt.normalize()
    trades = []
    seen = set()
    for d in sorted(r.date.unique()):
        d = pd.Timestamp(d)
        if d.weekday() != cfg.get("entry_weekday", 2):
            continue
        exps = sorted(r.loc[r.date == d, "expiry"].dropna().unique())
        e = next((pd.Timestamp(x) for x in exps if 1 <= (pd.Timestamp(x) - d).days <= 7), None)
        if e is None or e in seen:
            continue
        seen.add(e)
        strikes = choose(r, d, e, cfg["distance"], cfg["width"])
        if strikes is None:
            continue
        pl, ps, cs, cl = strikes
        ep = [_p(r, d, e, pl, "PE"), _p(r, d, e, ps, "PE"), _p(r, d, e, cs, "CE"), _p(r, d, e, cl, "CE")]
        if any(x is None or x <= 0 for x in ep):
            continue
        credit = ep[1] + ep[2] - ep[0] - ep[3]
        lot = 65 if d >= pd.Timestamp("2025-11-20") else 75
        max_loss = max(ps - pl, cl - cs) - credit
        if credit <= 0 or max_loss <= 0:
            continue
        exit_d, reason, mark = e, "expiry", None
        for day in sorted(pd.Timestamp(x) for x in r.loc[(r.date > d) & (r.date <= e), "date"].unique()):
            vals = [_p(r, day, e, pl, "PE"), _p(r, day, e, ps, "PE"), _p(r, day, e, cs, "CE"), _p(r, day, e, cl, "CE")]
            if any(x is None for x in vals):
                continue
            m = vals[1] + vals[2] - vals[0] - vals[3]
            pnl_pts = credit + m
            if pnl_pts >= cfg["take_profit"] * credit:
                exit_d, reason, mark = day, "take_profit", m
                break
            if pnl_pts <= -cfg["stop_loss"] * credit:
                exit_d, reason, mark = day, "stop_loss", m
                break
        if mark is None:
            vals = [_p(r, exit_d, e, pl, "PE"), _p(r, exit_d, e, ps, "PE"), _p(r, exit_d, e, cs, "CE"), _p(r, exit_d, e, cl, "CE")]
            if any(x is None for x in vals):
                continue
            mark = vals[1] + vals[2] - vals[0] - vals[3]
            xp = vals
        else:
            xp = [_p(r, exit_d, e, pl, "PE"), _p(r, exit_d, e, ps, "PE"), _p(r, exit_d, e, cs, "CE"), _p(r, exit_d, e, cl, "CE")]
            if any(x is None for x in xp):
                continue
        gross = (credit + mark) * lot
        legs = [(ep[0], xp[0], 1), (ep[1], xp[1], -1), (ep[2], xp[2], -1), (ep[3], xp[3], 1)]
        fee = CostModel().total(legs, lot)
        net = gross - fee
        trades.append({"entry_date": d, "exit_date": exit_d, "expiry": e, "put_long": pl, "put_short": ps, "call_short": cs, "call_long": cl, "credit": credit, "net_pnl": net, "costs": fee, "max_loss": max_loss * lot, "return_on_capital": net / cfg["capital"], "return_on_max_loss": net / (max_loss * lot), "reason": reason, "lot": lot})
    t = pd.DataFrame(trades)
    if t.empty:
        return t, {"data_ok": False, "trades": 0}
    weekly = t.groupby(t.exit_date.dt.to_period("W")).net_pnl.sum() / cfg["capital"]
    eq = cfg["capital"] + t.sort_values("exit_date").net_pnl.cumsum()
    peak = eq.cummax()
    s = {"data_ok": True, "trades": int(len(t)), "weeks": int(len(weekly)), "win_rate": float((t.net_pnl > 0).mean()), "avg_weekly_return": float(weekly.mean()), "median_weekly_return": float(weekly.median()), "p05_weekly_return": float(weekly.quantile(.05)), "worst_weekly_return": float(weekly.min()), "weeks_at_or_above_5pct": float((weekly >= .05).mean()), "weeks_nonnegative": float((weekly >= 0).mean()), "total_return": float(t.net_pnl.sum() / cfg["capital"]), "max_drawdown": float((eq / peak - 1).min()), "best_trade": float(t.net_pnl.max()), "worst_trade": float(t.net_pnl.min())}
    t["equity"] = eq.values
    return t, s
