from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# `python scripts/run_iron_condor_v2.py` puts scripts/ on sys.path rather than
# the repository root. Add ROOT explicitly so src/ imports are deterministic in
# GitHub Actions and local execution.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.iron_condor_engine_v1 import CostModel, lot_size_for_expiry

CFG = yaml.safe_load((ROOT / "config.iron_condor_v2.yaml").read_text())
CAPITAL = float(CFG["capital"])


class Market:
    def __init__(self, options: pd.DataFrame, futures: pd.DataFrame):
        o = options.copy()
        o["date"] = pd.to_datetime(o.date).dt.normalize()
        o["expiry"] = pd.to_datetime(o.expiry).dt.normalize()
        o["option_type"] = o.option_type.astype(str).str.upper()
        o = o.drop_duplicates(["date", "expiry", "strike", "option_type"], keep="last")
        f = futures.copy()
        f["date"] = pd.to_datetime(f.date).dt.normalize()
        f["expiry"] = pd.to_datetime(f.expiry).dt.normalize()
        f = f.sort_values(["date", "expiry"]).drop_duplicates(["date", "expiry"], keep="last")
        f = f[f.expiry >= f.date]
        spot = f.sort_values(["date", "expiry"]).groupby("date", as_index=False).first()[["date", "close"]]
        self.spot = dict(zip(spot.date, spot.close.astype(float)))
        self.dates = tuple(pd.Timestamp(x) for x in sorted(set(o.date) & set(spot.date)))
        self.expiries = {}
        self.strikes = {}
        self.px = {}
        for (d, e), g in o.groupby(["date", "expiry"], sort=False):
            d, e = pd.Timestamp(d), pd.Timestamp(e)
            if d not in self.spot:
                continue
            self.expiries.setdefault(d, []).append(e)
            self.strikes[(d, e)] = np.asarray(sorted(pd.to_numeric(g.strike, errors="coerce").dropna().unique()), dtype=float)
        for d in self.expiries:
            self.expiries[d] = tuple(sorted(self.expiries[d]))
        for d, e, k, t, close, high, low in o[["date", "expiry", "strike", "option_type", "close", "high", "low"]].itertuples(index=False, name=None):
            key = (pd.Timestamp(d), pd.Timestamp(e), float(k), str(t).upper())
            self.px[key] = {
                "close": float(close) if pd.notna(close) else np.nan,
                "high": float(high) if pd.notna(high) else np.nan,
                "low": float(low) if pd.notna(low) else np.nan,
            }

    def price(self, d, e, k, t, field="close"):
        r = self.px.get((pd.Timestamp(d), pd.Timestamp(e), float(k), str(t).upper()))
        if not r:
            return None
        v = r.get(field, np.nan)
        return None if pd.isna(v) else float(v)

    def expiry_for(self, d):
        for e in self.expiries.get(pd.Timestamp(d), ()):
            dte = (e - pd.Timestamp(d)).days
            if int(CFG["min_days_to_expiry"]) <= dte <= int(CFG["max_days_to_expiry"]):
                return e
        return None

    def choose(self, signal_date, expiry, distance, width):
        spot = self.spot.get(pd.Timestamp(signal_date))
        strikes = self.strikes.get((pd.Timestamp(signal_date), pd.Timestamp(expiry)))
        if spot is None or strikes is None or len(strikes) < 8:
            return None
        puts = strikes[strikes <= spot * (1.0 - distance)]
        calls = strikes[strikes >= spot * (1.0 + distance)]
        if len(puts) == 0 or len(calls) == 0:
            return None
        ps, cs = float(puts[-1]), float(calls[0])
        pls = strikes[strikes <= ps - width + 1e-9]
        cls = strikes[strikes >= cs + width - 1e-9]
        if len(pls) == 0 or len(cls) == 0:
            return None
        return float(pls[-1]), ps, cs, float(cls[0])


def score(trades: pd.DataFrame) -> float:
    if trades.empty:
        return -1e9
    t = trades.sort_values("exit_date").copy()
    weekly = t.groupby(t.exit_date.dt.to_period("W"))["net_pnl"].sum() / CAPITAL
    equity = CAPITAL + t.net_pnl.cumsum()
    dd = equity / equity.cummax() - 1.0
    win = float((t.net_pnl > 0).mean())
    return (
        2.0 * float(weekly.median())
        + float(weekly.mean())
        + 0.10 * float((weekly >= 0.05).mean())
        + 0.25 * float((weekly >= 0).mean())
        + 0.10 * win
        + 0.50 * float(weekly.min())
        + 0.50 * float(dd.min())
    )


def run_config(market: Market, params: dict, start=None, end=None, slippage=None, collect=True):
    slip = float(CFG["selection_slippage_per_leg_points"] if slippage is None else slippage)
    min_credit = float(params.get("minimum_credit", 0.0))
    rows = []
    active_until = None
    dates = [
        d for d in market.dates
        if d.weekday() == int(params["entry_weekday"])
        and (start is None or d >= start)
        and (end is None or d <= end)
    ]
    for signal_date in dates:
        if active_until is not None and signal_date <= active_until:
            continue
        expiry = market.expiry_for(signal_date)
        if expiry is None:
            continue
        legs4 = market.choose(signal_date, expiry, float(params["distance"]), int(params["width"]))
        if legs4 is None:
            continue
        leg_keys = [(legs4[0], "PE"), (legs4[1], "PE"), (legs4[2], "CE"), (legs4[3], "CE")]

        # Signal is generated from the completed signal session. Execution occurs
        # on the first subsequent eligible NSE session. A trade whose entry falls
        # outside the research window is excluded rather than allowed to borrow
        # information from the next window.
        entry_date = next((d for d in market.dates if d > signal_date and d <= expiry), None)
        if entry_date is None or (end is not None and entry_date > end):
            continue

        entry = [market.price(entry_date, expiry, k, t) for k, t in leg_keys]
        if any(v is None or v <= 0 for v in entry):
            continue
        raw_credit = entry[1] + entry[2] - entry[0] - entry[3]
        effective_credit = raw_credit - 4.0 * slip
        if raw_credit <= 0 or effective_credit < min_credit:
            continue
        lot = lot_size_for_expiry(expiry)
        max_loss_points = max(legs4[1] - legs4[0], legs4[3] - legs4[2]) - raw_credit
        if max_loss_points <= 0:
            continue

        exit_date = None
        reason = "expiry"
        exit_mark = None
        sim_days = [d for d in market.dates if entry_date < d <= expiry and (end is None or d <= end)]
        for day in sim_days:
            vals = [market.price(day, expiry, k, t) for k, t in leg_keys]
            if any(v is None for v in vals):
                continue
            mark = vals[1] + vals[2] - vals[0] - vals[3]
            exit_debit = -mark + 4.0 * slip
            pnl_points = effective_credit - exit_debit
            tp_target_debit = (1.0 - float(params["take_profit"])) * effective_credit
            sl_debit = (1.0 + float(params["stop_loss"])) * effective_credit
            if exit_debit <= tp_target_debit:
                exit_date, reason, exit_mark = day, "take_profit", mark
                break
            if exit_debit >= sl_debit:
                exit_date, reason, exit_mark = day, "stop_loss", mark
                break

        # Never let a training/OOS boundary trade be settled using information
        # after the requested end date. If no exit occurred inside the window and
        # expiry itself is beyond the window, the trade is omitted.
        if exit_date is None:
            if end is not None and expiry > end:
                continue
            exit_date = expiry
            vals = [market.price(exit_date, expiry, k, t) for k, t in leg_keys]
            if any(v is None for v in vals):
                continue
            exit_mark = vals[1] + vals[2] - vals[0] - vals[3]

        exit_px = [market.price(exit_date, expiry, k, t) for k, t in leg_keys]
        if any(v is None for v in exit_px):
            continue
        cost_model = CostModel()
        cost = cost_model.total([(entry[0], exit_px[0], 1), (entry[1], exit_px[1], -1), (entry[2], exit_px[2], -1), (entry[3], exit_px[3], 1)], lot)
        gross = (effective_credit + exit_mark - 4.0 * slip) * lot
        net = gross - cost
        active_until = pd.Timestamp(exit_date)
        if collect:
            rows.append({
                "signal_date": pd.Timestamp(signal_date),
                "entry_date": pd.Timestamp(entry_date),
                "exit_date": pd.Timestamp(exit_date),
                "expiry": pd.Timestamp(expiry),
                "entry_weekday": int(params["entry_weekday"]),
                "dte_at_signal": int((expiry - signal_date).days),
                "dte_at_entry": int((expiry - entry_date).days),
                "nifty_reference_signal": float(market.spot[signal_date]),
                "nifty_reference_entry": float(market.spot.get(entry_date, np.nan)),
                "put_long": legs4[0], "put_short": legs4[1], "call_short": legs4[2], "call_long": legs4[3],
                "put_long_entry": entry[0], "put_short_entry": entry[1], "call_short_entry": entry[2], "call_long_entry": entry[3],
                "put_long_exit": exit_px[0], "put_short_exit": exit_px[1], "call_short_exit": exit_px[2], "call_long_exit": exit_px[3],
                "raw_credit_points": raw_credit, "effective_credit_points": effective_credit,
                "minimum_credit": min_credit, "slippage_per_leg": slip,
                "wing_width": float(params["width"]), "distance": float(params["distance"]),
                "take_profit": float(params["take_profit"]), "stop_loss": float(params["stop_loss"]),
                "lot": int(lot), "max_loss": float(max_loss_points * lot),
                "lower_breakeven": float(legs4[1] - raw_credit), "upper_breakeven": float(legs4[2] + raw_credit),
                "gross_pnl": float(gross), "costs": float(cost), "net_pnl": float(net),
                "return_on_capital": float(net / CAPITAL), "return_on_max_loss": float(net / (max_loss_points * lot)),
                "reason": reason,
            })
    return pd.DataFrame(rows)


def summarize(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"trades": 0, "win_rate": 0.0, "net_pnl": 0.0, "return": 0.0, "max_drawdown": 0.0}
    t = t.sort_values(["exit_date", "entry_date"]).reset_index(drop=True)
    eq = CAPITAL + t.net_pnl.cumsum()
    dd = eq / eq.cummax() - 1.0
    return {
        "trades": int(len(t)),
        "win_rate": float((t.net_pnl > 0).mean()),
        "net_pnl": float(t.net_pnl.sum()),
        "return": float(t.net_pnl.sum() / CAPITAL),
        "median_trade": float(t.net_pnl.median()),
        "worst_trade": float(t.net_pnl.min()),
        "best_trade": float(t.net_pnl.max()),
        "max_drawdown": float(dd.min()),
        "take_profit_trades": int((t.reason == "take_profit").sum()),
        "stop_loss_trades": int((t.reason == "stop_loss").sum()),
        "expiry_trades": int((t.reason == "expiry").sum()),
    }


def select_config(market: Market, start, end):
    base = []
    for wd, distance, width, tp, sl in itertools.product(
        CFG["entry_weekdays"], CFG["short_distance_pct"], CFG["wing_width_points"],
        CFG["take_profit_credit_pct"], CFG["stop_loss_credit_multiple"]):
        p = {"entry_weekday": wd, "distance": distance, "width": width, "take_profit": tp, "stop_loss": sl, "minimum_credit": 0.0}
        t = run_config(market, p, start, end, slippage=CFG["selection_slippage_per_leg_points"])
        if len(t) >= int(CFG["min_training_trades"]):
            base.append((score(t), p, t))
    base.sort(key=lambda x: x[0], reverse=True)
    top = base[:10]
    candidates = []
    for _, p, _ in top:
        for mc in CFG["minimum_credit_points"]:
            q = {**p, "minimum_credit": mc}
            t = run_config(market, q, start, end, slippage=CFG["selection_slippage_per_leg_points"])
            if len(t) >= int(CFG["min_training_trades"]):
                candidates.append((score(t), q, t))
    if not candidates:
        raise RuntimeError("No V2 configuration met the minimum training trade requirement")
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0], base[:25]


def walk_forward(market: Market):
    dates = list(market.dates)
    n = len(dates)
    windows = int(CFG["walk_forward_windows"])
    results = []
    all_oos = []
    for i in range(windows):
        test_start_idx = int(round(n * (i + 1) / (windows + 1)))
        test_end_idx = int(round(n * (i + 2) / (windows + 1))) - 1
        train_start = dates[0]
        train_end = dates[test_start_idx - 1]
        test_start = dates[test_start_idx]
        test_end = dates[min(test_end_idx, n - 1)]
        (sc, params, train_trades), _ = select_config(market, train_start, train_end)
        oos = run_config(market, params, test_start, test_end, slippage=CFG["selection_slippage_per_leg_points"])
        s = summarize(oos)
        results.append({"window": i + 1, "train_start": str(train_start.date()), "train_end": str(train_end.date()), "test_start": str(test_start.date()), "test_end": str(test_end.date()), "selection_score": float(sc), **params, **s})
        if not oos.empty:
            oos["window"] = i + 1
            all_oos.append(oos)
    return pd.DataFrame(results), pd.concat(all_oos, ignore_index=True) if all_oos else pd.DataFrame()


def main():
    opt_path = ROOT / "data/cache/nifty_options_long.parquet"
    fut_path = ROOT / "data/cache/nifty_futures_long.parquet"
    if not opt_path.exists() or not fut_path.exists():
        raise FileNotFoundError("Run the existing NSE long-history downloader first")
    m = Market(pd.read_parquet(opt_path), pd.read_parquet(fut_path))
    dates = list(m.dates)
    if len(dates) < 100:
        raise RuntimeError("Not enough overlapping option/futures sessions for V2")
    split = int(len(dates) * 0.70)
    train_start, train_end = dates[0], dates[split - 1]
    oos_start, oos_end = dates[split], dates[-1]
    selected, top = select_config(m, train_start, train_end)
    selection_score, params, train_trades = selected
    oos_trades = run_config(m, params, oos_start, oos_end, slippage=CFG["selection_slippage_per_leg_points"])
    slippage_rows = []
    for s in CFG["slippage_per_leg_points"]:
        t = run_config(m, params, oos_start, oos_end, slippage=s)
        slippage_rows.append({"slippage_per_leg": s, **summarize(t)})
    wf, wf_trades = walk_forward(m)
    baseline = {"entry_weekday": 1, "distance": 0.0125, "width": 250, "take_profit": 0.40, "stop_loss": 1.0, "minimum_credit": 0.0}
    baseline_t = run_config(m, baseline, oos_start, oos_end, slippage=CFG["selection_slippage_per_leg_points"])
    out = ROOT / "backtest/results/iron-condor-v2"
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "strategy": "Iron Condor V2 — daily adaptive / delayed entry / credit quality / no overlap",
        "data_start": str(dates[0].date()), "data_end": str(dates[-1].date()),
        "holdout": {"train_start": str(train_start.date()), "train_end": str(train_end.date()), "oos_start": str(oos_start.date()), "oos_end": str(oos_end.date())},
        "selected_parameters": params,
        "selection_score": float(selection_score),
        "training": summarize(train_trades),
        "oos": summarize(oos_trades),
        "baseline_same_period": summarize(baseline_t),
        "slippage_sensitivity": slippage_rows,
        "walk_forward_windows": int(len(wf)),
    }
    (out / "v2_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    pd.DataFrame(top, columns=["score", "params", "trades"]).to_json(out / "training_top25.json", orient="records", date_format="iso")
    oos_trades.to_csv(out / "oos_trades.csv", index=False)
    baseline_t.to_csv(out / "baseline_oos_trades.csv", index=False)
    wf.to_csv(out / "walk_forward_results.csv", index=False)
    wf_trades.to_csv(out / "walk_forward_trades.csv", index=False)
    pd.DataFrame(slippage_rows).to_csv(out / "slippage_sensitivity.csv", index=False)
    (out / "oos_trades.json").write_text(oos_trades.to_json(orient="records", date_format="iso"))
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
