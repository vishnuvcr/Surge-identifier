from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.iron_condor_engine_v1 import CostModel, choose, lot_size_for_expiry

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "backtest/results/success-v1-validation"
OUT.mkdir(parents=True, exist_ok=True)


def load_data():
    options = pd.read_parquet(os.environ.get("IRON_CONDOR_DATA_PATH", "data/cache/nifty_options_long.parquet"))
    futures = pd.read_parquet(os.environ.get("IRON_CONDOR_FUTURES_PATH", "data/cache/nifty_futures_long.parquet"))
    for df, cols in ((options, ("date", "expiry")), (futures, ("date", "expiry"))):
        for c in cols:
            df[c] = pd.to_datetime(df[c]).dt.normalize()
    options = options.loc[options.option_type.isin(["CE", "PE"])].copy()
    futures["close"] = pd.to_numeric(futures["close"], errors="coerce")
    futures = futures.dropna(subset=["close"])
    futures["dte"] = (futures.expiry - futures.date).dt.days
    futures = futures.loc[futures.dte >= 0].sort_values(["date", "dte"])
    futures = futures.groupby("date", as_index=False).first()[["date", "close"]].rename(columns={"close": "underlying_close"})
    options = options.merge(futures, on="date", how="left", validate="many_to_one")
    options["dte"] = (options.expiry - options.date).dt.days
    options = options.loc[options.dte.between(0, 7)].drop(columns=["dte"])
    options = options.sort_values(["date", "expiry", "strike", "option_type"]).drop_duplicates(["date", "expiry", "strike", "option_type"], keep="last")
    return options.reset_index(drop=True)


def prepare(x):
    dates = tuple(pd.Timestamp(d) for d in sorted(x.date.unique()))
    prices, highs, lows, strikes, expiries, under = {}, {}, {}, {}, {}, {}
    for r in x.itertuples(index=False):
        d, e, k, typ = pd.Timestamp(r.date), pd.Timestamp(r.expiry), float(r.strike), str(r.option_type).upper()
        key = (d, e, k, typ)
        if pd.notna(r.close): prices[key] = float(r.close)
        if pd.notna(r.high): highs[key] = float(r.high)
        if pd.notna(r.low): lows[key] = float(r.low)
        strikes.setdefault((d, e), set()).add(k)
        expiries.setdefault(d, set()).add(e)
        if pd.notna(r.underlying_close): under[d] = float(r.underlying_close)
    strikes = {k: np.asarray(sorted(v), dtype=float) for k, v in strikes.items()}
    expiries = {k: tuple(sorted(v)) for k, v in expiries.items()}
    return {"dates": dates, "prices": prices, "highs": highs, "lows": lows, "strikes": strikes, "expiries": expiries, "under": under}


def price(m, d, e, k, typ):
    return m["prices"].get((pd.Timestamp(d), pd.Timestamp(e), float(k), typ))


def choose_legs(m, signal_date, expiry, distance, width):
    s = m["under"].get(pd.Timestamp(signal_date))
    ks = m["strikes"].get((pd.Timestamp(signal_date), pd.Timestamp(expiry)))
    if s is None or ks is None or len(ks) < 8:
        return None
    puts = ks[ks <= s * (1 - distance)]
    calls = ks[ks >= s * (1 + distance)]
    if not len(puts) or not len(calls):
        return None
    ps, cs = float(puts[-1]), float(calls[0])
    pls = ks[ks <= ps - width + 1e-9]
    cls = ks[ks >= cs + width - 1e-9]
    if not len(pls) or not len(cls):
        return None
    return float(pls[-1]), ps, cs, float(cls[0])


def mark(m, d, e, legs):
    vals = [price(m, d, e, k, t) for k, t in legs]
    if any(v is None for v in vals):
        return None
    return vals[1] + vals[2] - vals[0] - vals[3]


def bound(m, d, e, legs):
    hs = [m["highs"].get((pd.Timestamp(d), pd.Timestamp(e), k, t)) for k, t in legs]
    ls = [m["lows"].get((pd.Timestamp(d), pd.Timestamp(e), k, t)) for k, t in legs]
    if any(v is None for v in hs + ls):
        return None
    return ls[1] + ls[2] - hs[0] - hs[3]


def trade(m, signal_date, cfg, delayed=True, slippage_per_leg=0.0):
    exp = next((e for e in m["expiries"].get(pd.Timestamp(signal_date), ()) if cfg["min_dte"] <= (e-pd.Timestamp(signal_date)).days <= cfg["max_dte"]), None)
    if exp is None:
        return None
    legs = choose_legs(m, signal_date, exp, cfg["distance"], cfg["width"])
    if legs is None:
        return None
    pl, ps, cs, cl = legs
    lt = [(pl, "PE"), (ps, "PE"), (cs, "CE"), (cl, "CE")]
    dates_after = [d for d in m["dates"] if d > pd.Timestamp(signal_date) and d <= exp]
    if delayed:
        if not dates_after:
            return None
        entry_date = dates_after[0]
    else:
        entry_date = pd.Timestamp(signal_date)
    ep = [price(m, entry_date, exp, k, t) for k, t in lt]
    if any(v is None or v <= 0 for v in ep):
        return None
    credit = ep[1] + ep[2] - ep[0] - ep[3]
    lot = lot_size_for_expiry(exp)
    max_loss_points = max(ps-pl, cl-cs) - credit
    if credit <= 0 or max_loss_points <= 0:
        return None

    # Entry slippage: long buys worse, short sells worse.
    effective_credit = credit - 4.0 * slippage_per_leg
    exit_date, reason, mmark = exp, "expiry", None
    adverse = np.nan
    adverse_date = pd.NaT
    exit_start = entry_date if not delayed else entry_date
    exit_dates = [d for d in m["dates"] if d > exit_start and d <= exp]
    for d in exit_dates:
        b = bound(m, d, exp, lt)
        if b is not None:
            adverse = min(adverse if not np.isnan(adverse) else 1e99, effective_credit + b)
            if adverse == 1e99:
                adverse_date = d
        mm = mark(m, d, exp, lt)
        if mm is None:
            continue
        pnl_pts = effective_credit + mm - 4.0 * slippage_per_leg
        if pnl_pts >= cfg["tp"] * effective_credit:
            exit_date, reason, mmark = d, "take_profit", mm
            break
        if pnl_pts <= -cfg["sl"] * effective_credit:
            exit_date, reason, mmark = d, "stop_loss", mm
            break
    if mmark is None:
        mmark = mark(m, exp, exp, lt)
        if mmark is None:
            return None
    xp = [price(m, exit_date, exp, k, t) for k, t in lt]
    if any(v is None for v in xp):
        return None
    gross = (effective_credit + mmark - 4.0 * slippage_per_leg) * lot
    costs = CostModel().total([(ep[0], xp[0], 1), (ep[1], xp[1], -1), (ep[2], xp[2], -1), (ep[3], xp[3], 1)], lot)
    net = gross - costs
    min_bound = np.nan
    for d in m["dates"]:
        if d > entry_date and d <= exp:
            b = bound(m, d, exp, lt)
            if b is not None:
                v = effective_credit + b - 4.0 * slippage_per_leg
                if np.isnan(min_bound) or v < min_bound:
                    min_bound = v
                    adverse_date = d
    return {
        "signal_date": pd.Timestamp(signal_date), "entry_date": entry_date, "exit_date": exit_date, "expiry": exp,
        "put_long": pl, "put_short": ps, "call_short": cs, "call_long": cl,
        "credit_raw": credit, "credit_after_entry_slippage": effective_credit, "slippage_per_leg": slippage_per_leg,
        "lot": lot, "gross_pnl": gross, "costs": costs, "net_pnl": net, "return_on_capital": net / cfg["capital"],
        "reason": reason, "conservative_intraday_pnl_points": min_bound, "conservative_intraday_bound_date": adverse_date,
        "max_loss": max_loss_points * lot,
    }


def run_config(data, cfg, start, end, delayed=True, slippage_per_leg=0.0):
    x = data.loc[(data.date >= pd.Timestamp(start)) & (data.date <= pd.Timestamp(end))].copy()
    m = prepare(x)
    dates = [d for d in m["dates"] if d.weekday() == cfg["weekday"]]
    rows = []
    for d in dates:
        tr = trade(m, d, cfg, delayed=delayed, slippage_per_leg=slippage_per_leg)
        if tr is not None:
            rows.append(tr)
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    return t.sort_values(["exit_date", "signal_date"]).reset_index(drop=True)


def metrics(t, capital):
    if t.empty:
        return {"trades": 0}
    return {
        "trades": int(len(t)),
        "win_rate": float((t.net_pnl > 0).mean()),
        "negative_trades": int((t.net_pnl <= 0).sum()),
        "total_return": float(t.net_pnl.sum() / capital),
        "avg_trade": float(t.net_pnl.mean()),
        "median_trade": float(t.net_pnl.median()),
        "worst_trade": float(t.net_pnl.min()),
        "best_trade": float(t.net_pnl.max()),
        "negative_bound_trades": int((t.conservative_intraday_pnl_points < 0).sum()),
        "deep_bound_trades": int((t.conservative_intraday_pnl_points <= -0.5 * (t.max_loss / t.lot)).sum()),
        "tp_exits": int((t.reason == "take_profit").sum()),
        "sl_exits": int((t.reason == "stop_loss").sum()),
        "expiry_exits": int((t.reason == "expiry").sum()),
    }


def configs(cfg):
    for d, w, tp, sl in itertools.product(cfg["short_distance_pct"], cfg["wing_width_points"], cfg["take_profit_credit_pct"], cfg["stop_loss_credit_multiple"]):
        yield {"capital": cfg["capital"], "weekday": cfg["entry_weekday"], "min_dte": cfg["min_days_to_expiry"], "max_dte": cfg["max_days_to_expiry"], "distance": d, "width": w, "tp": tp, "sl": sl}


def score(t, capital):
    if t.empty:
        return -1e9
    m = metrics(t, capital)
    return 2*t.net_pnl.median()/capital + t.net_pnl.mean()/capital - 0.5*max(0, -t.net_pnl.min()/capital)


def select(data, cfg, start, end):
    rows = []
    best, best_score = None, -1e99
    for i, c in enumerate(configs(cfg)):
        t = run_config(data, c, start, end, delayed=True, slippage_per_leg=0)
        s = score(t, c["capital"])
        rows.append({**c, "config_id": i, **metrics(t, c["capital"]), "score": s})
        if s > best_score and not t.empty:
            best, best_score = c, s
    return best, pd.DataFrame(rows).sort_values("score", ascending=False)


def main():
    cfg = yaml.safe_load((ROOT / "config/success_v1.yaml").read_text())
    data = load_data()
    base = {"capital": cfg["capital"], "weekday": cfg["entry_weekday"], "min_dte": cfg["min_days_to_expiry"], "max_dte": cfg["max_days_to_expiry"], "distance": cfg["selected_distance"], "width": cfg["selected_wing_width"], "tp": cfg["selected_take_profit"], "sl": cfg["selected_stop_loss"]}

    dates = sorted(data.date.unique())
    split = pd.Timestamp("2024-02-02")
    oos = data.loc[data.date >= split]
    same = run_config(oos, base, oos.date.min(), oos.date.max(), delayed=False, slippage_per_leg=0)
    delayed = run_config(oos, base, oos.date.min(), oos.date.max(), delayed=True, slippage_per_leg=0)
    pd.DataFrame([{"model":"same_day_close","slippage":0, **metrics(same, base["capital"])}, {"model":"next_session_close","slippage":0, **metrics(delayed, base["capital"])}]).to_csv(OUT / "execution_bias_comparison.csv", index=False)
    delayed.to_csv(OUT / "delayed_entry_trades.csv", index=False)

    slips = []
    for s in (0.0, 0.25, 0.5, 1.0):
        t = run_config(oos, base, oos.date.min(), oos.date.max(), delayed=True, slippage_per_leg=s)
        slips.append({"slippage_per_leg": s, **metrics(t, base["capital"])})
    pd.DataFrame(slips).to_csv(OUT / "slippage_sensitivity.csv", index=False)

    # Conservative path-risk audit of the delayed-entry model.
    if not delayed.empty:
        delayed["bound_negative"] = delayed.conservative_intraday_pnl_points < 0
        delayed["bound_below_half_max_loss"] = delayed.conservative_intraday_pnl_points <= -0.5 * (delayed.max_loss / delayed.lot)
        delayed[["signal_date","entry_date","exit_date","expiry","net_pnl","reason","conservative_intraday_pnl_points","conservative_intraday_bound_date","bound_negative","bound_below_half_max_loss"]].to_csv(OUT / "path_risk_audit.csv", index=False)

    windows = [("2015-01-01","2018-12-31","2019-01-01","2019-12-31"),("2015-01-01","2019-12-31","2020-01-01","2020-12-31"),("2015-01-01","2020-12-31","2021-01-01","2021-12-31"),("2015-01-01","2021-12-31","2022-01-01","2022-12-31"),("2015-01-01","2022-12-31","2023-01-01","2023-12-31"),("2015-01-01","2023-12-31","2024-01-01","2024-12-31"),("2015-01-01","2024-12-31","2025-01-01","2025-12-31"),("2015-01-01","2025-12-31","2026-01-01","2026-09-15")]
    wf = []
    for tr0, tr1, te0, te1 in windows:
        best, board = select(data, cfg, tr0, tr1)
        test = run_config(data, best, te0, te1, delayed=True, slippage_per_leg=0)
        wf.append({"train_start": tr0, "train_end": tr1, "test_start": te0, "test_end": te1, "selected_distance": best["distance"], "selected_width": best["width"], "selected_tp": best["tp"], "selected_sl": best["sl"], **metrics(test, cfg["capital"])})
    pd.DataFrame(wf).to_csv(OUT / "walk_forward_results.csv", index=False)

    summary = {
        "method": "Execution/path validation of Success V1 using prior-session signal and next-session EOD entry",
        "data": {"start": str(data.date.min().date()), "end": str(data.date.max().date()), "rows_after_filter": int(len(data)), "dates": int(data.date.nunique())},
        "fixed_candidate": base,
        "same_day_close": metrics(same, base["capital"]),
        "next_session_close": metrics(delayed, base["capital"]),
        "slippage_points_per_leg": [0, 0.25, 0.5, 1.0],
        "walk_forward_windows": len(wf),
        "note": "This validation removes same-day strike-selection/entry coincidence. Daily OHLC still cannot reconstruct exact intraday leg ordering, so path flags are conservative bounds, not assertions of actual fills.",
    }
    (OUT / "validation_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
