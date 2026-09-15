from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.run_backtest_phase61 import execute, liquidity_table, load_events, summary
from src.moe_engine import add_point_in_time_events
from src.moe_engine_phase63 import add_cpr_features, add_targets, route_and_rank, score_experts, train_experts, train_router
from src.nse_data import load_prices
from src.nse_fo import load_fo
from src.surge_model import build_features


def main() -> None:
    started = time.perf_counter()
    cfg = yaml.safe_load((ROOT / "backtest/config_phase63.yaml").read_text())
    out = Path(os.environ.get("PHASE63_OUTPUT_DIR", ROOT / "backtest/results/phase63"))
    out.mkdir(parents=True, exist_ok=True)

    print("=== PHASE 6.3 ADAPTIVE CPR/CAMARILLA MoE ===", flush=True)
    print(f"Python={sys.version.split()[0]} workspace={ROOT}", flush=True)

    prices = load_prices("data/cache", int(cfg["lookback_days"]))
    prices.date = pd.to_datetime(prices.date).dt.normalize()
    fo = load_fo("data/cache", int(cfg["lookback_days"]))
    if not fo.empty:
        fo.date = pd.to_datetime(fo.date).dt.normalize()
        data = prices.merge(fo, on=["date", "symbol"], how="left")
    else:
        data = prices
    print(f"Loaded prices={len(prices):,}; f&o={len(fo):,}", flush=True)

    feat = build_features(data)
    feat = add_cpr_features(feat)
    news, corp = load_events(cfg)
    feat = add_point_in_time_events(feat, news, corp)
    feat = add_targets(feat, float(cfg["stop_loss_pct"]), tuple(cfg["target_levels"]))
    feat.date = pd.to_datetime(feat.date).dt.normalize()

    days = pd.DatetimeIndex(sorted(feat.date.unique()))
    eval_days = days[int(cfg["min_train_days"]):]
    if cfg.get("backtest_start"):
        eval_days = eval_days[eval_days >= pd.Timestamp(cfg["backtest_start"])]
    if cfg.get("backtest_end"):
        eval_days = eval_days[eval_days <= pd.Timestamp(cfg["backtest_end"])]
    next_day = dict(zip(days[:-1], days[1:]))
    prices_idx = prices.set_index(["date", "symbol"]).sort_index()
    eligible_set = liquidity_table(prices, int(cfg["liquidity_lookback_sessions"]), float(cfg["min_avg_turnover_cr"]))

    sizes = [int(x) for x in cfg["portfolio_sizes"]]
    states = {ps: {"capital": float(cfg["initial_capital"]), "trades": [], "daily": []} for ps in sizes}
    bundles = {}
    router = None
    last_period = None
    diagnostics = {
        "phase": "6.3",
        "signal_days": 0,
        "candidate_rows": 0,
        "qualified_rows": 0,
        "trade_days": 0,
        "no_trade_days": 0,
        "expert_refits": [],
        "cpr_enabled": bool(cfg.get("include_cpr_camarilla", True)),
        "information_cutoff": "signal-day close; CPR/Camarilla levels use prior completed session",
    }

    for i, signal_day in enumerate(eval_days, 1):
        diagnostics["signal_days"] += 1
        period = pd.Timestamp(signal_day).to_period("M")
        if period != last_period:
            prior = feat[(feat.date < signal_day) & feat.label_date.notna() & (feat.label_date < signal_day)].copy()
            train_days = sorted(prior.date.unique())[-int(cfg["training_lookback_sessions"]):]
            prior = prior[prior.date.isin(train_days)].copy()
            if len(train_days) >= int(cfg["min_train_days"]):
                bundles = train_experts(prior, tuple(cfg["target_levels"]), float(cfg["stop_loss_pct"]), int(cfg["seed"]), int(cfg["min_positive_labels"]))
                router = train_router(prior, tuple(cfg["target_levels"]), float(cfg["stop_loss_pct"]), int(cfg["seed"])) if len(bundles) >= 2 else None
            else:
                bundles, router = {}, None
            diagnostics["expert_refits"].append({"period": str(period), "train_days": len(train_days), "experts": sorted(bundles), "router": router is not None})
            print(f"Refit {period}: train_days={len(train_days)} experts={len(bundles)} router={router is not None}", flush=True)
            last_period = period

        daily_picks = None
        if bundles:
            symbols = {s for d, s in eligible_set if d == signal_day}
            universe = feat[(feat.date == signal_day) & feat.symbol.isin(symbols)].copy()
            diagnostics["candidate_rows"] += len(universe)
            if not universe.empty:
                scored = score_experts(bundles, universe, tuple(cfg["target_levels"]))
                picks = route_and_rank(scored, router, cfg)
                diagnostics["qualified_rows"] += len(picks)
                entry_day = next_day.get(signal_day)
                if entry_day is not None and entry_day in prices.index if False else True:
                    if entry_day in prices_idx.index.get_level_values(0):
                        entry_symbols = prices_idx.loc[entry_day].index
                        daily_picks = picks[picks.symbol.isin(entry_symbols)].copy()

        if daily_picks is None or daily_picks.empty:
            diagnostics["no_trade_days"] += 1
            for ps in sizes:
                st = states[ps]
                st["daily"].append({"signal_date": signal_day, "entry_date": next_day.get(signal_day), "executed": 0, "starting_equity": st["capital"], "ending_equity": st["capital"], "daily_pnl": 0.0, "daily_return_pct": 0.0, "selected_count": 0})
            continue

        diagnostics["trade_days"] += 1
        entry_day = next_day[signal_day]
        for ps in sizes:
            st = states[ps]
            chosen = daily_picks.head(min(ps, int(cfg["max_candidates_per_day"]))).copy()
            start_equity = st["capital"]
            allocation = start_equity / max(len(chosen), 1)
            pnl = 0.0
            for _, pick in chosen.iterrows():
                row = prices_idx.loc[(entry_day, pick.symbol)]
                result = execute(row, allocation, cfg["costs"], float(pick.target), float(cfg["stop_loss_pct"]))
                result.update(pick.to_dict())
                result.update({"signal_date": signal_day, "entry_date": entry_day, "portfolio_size": ps})
                st["trades"].append(result)
                pnl += result["net_pnl"]
            st["capital"] += pnl
            st["daily"].append({"signal_date": signal_day, "entry_date": entry_day, "executed": 1, "starting_equity": start_equity, "ending_equity": st["capital"], "daily_pnl": pnl, "daily_return_pct": pnl / max(start_equity, 1.0), "selected_count": len(chosen)})

        if i == 1 or i % 20 == 0 or i == len(eval_days):
            print(f"Progress {i}/{len(eval_days)} elapsed={time.perf_counter()-started:.1f}s", flush=True)

    summaries = {str(ps): summary(states[ps], cfg) for ps in sizes}
    result = {"phase": "6.3", "engine": "single_job_adaptive_tail_moe_cpr_camarilla_oof_router", "config": cfg, "diagnostics": diagnostics, "summaries": summaries, "runtime_seconds": time.perf_counter() - started}
    (out / "summary.json").write_text(json.dumps(result, indent=2, default=str))
    for ps in sizes:
        trades = pd.DataFrame(states[ps]["trades"])
        daily = pd.DataFrame(states[ps]["daily"])
        if not trades.empty:
            trades.to_csv(out / f"trades_p{ps}.csv", index=False)
        daily.to_csv(out / f"daily_p{ps}.csv", index=False)
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
