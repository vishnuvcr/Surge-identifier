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
from src.moe_engine_phase63 import add_cpr_features, add_targets, score_experts, train_experts, train_router
from src.moe_engine_phase63_safe import route_and_rank
from src.nse_data import load_prices
from src.nse_fo import load_fo
from src.surge_model import build_features


def log(message: str, started: float) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message} | elapsed={time.perf_counter() - started:.1f}s", flush=True)


def main() -> None:
    started = time.perf_counter()
    cfg = yaml.safe_load((ROOT / "backtest/config_phase63.yaml").read_text())
    out = Path(os.environ.get("PHASE63_OUTPUT_DIR", ROOT / "backtest/results/phase63"))
    out.mkdir(parents=True, exist_ok=True)

    log("=== PHASE 6.3 ADAPTIVE CPR/CAMARILLA MoE: START ===", started)
    log(
        f"Config: lookback={cfg['lookback_days']}d, train_lookback={cfg['training_lookback_sessions']} sessions, "
        f"targets={cfg['target_levels']}, portfolios={cfg['portfolio_sizes']}",
        started,
    )

    log("STAGE 1/8: Loading NSE cash master data", started)
    prices = load_prices("data/cache", int(cfg["lookback_days"]))
    prices.date = pd.to_datetime(prices.date).dt.normalize()
    log(f"Cash data loaded: {len(prices):,} rows, {prices.date.min().date()} to {prices.date.max().date()}", started)

    log("STAGE 2/8: Loading NSE F&O master data", started)
    fo = load_fo("data/cache", int(cfg["lookback_days"]))
    if not fo.empty:
        fo.date = pd.to_datetime(fo.date).dt.normalize()
        data = prices.merge(fo, on=["date", "symbol"], how="left")
    else:
        data = prices
    log(f"F&O data loaded: {len(fo):,} rows; merged dataset={len(data):,} rows", started)

    log("STAGE 3/8: Building leakage-safe market/stock features", started)
    t = time.perf_counter()
    feat = build_features(data)
    log(f"Base features complete: {len(feat):,} rows in {time.perf_counter() - t:.1f}s", started)

    log("STAGE 4/8: Adding CPR/Camarilla + point-in-time event features", started)
    t = time.perf_counter()
    feat = add_cpr_features(feat)
    log(f"CPR/Camarilla features complete in {time.perf_counter() - t:.1f}s", started)
    news, corp = load_events(cfg)
    log(f"Event inputs loaded: news_rows={len(news):,}, corporate_action_rows={len(corp):,}", started)
    feat = add_point_in_time_events(feat, news, corp)

    log("STAGE 5/8: Building strictly next-session targets", started)
    t = time.perf_counter()
    feat = add_targets(feat, float(cfg["stop_loss_pct"]), tuple(cfg["target_levels"]))
    feat.date = pd.to_datetime(feat.date).dt.normalize()
    log(f"Targets complete in {time.perf_counter() - t:.1f}s", started)

    days = pd.DatetimeIndex(sorted(feat.date.unique()))
    eval_days = days[int(cfg["min_train_days"]):]
    if cfg.get("backtest_start"):
        eval_days = eval_days[eval_days >= pd.Timestamp(cfg["backtest_start"])]
    if cfg.get("backtest_end"):
        eval_days = eval_days[eval_days <= pd.Timestamp(cfg["backtest_end"])]
    next_day = dict(zip(days[:-1], days[1:]))
    prices_idx = prices.set_index(["date", "symbol"]).sort_index()
    price_days = set(prices_idx.index.get_level_values(0).unique())

    log(
        f"STAGE 6/8: Backtest universe ready: {len(days)} sessions, {len(eval_days)} OOS signal days, "
        f"{prices['symbol'].nunique():,} symbols",
        started,
    )
    t = time.perf_counter()
    eligible_set = liquidity_table(prices, int(cfg["liquidity_lookback_sessions"]), float(cfg["min_avg_turnover_cr"]))
    log(f"Liquidity eligibility table complete: {len(eligible_set):,} eligible day-symbol pairs in {time.perf_counter() - t:.1f}s", started)

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

    log("STAGE 7/8: Starting walk-forward OOS backtest", started)
    log("Each monthly refit uses ONLY data strictly before the signal day; no forced trades", started)

    for i, signal_day in enumerate(eval_days, 1):
        signal_day = pd.Timestamp(signal_day)
        diagnostics["signal_days"] += 1
        period = signal_day.to_period("M")

        if period != last_period:
            refit_started = time.perf_counter()
            log(f"MONTHLY REFIT START: {period} | signal_day={signal_day.date()}", started)
            prior = feat[(feat.date < signal_day) & feat.label_date.notna() & (feat.label_date < signal_day)].copy()
            train_days = sorted(prior.date.unique())[-int(cfg["training_lookback_sessions"]):]
            prior = prior[prior.date.isin(train_days)].copy()
            log(f"Training window prepared: {len(train_days)} sessions, {len(prior):,} rows", started)
            if len(train_days) >= int(cfg["min_train_days"]):
                model_started = time.perf_counter()
                bundles = train_experts(
                    prior,
                    tuple(cfg["target_levels"]),
                    float(cfg["stop_loss_pct"]),
                    int(cfg["seed"]),
                    int(cfg["min_positive_labels"]),
                )
                log(f"Expert models trained: {len(bundles)} experts in {time.perf_counter() - model_started:.1f}s", started)
                if len(bundles) >= 2:
                    router_started = time.perf_counter()
                    router = train_router(prior, tuple(cfg["target_levels"]), float(cfg["stop_loss_pct"]), int(cfg["seed"]))
                    log(f"OOF router trained: {'YES' if router is not None else 'NO'} in {time.perf_counter() - router_started:.1f}s", started)
                else:
                    router = None
                    log("OOF router skipped: fewer than 2 trained experts", started)
            else:
                bundles, router = {}, None
                log(f"MODEL SKIP: only {len(train_days)} training sessions; minimum is {cfg['min_train_days']}", started)
            diagnostics["expert_refits"].append({
                "period": str(period),
                "train_days": len(train_days),
                "experts": sorted(bundles),
                "router": router is not None,
            })
            log(
                f"MONTHLY REFIT END: {period} | experts={len(bundles)}, router={router is not None}, "
                f"duration={time.perf_counter() - refit_started:.1f}s",
                started,
            )
            last_period = period

        daily_picks = None
        entry_day = next_day.get(signal_day)
        if bundles and entry_day is not None and entry_day in price_days:
            symbols = {s for d, s in eligible_set if d == signal_day}
            universe = feat[(feat.date == signal_day) & feat.symbol.isin(symbols)].copy()
            diagnostics["candidate_rows"] += len(universe)
            if not universe.empty:
                score_started = time.perf_counter()
                scored = score_experts(bundles, universe, tuple(cfg["target_levels"]))
                picks = route_and_rank(scored, router, cfg)
                diagnostics["qualified_rows"] += len(picks)
                entry_symbols = prices_idx.loc[entry_day].index
                daily_picks = picks[picks.symbol.isin(entry_symbols)].copy()
                if i == 1 or i % 10 == 0 or len(daily_picks):
                    log(
                        f"OOS SCORING {i}/{len(eval_days)}: signal={signal_day.date()} entry={entry_day.date()} "
                        f"candidates={len(universe):,} qualified={len(picks):,} selected={len(daily_picks):,} "
                        f"score_time={time.perf_counter() - score_started:.2f}s",
                        started,
                    )

        if daily_picks is None or daily_picks.empty:
            diagnostics["no_trade_days"] += 1
            for ps in sizes:
                st = states[ps]
                st["daily"].append({
                    "signal_date": signal_day,
                    "entry_date": entry_day,
                    "executed": 0,
                    "starting_equity": st["capital"],
                    "ending_equity": st["capital"],
                    "daily_pnl": 0.0,
                    "daily_return_pct": 0.0,
                    "selected_count": 0,
                })
        else:
            diagnostics["trade_days"] += 1
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
                st["daily"].append({
                    "signal_date": signal_day,
                    "entry_date": entry_day,
                    "executed": 1,
                    "starting_equity": start_equity,
                    "ending_equity": st["capital"],
                    "daily_pnl": pnl,
                    "daily_return_pct": pnl / max(start_equity, 1.0),
                    "selected_count": len(chosen),
                })

            chosen_symbols = ", ".join(str(x) for x in daily_picks.head(max(sizes)).symbol.tolist())
            log(
                f"TRADE DAY: signal={signal_day.date()} entry={entry_day.date()} selected={len(daily_picks)} "
                f"symbols=[{chosen_symbols}]",
                started,
            )

        if i == 1 or i % 10 == 0 or i == len(eval_days):
            capital_snapshot = ", ".join(f"P{ps}=Rs.{states[ps]['capital']:,.0f}" for ps in sizes)
            log(
                f"PROGRESS {i}/{len(eval_days)} | trade_days={diagnostics['trade_days']} "
                f"no_trade_days={diagnostics['no_trade_days']} | {capital_snapshot}",
                started,
            )

    log("STAGE 7/8 COMPLETE: Walk-forward simulation finished", started)
    log(
        f"Simulation diagnostics: signal_days={diagnostics['signal_days']}, trade_days={diagnostics['trade_days']}, "
        f"no_trade_days={diagnostics['no_trade_days']}, candidate_rows={diagnostics['candidate_rows']:,}, "
        f"qualified_rows={diagnostics['qualified_rows']:,}",
        started,
    )

    log("STAGE 8/8: Calculating summaries and writing result files", started)
    summaries = {str(ps): summary(states[ps], cfg) for ps in sizes}
    result = {
        "phase": "6.3",
        "engine": "single_job_adaptive_tail_moe_cpr_camarilla_oof_router",
        "config": cfg,
        "diagnostics": diagnostics,
        "summaries": summaries,
        "runtime_seconds": time.perf_counter() - started,
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2, default=str))
    for ps in sizes:
        trades = pd.DataFrame(states[ps]["trades"])
        daily = pd.DataFrame(states[ps]["daily"])
        if not trades.empty:
            trades.to_csv(out / f"trades_p{ps}.csv", index=False)
        daily.to_csv(out / f"daily_p{ps}.csv", index=False)
        log(f"P{ps} results written: trades={len(trades):,}, daily_rows={len(daily):,}, final_capital=Rs.{states[ps]['capital']:,.2f}", started)

    log(f"=== PHASE 6.3 COMPLETE === total_runtime={time.perf_counter() - started:.1f}s", started)
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
