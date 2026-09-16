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

from backtest.run_backtest_phase61 import liquidity_table, load_events, summary
from src.moe_engine import add_point_in_time_events
from src.moe_engine_phase65 import (
    add_bidirectional_targets,
    score_experts,
    select_top_predictions,
    train_contrarian_factors,
    train_experts,
    train_router,
)
from src.nse_data import load_prices
from src.nse_fo import load_fo
from src.surge_model import build_features


def log(message: str, started: float) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message} | elapsed={time.perf_counter() - started:.1f}s", flush=True)


def fees(buy: float, sell: float, c: dict) -> float:
    turnover = buy + sell
    exchange = turnover * c["exchange_turnover_rate"]
    brokerage = 2.0 * c["brokerage_per_order"]
    stt = sell * c["stt_sell_rate"]
    sebi = turnover * c["sebi_turnover_rate"]
    stamp = buy * c["stamp_buy_rate"]
    gst = c["gst_rate"] * (brokerage + exchange)
    return brokerage + exchange + stt + sebi + stamp + gst


def execute(row, allocation: float, costs: dict, direction: str, target: float, stop_loss: float) -> dict:
    raw_open, raw_high, raw_low, raw_close = map(float, [row.open, row.high, row.low, row.close])
    slip = float(costs["slippage_per_side"])

    if direction == "LONG":
        entry = raw_open * (1.0 + slip)
        tp = entry * (1.0 + target)
        sl = entry * (1.0 - stop_loss)
        both = raw_low <= sl and raw_high >= tp
        if both:
            exit_price = sl * (1.0 - slip)
            reason = "stop_and_target_same_bar_conservative_stop"
        elif raw_high >= tp:
            exit_price = tp * (1.0 - slip)
            reason = "take_profit"
        elif raw_low <= sl:
            exit_price = sl * (1.0 - slip)
            reason = "stop_loss"
        else:
            exit_price = raw_close * (1.0 - slip)
            reason = "close_proxy"
        variable_buy = costs["exchange_turnover_rate"] + costs["sebi_turnover_rate"] + costs["stamp_buy_rate"]
        qty = int(max((allocation - costs["brokerage_per_order"]) / (entry * (1.0 + variable_buy)), 0))
        if qty <= 0:
            return {"net_pnl": 0.0, "gross_pnl": 0.0, "fees": 0.0, "return_pct": 0.0, "exit_reason": "no_qty"}
        buy = qty * entry
        sell = qty * exit_price
    else:
        # Intraday short: sell first near the open, then cover later. Slippage is
        # adverse on both legs. The same conservative stop-first rule applies.
        entry = raw_open * (1.0 - slip)
        tp = entry * (1.0 - target)
        sl = entry * (1.0 + stop_loss)
        both = raw_high >= sl and raw_low <= tp
        if both:
            exit_price = sl * (1.0 + slip)
            reason = "stop_and_target_same_bar_conservative_stop"
        elif raw_low <= tp:
            exit_price = tp * (1.0 + slip)
            reason = "take_profit"
        elif raw_high >= sl:
            exit_price = sl * (1.0 + slip)
            reason = "stop_loss"
        else:
            exit_price = raw_close * (1.0 + slip)
            reason = "close_proxy"
        variable_cover = costs["exchange_turnover_rate"] + costs["sebi_turnover_rate"] + costs["stamp_buy_rate"]
        qty = int(max((allocation - costs["brokerage_per_order"]) / (entry * (1.0 + variable_cover)), 0))
        if qty <= 0:
            return {"net_pnl": 0.0, "gross_pnl": 0.0, "fees": 0.0, "return_pct": 0.0, "exit_reason": "no_qty"}
        sell = qty * entry
        buy = qty * exit_price

    gross = sell - buy if direction == "LONG" else sell - buy
    cost = fees(buy, sell, costs)
    net = gross - cost if direction == "LONG" else gross - cost
    return {
        "qty": qty,
        "entry_price": entry,
        "exit_price": exit_price,
        "buy_value": buy,
        "sell_value": sell,
        "gross_pnl": gross,
        "net_pnl": net,
        "fees": cost,
        "return_pct": net / max(allocation, 1.0),
        "exit_reason": reason,
    }


def main() -> None:
    started = time.perf_counter()
    cfg = yaml.safe_load((ROOT / "backtest/config_phase65.yaml").read_text())
    out = Path(os.environ.get("PHASE65_OUTPUT_DIR", ROOT / "backtest/results/phase65"))
    out.mkdir(parents=True, exist_ok=True)

    log("=== PHASE 6.5 BIDIRECTIONAL ADAPTIVE MoE: START ===", started)
    log(f"Config: lookback={cfg['lookback_days']}d train={cfg['training_lookback_sessions']} targets={cfg['target_levels']} portfolios={cfg['portfolio_sizes']} shorting={cfg['allow_short']}", started)

    prices = load_prices("data/cache", int(cfg["lookback_days"]))
    prices.date = pd.to_datetime(prices.date).dt.normalize()
    fo = load_fo("data/cache", int(cfg["lookback_days"]))
    if not fo.empty:
        fo.date = pd.to_datetime(fo.date).dt.normalize()
        data = prices.merge(fo, on=["date", "symbol"], how="left")
    else:
        data = prices
    log(f"Cash={len(prices):,} F&O={len(fo):,} merged={len(data):,}", started)

    feat = build_features(data)
    # CPR/Camarilla remains an expert, never a gate.
    from src.moe_engine_phase63 import add_cpr_features
    feat = add_cpr_features(feat)
    news, corp = load_events(cfg)
    feat = add_point_in_time_events(feat, news, corp)
    feat = add_bidirectional_targets(feat, float(cfg["stop_loss_pct"]), tuple(cfg["target_levels"]))
    feat.date = pd.to_datetime(feat.date).dt.normalize()
    log(f"Features/targets ready={len(feat):,}; news={len(news):,}; corp={len(corp):,}", started)

    days = pd.DatetimeIndex(sorted(feat.date.unique()))
    eval_days = days[int(cfg["min_train_days"]):]
    if cfg.get("backtest_start"):
        eval_days = eval_days[eval_days >= pd.Timestamp(cfg["backtest_start"])]
    if cfg.get("backtest_end"):
        eval_days = eval_days[eval_days <= pd.Timestamp(cfg["backtest_end"])]
    next_day = dict(zip(days[:-1], days[1:]))
    prices_idx = prices.set_index(["date", "symbol"]).sort_index()
    price_days = set(prices_idx.index.get_level_values(0).unique())

    eligible_raw = liquidity_table(prices, int(cfg["liquidity_lookback_sessions"]), float(cfg["min_avg_turnover_cr"]))
    eligible_set = set()
    for item in eligible_raw:
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], tuple):
            (day, sym), flag = item
            if bool(flag):
                eligible_set.add((pd.Timestamp(day).normalize(), sym))
        elif isinstance(item, tuple) and len(item) == 2:
            day, sym = item
            eligible_set.add((pd.Timestamp(day).normalize(), sym))
    eligible_by_day: dict[pd.Timestamp, set[str]] = {}
    for day, sym in eligible_set:
        eligible_by_day.setdefault(pd.Timestamp(day), set()).add(sym)
    log(f"Universe ready: {len(eval_days)} OOS sessions; {len(eligible_set):,} eligible pairs", started)

    sizes = [int(x) for x in cfg["portfolio_sizes"]]
    states = {ps: {"capital": float(cfg["initial_capital"]), "trades": [], "daily": []} for ps in sizes}
    bundles = {}
    router = {}
    contrarian = {}
    last_period = None
    diagnostics = {
        "phase": "6.5",
        "signal_days": 0,
        "universe_rows": 0,
        "expert_prediction_rows": 0,
        "selected_rows": 0,
        "trade_days": 0,
        "no_trade_days": 0,
        "long_trades": 0,
        "short_trades": 0,
        "direct_trades": 0,
        "contrarian_trades": 0,
        "missing_entry_rows_skipped": 0,
        "expert_refits": [],
        "router_weights_by_regime": {},
        "contrarian_factors_by_regime": {},
        "design": "all experts predict LONG and SHORT in parallel; direction-aware regime router; learned contrarian alternatives; one symbol can resolve to LONG, SHORT or NO TRADE; CPR/Camarilla is an expert",
    }

    for i, signal_day in enumerate(eval_days, 1):
        signal_day = pd.Timestamp(signal_day)
        diagnostics["signal_days"] += 1
        period = signal_day.to_period("M")
        if period != last_period:
            refit_started = time.perf_counter()
            prior = feat[(feat.date < signal_day) & feat.label_date.notna() & (feat.label_date < signal_day)].copy()
            train_days = sorted(prior.date.unique())[-int(cfg["training_lookback_sessions"]):]
            prior = prior[prior.date.isin(train_days)].copy()
            if len(train_days) >= int(cfg["min_train_days"]):
                bundles = train_experts(prior, tuple(cfg["target_levels"]), float(cfg["stop_loss_pct"]), int(cfg["seed"]), int(cfg["min_positive_labels"]))
                router = train_router(prior, bundles, tuple(cfg["target_levels"])) if bundles else {}
                contrarian = train_contrarian_factors(prior, bundles) if bundles and cfg.get("contrarian_enabled", True) else {}
            else:
                bundles, router, contrarian = {}, {}, {}
            diagnostics["expert_refits"].append({"period": str(period), "train_days": len(train_days), "experts": sorted(bundles), "regimes": sorted(router), "contrarian": contrarian})
            diagnostics["router_weights_by_regime"].update({str(k): v for k, v in router.items()})
            diagnostics["contrarian_factors_by_regime"].update({str(k): v for k, v in contrarian.items()})
            log(f"REFIT {period}: train_days={len(train_days)} experts={len(bundles)} regimes={sorted(router)} duration={time.perf_counter()-refit_started:.1f}s", started)
            last_period = period

        entry_day = next_day.get(signal_day)
        symbols = eligible_by_day.get(signal_day, set())
        universe = feat[(feat.date == signal_day) & feat.symbol.isin(symbols)].copy()
        diagnostics["universe_rows"] += len(universe)
        selected = pd.DataFrame()
        if bundles and entry_day is not None and entry_day in price_days and not universe.empty:
            scored = score_experts(bundles, universe, tuple(cfg["target_levels"]))
            diagnostics["expert_prediction_rows"] += len(scored)
            selected, _ = select_top_predictions(scored, router, contrarian, int(cfg["max_per_expert"]), max(sizes))
            diagnostics["selected_rows"] += len(selected)

        # Remove candidates without executable OHLC on entry session; never crash and never synthesize data.
        if not selected.empty and entry_day in price_days:
            available = set(prices_idx.loc[entry_day].index.astype(str))
            selected["symbol"] = selected["symbol"].astype(str)
            before = len(selected)
            selected = selected[selected.symbol.isin(available)].copy()
            diagnostics["missing_entry_rows_skipped"] += before - len(selected)

        if selected.empty:
            diagnostics["no_trade_days"] += 1
            for ps in sizes:
                st = states[ps]
                st["daily"].append({"signal_date": signal_day, "entry_date": entry_day, "executed": 0, "starting_equity": st["capital"], "ending_equity": st["capital"], "daily_pnl": 0.0, "daily_return_pct": 0.0, "selected_count": 0})
        else:
            diagnostics["trade_days"] += 1
            for ps in sizes:
                st = states[ps]
                chosen = selected.head(min(ps, int(cfg["max_candidates_per_day"]))).copy()
                start_equity = st["capital"]
                allocation = start_equity / max(len(chosen), 1)
                pnl = 0.0
                for _, pick in chosen.iterrows():
                    row = prices_idx.loc[(entry_day, pick.symbol)]
                    direction = str(pick["direction"])
                    # 2.5% is the minimum target; increase target only when the directional
                    # composite is strongly supported by the selected candidate's score.
                    score = float(pick.get("final_score", 0.0))
                    target_levels = [float(x) for x in cfg["target_levels"]]
                    if score >= 0.85:
                        target = target_levels[min(3, len(target_levels) - 1)]
                    elif score >= 0.70:
                        target = target_levels[min(2, len(target_levels) - 1)]
                    elif score >= 0.55:
                        target = target_levels[min(1, len(target_levels) - 1)]
                    else:
                        target = target_levels[0]
                    result = execute(row, allocation, cfg["costs"], direction, target, float(cfg["stop_loss_pct"]))
                    result.update(pick.to_dict())
                    result.update({"signal_date": signal_day, "entry_date": entry_day, "portfolio_size": ps, "target": target, "direction": direction})
                    st["trades"].append(result)
                    pnl += result["net_pnl"]
                    if direction == "LONG": diagnostics["long_trades"] += 1
                    else: diagnostics["short_trades"] += 1
                    if str(pick.get("selection_source", "direct")) == "contrarian": diagnostics["contrarian_trades"] += 1
                    else: diagnostics["direct_trades"] += 1
                st["capital"] += pnl
                st["daily"].append({"signal_date": signal_day, "entry_date": entry_day, "executed": 1, "starting_equity": start_equity, "ending_equity": st["capital"], "daily_pnl": pnl, "daily_return_pct": pnl / max(start_equity, 1.0), "selected_count": len(chosen)})
            log(f"TRADE DAY {i}/{len(eval_days)} signal={signal_day.date()} entry={entry_day.date()} selected={len(selected)} dirs={','.join(selected.head(5).direction.astype(str))}", started)

        if i == 1 or i % 20 == 0 or i == len(eval_days):
            snapshot = ", ".join(f"P{ps}=Rs.{states[ps]['capital']:,.0f}" for ps in sizes)
            log(f"PROGRESS {i}/{len(eval_days)} trade_days={diagnostics['trade_days']} no_trade_days={diagnostics['no_trade_days']} long={diagnostics['long_trades']} short={diagnostics['short_trades']} universe={diagnostics['universe_rows']:,} | {snapshot}", started)

    summaries = {str(ps): summary(states[ps], cfg) for ps in sizes}
    result = {
        "phase": "6.5",
        "engine": "bidirectional_long_short_adaptive_moe_with_contrarian_layer",
        "config": cfg,
        "diagnostics": diagnostics,
        "summaries": summaries,
        "runtime_seconds": time.perf_counter() - started,
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2, default=str))
    for ps in sizes:
        pd.DataFrame(states[ps]["trades"]).to_csv(out / f"trades_p{ps}.csv", index=False)
        pd.DataFrame(states[ps]["daily"]).to_csv(out / f"daily_p{ps}.csv", index=False)
    log(f"=== PHASE 6.5 COMPLETE === runtime={time.perf_counter()-started:.1f}s trade_days={diagnostics['trade_days']} long={diagnostics['long_trades']} short={diagnostics['short_trades']}", started)
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
