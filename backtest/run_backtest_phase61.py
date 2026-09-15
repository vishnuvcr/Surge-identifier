from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.moe_engine import add_point_in_time_events
from src.moe_engine_phase61 import add_targets, route_and_rank, score_experts_phase61, train_experts_phase61, train_router
from src.nse_data import load_prices
from src.nse_fo import load_fo
from src.surge_model import build_features


def fees(buy: float, sell: float, c: dict) -> float:
    turnover = buy + sell
    exchange = turnover * c["exchange_turnover_rate"]
    brokerage = 2 * c["brokerage_per_order"]
    stt = sell * c["stt_sell_rate"]
    sebi = turnover * c["sebi_turnover_rate"]
    stamp = buy * c["stamp_buy_rate"]
    gst = c["gst_rate"] * (brokerage + exchange)
    return brokerage + exchange + stt + sebi + stamp + gst


def execute(row, allocation: float, costs: dict, target: float, stop_loss: float) -> dict:
    raw_open, raw_high, raw_low, raw_close = map(float, [row.open, row.high, row.low, row.close])
    entry = raw_open * (1 + costs["slippage_per_side"])
    variable_buy = costs["exchange_turnover_rate"] + costs["sebi_turnover_rate"] + costs["stamp_buy_rate"]
    qty = int(max((allocation - costs["brokerage_per_order"]) / (entry * (1 + variable_buy)), 0))
    if qty <= 0:
        return {"net_pnl": 0.0, "gross_pnl": 0.0, "fees": 0.0, "return_pct": 0.0, "exit_reason": "no_qty"}
    buy = qty * entry
    tp = entry * (1 + target)
    sl = entry * (1 - stop_loss)
    both = raw_low <= sl and raw_high >= tp
    if both:
        exit_price = sl * (1 - costs["slippage_per_side"])
        reason = "stop_and_target_same_bar_conservative_stop"
    elif raw_high >= tp:
        exit_price = tp * (1 - costs["slippage_per_side"])
        reason = "take_profit"
    elif raw_low <= sl:
        exit_price = sl * (1 - costs["slippage_per_side"])
        reason = "stop_loss"
    else:
        exit_price = raw_close * (1 - costs["slippage_per_side"])
        reason = "close_proxy"
    sell = qty * exit_price
    gross = sell - buy
    cost = fees(buy, sell, costs)
    net = gross - cost
    return {"qty": qty, "entry_price": entry, "exit_price": exit_price, "buy_value": buy, "sell_value": sell, "gross_pnl": gross, "net_pnl": net, "fees": cost, "return_pct": net / max(allocation, 1.0), "exit_reason": reason}


def load_events(cfg):
    news_path = ROOT / cfg["event_inputs"]["news_path"]
    corp_path = ROOT / cfg["event_inputs"]["corporate_actions_path"]
    return (pd.read_csv(news_path) if news_path.exists() else pd.DataFrame(), pd.read_csv(corp_path) if corp_path.exists() else pd.DataFrame())


def liquidity_table(prices, sessions, min_turnover_cr):
    p = prices[["date", "symbol", "turnover"]].copy()
    p["date"] = pd.to_datetime(p.date).dt.normalize()
    p["turnover_cr"] = p.turnover.astype(float) / 1e7
    wide = p.pivot(index="date", columns="symbol", values="turnover_cr").sort_index()
    roll = wide.rolling(sessions, min_periods=1).median().shift(1)
    mask = roll >= float(min_turnover_cr)
    return {(d, s) for d, s in mask.stack().items() if bool(s)}


def summary(st, cfg):
    trades = pd.DataFrame(st["trades"])
    daily = pd.DataFrame(st["daily"])
    if daily.empty:
        return {}
    daily["peak"] = daily.ending_equity.cummax()
    daily["dd"] = daily.ending_equity / daily.peak - 1
    wins = float((trades.net_pnl > 0).mean()) if not trades.empty else 0.0
    gross_w = float(trades.loc[trades.net_pnl > 0, "net_pnl"].sum()) if not trades.empty else 0.0
    gross_l = float(-trades.loc[trades.net_pnl < 0, "net_pnl"].sum()) if not trades.empty else 0.0
    pf = gross_w / max(gross_l, 1e-9)
    exec_days = daily[daily.executed == 1]
    monthly_rows = []
    if not trades.empty:
        eq = float(cfg["initial_capital"])
        for period, g in trades.groupby(pd.to_datetime(trades.entry_date).dt.to_period("M")):
            start = eq
            eq += float(g.net_pnl.sum())
            monthly_rows.append({"month": str(period), "starting_equity": start, "net_pnl": float(g.net_pnl.sum()), "return_pct": eq / start - 1})
    expert_rows = []
    if not trades.empty:
        for expert, g in trades.groupby("expert"):
            loss = -float(g.loc[g.net_pnl < 0, "net_pnl"].sum())
            expert_rows.append({"expert": expert, "trades": len(g), "net_pnl": float(g.net_pnl.sum()), "win_rate": float((g.net_pnl > 0).mean()), "profit_factor": float(g.loc[g.net_pnl > 0, "net_pnl"].sum() / max(loss, 1e-9))})
    regime_rows = []
    if not trades.empty:
        for reg, g in trades.groupby("regime_code"):
            regime_rows.append({"regime_code": int(reg), "trades": len(g), "net_pnl": float(g.net_pnl.sum()), "win_rate": float((g.net_pnl > 0).mean())})
    return {"final_equity": float(st["capital"]), "net_pnl": float(st["capital"] - cfg["initial_capital"]), "net_return_pct": float(st["capital"] / cfg["initial_capital"] - 1), "trade_count": int(len(trades)), "execution_days": int(len(exec_days)), "no_trade_days": int((daily.executed == 0).sum()), "win_rate": wins, "profit_factor": float(pf), "max_drawdown_pct": float(daily.dd.min()), "target_hit_rate": float((trades.exit_reason == "take_profit").mean()) if not trades.empty else 0.0, "stop_hit_rate": float(trades.exit_reason.isin(["stop_loss", "stop_and_target_same_bar_conservative_stop"]).mean()) if not trades.empty else 0.0, "average_trade_net_pct": float(trades.return_pct.mean()) if not trades.empty else 0.0, "daily_2_5pct_attainment_on_trade_days": float((exec_days.daily_return_pct >= cfg["objective_daily_net_pct"]).mean()) if not exec_days.empty else 0.0, "monthly_30pct_target_hit": bool(any(x["return_pct"] >= cfg["objective_monthly_pct"] for x in monthly_rows)), "monthly_results": monthly_rows, "expert_results": expert_rows, "regime_results": regime_rows}


def main():
    started = time.perf_counter()
    cfg = yaml.safe_load((ROOT / "backtest/config_phase61.yaml").read_text())
    out = ROOT / "backtest/results/phase61"
    out.mkdir(parents=True, exist_ok=True)
    prices = load_prices("data/cache", cfg["lookback_days"])
    prices.date = pd.to_datetime(prices.date).dt.normalize()
    fo = load_fo("data/cache", cfg["lookback_days"])
    if not fo.empty:
        fo.date = pd.to_datetime(fo.date).dt.normalize()
        data = prices.merge(fo, on=["date", "symbol"], how="left")
    else:
        data = prices
    print(f"Loaded prices={len(prices):,} rows, f&o={len(fo):,} rows", flush=True)
    feat = build_features(data)
    news, corp = load_events(cfg)
    feat = add_point_in_time_events(feat, news, corp)
    feat = add_targets(feat, float(cfg["stop_loss_pct"]), tuple(cfg["target_levels"]))
    feat.date = pd.to_datetime(feat.date).dt.normalize()
    days = pd.DatetimeIndex(sorted(feat.date.unique()))
    eval_days = days[cfg["min_train_days"]:]
    if cfg.get("backtest_start"): eval_days = eval_days[eval_days >= pd.Timestamp(cfg["backtest_start"])]
    if cfg.get("backtest_end"): eval_days = eval_days[eval_days <= pd.Timestamp(cfg["backtest_end"])]
    next_day = dict(zip(days[:-1], days[1:]))
    prices_idx = prices.set_index(["date", "symbol"]).sort_index()
    eligible_set = liquidity_table(prices, int(cfg["liquidity_lookback_sessions"]), float(cfg["min_avg_turnover_cr"]))
    sizes = [int(x) for x in cfg["portfolio_sizes"]]
    states = {ps: {"capital": float(cfg["initial_capital"]), "trades": [], "daily": []} for ps in sizes}
    last_period = None
    bundles = {}
    router = None
    diagnostics = {"signal_days": 0, "candidate_rows": 0, "qualified_rows": 0, "trade_days": 0, "no_trade_days": 0, "expert_refits": []}
    for i, signal_day in enumerate(eval_days, 1):
        diagnostics["signal_days"] += 1
        period = pd.Timestamp(signal_day).to_period("M")
        if period != last_period:
            prior = feat[(feat.date < signal_day) & feat.label_date.notna() & (feat.label_date < signal_day)].copy()
            train_days = sorted(prior.date.unique())[-int(cfg["training_lookback_sessions"]):]
            prior = prior[prior.date.isin(train_days)].copy()
            if len(train_days) >= int(cfg["min_train_days"]):
                bundles = train_experts_phase61(prior, tuple(cfg["target_levels"]), float(cfg["stop_loss_pct"]), int(cfg["seed"]), int(cfg["min_positive_labels"]))
                router = train_router(prior, tuple(cfg["target_levels"]), float(cfg["stop_loss_pct"]), int(cfg["seed"])) if bundles else None
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
                scored = score_experts_phase61(bundles, universe, tuple(cfg["target_levels"]))
                picks = route_and_rank(scored, router, cfg)
                diagnostics["qualified_rows"] += len(picks)
                entry_day = next_day.get(signal_day)
                if entry_day is not None:
                    entry_symbols = prices_idx.loc[entry_day].index if entry_day in prices_idx.index.get_level_values(0) else pd.Index([])
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
            start = st["capital"]
            allocation = start / max(len(chosen), 1)
            pnl = 0.0
            for _, pick in chosen.iterrows():
                row = prices_idx.loc[(entry_day, pick.symbol)]
                result = execute(row, allocation, cfg["costs"], float(pick.target), float(cfg["stop_loss_pct"]))
                result.update(pick.to_dict())
                result.update({"signal_date": signal_day, "entry_date": entry_day, "portfolio_size": ps})
                st["trades"].append(result)
                pnl += result["net_pnl"]
            st["capital"] += pnl
            st["daily"].append({"signal_date": signal_day, "entry_date": entry_day, "executed": 1, "starting_equity": start, "ending_equity": st["capital"], "daily_pnl": pnl, "daily_return_pct": pnl / max(start, 1.0), "selected_count": len(chosen)})
        if i == 1 or i % 20 == 0 or i == len(eval_days):
            print(f"Progress {i}/{len(eval_days)} elapsed={time.perf_counter()-started:.1f}s", flush=True)
    summaries = {str(ps): summary(states[ps], cfg) for ps in sizes}
    result = {"phase": "6.1", "engine": "full_information_tail_moe_oof_router", "config": cfg, "diagnostics": diagnostics, "summaries": summaries, "runtime_seconds": time.perf_counter() - started}
    (out / "summary.json").write_text(json.dumps(result, indent=2, default=str))
    for ps in sizes:
        trades = pd.DataFrame(states[ps]["trades"])
        daily = pd.DataFrame(states[ps]["daily"])
        if not trades.empty: trades.to_csv(out / f"trades_p{ps}.csv", index=False)
        daily.to_csv(out / f"daily_p{ps}.csv", index=False)
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
