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

from src.moe_engine import add_forward_targets, add_point_in_time_events, score_experts, select_trades, train_experts
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


def load_optional_events(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    news_path = ROOT / cfg["event_inputs"]["news_path"]
    corp_path = ROOT / cfg["event_inputs"]["corporate_actions_path"]
    news = pd.read_csv(news_path) if news_path.exists() else pd.DataFrame()
    corp = pd.read_csv(corp_path) if corp_path.exists() else pd.DataFrame()
    return news, corp


def build_liquidity_table(prices: pd.DataFrame, lookback_sessions: int, min_turnover_cr: float) -> pd.DataFrame:
    """Build point-in-time trailing median turnover once, instead of per signal day."""
    p = prices[["date", "symbol", "turnover"]].copy()
    p["date"] = pd.to_datetime(p["date"]).dt.normalize()
    p["turnover_cr"] = p["turnover"].astype(float) / 1e7
    daily = p.pivot(index="date", columns="symbol", values="turnover_cr").sort_index()
    rolling = daily.rolling(lookback_sessions, min_periods=1).median().shift(1)
    mask = rolling >= float(min_turnover_cr)
    long = mask.stack(future_stack=True).rename("eligible").reset_index()
    long = long[long["eligible"]].drop(columns="eligible")
    return long


def main() -> None:
    started = time.perf_counter()
    cfg = yaml.safe_load((ROOT / "backtest/config_phase6.yaml").read_text())
    out = ROOT / "backtest/results/phase6"
    out.mkdir(parents=True, exist_ok=True)

    prices = load_prices("data/cache", cfg["lookback_days"])
    prices.date = pd.to_datetime(prices.date).dt.normalize()
    fo = load_fo("data/cache", cfg["lookback_days"])
    data = prices
    if not fo.empty:
        fo.date = pd.to_datetime(fo.date).dt.normalize()
        data = prices.merge(fo, on=["date", "symbol"], how="left")

    print(f"Loaded prices={len(prices):,} rows, f&o={len(fo):,} rows", flush=True)
    t0 = time.perf_counter()
    feat = build_features(data)
    print(f"Feature build: {time.perf_counter() - t0:.1f}s", flush=True)
    feat.date = pd.to_datetime(feat.date).dt.normalize()

    news, corp = load_optional_events(cfg)
    feat = add_point_in_time_events(feat, news, corp)
    feat = add_forward_targets(feat, tuple(cfg["target_levels"]), float(cfg["stop_loss_pct"]))

    days = pd.DatetimeIndex(sorted(pd.to_datetime(feat.date).unique()))
    eval_days = days[cfg["min_train_days"]:]
    if cfg.get("backtest_start"):
        eval_days = eval_days[eval_days >= pd.Timestamp(cfg["backtest_start"])]
    if cfg.get("backtest_end"):
        eval_days = eval_days[eval_days <= pd.Timestamp(cfg["backtest_end"])]

    next_day = dict(zip(days[:-1], days[1:]))
    prices_idx = prices.set_index(["date", "symbol"]).sort_index()
    portfolio_sizes = [int(x) for x in cfg["portfolio_sizes"]]
    states = {ps: {"capital": float(cfg["initial_capital"]), "trades": [], "daily": []} for ps in portfolio_sizes}

    print(f"Evaluation sessions: {len(eval_days)}", flush=True)
    t0 = time.perf_counter()
    liquidity = build_liquidity_table(prices, int(cfg["liquidity_lookback_sessions"]), float(cfg["min_avg_turnover_cr"]))
    liquidity_dates = {d: set(g["symbol"]) for d, g in liquidity.groupby("date")}
    print(f"Liquidity table: {time.perf_counter() - t0:.1f}s", flush=True)

    last_period = None
    bundles = {}
    monthly_refits = 0

    for idx, signal_day in enumerate(eval_days, 1):
        period = pd.Timestamp(signal_day).to_period("M")
        if period != last_period:
            refit_started = time.perf_counter()
            prior = feat[feat.date < signal_day].copy()
            prior["label_date"] = prior.groupby("symbol").date.shift(-1)
            prior = prior[prior.label_date.notna() & (prior.label_date < signal_day)].drop(columns=["label_date"])
            train_days = sorted(pd.to_datetime(prior.date).unique())[-cfg["training_lookback_sessions"]:]
            prior = prior[prior.date.isin(train_days)].copy()
            if len(train_days) < cfg["min_train_days"]:
                bundles = {}
            else:
                bundles = train_experts(
                    prior,
                    target_levels=tuple(cfg["target_levels"]),
                    stop_loss=float(cfg["stop_loss_pct"]),
                    seed=int(cfg["seed"]),
                )
            monthly_refits += 1
            print(
                f"Refit {monthly_refits}: period={period} train_days={len(train_days)} "
                f"experts={len(bundles)} elapsed={time.perf_counter() - refit_started:.1f}s",
                flush=True,
            )
            last_period = period

        if not bundles:
            continue

        d = feat[feat.date == signal_day].copy()
        if d.empty:
            continue

        eligible_symbols = liquidity_dates.get(signal_day, set())
        if not eligible_symbols:
            continue
        universe = d[d.symbol.isin(eligible_symbols)].copy()
        if universe.empty:
            continue

        scored = score_experts(bundles, universe, target_levels=tuple(cfg["target_levels"]))
        if scored.empty:
            continue
        scored = scored[scored.expert_rule_score >= 0.70].copy()
        if scored.empty:
            for ps in portfolio_sizes:
                st = states[ps]
                st["daily"].append({
                    "signal_date": signal_day, "entry_date": next_day.get(signal_day), "executed": 0,
                    "starting_equity": st["capital"], "ending_equity": st["capital"], "daily_pnl": 0.0,
                    "daily_return_pct": 0.0, "portfolio_size": ps, "selected_count": 0,
                })
            continue
        pred_cols = [f"pred_{str(t).replace('.', 'p').lstrip('0')}" for t in cfg["target_levels"]]
        scored = scored[scored[pred_cols].max(axis=1) >= cfg["min_expected_return_pct"]].copy()
        picks = select_trades(scored, cfg)
        if picks.empty:
            for ps in portfolio_sizes:
                st = states[ps]
                st["daily"].append({
                    "signal_date": signal_day, "entry_date": next_day.get(signal_day), "executed": 0,
                    "starting_equity": st["capital"], "ending_equity": st["capital"], "daily_pnl": 0.0,
                    "daily_return_pct": 0.0, "portfolio_size": ps, "selected_count": 0,
                })
            continue

        entry_day = next_day.get(signal_day)
        if entry_day is None:
            continue
        entry_symbols = prices_idx.loc[entry_day].index if entry_day in prices_idx.index.get_level_values(0) else pd.Index([])
        picks = picks[picks.symbol.isin(entry_symbols)].copy()
        if picks.empty:
            continue

        for ps in portfolio_sizes:
            st = states[ps]
            chosen = picks.head(min(ps, cfg["max_candidates_per_day"])).copy()
            start_capital = st["capital"]
            allocation = start_capital / max(len(chosen), 1)
            total_pnl = 0.0
            for _, pick in chosen.iterrows():
                row = prices_idx.loc[(entry_day, pick.symbol)]
                result = execute(row, allocation, cfg["costs"], float(pick.target), float(cfg["stop_loss_pct"]))
                result.update(pick.to_dict())
                result.update({
                    "signal_date": signal_day,
                    "entry_date": entry_day,
                    "configured_portfolio_size": ps,
                    "actual_portfolio_size": len(chosen),
                    "stop_loss_pct": cfg["stop_loss_pct"],
                })
                st["trades"].append(result)
                total_pnl += result["net_pnl"]

            st["capital"] += total_pnl
            st["daily"].append({
                "signal_date": signal_day, "entry_date": entry_day, "executed": 1,
                "starting_equity": start_capital, "ending_equity": st["capital"],
                "daily_pnl": total_pnl, "daily_return_pct": total_pnl / max(start_capital, 1.0),
                "portfolio_size": ps, "selected_count": len(chosen),
            })

        if idx == 1 or idx % 20 == 0 or idx == len(eval_days):
            elapsed = time.perf_counter() - started
            print(f"Progress {idx}/{len(eval_days)} signal_day={pd.Timestamp(signal_day).date()} total_elapsed={elapsed:.1f}s", flush=True)

    summaries = []
    expert_rows = []
    regime_rows = []
    for ps in portfolio_sizes:
        st = states[ps]
        trades = pd.DataFrame(st["trades"])
        daily = pd.DataFrame(st["daily"])
        if daily.empty:
            continue
        daily["peak_equity"] = daily.ending_equity.cummax()
        daily["drawdown_pct"] = daily.ending_equity / daily.peak_equity - 1
        if trades.empty:
            win_rate = pf = target_hit = stop_hit = avg_trade = 0.0
            expert_rows_local = []
        else:
            gross_wins = trades.loc[trades.net_pnl > 0, "net_pnl"].sum()
            gross_losses = -trades.loc[trades.net_pnl < 0, "net_pnl"].sum()
            pf = float(gross_wins / max(gross_losses, 1e-9))
            win_rate = float((trades.net_pnl > 0).mean())
            target_hit = float((trades.exit_reason == "take_profit").mean())
            stop_hit = float(trades.exit_reason.isin(["stop_loss", "stop_and_target_same_bar_conservative_stop"]).mean())
            avg_trade = float(trades.return_pct.mean())
            expert_rows_local = []
            for expert, g in trades.groupby("expert"):
                expert_rows_local.append({
                    "portfolio_size": ps, "expert": expert, "trades": len(g),
                    "net_pnl": float(g.net_pnl.sum()), "win_rate": float((g.net_pnl > 0).mean()),
                    "profit_factor": float(g.loc[g.net_pnl > 0, "net_pnl"].sum() / max(-g.loc[g.net_pnl < 0, "net_pnl"].sum(), 1e-9)),
                    "avg_return_pct": float(g.return_pct.mean()),
                })
        expert_rows.extend(expert_rows_local)
        if not trades.empty:
            for regime, g in trades.groupby("regime_code"):
                regime_rows.append({"portfolio_size": ps, "regime_code": int(regime), "trades": len(g), "net_pnl": float(g.net_pnl.sum()), "win_rate": float((g.net_pnl > 0).mean())})

        monthly = trades.groupby(pd.to_datetime(trades.entry_date).dt.to_period("M")).net_pnl.sum() if not trades.empty else pd.Series(dtype=float)
        eq = float(cfg["initial_capital"])
        monthly_rows = []
        for period, pnl in monthly.items():
            start = eq
            eq += float(pnl)
            monthly_rows.append({"month": str(period), "starting_equity": start, "net_pnl": float(pnl), "return_pct": eq / start - 1})

        execution_days = daily[daily.executed == 1]
        summaries.append({
            "portfolio_size": ps,
            "final_equity": float(st["capital"]),
            "net_pnl": float(st["capital"] - cfg["initial_capital"]),
            "net_return_pct": float(st["capital"] / cfg["initial_capital"] - 1),
            "trade_count": int(len(trades)),
            "execution_days": int(len(execution_days)),
            "no_trade_days": int((daily.executed == 0).sum()),
            "win_rate": win_rate,
            "profit_factor": pf,
            "max_drawdown_pct": float(daily.drawdown_pct.min()),
            "target_hit_rate": target_hit,
            "stop_hit_rate": stop_hit,
            "average_trade_net_pct": avg_trade,
            "daily_2_5pct_attainment_on_trade_days": float((execution_days.daily_return_pct >= cfg["objective_daily_net_pct"]).mean()) if not execution_days.empty else 0.0,
            "monthly_30pct_target_hit": bool(any(x["return_pct"] >= cfg["objective_monthly_pct"] for x in monthly_rows)),
            "monthly_results": monthly_rows,
        })

        daily.to_csv(out / f"daily_p{ps}.csv", index=False)
        trades.to_csv(out / f"trades_p{ps}.csv", index=False)

    pd.DataFrame(summaries).to_json(out / "summary.json", orient="records", indent=2)
    pd.DataFrame(summaries).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(expert_rows).to_csv(out / "expert_performance.csv", index=False)
    pd.DataFrame(regime_rows).to_csv(out / "regime_performance.csv", index=False)

    metadata = {
        "architecture": "regime-aware mixture-of-experts with expert-specific multi-target RandomForest return models",
        "experts": list(bundles.keys()),
        "event_inputs_present": {"news": not news.empty, "corporate_actions": not corp.empty},
        "monthly_refits": monthly_refits,
        "runtime_seconds": time.perf_counter() - started,
        "anti_lookahead": [
            "Training rows use labels whose next-session label date is strictly before the signal day.",
            "All strategy models are refit monthly using only prior sessions.",
            "Liquidity is precomputed with a trailing window shifted one session, so each signal uses only prior liquidity history.",
            "Signals are produced EOD and executed at the next session open.",
            "Daily-bar TP/SL conflicts are resolved conservatively as stop-first.",
            "Event inputs are optional and must carry timestamps; only information mapped to the signal date is used.",
        ],
        "optimization": [
            "One multi-output RandomForest per expert instead of one forest per expert/target.",
            "Point-in-time liquidity eligibility precomputed once for the backtest.",
            "Vectorized candidate/target selection.",
        ],
        "objective": {"daily_net_pct": cfg["objective_daily_net_pct"], "monthly_pct": cfg["objective_monthly_pct"]},
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    print(json.dumps({"summaries": summaries, "experts": list(bundles.keys()), "runtime_seconds": time.perf_counter() - started}, indent=2, default=str))


if __name__ == "__main__":
    main()
