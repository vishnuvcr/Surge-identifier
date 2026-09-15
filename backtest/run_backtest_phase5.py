from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mfe_model import score_mfe_calibrated, train_mfe_walk_forward
from src.nse_data import load_prices
from src.nse_fo import load_fo
from src.surge_model import build_features, score_latest, train_walk_forward


def fees(buy, sell, c):
    turnover = buy + sell
    exchange = turnover * c["exchange_turnover_rate"]
    brokerage = 2 * c["brokerage_per_order"]
    stt = sell * c["stt_sell_rate"]
    sebi = turnover * c["sebi_turnover_rate"]
    stamp = buy * c["stamp_buy_rate"]
    gst = c["gst_rate"] * (brokerage + exchange)
    return brokerage + exchange + stt + sebi + stamp + gst


def execute(row, allocation, c, target, stop_loss):
    raw_open, raw_high, raw_low, raw_close = map(float, [row.open, row.high, row.low, row.close])
    entry = raw_open * (1 + c["slippage_per_side"])
    variable_buy = c["exchange_turnover_rate"] + c["sebi_turnover_rate"] + c["stamp_buy_rate"]
    q = int(max((allocation - c["brokerage_per_order"]) / (entry * (1 + variable_buy)), 0))
    if q <= 0:
        return {"net_pnl": 0.0, "gross_pnl": 0.0, "fees": 0.0, "return_pct": 0.0, "exit_reason": "no_qty"}

    buy = q * entry
    tp = entry * (1 + target)
    sl = entry * (1 - stop_loss)

    # Conservative daily-bar path assumption: if both TP and SL are touched,
    # assume the stop was reached first. This avoids intraday path look-ahead.
    if raw_low <= sl and raw_high >= tp:
        exit_price = sl * (1 - c["slippage_per_side"])
        reason = "stop_and_target_same_bar_conservative_stop"
    elif raw_high >= tp:
        exit_price = tp * (1 - c["slippage_per_side"])
        reason = "take_profit"
    elif raw_low <= sl:
        exit_price = sl * (1 - c["slippage_per_side"])
        reason = "stop_loss"
    else:
        exit_price = raw_close * (1 - c["slippage_per_side"])
        reason = "close_proxy"

    sell = q * exit_price
    gross = sell - buy
    cost = fees(buy, sell, c)
    net = gross - cost
    return {
        "qty": q,
        "entry_price": entry,
        "exit_price": exit_price,
        "buy_value": buy,
        "sell_value": sell,
        "gross_pnl": gross,
        "net_pnl": net,
        "fees": cost,
        "return_pct": net / max(allocation, 1),
        "exit_reason": reason,
    }


def regime_score(scored):
    breadth = float(scored.market_breadth.median())
    ret5 = float(scored.market_ret5.median())
    vol = float(scored.volatility20.median())
    breadth_component = np.clip((breadth - 0.35) / 0.30, 0, 1)
    momentum_component = np.clip((ret5 + 0.03) / 0.08, 0, 1)
    vol_component = 1 - np.clip((vol - 0.01) / 0.05, 0, 1)
    score = 0.45 * breadth_component + 0.40 * momentum_component + 0.15 * vol_component
    if score >= 0.65:
        label = "risk_on"
    elif score >= 0.40:
        label = "neutral"
    else:
        label = "risk_off"
    return float(score), label


def make_candidates(scored, cfg, stop_loss):
    rscore, rlabel = regime_score(scored)
    x = scored.copy()
    risk_multiplier = cfg["risk_off_probability_multiplier"] if rlabel == "risk_off" else 1.0
    pcol = "p_hit_2_5"

    candidates = []
    for _, row in x.iterrows():
        p = float(row[pcol])
        mfe = float(row.predicted_mfe)
        p_gate = cfg["min_candidate_probability"] * risk_multiplier
        if p < p_gate:
            continue
        # Require the predicted intraday MFE to reach the actual target.
        if mfe < cfg["target_pct"]:
            continue

        # Binary target/stop expected value with explicit transaction-cost hurdle.
        expected_net = p * cfg["target_pct"] - (1.0 - p) * stop_loss - cfg["estimated_cost_hurdle_pct"]
        if expected_net <= 0:
            continue

        score = expected_net + 0.10 * p + 0.10 * mfe + 0.05 * rscore
        candidates.append({
            "score": float(score),
            "expected_net": float(expected_net),
            "symbol": row.symbol,
            "surge_probability": float(row.surge_probability),
            "predicted_mfe": mfe,
            "p_hit_2_5": p,
            "p_hit_3": float(row.p_hit_3),
            "p_hit_4": float(row.p_hit_4),
            "p_hit_5": float(row.p_hit_5),
            "regime": rlabel,
            "regime_score": rscore,
            "validation_precision": float(row.validation_precision),
            "mfe_validation_mae": float(row.mfe_validation_mae),
            "mfe_validation_rank_ic": float(row.mfe_validation_rank_ic),
        })

    if not candidates:
        return []
    return sorted(candidates, key=lambda z: (z["score"], z["p_hit_2_5"], z["predicted_mfe"]), reverse=True)


def fit_models_for_day(feat, day, cfg):
    prior = feat[feat.date < day].copy()
    prior["label_date"] = prior.groupby("symbol").date.shift(-1)
    prior = prior[prior.label_date.notna() & (prior.label_date < day)].drop(columns=["label_date"])
    days = sorted(pd.to_datetime(prior.date).unique())[-cfg["training_lookback_sessions"]:]
    prior = prior[prior.date.isin(days)]
    if prior.empty:
        return None
    return (
        train_walk_forward(prior, validation_days=cfg["validation_days"], seed=cfg["seed"]),
        train_mfe_walk_forward(prior, validation_days=cfg["validation_days"], seed=cfg["seed"]),
    )


def metric_value(bundle, name):
    value = getattr(bundle, name, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def main():
    cfg = yaml.safe_load((ROOT / "backtest/config_phase5.yaml").read_text())
    out = ROOT / "backtest/results/phase5"
    out.mkdir(parents=True, exist_ok=True)

    prices = load_prices("data/cache", cfg["lookback_days"])
    prices.date = pd.to_datetime(prices.date).dt.normalize()
    fo = load_fo("data/cache", cfg["lookback_days"])
    data = prices
    if not fo.empty:
        fo.date = pd.to_datetime(fo.date).dt.normalize()
        data = prices.merge(fo, on=["date", "symbol"], how="left")
    feat = build_features(data)
    feat.date = pd.to_datetime(feat.date).dt.normalize()

    days = sorted(pd.to_datetime(feat.date).dt.normalize().unique())
    eval_days = days[cfg["min_train_days"] + cfg["validation_days"]:]
    if cfg.get("backtest_start"):
        eval_days = [d for d in eval_days if d >= pd.Timestamp(cfg["backtest_start"])]
    if cfg.get("backtest_end"):
        eval_days = [d for d in eval_days if d <= pd.Timestamp(cfg["backtest_end"])]

    next_day = {days[i]: days[i + 1] for i in range(len(days) - 1)}
    prices_idx = prices.set_index(["date", "symbol"]).sort_index()
    variants = [(float(sl), int(ps)) for sl in cfg["stop_loss_pcts"] for ps in cfg["portfolio_sizes"]]
    state = {
        (sl, ps): {"capital": float(cfg["initial_capital"]), "trades": [], "daily": []}
        for sl, ps in variants
    }
    last_period, bundle = None, None

    for signal_day in eval_days:
        period = pd.Timestamp(signal_day).to_period("M")
        if period != last_period:
            bundle = fit_models_for_day(feat, pd.Timestamp(signal_day), cfg)
            last_period = period
        if bundle is None:
            continue

        surge_bundle, mfe_bundle = bundle
        d = feat[feat.date == signal_day].copy()
        if d.empty:
            continue

        hist = prices[prices.date < signal_day]
        hdays = sorted(hist.date.unique())[-cfg["liquidity_lookback_sessions"]:]
        med_turnover = hist[hist.date.isin(hdays)].groupby("symbol").turnover.median() / 1e7
        eligible = med_turnover[med_turnover >= cfg["min_avg_turnover_cr"]].index
        universe = d[d.symbol.isin(eligible)].copy()
        if universe.empty:
            continue

        scored = score_latest(surge_bundle, universe)
        mfe = score_mfe_calibrated(mfe_bundle, universe)
        keep = ["symbol", "predicted_mfe", "p_hit_2_5", "p_hit_3", "p_hit_4", "p_hit_5"]
        scored = scored.merge(mfe[keep], on="symbol", how="inner")
        scored = scored[scored.risk_flag != "low_liquidity"].copy()
        if scored.empty:
            continue
        scored["threshold_used"] = max(float(surge_bundle.threshold), cfg["min_candidate_probability"])
        scored["validation_precision"] = metric_value(surge_bundle, "validation_precision")
        scored["mfe_validation_mae"] = metric_value(mfe_bundle, "validation_mae")
        scored["mfe_validation_rank_ic"] = metric_value(mfe_bundle, "validation_rank_ic")

        entry_day = next_day.get(signal_day)
        if entry_day is None:
            continue

        for stop_loss, portfolio_size in variants:
            st = state[(stop_loss, portfolio_size)]
            start_capital = st["capital"]
            candidates = make_candidates(scored, cfg, stop_loss)
            picks = []
            for candidate in candidates[:cfg["max_candidates_per_day"]]:
                if (entry_day, candidate["symbol"]) not in prices_idx.index:
                    continue
                if candidate["symbol"] in {p["symbol"] for p in picks}:
                    continue
                picks.append(candidate)
                if len(picks) >= portfolio_size:
                    break

            if not picks:
                rlabel = regime_score(scored)[1]
                st["daily"].append({
                    "signal_date": signal_day, "executed": 0,
                    "starting_equity": start_capital, "ending_equity": start_capital,
                    "daily_pnl": 0.0, "daily_return_pct": 0.0,
                    "portfolio_size": portfolio_size, "stop_loss_pct": stop_loss,
                    "regime": rlabel,
                })
                continue

            allocation = start_capital / len(picks)
            total_pnl = 0.0
            for candidate in picks:
                result = execute(prices_idx.loc[(entry_day, candidate["symbol"])], allocation, cfg["costs"], cfg["target_pct"], stop_loss)
                total_pnl += result["net_pnl"]
                result.update(candidate)
                result.update({
                    "signal_date": signal_day, "entry_date": entry_day,
                    "portfolio_size": len(picks), "configured_portfolio_size": portfolio_size,
                    "stop_loss_pct": stop_loss, "target_pct": cfg["target_pct"],
                })
                st["trades"].append(result)

            st["capital"] += total_pnl
            st["daily"].append({
                "signal_date": signal_day, "entry_date": entry_day, "executed": 1,
                "starting_equity": start_capital, "ending_equity": st["capital"],
                "daily_pnl": total_pnl, "daily_return_pct": total_pnl / max(start_capital, 1.0),
                "portfolio_size": len(picks), "stop_loss_pct": stop_loss,
                "regime": picks[0]["regime"],
            })

    summaries = []
    for stop_loss, portfolio_size in variants:
        st = state[(stop_loss, portfolio_size)]
        trades = pd.DataFrame(st["trades"])
        daily = pd.DataFrame(st["daily"])
        if daily.empty:
            continue
        daily["peak_equity"] = daily.ending_equity.cummax()
        daily["drawdown_pct"] = daily.ending_equity / daily.peak_equity - 1

        if not trades.empty:
            wins = trades[trades.net_pnl > 0].net_pnl.sum()
            losses = -trades[trades.net_pnl < 0].net_pnl.sum()
            pf = float(wins / max(losses, 1e-9))
            target_hits = float((trades.exit_reason == "take_profit").mean())
            stop_hits = float(trades.exit_reason.isin(["stop_loss", "stop_and_target_same_bar_conservative_stop"]).mean())
            avg_trade = float(trades.net_pnl.mean())
            avg_win = float(trades.loc[trades.net_pnl > 0, "net_pnl"].mean()) if (trades.net_pnl > 0).any() else 0.0
            avg_loss = float(trades.loc[trades.net_pnl < 0, "net_pnl"].mean()) if (trades.net_pnl < 0).any() else 0.0
            monthly = trades.groupby(pd.to_datetime(trades.entry_date).dt.to_period("M")).net_pnl.sum()
        else:
            pf = target_hits = stop_hits = avg_trade = avg_win = avg_loss = 0.0
            monthly = pd.Series(dtype=float)

        meq = float(cfg["initial_capital"])
        monthly_rows = []
        for period, pnl in monthly.items():
            start_eq = meq
            meq += float(pnl)
            monthly_rows.append({"month": str(period), "starting_equity": start_eq,
                                 "net_pnl": float(pnl), "return_pct": meq / start_eq - 1})

        execution_days = daily[daily.executed == 1]
        daily_target_attainment = float((execution_days.daily_return_pct >= cfg["objective_daily_net_pct"]).mean()) if not execution_days.empty else 0.0
        summaries.append({
            "stop_loss_pct": stop_loss,
            "portfolio_size": portfolio_size,
            "final_equity": float(st["capital"]),
            "net_pnl": float(st["capital"] - cfg["initial_capital"]),
            "net_return_pct": float(st["capital"] / cfg["initial_capital"] - 1),
            "trade_count": int(len(trades)),
            "execution_days": int(len(execution_days)),
            "no_trade_days": int((daily.executed == 0).sum()),
            "win_rate": float((trades.net_pnl > 0).mean()) if not trades.empty else 0.0,
            "profit_factor": pf,
            "max_drawdown_pct": float(daily.drawdown_pct.min()),
            "target_hit_rate": target_hits,
            "stop_hit_rate": stop_hits,
            "average_trade_net_pct": float(trades.return_pct.mean()) if not trades.empty else 0.0,
            "average_win_pnl": avg_win,
            "average_loss_pnl": avg_loss,
            "daily_2_5pct_attainment_on_trade_days": daily_target_attainment,
            "monthly_30pct_target_hit": bool(any(m["return_pct"] >= cfg["objective_monthly_pct"] for m in monthly_rows)),
            "monthly_results": monthly_rows,
        })

        trades.to_csv(out / f"trades_sl{stop_loss:.3f}_p{portfolio_size}.csv", index=False)
        daily.to_csv(out / f"daily_sl{stop_loss:.3f}_p{portfolio_size}.csv", index=False)

    summary_df = pd.DataFrame([{k: v for k, v in s.items() if k != "monthly_results"} for s in summaries])
    summary_df = summary_df.sort_values(["net_return_pct", "max_drawdown_pct"], ascending=[False, False])
    summary_df.to_csv(out / "summary.csv", index=False)
    best = summaries[int(summary_df.index[0])] if not summary_df.empty else None
    (out / "summary.json").write_text(json.dumps({
        "objective": {
            "daily_net_target_pct": cfg["objective_daily_net_pct"],
            "monthly_target_pct": cfg["objective_monthly_pct"],
        },
        "calibration": "Isotonic calibration is fit only on each model's trailing 60-session validation block and then applied to later signal days.",
        "anti_lookahead": [
            "All labels used for model fitting have label dates strictly before each signal day.",
            "Liquidity is computed only from sessions strictly before the signal day.",
            "Signals are generated at EOD and entered at the next session open.",
            "Portfolio allocation is based only on starting equity for the trading day.",
            "If TP and SL are both touched in a daily bar, the conservative simulator assumes SL first.",
        ],
        "variants": summaries,
        "best_variant": best,
        "costs": cfg["costs"],
    }, indent=2, default=str))
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
