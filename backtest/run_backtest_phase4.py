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

from src.mfe_model import score_mfe, train_mfe_walk_forward
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


def execute(row, allocation, c, target):
    raw_open, raw_high, raw_close = map(float, [row.open, row.high, row.close])
    entry = raw_open * (1 + c["slippage_per_side"])
    variable_buy = c["exchange_turnover_rate"] + c["sebi_turnover_rate"] + c["stamp_buy_rate"]
    q = int(max((allocation - c["brokerage_per_order"]) / (entry * (1 + variable_buy)), 0))
    if q <= 0:
        return {"net_pnl": 0.0, "gross_pnl": 0.0, "fees": 0.0, "return_pct": 0.0, "exit_reason": "no_qty"}
    buy = q * entry
    tp = entry * (1 + target)
    if raw_high >= tp:
        exit_price = tp * (1 - c["slippage_per_side"])
        reason = "take_profit"
    else:
        exit_price = raw_close * (1 - c["slippage_per_side"])
        reason = "close_proxy"
    sell = q * exit_price
    gross = sell - buy
    cost = fees(buy, sell, c)
    net = gross - cost
    return {"qty": q, "entry_price": entry, "exit_price": exit_price, "buy_value": buy,
            "sell_value": sell, "gross_pnl": gross, "net_pnl": net, "fees": cost,
            "return_pct": net / max(allocation, 1), "exit_reason": reason}


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


def choose_signal(scored, cfg):
    rscore, rlabel = regime_score(scored)
    x = scored.copy()
    rules = [
        (0.05, "p_hit_5", cfg["p_hit_5"], cfg["mfe_min_5"], 0.72),
        (0.04, "p_hit_4", cfg["p_hit_4"], cfg["mfe_min_4"], 0.64),
        (0.03, "p_hit_3", cfg["p_hit_3"], cfg["mfe_min_3"], 0.56),
    ]
    risk_multiplier = cfg["risk_off_probability_multiplier"] if rlabel == "risk_off" else 1.0
    x["mfe_score"] = x["predicted_mfe"] * (0.60 + 0.40 * x["surge_probability"])
    x["regime_score"] = rscore
    x["regime"] = rlabel
    picks = []
    for _, row in x.iterrows():
        for target, pcol, pmin, mfemin, surge_min in rules:
            surge_gate = max(surge_min, float(row.threshold_used)) * risk_multiplier
            if float(row[pcol]) >= pmin * risk_multiplier and float(row.predicted_mfe) >= mfemin and float(row.surge_probability) >= surge_gate:
                expected_edge = float(row[pcol]) * target - cfg["estimated_cost_hurdle_pct"]
                if expected_edge > 0:
                    picks.append((expected_edge + 0.25 * float(row[pcol]) + 0.20 * float(row.regime_score), target, row.copy()))
                break
    if not picks:
        return None
    picks.sort(key=lambda z: z[0], reverse=True)
    edge, target, row = picks[0]
    row["target_pct"] = target
    row["selection_score"] = edge
    row["selection_type"] = "mfe_gated"
    return row


def _metric_value(obj, name, default=0.0):
    value = getattr(obj, name, default)
    if isinstance(value, (pd.Series, pd.DataFrame, np.ndarray, list, tuple)):
        arr = np.asarray(value, dtype=float).reshape(-1)
        return float(arr[-1]) if arr.size else float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def main():
    cfg = yaml.safe_load((ROOT / "backtest/config_phase4.yaml").read_text())
    out = ROOT / "backtest/results/phase4"
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
    capital = float(cfg["initial_capital"])
    trades, daily = [], []
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
        mfe = score_mfe(mfe_bundle, universe)
        keep = ["symbol", "predicted_mfe", "p_hit_3", "p_hit_4", "p_hit_5"]
        scored = scored.merge(mfe[keep], on="symbol", how="inner")
        scored = scored[scored.risk_flag != "low_liquidity"].copy()
        if scored.empty:
            continue
        scored["threshold_used"] = max(float(surge_bundle.threshold), cfg["min_candidate_probability"])
        pick = choose_signal(scored, cfg)
        if pick is None:
            daily.append({"signal_date": signal_day, "executed": 0, "starting_equity": capital,
                          "ending_equity": capital, "daily_pnl": 0.0, "daily_return_pct": 0.0,
                          "regime": regime_score(scored)[1]})
            continue
        entry_day = next_day.get(signal_day)
        if entry_day is None or (entry_day, pick.symbol) not in prices_idx.index:
            continue
        start = capital
        trade = execute(prices_idx.loc[(entry_day, pick.symbol)], start, cfg["costs"], float(pick.target_pct))
        trade.update({"signal_date": signal_day, "entry_date": entry_day, "symbol": pick.symbol,
                      "surge_probability": float(pick.surge_probability), "predicted_mfe": float(pick.predicted_mfe),
                      "p_hit_3": float(pick.p_hit_3), "p_hit_4": float(pick.p_hit_4), "p_hit_5": float(pick.p_hit_5),
                      "target_pct": float(pick.target_pct), "selection_score": float(pick.selection_score),
                      "regime": str(pick.regime), "regime_score": float(pick.regime_score),
                      "validation_precision": _metric_value(surge_bundle, "validation_precision"),
                      "mfe_validation_mae": _metric_value(mfe_bundle, "validation_mae"),
                      "mfe_validation_rank_ic": _metric_value(mfe_bundle, "validation_rank_ic")})
        trades.append(trade)
        capital += trade["net_pnl"]
        daily.append({"signal_date": signal_day, "entry_date": entry_day, "executed": 1,
                      "starting_equity": start, "ending_equity": capital, "daily_pnl": capital - start,
                      "daily_return_pct": capital / start - 1, "regime": str(pick.regime),
                      "target_pct": float(pick.target_pct)})

    trades = pd.DataFrame(trades)
    daily = pd.DataFrame(daily)
    if trades.empty:
        raise RuntimeError("Phase 4 produced zero trades; gates may be too strict")
    daily["peak_equity"] = daily.ending_equity.cummax()
    daily["drawdown_pct"] = daily.ending_equity / daily.peak_equity - 1
    monthly = trades.groupby(pd.to_datetime(trades.entry_date).dt.to_period("M")).net_pnl.sum().rename("net_pnl")
    monthly_equity = float(cfg["initial_capital"])
    monthly_rows = []
    for period, pnl in monthly.items():
        start_eq = monthly_equity
        monthly_equity += float(pnl)
        monthly_rows.append({"month": str(period), "starting_equity": start_eq, "net_pnl": float(pnl),
                             "return_pct": monthly_equity / start_eq - 1})
    qualified_daily = daily[daily.executed == 1]
    summary = {
        "initial_capital": cfg["initial_capital"], "final_equity": float(capital),
        "net_pnl": float(trades.net_pnl.sum()), "net_return_pct": float(capital / cfg["initial_capital"] - 1),
        "trade_count": int(len(trades)), "win_rate": float((trades.net_pnl > 0).mean()),
        "profit_factor": float(trades.loc[trades.net_pnl > 0, "net_pnl"].sum() / max(-trades.loc[trades.net_pnl < 0, "net_pnl"].sum(), 1e-9)),
        "max_drawdown_pct": float(daily.drawdown_pct.min()),
        "profitable_execution_days": int((qualified_daily.daily_pnl > 0).sum()),
        "losing_execution_days": int((qualified_daily.daily_pnl < 0).sum()),
        "no_trade_days": int((daily.executed == 0).sum()),
        "target_distribution": trades.target_pct.value_counts().sort_index().to_dict(),
        "mean_target_pct": float(trades.target_pct.mean()),
        "daily_target_hit_rate": {str(t): float(((trades.target_pct == t) & (trades.exit_reason == "take_profit")).mean()) for t in sorted(trades.target_pct.unique())},
        "monthly_results": monthly_rows,
        "objective": {"daily_target_min_pct": 0.03, "daily_target_max_pct": 0.05, "monthly_target_pct": 0.30},
        "warning": "The requested 3-5% daily and 30% monthly targets are research objectives, not guaranteed outcomes; the backtest must prove them out-of-sample.",
        "anti_lookahead": [
            "Model labels are required to have their next-session label date strictly before the prediction day.",
            "Training uses only the trailing configured history.",
            "Liquidity is computed only from data strictly before the signal day.",
            "Signal is generated at EOD and executed on the next session open.",
            "Next-session OHLC is never used in model fitting or signal selection.",
            "Monthly model refits use only information available before that month.",
        ],
        "costs": cfg["costs"],
    }
    trades.to_csv(out / "trades.csv", index=False)
    daily.to_csv(out / "daily_equity.csv", index=False)
    pd.DataFrame(monthly_rows).to_csv(out / "monthly.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
