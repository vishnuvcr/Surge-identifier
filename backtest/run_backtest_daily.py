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
    total = brokerage + exchange + stt + sebi + stamp + gst
    return {"brokerage": brokerage, "exchange": exchange, "stt": stt, "sebi": sebi, "stamp": stamp, "gst": gst, "total": total}


def qty_for_allocation(allocation, raw_open, c):
    entry = raw_open * (1 + c["slippage_per_side"])
    variable_buy = c["exchange_turnover_rate"] + c["sebi_turnover_rate"] + c["stamp_buy_rate"]
    q = int(max((allocation - c["brokerage_per_order"]) / (entry * (1 + variable_buy)), 0))
    return q, entry


def execute(row, allocation, c, mode, target):
    raw_open, raw_high, raw_close = map(float, [row.open, row.high, row.close])
    q, entry = qty_for_allocation(allocation, raw_open, c)
    if q <= 0:
        return {"qty": 0, "net_pnl": 0.0, "gross_pnl": 0.0, "fees": 0.0, "return_pct": 0.0, "exit_reason": "no_qty"}
    buy = q * entry
    tp = entry * (1 + target)
    hit = raw_high >= tp
    if mode == "tp3_close" and hit:
        raw_exit, reason = tp, "take_profit"
    else:
        raw_exit, reason = raw_close, "close_proxy"
    exit_price = raw_exit * (1 - c["slippage_per_side"])
    sell = q * exit_price
    gross = sell - buy
    f = fees(buy, sell, c)
    net = gross - f["total"]
    return {"qty": q, "entry_price": entry, "exit_price": exit_price, "raw_open": raw_open, "raw_high": raw_high, "raw_close": raw_close, "buy_value": buy, "sell_value": sell, "gross_pnl": gross, "net_pnl": net, "fees": f["total"], **{f"fee_{k}": v for k, v in f.items() if k != "total"}, "return_pct": net / max(allocation, 1), "exit_reason": reason}


def fit_model_for_day(feat, day, cfg):
    prior = feat[feat.date < day].copy()
    prior["label_date"] = prior.groupby("symbol")["date"].shift(-1)
    prior = prior[prior["label_date"].notna() & (prior["label_date"] < day)].drop(columns=["label_date"])
    prior_days = sorted(pd.to_datetime(prior.date).unique())
    prior = prior[prior.date.isin(prior_days[-cfg["training_lookback_sessions"]:])]
    if prior.empty:
        return None
    return train_walk_forward(prior, validation_days=cfg["validation_days"], seed=cfg["seed"])


def trailing_correlation(prices, day, symbols, lookback):
    hist = prices[prices.date < day].copy()
    dates = sorted(pd.to_datetime(hist.date).dt.normalize().unique())[-lookback:]
    hist = hist[hist.date.isin(dates) & hist.symbol.isin(symbols)]
    if hist.empty:
        return pd.DataFrame(index=symbols, columns=symbols, dtype=float).fillna(0.0)
    ret = hist.pivot(index="date", columns="symbol", values="close").pct_change()
    corr = ret.corr(min_periods=15).reindex(index=symbols, columns=symbols).fillna(0.0)
    np.fill_diagonal(corr.values, 0.0)
    return corr


def select_portfolio(scored, corr, n_positions, diversification_lambda):
    scored = scored.sort_values(["utility", "surge_probability"], ascending=False).copy()
    qualified = scored[scored["signal"]].copy()
    if qualified.empty:
        selected = scored.head(1).copy()
        selected["selection_type"] = "forced_fallback"
        return selected

    selected_rows = [qualified.iloc[0]]
    remaining = qualified.iloc[1:].copy()
    while len(selected_rows) < min(n_positions, len(qualified)) and not remaining.empty:
        names = [str(x.symbol) for x in selected_rows]
        candidates = []
        for _, r in remaining.iterrows():
            corr_vals = [abs(float(corr.loc[str(r.symbol), s])) for s in names if str(r.symbol) in corr.index and s in corr.columns]
            max_corr = max(corr_vals) if corr_vals else 0.0
            adjusted = float(r.utility) / (1.0 + diversification_lambda * max_corr)
            candidates.append((adjusted, float(r.utility), float(r.surge_probability), r))
        candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        chosen = candidates[0][-1]
        selected_rows.append(chosen)
        remaining = remaining[remaining.symbol != chosen.symbol]

    selected = pd.DataFrame(selected_rows).copy()
    selected["selection_type"] = "qualified"
    return selected


def build_daily_universes(feat, prices, cfg):
    days = sorted(pd.to_datetime(feat.date).dt.normalize().unique())
    min_hist = cfg["min_train_days"] + cfg["validation_days"]
    eval_days = days[min_hist:]
    if cfg.get("backtest_start"):
        eval_days = [d for d in eval_days if d >= pd.Timestamp(cfg["backtest_start"])]
    if cfg.get("backtest_end"):
        eval_days = [d for d in eval_days if d <= pd.Timestamp(cfg["backtest_end"])]

    universes = {}
    last_period = None
    bundle = None
    min_liq = float(cfg["min_avg_turnover_cr"])
    fallback_liq = float(cfg.get("fallback_min_turnover_cr", 1.0))

    for day in eval_days:
        period = pd.Timestamp(day).to_period("M")
        if period != last_period:
            bundle = fit_model_for_day(feat, pd.Timestamp(day), cfg)
            last_period = period
        if bundle is None:
            continue
        d = feat[feat.date == day].copy()
        if d.empty:
            continue
        hist = prices[prices.date < day]
        hdays = sorted(pd.to_datetime(hist.date).dt.normalize().unique())[-cfg["liquidity_lookback_sessions"]:]
        med = hist[hist.date.isin(hdays)].groupby("symbol").turnover.median() / 1e7
        eligible = med[med >= min_liq].index
        universe = d[d.symbol.isin(eligible)].copy()
        liquidity_mode = "strict"
        if universe.empty:
            eligible = med[med >= fallback_liq].index
            universe = d[d.symbol.isin(eligible)].copy()
            liquidity_mode = "fallback_liquidity"
        if universe.empty:
            universe = d.copy()
            liquidity_mode = "all_available"
        scored = score_latest(bundle, universe)
        scored = scored[scored.risk_flag != "low_liquidity"].copy()
        if scored.empty:
            continue
        scored["threshold_used"] = max(float(bundle.threshold), float(cfg.get("min_candidate_probability", 0.0)))
        scored["signal"] = scored.surge_probability >= scored["threshold_used"]
        scored["signal_date"] = day
        scored["liquidity_mode"] = liquidity_mode
        scored["validation_precision"] = float(bundle.validation_precision)
        scored["validation_recall"] = float(bundle.validation_recall)
        scored["validation_specificity"] = float(bundle.validation_specificity)
        universes[day] = scored
    return universes


def run_portfolio(cfg, universes, prices, mode, n_positions):
    trading_days = sorted(pd.to_datetime(prices.date).dt.normalize().unique())
    next_day = {trading_days[i]: trading_days[i + 1] for i in range(len(trading_days) - 1)}
    by_key = prices.set_index(["date", "symbol"]).sort_index()
    capital = float(cfg["initial_capital"])
    trades, daily = [], []

    for signal_day, scored in universes.items():
        entry_day = next_day.get(signal_day)
        if entry_day is None:
            continue
        symbols = list(scored.symbol.astype(str).unique())
        corr = trailing_correlation(prices, signal_day, symbols, cfg["correlation_lookback_sessions"])
        selected = select_portfolio(scored, corr, n_positions, cfg["diversification_lambda"])
        start = capital
        allocation = start / max(len(selected), 1)
        executed = 0
        fallback = False
        for _, sig in selected.iterrows():
            key = (entry_day, sig.symbol)
            if key not in by_key.index:
                continue
            trade = execute(by_key.loc[key], allocation, cfg["costs"], mode, cfg["take_profit_pct"])
            trade.update({"signal_date": signal_day, "entry_date": entry_day, "symbol": sig.symbol, "surge_probability": float(sig.surge_probability), "utility": float(sig.utility), "atr_pct": float(sig.atr_pct), "relvol20": float(sig.relvol20), "turnover_cr": float(sig.turnover_cr), "risk_flag": str(sig.risk_flag), "selection_type": str(sig.selection_type), "threshold_used": float(sig.threshold_used), "validation_precision": float(sig.validation_precision), "validation_recall": float(sig.validation_recall), "validation_specificity": float(sig.validation_specificity)})
            trades.append(trade)
            capital += trade["net_pnl"]
            executed += 1
            fallback = fallback or sig.selection_type == "forced_fallback"
        if executed:
            daily.append({"date": entry_day, "starting_equity": start, "ending_equity": capital, "daily_pnl": capital - start, "daily_return_pct": capital / start - 1, "n_trades": executed, "used_forced_fallback": fallback})

    trades = pd.DataFrame(trades)
    daily = pd.DataFrame(daily)
    if trades.empty or daily.empty:
        raise RuntimeError(f"No executable trades for portfolio size {n_positions}")
    daily["peak_equity"] = daily.ending_equity.cummax()
    daily["drawdown_pct"] = daily.ending_equity / daily.peak_equity - 1
    initial = float(cfg["initial_capital"])
    qualified = trades[trades.selection_type == "qualified"]
    fallback = trades[trades.selection_type == "forced_fallback"]
    win_profit = float(trades.loc[trades.net_pnl > 0, "net_pnl"].sum())
    loss_abs = float(-trades.loc[trades.net_pnl < 0, "net_pnl"].sum())
    summary = {
        "portfolio_size": n_positions,
        "mode": mode,
        "initial_capital": initial,
        "final_equity": float(capital),
        "net_pnl": float(trades.net_pnl.sum()),
        "net_return_pct": float(capital / initial - 1),
        "gross_pnl": float(trades.gross_pnl.sum()),
        "fees": float(trades.fees.sum()),
        "trade_count": int(len(trades)),
        "qualified_trade_count": int(len(qualified)),
        "fallback_trade_count": int(len(fallback)),
        "fallback_trade_fraction": float(len(fallback) / max(len(trades), 1)),
        "win_rate": float((trades.net_pnl > 0).mean()),
        "profit_factor": float(win_profit / loss_abs) if loss_abs > 0 else None,
        "max_drawdown_pct": float(daily.drawdown_pct.min()),
        "profitable_days": int((daily.daily_pnl > 0).sum()),
        "losing_days": int((daily.daily_pnl < 0).sum()),
        "trading_days_with_execution": int(len(daily)),
        "average_trades_per_day": float(daily.n_trades.mean()),
        "portfolio_fullness_ratio": float((daily.n_trades == n_positions).mean()),
        "daily_fallback_days": int(daily.used_forced_fallback.sum()),
        "anti_lookahead": [
            "Model fitting uses only feature rows whose realized next-session label date is strictly before the prediction day.",
            "Training history is limited to the configured trailing sessions.",
            "Threshold is selected only inside the historical validation segment preceding the prediction month.",
            "Liquidity is computed only from observations strictly before the signal date.",
            "Correlation diversification uses only trailing returns strictly before the signal date.",
            "Signal is generated at EOD; entry is the next trading-session open.",
            "Next-day OHLC is used only after the portfolio has been selected, solely for trade outcome calculation.",
            "A forced fallback is exactly one trade when no qualified signal exists; it is never expanded to five low-confidence trades.",
        ],
        "costs": cfg["costs"],
    }
    return trades, daily, summary


def main():
    cfg = yaml.safe_load((ROOT / "backtest/config.yaml").read_text())
    out = ROOT / "backtest/results"
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
    universes = build_daily_universes(feat, prices, cfg)
    if not universes:
        raise RuntimeError("No historical point-in-time scoring days")

    rows = []
    for mode in ["close_exit", "tp3_close"]:
        for n_positions in cfg["portfolio_sizes"]:
            trades, daily, summary = run_portfolio(cfg, universes, prices, mode, int(n_positions))
            tag = f"{mode}_p{n_positions}"
            trades.to_csv(out / f"trades_{tag}.csv", index=False)
            daily.to_csv(out / f"daily_equity_{tag}.csv", index=False)
            (out / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2, default=str))
            rows.append({"mode": mode, "portfolio_size": n_positions, "final_equity": summary["final_equity"], "net_pnl": summary["net_pnl"], "return_pct": summary["net_return_pct"], "fees": summary["fees"], "trades": summary["trade_count"], "qualified_trades": summary["qualified_trade_count"], "fallback_trades": summary["fallback_trade_count"], "win_rate": summary["win_rate"], "profit_factor": summary["profit_factor"], "max_drawdown_pct": summary["max_drawdown_pct"], "profitable_days": summary["profitable_days"], "losing_days": summary["losing_days"], "avg_trades_per_day": summary["average_trades_per_day"], "full_portfolio_days": summary["portfolio_fullness_ratio"]})

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out / "summary.csv", index=False)
    lines = [
        "# Phase 2 — Diversified Point-in-Time Intraday Backtest", "",
        "Initial capital: ₹1,00,000. Portfolio sizes tested: 1, 3 and 5.",
        "Capital is split equally across the selected positions each day.",
        "At least one executable trade is selected each trading day; when no qualified signal exists, exactly one fallback trade is used.",
        "For portfolios of 3 or 5, only threshold-qualified candidates can fill additional positions; the system never pads a portfolio with multiple weak fallback trades.",
        "Additional positions are selected using trailing correlation to reduce concentration among highly correlated candidates.",
        "",
        "| Exit mode | Portfolio | Final equity | Return | Fees | Trades | Qualified | Fallback | Win rate | PF | Max DD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary_df.iterrows():
        pf = "n/a" if pd.isna(r["profit_factor"]) else f"{r['profit_factor']:.2f}"
        lines.append(f"| {r['mode']} | {int(r['portfolio_size'])} | ₹{r['final_equity']:,.2f} | {r['return_pct']*100:.2f}% | ₹{r['fees']:,.2f} | {int(r['trades'])} | {int(r['qualified_trades'])} | {int(r['fallback_trades'])} | {r['win_rate']*100:.2f}% | {pf} | {r['max_drawdown_pct']*100:.2f}% |")
    lines += [
        "", "## Anti-lookahead controls",
        "- Historical labels are fully realized before model fitting for each prediction day.",
        "- Trailing liquidity and trailing return correlations are computed strictly before the signal day.",
        "- EOD signal → next-session-open entry.",
        "- Future OHLC is used only after portfolio selection for outcome simulation.",
        "- Integer-share quantities; no leverage.",
        "- 0.05% slippage per side plus modeled brokerage/statutory charges.",
        "", "## Daily-bar limitation",
        "The close-exit path is a close-price proxy because daily OHLC does not contain the actual intraday auto-square-off price. The +3% target path is reported separately."
    ]
    (out / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
