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
    return {"brokerage": brokerage, "exchange": exchange, "stt": stt,
            "sebi": sebi, "stamp": stamp, "gst": gst, "total": total}


def qty_for_allocation(allocation, raw_open, c):
    entry = raw_open * (1 + c["slippage_per_side"])
    v = c["exchange_turnover_rate"] + c["sebi_turnover_rate"] + c["stamp_buy_rate"]
    q = int(max((allocation - c["brokerage_per_order"]) / (entry * (1 + v)), 0))
    return q, entry


def execute(row, allocation, c, mode, target):
    raw_open, raw_high, raw_close = map(float, [row.open, row.high, row.close])
    q, entry = qty_for_allocation(allocation, raw_open, c)
    if q <= 0:
        return {"qty": 0, "net_pnl": 0.0, "gross_pnl": 0.0, "fees": 0.0,
                "return_pct": 0.0, "exit_reason": "no_qty"}
    buy = q * entry
    tp = entry * (1 + target)
    hit = raw_high >= tp
    if mode == "tp3_close" and hit:
        raw_exit, reason = tp, "take_profit"
    else:
        raw_exit, reason = raw_close, "close"
    exit_price = raw_exit * (1 - c["slippage_per_side"])
    sell = q * exit_price
    gross = sell - buy
    f = fees(buy, sell, c)
    net = gross - f["total"]
    return {
        "qty": q, "entry_price": entry, "exit_price": exit_price,
        "raw_open": raw_open, "raw_high": raw_high, "raw_close": raw_close,
        "buy_value": buy, "sell_value": sell, "gross_pnl": gross,
        "net_pnl": net, "fees": f["total"],
        **{f"fee_{k}": v for k, v in f.items() if k != "total"},
        "return_pct": net / max(allocation, 1), "exit_reason": reason,
    }


def fit_model_for_day(feat, day, cfg):
    prior = feat[feat.date < day].copy()
    # build_features labels each row with the next available observation for that symbol.
    # Only rows whose realized label date is strictly before the prediction day are legal training rows.
    prior["label_date"] = prior.groupby("symbol")["date"].shift(-1)
    prior = prior[prior["label_date"].notna() & (prior["label_date"] < day)].drop(columns=["label_date"])
    prior_days = sorted(pd.to_datetime(prior.date).unique())
    prior = prior[prior.date.isin(prior_days[-cfg["training_lookback_sessions"]:])]
    if prior.empty:
        return None
    return train_walk_forward(prior, validation_days=cfg["validation_days"], seed=cfg["seed"])


def build_predictions(feat, prices, cfg):
    days = sorted(pd.to_datetime(feat.date).dt.normalize().unique())
    min_hist = cfg["min_train_days"] + cfg["validation_days"]
    eval_days = days[min_hist:]
    if cfg.get("backtest_start"):
        eval_days = [d for d in eval_days if d >= pd.Timestamp(cfg["backtest_start"])]
    if cfg.get("backtest_end"):
        eval_days = [d for d in eval_days if d <= pd.Timestamp(cfg["backtest_end"])]

    predictions = []
    last_period = None
    bundle = None
    max_trades = int(cfg["max_trades_per_day"])
    force_min_one = bool(cfg.get("force_min_one_trade", True))
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
        strict = d[d.symbol.isin(eligible)].copy()
        if strict.empty:
            eligible = med[med >= fallback_liq].index
            strict = d[d.symbol.isin(eligible)].copy()
        if strict.empty:
            strict = d.copy()
        if strict.empty:
            continue

        scored = score_latest(bundle, strict)
        scored = scored[scored.risk_flag != "low_liquidity"].copy()
        if scored.empty:
            scored = score_latest(bundle, d)
            scored = scored[scored.risk_flag != "low_liquidity"].copy()
        if scored.empty:
            continue

        qualified = scored[scored.signal].sort_values(
            ["utility", "surge_probability"], ascending=False
        )
        if len(qualified) >= 1:
            selected = qualified.head(max_trades).copy()
            selection_type = "qualified"
        elif force_min_one:
            selected = scored.sort_values(
                ["utility", "surge_probability"], ascending=False
            ).head(max_trades).copy()
            selected["fallback"] = True
            selection_type = "forced_fallback"
        else:
            continue

        for _, r in selected.iterrows():
            predictions.append({
                "signal_date": day,
                "symbol": r.symbol,
                "surge_probability": float(r.surge_probability),
                "utility": float(r.utility),
                "atr_pct": float(r.atr_pct),
                "relvol20": float(r.relvol20),
                "turnover_cr": float(r.turnover_cr),
                "risk_flag": str(r.risk_flag),
                "threshold": float(bundle.threshold),
                "validation_precision": float(bundle.validation_precision),
                "validation_recall": float(bundle.validation_recall),
                "validation_specificity": float(bundle.validation_specificity),
                "selection_type": selection_type,
            })
    return pd.DataFrame(predictions)


def run(cfg, mode):
    prices = load_prices("data/cache", cfg["lookback_days"])
    prices.date = pd.to_datetime(prices.date).dt.normalize()
    fo = load_fo("data/cache", cfg["lookback_days"])
    if not fo.empty:
        fo.date = pd.to_datetime(fo.date).dt.normalize()
        data = prices.merge(fo, on=["date", "symbol"], how="left")
    else:
        data = prices
    feat = build_features(data)
    feat.date = pd.to_datetime(feat.date).dt.normalize()

    pred = build_predictions(feat, prices, cfg)
    if pred.empty:
        raise RuntimeError("No historical predictions after point-in-time filters")

    trading_days = sorted(prices.date.unique())
    next_day = {trading_days[i]: trading_days[i + 1] for i in range(len(trading_days) - 1)}
    pred["entry_date"] = pred.signal_date.map(next_day)
    pred = pred.dropna(subset=["entry_date"])

    by_key = prices.set_index(["date", "symbol"]).sort_index()
    capital = float(cfg["initial_capital"])
    trades, daily = [], []

    for entry_date, sigs in pred.groupby("entry_date", sort=True):
        sigs = sigs.drop_duplicates("symbol").head(cfg["max_trades_per_day"])
        start = capital
        allocation = start / max(len(sigs), 1)
        day_trades = 0
        day_fallback = False

        for _, sig in sigs.iterrows():
            key = (entry_date, sig.symbol)
            if key not in by_key.index:
                continue
            t = execute(by_key.loc[key], allocation, cfg["costs"], mode, cfg["take_profit_pct"])
            t.update({
                "signal_date": sig.signal_date, "entry_date": entry_date, "symbol": sig.symbol,
                "surge_probability": sig.surge_probability, "utility": sig.utility,
                "atr_pct": sig.atr_pct, "relvol20": sig.relvol20, "turnover_cr": sig.turnover_cr,
                "risk_flag": sig.risk_flag, "selection_type": sig.selection_type,
                "model_threshold": sig.threshold,
                "validation_precision": sig.validation_precision,
                "validation_recall": sig.validation_recall,
                "validation_specificity": sig.validation_specificity,
            })
            trades.append(t)
            capital += t["net_pnl"]
            day_trades += 1
            day_fallback = day_fallback or sig.selection_type == "forced_fallback"

        if day_trades:
            daily.append({
                "date": entry_date, "starting_equity": start, "ending_equity": capital,
                "daily_pnl": capital - start, "daily_return_pct": capital / start - 1,
                "n_trades": day_trades, "used_forced_fallback": day_fallback,
            })

    trades = pd.DataFrame(trades)
    daily = pd.DataFrame(daily)
    if trades.empty or daily.empty:
        raise RuntimeError("No executable historical trades after point-in-time filters")

    daily["peak_equity"] = daily.ending_equity.cummax()
    daily["drawdown_pct"] = daily.ending_equity / daily.peak_equity - 1
    initial = cfg["initial_capital"]
    summary = {
        "mode": mode,
        "initial_capital": initial,
        "final_equity": float(capital),
        "net_pnl": float(trades.net_pnl.sum()),
        "net_return_pct": float(capital / initial - 1),
        "gross_pnl": float(trades.gross_pnl.sum()),
        "fees": float(trades.fees.sum()),
        "trade_count": int(len(trades)),
        "win_rate": float((trades.net_pnl > 0).mean()),
        "max_drawdown_pct": float(daily.drawdown_pct.min()),
        "profitable_days": int((daily.daily_pnl > 0).sum()),
        "losing_days": int((daily.daily_pnl < 0).sum()),
        "flat_days": int((daily.daily_pnl == 0).sum()),
        "trading_days_with_execution": int(len(daily)),
        "avg_trades_per_day": float(daily.n_trades.mean()),
        "forced_fallback_days": int(daily.used_forced_fallback.sum()),
        "forced_fallback_trade_fraction": float((trades.selection_type == "forced_fallback").mean()),
        "anti_lookahead": [
            "For each prediction month, model fitting uses only rows whose signal date and realized label date are both strictly before the prediction day.",
            "The training window is limited to a trailing number of historical sessions only.",
            "Threshold selection occurs only inside the historical validation segment available before the prediction month.",
            "Liquidity uses only trailing turnover observations strictly before each signal date.",
            "The signal is EOD and the trade enters on the next trading session open.",
            "Next-day OHLC is used only after the signal is fixed to calculate trade outcome.",
            "Forced-fallback selection uses only the EOD model score and historical liquidity; it does not inspect the next day's price.",
            "No leverage; capital is split equally across selected trades using starting equity for that day.",
        ],
        "daily_trade_requirement": {
            "enabled": bool(cfg.get("force_min_one_trade", True)),
            "requested_min_trades_per_day": 1,
            "note": "When no model-qualified signal exists, the highest-utility eligible candidate is traded and explicitly marked forced_fallback.",
        },
        "costs": cfg["costs"],
    }
    return trades, daily, summary


def main():
    cfg = yaml.safe_load((ROOT / "backtest/config.yaml").read_text())
    out = ROOT / "backtest/results"
    out.mkdir(parents=True, exist_ok=True)
    rows, summaries = [], {}
    for mode in ["close_exit", "tp3_close"]:
        trades, daily, summary = run(cfg, mode)
        trades.to_csv(out / f"trades_{mode}.csv", index=False)
        daily.to_csv(out / f"daily_equity_{mode}.csv", index=False)
        (out / f"summary_{mode}.json").write_text(json.dumps(summary, indent=2, default=str))
        summaries[mode] = summary
        rows.append({
            "mode": mode, "final_equity": summary["final_equity"],
            "net_pnl": summary["net_pnl"], "return_pct": summary["net_return_pct"],
            "gross_pnl": summary["gross_pnl"], "fees": summary["fees"],
            "trades": summary["trade_count"], "win_rate": summary["win_rate"],
            "max_drawdown_pct": summary["max_drawdown_pct"],
            "profitable_days": summary["profitable_days"],
            "losing_days": summary["losing_days"],
            "fallback_days": summary["forced_fallback_days"],
            "fallback_trade_fraction": summary["forced_fallback_trade_fraction"],
        })

    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    lines = [
        "# Point-in-Time Intraday Backtest — Daily Trade Mode", "",
        "Initial capital: ₹1,00,000; maximum 5 simultaneous trades; equal capital split.",
        "At least 1 trade is selected each trading day whenever an executable NSE equity candidate exists.",
        "Days where the model has no threshold-qualified signal use an explicitly marked `forced_fallback` top-ranked candidate.",
        "",
        "| Mode | Final equity | Net P&L | Return | Fees | Trades | Win rate | Max DD | Fallback days |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['mode']} | ₹{r['final_equity']:,.2f} | ₹{r['net_pnl']:,.2f} | {r['return_pct']*100:.2f}% | "
            f"₹{r['fees']:,.2f} | {r['trades']} | {r['win_rate']*100:.2f}% | {r['max_drawdown_pct']*100:.2f}% | {r['fallback_days']} |"
        )
    lines += [
        "", "## Key safeguards",
        "", "- Point-in-time model retraining; no future labels cross the prediction-day boundary.",
        "- Trailing liquidity only; no full-sample liquidity filter.",
        "- EOD signal, next-session-open entry.",
        "- 0.05% slippage per side.",
        "- Paytm Money brokerage modeled at ₹20 per executed order; two orders per completed trade.",
        "- Intraday statutory charges included: exchange turnover, 0.025% sell-side STT, SEBI turnover fee, buy-side stamp duty, and 18% GST on brokerage + exchange charges.",
        "- Daily-bar limitation: exact intraday price path is unavailable; the +3% target case is therefore a separate scenario.",
        "", "## Interpretation", "",
        "The forced-fallback days are intentionally reported separately. A daily-trade requirement can materially reduce the quality of the trading edge because it prevents the model from staying flat when no strong signal exists.",
    ]
    (out / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
