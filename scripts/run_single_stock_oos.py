from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.nse_data import load_prices
from src.nse_fo import load_fo
from src.single_stock_strategy import build_single_stock_frame, walk_forward_oos

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description="Single-stock next-open ML walk-forward OOS research")
    p.add_argument("--symbol", default="RELIANCE", help="NSE EQ symbol")
    p.add_argument("--lookback-days", type=int, default=1800)
    p.add_argument("--train-days", type=int, default=504)
    p.add_argument("--test-days", type=int, default=21)
    p.add_argument("--step-days", type=int, default=21)
    p.add_argument("--target", type=float, default=0.0275)
    p.add_argument("--stop", type=float, default=0.0125)
    p.add_argument("--probability-gate", type=float, default=0.60)
    return p.parse_args()


def context_from_equities(prices: pd.DataFrame, symbol: str) -> pd.DataFrame:
    x = prices.copy()
    x["date"] = pd.to_datetime(x["date"])
    idx = x.groupby("date").agg(
        nifty_ret1=("ret_placeholder", "mean")
    ) if "ret_placeholder" in x else None
    # The NSE cash archive does not itself contain NIFTY index rows in a portable way.
    # These columns are intentionally neutral until index/context adapters are populated.
    out = pd.DataFrame({"date": sorted(x["date"].unique())})
    for c, v in {
        "nifty_ret1": 0.0, "nifty_ret5": 0.0, "nifty_gap": 0.0,
        "india_vix_ret1": 0.0, "sector_ret1": 0.0, "sector_ret5": 0.0,
        "breadth": 0.5, "news_score": 0.0, "news_count": 0.0,
        "corporate_action_flag": 0.0,
    }.items():
        out[c] = v
    return out


def add_fo_context(stock: pd.DataFrame, fo: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if fo.empty:
        return stock
    z = fo[fo["symbol"].eq(symbol)].sort_values("date").copy()
    if z.empty:
        return stock
    z["fo_oi_z20"] = (z["fut_oi"] - z["fut_oi"].rolling(20, min_periods=5).mean()) / z["fut_oi"].rolling(20, min_periods=5).std().replace(0, pd.NA)
    z["fo_oi_change_z20"] = (z["fut_oi_change"] - z["fut_oi_change"].rolling(20, min_periods=5).mean()) / z["fut_oi_change"].rolling(20, min_periods=5).std().replace(0, pd.NA)
    z["fo_volume_rel20"] = z["fut_volume"] / z["fut_volume"].rolling(20, min_periods=5).mean()
    pcr_mean = z["pcr"].rolling(20, min_periods=5).mean()
    z["pcr_dev20"] = z["pcr"] - pcr_mean
    keep = z[["date", "fo_oi_z20", "fo_oi_change_z20", "fo_volume_rel20", "pcr_dev20"]]
    return stock.merge(keep, on="date", how="left")


def main():
    a = parse_args()
    cache = ROOT / "data" / "cache"
    prices = load_prices(str(cache), a.lookback_days)
    prices["date"] = pd.to_datetime(prices["date"])
    stock = prices[(prices["symbol"] == a.symbol) & (prices["series"].astype(str).str.upper() == "EQ")].copy()
    if stock.empty:
        raise SystemExit(f"No EQ history found for {a.symbol}")
    stock = stock.sort_values("date")
    stock["prev_day_high"] = stock["high"].shift(1)
    stock["prev_day_low"] = stock["low"].shift(1)
    fo = load_fo(str(cache), a.lookback_days)
    stock = add_fo_context(stock, fo, a.symbol)
    ctx = context_from_equities(prices, a.symbol)
    frame = build_single_stock_frame(stock, ctx)
    report = walk_forward_oos(
        frame,
        train_days=a.train_days,
        test_days=a.test_days,
        step_days=a.step_days,
        target=a.target,
        stop=a.stop,
        probability_gate=a.probability_gate,
    )
    out = ROOT / "reports" / "single_stock"
    out.mkdir(parents=True, exist_ok=True)
    report.predictions.to_csv(out / f"{a.symbol}_oos_predictions.csv", index=False)
    report.trades.to_csv(out / f"{a.symbol}_oos_trades.csv", index=False)
    summary = {
        "symbol": a.symbol,
        "target": a.target,
        "stop": a.stop,
        "probability_gate": a.probability_gate,
        "train_days": a.train_days,
        "test_days": a.test_days,
        "step_days": a.step_days,
        "metrics": report.metrics,
        "feature_count":  len(frame.columns),
        "note": "Context adapters for index/news/corporate-action layers are neutral placeholders until their dated data sources are wired. OOS engine never uses future rows for training.",
    }
    (out / f"{a.symbol}_oos_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
