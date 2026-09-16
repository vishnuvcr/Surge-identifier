from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from src.iron_condor_engine_v1 import backtest_prepared, prepare_market


def combinations(cfg):
    for distance, width, tp, sl in itertools.product(
        cfg["short_distance_pct"], cfg["wing_width_points"],
        cfg["take_profit_credit_pct"], cfg["stop_loss_credit_multiple"]
    ):
        yield {
            "capital": cfg["capital"], "entry_weekday": cfg["entry_weekday"],
            "distance": distance, "width": width, "take_profit": tp, "stop_loss": sl,
        }


def rank(s):
    if not s.get("data_ok"):
        return -1e9
    # Prioritize consistency and drawdown, while retaining return as a component.
    return (
        2.0 * s["median_weekly_return"]
        + 1.0 * s["avg_weekly_return"]
        + 0.50 * s["worst_weekly_return"]
        + 0.25 * s["weeks_at_or_above_5pct"]
        + 0.50 * s["weeks_nonnegative"]
        + 0.10 * s["win_rate"]
        + 0.10 * s["theoretical_compounded_return"]
        + 0.50 * s["theoretical_compounded_max_drawdown"]
    )


def load_dataset(path: Path) -> pd.DataFrame:
    data = pd.read_parquet(path)
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    data["expiry"] = pd.to_datetime(data["expiry"]).dt.normalize()
    data = data.loc[data["option_type"].isin(["CE", "PE"])].copy()
    futures_path = Path(os.environ.get("IRON_CONDOR_FUTURES_PATH", "data/cache/nifty_futures_long.parquet"))
    if not futures_path.exists():
        raise RuntimeError(f"Missing NSE futures proxy: {futures_path}")
    futures = pd.read_parquet(futures_path)
    futures["date"] = pd.to_datetime(futures["date"]).dt.normalize()
    futures["expiry"] = pd.to_datetime(futures["expiry"]).dt.normalize()
    futures["close"] = pd.to_numeric(futures["close"], errors="coerce")
    futures = futures.dropna(subset=["date", "expiry", "close"])
    futures["dte"] = (futures["expiry"] - futures["date"]).dt.days
    futures = futures.loc[futures["dte"] >= 0].sort_values(["date", "dte"])
    futures = futures.groupby("date", as_index=False).first()[["date", "close"]].rename(columns={"close":"underlying_close"})
    data = data.merge(futures, on="date", how="left", validate="many_to_one")
    coverage = float(data["underlying_close"].notna().mean())
    if coverage < 0.995:
        raise RuntimeError(f"Futures underlying coverage too low: {coverage:.4f}")
    data["dte"] = (data["expiry"] - data["date"]).dt.days
    data = data.loc[data["dte"].between(0, 7)].copy()
    rel_band = 0.03 + 250.0 / data["underlying_close"] + 0.01
    data["rel_strike_distance"] = (data["strike"] / data["underlying_close"] - 1.0).abs()
    data = data.loc[data["rel_strike_distance"] <= rel_band]
    data = data.drop(columns=["dte", "rel_strike_distance"])
    return data.sort_values(["date","expiry","strike","option_type"]).reset_index(drop=True)


def main():
    cfg = yaml.safe_load(Path("config/success_v1.yaml").read_text())
    out = Path("backtest/results/success-v1")
    out.mkdir(parents=True, exist_ok=True)
    data_path = Path(os.environ.get("IRON_CONDOR_DATA_PATH", "data/cache/nifty_options_long.parquet"))
    data = load_dataset(data_path)
    dates = sorted(data.date.unique())
    split = pd.Timestamp(dates[int(len(dates) * 0.70)])
    train = data[data.date < split].copy()
    test = data[data.date >= split].copy()
    train_market, test_market = prepare_market(train), prepare_market(test)

    rows, best, best_score = [], None, -1e18
    for i, c in enumerate(combinations(cfg)):
        _, s = backtest_prepared(train_market, c)
        rec = {**s, "config_id": i, **c, "train_score": rank(s)}
        rows.append(rec)
        if s.get("data_ok") and rec["train_score"] > best_score:
            best_score, best = rec["train_score"], c
    if best is None:
        raise RuntimeError("No viable configuration")

    oos_trades, oos = backtest_prepared(test_market, best)
    oos.update({
        "split_date": str(split.date()), "selected_parameters": best,
        "dataset_start": str(data.date.min().date()), "dataset_end": str(data.date.max().date()),
        "dataset_rows": int(len(data)), "dataset_dates": int(len(dates)),
        "oos_start": str(test.date.min().date()), "oos_end": str(test.date.max().date()),
    })

    leaderboard = pd.DataFrame(rows).sort_values("train_score", ascending=False).reset_index(drop=True)
    leaderboard.to_csv(out / "training_leaderboard.csv", index=False)
    oos_trades.to_csv(out / "oos_trades.csv", index=False)

    yearly = []
    oos_trades["exit_date"] = pd.to_datetime(oos_trades["exit_date"])
    for year, g in oos_trades.groupby(oos_trades["exit_date"].dt.year):
        yearly.append({
            "year": int(year), "trades": len(g), "win_rate": float((g.net_pnl > 0).mean()),
            "net_pnl": float(g.net_pnl.sum()), "fixed_lot_return": float(g.net_pnl.sum()/cfg["capital"]),
            "worst_trade": float(g.net_pnl.min()),
            "negative_intraday_bound_rate": float((g.conservative_intraday_pnl < 0).mean()),
        })
    pd.DataFrame(yearly).to_csv(out / "oos_year_breakdown.csv", index=False)

    # TP/SL invariance audit: many identical rows across TP/SL is a warning that
    # daily EOD data cannot distinguish the path-dependent exits.
    fp = leaderboard[["take_profit","stop_loss","trades","take_profit_trades","stop_loss_trades","expiry_trades","fixed_lot_total_return"]].copy()
    fp.to_csv(out / "tp_sl_behavior_audit.csv", index=False)
    unique_exit_profiles = int(fp[["take_profit","stop_loss","take_profit_trades","stop_loss_trades","expiry_trades"]].drop_duplicates().shape[0])

    audit = {
        "purpose": "Corrected long-history research plus paper-trading candidate",
        "data_frequency": "daily EOD NSE F&O bhavcopy",
        "underlying_series": "nearest non-expired NIFTY index-futures close from NSE bhavcopy",
        "chronology": "70/30 chronological train/OOS split; OOS untouched during selection",
        "parameter_grid": 512,
        "accounting": "fixed-lot cumulative equity and separate theoretical per-trade compounded equity are both reported",
        "execution": "EOD close exits only; intraday OHLC is used only as a conservative path-risk flag",
        "tp_sl_behavior_unique_profiles": unique_exit_profiles,
        "tp_sl_path_warning": unique_exit_profiles < 4,
        "paper_status": "NOT automatically executable; prospective validation only",
        "cost_model": "Paytm Money / India F&O modeled costs; actual contract-note charges should be compared after paper trades",
    }
    (out / "backtest_audit.json").write_text(json.dumps(audit, indent=2))
    (out / "oos_summary.json").write_text(json.dumps(oos, indent=2, default=str))
    print(json.dumps({"selected": best, "oos": oos, "audit": audit}, indent=2, default=str))


if __name__ == "__main__":
    main()
