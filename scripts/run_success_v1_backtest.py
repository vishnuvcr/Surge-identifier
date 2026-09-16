from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.iron_condor_engine_v1 import backtest_prepared, prepare_market


def combinations(cfg):
    for distance, width, tp, sl in itertools.product(
        cfg["short_distance_pct"], cfg["wing_width_points"],
        cfg["take_profit_credit_pct"], cfg["stop_loss_credit_multiple"],
    ):
        yield {
            "capital": cfg["capital"],
            "entry_weekday": cfg["entry_weekday"],
            "min_days_to_expiry": cfg["min_days_to_expiry"],
            "max_days_to_expiry": cfg["max_days_to_expiry"],
            "distance": distance,
            "width": width,
            "take_profit": tp,
            "stop_loss": sl,
        }


def rank(s):
    if not s.get("data_ok"):
        return -1e18
    return (
        2.0 * s["median_weekly_return"]
        + s["avg_weekly_return"]
        + 0.10 * s["weeks_at_or_above_5pct"]
        + 0.25 * s["weeks_nonnegative"]
        + 0.10 * s["win_rate"]
        + 0.50 * s["worst_weekly_return"]
        + 0.25 * s["theoretical_compounded_return"]
        + 0.50 * s["fixed_lot_max_drawdown"]
    )


def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_parquet(path)
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    data["expiry"] = pd.to_datetime(data["expiry"]).dt.normalize()
    data["option_type"] = data["option_type"].astype(str).str.upper()
    data = data.loc[data.option_type.isin(["CE", "PE"])].copy()

    futures_path = Path(os.environ.get("IRON_CONDOR_FUTURES_PATH", "data/cache/nifty_futures_long.parquet"))
    futures = pd.read_parquet(futures_path)
    futures["date"] = pd.to_datetime(futures["date"]).dt.normalize()
    futures["expiry"] = pd.to_datetime(futures["expiry"]).dt.normalize()
    futures["close"] = pd.to_numeric(futures["close"], errors="coerce")
    futures = futures.dropna(subset=["date", "expiry", "close"])
    futures["dte"] = (futures.expiry - futures.date).dt.days
    futures = futures.loc[futures.dte >= 0].sort_values(["date", "dte"])
    futures = futures.groupby("date", as_index=False).first()[["date", "close"]]
    futures = futures.rename(columns={"close": "underlying_close"})

    data = data.merge(futures, on="date", how="left", validate="many_to_one")
    coverage = float(data.underlying_close.notna().mean())
    if coverage < 0.995:
        raise RuntimeError(f"NIFTY futures coverage too low: {coverage:.4f}")

    data["dte"] = (data.expiry - data.date).dt.days
    data = data.loc[data.dte.between(0, 7)].copy()

    max_distance = max(y for y in cfg["short_distance_pct"])
    max_width = max(cfg["wing_width_points"])
    rel_band = max_distance + max_width / data["underlying_close"] + 0.01
    data["rel_strike_distance"] = (data.strike / data.underlying_close - 1.0).abs()
    data = data.loc[data.rel_strike_distance <= rel_band]
    data = data.drop(columns=["dte", "rel_strike_distance"])
    return data.sort_values(["date", "expiry", "strike", "option_type"]).reset_index(drop=True)


def main():
    global cfg
    cfg_path = ROOT / "config/success_v1.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    data_path = Path(os.environ.get("IRON_CONDOR_DATA_PATH", "data/cache/nifty_options_long.parquet"))
    data = load_data(data_path)
    out = ROOT / "backtest/results/success-v1"
    out.mkdir(parents=True, exist_ok=True)

    dates = sorted(pd.to_datetime(data.date.unique()))
    if len(dates) < 500:
        raise RuntimeError(f"Too few dates: {len(dates)}")
    split_index = min(len(dates) - 1, max(1, int(len(dates) * 0.70)))
    split = pd.Timestamp(dates[split_index])
    train = data.loc[data.date < split].copy()
    test = data.loc[data.date >= split].copy()

    train_market = prepare_market(train)
    test_market = prepare_market(test)

    rows = []
    best = None
    best_score = -1e18
    for i, c in enumerate(combinations(cfg)):
        _, summary = backtest_prepared(train_market, c)
        rec = {**summary, "config_id": i, **c, "train_score": rank(summary)}
        rows.append(rec)
        if summary.get("data_ok") and rec["train_score"] > best_score:
            best_score = rec["train_score"]
            best = c
    if best is None:
        raise RuntimeError("No viable training configuration")

    oos_trades, oos = backtest_prepared(test_market, best)
    if not oos.get("data_ok"):
        raise RuntimeError("Selected configuration produced no OOS trades")

    # Compare the checked-in paper candidate with the training-selected configuration.
    candidate = {
        "capital": cfg["capital"],
        "entry_weekday": cfg["entry_weekday"],
        "min_days_to_expiry": cfg["min_days_to_expiry"],
        "max_days_to_expiry": cfg["max_days_to_expiry"],
        "distance": cfg["selected_distance"],
        "width": cfg["selected_wing_width"],
        "take_profit": cfg["selected_take_profit"],
        "stop_loss": cfg["selected_stop_loss"],
    }

    oos.update({
        "split_date": str(split.date()),
        "selected_parameters": best,
        "paper_candidate_parameters": candidate,
        "paper_candidate_matches_selected": candidate == best,
        "dataset_start": str(data.date.min().date()),
        "dataset_end": str(data.date.max().date()),
        "dataset_rows": int(len(data)),
        "dataset_dates": int(len(dates)),
        "oos_start": str(test.date.min().date()),
        "oos_end": str(test.date.max().date()),
        "lookback_days": int((data.date.max() - data.date.min()).days),
    })

    leaderboard = pd.DataFrame(rows).sort_values("train_score", ascending=False).reset_index(drop=True)
    leaderboard.to_csv(out / "training_leaderboard.csv", index=False)
    oos_trades.to_csv(out / "oos_trades.csv", index=False)

    oos_trades["exit_date"] = pd.to_datetime(oos_trades["exit_date"])
    years = []
    for year, g in oos_trades.groupby(oos_trades.exit_date.dt.year):
        years.append({
            "year": int(year),
            "trades": int(len(g)),
            "win_rate": float((g.net_pnl > 0).mean()),
            "net_pnl": float(g.net_pnl.sum()),
            "fixed_lot_return": float(g.net_pnl.sum() / cfg["capital"]),
            "worst_trade": float(g.net_pnl.min()),
            "negative_intraday_bound_rate": float((g.conservative_intraday_pnl < 0).mean()),
        })
    pd.DataFrame(years).to_csv(out / "oos_year_breakdown.csv", index=False)

    profiles = leaderboard.groupby(["take_profit", "stop_loss"], as_index=False).agg(
        train_score_min=("train_score", "min"),
        train_score_max=("train_score", "max"),
        trades_min=("trades", "min"),
        trades_max=("trades", "max"),
        fixed_return_min=("fixed_lot_total_return", "min"),
        fixed_return_max=("fixed_lot_total_return", "max"),
    )
    profiles.to_csv(out / "tp_sl_behavior_audit.csv", index=False)

    audit = {
        "purpose": "Corrected Success V1 long-history research plus prospective paper candidate",
        "data_frequency": "daily EOD NSE F&O bhavcopy",
        "underlying_series": "nearest non-expired NIFTY index-futures close from the same NSE session",
        "chronology": "70/30 chronological train/OOS split; OOS is not used for parameter selection",
        "parameter_grid": len(rows),
        "entry_weekday_convention": "Python datetime.weekday(): Monday=0, Tuesday=1",
        "entry_weekday_used": int(cfg["entry_weekday"]),
        "dte_window": [int(cfg["min_days_to_expiry"]), int(cfg["max_days_to_expiry"])],
        "accounting": "fixed-lot cumulative equity plus separate theoretical per-trade compounding sensitivity",
        "execution": "EOD close exits; OHLC is a conservative path-risk bound only and does not prove intraday order execution",
        "paper_status": "prospective validation only",
        "paper_candidate_matches_training_selection": candidate == best,
        "candidate": candidate,
        "selected": best,
        "cost_model": "Paytm Money / India F&O modeled costs; compare with actual contract-note charges in paper validation",
    }
    (out / "backtest_audit.json").write_text(json.dumps(audit, indent=2))
    (out / "oos_summary.json").write_text(json.dumps(oos, indent=2, default=str))
    print(json.dumps(oos, indent=2, default=str))


if __name__ == "__main__":
    main()
