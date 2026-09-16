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

from src.iron_condor_engine import backtest_prepared, prepare_market
from src.nse_options import _download_nifty_spot_proxy


def combinations(cfg):
    for distance, width, tp, sl in itertools.product(
        cfg["short_distance_pct"],
        cfg["wing_width_points"],
        cfg["take_profit_credit_pct"],
        cfg["stop_loss_credit_multiple"],
    ):
        yield {
            "capital": cfg["capital"],
            "entry_weekday": cfg["entry_weekday"],
            "distance": distance,
            "width": width,
            "take_profit": tp,
            "stop_loss": sl,
        }


def rank(s):
    if not s.get("data_ok"):
        return -1e9
    return (
        2.0 * s["median_weekly_return"]
        + s["avg_weekly_return"]
        + 0.10 * s["weeks_at_or_above_5pct"]
        + 0.25 * s["weeks_nonnegative"]
        + 0.10 * s["win_rate"]
        + 0.50 * s["worst_weekly_return"]
        + 0.25 * s["compounded_return"]
        + 0.50 * s["max_drawdown"]
    )


def _load_long_dataset(path: Path) -> pd.DataFrame:
    data = pd.read_parquet(path)
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    data["expiry"] = pd.to_datetime(data["expiry"]).dt.normalize()
    data = data.loc[data["option_type"].isin(["CE", "PE"])].copy()
    if data.empty:
        raise RuntimeError("Long-history parquet contains no CE/PE NIFTY options")

    # Add an independent daily NIFTY spot series for strike selection.
    proxy = _download_nifty_spot_proxy(data.date.min().date(), data.date.max().date())
    if proxy.empty:
        raise RuntimeError("Unable to obtain NIFTY spot proxy for long-history backtest")
    data = data.merge(proxy, on="date", how="left", validate="many_to_one")
    coverage = float(data["underlying_close"].notna().mean())
    if coverage < 0.995:
        raise RuntimeError(f"NIFTY spot coverage too low: {coverage:.4f}")

    # The strategy enters on Tuesday and chooses a contract 1-7 calendar days
    # from entry. Once selected, only that same near-term expiry is needed on
    # subsequent daily marks. Keeping only 0-7 DTE rows dramatically reduces
    # memory while preserving every price required by the strategy.
    data["dte"] = (data["expiry"] - data["date"]).dt.days
    data = data.loc[data["dte"].between(0, 7)].copy()

    # Keep enough strikes to cover the whole configured search grid, including
    # the far wing. The extra 1% cushion protects against strike-grid gaps.
    max_distance = max(cfg_distance for cfg_distance in [0.03])
    max_width = 250.0
    rel_band = max_distance + max_width / data["underlying_close"] + 0.01
    data["rel_strike_distance"] = (data["strike"] / data["underlying_close"] - 1.0).abs()
    data = data.loc[data["rel_strike_distance"] <= rel_band]
    data = data.drop(columns=["dte", "rel_strike_distance"])

    if data.empty:
        raise RuntimeError("Long-history filter removed all relevant weekly contracts")
    return data.sort_values(["date", "expiry", "strike", "option_type"]).reset_index(drop=True)


def main():
    cfg = yaml.safe_load(Path("config/iron_condor.yaml").read_text())
    out = Path("backtest/results/iron-condor")
    out.mkdir(parents=True, exist_ok=True)

    data_path = Path(os.environ.get("IRON_CONDOR_DATA_PATH", "data/cache/nifty_options_long.parquet"))
    data = _load_long_dataset(data_path)
    data.to_parquet("data/cache/nifty_options.parquet", index=False)

    dates = sorted(data.date.unique())
    if len(dates) < 500:
        raise RuntimeError(f"Too few dates for long-history research: {len(dates)}")
    split_index = min(len(dates) - 1, max(1, int(len(dates) * 0.70)))
    split = pd.Timestamp(dates[split_index])
    train = data[data.date < split].copy()
    test = data[data.date >= split].copy()
    print(f"Filtered dataset: {len(data):,} rows, {len(dates):,} dates, {data.date.min().date()} -> {data.date.max().date()}")
    print(f"70/30 split: {split.date()} | train={train.date.min().date()}->{train.date.max().date()} | oos={test.date.min().date()}->{test.date.max().date()}")

    train_market = prepare_market(train)
    test_market = prepare_market(test)

    rows = []
    best = None
    best_score = -1e18
    for i, c in enumerate(combinations(cfg)):
        _, s = backtest_prepared(train_market, c)
        rec = {**s, "config_id": i, **c, "train_score": rank(s)}
        rows.append(rec)
        if s.get("data_ok") and rec["train_score"] > best_score:
            best_score = rec["train_score"]
            best = c
    if best is None:
        raise RuntimeError("No viable iron condor configuration in training period")

    oos_trades, oos = backtest_prepared(test_market, best)
    if not oos.get("data_ok"):
        raise RuntimeError("Selected configuration produced no OOS trades")

    oos.update({
        "split_date": str(split.date()),
        "selected_parameters": best,
        "dataset_start": str(data.date.min().date()),
        "dataset_end": str(data.date.max().date()),
        "dataset_rows": int(len(data)),
        "dataset_dates": int(len(dates)),
        "oos_start": str(test.date.min().date()),
        "oos_end": str(test.date.max().date()),
        "lookback_days": int((data.date.max() - data.date.min()).days),
        "oos_trades": int(oos.get("trades", 0)),
        "oos_simple_return": float(oos.get("simple_total_return", 0.0)),
        "oos_compounded_return": float(oos.get("compounded_return", 0.0)),
    })

    leaderboard = pd.DataFrame(rows).sort_values("train_score", ascending=False).reset_index(drop=True)
    leaderboard.to_csv(out / "training_leaderboard.csv", index=False)
    oos_trades.to_csv(out / "oos_trades.csv", index=False)

    oos_trades["exit_date"] = pd.to_datetime(oos_trades["exit_date"])
    year_rows = []
    for year, g in oos_trades.groupby(oos_trades["exit_date"].dt.year):
        year_rows.append({
            "year": int(year),
            "trades": int(len(g)),
            "win_rate": float((g.net_pnl > 0).mean()),
            "net_pnl": float(g.net_pnl.sum()),
            "simple_return": float(g.net_pnl.sum() / cfg["capital"]),
            "worst_trade": float(g.net_pnl.min()),
            "conservative_intraday_negative_bound_rate": float((g.conservative_intraday_pnl < 0).mean()),
        })
    pd.DataFrame(year_rows).to_csv(out / "oos_year_breakdown.csv", index=False)

    audit = {
        "data_frequency": "daily EOD NSE F&O bhavcopy; intraday path is not observable",
        "spot_series": "Yahoo Finance ^NSEI daily close used as an independent spot proxy",
        "history": "NSE NIFTY index-option bhavcopy history from 2015-01-01 through the latest available date",
        "selection": "512 configurations selected using 70% chronological training segment only",
        "oos": "remaining 30% chronological segment",
        "execution": "entry/exit uses daily close prices with modeled transaction costs",
        "intraday_bound": "conservative same-day OHLC bound; four leg extremes are not assumed simultaneous",
        "lot_size": "historical NIFTY lot-size schedule applied by option expiry vintage",
        "important": "The compounded return is the portfolio-equity metric; simple_total_return is the arithmetic sum of trade PnL divided by initial capital.",
    }
    (out / "backtest_audit.json").write_text(json.dumps(audit, indent=2))
    (out / "oos_summary.json").write_text(json.dumps(oos, indent=2, default=str))
    print(json.dumps({"top_training_configs": leaderboard.head(10).to_dict("records"), "oos": oos}, indent=2, default=str))


if __name__ == "__main__":
    main()
