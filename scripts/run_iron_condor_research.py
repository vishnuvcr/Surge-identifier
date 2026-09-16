from __future__ import annotations

import itertools
import json
from pathlib import Path

import pandas as pd
import yaml

from src.iron_condor_engine import backtest
from src.nse_options import load_nifty_options


def combinations(cfg):
    for distance, width, tp, sl in itertools.product(
        cfg["short_distance_pct"], cfg["wing_width_points"], cfg["take_profit_credit_pct"], cfg["stop_loss_credit_multiple"]
    ):
        yield {"capital": cfg["capital"], "entry_weekday": cfg["entry_weekday"], "distance": distance, "width": width, "take_profit": tp, "stop_loss": sl}


def rank(s):
    if not s.get("data_ok"):
        return -1e9
    return 2*s["median_weekly_return"] + s["avg_weekly_return"] + 0.05*s["weeks_at_or_above_5pct"] + 0.25*s["weeks_nonnegative"] + 0.10*s["win_rate"] + 0.5*s["worst_weekly_return"] + 0.5*s["max_drawdown"]


def main():
    cfg = yaml.safe_load(Path("config/iron_condor.yaml").read_text())
    out = Path("backtest/results/iron-condor")
    out.mkdir(parents=True, exist_ok=True)
    data = load_nifty_options("data/cache", lookback_days=600)
    if data.empty:
        raise RuntimeError("No NIFTY option data downloaded")
    data.to_parquet("data/cache/nifty_options.parquet", index=False)
    dates = sorted(pd.to_datetime(data.date).dt.normalize().unique())
    split = dates[max(1, int(len(dates) * .70))]
    train = data[data.date < split]
    test = data[data.date >= split]

    rows = []
    best = None
    best_score = -1e18
    for i, c in enumerate(combinations(cfg)):
        _, s = backtest(train, c)
        rec = {**s, "config_id": i, **c, "train_score": rank(s)}
        rows.append(rec)
        if s.get("data_ok") and rec["train_score"] > best_score:
            best_score = rec["train_score"]
            best = c
    if best is None:
        raise RuntimeError("No viable iron condor configuration in training period")

    _, oos = backtest(test, best)
    oos.update({"split_date": str(split.date()), "selected_parameters": best})
    leaderboard = pd.DataFrame(rows).sort_values("train_score", ascending=False)
    leaderboard.to_csv(out / "training_leaderboard.csv", index=False)
    (out / "oos_summary.json").write_text(json.dumps(oos, indent=2, default=str))
    print(json.dumps({"top_training_configs": leaderboard.head(10).to_dict("records"), "oos": oos}, indent=2, default=str))


if __name__ == "__main__":
    main()
