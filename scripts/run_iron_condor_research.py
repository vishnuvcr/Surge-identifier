from __future__ import annotations

import itertools
import json
from pathlib import Path

import pandas as pd
import yaml

from src.iron_condor_backtest import DEFAULT_CONFIG, backtest
from src.nse_options import load_nifty_options


def combinations(cfg):
    keys = ["short_distance_pct", "wing_width_points", "take_profit_credit_pct", "stop_loss_credit_multiple"]
    for vals in itertools.product(*(cfg[k] for k in keys)):
        x = dict(cfg)
        for k, v in zip(keys, vals):
            x[k] = [v]
        yield x


def rank(summary):
    if not summary.get("data_ok"):
        return -1e9
    # Favor positive weekly results, but explicitly penalize tail loss and drawdown.
    return (
        2.0 * summary["median_weekly_return"]
        + 1.0 * summary["avg_weekly_return"]
        + 0.75 * summary["weeks_at_or_above_5pct"] * 0.05
        + 0.25 * summary["weeks_at_or_above_0pct"]
        + 0.10 * summary["win_rate"]
        + 0.5 * summary["worst_weekly_return"]
        + 0.5 * summary["max_drawdown"]
    )


def main():
    cfg = DEFAULT_CONFIG.copy()
    cfg.update(yaml.safe_load(Path("config/iron_condor.yaml").read_text()) or {})
    out = Path("backtest/results/iron-condor")
    out.mkdir(parents=True, exist_ok=True)
    data = load_nifty_options("data/cache", lookback_days=600)
    if data.empty:
        raise RuntimeError("No NIFTY option data downloaded")
    data.to_parquet("data/cache/nifty_options.parquet", index=False)

    dates = sorted(pd.to_datetime(data["date"]).dt.normalize().unique())
    split = dates[max(1, int(len(dates) * 0.70))]
    train = data[data["date"] < split]
    test = data[data["date"] >= split]

    rows = []
    best_cfg = None
    best_score = -1e18
    for i, c in enumerate(combinations(cfg)):
        _, s = backtest(train, cfg=c)
        s["config_id"] = i
        s["short_distance_pct"] = c["short_distance_pct"][0]
        s["wing_width_points"] = c["wing_width_points"][0]
        s["take_profit_credit_pct"] = c["take_profit_credit_pct"][0]
        s["stop_loss_credit_multiple"] = c["stop_loss_credit_multiple"][0]
        s["train_score"] = rank(s)
        rows.append(s)
        if s.get("data_ok") and s["train_score"] > best_score:
            best_score = s["train_score"]
            best_cfg = c

    if best_cfg is None:
        raise RuntimeError("No viable iron condor configuration in training period")

    _, oos = backtest(test, cfg=best_cfg)
    oos.update({
        "selected_short_distance_pct": best_cfg["short_distance_pct"][0],
        "selected_wing_width_points": best_cfg["wing_width_points"][0],
        "selected_take_profit_credit_pct": best_cfg["take_profit_credit_pct"][0],
        "selected_stop_loss_credit_multiple": best_cfg["stop_loss_credit_multiple"][0],
        "split_date": str(split.date()),
    })

    leaderboard = pd.DataFrame(rows).sort_values("train_score", ascending=False)
    leaderboard.to_csv(out / "training_leaderboard.csv", index=False)
    (out / "oos_summary.json").write_text(json.dumps(oos, indent=2, default=str))
    pd.DataFrame()  # keep pandas imported for CI diagnostics
    print(json.dumps({"training_top": leaderboard.head(10).to_dict("records"), "oos": oos}, indent=2, default=str))


if __name__ == "__main__":
    main()
