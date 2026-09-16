from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Make repository-root imports deterministic for both local and GitHub Actions runs.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.iron_condor_engine_v1 import CostModel, lot_size_for_expiry

CFG = yaml.safe_load((ROOT / "config.iron_condor_v2.yaml").read_text())
CAPITAL = float(CFG["capital"])


class Market:
    def __init__(self, options: pd.DataFrame, futures: pd.DataFrame):
        o = options.copy()
        o["date"] = pd.to_datetime(o.date).dt.normalize()
        o["expiry"] = pd.to_datetime(o.expiry).dt.normalize()
        o["option_type"] = o.option_type.astype(str).str.upper()
        o = o.drop_duplicates(["date", "expiry", "strike", "option_type"], keep="last")

        f = futures.copy()
        f["date"] = pd.to_datetime(f.date).dt.normalize()
        f["expiry"] = pd.to_datetime(f.expiry).dt.normalize()
        f = f.sort_values(["date", "expiry"]).drop_duplicates(["date", "expiry"], keep="last")
        f = f[f.expiry >= f.date]

        # Use the nearest non-expired NIFTY index-future close as the underlying proxy.
        spot = f.sort_values(["date", "expiry"]).groupby("date", as_index=False).first()[["date", "close"]]
        self.spot = dict(zip(spot.date, spot.close.astype(float)))
        self.dates = tuple(pd.Timestamp(x) for x in sorted(set(o.date) & set(spot.date)))
        self.expiries = {}
        self.strikes = {}
        self.px = {}

        for (d, e), g in o.groupby(["date", "expiry"], sort=False):
            d, e = pd.Timestamp(d), pd.Timestamp(e)
            if d not in self.spot:
                continue
            self.expiries.setdefault(d, []).append(e)
            self.strikes[(d, e)] = np.asarray(
                sorted(pd.to_numeric(g.strike, errors="coerce").dropna().unique()), dtype=float
            )
        for d in self.expiries:
            self.expiries[d] = tuple(sorted(self.expiries[d]))

        for d, e, k, t, close, high, low in o[
            ["date", "expiry", "strike", "option_type", "close", "high", "low"]
        ].itertuples(index=False, name=None):
            key = (pd.Timestamp(d), pd.Timestamp(e), float(k), str(t).upper())
            self.px[key] = {
                "close": float(close) if pd.notna(close) else np.nan,
                "high": float(high) if pd.notna(high) else np.nan,
                "low": float(low) if pd.notna(low) else np.nan,
            }

    def price(self, d, e, k, t, field="close"):
        row = self.px.get((pd.Timestamp(d), pd.Timestamp(e), float(k), str(t).upper()))
        if not row:
            return None
        value = row.get(field, np.nan)
        return None if pd.isna(value) else float(value)

    def expiry_for(self, d):
        for expiry in self.expiries.get(pd.Timestamp(d), ()):
            dte = (expiry - pd.Timestamp(d)).days
            if int(CFG["min_days_to_expiry"]) <= dte <= int(CFG["max_days_to_expiry"]):
                return expiry
        return None

    def choose(self, signal_date, expiry, distance, width):
        spot = self.spot.get(pd.Timestamp(signal_date))
        strikes = self.strikes.get((pd.Timestamp(signal_date), pd.Timestamp(expiry)))
        if spot is None or strikes is None or len(strikes) < 8:
            return None

        puts = strikes[strikes <= spot * (1.0 - distance)]
        calls = strikes[strikes >= spot * (1.0 + distance)]
        if len(puts) == 0 or len(calls) == 0:
            return None

        put_short, call_short = float(puts[-1]), float(calls[0])
        put_longs = strikes[strikes <= put_short - width + 1e-9]
        call_longs = strikes[strikes >= call_short + width - 1e-9]
        if len(put_longs) == 0 or len(call_longs) == 0:
            return None
        return float(put_longs[-1]), put_short, call_short, float(call_longs[0])


def score(trades: pd.DataFrame) -> float:
    if trades.empty:
        return -1e9
    t = trades.sort_values("exit_date").copy()
    weekly = t.groupby(t.exit_date.dt.to_period("W"))["net_pnl"].sum() / CAPITAL
    equity = CAPITAL + t.net_pnl.cumsum()
    drawdown = equity / equity.cummax() - 1.0
    win = float((t.net_pnl > 0).mean())
    return (
        2.0 * float(weekly.median())
        + float(weekly.mean())
        + 0.10 * float((weekly >= 0.05).mean())
        + 0.25 * float((weekly >= 0).mean())
        + 0.10 * win
        + 0.50 * float(weekly.min())
        + 0.50 * float(drawdown.min())
    )


def run_config(market: Market, params: dict, start=None, end=None, slippage=None, collect=True):
    slip = float(CFG["selection_slippage_per_leg_points"] if slippage is None else slippage)
    min_credit = float(params.get("minimum_credit", 0.0))
    rows = []
    active_until = None

    dates = [
        d for d in market.dates
        if d.weekday() == int(params["entry_weekday"])
        and (start is None or d >= start)
        and (end is None or d <= end)
    ]

    for signal_date in dates:
        # No overlapping positions. The next signal must occur strictly after the prior exit.
        if active_until is not None and signal_date <= active_until:
            continue

        expiry = market.expiry_for(signal_date)
        if expiry is None:
            continue

        legs4 = market.choose(signal_date, expiry, float(params["distance"]), int(params["width"]))
        if legs4 is None:
            continue
        leg_keys = [(legs4[0], "PE"), (legs4[1], "PE"), (legs4[2], "CE"), (legs4[3], "CE")]

        # Delayed execution: signal is built from the completed signal session,
        # but the paper entry occurs at the next eligible session.
        entry_date = next((d for d in market.dates if d > signal_date and d <= expiry), None)
        if entry_date is None or (end is not None and entry_date > end):
            continue

        entry = [market.price(entry_date, expiry, k, t) for k, t in leg_keys]
        if any(v is None or v <= 0 for v in entry):
            continue

        raw_credit = entry[1] + entry[2] - entry[0] - entry[3]
        effective_credit = raw_credit - 4.0 * slip
        if raw_credit <= 0 or effective_credit < min_credit:
            continue

        lot = lot_size_for_expiry(expiry)
        max_loss_points = max(legs4[1] - legs4[0], legs4[3] - legs4[2]) - raw_credit
        if max_loss_points <= 0:
            continue

        exit_date = None
        reason = "expiry"
        exit_mark = None
        sim_days = [
            d for d in market.dates
            if entry_date < d <= expiry and (end is None or d <= end)
        ]
        for day in sim_days:
            values = [market.price(day, expiry, k, t) for k, t in leg_keys]
            if any(v is None for v in values):
                continue
            mark = values[1] + values[2] - values[0] - values[3]
            exit_debit = -mark + 4.0 * slip
            pnl_points = effective_credit - exit_debit
            tp_target_debit = (1.0 - float(params["take_profit"])) * effective_credit
            sl_debit = (1.0 + float(params["stop_loss"])) * effective_credit
            if exit_debit <= tp_target_debit:
                exit_date, reason, exit_mark = day, "take_profit", mark
                break
            if exit_debit >= sl_debit:
                exit_date, reason, exit_mark = day, "stop_loss", mark
                break

        # Hard research-window boundary: a position is included only if it can be
        # fully settled within the requested window. This prevents train/OOS contamination.
        if exit_date is None:
            if end is not None and expiry > end:
                continue
            exit_date = expiry
            values = [market.price(exit_date, expiry, k, t) for k, t in leg_keys]
            if any(v is None for v in values):
                continue
            exit_mark = values[1] + values[2] - values[0] - values[3]

        exit_px = [market.price(exit_date, expiry, k, t) for k, t in leg_keys]
        if any(v is None for v in exit_px):
            continue

        cost = CostModel().total(
            [
                (entry[0], exit_px[0], 1),
                (entry[1], exit_px[1], -1),
                (entry[2], exit_px[2], -1),
                (entry[3], exit_px[3], 1),
            ],
            lot,
        )
        gross = (effective_credit + exit_mark - 4.0 * slip) * lot
        net = gross - cost
        active_until = pd.Timestamp(exit_date)

        if collect:
            rows.append(
                {
                    "signal_date": pd.Timestamp(signal_date),
                    "entry_date": pd.Timestamp(entry_date),
                    "exit_date": pd.Timestamp(exit_date),
                    "expiry": pd.Timestamp(expiry),
                    "entry_weekday": int(params["entry_weekday"]),
                    "dte_at_signal": int((expiry - signal_date).days),
                    "dte_at_entry": int((expiry - entry_date).days),
                    "nifty_reference_signal": float(market.spot[signal_date]),
                    "nifty_reference_entry": float(market.spot.get(entry_date, np.nan)),
                    "put_long": legs4[0], "put_short": legs4[1],
                    "call_short": legs4[2], "call_long": legs4[3],
                    "put_long_entry": entry[0], "put_short_entry": entry[1],
                    "call_short_entry": entry[2], "call_long_entry": entry[3],
                    "put_long_exit": exit_px[0], "put_short_exit": exit_px[1],
                    "call_short_exit": exit_px[2], "call_long_exit": exit_px[3],
                    "raw_credit_points": raw_credit,
                    "effective_credit_points": effective_credit,
                    "minimum_credit": min_credit,
                    "slippage_per_leg": slip,
                    "wing_width": float(params["width"]),
                    "distance": float(params["distance"]),
                    "take_profit": float(params["take_profit"]),
                    "stop_loss": float(params["stop_loss"]),
                    "lot": int(lot),
                    "max_loss": float(max_loss_points * lot),
                    "lower_breakeven": float(legs4[1] - raw_credit),
                    "upper_breakeven": float(legs4[2] + raw_credit),
                    "gross_pnl": float(gross),
                    "costs": float(cost),
                    "net_pnl": float(net),
                    "return_on_capital": float(net / CAPITAL),
                    "return_on_max_loss": float(net / (max_loss_points * lot)),
                    "reason": reason,
                }
            )
    return pd.DataFrame(rows)


def summarize(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"trades": 0, "win_rate": 0.0, "net_pnl": 0.0, "return": 0.0, "max_drawdown": 0.0}
    t = t.sort_values(["exit_date", "entry_date"]).reset_index(drop=True)
    equity = CAPITAL + t.net_pnl.cumsum()
    drawdown = equity / equity.cummax() - 1.0
    return {
        "trades": int(len(t)),
        "win_rate": float((t.net_pnl > 0).mean()),
        "net_pnl": float(t.net_pnl.sum()),
        "return": float(t.net_pnl.sum() / CAPITAL),
        "median_trade": float(t.net_pnl.median()),
        "worst_trade": float(t.net_pnl.min()),
        "best_trade": float(t.net_pnl.max()),
        "max_drawdown": float(drawdown.min()),
        "take_profit_trades": int((t.reason == "take_profit").sum()),
        "stop_loss_trades": int((t.reason == "stop_loss").sum()),
        "expiry_trades": int((t.reason == "expiry").sum()),
    }


def select_config(market: Market, start, end):
    base = []
    for wd, distance, width, tp, sl in itertools.product(
        CFG["entry_weekdays"], CFG["short_distance_pct"], CFG["wing_width_points"],
        CFG["take_profit_credit_pct"], CFG["stop_loss_credit_multiple"]
    ):
        params = {
            "entry_weekday": wd, "distance": distance, "width": width,
            "take_profit": tp, "stop_loss": sl, "minimum_credit": 0.0,
        }
        trades = run_config(
            market, params, start, end,
            slippage=CFG["selection_slippage_per_leg_points"]
        )
        if len(trades) >= int(CFG["min_training_trades"]):
            base.append((score(trades), params, trades))

    if not base:
        raise RuntimeError(
            f"No base V2 configuration met min_training_trades={CFG['min_training_trades']} "
            f"for training window {start.date()} to {end.date()}"
        )

    base.sort(key=lambda x: x[0], reverse=True)
    top = base[:10]
    candidates = []
    for _, params, _ in top:
        for minimum_credit in CFG["minimum_credit_points"]:
            candidate = {**params, "minimum_credit": minimum_credit}
            trades = run_config(
                market, candidate, start, end,
                slippage=CFG["selection_slippage_per_leg_points"]
            )
            if len(trades) >= int(CFG["min_training_trades"]):
                candidates.append((score(trades), candidate, trades))

    if not candidates:
        raise RuntimeError(
            f"No V2 minimum-credit candidate met min_training_trades={CFG['min_training_trades']} "
            f"for training window {start.date()} to {end.date()}"
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0], base[:25]


def resolve_windows(dates):
    """Return exactly the predefined expanding walk-forward windows.

    Each test period is a full calendar year except the final 2026 YTD window.
    Training uses only dates strictly before each test start.
    """
    configured = CFG.get("walk_forward_schedule")
    if not configured:
        raise RuntimeError("walk_forward_schedule is required; refusing ambiguous equal-sized windows")

    available = set(pd.Timestamp(x) for x in dates)
    latest = max(dates)
    windows = []
    for i, spec in enumerate(configured, start=1):
        train_end = pd.Timestamp(spec["train_end"]).normalize()
        test_start = pd.Timestamp(spec["test_start"]).normalize()
        test_end = latest if str(spec["test_end"]).lower() == "latest" else pd.Timestamp(spec["test_end"]).normalize()
        if not (train_end < test_start <= test_end):
            raise ValueError(f"Invalid walk-forward window {i}")
        train_dates = [d for d in dates if d <= train_end]
        test_dates = [d for d in dates if test_start <= d <= test_end]
        if not train_dates or not test_dates:
            raise ValueError(f"Walk-forward window {i} has no usable train/test dates")
        # Keep a hard chronological gap: training ends before test starts.
        if max(train_dates) >= min(test_dates):
            raise AssertionError(f"Window {i} leaks across train/test boundary")
        windows.append((train_dates[0], train_dates[-1], test_dates[0], test_dates[-1]))
    if len(windows) != int(CFG["walk_forward_windows"]):
        raise ValueError("Configured walk-forward schedule count does not match walk_forward_windows")
    return windows


def walk_forward(market: Market):
    windows = resolve_windows(list(market.dates))
    results = []
    all_oos = []
    for i, (train_start, train_end, test_start, test_end) in enumerate(windows, start=1):
        selected, _ = select_config(market, train_start, train_end)
        selection_score, params, train_trades = selected
        oos = run_config(
            market, params, test_start, test_end,
            slippage=CFG["selection_slippage_per_leg_points"]
        )
        summary = summarize(oos)
        results.append(
            {
                "window": i,
                "train_start": str(train_start.date()),
                "train_end": str(train_end.date()),
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                "selection_score": float(selection_score),
                "training_trades": int(len(train_trades)),
                **params,
                **summary,
            }
        )
        if not oos.empty:
            oos = oos.copy()
            oos["window"] = i
            all_oos.append(oos)

    return pd.DataFrame(results), pd.concat(all_oos, ignore_index=True) if all_oos else pd.DataFrame()


def main():
    opt_path = ROOT / "data/cache/nifty_options_long.parquet"
    fut_path = ROOT / "data/cache/nifty_futures_long.parquet"
    if not opt_path.exists() or not fut_path.exists():
        raise FileNotFoundError("Run the existing NSE long-history downloader first")

    market = Market(pd.read_parquet(opt_path), pd.read_parquet(fut_path))
    dates = list(market.dates)
    if len(dates) < 1000:
        raise RuntimeError(f"Insufficient aligned market history: {len(dates)} dates")

    split = int(len(dates) * 0.70)
    train_start, train_end = dates[0], dates[split - 1]
    oos_start, oos_end = dates[split], dates[-1]

    # Primary 70/30 holdout: parameters selected only inside the 70% training set.
    selected, top = select_config(market, train_start, train_end)
    selection_score, params, train_trades = selected
    oos_trades = run_config(
        market, params, oos_start, oos_end,
        slippage=CFG["selection_slippage_per_leg_points"]
    )
    if oos_trades.empty:
        raise RuntimeError("70/30 OOS produced zero complete trades")

    slippage_rows = []
    for slippage in CFG["slippage_per_leg_points"]:
        trades = run_config(market, params, oos_start, oos_end, slippage=slippage)
        slippage_rows.append({"slippage_per_leg": slippage, **summarize(trades)})

    # Expanding walk-forward validation with a predefined chronological schedule.
    wf, wf_trades = walk_forward(market)
    if len(wf) != int(CFG["walk_forward_windows"]):
        raise RuntimeError("Walk-forward window count mismatch")
    if wf.empty or (wf["trades"] <= 0).any():
        raise RuntimeError("At least one walk-forward OOS window has zero trades")

    baseline = {
        "entry_weekday": 1, "distance": 0.0125, "width": 250,
        "take_profit": 0.40, "stop_loss": 1.0, "minimum_credit": 0.0,
    }
    baseline_trades = run_config(
        market, baseline, oos_start, oos_end,
        slippage=CFG["selection_slippage_per_leg_points"]
    )

    out = ROOT / "backtest/results/iron-condor-v2"
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "strategy": "Iron Condor V2 — daily adaptive / delayed entry / credit quality / no overlap",
        "data_start": str(dates[0].date()),
        "data_end": str(dates[-1].date()),
        "holdout": {
            "train_start": str(train_start.date()), "train_end": str(train_end.date()),
            "oos_start": str(oos_start.date()), "oos_end": str(oos_end.date()),
        },
        "selected_parameters": params,
        "selection_score": float(selection_score),
        "training": summarize(train_trades),
        "oos": summarize(oos_trades),
        "baseline_same_period": summarize(baseline_trades),
        "slippage_sensitivity": slippage_rows,
        "walk_forward_windows": int(len(wf)),
    }

    # JSON is written with allow_nan=False so invalid Infinity/NaN can never reach Pages.
    (out / "v2_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    pd.DataFrame(top, columns=["score", "params", "trades"]).to_json(
        out / "training_top25.json", orient="records", date_format="iso"
    )
    oos_trades.to_csv(out / "oos_trades.csv", index=False)
    baseline_trades.to_csv(out / "baseline_oos_trades.csv", index=False)
    wf.to_csv(out / "walk_forward_results.csv", index=False)
    wf_trades.to_csv(out / "walk_forward_trades.csv", index=False)
    pd.DataFrame(slippage_rows).to_csv(out / "slippage_sensitivity.csv", index=False)
    out.joinpath("oos_trades.json").write_text(
        oos_trades.to_json(orient="records", date_format="iso")
    )
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
