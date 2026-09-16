from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "v3" / "config.yaml").read_text())
CAPITAL = float(CFG["capital"])


@dataclass(frozen=True)
class CostModel:
    brokerage_per_order: float = 10.0
    stt_sell_pct: float = 0.0010
    stamp_buy_pct: float = 0.00003
    sebi_turnover_pct: float = 0.000001
    exchange_txn_pct: float = 0.00035
    gst_pct: float = 0.18

    def total(self, legs, lot: int) -> float:
        # Four-leg entry + four-leg exit = 8 orders.
        brokerage = 8.0 * self.brokerage_per_order
        sell_value = sum(entry if side < 0 else exit for entry, exit, side in legs) * lot
        buy_value = sum(entry if side > 0 else exit for entry, exit, side in legs) * lot
        turnover = sum(abs(entry) + abs(exit) for entry, exit, _ in legs) * lot
        txn = turnover * self.exchange_txn_pct
        stt = max(sell_value, 0.0) * self.stt_sell_pct
        stamp = max(buy_value, 0.0) * self.stamp_buy_pct
        sebi = turnover * self.sebi_turnover_pct
        gst = (brokerage + txn + sebi) * self.gst_pct
        return brokerage + stt + stamp + sebi + txn + gst


def lot_size_for_expiry(expiry: pd.Timestamp) -> int:
    e = pd.Timestamp(expiry).normalize()
    # Historical NIFTY lot-size changes used in the prior research line.
    if e < pd.Timestamp("2015-10-30"):
        return 25
    if e < pd.Timestamp("2021-08-01"):
        return 75
    if e < pd.Timestamp("2024-05-02"):
        return 50
    if e < pd.Timestamp("2024-11-20"):
        return 25
    if e < pd.Timestamp("2026-01-06"):
        return 75
    return 65


class Market:
    def __init__(self, options: pd.DataFrame, futures: pd.DataFrame):
        o = options.copy()
        o["date"] = pd.to_datetime(o["date"]).dt.normalize()
        o["expiry"] = pd.to_datetime(o["expiry"]).dt.normalize()
        o["strike"] = pd.to_numeric(o["strike"], errors="coerce")
        o["option_type"] = o["option_type"].astype(str).str.upper()
        o = o.dropna(subset=["date", "expiry", "strike", "option_type"])
        o = o.sort_values(["date", "expiry", "strike", "option_type"])
        o = o.drop_duplicates(["date", "expiry", "strike", "option_type"], keep="last")

        f = futures.copy()
        f["date"] = pd.to_datetime(f["date"]).dt.normalize()
        f["expiry"] = pd.to_datetime(f["expiry"]).dt.normalize()
        f = f.sort_values(["date", "expiry"]).drop_duplicates(["date", "expiry"], keep="last")
        f = f[f["expiry"] >= f["date"]]
        spot = f.sort_values(["date", "expiry"]).groupby("date", as_index=False).first()[["date", "close"]]
        self.spot = dict(zip(spot["date"], spot["close"].astype(float)))

        self.dates = tuple(pd.Timestamp(x) for x in sorted(set(o["date"]) & set(self.spot)))
        self.expiries: dict[pd.Timestamp, tuple[pd.Timestamp, ...]] = {}
        self.strikes: dict[tuple[pd.Timestamp, pd.Timestamp], np.ndarray] = {}
        self.px: dict[tuple[pd.Timestamp, pd.Timestamp, float, str], dict[str, float]] = {}

        for (d, e), g in o.groupby(["date", "expiry"], sort=False):
            d, e = pd.Timestamp(d), pd.Timestamp(e)
            if d not in self.spot:
                continue
            self.expiries.setdefault(d, []).append(e)
            self.strikes[(d, e)] = np.asarray(sorted(g["strike"].dropna().unique()), dtype=float)
        for d in self.expiries:
            self.expiries[d] = tuple(sorted(self.expiries[d]))

        fields = ["date", "expiry", "strike", "option_type", "close", "high", "low"]
        for d, e, k, typ, close, high, low in o[fields].itertuples(index=False, name=None):
            key = (pd.Timestamp(d), pd.Timestamp(e), float(k), str(typ).upper())
            self.px[key] = {
                "close": float(close) if pd.notna(close) else math.nan,
                "high": float(high) if pd.notna(high) else math.nan,
                "low": float(low) if pd.notna(low) else math.nan,
            }

    def price(self, d, e, k, typ, field="close"):
        row = self.px.get((pd.Timestamp(d), pd.Timestamp(e), float(k), str(typ).upper()))
        if not row:
            return None
        value = row.get(field, math.nan)
        return None if pd.isna(value) else float(value)

    def next_session(self, d) -> pd.Timestamp | None:
        d = pd.Timestamp(d)
        idx = self._date_index.get(d)
        if idx is None or idx + 1 >= len(self.dates):
            return None
        return self.dates[idx + 1]

    def setup_index(self):
        self._date_index = {d: i for i, d in enumerate(self.dates)}

    def expiry_for(self, signal_date: pd.Timestamp) -> pd.Timestamp | None:
        for expiry in self.expiries.get(signal_date, ()):
            dte = int((expiry - signal_date).days)
            if int(CFG["min_days_to_expiry"]) <= dte <= int(CFG["max_days_to_expiry"]):
                return expiry
        return None

    def choose(self, signal_date: pd.Timestamp, expiry: pd.Timestamp, distance: float, width: int):
        spot = self.spot.get(signal_date)
        strikes = self.strikes.get((signal_date, expiry))
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

    def mark(self, day, expiry, legs):
        vals = [self.price(day, expiry, k, typ, "close") for k, typ in legs]
        if any(v is None for v in vals):
            return None
        return vals[1] + vals[2] - vals[0] - vals[3]

    def conservative_bound(self, day, expiry, legs):
        lows = [self.price(day, expiry, k, typ, "low") for k, typ in legs]
        highs = [self.price(day, expiry, k, typ, "high") for k, typ in legs]
        if any(v is None for v in lows + highs):
            return None
        # Most-adverse leg-wise combination. This is a risk bound, not a realizable
        # path or proof that every extreme occurred simultaneously.
        return lows[1] + lows[2] - highs[0] - highs[3]


def config_grid():
    for d, w, tp, sl in itertools.product(
        CFG["short_distance_pct"],
        CFG["wing_width_points"],
        CFG["take_profit_credit_pct"],
        CFG["stop_loss_credit_multiple"],
    ):
        yield {
            "distance": float(d),
            "width": int(w),
            "take_profit": float(tp),
            "stop_loss": float(sl),
            "minimum_credit": float(CFG["minimum_credit_points"]),
        }


def simulate_config(market: Market, params: dict, start: pd.Timestamp, end: pd.Timestamp, slippage_per_leg: float):
    decisions = []
    trades = []
    active_until: pd.Timestamp | None = None

    for signal_date in market.dates:
        if signal_date < start or signal_date > end:
            continue

        row = {
            "signal_date": signal_date,
            "entry_weekday": int(signal_date.weekday()),
            "status": "SKIP",
            "reason": None,
            **params,
        }

        if active_until is not None and signal_date <= active_until:
            row["reason"] = "ACTIVE_POSITION"
            decisions.append(row)
            continue

        expiry = market.expiry_for(signal_date)
        if expiry is None:
            row["reason"] = "NO_ELIGIBLE_EXPIRY"
            decisions.append(row)
            continue

        legs4 = market.choose(signal_date, expiry, params["distance"], params["width"])
        if legs4 is None:
            row["reason"] = "NO_VALID_STRIKES"
            decisions.append(row)
            continue

        entry_date = market.next_session(signal_date)
        if entry_date is None or entry_date > expiry or entry_date > end:
            row["reason"] = "NO_NEXT_SESSION_ENTRY"
            decisions.append(row)
            continue
        if entry_date == expiry:
            row["reason"] = "ENTRY_ON_EXPIRY_EXCLUDED"
            decisions.append(row)
            continue

        leg_keys = [(legs4[0], "PE"), (legs4[1], "PE"), (legs4[2], "CE"), (legs4[3], "CE")]
        entry_px = [market.price(entry_date, expiry, k, typ) for k, typ in leg_keys]
        if any(v is None or v <= 0 for v in entry_px):
            row["reason"] = "MISSING_ENTRY_PRICE"
            decisions.append(row)
            continue

        raw_credit = entry_px[1] + entry_px[2] - entry_px[0] - entry_px[3]
        effective_credit = raw_credit - 4.0 * slippage_per_leg
        if raw_credit <= 0 or effective_credit < params["minimum_credit"]:
            row["reason"] = "CREDIT_GATE"
            decisions.append(row)
            continue

        lot = lot_size_for_expiry(expiry)
        width_points = max(legs4[1] - legs4[0], legs4[3] - legs4[2])
        max_loss_points = width_points - raw_credit
        if max_loss_points <= 0:
            row["reason"] = "INVALID_MAX_LOSS"
            decisions.append(row)
            continue

        hold_days = [d for d in market.dates if entry_date < d <= expiry]
        hold_days = hold_days[: int(CFG["max_holding_sessions"])]
        if not hold_days:
            row["reason"] = "NO_EXIT_SESSION"
            decisions.append(row)
            continue

        exit_date = None
        exit_reason = None
        exit_mark = None
        minimum_bound = math.nan
        minimum_bound_day = None

        for day in hold_days:
            bound = market.conservative_bound(day, expiry, leg_keys)
            if bound is not None:
                bound_pnl_points = effective_credit + bound - 4.0 * slippage_per_leg
                if math.isnan(minimum_bound) or bound_pnl_points < minimum_bound:
                    minimum_bound = bound_pnl_points
                    minimum_bound_day = day

            mark = market.mark(day, expiry, leg_keys)
            if mark is None:
                continue
            pnl_points = effective_credit + mark - 4.0 * slippage_per_leg
            target_pnl_points = params["take_profit"] * effective_credit
            stop_pnl_points = -params["stop_loss"] * effective_credit
            if pnl_points >= target_pnl_points:
                exit_date, exit_reason, exit_mark = day, "take_profit_eod_mark", mark
                break
            if pnl_points <= stop_pnl_points:
                exit_date, exit_reason, exit_mark = day, "stop_loss_eod_mark", mark
                break

        if exit_date is None:
            exit_date = hold_days[-1]
            exit_reason = "max_holding_or_expiry"
            exit_mark = market.mark(exit_date, expiry, leg_keys)
            if exit_mark is None:
                row["reason"] = "MISSING_EXIT_PRICE"
                decisions.append(row)
                continue

        # A test-window trade must settle within the test window. Otherwise it would
        # leak information from the next walk-forward window.
        if exit_date > end:
            row["reason"] = "CROSSES_TEST_WINDOW"
            decisions.append(row)
            continue

        exit_px = [market.price(exit_date, expiry, k, typ) for k, typ in leg_keys]
        if any(v is None for v in exit_px):
            row["reason"] = "MISSING_EXIT_LEG_PRICE"
            decisions.append(row)
            continue

        cost = CostModel().total(
            [
                (entry_px[0], exit_px[0], +1),
                (entry_px[1], exit_px[1], -1),
                (entry_px[2], exit_px[2], -1),
                (entry_px[3], exit_px[3], +1),
            ],
            lot,
        )
        gross_points = effective_credit + exit_mark - 4.0 * slippage_per_leg
        gross_pnl = gross_points * lot
        net_pnl = gross_pnl - cost

        active_until = exit_date
        row.update({
            "status": "TRADE",
            "reason": "EXECUTED",
            "expiry": expiry,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "lot": lot,
            "raw_credit_points": raw_credit,
            "effective_credit_points": effective_credit,
            "minimum_credit": params["minimum_credit"],
            "gross_pnl": gross_pnl,
            "costs": cost,
            "net_pnl": net_pnl,
        })
        decisions.append(row)

        trades.append({
            **row,
            "dte_at_signal": int((expiry - signal_date).days),
            "dte_at_entry": int((expiry - entry_date).days),
            "nifty_reference_signal": float(market.spot[signal_date]),
            "nifty_reference_entry": float(market.spot.get(entry_date, math.nan)),
            "put_long": legs4[0],
            "put_short": legs4[1],
            "call_short": legs4[2],
            "call_long": legs4[3],
            "put_long_entry": entry_px[0],
            "put_short_entry": entry_px[1],
            "call_short_entry": entry_px[2],
            "call_long_entry": entry_px[3],
            "put_long_exit": exit_px[0],
            "put_short_exit": exit_px[1],
            "call_short_exit": exit_px[2],
            "call_long_exit": exit_px[3],
            "wing_width_points": width_points,
            "max_loss_points": max_loss_points,
            "max_loss": max_loss_points * lot,
            "lower_breakeven": legs4[1] - raw_credit,
            "upper_breakeven": legs4[2] + raw_credit,
            "return_on_capital": net_pnl / CAPITAL,
            "return_on_max_loss": net_pnl / (max_loss_points * lot),
            "eod_exit_reason": exit_reason,
            "conservative_intraday_min_pnl_points": minimum_bound,
            "conservative_intraday_min_pnl": minimum_bound * lot if not math.isnan(minimum_bound) else math.nan,
            "conservative_intraday_min_pnl_day": minimum_bound_day,
            "conservative_bound_is_not_fill": True,
        })

    return pd.DataFrame(decisions), pd.DataFrame(trades)


def metrics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "net_pnl": 0.0,
            "return_on_capital": 0.0,
            "median_trade": 0.0,
            "worst_trade": 0.0,
            "best_trade": 0.0,
            "profit_factor": None,
            "max_drawdown": 0.0,
            "negative_conservative_bound_trades": 0,
            "deep_conservative_bound_trades": 0,
        }
    t = trades.sort_values(["exit_date", "entry_date"]).reset_index(drop=True)
    equity = CAPITAL + t["net_pnl"].cumsum()
    dd = equity / equity.cummax() - 1.0
    gains = float(t.loc[t.net_pnl > 0, "net_pnl"].sum())
    losses = float(-t.loc[t.net_pnl < 0, "net_pnl"].sum())
    bound = t["conservative_intraday_min_pnl"]
    return {
        "trades": int(len(t)),
        "win_rate": float((t.net_pnl > 0).mean()),
        "net_pnl": float(t.net_pnl.sum()),
        "return_on_capital": float(t.net_pnl.sum() / CAPITAL),
        "median_trade": float(t.net_pnl.median()),
        "worst_trade": float(t.net_pnl.min()),
        "best_trade": float(t.net_pnl.max()),
        "profit_factor": float(gains / losses) if losses > 0 else None,
        "max_drawdown": float(dd.min()),
        "take_profit_eod_marks": int((t.eod_exit_reason == "take_profit_eod_mark").sum()),
        "stop_loss_eod_marks": int((t.eod_exit_reason == "stop_loss_eod_mark").sum()),
        "forced_expiry_or_max_hold": int((t.eod_exit_reason == "max_holding_or_expiry").sum()),
        "negative_conservative_bound_trades": int((bound < 0).sum()),
        "deep_conservative_bound_trades": int((bound <= -0.5 * t.max_loss).sum()),
    }


def selection_score(trades: pd.DataFrame) -> float:
    if len(trades) < int(CFG["minimum_training_trades"]):
        return -1e9
    t = trades.sort_values(["exit_date", "entry_date"]).reset_index(drop=True)
    returns = t["net_pnl"] / CAPITAL
    equity = CAPITAL + t["net_pnl"].cumsum()
    drawdown = equity / equity.cummax() - 1.0
    median = float(returns.median())
    mean = float(returns.mean())
    win = float((returns > 0).mean())
    worst = float(returns.min())
    dd = float(drawdown.min())
    return (
        float(CFG["score_median_weight"]) * median
        + float(CFG["score_mean_weight"]) * mean
        + float(CFG["score_winrate_weight"]) * win
        + float(CFG["score_worst_trade_weight"]) * worst
        + float(CFG["score_drawdown_weight"]) * dd
    )


def choose_best_config(market: Market, train_start: pd.Timestamp, train_end: pd.Timestamp):
    leaderboard = []
    best = None
    for params in config_grid():
        _, trades = simulate_config(
            market,
            params,
            train_start,
            train_end,
            slippage_per_leg=float(CFG["selection_slippage_per_leg_points"]),
        )
        if len(trades) < int(CFG["minimum_training_trades"]):
            continue
        sc = selection_score(trades)
        m = metrics(trades)
        record = {**params, "score": sc, **{f"train_{k}": v for k, v in m.items()}}
        leaderboard.append(record)
        if best is None or sc > best[0]:
            best = (sc, params, trades)

    if best is None:
        raise RuntimeError(
            f"No V3 configuration met minimum_training_trades={CFG['minimum_training_trades']} "
            f"for training window {train_start.date()} to {train_end.date()}"
        )
    lb = pd.DataFrame(leaderboard).sort_values(["score", "train_net_pnl"], ascending=False).reset_index(drop=True)
    return best[1], best[2], lb


def window_specs(latest: pd.Timestamp):
    for spec in CFG["walk_forward_reporting"]:
        start = pd.Timestamp(spec["start"]).normalize()
        end = latest if str(spec["end"]).lower() == "latest" else pd.Timestamp(spec["end"]).normalize()
        yield str(spec["name"]), start, end


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(obj).date())
    if isinstance(obj, (np.floating, float)):
        return None if (isinstance(obj, float) and math.isnan(obj)) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def main():
    options_path = ROOT / "data" / "cache" / "nifty_options.parquet"
    futures_path = ROOT / "data" / "cache" / "nifty_futures.parquet"
    if not options_path.exists() or not futures_path.exists():
        raise FileNotFoundError("Expected cached NSE option/futures parquet files are missing")

    options = pd.read_parquet(options_path)
    futures = pd.read_parquet(futures_path)
    market = Market(options, futures)
    market.setup_index()
    if len(market.dates) < int(CFG["initial_training_sessions"]) + 20:
        raise RuntimeError("Insufficient synchronized option/futures history for V3")

    dataset_start, dataset_end = market.dates[0], market.dates[-1]
    min_train_start = dataset_start
    session_index = {d: i for i, d in enumerate(market.dates)}

    all_oos_decisions = []
    all_oos_trades = []
    window_records = []
    leaderboards = []

    for window_name, test_start, test_end in window_specs(dataset_end):
        # Expanding training: the training end is the session immediately before the test start.
        test_dates = [d for d in market.dates if test_start <= d <= test_end]
        if not test_dates:
            continue
        first_test = test_dates[0]
        idx = session_index[first_test]
        if idx < int(CFG["initial_training_sessions"]):
            train_start = market.dates[0]
        else:
            train_start = market.dates[max(0, idx - int(CFG["initial_training_sessions"]))]
        train_end = market.dates[idx - 1]

        params, train_trades, leaderboard = choose_best_config(market, train_start, train_end)
        leaderboard["window"] = window_name
        leaderboards.append(leaderboard.head(25))

        decisions, oos_trades = simulate_config(
            market,
            params,
            test_start,
            test_end,
            slippage_per_leg=float(CFG["selection_slippage_per_leg_points"]),
        )
        decisions["window"] = window_name
        oos_trades["window"] = window_name
        oos_trades["selection_train_start"] = train_start
        oos_trades["selection_train_end"] = train_end
        oos_trades["selected_score"] = selection_score(train_trades)

        all_oos_decisions.append(decisions)
        all_oos_trades.append(oos_trades)
        window_records.append({
            "window": window_name,
            "train_start": train_start,
            "train_end": train_end,
            "oos_start": test_start,
            "oos_end": test_end,
            "selected_score": selection_score(train_trades),
            "selected_parameters": params,
            "training": metrics(train_trades),
            "oos": metrics(oos_trades),
            "decision_days": int(len(decisions)),
            "executed_days": int((decisions.status == "TRADE").sum()),
            "trade_coverage": float((decisions.status == "TRADE").mean()) if len(decisions) else 0.0,
        })

    if not all_oos_trades:
        raise RuntimeError("No OOS windows produced results")

    decisions_all = pd.concat(all_oos_decisions, ignore_index=True)
    oos = pd.concat(all_oos_trades, ignore_index=True).sort_values(["exit_date", "entry_date"]).reset_index(drop=True)

    equity = CAPITAL + oos["net_pnl"].cumsum()
    oos["oos_fixed_lot_equity"] = equity
    oos["oos_fixed_lot_drawdown"] = equity / equity.cummax() - 1.0

    stress_rows = []
    final_params = window_records[-1]["selected_parameters"]
    # Stress each walk-forward window with the same selected parameters and varying execution slippage.
    for slip in CFG["stress_slippage_per_leg_points"]:
        stress_trades = []
        for wr in window_records:
            _, t = simulate_config(
                market,
                wr["selected_parameters"],
                pd.Timestamp(wr["oos_start"]),
                pd.Timestamp(wr["oos_end"]),
                slippage_per_leg=float(slip),
            )
            if not t.empty:
                t["window"] = wr["window"]
                stress_trades.append(t)
        combined = pd.concat(stress_trades, ignore_index=True) if stress_trades else pd.DataFrame()
        stress_rows.append({"slippage_per_leg": float(slip), **metrics(combined)})

    out = ROOT / "v3" / "research"
    out.mkdir(parents=True, exist_ok=True)
    decisions_all.to_csv(out / "daily_decisions.csv", index=False)
    oos.to_csv(out / "oos_trade_ledger.csv", index=False)
    pd.DataFrame(window_records).to_json(out / "window_summary.json", orient="records", indent=2, date_format="iso")
    pd.DataFrame(window_records).to_csv(out / "window_summary.csv", index=False)
    pd.concat(leaderboards, ignore_index=True).to_csv(out / "training_leaderboards_top25.csv", index=False)

    if "conservative_intraday_min_pnl" in oos.columns:
        oos[[
            "window", "signal_date", "entry_date", "exit_date", "expiry",
            "net_pnl", "max_loss", "conservative_intraday_min_pnl",
            "conservative_intraday_min_pnl_day", "conservative_bound_is_not_fill",
        ]].to_csv(out / "path_risk_audit.csv", index=False)

    pd.DataFrame(stress_rows).to_csv(out / "slippage_stress.csv", index=False)

    summary = {
        "strategy": "Iron Condor V3 - daily-decision walk-forward",
        "methodology": {
            "dataset_start": dataset_start,
            "dataset_end": dataset_end,
            "expanding_training": True,
            "daily_decision_days": int(len(decisions_all)),
            "daily_executed_trades": int(len(oos)),
            "entry_execution": "next available session EOD reference after signal-day close",
            "exit_execution": "EOD option mark only; no intraday fills assumed",
            "strike_selection": "signal-day NIFTY reference with fixed distance and wing-width parameters",
            "expiry_selection": "nearest actual available expiry with configured DTE range",
            "historical_lot_sizes": True,
            "modeled_costs": True,
            "selection_slippage_per_leg_points": float(CFG["selection_slippage_per_leg_points"]),
            "path_risk_note": "Leg-wise OHLC adverse bounds are reported as risk audits only; they do not prove TP/SL fills or simultaneity.",
            "test_contamination_protection": "Trades crossing an OOS reporting-window boundary are excluded rather than settled using future-window information.",
        },
        "overall_oos": metrics(oos),
        "oos_start": min(pd.Timestamp(w["oos_start"]) for w in window_records),
        "oos_end": max(pd.Timestamp(w["oos_end"]) for w in window_records),
        "windows": window_records,
        "slippage_stress": stress_rows,
        "final_window_selected_parameters": final_params,
        "artifact_files": {
            "daily_decisions": "v3/research/daily_decisions.csv",
            "oos_trade_ledger": "v3/research/oos_trade_ledger.csv",
            "window_summary": "v3/research/window_summary.csv",
            "training_leaderboards": "v3/research/training_leaderboards_top25.csv",
            "path_risk_audit": "v3/research/path_risk_audit.csv",
            "slippage_stress": "v3/research/slippage_stress.csv",
        },
    }
    (out / "daily_walk_forward_summary.json").write_text(json.dumps(json_safe(summary), indent=2))
    print(json.dumps(json_safe(summary), indent=2))


if __name__ == "__main__":
    main()
