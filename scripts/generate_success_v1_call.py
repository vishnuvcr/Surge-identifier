from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pandas as pd
import yaml

from src.iron_condor_engine_v1 import CostModel, choose, lot_size_for_expiry, prepare_market

IST = timezone(timedelta(hours=5, minutes=30))
ROOT = Path(__file__).resolve().parents[1]
SIGNALS = ROOT / "signals"
SIGNALS.mkdir(parents=True, exist_ok=True)
OPEN_PATH = SIGNALS / "open_trade.json"
HISTORY_PATH = SIGNALS / "trade_history.csv"
MD_PATH = SIGNALS / "latest_trade_call.md"
JSON_PATH = SIGNALS / "latest_trade_call.json"


def load_cfg():
    return yaml.safe_load((ROOT / "config/success_v1.yaml").read_text())


def load_data():
    options_path = Path(os.environ.get("IRON_CONDOR_DATA_PATH", "data/cache/nifty_options_long.parquet"))
    futures_path = Path(os.environ.get("IRON_CONDOR_FUTURES_PATH", "data/cache/nifty_futures_long.parquet"))
    options = pd.read_parquet(options_path)
    futures = pd.read_parquet(futures_path)
    options["date"] = pd.to_datetime(options["date"]).dt.normalize()
    options["expiry"] = pd.to_datetime(options["expiry"]).dt.normalize()
    futures["date"] = pd.to_datetime(futures["date"]).dt.normalize()
    futures["expiry"] = pd.to_datetime(futures["expiry"]).dt.normalize()
    futures["dte"] = (futures["expiry"] - futures["date"]).dt.days
    futures = futures.loc[futures["dte"] >= 0].sort_values(["date", "dte"])
    futures = futures.groupby("date", as_index=False).first()[["date", "close"]].rename(columns={"close": "underlying_close"})
    options = options.merge(futures, on="date", how="left", validate="many_to_one")
    options["dte"] = (options["expiry"] - options["date"]).dt.days
    options = options.loc[options["dte"].between(0, 7)].drop(columns=["dte"])
    options = options.sort_values(["date", "expiry", "strike", "option_type"]).reset_index(drop=True)
    return options, futures


def current_trade_state():
    if not OPEN_PATH.exists():
        return None
    try:
        return json.loads(OPEN_PATH.read_text())
    except Exception:
        return None


def save_history(row):
    exists = HISTORY_PATH.exists()
    with HISTORY_PATH.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_outputs(call, status):
    payload = {"status": status, **call}
    JSON_PATH.write_text(json.dumps(payload, indent=2, default=str))
    lines = [
        "# SUCCESS V1 — NIFTY IRON CONDOR PAPER CALL",
        "",
        f"**Status:** {status}",
        f"**Generated (IST):** {call['generated_at_ist']}",
        f"**Market data date:** {call['data_date']}",
        "",
    ]
    for key in [
        "signal_id","underlying_close","expiry","dte","put_long","put_short","call_short","call_long",
        "lot","entry_credit","entry_credit_value","target_debit_to_close","stop_debit_to_close",
        "max_loss_value","breakeven_lower","breakeven_upper","model_profit_at_target","model_loss_at_stop",
        "entry_cost_estimate","estimated_roundtrip_cost_note","return_if_target","return_at_max_loss",
    ]:
        if key in call:
            lines.append(f"- **{key}:** {call[key]}")
    lines += [
        "",
        "## Execution checklist",
        "1. PAPER TEST ONLY. Do not treat this as a guaranteed or broker-approved recommendation.",
        "2. Verify all four option contracts exist and are tradable in Paytm Money before any paper entry.",
        "3. Record actual fill prices for each leg; do not replace fills with the EOD prices in this file.",
        "4. Track each leg separately and record all charges from the contract note.",
        "5. Exit at the model TP/SL/expiry rule only if the live paper test uses the same rule.",
        "",
        "## Entry legs",
        f"- BUY {call['lot']} x NIFTY {call['expiry']} {call['put_long']} PE",
        f"- SELL {call['lot']} x NIFTY {call['expiry']} {call['put_short']} PE",
        f"- SELL {call['lot']} x NIFTY {call['expiry']} {call['call_short']} CE",
        f"- BUY {call['lot']} x NIFTY {call['expiry']} {call['call_long']} CE",
        "",
        "This branch is a prospective paper-validation system. Historical backtest results do not guarantee future performance.",
    ]
    MD_PATH.write_text("\n".join(lines) + "\n")


def main():
    cfg = load_cfg()
    options, futures = load_data()
    latest_date = pd.Timestamp(options.date.max()).normalize()
    generated = datetime.now(IST).isoformat(timespec="seconds")
    existing = current_trade_state()

    # Update an existing paper position first.
    if existing and existing.get("status") == "OPEN":
        expiry = pd.Timestamp(existing["expiry"])
        legs = [(existing["put_long"],"PE"),(existing["put_short"],"PE"),(existing["call_short"],"CE"),(existing["call_long"],"CE")]
        row = options.loc[options.date.eq(latest_date)]
        price_map = {(float(r.strike), str(r.option_type)): float(r.close) for r in row.itertuples() if pd.notna(r.close)}
        vals = [price_map.get((float(k),t)) for k,t in legs]
        status = "OPEN"
        model_pnl_points = None
        close_reason = None
        if all(v is not None for v in vals):
            mark = vals[1] + vals[2] - vals[0] - vals[3]
            credit = float(existing["entry_credit"])
            model_pnl_points = credit + mark
            if model_pnl_points >= cfg["selected_take_profit"] * credit:
                status, close_reason = "MODEL_TP", "take_profit"
            elif model_pnl_points <= -cfg["selected_stop_loss"] * credit:
                status, close_reason = "MODEL_SL", "stop_loss"
            elif latest_date >= expiry:
                status, close_reason = "MODEL_EXPIRY", "expiry"
        call = dict(existing)
        call.update({
            "generated_at_ist": generated, "data_date": str(latest_date.date()),
            "current_option_prices": vals, "model_pnl_points": model_pnl_points,
            "status": status, "close_reason": close_reason,
        })
        if status != "OPEN":
            OPEN_PATH.unlink(missing_ok=True)
            save_history(call)
        write_outputs(call, status)
        return

    entry_weekday = int(cfg["entry_weekday"])
    weekday_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][entry_weekday]
    if latest_date.weekday() != entry_weekday:
        call = {
            "generated_at_ist": generated,
            "data_date": str(latest_date.date()),
            "entry_weekday": entry_weekday,
            "message": f"NO NEW ENTRY: latest available NSE session is not {weekday_name}.",
        }
        write_outputs(call, "NO_NEW_ENTRY")
        print(json.dumps({"status":"NO_NEW_ENTRY", **call}, indent=2))
        return

    # Prepare market for strike selection on the latest session.
    market = prepare_market(options)
    expiry = next((x for x in market.expiries_by_date.get(latest_date, ()) if cfg["min_days_to_expiry"] <= (x-latest_date).days <= cfg["max_days_to_expiry"]), None)
    if expiry is None:
        raise RuntimeError(f"No eligible NIFTY expiry in configured {cfg['min_days_to_expiry']}-{cfg['max_days_to_expiry']} DTE window")
    strikes = choose(market, latest_date, expiry, cfg["selected_distance"], cfg["selected_wing_width"])
    if strikes is None:
        raise RuntimeError("Could not construct the configured iron condor from the latest NSE option chain")
    pl, ps, cs, cl = strikes
    leg_tuples = [(pl,"PE"),(ps,"PE"),(cs,"CE"),(cl,"CE")]
    price = {(float(r.strike), str(r.option_type)): float(r.close) for r in options.loc[options.date.eq(latest_date)].itertuples() if pd.notna(r.close) and r.close > 0}
    ep = [price.get((k,t)) for k,t in leg_tuples]
    if any(x is None for x in ep):
        raise RuntimeError(f"Missing EOD price for one or more selected legs: {ep}")
    credit = ep[1] + ep[2] - ep[0] - ep[3]
    lot = lot_size_for_expiry(expiry)
    max_loss_points = max(ps-pl, cl-cs) - credit
    if credit <= 0 or max_loss_points <= 0:
        raise RuntimeError("Constructed condor has invalid credit/max-loss")

    # Entry-side cost estimate: four executed orders; statutory components are approximate.
    cm = CostModel()
    entry_turnover = sum(ep) * lot
    entry_sell = (ep[1] + ep[2]) * lot
    entry_buy = (ep[0] + ep[3]) * lot
    entry_brokerage = 4 * cm.brokerage_per_order
    entry_cost = entry_brokerage + max(entry_sell,0)*cm.stt_sell_pct + entry_buy*cm.stamp_buy_pct + entry_turnover*cm.exchange_txn_pct + entry_turnover*cm.sebi_turnover_pct + (entry_brokerage + entry_turnover*cm.exchange_txn_pct + entry_turnover*cm.sebi_turnover_pct)*cm.gst_pct

    signal_id = f"SV1-{latest_date:%Y%m%d}-{expiry:%Y%m%d}-{int(ps)}-{int(cs)}"
    call = {
        "signal_id": signal_id, "generated_at_ist": generated, "data_date": str(latest_date.date()),
        "underlying_close": float(market.spot[latest_date]), "expiry": str(expiry.date()), "dte": int((expiry-latest_date).days),
        "distance": cfg["selected_distance"], "wing_width": cfg["selected_wing_width"],
        "take_profit_credit_pct": cfg["selected_take_profit"], "stop_loss_credit_multiple": cfg["selected_stop_loss"],
        "put_long": pl, "put_short": ps, "call_short": cs, "call_long": cl, "lot": lot,
        "entry_put_long": ep[0], "entry_put_short": ep[1], "entry_call_short": ep[2], "entry_call_long": ep[3],
        "entry_credit": float(credit), "entry_credit_value": float(credit*lot),
        "target_debit_to_close": float(credit*(1-cfg["selected_take_profit"])),
        "stop_debit_to_close": float(credit*(1+cfg["selected_stop_loss"])),
        "max_loss_value": float(max_loss_points*lot),
        "breakeven_lower": float(ps-credit), "breakeven_upper": float(cs+credit),
        "model_profit_at_target": float(cfg["selected_take_profit"]*credit*lot),
        "model_loss_at_stop": float(-cfg["selected_stop_loss"]*credit*lot),
        "entry_cost_estimate": float(entry_cost),
        "estimated_roundtrip_cost_note": "Entry-side estimate only; final round-trip charges depend on actual exit fills and broker/exchange charges.",
        "return_if_target": float((cfg["selected_take_profit"]*credit*lot-entry_cost)/cfg["capital"]),
        "return_at_max_loss": float((-max_loss_points*lot-entry_cost)/cfg["capital"]),
        "status": "OPEN",
    }
    OPEN_PATH.write_text(json.dumps(call, indent=2, default=str))
    write_outputs(call, "NEW_PAPER_ENTRY")
    print(json.dumps(call, indent=2, default=str))


if __name__ == "__main__":
    main()
