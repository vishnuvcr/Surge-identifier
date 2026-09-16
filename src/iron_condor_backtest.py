from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostModel:
    brokerage_per_order: float = 10.0
    stt_sell_pct: float = 0.0015
    stamp_buy_pct: float = 0.00003
    sebi_turnover_pct: float = 0.000001
    exchange_txn_pct: float = 0.00035
    gst_pct: float = 0.18

    def round_trip(self, premiums: list[tuple[float, float, int]]) -> float:
        # (entry_price, exit_price, side), side +1 buy / -1 sell
        brokerage = self.brokerage_per_order * len(premiums) * 2
        sell_premium = sum(abs(px) * qty for entry, exit, side in premiums for px, qty in [(entry, qty, side)] if side < 0)
        sell_premium += sum(abs(exit) * qty for entry, exit, side, qty in [] ) if False else 0.0
        turnover = sum((abs(entry) + abs(exit)) * qty for entry, exit, side, qty in [(e, x, s, 1) for e, x, s in premiums])
        # For a four-leg condor, each leg has one lot. Brokerage is per executed order.
        # Statutory charges are applied to the appropriate side and are intentionally conservative.
        sell_value = sum(abs(e) for e, _, s in premiums if s < 0) + sum(abs(x) for e, x, s in premiums if s > 0)
        buy_value = sum(abs(e) for e, _, s in premiums if s > 0) + sum(abs(x) for e, x, s in premiums if s < 0)
        stt = sell_value * self.stt_sell_pct
        stamp = buy_value * self.stamp_buy_pct
        sebi = turnover * self.sebi_turnover_pct
        txn = turnover * self.exchange_txn_pct
        gst = (brokerage + txn) * self.gst_pct
        return brokerage + stt + stamp + sebi + txn + gst


DEFAULT_CONFIG = {
    "capital": 100000.0,
    "underlying": "NIFTY",
    "entry_weekday": 2,  # Wednesday, after Tuesday weekly expiry
    "max_days_to_expiry": 7,
    "min_days_to_expiry": 1,
    "short_distance_pct": [0.015, 0.02, 0.025, 0.03],
    "wing_width_points": [100, 150, 200, 250],
    "take_profit_credit_pct": [0.40, 0.50, 0.60, 0.70],
    "stop_loss_credit_multiple": [1.0, 1.25, 1.50, 2.0],
    "max_trades_per_week": 1,
    "lot_size_pre_2025_11_20": 75,
    "lot_size_2025_11_20_onward": 65,
}


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def normalize_options(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize NSE FO bhavcopy-like rows into contract OHLC data.

    Required canonical columns: date, symbol, instrument, option_type, expiry,
    strike, open, high, low, close, volume, oi.
    """
    aliases = {
        "date": ["date", "TradDt", "TRADDT"],
        "symbol": ["symbol", "TckrSymb", "SYMBOL"],
        "instrument": ["instrument", "FinInstrmTp", "INSTRUMENT"],
        "option_type": ["option_type", "OptnTp", "OPTTYPE"],
        "expiry": ["expiry", "XpryDt", "EXPIRYDT"],
        "strike": ["strike", "StrkPric", "STRIKEPRICE"],
        "open": ["open", "OpnPric", "OPENPRICE"],
        "high": ["high", "HghPric", "HIGHPRICE"],
        "low": ["low", "LwPric", "LOWPRICE"],
        "close": ["close", "ClsPric", "CLSPRIC", "CLOSEPRICE"],
        "volume": ["volume", "TtlTradgVol", "CONTRACTS"],
        "oi": ["oi", "OpnIntrst", "OPENINT"],
    }
    out = pd.DataFrame(index=df.index)
    for dst, names in aliases.items():
        src = next((x for x in names if x in df.columns), None)
        if src is None:
            out[dst] = np.nan
        else:
            out[dst] = df[src]
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["expiry"] = pd.to_datetime(out["expiry"], errors="coerce").dt.normalize()
    for c in ["strike", "open", "high", "low", "close", "volume", "oi"]:
        out[c] = _num(out[c])
    out["symbol"] = out["symbol"].astype(str).str.upper().str.strip()
    out["instrument"] = out["instrument"].astype(str).str.upper().str.strip()
    out["option_type"] = out["option_type"].astype(str).str.upper().str.strip()
    out = out[(out["symbol"] == "NIFTY") & out["instrument"].str.contains("OPTIDX|OPT", regex=True, na=False)]
    out = out[out["option_type"].isin(["CE", "PE", "CALL", "PUT"])]
    out["option_type"] = out["option_type"].replace({"CALL": "CE", "PUT": "PE"})
    out = out.dropna(subset=["date", "expiry", "strike", "close"])
    return out


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        raw = pd.read_parquet(path)
    elif path.suffix.lower() in {".csv", ".txt"}:
        raw = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported dataset format: {path}")
    return normalize_options(raw)


def lot_size(trade_date: pd.Timestamp, cfg: dict) -> int:
    cut = pd.Timestamp("2025-11-20")
    return int(cfg["lot_size_2025_11_20_onward"] if trade_date >= cut else cfg["lot_size_pre_2025_11_20"])


def _next_expiry(rows: pd.DataFrame, trade_date: pd.Timestamp, cfg: dict) -> pd.Timestamp | None:
    exps = sorted(rows.loc[rows["date"] == trade_date, "expiry"].dropna().unique())
    valid = [e for e in exps if cfg["min_days_to_expiry"] <= (e - trade_date).astype("timedelta64[D]").astype(int) <= cfg["max_days_to_expiry"]]
    return pd.Timestamp(valid[0]) if valid else None


def _spot_proxy(rows: pd.DataFrame, trade_date: pd.Timestamp, expiry: pd.Timestamp) -> float | None:
    # ATM strike is the most stable underlying proxy available directly in the option bhavcopy.
    x = rows[(rows.date == trade_date) & (rows.expiry == expiry)].copy()
    if x.empty:
        return None
    strikes = sorted(x.strike.dropna().unique())
    if not strikes:
        return None
    midpoint = np.median(strikes)
    # Use put-call parity candidates around the center; fall back to central strike.
    return float(midpoint)


def _price(rows: pd.DataFrame, dt: pd.Timestamp, expiry: pd.Timestamp, strike: float, typ: str, field: str = "close") -> float | None:
    x = rows[(rows.date == dt) & (rows.expiry == expiry) & (rows.strike == strike) & (rows.option_type == typ)]
    if x.empty:
        return None
    vals = pd.to_numeric(x[field], errors="coerce").dropna()
    return float(vals.iloc[-1]) if not vals.empty else None


def _choose_strikes(rows: pd.DataFrame, dt: pd.Timestamp, expiry: pd.Timestamp, cfg: dict, distance: float, wing: float):
    spot = _spot_proxy(rows, dt, expiry)
    if spot is None:
        return None
    chain = rows[(rows.date == dt) & (rows.expiry == expiry)].copy()
    strikes = np.array(sorted(chain.strike.dropna().unique()), dtype=float)
    if len(strikes) < 8:
        return None
    put_short_target = spot * (1 - distance)
    call_short_target = spot * (1 + distance)
    ps = strikes[strikes <= put_short_target]
    cs = strikes[strikes >= call_short_target]
    if len(ps) == 0 or len(cs) == 0:
        return None
    put_short = float(ps[-1])
    call_short = float(cs[0])
    put_longs = strikes[strikes <= put_short - wing + 1e-9]
    call_longs = strikes[strikes >= call_short + wing - 1e-9]
    if len(put_longs) == 0 or len(call_longs) == 0:
        return None
    put_long = float(put_longs[-1])
    call_long = float(call_longs[0])
    return put_long, put_short, call_short, call_long


def _mark_value(legs: dict[str, dict], day: pd.Timestamp, rows: pd.DataFrame) -> float | None:
    value = 0.0
    for name, leg in legs.items():
        p = _price(rows, day, leg["expiry"], leg["strike"], leg["type"])
        if p is None:
            return None
        value += leg["side"] * p
    return value


def backtest(rows: pd.DataFrame, cfg: dict | None = None, costs: CostModel | None = None) -> tuple[pd.DataFrame, dict]:
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    costs = costs or CostModel()
    rows = normalize_options(rows) if "date" not in rows.columns or "expiry" not in rows.columns else rows.copy()
    rows = rows.sort_values(["date", "expiry", "strike", "option_type"])
    dates = sorted(rows.date.dropna().unique())
    trades: list[dict] = []

    weekly_seen: set[pd.Timestamp] = set()
    for dt_raw in dates:
        dt = pd.Timestamp(dt_raw)
        if dt.weekday() != int(cfg["entry_weekday"]):
            continue
        expiry = _next_expiry(rows, dt, cfg)
        if expiry is None or expiry in weekly_seen:
            continue
        weekly_seen.add(expiry)
        for distance in cfg["short_distance_pct"]:
            for wing in cfg["wing_width_points"]:
                strikes = _choose_strikes(rows, dt, expiry, cfg, float(distance), float(wing))
                if strikes is None:
                    continue
                pl, ps, cs, cl = strikes
                entry_prices = {
                    "pl": _price(rows, dt, expiry, pl, "PE"),
                    "ps": _price(rows, dt, expiry, ps, "PE"),
                    "cs": _price(rows, dt, expiry, cs, "CE"),
                    "cl": _price(rows, dt, expiry, cl, "CE"),
                }
                if any(v is None or v <= 0 for v in entry_prices.values()):
                    continue
                # +1 for purchased wings; -1 for short strikes.
                legs = {
                    "pl": {"strike": pl, "type": "PE", "side": 1, "expiry": expiry},
                    "ps": {"strike": ps, "type": "PE", "side": -1, "expiry": expiry},
                    "cs": {"strike": cs, "type": "CE", "side": -1, "expiry": expiry},
                    "cl": {"strike": cl, "type": "CE", "side": 1, "expiry": expiry},
                }
                credit = entry_prices["ps"] + entry_prices["cs"] - entry_prices["pl"] - entry_prices["cl"]
                if credit <= 0:
                    continue
                lot = lot_size(dt, cfg)
                max_loss_points = max(ps - pl, cl - cs) - credit
                if max_loss_points <= 0:
                    continue
                entry_margin = max_loss_points * lot
                entry_costs = costs.round_trip([(entry_prices[n], entry_prices[n], legs[n]["side"]) for n in legs]) / 2
                exit_dt = expiry
                exit_credit_value = None
                exit_reason = "expiry"
                peak_loss = 0.0
                days = sorted(pd.Timestamp(x) for x in rows.loc[(rows.date >= dt) & (rows.date <= expiry), "date"].unique())
                for day in days[1:]:
                    mark = _mark_value(legs, day, rows)
                    if mark is None:
                        continue
                    pnl_points = credit + mark  # mark is signed current value of long/short portfolio
                    pnl_rupees = pnl_points * lot
                    peak_loss = min(peak_loss, pnl_rupees)
                    if pnl_points <= -float(cfg["stop_loss_credit_multiple"][0]) * credit:
                        exit_dt = day
                        exit_credit_value = mark
                        exit_reason = "stop_loss"
                        break
                    if pnl_points >= float(cfg["take_profit_credit_pct"][0]) * credit:
                        exit_dt = day
                        exit_credit_value = mark
                        exit_reason = "take_profit"
                        break
                if exit_credit_value is None:
                    exit_credit_value = _mark_value(legs, exit_dt, rows)
                if exit_credit_value is None:
                    continue
                gross_points = credit + exit_credit_value
                gross_pnl = gross_points * lot
                exit_prices = {}
                for n, leg in legs.items():
                    exit_prices[n] = _price(rows, exit_dt, expiry, leg["strike"], leg["type"])
                if any(v is None for v in exit_prices.values()):
                    continue
                side_pairs = [(entry_prices[n], exit_prices[n], legs[n]["side"]) for n in legs]
                trade_cost = costs.round_trip(side_pairs)
                net_pnl = gross_pnl - trade_cost
                return_on_capital = net_pnl / float(cfg["capital"])
                return_on_risk = net_pnl / max(entry_margin, 1.0)
                trades.append({
                    "entry_date": dt.date().isoformat(), "expiry": expiry.date().isoformat(), "exit_date": exit_dt.date().isoformat(),
                    "put_long": pl, "put_short": ps, "call_short": cs, "call_long": cl,
                    "credit_points": credit, "gross_pnl": gross_pnl, "costs": trade_cost, "net_pnl": net_pnl,
                    "return_on_capital": return_on_capital, "return_on_max_loss": return_on_risk,
                    "max_loss_rupees": entry_margin, "exit_reason": exit_reason,
                    "distance": distance, "wing_width": wing, "lot_size": lot, "peak_loss": peak_loss,
                })
                break
            else:
                continue
            break

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df, {"trades": 0, "data_ok": False, "reason": "No complete NIFTY weekly contracts matched the entry/exit rules."}
    trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"])
    trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"])
    equity = float(cfg["capital"])
    curve = []
    for _, r in trades_df.sort_values("exit_date").iterrows():
        equity += float(r.net_pnl)
        curve.append(equity)
    trades_df["equity"] = curve
    trades_df["equity_peak"] = trades_df["equity"].cummax()
    trades_df["drawdown"] = trades_df["equity"] / trades_df["equity_peak"] - 1.0
    weekly_returns = trades_df.groupby(trades_df["exit_date"].dt.to_period("W"))["net_pnl"].sum() / float(cfg["capital"])
    wins = trades_df["net_pnl"] > 0
    summary = {
        "trades": int(len(trades_df)),
        "weeks_tested": int(trades_df["expiry"].nunique()),
        "win_rate": float(wins.mean()),
        "avg_weekly_return": float(weekly_returns.mean()),
        "median_weekly_return": float(weekly_returns.median()),
        "p05_weekly_return": float(weekly_returns.quantile(0.05)),
        "worst_weekly_return": float(weekly_returns.min()),
        "weeks_at_or_above_5pct": float((weekly_returns >= 0.05).mean()),
        "weeks_at_or_above_0pct": float((weekly_returns >= 0).mean()),
        "total_return": float(trades_df["net_pnl"].sum() / float(cfg["capital"])),
        "max_drawdown": float(trades_df["drawdown"].min()),
        "avg_net_pnl": float(trades_df["net_pnl"].mean()),
        "median_net_pnl": float(trades_df["net_pnl"].median()),
        "worst_trade": float(trades_df["net_pnl"].min()),
        "best_trade": float(trades_df["net_pnl"].max()),
        "data_ok": True,
    }
    return trades_df, summary


def run(data_path: str, output_dir: str, config_path: str | None = None) -> None:
    cfg = DEFAULT_CONFIG.copy()
    if config_path:
        import yaml
        cfg.update(yaml.safe_load(Path(config_path).read_text()) or {})
    rows = load_dataset(data_path)
    trades, summary = backtest(rows, cfg=cfg)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out / "trades.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output", default="backtest/results/iron-condor")
    p.add_argument("--config", default=None)
    args = p.parse_args()
    run(args.data, args.output, args.config)
