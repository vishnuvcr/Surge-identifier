from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "v5/config.yaml").read_text())
CAPITAL = float(CFG["capital"])

STRATEGIES = [
    "LONG_CALL",
    "LONG_PUT",
    "BULL_CALL_SPREAD",
    "BEAR_PUT_SPREAD",
    "BULL_PUT_SPREAD",
    "BEAR_CALL_SPREAD",
    "LONG_STRADDLE",
    "LONG_STRANGLE",
    "IRON_CONDOR",
    "IRON_BUTTERFLY",
]


def lot_size(expiry: pd.Timestamp) -> int:
    e = pd.Timestamp(expiry).normalize()
    # Historical NIFTY lot-size schedule used by this research pipeline.
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


class Costs:
    def total(self, entry_prices, exit_prices, signs, qty_per_leg):
        buy_turnover = 0.0
        sell_turnover = 0.0
        for ep, xp, s in zip(entry_prices, exit_prices, signs):
            ep = max(float(ep), 0.0)
            xp = max(float(xp), 0.0)
            if s > 0:  # long at entry, sold at exit
                buy_turnover += ep * qty_per_leg
                sell_turnover += xp * qty_per_leg
            else:  # short at entry, bought back at exit
                sell_turnover += ep * qty_per_leg
                buy_turnover += xp * qty_per_leg
        turnover = buy_turnover + sell_turnover
        orders = 2 * len(signs)
        brokerage = orders * float(CFG["brokerage_per_order"])
        txn = turnover * float(CFG["exchange_txn_pct"])
        sebi = turnover * float(CFG["sebi_turnover_pct"])
        stt = sell_turnover * float(CFG["stt_sell_pct"])
        stamp = buy_turnover * float(CFG["stamp_buy_pct"])
        gst = (brokerage + txn + sebi) * float(CFG["gst_pct"])
        return brokerage + txn + sebi + stt + stamp + gst


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
        f = f[f.expiry >= f.date].sort_values(["date", "expiry"])
        f = f.drop_duplicates(["date", "expiry"], keep="last")
        spot = (
            f.sort_values(["date", "expiry"])
            .groupby("date", as_index=False)
            .first()[["date", "close"]]
        )
        self.spot = dict(zip(spot.date, spot.close.astype(float)))
        self.dates = tuple(pd.Timestamp(x) for x in sorted(set(o.date) & set(spot.date)))
        self.expiries = {}
        self.prices = {}
        for (d, e), g in o.groupby(["date", "expiry"], sort=False):
            d, e = pd.Timestamp(d), pd.Timestamp(e)
            self.expiries.setdefault(d, []).append(e)
        for d in self.expiries:
            self.expiries[d] = tuple(sorted(self.expiries[d]))
        for d, e, k, t, c in o[["date", "expiry", "strike", "option_type", "close"]].itertuples(index=False, name=None):
            self.prices[(pd.Timestamp(d), pd.Timestamp(e), float(k), str(t).upper())] = (
                float(c) if pd.notna(c) else np.nan
            )
        self.strike_cache = {}

    def price(self, d, e, k, t):
        v = self.prices.get((pd.Timestamp(d), pd.Timestamp(e), float(k), str(t).upper()))
        return None if v is None or pd.isna(v) or v <= 0 else float(v)

    def strikes(self, d, e, t):
        key = (pd.Timestamp(d), pd.Timestamp(e), str(t).upper())
        if key not in self.strike_cache:
            vals = [k for (dd, ee, k, tt), v in self.prices.items() if dd == key[0] and ee == key[1] and tt == key[2] and np.isfinite(v) and v > 0]
            self.strike_cache[key] = np.array(sorted(vals), dtype=float)
        return self.strike_cache[key]

    def nearest_strike(self, d, e, t, target):
        ks = self.strikes(d, e, t)
        if len(ks) == 0:
            return None
        return float(ks[np.argmin(np.abs(ks - target))])

    def expiry_for(self, d, min_dte, max_dte):
        d = pd.Timestamp(d)
        for e in self.expiries.get(d, ()):
            dte = (e - d).days
            if min_dte <= dte <= max_dte:
                return e
        return None

    def next_date(self, d, n=1, end=None):
        vals = [x for x in self.dates if x > pd.Timestamp(d)]
        if end is not None:
            vals = [x for x in vals if x <= pd.Timestamp(end)]
        return vals[n - 1] if len(vals) >= n else None


# ---------- feature construction ----------

def rsi(s, n=14):
    delta = s.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / (down + 1e-12)
    return 100 - 100 / (1 + rs)


def features_for_day(m: Market, d: pd.Timestamp):
    s = m.spot_series
    hist = s.loc[:d]
    if len(hist) < 60:
        return None
    x = hist.astype(float)
    ema10 = x.ewm(span=10, adjust=False).mean().iloc[-1]
    ema20 = x.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = x.ewm(span=50, adjust=False).mean().iloc[-1]
    vol10 = x.pct_change().rolling(10).std().iloc[-1] * math.sqrt(252)
    vol20 = x.pct_change().rolling(20).std().iloc[-1] * math.sqrt(252)
    r = rsi(x, 14).iloc[-1]
    hh20 = x.shift(1).rolling(20).max().iloc[-1]
    ll20 = x.shift(1).rolling(20).min().iloc[-1]
    hh60 = x.shift(1).rolling(60).max().iloc[-1]
    ll60 = x.shift(1).rolling(60).min().iloc[-1]
    ret1 = x.pct_change(1).iloc[-1]
    ret5 = x.pct_change(5).iloc[-1]
    ret10 = x.pct_change(10).iloc[-1]
    ret20 = x.pct_change(20).iloc[-1]
    slope20 = np.polyfit(np.arange(20), x.iloc[-20:].values, 1)[0] / x.iloc[-1]

    expiry = m.expiry_for(d, int(CFG["selection_dte_min"]), int(CFG["selection_dte_max"]))
    atm_straddle_pct = np.nan
    skew_proxy = np.nan
    dte = np.nan
    if expiry is not None:
        dte = (expiry - d).days
        atm = m.nearest_strike(d, expiry, "CE", m.spot[d])
        if atm is not None:
            c = m.price(d, expiry, atm, "CE")
            p = m.price(d, expiry, atm, "PE")
            if c and p:
                atm_straddle_pct = (c + p) / m.spot[d]
        otm = 100.0
        pc = m.nearest_strike(d, expiry, "CE", m.spot[d] + otm)
        pp = m.nearest_strike(d, expiry, "PE", m.spot[d] - otm)
        if pc and pp:
            c = m.price(d, expiry, pc, "CE")
            p = m.price(d, expiry, pp, "PE")
            if c and p:
                skew_proxy = (p - c) / max((p + c), 1e-9)

    return {
        "ret1": ret1,
        "ret5": ret5,
        "ret10": ret10,
        "ret20": ret20,
        "ema10_gap": (x.iloc[-1] / ema10) - 1,
        "ema20_gap": (x.iloc[-1] / ema20) - 1,
        "ema50_gap": (x.iloc[-1] / ema50) - 1,
        "ema10_20_gap": (ema10 / ema20) - 1,
        "ema20_50_gap": (ema20 / ema50) - 1,
        "rsi14": r / 100.0,
        "vol10": vol10,
        "vol20": vol20,
        "breakout20": (x.iloc[-1] / hh20) - 1,
        "breakdown20": (x.iloc[-1] / ll20) - 1,
        "range60": (x.iloc[-1] - ll60) / max(hh60 - ll60, 1e-9),
        "slope20": slope20,
        "atm_straddle_pct": atm_straddle_pct,
        "skew_proxy": skew_proxy,
        "dte": dte,
    }


# ---------- strategy construction ----------

def _legs(m, d, expiry, specs):
    spot = m.spot[d]
    # Build strikes from actual chain, never from an assumed historical strike grid.
    atm_c = m.nearest_strike(d, expiry, "CE", spot)
    atm_p = m.nearest_strike(d, expiry, "PE", spot)
    if atm_c is None or atm_p is None:
        return None
    out = []
    for typ, offset, sign in specs:
        target = spot + offset if typ == "CE" else spot + offset
        k = m.nearest_strike(d, expiry, typ, target)
        if k is None:
            return None
        px = m.price(d, expiry, k, typ)
        if px is None:
            return None
        out.append((typ, float(k), int(sign), float(px)))
    return out


def strategy_legs(m, d, expiry, name):
    w = float(CFG["wing_width_points"])
    sw = float(CFG["spread_width_points"])
    otm = float(CFG["otm_points"])
    specs = {
        "LONG_CALL": [("CE", 0, +1)],
        "LONG_PUT": [("PE", 0, +1)],
        "BULL_CALL_SPREAD": [("CE", 0, +1), ("CE", sw, -1)],
        "BEAR_PUT_SPREAD": [("PE", 0, +1), ("PE", -sw, -1)],
        "BULL_PUT_SPREAD": [("PE", 0, -1), ("PE", -sw, +1)],
        "BEAR_CALL_SPREAD": [("CE", 0, -1), ("CE", sw, +1)],
        "LONG_STRADDLE": [("CE", 0, +1), ("PE", 0, +1)],
        "LONG_STRANGLE": [("CE", otm, +1), ("PE", -otm, +1)],
        "IRON_CONDOR": [
            ("PE", -otm, -1), ("PE", -otm - w, +1),
            ("CE", otm, -1), ("CE", otm + w, +1),
        ],
        "IRON_BUTTERFLY": [
            ("PE", 0, -1), ("PE", -w, +1),
            ("CE", 0, -1), ("CE", w, +1),
        ],
    }[name]
    return _legs(m, d, expiry, specs)


def _intrinsic(typ, strike, spot):
    return max(spot - strike, 0.0) if typ == "CE" else max(strike - spot, 0.0)


def max_loss_points(legs):
    if not legs:
        return np.nan
    strikes = [x[1] for x in legs]
    lo, hi = min(strikes) - 400, max(strikes) + 400
    grid = np.linspace(lo, hi, 401)
    entry_cash = sum(sign * premium for _, _, sign, premium in legs)
    pnl = []
    for spot in grid:
        payoff = sum(sign * _intrinsic(typ, k, spot) for typ, k, sign, _ in legs)
        pnl.append(payoff - entry_cash)
    return max(0.01, -min(pnl))


def realized_strategy(m: Market, signal_date, strategy, end_date=None):
    entry_date = m.next_date(signal_date, 1, end_date=end_date)
    if entry_date is None:
        return None
    expiry = m.expiry_for(entry_date, int(CFG["selection_dte_min"]), int(CFG["selection_dte_max"]))
    if expiry is None or expiry <= entry_date:
        return None
    # Horizon is calendar-independent: choose the fifth available market session.
    exit_date = m.next_date(entry_date, int(CFG["holding_days"]), end=end_date)
    if exit_date is None or exit_date >= expiry:
        exit_date = m.next_date(entry_date, max(1, int(CFG["holding_days"]) - 1), end=end_date)
    if exit_date is None or exit_date >= expiry:
        return None
    legs = strategy_legs(m, entry_date, expiry, strategy)
    if legs is None:
        return None

    exit_prices = []
    signs = []
    entry_prices = []
    for typ, strike, sign, entry_raw in legs:
        xp = m.price(exit_date, expiry, strike, typ)
        if xp is None:
            return None
        # Slippage is adverse at both entry and exit.
        entry_exec = entry_raw + float(CFG["entry_slippage_points"]) if sign > 0 else max(entry_raw - float(CFG["entry_slippage_points"]), 0.01)
        exit_exec = max(xp - float(CFG["exit_slippage_points"]), 0.01) if sign > 0 else xp + float(CFG["exit_slippage_points"])
        entry_prices.append(entry_exec)
        exit_prices.append(exit_exec)
        signs.append(sign)

    qty_per_leg = lot_size(expiry)
    gross = sum(sign * (xp - ep) for xp, ep, sign in zip(exit_prices, entry_prices, signs)) * qty_per_leg
    costs = Costs().total(entry_prices, exit_prices, signs, qty_per_leg)
    net = gross - costs
    risk_points = max_loss_points(legs)
    risk_cash = max(risk_points * qty_per_leg, 1.0)
    return {
        "entry_date": entry_date,
        "exit_date": exit_date,
        "expiry": expiry,
        "net_pnl_1lot": float(net),
        "risk_cash_1lot": float(risk_cash),
        "risk_return": float(net / risk_cash),
        "capital_return_1lot": float(net / CAPITAL),
    }


FEATURE_COLUMNS = [
    "ret1", "ret5", "ret10", "ret20", "ema10_gap", "ema20_gap", "ema50_gap",
    "ema10_20_gap", "ema20_50_gap", "rsi14", "vol10", "vol20", "breakout20",
    "breakdown20", "range60", "slope20", "atm_straddle_pct", "skew_proxy", "dte"
]


def build_dataset(m: Market):
    rows = []
    start_min = max(pd.Timestamp(m.dates[0]), pd.Timestamp("2018-01-01"))
    max_h = int(CFG["holding_days"]) + 2
    for i, d in enumerate(m.dates):
        if d < start_min:
            continue
        f = features_for_day(m, d)
        if f is None:
            continue
        labels = {}
        details = {}
        for s in STRATEGIES:
            r = realized_strategy(m, d, s)
            if r is None:
                labels[s] = np.nan
            else:
                labels[s] = r["risk_return"]
                details[s] = r
        if sum(pd.notna(v) for v in labels.values()) < 8:
            continue
        row = {"signal_date": d, **f, **labels}
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def fit_model(train):
    valid = train.dropna(subset=STRATEGIES).copy()
    if len(valid) < int(CFG["min_training_samples"]):
        raise RuntimeError(f"Only {len(valid)} valid training rows; need {CFG['min_training_samples']}")
    X = valid[FEATURE_COLUMNS]
    y = valid[STRATEGIES]
    mcfg = CFG["model"]
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(
            n_estimators=int(mcfg["n_estimators"]),
            max_depth=int(mcfg["max_depth"]),
            min_samples_leaf=int(mcfg["min_samples_leaf"]),
            random_state=int(mcfg["random_state"]),
            n_jobs=-1,
        ),
    )
    model.fit(X, y)
    return model, valid


def execute_selected(m: Market, signal_date, strategy, end_date=None):
    entry_date = m.next_date(signal_date, 1, end_date=end_date)
    if entry_date is None:
        return None
    expiry = m.expiry_for(entry_date, int(CFG["selection_dte_min"]), int(CFG["selection_dte_max"]))
    if expiry is None:
        return None
    exit_date = m.next_date(entry_date, int(CFG["holding_days"]), end=end_date)
    if exit_date is None or exit_date >= expiry:
        return None
    legs = strategy_legs(m, entry_date, expiry, strategy)
    if legs is None:
        return None
    exit_prices = []
    entry_prices = []
    signs = []
    for typ, strike, sign, ep0 in legs:
        xp0 = m.price(exit_date, expiry, strike, typ)
        if xp0 is None:
            return None
        ep = ep0 + float(CFG["entry_slippage_points"]) if sign > 0 else max(ep0 - float(CFG["entry_slippage_points"]), 0.01)
        xp = max(xp0 - float(CFG["exit_slippage_points"]), 0.01) if sign > 0 else xp0 + float(CFG["exit_slippage_points"])
        entry_prices.append(ep); exit_prices.append(xp); signs.append(sign)
    lot = lot_size(expiry)
    risk_points = max_loss_points(legs)
    risk_per_lot = max(risk_points * lot, 1.0)
    risk_budget = CAPITAL * float(CFG["risk_per_trade_pct"])
    lots = max(1, min(int(CFG["max_lots"]), int(math.floor(risk_budget / risk_per_lot))))
    qty = lots * lot
    gross_per_lot = sum(sign * (xp - ep) for xp, ep, sign in zip(exit_prices, entry_prices, signs)) * lot
    costs_per_lot = Costs().total(entry_prices, exit_prices, signs, lot)
    net = (gross_per_lot - costs_per_lot) * lots
    risk_cash = risk_per_lot * lots
    return {
        "signal_date": signal_date,
        "entry_date": entry_date,
        "exit_date": exit_date,
        "strategy": strategy,
        "expiry": expiry,
        "lot_size": lot,
        "lots": lots,
        "qty": qty,
        "risk_cash": risk_cash,
        "gross_pnl": gross_per_lot * lots,
        "costs": costs_per_lot * lots,
        "net_pnl": net,
        "capital_return": net / CAPITAL,
    }


def metrics(t):
    if t.empty:
        return {
            "trades": 0, "net_pnl": 0.0, "return": 0.0, "win_rate": 0.0,
            "max_drawdown": 0.0, "profit_factor": 0.0, "months": 0,
            "median_monthly_return": 0.0, "mean_monthly_return": 0.0,
            "months_ge_30pct": 0.0, "months_nonnegative": 0.0,
            "worst_month": 0.0, "best_month": 0.0,
        }
    eq = CAPITAL + t.net_pnl.cumsum()
    dd = eq / eq.cummax() - 1
    wins = t.net_pnl > 0
    monthly = t.groupby(t.exit_date.dt.to_period("M")).net_pnl.sum() / CAPITAL
    gains = float(t.loc[wins, "net_pnl"].sum())
    losses = float(-t.loc[~wins, "net_pnl"].sum())
    return {
        "trades": int(len(t)),
        "net_pnl": float(t.net_pnl.sum()),
        "return": float(t.net_pnl.sum() / CAPITAL),
        "win_rate": float(wins.mean()),
        "median_trade": float(t.net_pnl.median()),
        "worst_trade": float(t.net_pnl.min()),
        "best_trade": float(t.net_pnl.max()),
        "max_drawdown": float(dd.min()),
        "profit_factor": float(gains / losses) if losses > 0 else math.inf,
        "months": int(len(monthly)),
        "median_monthly_return": float(monthly.median()),
        "mean_monthly_return": float(monthly.mean()),
        "months_ge_30pct": float((monthly >= 0.30).mean()),
        "months_nonnegative": float((monthly >= 0).mean()),
        "worst_month": float(monthly.min()),
        "best_month": float(monthly.max()),
    }


def main():
    opt = pd.read_parquet(ROOT / "data/cache/nifty_options_long.parquet")
    fut = pd.read_parquet(ROOT / "data/cache/nifty_futures_long.parquet")
    fut = fut.rename(columns={c: c.lower() for c in fut.columns})
    opt = opt.rename(columns={c: c.lower() for c in opt.columns})
    m = Market(opt, fut)
    m.spot_series = pd.Series(m.spot).sort_index()

    research = ROOT / "v5/research"
    research.mkdir(parents=True, exist_ok=True)
    dataset_path = research / "candidate_dataset.parquet"
    if dataset_path.exists():
        dataset = pd.read_parquet(dataset_path)
        dataset["signal_date"] = pd.to_datetime(dataset.signal_date).dt.normalize()
    else:
        dataset = build_dataset(m)
        dataset.to_parquet(dataset_path, index=False)
    if dataset.empty:
        raise RuntimeError("V5 candidate dataset is empty")

    all_trades = []
    wf = []
    for wi, w in enumerate(CFG["walk_forward"], 1):
        train_end = pd.Timestamp(w["train_end"])
        test_start = pd.Timestamp(w["test_start"])
        test_end = m.dates[-1] if str(w["test_end"]).lower() == "latest" else pd.Timestamp(w["test_end"])
        train_start = train_end - pd.DateOffset(years=int(CFG["rolling_train_years"]))
        train = dataset[(dataset.signal_date > train_start) & (dataset.signal_date <= train_end)]
        model, train_valid = fit_model(train)
        test = dataset[(dataset.signal_date >= test_start) & (dataset.signal_date <= test_end)].copy()
        preds = model.predict(test[FEATURE_COLUMNS])
        pred_df = pd.DataFrame(preds, columns=STRATEGIES, index=test.index)
        chosen = []
        for idx in test.index:
            vals = pred_df.loc[idx]
            ordered = vals.sort_values(ascending=False)
            top = ordered.index[0]
            second = ordered.iloc[1]
            edge = float(ordered.iloc[0] - second)
            if float(ordered.iloc[0]) < float(CFG["no_trade_threshold"]) or edge < float(CFG["min_predicted_edge_vs_second"]):
                chosen.append("NO_TRADE")
            else:
                chosen.append(top)
        test["selected_strategy"] = chosen

        busy_until = None
        for idx, row in test.sort_values("signal_date").iterrows():
            d = pd.Timestamp(row.signal_date)
            if busy_until is not None and d <= busy_until:
                continue
            selected = row.selected_strategy
            if selected == "NO_TRADE":
                continue
            trade = execute_selected(m, d, selected, end_date=test_end)
            if trade is None:
                continue
            trade["window"] = wi
            trade["model_predicted_risk_return"] = float(pred_df.loc[idx, selected])
            realized_row = dataset.loc[idx]
            trade["actual_selected_risk_return"] = float(realized_row[selected])
            trade["selection_confidence"] = float(pred_df.loc[idx].sort_values(ascending=False).iloc[0] - pred_df.loc[idx].sort_values(ascending=False).iloc[1])
            all_trades.append(trade)
            busy_until = pd.Timestamp(trade["exit_date"])

        selected_counts = test.selected_strategy.value_counts().to_dict()
        selected_only = test[test.selected_strategy != "NO_TRADE"]
        if not selected_only.empty:
            actual_best = selected_only[STRATEGIES].max(axis=1)
            actual_selected = [row[s] for _, row in selected_only.iterrows() for s in [row.selected_strategy]]
            selector_capture = float(np.nanmean(actual_selected - actual_best))
        else:
            selector_capture = np.nan
        wf.append({
            "window": wi,
            "train_start": str(train_start.date()),
            "train_end": str(train_end.date()),
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "training_rows": int(len(train_valid)),
            "test_rows": int(len(test)),
            "selected_counts": selected_counts,
            "selector_capture_vs_daily_best": selector_capture,
        })

    trades = pd.DataFrame(all_trades)
    if not trades.empty:
        trades["signal_date"] = pd.to_datetime(trades.signal_date)
        trades["entry_date"] = pd.to_datetime(trades.entry_date)
        trades["exit_date"] = pd.to_datetime(trades.exit_date)
        trades = trades.sort_values("entry_date").reset_index(drop=True)

    summary = {
        "strategy": "V5 Adaptive Multi-Strategy Options Engine",
        "description": "Walk-forward multi-output ML selector over ten defined option structures; one selected trade at a time; long and defined-risk strategies only.",
        "capital": CAPITAL,
        "selection_target": "Predict forward 5-session risk-adjusted return, then trade only when predicted edge and confidence clear thresholds.",
        "candidate_strategies": STRATEGIES,
        "overall_oos": metrics(trades) if not trades.empty else metrics(pd.DataFrame()),
        "windows": wf,
        "notes": [
            "Features use data available on the signal date; entry is next available session.",
            "Historical strikes are selected from the actual option chain rather than assuming a fixed strike grid.",
            "Labels use a fixed five-session holding horizon and transaction-cost/slippage assumptions.",
            "V5.0 is a strategy selector research baseline; intraday OHLC path and IV-surface reconstruction are intentionally deferred to V5.1.",
        ],
    }
    (research / "summary.json").write_text(json.dumps(summary, indent=2, default=str, allow_nan=True))
    (research / "walk_forward_summary.json").write_text(json.dumps(wf, indent=2, default=str, allow_nan=True))
    trades.to_csv(research / "oos_trade_ledger.csv", index=False)
    print(json.dumps(summary, indent=2, default=str, allow_nan=True))


if __name__ == "__main__":
    main()
