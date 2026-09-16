# Iron Condor V3 — Daily Walk-Forward Research

This is an isolated V3 research line. It does **not** modify the V1/V2 production paper-trading logic.

## Objective

Backtest a **daily-decision NIFTY iron condor** using strict chronological walk-forward selection.

Each trading session is treated as a decision date. The model:

1. Uses only information available at the completed decision session.
2. Selects the configuration using only trades whose signal and exit occurred before the current decision date.
3. Determines strikes from the decision-session NIFTY reference and enters on the **next available trading session**.
4. Uses EOD option marks for research exits. Intraday OHLC is used only for a conservative path-risk audit and is **never** treated as proof of an intraday fill.
5. Applies modeled brokerage, STT, stamp duty, SEBI, exchange charges and slippage.
6. Uses the historical NIFTY lot size schedule rather than today's lot size for the entire history.
7. Produces a complete daily decision ledger, executable trade ledger, walk-forward window summaries and slippage stress results.

## What V3 explicitly prevents

- Same-session signal selection + same-session entry lookahead.
- Selecting parameters using the test period.
- Using future expiry/strike information unavailable on the signal date.
- Claiming an SL/TP hit from daily OHLC extremes alone.
- Treating theoretical compounding as executable P&L.
- Hiding skipped days or failed contract/price availability.
- Silent dataset truncation.
- Hard-coded weekday assumptions.
- Mixing prospective paper trades with historical OOS trades.

## Primary output

`v3/research/daily_walk_forward_summary.json`

Additional outputs include:

- `v3/research/daily_decisions.csv`
- `v3/research/oos_trade_ledger.csv`
- `v3/research/window_summary.csv`
- `v3/research/slippage_stress.csv`
- `v3/research/path_risk_audit.csv`

The live/prospective paper-trading book is intentionally **not** part of this V3 research branch.
