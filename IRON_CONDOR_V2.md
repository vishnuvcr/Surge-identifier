# Iron Condor V2

V2 is a separate research branch built from the validated Success V1 implementation.

## What changes from V1

1. **Daily candidate generation** across Monday–Friday rather than a Tuesday-only entry rule.
2. **Delayed execution**: strikes are selected from the completed signal session, but the position is entered on the next eligible NSE session.
3. **No overlapping positions**: one open iron condor at a time.
4. **Credit-quality filter**: minimum entry credit is optimized after the main parameter search.
5. **Execution stress**: selection uses 0.50-point slippage per leg; OOS sensitivity is tested at 0 / 0.25 / 0.50 / 1.00 points per leg.
6. **Walk-forward validation**: eight expanding chronological windows, with parameters selected only from each prior training period.
7. **Baseline comparison**: the V1 structure is replayed on the same OOS period under the same delayed-entry/slippage convention.

## Core search

The primary grid covers 5 weekdays × 4 short-distance levels × 4 wing widths × 4 take-profit levels × 4 stop-loss levels = **1,280 configurations**. The minimum-credit filter is then optimized among the top 10 configurations from the primary grid.

## Execution convention

Option premiums are daily EOD reference values. The V2 backtest does not claim intraday fills. The delayed-entry model removes the same-day close/fill coincidence identified in V1 validation. Slippage is applied pessimistically to the four legs at entry and exit.

## Output

Results are written to `backtest/results/iron-condor-v2/`:

- `v2_summary.json`
- `oos_trades.csv` and `oos_trades.json`
- `baseline_oos_trades.csv`
- `walk_forward_results.csv`
- `walk_forward_trades.csv`
- `slippage_sensitivity.csv`
- `training_top25.json`

No V2 profitability claim should be made until the manual research workflow completes successfully and the artifacts are inspected.
