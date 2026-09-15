# Phase 2 — Diversified Point-in-Time Intraday Backtest

Initial capital: ₹1,00,000. Portfolio sizes tested: 1, 3 and 5.
Capital is split equally across the selected positions each day.
At least one executable trade is selected each trading day; when no qualified signal exists, exactly one fallback trade is used.
For portfolios of 3 or 5, only threshold-qualified candidates can fill additional positions; the system never pads a portfolio with multiple weak fallback trades.
Additional positions are selected using trailing correlation to reduce concentration among highly correlated candidates.

| Exit mode | Portfolio | Final equity | Return | Fees | Trades | Qualified | Fallback | Win rate | PF | Max DD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| close_exit | 1 | ₹42,277.29 | -57.72% | ₹6,823.08 | 102 | 15 | 87 | 38.24% | 0.53 | -67.94% |
| close_exit | 3 | ₹33,950.65 | -66.05% | ₹7,046.44 | 107 | 20 | 87 | 37.38% | 0.47 | -67.94% |
| close_exit | 5 | ₹33,915.48 | -66.08% | ₹7,093.85 | 108 | 21 | 87 | 37.04% | 0.47 | -67.94% |
| tp3_close | 1 | ₹75,376.91 | -24.62% | ₹8,321.37 | 102 | 15 | 87 | 52.94% | 0.85 | -44.09% |
| tp3_close | 3 | ₹70,114.64 | -29.89% | ₹8,548.52 | 107 | 20 | 87 | 53.27% | 0.82 | -44.09% |
| tp3_close | 5 | ₹70,097.03 | -29.90% | ₹8,596.21 | 108 | 21 | 87 | 52.78% | 0.82 | -44.09% |

## Anti-lookahead controls
- Historical labels are fully realized before model fitting for each prediction day.
- Trailing liquidity and trailing return correlations are computed strictly before the signal day.
- EOD signal → next-session-open entry.
- Future OHLC is used only after portfolio selection for outcome simulation.
- Integer-share quantities; no leverage.
- 0.05% slippage per side plus modeled brokerage/statutory charges.

## Daily-bar limitation
The close-exit path is a close-price proxy because daily OHLC does not contain the actual intraday auto-square-off price. The +3% target path is reported separately.