# Phase 3 — Two-Layer Point-in-Time Intraday Backtest

Initial capital: ₹1,00,000. Portfolio sizes tested: 1, 3 and 5.
Layer 1: high-precision >=3% surge classifier. Layer 2: continuous next-session intraday-return ranker for coverage.
The surge probability threshold is never relaxed to manufacture trades.

| Exit mode | Portfolio | Final equity | Return | Fees | Trades | Qualified | Rank layer | Win rate | PF | Max DD | Rank IC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| close_exit | 1 | ₹96,425.90 | -3.57% | ₹7,794.26 | 102 | 15 | 87 | 33.33% | 0.96 | -39.99% | 0.274 |
| close_exit | 3 | ₹63,114.17 | -36.89% | ₹17,167.05 | 306 | 20 | 286 | 31.05% | 0.59 | -39.48% | 0.274 |
| close_exit | 5 | ₹54,778.03 | -45.22% | ₹26,391.75 | 510 | 21 | 489 | 28.63% | 0.54 | -44.81% | 0.274 |
| tp3_close | 1 | ₹100,125.49 | 0.13% | ₹8,626.57 | 102 | 15 | 87 | 40.20% | 1.00 | -28.98% | 0.274 |
| tp3_close | 3 | ₹71,839.42 | -28.16% | ₹17,469.67 | 306 | 20 | 286 | 37.25% | 0.67 | -31.65% | 0.274 |
| tp3_close | 5 | ₹54,664.63 | -45.34% | ₹26,417.42 | 510 | 21 | 489 | 33.33% | 0.50 | -45.09% | 0.274 |

## Anti-lookahead
- Both models train only on fully realized historical next-session labels strictly before each prediction day.
- Trailing liquidity and correlations use only information available before the signal.
- EOD signal -> next-session-open entry.
- Future OHLC is used only after selection for trade outcome simulation.
- Integer-share quantities; no leverage; modeled brokerage/statutory charges and 0.05% slippage per side.

## Daily-bar limitation
The close-exit path is a close-price proxy; daily OHLC cannot reproduce the true intraday auto-square-off. The +3% target path is reported separately.