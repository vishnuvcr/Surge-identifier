# Phase 3 — Two-Layer Point-in-Time Intraday Backtest

Initial capital: ₹1,00,000. Portfolio sizes tested: 1, 3 and 5.
Layer 1: high-precision >=3% surge classifier. Layer 2: continuous next-session intraday-return ranker for coverage.
The surge probability threshold is never relaxed to manufacture trades.

| Exit mode | Portfolio | Final equity | Return | Fees | Trades | Qualified | Rank layer | Win rate | PF | Max DD | Rank IC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| close_exit | 1 | ₹94,402.92 | -5.60% | ₹7,875.61 | 103 | 16 | 87 | 33.01% | 0.94 | -39.99% | 0.274 |
| close_exit | 3 | ₹61,705.02 | -38.29% | ₹17,330.92 | 309 | 21 | 288 | 30.74% | 0.58 | -39.48% | 0.274 |
| close_exit | 5 | ₹53,494.88 | -46.51% | ₹26,646.64 | 515 | 22 | 493 | 28.35% | 0.53 | -46.10% | 0.274 |
| tp3_close | 1 | ₹102,977.32 | 2.98% | ₹8,710.81 | 103 | 16 | 87 | 40.78% | 1.03 | -28.98% | 0.274 |
| tp3_close | 3 | ₹71,418.13 | -28.58% | ₹17,636.70 | 309 | 21 | 288 | 37.22% | 0.67 | -31.65% | 0.274 |
| tp3_close | 5 | ₹53,888.14 | -46.11% | ₹26,672.45 | 515 | 22 | 493 | 33.20% | 0.49 | -45.87% | 0.274 |

## Anti-lookahead
- Both models train only on fully realized historical next-session labels strictly before each prediction day.
- Trailing liquidity and correlations use only information available before the signal.
- EOD signal -> next-session-open entry.
- Future OHLC is used only after selection for trade outcome simulation.
- Integer-share quantities; no leverage; modeled brokerage/statutory charges and 0.05% slippage per side.

## Daily-bar limitation
The close-exit path is a close-price proxy; daily OHLC cannot reproduce the true intraday auto-square-off. The +3% target path is reported separately.