# Point-in-Time Intraday Backtest — Daily Trade Mode

Initial capital: ₹1,00,000; maximum 5 simultaneous trades; equal capital split.
At least 1 trade is selected each trading day whenever an executable NSE equity candidate exists.
Days where the model has no threshold-qualified signal use an explicitly marked `forced_fallback` top-ranked candidate.

| Mode | Final equity | Net P&L | Return | Fees | Trades | Win rate | Max DD | Fallback days |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| close_exit | ₹41,049.00 | ₹-58,951.00 | -58.95% | ₹23,620.41 | 455 | 32.31% | -60.87% | 87 |
| tp3_close | ₹62,522.88 | ₹-37,477.12 | -37.48% | ₹24,543.95 | 455 | 49.45% | -42.45% | 87 |

## Key safeguards

- Point-in-time model retraining; no future labels cross the prediction-day boundary.
- Trailing liquidity only; no full-sample liquidity filter.
- EOD signal, next-session-open entry.
- 0.05% slippage per side.
- Paytm Money brokerage modeled at ₹20 per executed order; two orders per completed trade.
- Intraday statutory charges included: exchange turnover, 0.025% sell-side STT, SEBI turnover fee, buy-side stamp duty, and 18% GST on brokerage + exchange charges.
- Daily-bar limitation: exact intraday price path is unavailable; the +3% target case is therefore a separate scenario.

## Interpretation

The forced-fallback days are intentionally reported separately. A daily-trade requirement can materially reduce the quality of the trading edge because it prevents the model from staying flat when no strong signal exists.