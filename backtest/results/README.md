# Point-in-Time Intraday Backtest

Initial capital: ₹1,00,000; maximum 5 simultaneous trades; equal capital split among selected trades.

## Results

| Mode | Final equity | Net P&L | Return | Fees | Trades | Win rate | Max DD |
|---|---:|---:|---:|---:|---:|---:|---:|
| close_exit | ₹119,971.72 | ₹19,971.72 | 19.97% | ₹505.90 | 6 | 83.33% | -4.31% |
| tp3_close | ₹110,016.94 | ₹10,016.94 | 10.02% | ₹508.02 | 6 | 83.33% | -4.40% |

## Execution assumptions

- Entry: next trading day open after the EOD signal.
- Exit 1: same-day close.
- Exit 2: take profit at +3% when the next-day high reaches the target; otherwise close.
- Slippage: 0.05% per side.
- Paytm Money brokerage: ₹20 per executed order; two orders per completed trade.
- Modeled statutory charges: exchange turnover, 0.025% STT on sell, 0.0001% SEBI turnover fee, 0.003% stamp duty on buy, and 18% GST on brokerage + exchange charges.

## Anti-lookahead controls
- For each prediction month, training uses only feature rows dated strictly before the prediction day.
- Training rows are additionally removed when their next-session label outcome would occur on or after the prediction day.
- The model threshold is selected only inside the historical training/validation window preceding that prediction month.
- Liquidity uses only turnover observations from the trailing sessions strictly before each signal date.
- The signal is generated from EOD data; entry occurs on the next trading session open.
- Next-day OHLC is used only after the signal is fixed to calculate realized trade outcome.
- Capital is split equally across the selected trades using starting equity for that day; no leverage is used.
- Daily bars cannot reconstruct the exact intraday path; the +3% target scenario is therefore reported separately.

The backtest uses daily OHLC data. Exact intraday timestamp/fill quality cannot be reconstructed from daily bars, so the +3% target mode is reported separately rather than silently treated as the only result.