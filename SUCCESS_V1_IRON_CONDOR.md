# Success V1 — NIFTY Iron Condor

This branch is an isolated research and prospective paper-testing system for a weekly NIFTY iron condor.

## Historical research

- Initial capital: Rs.100,000
- Entry day: Tuesday
- Eligible expiry: 1–7 calendar DTE
- Parameter grid: 512 configurations
- Chronological split: 70% training / 30% OOS
- Data: daily NSE NIFTY option and NIFTY index-futures bhavcopy history
- Historical lot size is selected by expiry vintage
- EOD option closes are used for fills; daily OHLC is used only as a conservative intraday path-risk flag
- Fixed-lot cumulative equity and a separate theoretical per-trade compounded equity curve are both reported
- Python `weekday()` convention is explicit: Tuesday = `1`

The corrected engine reads the entry weekday and expiry DTE window from configuration rather than silently hard-coding them. The theoretical compounded curve assumes percentage re-sizing after every trade and is therefore a sensitivity metric, not a live-trading guarantee.

## Candidate paper configuration

- Short-strike distance: 1.25%
- Wing width: 250 points
- Take profit: 40% of initial credit retained as profit
- Stop loss: 1.0 × initial credit loss
- One market lot for the initial prospective test
- Maximum one trade per week

## Prospective paper calls

Run **Actions → Success V1 NIFTY Iron Condor Paper Call → Run workflow** after the latest NSE session is available. The workflow downloads the recent NSE bhavcopy, creates or updates the model position, and writes:

- `signals/latest_trade_call.md` — human-readable trade call
- `signals/latest_trade_call.json` — all structured trade details
- `signals/open_trade.json` — currently tracked model position
- `signals/trade_history.csv` — model exit journal

Each call contains the underlying proxy, expiry/DTE, all four strikes, all four reference premiums, applicable market lot size, entry credit, credit value, target/stop close debit, breakevens, maximum expiry loss, estimated entry-side costs, target and stop P&L, modeled returns, and status.

## Prospective validation rule

The EOD premiums are reference marks, not guaranteed fills. For a real paper test, record the actual four leg fills and the broker contract-note charges separately. Compare the live paper result with the model result and record slippage.

This is a research/paper-testing system, not an execution bot and not a guarantee of future returns.
