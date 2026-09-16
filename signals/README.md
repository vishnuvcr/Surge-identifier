# Success V1 prospective paper-trading signals

These files are generated from the `success-v1-iron-condor` branch for prospective paper testing.

`latest_trade_call.md` is the human-readable call. `latest_trade_call.json` contains the machine-readable fields. `open_trade.json` stores the currently tracked model position. `trade_history.csv` stores model exits.

The system uses daily NSE F&O bhavcopy data and a same-session nearest non-expired NIFTY futures close as the underlying proxy. Historical backtests and these calls are research outputs, not guaranteed returns or execution instructions.

For prospective validation, record the actual four-leg fills and all charges separately. The generated EOD prices are reference prices, not guaranteed fills.
