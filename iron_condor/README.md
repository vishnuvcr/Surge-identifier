# Iron Condor Weekly Research

Isolated research branch for testing weekly index iron-condor strategies using NSE F&O data where contract-level historical option prices are available.

## Research principles
- A 5% weekly return is a target, not an assurance.
- Backtests must report profitable weeks, losing weeks, max drawdown, tail losses, average return, median return, compounded return, and worst week.
- Include transaction costs, STT, exchange charges, GST, SEBI charges, stamp duty, slippage, and margin assumptions.
- Avoid look-ahead bias: strikes, premiums, IV and signals must use only information available at entry time.
- Prefer NIFTY index options initially; expand to BANKNIFTY/other eligible indices only after the base engine is validated.

## Strategy family to test
1. Symmetric and asymmetric iron condors.
2. Short strikes selected by delta, percentile distance, and volatility-adjusted distance.
3. Wings selected by fixed width and risk-budget constraints.
4. Entry windows from Monday through Thursday, with separate same-day/weekly-expiry handling.
5. Profit-taking at configurable percentages of max credit.
6. Stop-loss based on credit multiple, loss amount, underlying movement, and short-leg delta expansion.
7. Time-based exits before expiry to reduce expiry-gap/tail risk.
8. Regime filter using the existing Surge-identifier features plus realized volatility/trend.
9. Optional ML probability filter estimating probability of staying inside the short strikes through exit.

## Required evaluation
The test must compare parameter sets on out-of-sample periods. Do not select a strategy solely because it exceeds 5% in-sample. Report stability by month/quarter/year and include the distribution of weekly returns.
