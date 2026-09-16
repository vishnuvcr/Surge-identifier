# V4 NIFTY Option Buying Research

Goal: research whether a directionally selective NIFTY option-buying system can produce strong monthly returns, with **30% monthly return as a research target, not a guaranteed outcome**.

Constraints:
- Daily EOD NSE option/futures data.
- No look-ahead: signal on completed session, entry reference on next session.
- Long calls/puts only; no option selling.
- Explicit brokerage, STT, exchange, SEBI, GST and slippage model.
- Expanding walk-forward OOS validation.
- Monthly aggregation and drawdown reporting.
- Intraday fills are not claimed from EOD data.
