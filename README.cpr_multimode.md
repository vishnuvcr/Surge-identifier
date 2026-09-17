# CPR Multi-Mode Research

This branch extends the CPR research into three explicitly different trading horizons.

## Modes

### 1. Intraday
- 5-minute execution bars.
- Long and short.
- Daily CPR is the primary intraday level set.
- Weekly CPR is used for multi-timeframe context where appropriate.
- Positions are forcibly closed at the end of the session.

### 2. Swing
- Daily execution bars.
- Long and short.
- Weekly CPR is the primary swing level set.
- Monthly CPR is used as higher-timeframe context.
- Maximum holding period is configurable in trading days.

### 3. BTST
- Signal is evaluated using the completed trading day's close.
- Entry is the same day's final 5-minute close (BTST proxy).
- Exit is the next trading session using 5-minute OHLC for stop/target, with forced exit at session close.
- Long-only by default because ordinary cash-equity BTST is a long overnight position. An STBT/derivative mode should be tested separately rather than silently treating a cash short as overnight.

The same CPR strategy catalogue is evaluated separately under each mode so a strategy is not forced to use intraday levels/holding assumptions when tested as swing or BTST.

The implementation uses only prior-period CPR inputs for each decision and records the exact setup timeframe, entry, stop, target, exit reason, and mode.
