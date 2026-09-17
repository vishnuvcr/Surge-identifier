# CPR v1.0 — Canonical Execution Policy

The supplied transcripts often describe entries and exits qualitatively. To make strategies comparable, this branch uses the following research execution policy unless a strategy-specific rule is explicitly defined.

## Signal and entry
- Signal is created only after a completed candle close.
- Equity/index intraday signal: execute at the next 5-minute bar open.
- Swing/BTST signal: execute at the next regular-session open.
- No same-bar look-ahead.

## Intraday exits
- One open position per symbol.
- No new positions after the configured cut-off.
- Force-close all intraday positions at 15:25 IST unless a strategy explicitly requires an earlier exit.
- If both target and stop are touched in the same OHLC bar, assume stop first. This is conservative in the absence of tick data.
- Slippage and transaction costs are deducted on both entry and exit.

## Canonical strategy exit mapping
- CPR01 Camarilla-inside reversal: target = CPR-side S1/R1; stop = signal/rejection extreme plus buffer.
- CPR02 R1/PDH/S1/PDL mean reversion: target = CPR; stop = rejection-zone extreme plus buffer.
- CPR03 key levels: treated as a context filter; entries require another trigger family.
- CPR04 candle psychology: target = next configured CPR level or EOD; stop = signal candle extreme plus buffer.
- CPR05 narrow-CPR breakout: target = next CPR resistance/support level; stop = re-entry into CPR plus buffer.
- CPR06 virgin CPR reversal: target = current-session CPR; stop = virgin-zone extreme plus buffer.
- CPR07 combined mean-reversion family: use the selected component's exit mapping above.
- CPR08 narrow/inside/ascending/descending breakout: target = next CPR level; stop = CPR re-entry plus buffer.
- CPR09 short-strangle options: premium stop = 10% of collected premium as stated in transcript; strike/expiry selection remains an explicit config parameter.
- CPR10 MTF day-trade: high-probability trades may run to 15:25 or the nearest CPR resistance/support; low-probability trades use a maximum holding-time limit of 30 minutes.
- CPR11 opening/regime rules: continuation setups target the next CPR level; far-gap reversal targets CPR; stop = structure level plus buffer.
- CPR12 additional masterclass: follow weekly/daily CPR hierarchy, use R2/S2 continuation when enabled, and default to EOD exit; gap-specific stop can be selected explicitly.

## Risk accounting
The primary report will use:
- Initial capital: ₹1,00,000.
- Maximum risk per trade: 1% of current equity.
- Maximum capital allocation: 50% unless strategy config overrides it.
- Compounding on.

## Important limitations
- The transcripts do not define a single universal stop/target/entry-buffer formula for every strategy. Those items are therefore explicit assumptions, not hidden claims about the videos.
- The first-pass research is intended to identify robust strategy families across symbols and regimes. Any parameter optimization must be walk-forward and must not touch the final OOS window.
