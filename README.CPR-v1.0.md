# CPR v1.0 — Strategy Research Branch

## Scope

This branch is dedicated to converting and testing the user-supplied CPR / price-action videos across:

- Intraday LONG and SHORT
- BTST
- Swing
- NIFTY 50
- NIFTY 100
- NIFTY 500
- NIFTY index/options strategies

The objective is empirical validation. The requested 10% weekly and 30–40% monthly return levels are treated as research targets, not assumptions or guarantees.

## Videos supplied

1. https://youtu.be/gJDe4H7C_oI
2. https://youtu.be/jip0tBDHLHM
3. https://youtu.be/DEJk1kUJo68
4. https://youtu.be/fWFK-Gn0AiE
5. https://youtu.be/1qgI1qOfJaI
6. https://youtu.be/xIXI0NKTygg
7. https://youtu.be/Q1Pq2LYQ8fM
8. https://youtu.be/Fvem1Ai2Z8w
9. https://youtu.be/x_VT3HALjB8
10. https://youtu.be/pUOoDLdJSA4
11. https://youtu.be/0_349o_Y7O4

## Strategy extraction status

Public search results currently expose reliable title/context for some of the supplied CPR BY KGS videos, including:

- Most Powerful CPR strategy — breakout and reversal
- Secrets of Price Action Trading | CPR Strategy
- Why Virgin CPR Beats Average Trading Strategies Every Time
- 3 Common Mistakes Using Candlestick Patterns That Ruin Your Trading

Direct YouTube page/transcript retrieval is not available in the current execution environment. Therefore the remaining video rules must not be inferred or fabricated. Each strategy must be converted from its actual stated rules before backtesting.

## Planned test design

For each strategy, build an explicit rule card containing:

- instrument/universe
- timeframe
- entry conditions
- confirmation conditions
- long rules
- short rules
- target
- stop
- trailing/exit rules
- session cutoff
- BTST/swing holding rules
- option contract selection rules
- position sizing
- transaction costs/slippage

Backtests will use point-in-time safe data and will report:

- CAGR / total return where applicable
- weekly return distribution
- monthly return distribution
- hit rate
- expectancy
- profit factor
- max drawdown
- worst week/month
- consecutive losses
- turnover
- exposure
- trade count
- option-specific cost and liquidity effects

Walk-forward / OOS validation is required for any ML-enhanced version. No parameter optimization may use the final OOS period.
