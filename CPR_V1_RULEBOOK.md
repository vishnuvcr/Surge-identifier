# CPR v1.0 — Transcript-Derived Rulebook

This branch converts the user-supplied video transcripts into explicit, backtestable rule families. The wording below is a research specification, not a claim that the strategies work.

## 1. CPR + Camarilla R3/S3 inside CPR reversal
Source: gJDe4H7C_oI
- Calculate current-session daily CPR from the prior session OHLC.
- Calculate prior-session Camarilla R3 and S3.
- Setup: R3 lies inside the CPR => monitor reversal short near/at R3. S3 lies inside CPR => monitor reversal long near/at S3.
- Entry confirmation from transcript: wait for a 5-minute candle close back out of/through the CPR in the reversal direction; do not blindly enter on the first candle.
- Targets mentioned: opposite Camarilla S3/R3, CPR S1/R1, and previous day low/high.
- Explicitly parameterise target selection and stop placement because the transcript does not define one universal stop formula.

## 2. CPR mean reversion from R1/PDH or S1/PDL
Source: DEJk1kUJo68
- Daily CPR plus prior-day high (PDH) and prior-day low (PDL).
- Short setup: price breaks/extends above the R1+PDH resistance zone, then returns below that zone with bearish confirmation; target current CPR.
- Long setup: price breaks/extends below the S1+PDL support zone, then returns above that zone with bullish confirmation; target current CPR.
- Weekly version: use weekly CPR with prior-week high/low on hourly charts.
- Monthly version: use monthly CPR with prior-month high/low on daily charts.
- Buy-on-dips interpretation: in an ascending monthly CPR regime, long when price retests/supports monthly CPR.

## 3. Key price-action levels with CPR
Source: jip0tBDHLHM
- Plot PDH/PDL, previous-week high/low, previous-month high/low and 52-week high/low.
- Use these as directional context and position-sizing/risk-management levels.
- Intraday bullish context: price remains above the relevant prior-period level and above CPR/R1/R2; bearish context is the inverse.
- Break-and-hold above/below the level increases confidence; failed breaks/retests are treated as warning/reversal context.
- Position sizing is directional/contextual rather than a fixed signal by itself; executable entries must be paired with one of the CPR trigger strategies.

## 4. Candlestick psychology + key candle break
Source: fWFK-Gn0AiE
- Identify a strong bullish daily candle. Mark its high/low and 50% level.
- Bullish continuation condition: following price breaks the strong bullish candle high.
- Bearish confirmation condition after a strong bullish candle: price moves below its 50% area and then breaks the low of a subsequent bearish confirmation candle.
- Mirror logic for a strong bearish candle.
- Intraday implementation: carry the marked higher-timeframe level to 5-minute data and trade breaks/retests consistent with the higher-timeframe bias.
- Exact stop/target remains a configurable assumption where not explicitly defined in the transcript.

## 5. Narrow CPR breakout / wide CPR breakout filter
Source: 1qgI1qOfJaI
- Narrow CPR: prior-session CPR width is small => breakout/breakdown setups are permitted.
- Wide CPR: breakout/breakdown setups are filtered out as low-quality/fake-breakout candidates.
- Intraday trigger: break above resistance for long or below support for short.
- Higher timeframe adaptation: daily chart with monthly CPR; hourly chart with weekly CPR; 5-minute chart with daily CPR.
- Transcript says medium CPR can also be acceptable on higher timeframes, so width bands are configurable.

## 6. Virgin CPR reversal
Source: xIXI0NKTygg
- A prior-session CPR is "virgin" when price did not touch/enter that CPR during the originating session.
- Extend the virgin CPR zone into following sessions.
- When price later reaches the virgin CPR, treat it as a reversal zone rather than a breakout/continuation entry.
- The transcript explicitly warns against using virgin CPR as a simple momentum target; the trigger is reversal at/around the virgin zone.
- Track age of virgin CPR; transcript says the effect weakens as more sessions pass.

## 7. Combined CPR mean-reversion family
Source: Q1Pq2LYQ8fM
- Combined families: trend-reversal CPR, virgin CPR, Camarilla-inside-CPR, and CPR-attract-candle.
- Trend-reversal CPR: use ascending/descending CPR structure as directional bias; reversal at CPR/outer bands is the trigger.
- Camarilla-inside-CPR follows Strategy 1.
- Virgin CPR follows Strategy 6.
- CPR-attract candle: monitor a prior candle high/low together with CPR and prior-day high/low; trade the relevant break/retest. Exact candle definition is ambiguous in the supplied auto-caption transcript and is therefore kept configurable.

## 8. Narrow / Inside / Ascending / Descending CPR breakout family
Source: Fvem1Ai2Z8w
- Narrow CPR breakout: follow Strategy 5.
- Inside CPR: detect CPR compression/inside relationship between consecutive sessions and allow directional breakout setups.
- Ascending CPR: current CPR is above prior CPR; favor long breakouts/support retests.
- Descending CPR: current CPR is below prior CPR; favor short breakdowns/resistance retests.
- Use R1/R2/R3/R4 and S1/S2/S3/S4 as progressively farther levels.

## 9. CPR + options short strangle
Source: x_VT3HALjB8
- Narrow CPR day is treated as directional/trending context; wide/sideways CPR is the preferred environment for the transcript's non-directional short-strangle concept.
- Transcript example: sell OTM call and put around the selected premium, with a 10% premium stop-loss.
- Because the transcript does not fully specify strike-selection, expiry-selection, adjustment rules, and exact entry time, these are configuration parameters and must not be filled in silently.
- Must model option liquidity, slippage, bid/ask, expiry, lot size, and assignment/exercise mechanics before performance is considered usable.

## 10. Multi-timeframe CPR day-trading system
Source: pUOoDLdJSA4
- Day trading timeframes: hourly chart with weekly CPR + 5-minute chart with daily CPR.
- High-probability long: price above weekly CPR AND above daily CPR; enter on support at daily CPR or break of R1 + PDH.
- High-probability short: price below weekly CPR AND below daily CPR; enter on resistance at daily CPR or break of S1 + PDL.
- High-probability trades can be held until 15:30 IST or exited at nearby CPR resistance/support.
- Low-probability countertrend/scalp: weekly and daily CPR disagree. Limit holding period (transcript says less than ~30 minutes) and use smaller capital allocation.
- The transcript mentions 50% capital for high-probability and 25% for low-probability trades; this will be tested as a risk-budget input, not as a recommendation.

## 11. CPR masterclass / opening and regime rules
Source: 0_349o_Y7O4 visual notes
- Narrow CPR => momentum/trend day candidate; wide CPR => range/mean-reversion candidate.
- Virgin CPR acts as a strong reaction zone.
- Ascending CPR => bullish bias; descending CPR => bearish bias.
- Open between CPR and R1; break above R1 => bullish continuation candidate.
- Open far above R1 => overextension/reversion candidate toward CPR.
- Open between CPR and S1; break below S1 => bearish continuation candidate.
- Open far below S1 => overextension/reversion candidate toward CPR.

## 12. CPR multi-rule backtest workflow from additional transcript
Source: jzSV1IiDwAc
- 5-minute execution with weekly CPR + daily CPR.
- If price is above weekly CPR: look to buy above narrow daily CPR.
- If price is below weekly CPR: look to sell below narrow daily CPR.
- If price breaks weekly R2: permit long continuation above narrow daily CPR.
- If price breaks weekly S2: permit short continuation below narrow daily CPR.
- Avoid gaps unless gap-handling is explicitly enabled by the test configuration.
- If both daily and weekly CPR are wide: no-trade.
- If daily CPR collapses to a single/near-single line: optional no-trade filter.
- Entry confirmation example: break of first 5-minute candle high/low, with a configurable buffer.
- Exits: daily CPR R2/S2, weekly R1/R2/S1/S2, or end-of-day; exact priority is configurable.
- Risk: use CPR/gap/structure-based stop; the transcript itself uses several variants, so backtests must report which stop mode was used.

## Research discipline
- These rules are derived from the supplied transcripts. No performance claim is implied.
- Ambiguous items are parameterised and logged instead of being silently invented.
- All model/parameter selection must be inside training/validation data; the final OOS period remains untouched.
- Intraday rules require point-in-time 5-minute data. Daily data alone must not be presented as a true 5-minute execution backtest.
