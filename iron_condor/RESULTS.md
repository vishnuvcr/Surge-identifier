# Current status

## Branch
`iron-condor-weekly-research`

## What is implemented
- Isolated weekly iron-condor research specification.
- Parameter grid for short-strike delta, wing width, credit threshold, profit target, stop loss, expiry exit and slippage.
- Payoff calculation and result-summary functions.
- Input validation so the backtest cannot silently run on fabricated premiums.
- Data-source documentation.

## What is not yet claimed
No historical performance result is claimed yet. A robust 5% weekly test requires contract-level historical option prices. The current repository's F&O loader primarily aggregates daily futures OI/volume and option OI into PCR, which is useful for signal features but is not enough to reconstruct actual weekly option entry/exit prices. See `DATA_SOURCES.md`.

## Candidate data paths
- NSE F&O bhavcopy: suitable for daily contract information and settlement/EOD analysis.
- Public research repositories that reconstruct NIFTY option chains from NSE bhavcopy can accelerate validation, but provenance and completeness must be checked before using them as ground truth.
- Intraday historical option data from a broker/vendor can support a realistic 1-minute or finer backtest.

## Next research pass
1. Acquire a long history of NIFTY weekly contract-level option prices.
2. Normalize contracts and expiry calendars.
3. Build chronological weekly lifecycle simulator.
4. Add actual statutory cost model and conservative bid/ask slippage.
5. Walk-forward test strategy families and stress them with larger slippage and tail moves.
6. Publish full weekly return distribution instead of a single headline percentage.
