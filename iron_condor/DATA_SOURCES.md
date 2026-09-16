# Data sources and limitations

## NSE
NSE publishes F&O bhavcopy and related derivatives reports. The existing project already downloads NSE F&O daily files for futures/options OI and volume. Those EOD files are useful for regime features but are not sufficient on their own to reconstruct an intraday weekly iron-condor entry/exit backtest with realistic option premiums.

For a valid backtest, source contract-level historical option prices (at least timestamp/date, expiry, strike, CE/PE, OHLC or bid/ask/mid, volume and OI). When only end-of-day data are available, the engine should explicitly mark execution as EOD-only and must not pretend to have intraday fills.

## Data hierarchy
1. Exchange-grade historical contract data.
2. Broker/vendor historical option OHLC with documented survivorship and timestamp conventions.
3. NSE EOD OI/volume as supporting features.
4. Derived/estimated IV and Greeks only when the underlying price and option price are observed at the same timestamp.

## Required checks
- Correct weekly expiry calendar by instrument and historical period.
- Strike availability and lot-size changes by date.
- Expired/illiquid contracts and zero-volume observations.
- Bid/ask or a conservative slippage model.
- Corporate actions are less relevant for index options but must be handled if stock options are later included.
- Avoid using future OI, IV, or settlement values to select entry strikes.
