# Single-stock Open ML Walk-Forward Branch

Branch: `single-stock-open-ml-walkforward`

## Objective
Research a configurable NSE stock for next-session entry at the open, targeting an intraday move in the approximate 2.5–3% range. The system estimates both:

1. `P(next-session high >= target from next-session open)`
2. `expected next-session open-to-high return`

The research engine is designed to answer whether this setup has stable out-of-sample edge. It does **not** assume that a 2.5–3% daily return or 30% monthly return is achievable.

## Walk-forward design

- Training window: 504 trading days by default.
- OOS test block: 21 trading days.
- Refit every 21 trading days.
- No shuffling across time.
- Features are calculated using information available before the forecasted session.
- The open-to-high target uses the following session's OHLC only as the label, never as an input.
- If both target and stop are touched within the same daily candle, the engine uses a conservative stop-first assumption because OHLC alone cannot establish intraday order.

## Feature layers

### Stock
Gap, multi-horizon returns, ATR, RSI, volatility, relative volume, trend, range/body structure, distance from recent extremes, skew.

### Derivatives
Futures OI, OI change, futures volume, put/call OI ratio and rolling standardized/deviation measures from the existing NSE F&O archive loader.

### Market / regime
NIFTY return/gap, India VIX change, sector return and market breadth fields are part of the schema. The first implementation currently uses explicit neutral placeholders until dated index/VIX/sector data adapters are wired in.

### CPR / price structure
Previous-day pivot/CPR width, price location vs pivot, R1/S1 distances and previous-day range.

### News / corporate actions
The schema contains `news_score`, `news_count`, and `corporate_action_flag`. These must be populated from timestamped historical sources before being used in a production/OOS conclusion. The runner currently initializes them to neutral values rather than introducing look-ahead or undocumented proxies.

## Run locally

```bash
python scripts/run_single_stock_oos.py \
  --symbol RELIANCE \
  --target 0.0275 \
  --stop 0.0125 \
  --probability-gate 0.60
```

## GitHub Actions

Use the `Single Stock ML Walk-Forward OOS` workflow and select the symbol/target/stop/probability gate. The workflow uploads CSV/JSON OOS artifacts.

## Important interpretation

A 30% monthly result requires repeated compounding and can be produced by very different combinations of hit rate, average win, average loss and trade frequency. The OOS report therefore exposes hit rate, mean/median trade return, compounded return, drawdown, AUC, average precision and prediction error instead of optimizing directly for a headline monthly percentage.
