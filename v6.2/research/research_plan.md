# V6.2 Research Plan

## Phase 0 — Data audit
- Freeze source URLs/versions and retrieval timestamps.
- Build point-in-time NIFTY 50 membership and free-float weight history.
- Audit every dataset for start/end dates, duplicates, missingness and publication/release timestamps.
- Define a feature availability timestamp for every observation.

## Phase 1 — Baselines
Run next-expiry return/close prediction with:
- naive last-value/zero-return;
- historical mean return;
- Ridge;
- Random Forest;
- Extra Trees;
- HistGradientBoosting.

## Phase 2 — Feature ablation
Evaluate incrementally:
1. index-only;
2. index + constituents;
3. + technical/regime;
4. + FII/DII;
5. + futures;
6. + options microstructure;
7. + cross-asset;
8. + news sentiment;
9. full multimodal.

Every addition is measured on the same chronological folds, using identical target and scoring definitions.

## Phase 3 — Deep learning
Only after tabular baselines are established:
- LSTM/GRU;
- TFT;
- optional TKAN;
- optional DWT-CNN-LSTM hybrid.

Use sequence lengths and hyperparameters selected only from training folds. Do not tune on the new OOS period.

## Phase 4 — Probability/range forecasting
Produce:
- point forecast;
- P(BULLISH), P(RANGE_BOUND), P(BEARISH);
- lower/median/upper quantiles;
- empirical coverage of prediction intervals.

Compare nominal versus realized coverage on OOS data.

## Phase 5 — Options strategy research
For each forecast, map to explicitly defined candidate structures. Evaluate:
- entry time;
- strike selection;
- expiry;
- quantity/lot size;
- premium paid/received;
- maximum loss;
- maximum profit;
- breakevens;
- realistic execution assumptions;
- transaction costs;
- slippage;
- exit rules.

Strategy selection must be learned only from historical training/WFO data and then frozen for each OOS segment.

## Phase 6 — Live paper-trading output
GitHub Actions generates a daily/expiry-cycle JSON and static dashboard containing:
- current spot;
- forecast close/range;
- regime probabilities;
- model/version;
- confidence/calibration information;
- feature availability timestamp;
- candidate paper-trades with full payoff details;
- explicit 'NO TRADE' state when confidence or data quality gates fail.

## Primary scientific test
H0: Multimodal features do not improve next-expiry NIFTY forecasting versus the strongest price-only baseline.

H1: Multimodal features improve out-of-sample forecasting performance and/or calibration by a statistically and economically meaningful amount, without leakage.

The decision criterion is based on untouched OOS error, direction/regime metrics, calibration and stability across walk-forward windows—not a single backtest equity curve.
