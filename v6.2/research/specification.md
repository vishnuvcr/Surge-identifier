# V6.2 — Multimodal NIFTY Expiry Research

## Objective
Build and test a leakage-controlled multimodal forecasting system for the NIFTY 50 next-expiry settlement/close, with a separate options-strategy evaluation layer.

The project is research-first: every model must be compared against simple baselines and evaluated using chronological walk-forward and a completely untouched OOS period. No model or strategy is promoted because of in-sample performance alone.

## Research hypothesis
A multimodal representation combining constituent-level NIFTY 50 dynamics, market-wide technical/regime features, institutional flows, futures/options microstructure, global context and point-in-time news sentiment should improve next-expiry NIFTY return/close forecasting versus price-only baselines, provided that all feature timestamps precede the prediction cutoff and validation is purged/embargoed where label windows overlap.

The uploaded research review motivates this hypothesis through free-float index construction, heavy constituent influence, FII/DII flows, RSI/MACD/CPR/HMA, option-chain Max Pain/PCR/GEX/walls, FinBERT sentiment, XGBoost feature selection, TFT/TKAN, wavelet/CNN/LSTM hybrids, purged time-series validation, transaction-cost modeling and GitHub Actions/Pages automation. See the source PDF supplied with the project. fileciteturn227file0

## Feature groups
1. Index/price: OHLCV, returns, realized volatility, ATR, RSI, MACD, EMA gaps, HMA, CPR, breakout/breakdown, trend slope.
2. Constituents: all available NIFTY 50 constituents with point-in-time membership/weights; weighted returns, volatility, breadth, dispersion, sector aggregates, top-contributor shocks and concentration measures.
3. Institutional: FII/DII cash flows; rolling 1/5/20/60-day flows; standardized flows; derivative positioning where historical coverage permits.
4. Futures: basis, futures return, OI/change-OI, volume, roll/basis regime.
5. Options: expiry-aligned chain features; PCR OI/volume, Max Pain and spot-distance, call/put walls, OI concentration, IV/skew/term structure, ATM straddle, estimated dealer gamma exposure where reliable inputs exist.
6. Cross-asset: India VIX, USDINR, Brent, gold, US indices/volatility, relevant rates/yield proxies.
7. News/NLP: point-in-time NIFTY/constituent/global macro headlines; FinBERT polarity plus volume, dispersion, negative-share and aspect/event tags where feasible.
8. Calendar: DTE, weekday, month/quarter-end, expiry cycle, scheduled macro-event indicators.

## Targets
Primary: next-expiry NIFTY settlement/close and return.
Secondary: probability of bullish/range/bearish regime, conditional quantiles for the expiry close, and a calibrated prediction interval.
Optional extension: expiry-day intraday high/low and realized range, using only data available before the corresponding forecast timestamp.

## Model ladder
A. Naive baselines: last price, historical mean return, volatility-scaled zero return.
B. Statistical/linear: Ridge/ElasticNet.
C. Tree ensembles: Random Forest, Extra Trees, HistGradientBoosting, XGBoost/LightGBM where installation permits.
D. Sequence models: LSTM/GRU.
E. TFT using PyTorch Forecasting.
F. Optional TKAN/hybrid model only after strong baselines pass leakage and stability checks.
G. Ensemble selected only from OOS/WFO-valid predictions; no test-period tuning.

## Validation protocol
- Chronological expiry-cycle samples.
- Expanding/rolling walk-forward windows.
- Purge/embargo whenever training labels can overlap validation prediction horizons.
- 70/30 chronological split as a secondary benchmark.
- Completely new OOS period beginning 2026-04-01, never used for feature selection, hyperparameter tuning, production-model selection or strategy selection.
- Report MAE/RMSE, directional accuracy, regime accuracy, correlation, calibration/coverage and error by market regime.
- Compare against naive baselines.
- Bootstrap confidence intervals where sample size permits.

## Options layer
Only after prediction research is validated. Evaluate a rules-based strategy mapper using the forecast distribution/regime and actual option chain. Candidate families include debit spreads, credit spreads, straddles/strangles, iron condors/butterflies and asymmetric structures. Leg selection must be explicit and payoff-invariant.

Backtests must use historical bid/ask or a documented conservative proxy, current-at-the-time lot sizes, brokerage/exchange charges, GST, STT, SEBI fees, stamp duty and slippage. Never use today's chain to construct historical trades.

## Automation
GitHub Actions should:
1. download/refresh permitted datasets;
2. run data-quality and leakage tests;
3. build features;
4. train/evaluate models;
5. generate latest forecast;
6. run options strategy simulation separately;
7. produce CSV/JSON/HTML research artifacts;
8. publish a GitHub Pages dashboard.

No live order execution is part of V6.2. The output is a research/paper-trading signal until separately validated.

## Acceptance gates
A model can be promoted only if:
- no feature is timestamped after its prediction cutoff;
- constituent membership/weights are point-in-time correct;
- validation is chronological and purged where necessary;
- untouched OOS results are preserved;
- it beats or meaningfully complements naive baselines on OOS metrics;
- calibration is reported for probabilistic outputs;
- options profitability, if any, survives realistic costs and slippage.
