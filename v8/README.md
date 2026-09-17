# V8 Deep ML Iron Condor Research

Goal: train and evaluate deep-learning models to construct defined-risk NIFTY iron condors from predicted expiry range and touch-risk distributions.

Core principle: profit/loss is determined exactly from the four option legs. Candidate structures are evaluated with realistic premiums, lot size, slippage, brokerage, exchange charges, STT, stamp duty and GST. The research optimizer prefers high reward-to-risk candidates but never sacrifices minimum probability-of-profit and tail-risk constraints merely to maximize the ratio.

## Model targets

1. Expiry close distribution.
2. Expiry-day high distribution.
3. Expiry-day low distribution.
4. Probability of touching candidate put/call strikes before expiry.
5. Joint range probability for the selected short strikes.

## Candidate IC

- Buy lower put
- Sell higher put
- Sell lower call
- Buy higher call

For equal wings:
- Gross max profit = net credit * lot size.
- Gross max loss = (wing width - net credit) * lot size.
- Reward/risk = net credit / (wing width - net credit).

The optimizer will screen for configurable minimum reward/risk (default target >= 1.5x), while separately requiring acceptable modeled loss probability and sufficient probability of remaining inside the short strikes.

## Evaluation

- Chronological train/test split.
- 6-window expiry-level walk-forward validation.
- New OOS from 2026-04-01 through the available historical cutoff.
- Baselines: delta/range heuristics and naive volatility/range models.
- No option-strategy performance claim without post-cost OOS evidence.
