# NSE Surge Identifier

An end-of-day ML research pipeline for identifying NSE equities with an elevated probability of an unusually strong move during the next trading session.

## What it does

- Downloads official NSE CM bhavcopy archives and maintains a rolling local data cache.
- Downloads NSE F&O bhavcopy context and derives futures open-interest and options put/call positioning features when available.
- Builds leakage-safe technical, volume, volatility and cross-sectional features.
- Trains a PyTorch deep-learning classifier on the historical cross-section.
- Uses walk-forward validation and probability-threshold selection.
- Ranks 1–5 liquid candidates for the next session.
- Produces an auditable JSON report with probability, expected-move proxy, derivatives context, risk flags and validation statistics.
- Runs automatically after the NSE session and prepares a GitHub Pages deployment on every scheduled run.

## Event being predicted

The model's primary event is **whether the next trading session's intraday high reaches at least +3% from that session's open**. This creates an objective target that can be evaluated with sensitivity/recall, specificity and precision without leaking the next day's information into the prediction.

## Important research note

The objective is to push the model toward unusually high-quality signals, not to claim perfect classification or guaranteed daily/monthly profits. Historical sensitivity, specificity and precision are published from the walk-forward validation window. A no-trade outcome is allowed when the confidence/liquidity gates are not satisfied.

## GitHub Pages

One repository setting is required once: **Settings → Pages → Source → GitHub Actions**. GitHub's documented custom Pages workflow uses `configure-pages`, `upload-pages-artifact`, and `deploy-pages`, with `pages: write` and `id-token: write` permissions.

The scheduled workflow runs at 18:00 IST on NSE weekdays (`12:30 UTC`) and also supports manual `workflow_dispatch` runs.

## Local run

```bash
pip install -r requirements.txt
python scripts/run_daily.py
```

The main artifacts are written to `site/` and `reports/`.
