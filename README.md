# NSE Surge Identifier

An end-of-day ML research pipeline for identifying NSE equities with an elevated probability of an unusually strong move during the next trading session.

## What it does

- Downloads official NSE CM bhavcopy archives and maintains a rolling local data cache.
- Builds leakage-safe technical, volume, volatility and cross-sectional features.
- Trains a PyTorch deep-learning classifier on the historical cross-section.
- Uses a walk-forward validation split and probability threshold selection.
- Ranks 1–5 liquid candidates for the next session.
- Produces an auditable JSON/CSV report with model probability, expected move, risk flags and validation statistics.
- Runs automatically after the NSE session and publishes the latest report to GitHub Pages.

## Important research note

The pipeline is optimized for high-quality candidate selection, not a promise of 100% sensitivity, 100% specificity, or positive returns every day. It reports those metrics from genuinely out-of-sample validation and will suppress trades when the model does not meet its confidence/liquidity gates.

## GitHub Pages

After enabling **Settings → Pages → Source → GitHub Actions**, the scheduled workflow deploys the `site/` output automatically. GitHub's documented Pages deployment requires `pages: write`, `id-token: write`, a `github-pages` environment, and `configure-pages`/`upload-pages-artifact`/`deploy-pages`. 

## Local run

```bash
pip install -r requirements.txt
python scripts/run_daily.py
```

The main artifacts are written to `site/` and `reports/`.
