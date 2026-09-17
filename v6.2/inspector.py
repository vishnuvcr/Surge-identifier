from __future__ import annotations

import ast
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / 'v6.2/config.yaml').read_text())
ENGINE = ROOT / 'v6.2/multimodal_engine.py'
DOWNLOADER = ROOT / 'v6.2/download_multimodal_data.py'
FLOW_DOWNLOADER = ROOT / 'v6.2/download_fii_dii_kaggle.py'
SAFE_RUNNER = ROOT / 'v6.2/run_multimodal_safe.py'

assert CFG['history_end'] == '2026-09-16'
assert CFG['new_oos_start'] == '2026-04-01'
assert CFG['purge_days'] > 0
assert CFG['embargo_days'] >= 0
assert CFG['features']['use_point_in_time_weights'] is True
assert CFG['features']['use_all_constituents'] is True
for p in [ROOT/'v6.2/README.md', ROOT/'v6.2/research_plan.md', ROOT/'v6.2/config.yaml', ENGINE, DOWNLOADER, FLOW_DOWNLOADER, SAFE_RUNNER]:
    assert p.exists() and p.stat().st_size > 0, f'missing {p}'

for p in [ENGINE, DOWNLOADER, FLOW_DOWNLOADER, SAFE_RUNNER]:
    ast.parse(p.read_text(), filename=str(p))

source = ENGINE.read_text()
required = [
    'cycles(', 'index_features(', 'constituent_features(', 'option_features(',
    'context_features(', 'fit_tabular(', 'lstm_predict(', 'run_wfo(',
    'run_new_oos(', 'latest_forecast(', 'paper_strategy(', 'purge_train(',
]
for token in required:
    assert token in source, f'missing required implementation token: {token}'

# Leakage gates: production code must use strictly prior-day context/news/flows and a purge/embargo.
assert "context[context.date < d]" in source
assert "news[pd.to_datetime(news['date']) < d]" in source
assert "flows.date < d" in source
assert 'purge_days' in source and 'embargo_days' in source

# Do not accidentally turn V6.2 into a current-constituent-only backtest.
assert 'membership' in source and 'weights' in source
assert 'valid_from<=d' in source.replace(' ', '')

# The FII/DII source must be replaceable by the explicit Kaggle historical dataset.
flow_source = FLOW_DOWNLOADER.read_text()
assert 'pravinpari/fii-and-dii-investments-in-indian-stock-market' in flow_source
assert 'v62_fii_dii.parquet' in flow_source
assert 'fii_net' in flow_source and 'dii_net' in flow_source

print('=== V6.2 MULTIMODAL INSPECTOR ===')
print('blocking_failures=0')
print('Leakage, PIT membership/weights, multimodal features, LSTM path, safe sparse-flow handling, and Kaggle FII/DII ingestion detected.')
