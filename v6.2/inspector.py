from __future__ import annotations

import ast
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / 'v6.2/config.yaml').read_text())
ENGINE = ROOT / 'v6.2/multimodal_engine.py'
DOWNLOADER = ROOT / 'v6.2/download_multimodal_data.py'

assert CFG['history_end'] == '2026-09-16'
assert CFG['new_oos_start'] == '2026-04-01'
assert CFG['purge_days'] > 0
assert CFG['embargo_days'] >= 0
assert CFG['features']['use_point_in_time_weights'] is True
assert CFG['features']['use_all_constituents'] is True
for p in [ROOT/'v6.2/README.md', ROOT/'v6.2/research_plan.md', ROOT/'v6.2/config.yaml', ENGINE, DOWNLOADER]:
    assert p.exists() and p.stat().st_size > 0, f'missing {p}'

for p in [ENGINE, DOWNLOADER]:
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

print('=== V6.2 MULTIMODAL INSPECTOR ===')
print('blocking_failures=0')
print('Leakage, PIT membership/weights, multimodal features, tabular ML and LSTM paths detected.')
