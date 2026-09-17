from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'v6.2/research'
CFG = ROOT / 'v6.2/config.yaml'

forecast_path = OUT / 'latest_forecast.json'
summary_path = OUT / 'summary.json'
assert forecast_path.exists() and forecast_path.stat().st_size > 0
f = json.loads(forecast_path.read_text())
s = json.loads(summary_path.read_text())
assert f.get('available') is True
assert f.get('version') == 'V7'
assert f.get('forecast_feature_date_verified') is True
assert f.get('forecast_type') == 'fresh_current_cutoff_deep_multimodal'
assert pd.Timestamp(f['latest_feature_date']) == pd.Timestamp(f['signal_date'])
assert pd.Timestamp(f['next_expiry']) > pd.Timestamp('2026-09-16')
assert s.get('selected_production_model') == 'deep_transformer'
print(json.dumps({'latest_forecast': f, 'research_gate': s.get('research_gate')}, indent=2))
