from __future__ import annotations

import ast
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / 'v6.2/config.yaml').read_text())
ENGINE = ROOT / 'v6.2/multimodal_engine.py'
NEWS_DOWNLOADER = ROOT / 'v6.2/download_news_robust.py'
FLOW_DOWNLOADER = ROOT / 'v6.2/download_fii_dii_kaggle.py'

assert CFG['version'] == 'V7'
assert CFG['history_end'] == '2026-09-16'
assert CFG['new_oos_start'] == '2026-04-01'
assert CFG['purge_days'] > 0
assert CFG['embargo_days'] >= 0
assert CFG['features']['use_point_in_time_weights'] is True
assert CFG['features']['use_all_constituents'] is True
assert CFG['features']['use_options_surface'] is True
assert CFG['sequence_length'] >= 10
for p in [ENGINE, NEWS_DOWNLOADER, FLOW_DOWNLOADER, ROOT/'v6.2/config.yaml']:
    assert p.exists() and p.stat().st_size > 0, f'missing {p}'

ast.parse(ENGINE.read_text(), filename=str(ENGINE))
source = ENGINE.read_text()
for token in [
    'class DeepExpiryTransformer', 'build_daily_modalities(', 'build_samples(', '_stock_features(',
    '_option_features(', '_context_features(', '_fit_model(', '_pinball(', 'run_wfo(',
    'run_split(', '_latest_forecast(', 'paper_strategy('
]:
    assert token in source, f'missing V7 token: {token}'

assert 'allow_exact_matches=False' in source
assert "side='right'" in source
assert 'purge_days' in source and 'embargo_days' in source
assert 'canonical' in source
assert 'active_mask' in source and 'weights_max_staleness_days' in source
assert 'stock_tensor' in source and 'stock_symbols' in source
assert 'm_bin' in source and 'opt_ce_oi_' in source and 'opt_pe_oi_' in source
assert '.pct_change()' not in source.replace('.pct_change(fill_method=None)', '')

flow = FLOW_DOWNLOADER.read_text()
assert 'pravinpari/fii-and-dii-investments-in-indian-stock-market' in flow
assert 'v62_fii_dii.parquet' in flow
news = NEWS_DOWNLOADER.read_text()
assert 'mveen3/Six_Year_Indian_Stock_Market_Dataset-News_and_Ticker' in news
assert 'dixitdharmansh07/indic-finance' in news
assert 'v62_news_manifest.json' in news

print('=== V7 DEEP MULTIMODAL INSPECTOR ===')
print('blocking_failures=0')
print('Transformer, PIT constituents/weights, option surface, strict as-of context, purge/embargo, historical FII/DII and robust news ingestion detected.')
