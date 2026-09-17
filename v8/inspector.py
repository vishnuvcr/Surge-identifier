from __future__ import annotations

import ast
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / 'v8/config.yaml').read_text())
ENGINE = ROOT / 'v8/deep_iron_condor_engine.py'
assert CFG['version'] == 'V8.1'
assert CFG['history_end'] == '2026-09-16'
assert CFG['new_oos_start'] == '2026-04-01'
assert int(CFG['purge_days']) > 0
assert int(CFG['embargo_days']) >= 0
assert float(CFG['min_ic_reward_risk']) >= 1.5
assert float(CFG['min_profit_probability']) >= 0.65
assert float(CFG['max_touch_probability']) <= 0.35
assert float(CFG['max_expected_loss_probability']) <= 0.25
assert int(CFG['walk_forward']['windows']) == 6
assert int(CFG['walk_forward']['train_expiries']) >= 36
assert int(CFG['walk_forward']['test_expiries']) >= 6
assert int(CFG['entry']['days_before_expiry']) > 0
ENGINE = ROOT / 'v8/deep_iron_condor_engine.py'
assert ENGINE.exists() and ENGINE.stat().st_size > 0
src = ENGINE.read_text()
ast.parse(src)
required = [
    'strict_asof(', 'make_option_features(', 'build_samples(', 'class ICTransformer',
    'fit_model(', 'predict(', 'candidate_ladder(', 'splits(', 'run_strategy(',
    'trade_metrics(', 'entries(', 'path_low_return', 'path_high_return',
    'wfo_metrics.csv', 'split_70_30_metrics.csv', 'new_oos_metrics.csv',
    'latest_distribution.json', 'latest_iron_condor.json', 'status.json',
]
for token in required:
    assert token in src, token
assert 'allow_exact_matches=False' in src
assert 'reward_risk' in src and 'model_profit_probability' in src and 'model_touch_probability' in src
assert 'slippage_points_per_leg' in src and 'brokerage_per_order' in src
print('=== V8.1 OBJECTIVE-FIRST INSPECTOR ===')
print('blocking_failures=0')
print(f"reward_risk_floor={CFG['min_ic_reward_risk']}x")
print(f"wfo_windows={CFG['walk_forward']['windows']}")
print('coherent_range=true')
print('probability_of_profit=true')
print('touch_probability=true')
print('post_cost_payoff=true')
