from __future__ import annotations

import ast
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
CFG=yaml.safe_load((ROOT/'v8/config.yaml').read_text())
ENGINE=ROOT/'v8/deep_iron_condor_engine.py'
assert CFG['version']=='V8'
assert CFG['history_end']=='2026-09-16'
assert CFG['new_oos_start']=='2026-04-01'
assert int(CFG['purge_days'])>0
assert int(CFG['embargo_days'])>=0
assert float(CFG['min_ic_reward_risk'])>=1.5
assert float(CFG['min_profit_probability'])>=0.5
assert float(CFG['max_touch_probability'])<0.5
assert ENGINE.exists() and ENGINE.stat().st_size>0
src=ENGINE.read_text(); ast.parse(src)
for token in ['class ICTransformer','pinball(','candidate_ics(','payoff_ic(','reward_risk(','strict_asof(','fit_model(','predict(']:
    assert token in src, token
assert 'width-net_credit' in src
assert 'max_profit=net_credit*lot' in src
assert 'max_loss=(width-net_credit)*lot' in src
assert 'allow_exact_matches=False' in src
print('=== V8 IRON CONDOR INSPECTOR ===')
print(f"blocking_failures=0 reward_risk_floor={CFG['min_ic_reward_risk']}x")
