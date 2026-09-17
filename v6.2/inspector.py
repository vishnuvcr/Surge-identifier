from pathlib import Path
import ast, yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT/'v6.2/config.yaml').read_text())

assert CFG['history_end'] == '2026-09-16'
assert CFG['new_oos_start'] == '2026-04-01'
assert CFG['purge_days'] > 0
assert CFG['embargo_days'] >= 0
assert CFG['features']['use_point_in_time_weights'] is True

for p in [ROOT/'v6.2/README.md', ROOT/'v6.2/research_plan.md', ROOT/'v6.2/config.yaml']:
    assert p.exists() and p.stat().st_size > 0, f'missing {p}'

# Compile every Python file already used by the prediction baseline.
for p in [ROOT/'v6.1/nifty_prediction_engine_v2.py']:
    ast.parse(p.read_text(), filename=str(p))

print('=== V6.2 INSPECTOR ===')
print('blocking_failures=0')
print('Research is specification-ready; prediction engine implementation is the next phase.')
