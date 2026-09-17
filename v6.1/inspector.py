from __future__ import annotations

import ast
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'v6.1/nifty_prediction_engine_v2.py'
CFG = ROOT / 'v6.1/config.yaml'
WF = ROOT / '.github/workflows/v6.1-nifty-prediction.yml'


def main():
    failures = []
    for p in [ENGINE, CFG, WF]:
        if not p.exists():
            failures.append(f'missing {p.relative_to(ROOT)}')
    if failures:
        for x in failures: print('BLOCK:', x)
        return 1
    cfg = yaml.safe_load(CFG.read_text())
    text = ENGINE.read_text()
    wf = WF.read_text()
    try:
        ast.parse(text)
    except SyntaxError as e:
        failures.append(f'engine syntax error: {e}')
    if cfg['new_oos_start'] != '2026-04-01': failures.append('new OOS start must remain 2026-04-01')
    if cfg['history_end'] != '2026-09-16': failures.append('history end must remain 2026-09-16')
    for required in ['completed_cycles','fit_predict','run_wfo','run_split','run_new_oos','latest_forecast','target_return','target_close']:
        if required not in text: failures.append(f'missing prediction component: {required}')
    if 'options strategy' in text.lower() or 'strategy_map' in text: failures.append('prediction engine must not contain options strategy selection logic')
    if 'history_end' not in text: failures.append('history-end leakage gate missing')
    if 'python v6.1/nifty_prediction_engine_v2.py' not in wf: failures.append('workflow does not execute V6.1 engine v2')
    if 'PYTHONPATH' not in wf: failures.append('workflow PYTHONPATH gate missing')
    if 'latest_forecast.json' not in wf: failures.append('latest forecast output missing from workflow')
    if 'new_oos_model_metrics.csv' not in wf: failures.append('new OOS model metrics output missing')
    print('=== V6.1 NIFTY PREDICTION INSPECTOR ===')
    print('Blocking failures:', len(failures))
    for x in failures: print('FAIL:', x)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
