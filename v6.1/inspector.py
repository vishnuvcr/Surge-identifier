from __future__ import annotations

import ast
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'v6.1/nifty_prediction_engine.py'
CFG = ROOT / 'v6.1/config.yaml'
WF = ROOT / '.github/workflows/v6.1-nifty-prediction.yml'


def fail(msg, out):
    out.append(msg)
    print('BLOCK:', msg)


def main():
    failures=[]
    for p in [ENGINE, CFG, WF]:
        if not p.exists():
            fail(f'missing {p.relative_to(ROOT)}', failures)
    if failures:
        return 1
    cfg=yaml.safe_load(CFG.read_text())
    text=ENGINE.read_text()
    wf=WF.read_text()
    try:
        ast.parse(text)
    except SyntaxError as e:
        fail(f'engine syntax error: {e}', failures)
    if cfg['new_oos_start'] != '2026-04-01':
        fail('new OOS start must remain 2026-04-01', failures)
    if cfg['history_end'] != '2026-09-16':
        fail('history end must remain 2026-09-16', failures)
    for required in ['completed_cycles','fit_predict','run_wfo','run_split','run_new_oos','latest_forecast','target_return','target_close']:
        if required not in text:
            fail(f'missing prediction component: {required}', failures)
    if 'options strategy' in text.lower() or 'strategy_map' in text:
        fail('prediction engine must not contain options strategy selection logic', failures)
    if "entry_date" in text and "target_date" not in text:
        fail('target-date handling missing', failures)
    if "history_end" not in text:
        fail('history-end leakage gate missing', failures)
    if 'python v6.1/nifty_prediction_engine.py' not in wf:
        fail('workflow does not execute V6.1 engine', failures)
    if 'PYTHONPATH' not in wf:
        fail('workflow PYTHONPATH gate missing', failures)
    if 'latest_forecast.json' not in wf:
        fail('latest forecast output missing from workflow', failures)
    if 'new_oos_model_metrics.csv' not in wf:
        fail('new OOS model metrics output missing', failures)
    print('=== V6.1 NIFTY PREDICTION INSPECTOR ===')
    print('Blocking failures:', len(failures))
    for x in failures:
        print('FAIL:', x)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
