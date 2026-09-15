from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REQUIRED = [
    'numpy', 'pandas', 'pyarrow', 'yaml', 'sklearn', 'torch', 'requests'
]


def check_imports():
    bad = []
    for name in REQUIRED:
        try:
            importlib.import_module(name)
        except Exception as e:
            bad.append(f'{name}: {type(e).__name__}: {e}')
    if bad:
        raise RuntimeError('Dependency preflight failed:\n' + '\n'.join(bad))


def check_files():
    paths = [
        ROOT / 'backtest/config_phase62.yaml',
        ROOT / 'backtest/config_phase61.yaml',
        ROOT / 'backtest/run_backtest_phase61.py',
        ROOT / 'backtest/run_backtest_phase62.py',
        ROOT / 'src/moe_engine_phase61.py',
        ROOT / 'src/surge_model.py',
        ROOT / 'src/nse_data.py',
        ROOT / 'src/nse_fo.py',
    ]
    missing = [str(p.relative_to(ROOT)) for p in paths if not p.exists()]
    if missing:
        raise RuntimeError('Required files missing: ' + ', '.join(missing))


def check_config():
    cfg = yaml.safe_load((ROOT / 'backtest/config_phase62.yaml').read_text())
    required = ['initial_capital','lookback_days','training_lookback_sessions','portfolio_sizes','target_levels','stop_loss_pct','costs']
    missing = [x for x in required if x not in cfg]
    if missing:
        raise RuntimeError('Phase 6.2 config missing keys: ' + ', '.join(missing))
    if int(cfg['lookback_days']) <= int(cfg['training_lookback_sessions']):
        raise RuntimeError('lookback_days must exceed training_lookback_sessions')
    if not cfg['target_levels']:
        raise RuntimeError('No target levels configured')
    if float(cfg['stop_loss_pct']) <= 0:
        raise RuntimeError('Invalid stop loss')


def check_block():
    start = os.environ.get('PHASE62_BLOCK_START')
    end = os.environ.get('PHASE62_BLOCK_END')
    if not start or not end:
        raise RuntimeError('PHASE62_BLOCK_START/END not set')
    if start >= end:
        raise RuntimeError(f'Invalid block: {start} >= {end}')


def main():
    print('=== Phase 6.2 PRE-FLIGHT ===', flush=True)
    check_imports(); print('imports: OK', flush=True)
    check_files(); print('files: OK', flush=True)
    check_config(); print('config: OK', flush=True)
    check_block(); print('block: OK', flush=True)
    print('PRE-FLIGHT PASSED', flush=True)


if __name__ == '__main__':
    main()
