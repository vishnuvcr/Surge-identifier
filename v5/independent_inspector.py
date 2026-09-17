from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / '.github/workflows/v5.1-regime-adaptive.yml'
INSPECTOR_WORKFLOW = ROOT / '.github/workflows/v5.1-independent-inspector.yml'
SAFE = ROOT / 'v5/run_v5_1_safe.py'
BASE = ROOT / 'v5/run.py'
MEMORY = ROOT / 'v5/inspector_memory.json'

REQUIRED = [BASE, SAFE, ROOT / 'v5/config.yaml', ROOT / 'v5/context_config.yaml', ROOT / 'scripts/download_nifty_nse_history.py', ROOT / 'scripts/download_v5_market_context.py', ROOT / 'scripts/download_v5_flow_context.py', MAIN, INSPECTOR_WORKFLOW, MEMORY]
PY_FILES = [BASE, SAFE, ROOT / 'scripts/download_nifty_nse_history.py', ROOT / 'scripts/download_v5_market_context.py', ROOT / 'scripts/download_v5_flow_context.py', ROOT / 'v5/independent_inspector.py']

HISTORICAL_UNSUPPORTED_FEATURES = {
    'fii_net_z', 'dii_net_z', 'fii_fut_long_short', 'pcr', 'flow_sentiment'
}


def fail(msg: str, failures: list[str]) -> None:
    failures.append(msg)
    print(f'BLOCK: {msg}')


def warn(msg: str, warnings: list[str]) -> None:
    warnings.append(msg)
    print(f'WARN: {msg}')


def main() -> int:
    failures: list[str] = []
    warnings: list[str] = []
    print('=== V5.1 INDEPENDENT INSPECTOR ===')

    for p in REQUIRED:
        if not p.exists():
            fail(f'missing required file: {p.relative_to(ROOT)}', failures)
    if failures:
        return 1

    try:
        memory = json.loads(MEMORY.read_text())
    except Exception as exc:
        fail(f'inspector memory cannot be parsed: {exc}', failures)
        memory = {}
    lessons = memory.get('lessons', [])
    if len(lessons) < 5:
        fail('inspector memory contains too few learned failure rules', failures)
    else:
        print(f'Loaded {len(lessons)} learned failure rules.')

    for p in PY_FILES:
        r = subprocess.run(['python', '-m', 'py_compile', str(p)], cwd=ROOT, text=True, capture_output=True)
        if r.returncode:
            fail(f'py_compile failed for {p.relative_to(ROOT)}: {r.stderr.strip()}', failures)
        else:
            print(f'OK compile: {p.relative_to(ROOT)}')

    main_text = MAIN.read_text()
    inspector_text = INSPECTOR_WORKFLOW.read_text()
    safe_text = SAFE.read_text()
    base_text = BASE.read_text()

    if 'uses: ./.github/workflows/v5.1-independent-inspector.yml' not in main_text:
        fail('main V5.1 workflow is not gated by the independent inspector', failures)
    if not re.search(r'backtest:\s*\n\s*needs:\s*inspector', main_text):
        fail('backtest job does not depend on inspector job', failures)
    if 'workflow_call:' not in inspector_text:
        fail('independent inspector workflow is not reusable via workflow_call', failures)
    if 'timeout-minutes: 5' not in inspector_text:
        fail('inspector does not have a bounded fail-fast runtime', failures)

    if re.search(r'^\s*run:\s*python v5/run\.py\s*$', main_text, re.MULTILINE):
        fail('workflow directly executes python v5/run.py instead of the guarded runner', failures)
    if 'run: python v5/run_v5_1_safe.py' not in main_text:
        fail('guarded V5.1 runner is not the production backtest command', failures)

    if "n.columns=[str(c).strip().lower() for c in n.columns]" not in main_text:
        fail('news Date/date normalization is missing', failures)
    if "assert 'date' in n.columns" not in main_text:
        fail('normalized news date is not validated', failures)

    if 'PAYOFF_INVARIANT_FAILED' not in base_text:
        fail('payoff invariant was removed from base runner', failures)
    if 'base.outcome = safe_outcome' not in safe_text:
        fail('safe runner does not install payoff guard', failures)
    if 'if msg.startswith("PAYOFF_INVARIANT_FAILED"):' not in safe_text:
        fail('safe runner does not specifically isolate payoff invariant failures', failures)
    if '\n        raise\n' not in safe_text:
        fail('safe runner may swallow non-payoff RuntimeErrors', failures)

    raw_symbol_mismatch = "c.get('INR=X_ret5'" in base_text or "c.get('BZ=F_ret5'" in base_text
    if raw_symbol_mismatch:
        if 'usd_inr_ret5' not in safe_text or 'brent_ret5' not in safe_text:
            fail('legacy Yahoo-symbol lookups have no safe context-key repair', failures)
        else:
            warn('base runner still contains legacy Yahoo-symbol lookups; safe runner repairs them from point-in-time context', warnings)

    # Historical feature integrity: unsupported all-NaN flow features must be
    # explicitly disabled rather than silently handed to sklearn imputation.
    if 'HISTORICAL_UNSUPPORTED_FEATURES' not in safe_text:
        fail('safe runner lacks explicit historical feature-integrity policy', failures)
    else:
        for feat in HISTORICAL_UNSUPPORTED_FEATURES:
            if feat not in safe_text:
                fail(f'historical feature policy missing {feat}', failures)
    if 'base.FEATURES = [f for f in base.FEATURES if f not in HISTORICAL_UNSUPPORTED_FEATURES]' not in safe_text:
        fail('safe runner does not remove unsupported all-NaN features from model inputs', failures)
    if 'historical_feature_integrity' not in safe_text:
        fail('safe runner does not publish historical feature-integrity status', failures)

    if "cutoff=pd.Timestamp(d)-pd.Timedelta(days=1)" not in base_text:
        fail('context/news as-of cutoff guard is missing', failures)

    for needle, label in [("'lots':1", 'one-lot trade control'), ("'qty':lot", 'quantity equals lot size'), ('theoretical_max_profit', 'analytical payoff bounds')]:
        if needle not in base_text and needle not in main_text:
            fail(f'missing {label}', failures)
    if "len(s['walk_forward']) == 6" not in main_text and "len(s['walk_forward']) == 6" not in base_text:
        fail('six-window validation is missing', failures)
    if 'timeout-minutes: 360' not in main_text:
        fail('backtest lacks an explicit maximum runtime', failures)

    if '.pct_change()' in base_text and 'fill_method=None' not in base_text:
        warn('pandas pct_change FutureWarning remains; non-blocking', warnings)

    for path, names in [(BASE, {'outcome', 'build_dataset', 'fit', 'execute', 'main'}), (ROOT / 'v5/independent_inspector.py', {'main', 'fail', 'warn'})]:
        try:
            tree = ast.parse(path.read_text())
            funcs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
            for name in names:
                if name not in funcs:
                    fail(f'expected function {name} missing from {path.relative_to(ROOT)}', failures)
        except SyntaxError as exc:
            fail(f'AST parse failed for {path.relative_to(ROOT)}: {exc}', failures)

    print('=== INSPECTOR RESULT ===')
    print(f'Blocking failures: {len(failures)}')
    print(f'Warnings: {len(warnings)}')
    for f in failures:
        print(f'FAIL: {f}')
    for w in warnings:
        print(f'WARN: {w}')

    report = {
        'blocking_failures': failures,
        'warnings': warnings,
        'memory_rules': len(lessons),
        'historical_unsupported_features': sorted(HISTORICAL_UNSUPPORTED_FEATURES),
        'policy': 'Never spend hours downloading/backtesting when a cheap structural check can detect the problem first.'
    }
    (ROOT / 'v5/inspector_report.json').write_text(json.dumps(report, indent=2))
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
