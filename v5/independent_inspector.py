from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / '.github/workflows/v5.1-regime-adaptive.yml'
SAFE = ROOT / 'v5/run_v5_1_safe.py'
BASE = ROOT / 'v5/run.py'
MEMORY = ROOT / 'v5/inspector_memory.json'

REQUIRED = [
    BASE,
    SAFE,
    ROOT / 'v5/config.yaml',
    ROOT / 'v5/context_config.yaml',
    ROOT / 'scripts/download_nifty_nse_history.py',
    ROOT / 'scripts/download_v5_market_context.py',
    ROOT / 'scripts/download_v5_flow_context.py',
    MAIN,
]

PY_FILES = [
    BASE,
    SAFE,
    ROOT / 'scripts/download_nifty_nse_history.py',
    ROOT / 'scripts/download_v5_market_context.py',
    ROOT / 'scripts/download_v5_flow_context.py',
]


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

    memory = json.loads(MEMORY.read_text())
    if len(memory.get('lessons', [])) < 5:
        fail('inspector memory contains too few learned failure rules', failures)
    else:
        print(f"Loaded {len(memory['lessons'])} learned failure rules.")

    for p in PY_FILES:
        r = subprocess.run(['python', '-m', 'py_compile', str(p)], cwd=ROOT, text=True, capture_output=True)
        if r.returncode:
            fail(f'py_compile failed for {p.relative_to(ROOT)}: {r.stderr.strip()}', failures)
        else:
            print(f'OK compile: {p.relative_to(ROOT)}')

    main_text = MAIN.read_text()
    safe_text = SAFE.read_text()
    base_text = BASE.read_text()

    # Architecture: cheap inspector must gate the expensive job.
    if 'uses: ./.github/workflows/v5.1-independent-inspector.yml' not in main_text:
        fail('main V5.1 workflow is not gated by the independent inspector', failures)
    if not re.search(r'backtest:\s*\n\s*needs:\s*inspector', main_text):
        fail('backtest job does not depend on inspector job', failures)

    # Never allow the expensive production step to bypass the hardened runner.
    direct_raw_run = re.search(r'^\s*run:\s*python v5/run\.py\s*$', main_text, flags=re.MULTILINE)
    if direct_raw_run:
        fail('workflow directly executes python v5/run.py instead of the guarded runner', failures)
    if 'run: python v5/run_v5_1_safe.py' not in main_text:
        fail('guarded V5.1 runner is not the production backtest command', failures)

    # Known external schema failure: Date vs date.
    if "n.columns=[str(c).strip().lower() for c in n.columns]" not in main_text:
        fail('news schema normalization is missing before model ingestion', failures)
    if "assert 'date' in n.columns" not in main_text:
        fail('news validation is not asserting normalized date', failures)

    # Known payoff failure: one bad historical quote must not kill the full run.
    if 'PAYOFF_INVARIANT_FAILED' not in base_text:
        fail('base research runner lost the payoff invariant', failures)
    if 'base.outcome = safe_outcome' not in safe_text:
        fail('safe runner does not intercept outcome() failures', failures)
    if 'if msg.startswith("PAYOFF_INVARIANT_FAILED"):' not in safe_text:
        fail('safe runner does not specifically isolate payoff invariant failures', failures)
    if '\n        raise\n' not in safe_text:
        fail('safe runner may be swallowing non-payoff RuntimeErrors', failures)

    # Known context-key drift: external Yahoo symbols != internal feature keys.
    raw_symbol_mismatch = ("c.get('INR=X_ret5'" in base_text) or ("c.get('BZ=F_ret5'" in base_text))
    if raw_symbol_mismatch:
        if "f['usd_inr_ret5']" not in safe_text or "f['brent_ret5']" not in safe_text:
            fail('legacy raw Yahoo-symbol feature lookups remain without a safe-runner repair', failures)
        else:
            warn('base runner still contains legacy Yahoo-symbol lookups; safe runner must repair them point-in-time', warnings)

    # Anti-leakage architecture checks.
    if "cutoff=pd.Timestamp(d)-pd.Timedelta(days=1)" not in base_text:
        fail('context/news as-of cutoff guard is missing', failures)
    if 'future-filled' not in base_text:
        warn('future-fill policy is not explicitly documented in base runner text', warnings)

    # Structural guardrails from prior runs.
    for needle, label in [
        ("'lots':1", 'one-lot trade control'),
        ("'qty':lot", 'quantity equals current lot size'),
        ("assert len(s['walk_forward']) == 6", 'six-window validation'),
        ("theoretical_max_profit", 'analytical payoff bounds'),
    ]:
        if needle not in main_text and needle not in base_text:
            fail(f'missing {label}', failures)

    # Warn on known non-fatal noise that should eventually be cleaned up.
    if '.pct_change()' in base_text and 'fill_method=None' not in base_text:
        warn('pandas pct_change FutureWarning remains; this is not a research-blocking error but should be cleaned up', warnings)

    # AST sanity: expected top-level functions still exist.
    try:
        tree = ast.parse(base_text)
        funcs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        for name in {'outcome', 'build_dataset', 'fit', 'execute', 'main'}:
            if name not in funcs:
                fail(f'expected function {name} missing from v5/run.py', failures)
    except SyntaxError as exc:
        fail(f'AST parse failed: {exc}', failures)

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
        'memory_rules': len(memory.get('lessons', [])),
        'policy': 'Do not spend hours downloading/backtesting when a cheap structural check can detect the problem first.',
    }
    out = ROOT / 'v5/inspector_report.json'
    out.write_text(json.dumps(report, indent=2))
    print(f'Report: {out.relative_to(ROOT)}')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
