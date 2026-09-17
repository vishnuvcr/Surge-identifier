from __future__ import annotations
import ast, re
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
ENGINE=ROOT/'v6/expiry_regime_engine.py'; CFG=ROOT/'v6/config.yaml'; WF=ROOT/'.github/workflows/v6-expiry-regime-research.yml'

def fail(msg, out): out.append(msg); print('BLOCK:',msg)
def main():
    f=[]; cfg=yaml.safe_load(CFG.read_text()); text=ENGINE.read_text(); wf=WF.read_text()
    for p in [ENGINE,CFG,WF]:
        if not p.exists(): fail(f'missing {p.relative_to(ROOT)}',f)
    if f:return 1
    try: ast.parse(text)
    except SyntaxError as e: fail(f'engine syntax error: {e}',f)
    if cfg['new_oos_start']!='2026-04-01': fail('new OOS start must remain 2026-04-01',f)
    if cfg['top_k']<1: fail('top_k must be positive',f)
    if 'target_return' not in text or 'pred_return' not in text: fail('expiry-return prediction path missing',f)
    if "new_oos_start" not in text: fail('new OOS gate missing from engine',f)
    if 'pred_regime' not in text: fail('regime classifier missing',f)
    if "groupby(['actual_regime','strategy'])" not in text: fail('regime-strategy learning map missing',f)
    if 'run_wfo' not in text or 'run_7030' not in text or 'run_new_oos' not in text: fail('required evaluation modes missing',f)
    if re.search(r'entry_date>=pd\.Timestamp\(CFG\[.new_oos_start.\]\)',text) is None: fail('new OOS selection boundary is not explicit',f)
    if "python v6/expiry_regime_engine.py" not in wf: fail('workflow does not execute V6 engine',f)
    print('=== V6 INSPECTOR ===')
    print('Blocking failures:',len(f))
    for x in f: print('FAIL:',x)
    return 1 if f else 0

if __name__=='__main__': raise SystemExit(main())
