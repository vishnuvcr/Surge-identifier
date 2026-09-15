"""Phase 6.2 block launcher.

Runs the full Phase 6.1 research engine with the Phase 6.2 configuration and
an isolated OOS calendar block. Parallelism is orchestration only: features,
labels, models, costs and execution rules remain unchanged.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG62 = ROOT / "backtest" / "config_phase62.yaml"
CFG61 = ROOT / "backtest" / "config_phase61.yaml"
PHASE61_OUT = ROOT / "backtest" / "results" / "phase61"


def main() -> None:
    start = os.environ.get("PHASE62_BLOCK_START")
    end = os.environ.get("PHASE62_BLOCK_END")
    output_dir = Path(os.environ.get("PHASE62_OUTPUT_DIR", str(ROOT / "backtest" / "results" / "phase62")))
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(CFG62.read_text(encoding="utf-8"))
    if start:
        cfg["backtest_start"] = start
    if end:
        cfg["backtest_end"] = end

    # The proven Phase 6.1 engine is reused unchanged. We temporarily provide
    # it with the Phase 6.2 config; each matrix runner is an isolated VM.
    original_cfg = CFG61.read_text(encoding="utf-8") if CFG61.exists() else None
    shutil.copy2(CFG62, CFG61)
    if original_cfg is not None:
        phase_cfg = yaml.safe_load(original_cfg)
    else:
        phase_cfg = None
    try:
        CFG61.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        # Remove stale output from any prior invocation in this VM.
        if PHASE61_OUT.exists():
            shutil.rmtree(PHASE61_OUT)
        from backtest import run_backtest_phase61 as engine
        print(f"Phase 6.2 block start={start!r} end={end!r} output={output_dir}", flush=True)
        engine.main()
        PHASE61_OUT.mkdir(parents=True, exist_ok=True)
        for item in PHASE61_OUT.iterdir():
            target = output_dir / item.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(item), str(target))
    finally:
        if phase_cfg is not None:
            CFG61.write_text(original_cfg, encoding="utf-8")
        else:
            CFG61.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
