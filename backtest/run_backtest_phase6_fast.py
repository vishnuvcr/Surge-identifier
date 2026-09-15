from __future__ import annotations

import sys
from pathlib import Path

# When a file inside backtest/ is executed directly, Python puts backtest/
# on sys.path rather than the repository root. Add the root explicitly so the
# package imports below are reliable in GitHub Actions and local execution.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backtest.run_backtest_phase6 as baseline
from src.moe_engine_fast import train_experts

# Reuse the validated baseline execution, cost model, walk-forward boundaries,
# event handling, liquidity logic, outputs and anti-lookahead controls. Only the
# expensive expert-training implementation is swapped for the optimized one.
baseline.train_experts = train_experts

if __name__ == "__main__":
    baseline.main()
