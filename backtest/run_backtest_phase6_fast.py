from __future__ import annotations

import backtest.run_backtest_phase6 as baseline
from src.moe_engine_fast import train_experts

# Reuse the validated baseline execution, cost model, walk-forward boundaries,
# event handling, liquidity logic, outputs and anti-lookahead controls. Only the
# expensive expert-training implementation is swapped for the optimized one.
baseline.train_experts = train_experts

if __name__ == "__main__":
    baseline.main()
