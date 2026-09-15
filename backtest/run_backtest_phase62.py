"""Phase 6.2 launcher.

This is intentionally a thin orchestration layer. The heavy work is partitioned by
walk-forward calendar blocks so independent OOS blocks can execute concurrently.
Every block independently rebuilds its prior-information training set, features,
liquidity universe, masters, OOF router and execution ledger; a final aggregation
step combines the disjoint OOS ledgers in chronological order and recomputes the
portfolio/equity statistics. The methodology is therefore unchanged by parallelism.

The implementation imports the Phase 6.1 engine rather than duplicating model logic.
The month/block argument is supplied through PHASE62_BLOCK_START/END. A future
optimization can cache the common feature table as an artifact, but this launcher
keeps that optimization separate from model semantics.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

from backtest.run_backtest_phase61 import run_backtest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main() -> None:
    config_path = ROOT / "backtest" / "config_phase62.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    start = os.environ.get("PHASE62_BLOCK_START")
    end = os.environ.get("PHASE62_BLOCK_END")
    output_dir = os.environ.get("PHASE62_OUTPUT_DIR", str(ROOT / "backtest" / "results" / "phase62"))

    print(f"Phase 6.2 block start={start!r} end={end!r} output={output_dir}", flush=True)

    # The Phase 6.1 engine accepts optional environment/date controls. We deliberately
    # do not alter features, labels, model hyperparameters, costs or execution rules.
    if start:
        os.environ["BACKTEST_START"] = start
    if end:
        os.environ["BACKTEST_END"] = end
    os.environ["PHASE62_OUTPUT_DIR"] = output_dir

    run_backtest(cfg)


if __name__ == "__main__":
    main()
