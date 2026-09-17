from __future__ import annotations

"""V5.2 research harness.

Runs the existing V5.1 engine under several explicitly defined regime/strategy
eligibility configurations without changing the V5.1 baseline branch.

This file is intentionally a research harness: it rewrites a temporary config,
runs the safe V5.1 engine for each configuration, and stores a compact comparison
report. No live-trading behavior is introduced.
"""

import json
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE_CFG = ROOT / "v5/config.yaml"
SAFE_RUNNER = ROOT / "v5/run_v5_1_safe.py"
OUT = ROOT / "v5/research/v5_2_config_comparison.json"

# Strategy eligibility families derived from the completed V5.1 learning report.
# Positive/negative here are descriptive historical median classifications from
# the V5.1 training windows, not guarantees for future performance.
REGIME_GATES = {
    "BULL_LOW_VOL": {
        "allowed": ["BULL_PUT_SPREAD", "BULL_CALL_SPREAD", "IRON_CONDOR"],
    },
    "BULL_HIGH_VOL": {
        "allowed": ["BULL_PUT_SPREAD", "BULL_CALL_SPREAD", "LONG_CALL"],
    },
    "BEAR_LOW_VOL": {
        "allowed": ["BEAR_CALL_SPREAD", "BEAR_PUT_SPREAD", "IRON_CONDOR"],
    },
    "BEAR_HIGH_VOL": {
        "allowed": ["IRON_CONDOR", "BEAR_CALL_SPREAD", "BEAR_PUT_SPREAD"],
    },
    "RANGE_LOW_VOL": {
        "allowed": ["BEAR_CALL_SPREAD", "IRON_CONDOR", "IRON_BUTTERFLY"],
    },
    "RANGE_HIGH_VOL": {
        "allowed": ["IRON_CONDOR", "IRON_BUTTERFLY", "LONG_STRADDLE"],
    },
}


def load_yaml():
    return yaml.safe_load(BASE_CFG.read_text())


def write_cfg(cfg):
    BASE_CFG.write_text(yaml.safe_dump(cfg, sort_keys=False))


def run_variant(name: str, cfg: dict) -> dict:
    write_cfg(cfg)
    p = subprocess.run(
        ["python", str(SAFE_RUNNER)],
        cwd=ROOT / "v5",
        text=True,
        capture_output=True,
    )
    result = {
        "variant": name,
        "returncode": p.returncode,
        "stdout_tail": p.stdout[-4000:],
        "stderr_tail": p.stderr[-4000:],
    }
    summary_path = ROOT / "v5/research/summary.json"
    if p.returncode == 0 and summary_path.exists():
        s = json.loads(summary_path.read_text())
        result["overall_oos"] = s.get("overall_oos", {})
        result["walk_forward"] = s.get("walk_forward", [])
    return result


def main():
    original = load_yaml()
    variants = []

    try:
        # Baseline-like stricter selection: increase required edge separation.
        cfg = dict(original)
        cfg["min_predicted_edge_vs_second"] = 0.002
        cfg["no_trade_threshold"] = 0.005
        cfg["regime_blend_weight"] = 0.50
        variants.append(run_variant("strict_edge_regime_blend", cfg))

        # Slightly longer holding period to test whether five sessions is the
        # main source of negative expectancy.
        cfg = dict(original)
        cfg["holding_days"] = 7
        cfg["min_predicted_edge_vs_second"] = 0.001
        cfg["no_trade_threshold"] = 0.003
        cfg["regime_blend_weight"] = 0.50
        variants.append(run_variant("7d_hold_regime_blend", cfg))

        # Reduced trade frequency with a stronger model edge gate.
        cfg = dict(original)
        cfg["min_predicted_edge_vs_second"] = 0.003
        cfg["no_trade_threshold"] = 0.008
        cfg["regime_blend_weight"] = 0.60
        variants.append(run_variant("high_selectivity_regime", cfg))

        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"variants": variants}, indent=2, default=str))
    finally:
        write_cfg(original)


if __name__ == "__main__":
    main()
