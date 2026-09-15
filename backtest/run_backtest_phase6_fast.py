from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

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

# Phase 6 diagnostic layer: quantify exactly where candidates disappear before
# changing any threshold. This prevents a zero-trade result from being "fixed"
# by blindly loosening the edge gates.
_cfg = yaml.safe_load((ROOT / "backtest/config_phase6.yaml").read_text())
_target_levels = [float(x) for x in _cfg["target_levels"]]
_pred_cols = [f"pred_{str(t).replace('.', 'p').lstrip('0')}" for t in _target_levels]
_min_expected = float(_cfg["min_expected_return_pct"])
_hurdle = float(_cfg["cost_hurdle_pct"]) + float(_cfg["risk_penalty_pct"])

_diag = {
    "score_calls": 0,
    "rows_scored": 0,
    "rule_ge_070": 0,
    "pred_ge_min_expected": 0,
    "pred_gt_cost_plus_risk_hurdle": 0,
    "global_max_prediction": float("-inf"),
    "global_min_prediction": float("inf"),
    "global_max_rule_score": float("-inf"),
    "per_expert": {},
}

_orig_score_experts = baseline.score_experts


def score_experts_diagnostic(bundles, latest, target_levels=tuple(_target_levels)):
    out = _orig_score_experts(bundles, latest, target_levels=target_levels)
    if out.empty:
        return out

    _diag["score_calls"] += 1
    _diag["rows_scored"] += len(out)

    rules = out["expert_rule_score"].astype(float).to_numpy()
    preds = out[_pred_cols].to_numpy(dtype=float)
    best = np.nanmax(preds, axis=1)

    _diag["rule_ge_070"] += int((rules >= 0.70).sum())
    _diag["pred_ge_min_expected"] += int((best >= _min_expected).sum())
    _diag["pred_gt_cost_plus_risk_hurdle"] += int((best > _hurdle).sum())
    _diag["global_max_prediction"] = max(_diag["global_max_prediction"], float(np.nanmax(best)))
    _diag["global_min_prediction"] = min(_diag["global_min_prediction"], float(np.nanmin(best)))
    _diag["global_max_rule_score"] = max(_diag["global_max_rule_score"], float(np.nanmax(rules)))

    for expert, g in out.groupby("expert"):
        stats = _diag["per_expert"].setdefault(
            str(expert),
            {
                "rows": 0,
                "rule_ge_070": 0,
                "pred_ge_min_expected": 0,
                "pred_gt_hurdle": 0,
                "max_prediction": float("-inf"),
                "max_rule_score": float("-inf"),
            },
        )
        gp = g[_pred_cols].to_numpy(dtype=float)
        gb = np.nanmax(gp, axis=1)
        gr = g["expert_rule_score"].astype(float).to_numpy()
        stats["rows"] += len(g)
        stats["rule_ge_070"] += int((gr >= 0.70).sum())
        stats["pred_ge_min_expected"] += int((gb >= _min_expected).sum())
        stats["pred_gt_hurdle"] += int((gb > _hurdle).sum())
        stats["max_prediction"] = max(stats["max_prediction"], float(np.nanmax(gb)))
        stats["max_rule_score"] = max(stats["max_rule_score"], float(np.nanmax(gr)))

    if _diag["score_calls"] == 1 or _diag["score_calls"] % 25 == 0:
        print(
            "DIAG "
            f"calls={_diag['score_calls']} rows={_diag['rows_scored']} "
            f"rule>=0.70={_diag['rule_ge_070']} "
            f"pred>=min={_diag['pred_ge_min_expected']} "
            f"pred>hurdle={_diag['pred_gt_cost_plus_risk_hurdle']} "
            f"max_pred={_diag['global_max_prediction']:.5f}",
            flush=True,
        )

    return out


baseline.score_experts = score_experts_diagnostic


if __name__ == "__main__":
    baseline.main()
    print(
        json.dumps(
            {
                "diagnostics": _diag,
                "thresholds": {
                    "min_expected_return_pct": _min_expected,
                    "cost_plus_risk_hurdle": _hurdle,
                    "rule_score_floor": 0.70,
                },
            },
            indent=2,
        ),
        flush=True,
    )
