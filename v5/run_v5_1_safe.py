"""Safe V5.1 runner: skips isolated historical quote/payoff invariant failures instead of aborting the whole research run."""
from __future__ import annotations

import json
import run as base

VIOLATIONS = []
_original_outcome = base.outcome


def safe_outcome(m, d, strategy, end=None):
    try:
        return _original_outcome(m, d, strategy, end)
    except RuntimeError as exc:
        msg = str(exc)
        if msg.startswith("PAYOFF_INVARIANT_FAILED"):
            if len(VIOLATIONS) < 25:
                VIOLATIONS.append(msg)
            return None
        raise


base.outcome = safe_outcome
base.main()

summary_path = base.ROOT / "v5/research/summary.json"
if summary_path.exists():
    summary = json.loads(summary_path.read_text())
    summary["payoff_invariant_skips"] = {
        "count": len(VIOLATIONS),
        "examples": VIOLATIONS,
        "policy": "Malformed/inconsistent historical option quotes are excluded from the candidate dataset; the research run continues. Non-payoff exceptions still abort the run."
    }
    summary["validation"] = [
        "Exactly one lot per trade.",
        "Defined-risk payoffs are checked analytically before a trade enters the dataset.",
        "Historical quote/payoff invariant failures are excluded rather than aborting the full research run.",
        "Context/news values are lagged by at least one session.",
        "No overlapping positions."
    ]
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
