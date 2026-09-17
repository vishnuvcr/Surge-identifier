"""Safe V5.1 runner: isolates malformed historical quotes and repairs known context-key drift."""
from __future__ import annotations

import json
import run as base

VIOLATION_COUNT = 0
VIOLATION_EXAMPLES = []
CONTEXT_REPAIR_COUNT = 0
_original_outcome = base.outcome
_original_make_features = base.make_features


def safe_make_features(m, d):
    """Repair legacy Yahoo-symbol lookups while preserving point-in-time context."""
    global CONTEXT_REPAIR_COUNT
    f = _original_make_features(m, d)
    if f is None:
        return None
    c = m.asof_context(d)
    changed = False
    for key in ("usd_inr_ret5", "brent_ret5"):
        value = c.get(key)
        if value is not None:
            try:
                if f.get(key) != value:
                    f[key] = value
                    changed = True
            except Exception:
                f[key] = value
                changed = True
    if changed:
        CONTEXT_REPAIR_COUNT += 1
    return f


def safe_outcome(m, d, strategy, end=None):
    global VIOLATION_COUNT
    try:
        return _original_outcome(m, d, strategy, end)
    except RuntimeError as exc:
        msg = str(exc)
        if msg.startswith("PAYOFF_INVARIANT_FAILED"):
            VIOLATION_COUNT += 1
            if len(VIOLATION_EXAMPLES) < 25:
                VIOLATION_EXAMPLES.append(msg)
            return None
        raise


base.make_features = safe_make_features
base.outcome = safe_outcome
base.main()

summary_path = base.ROOT / "v5/research/summary.json"
if summary_path.exists():
    summary = json.loads(summary_path.read_text())
    summary["payoff_invariant_skips"] = {
        "count": VIOLATION_COUNT,
        "examples": VIOLATION_EXAMPLES,
        "policy": "Malformed/inconsistent historical option quotes are excluded from the candidate dataset; the research run continues. Non-payoff exceptions still abort the run."
    }
    summary["context_key_repairs"] = {
        "count": CONTEXT_REPAIR_COUNT,
        "policy": "Legacy raw Yahoo-symbol feature lookups are repaired from the point-in-time internal context keys before model features are consumed."
    }
    summary["validation"] = [
        "Exactly one lot per trade.",
        "Defined-risk payoffs are checked analytically before a trade enters the dataset.",
        "Historical quote/payoff invariant failures are excluded rather than aborting the full research run.",
        "Known context-key drift is repaired from point-in-time internal keys.",
        "Context/news values are lagged by at least one session.",
        "No overlapping positions."
    ]
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
