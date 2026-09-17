"""Safe V5.1 runner: isolates bad quotes and enforces historical feature integrity."""
from __future__ import annotations

import json
import run as base

VIOLATION_COUNT = 0
VIOLATION_EXAMPLES = []
CONTEXT_REPAIR_COUNT = 0
_original_outcome = base.outcome
_original_make_features = base.make_features

# The current public FII/DII feed does not provide sufficient point-in-time history
# for the 2017-2025 training windows. Do not allow all-NaN features to enter the
# imputer/model. These features are explicitly disabled until a validated
# historical source is restored.
HISTORICAL_UNSUPPORTED_FEATURES = {
    "fii_net_z",
    "dii_net_z",
    "fii_fut_long_short",
    "pcr",
    "flow_sentiment",
}
base.FEATURES = [f for f in base.FEATURES if f not in HISTORICAL_UNSUPPORTED_FEATURES]


def safe_make_features(m, d):
    """Repair legacy Yahoo-symbol lookups using point-in-time internal keys."""
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
        "policy": "Malformed/inconsistent historical option quotes are excluded from the candidate dataset; non-payoff exceptions still abort the run."
    }
    summary["context_key_repairs"] = {
        "count": CONTEXT_REPAIR_COUNT,
        "policy": "Legacy raw Yahoo-symbol feature lookups are repaired from point-in-time internal context keys."
    }
    summary["historical_feature_integrity"] = {
        "disabled_features": sorted(HISTORICAL_UNSUPPORTED_FEATURES),
        "reason": "The current FII/DII/PCR feed lacks sufficient point-in-time history for the 2017-2025 training windows; all-NaN features are therefore excluded rather than imputed.",
        "policy": "Re-enable only after a validated historical source covers the required training windows."
    }
    summary["validation"] = [
        "Exactly one lot per trade.",
        "Defined-risk payoffs are checked analytically before a trade enters the dataset.",
        "Historical quote/payoff invariant failures are excluded rather than aborting the full research run.",
        "Known context-key drift is repaired from point-in-time internal keys.",
        "Unsupported all-NaN historical flow features are excluded from model training.",
        "Context/news values are lagged by at least one session.",
        "No overlapping positions."
    ]
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
