from __future__ import annotations

import numpy as np

from src.moe_engine_phase63 import route_and_rank as _route_and_rank


def route_and_rank(scored, router, cfg):
    """Use the trained OOF router when available; otherwise use a transparent p25 tail-probability fallback."""
    if router is not None:
        return _route_and_rank(scored, router, cfg)
    if scored is None or scored.empty:
        return scored
    out = scored.copy()
    out["p25"] = out["p_hit_025"]
    out["p30"] = out["p_hit_03"]
    out["p40"] = out["p_hit_04"]
    out["p50"] = out["p_hit_05"]
    out["target"] = np.select([out.p50 >= .08, out.p40 >= .12, out.p30 >= .18], [.05, .04, .03], default=.025)
    hurdle = float(cfg.get("cost_hurdle_pct", .0015)) + float(cfg.get("risk_penalty_pct", .001))
    out["risk_adjusted_edge"] = out.target * out.p25 - hurdle - .25 * out.p_stop
    out["router_expert"] = out.expert
    out["router_confidence"] = out.p25
    out["selection_score"] = out.risk_adjusted_edge + .10 * out.p30 + .05 * np.clip(out.pred_mfe, 0, .10)
    out = out[(out.p25 >= float(cfg.get("min_hit_probability", .08))) & (out.risk_adjusted_edge >= float(cfg.get("min_risk_adjusted_edge", .001)))]
    return out.sort_values(["selection_score", "p25", "pred_mfe"], ascending=False).drop_duplicates("symbol")
