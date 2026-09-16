from __future__ import annotations

"""Skeleton weekly iron-condor research engine.

The engine expects contract-level option OHLC/close data. It deliberately does not
invent historical option premiums when the data source only contains EOD OI/volume.
"""

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class Condor:
    expiry: pd.Timestamp
    spot: float
    put_short: float
    put_long: float
    call_short: float
    call_long: float
    entry_credit: float
    max_loss: float


def iron_condor_payoff(spot: float, c: Condor, qty: int = 1) -> float:
    put_value = max(c.put_short - spot, 0) - max(c.put_long - spot, 0)
    call_value = max(spot - c.call_short, 0) - max(spot - c.call_long, 0)
    return qty * (c.entry_credit - put_value - call_value)


def build_condor(option_chain: pd.DataFrame, spot: float, expiry: pd.Timestamp,
                 short_delta: float = 0.15, wing_width: float = 200.0) -> Condor | None:
    """Choose nearest strikes to a target absolute delta on each side.

    Expected columns: expiry, strike, option_type, delta, mid.
    """
    x = option_chain.loc[option_chain["expiry"] == expiry].copy()
    if x.empty:
        return None
    puts = x[x["option_type"].isin(["PE", "PUT"])].copy()
    calls = x[x["option_type"].isin(["CE", "CALL"])].copy()
    if puts.empty or calls.empty:
        return None
    ps = puts.iloc[(puts["delta"].abs() - short_delta).abs().argsort()[:1]]
    cs = calls.iloc[(calls["delta"].abs() - short_delta).abs().argsort()[:1]]
    ps_row, cs_row = ps.iloc[0], cs.iloc[0]
    pl = puts.loc[(puts["strike"] - (ps_row["strike"] - wing_width)).abs().idxmin()]
    cl = calls.loc[(calls["strike"] - (cs_row["strike"] + wing_width)).abs().idxmin()]
    credit = float(ps_row["mid"] + cs_row["mid"] - pl["mid"] - cl["mid"])
    width = float(min(ps_row["strike"] - pl["strike"], cl["strike"] - cs_row["strike"]))
    max_loss = width - credit
    if credit <= 0 or max_loss <= 0:
        return None
    return Condor(expiry, spot, float(ps_row["strike"]), float(pl["strike"]),
                  float(cs_row["strike"]), float(cl["strike"]), credit, max_loss)


def summarize(weekly: pd.DataFrame) -> dict:
    r = weekly["net_return"].astype(float)
    if r.empty:
        return {"weeks": 0}
    wins = r[r > 0]
    return {
        "weeks": int(r.size),
        "mean_weekly_return": float(r.mean()),
        "median_weekly_return": float(r.median()),
        "win_rate": float((r > 0).mean()),
        "target_5pct_hit_rate": float((r >= 0.05).mean()),
        "worst_week": float(r.min()),
        "best_week": float(r.max()),
        "profit_factor": float(wins.sum() / max(-r[r < 0].sum(), 1e-12)),
        "compound_return": float(np.prod(1 + r) - 1),
    }


if __name__ == "__main__":
    print("Use the repository workflow after loading contract-level option history.")
