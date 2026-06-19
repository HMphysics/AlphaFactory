"""Transaction-cost models.

Default model (the assignment's, usable without justification):
    charge the realized spread from the <SPREAD> column at the fill bar -- half
    the spread on each side of a round trip -- plus a commission of 0.5 bps per
    side.  A position change of magnitude |dw| at bar t is charged

        cost_t = |dw| * (0.5 * spread_frac_t + commission_per_side)

    so a full round trip (open then close) pays one full spread plus two
    commissions, exactly as specified.

Flat fallback (also accepted by the assignment): a fixed round-trip cost of
1 bps for index/FX/gold and 8 bps for crypto, i.e. 0.5 bps / 4 bps per side.
"""
from __future__ import annotations

import numpy as np

COMMISSION_PER_SIDE = 0.5e-4          # 0.5 bps
FLAT_ROUND_TRIP = {"index": 1e-4, "fx": 1e-4, "gold": 1e-4, "crypto": 8e-4}


def trade_costs(turnover: np.ndarray, spread_frac: np.ndarray, model: str,
                asset_class: str, commission_per_side: float = COMMISSION_PER_SIDE
                ) -> np.ndarray:
    """Per-bar cost given per-bar turnover |dw| and realized spread fraction."""
    turnover = np.asarray(turnover, float)
    if model == "realized_spread":
        spread_frac = np.asarray(spread_frac, float)
        per_side = 0.5 * spread_frac + commission_per_side
        return turnover * per_side
    if model == "flat":
        rt = FLAT_ROUND_TRIP.get(asset_class, 1e-4)
        return turnover * (rt / 2.0)
    raise ValueError(f"unknown cost model {model!r}")
