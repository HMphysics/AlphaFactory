"""Layer 1 -- In-sample / out-of-sample with parameter-sensitivity.

Each instance is evaluated on the out-of-sample (last 30%) slice.  It passes only
if (i) its OoS performance clears the pre-registered bar AND (ii) it is robust to
a +/-20% perturbation of each free parameter: the median neighbour OoS Sharpe
clears a floor and no neighbour has a losing OoS Sharpe -- a genuine optimum is
not perched on a cliff edge.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import families as fam_mod
from .. import stats as stats_mod
from . import common


def run(pop, data: dict, cfg: dict, cost_model: str = "realized_spread") -> pd.DataFrame:
    c = cfg["layer1"]
    rows = []
    for inst in pop:
        df = data[inst.symbol]
        ppy = df.attrs.get("bars_per_year", 252 * 24)
        n = len(df)
        is_mask, oos_mask = common.chronological_split(n, c["is_fraction"])

        net = common.full_net_returns(df, inst.family, inst.params, cost_model)
        oos = net[oos_mask]
        oos_sharpe = stats_mod.sharpe_annual(oos, ppy)
        oos_ret = float(np.prod(1.0 + oos) - 1.0)

        fam = fam_mod.get(inst.family)
        neigh = fam.perturb(inst.params, c["perturb_frac"])
        nsh = []
        for q in neigh:
            nnet = common.full_net_returns(df, inst.family, q, cost_model)
            nsh.append(stats_mod.sharpe_annual(nnet[oos_mask], ppy))
        nsh = np.array(nsh) if nsh else np.array([oos_sharpe])
        nmed = float(np.median(nsh))
        nmin = float(np.min(nsh))

        ok_oos = oos_sharpe >= c["min_oos_sharpe"]
        ok_ret = (oos_ret > 0) if c["require_positive_oos_return"] else True
        ok_med = nmed >= c["neighbour_median_min_sharpe"]
        ok_floor = nmin >= c["neighbour_floor_sharpe"]
        passed = bool(ok_oos and ok_ret and ok_med and ok_floor)

        if passed:
            reason = ""
        elif not ok_oos:
            reason = "OoS Sharpe below threshold"
        elif not ok_ret:
            reason = "negative OoS return"
        elif not ok_floor:
            reason = "fragile: a +/-20% neighbour loses OoS"
        else:
            reason = "weak neighbourhood (median Sharpe)"

        rows.append({
            "id": inst.id, "layer": "L1", "passed": passed, "reason": reason,
            "oos_sharpe": oos_sharpe, "oos_return": oos_ret,
            "neigh_median_sharpe": nmed, "neigh_min_sharpe": nmin,
            "n_neighbours": len(neigh),
        })
    return pd.DataFrame(rows)
