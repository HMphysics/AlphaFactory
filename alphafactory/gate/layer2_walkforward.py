"""Layer 2 -- Walk-forward stability (per-instance, fixed parameters).

For each instance we roll a TRAIN/TEST scheme defined in calendar weeks and, using
the instance's *fixed* parameters (no re-optimisation), evaluate the strategy on
each out-of-window TEST segment.  The instance is judged on its OWN distribution of
out-of-window results:

  * level       -- the aggregated out-of-window Sharpe clears `wfo_min_sharpe`
                   (and the aggregated return is positive), AND
  * consistency -- the strategy is profitable in at least
                   `min_positive_window_fraction` of the windows, so an edge driven
                   by a single lucky window fails even if the average looks good.

This verdict is INDEPENDENT per instance: every instance is judged on its own
rolling out-of-sample record, all instances can pass in principle, and none is
rejected for losing an in-sample contest to a sibling.  It is the temporal-
stability complement to Layer 1's single holdout + parameter-robustness check.

A rolling RE-OPTIMISATION is still computed -- one walk-forward curve per
(family, symbol) plus, per instance, the fraction of windows in which its
parameters would have been the optimiser's pick -- but it is reported as a
*diagnostic* only and does NOT drive the pass/fail.  An earlier design used the
re-optimisation selection as the gate; that verdict was relative across the grid
(its pass count was capped by grid arithmetic and a strong stand-alone instance
could fail for being the runner-up), so it was demoted -- see DECISIONS.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import families as fam_mod
from .. import stats as stats_mod
from . import common


def _param_key(p: dict) -> tuple:
    return tuple(sorted((k, float(v)) for k, v in p.items()))


def _windows(index: pd.DatetimeIndex, train_w: int, test_w: int, step_w: int):
    t0 = index[0]
    tic = pd.Timedelta(weeks=1)
    cur = t0
    out = []
    while True:
        tr_start = cur
        tr_end = tr_start + train_w * tic
        te_start = tr_end
        te_end = te_start + test_w * tic
        if te_start >= index[-1]:
            break
        tr_mask = np.asarray((index >= tr_start) & (index < tr_end))
        te_mask = np.asarray((index >= te_start) & (index < te_end))
        if tr_mask.sum() > 200 and te_mask.sum() > 50:
            out.append((tr_mask, te_mask))
        cur = cur + step_w * tic
    return out


def _diagnostic_reopt(df: pd.DataFrame, family: str, grid: list[dict],
                      cfg2: dict, cost_model: str):
    """Rolling re-optimisation -- kept as a DIAGNOSTIC, not a gate.

    Returns the family/symbol re-optimised out-of-window curve, the per-parameter
    selection fraction, and (for reuse) the precomputed full-series net returns
    per grid pair and the window masks.
    """
    ppy = df.attrs.get("bars_per_year", 252 * 24)
    grid_net = {_param_key(p): common.full_net_returns(df, family, p, cost_model)
                for p in grid}
    keys = list(grid_net.keys())
    windows = _windows(df.index, cfg2["train_weeks"], cfg2["test_weeks"],
                       cfg2["step_weeks"])

    test_segments, selections = [], []
    for tr_mask, te_mask in windows:
        best_key, best_sh = None, -np.inf
        for k in keys:
            sh = stats_mod.sharpe_annual(grid_net[k][tr_mask], ppy)
            if sh > best_sh:
                best_sh, best_key = sh, k
        selections.append(best_key)
        test_segments.append(grid_net[best_key][te_mask])

    if test_segments:
        agg = np.concatenate(test_segments)
        nwin = len(selections)
        reopt_sharpe = stats_mod.sharpe_annual(agg, ppy)
        sel_frac = {k: selections.count(k) / nwin for k in keys}
    else:
        nwin, reopt_sharpe = 0, 0.0
        sel_frac = {k: 0.0 for k in keys}

    return {"reopt_sharpe": reopt_sharpe, "select_fraction": sel_frac,
            "grid_net": grid_net, "windows": windows, "ppy": ppy,
            "n_windows": nwin}


def _instance_oos(net: np.ndarray, windows, ppy: float):
    """Per-instance fixed-parameter out-of-window record.

    Returns (aggregated out-of-window Sharpe, aggregated out-of-window return,
    fraction of windows with a positive return, number of windows).
    """
    if not windows:
        return 0.0, 0.0, 0.0, 0
    seg = [net[te_mask] for _, te_mask in windows]
    win_ret = np.array([float(np.prod(1.0 + s) - 1.0) for s in seg])
    agg = np.concatenate(seg)
    oos_sharpe = stats_mod.sharpe_annual(agg, ppy)
    oos_return = float(np.prod(1.0 + agg) - 1.0)
    pos_frac = float(np.mean(win_ret > 0.0))
    return oos_sharpe, oos_return, pos_frac, len(seg)


def run(pop, data: dict, cfg: dict, cost_model: str = "realized_spread") -> pd.DataFrame:
    c = cfg["layer2"]
    # one re-optimisation diagnostic per (family, symbol); reused for the per-instance eval
    fs_cache: dict[tuple, dict] = {}
    for fname, fam in fam_mod.all_families().items():
        for sym in {i.symbol for i in pop if i.family == fname}:
            fs_cache[(fname, sym)] = _diagnostic_reopt(data[sym], fname, fam.grid, c,
                                                       cost_model)

    rows = []
    for inst in pop:
        d = fs_cache[(inst.family, inst.symbol)]
        key = _param_key(inst.params)
        net = d["grid_net"][key]
        oos_sharpe, oos_return, pos_frac, nwin = _instance_oos(net, d["windows"], d["ppy"])

        ok_level = (oos_sharpe >= c["wfo_min_sharpe"]) and (
            (oos_return > 0) if c["require_positive_wfo_return"] else True)
        ok_consistency = pos_frac >= c["min_positive_window_fraction"]
        passed = bool(ok_level and ok_consistency)
        if passed:
            reason = ""
        elif not ok_level:
            reason = "out-of-window Sharpe/return below bar"
        else:
            reason = "edge not consistent across windows"

        rows.append({
            "id": inst.id, "layer": "L2", "passed": passed, "reason": reason,
            "oos_sharpe": oos_sharpe, "oos_return": oos_return,
            "positive_window_fraction": pos_frac, "n_windows": nwin,
            # --- diagnostics (not used for pass/fail) ---
            "reopt_sharpe": d["reopt_sharpe"],
            "select_fraction": d["select_fraction"].get(key, 0.0),
        })
    return pd.DataFrame(rows)
