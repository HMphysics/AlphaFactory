"""Combine the four independent layer verdicts into a survival funnel.

The assignment is explicit that every layer is run on the *full* population
independently, and that the funnel is the intersection of the per-layer
verdicts.  We therefore keep two distinct views:

* ``per_layer`` -- for each layer, how many of all N candidates pass it when
  judged on its own (this is the honest, order-independent statistic).
* ``waterfall`` -- a presentational L1->L2->L3->L4 cascade: of the instances
  still alive after every *previous* layer, how many also clear the current
  layer, and what is the dominant reason the rest die.  The cascade is purely
  for communication; survivors are defined by the intersection, not the order.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAYER_ORDER = ["L1", "L2", "L3", "L4"]
LAYER_NAME = {
    "L1": "IS/OoS + parameter robustness",
    "L2": "Walk-forward re-optimization",
    "L3": "Stress replay (real + synthetic)",
    "L4": "Monte Carlo + Deflated Sharpe",
}


def _dominant_reason(reasons: pd.Series) -> str:
    r = reasons[reasons.astype(bool)]
    if r.empty:
        return ""
    return r.value_counts().idxmax()


def combine(layer_results: dict[str, pd.DataFrame]) -> dict:
    """Merge per-layer DataFrames keyed by 'L1'..'L4'.

    Each input frame must have columns ['id', 'passed', 'reason'].  Returns a
    dict with the merged per-instance verdict table, the per-layer summary, the
    presentational waterfall, and the final survivor ids.
    """
    ids = list(layer_results["L1"]["id"])
    verdict = pd.DataFrame({"id": ids})

    for layer in LAYER_ORDER:
        df = layer_results[layer][["id", "passed", "reason"]].rename(
            columns={"passed": f"{layer}_pass", "reason": f"{layer}_reason"})
        verdict = verdict.merge(df, on="id", how="left")

    pass_cols = [f"{layer}_pass" for layer in LAYER_ORDER]
    verdict["n_layers_passed"] = verdict[pass_cols].sum(axis=1).astype(int)
    verdict["survivor"] = verdict[pass_cols].all(axis=1)

    # ---- per-layer independent view --------------------------------------
    n_total = len(verdict)
    per_layer_rows = []
    for layer in LAYER_ORDER:
        passed = int(verdict[f"{layer}_pass"].sum())
        per_layer_rows.append({
            "layer": layer,
            "name": LAYER_NAME[layer],
            "evaluated": n_total,
            "passed": passed,
            "failed": n_total - passed,
            "pass_rate": passed / n_total if n_total else 0.0,
            "dominant_failure": _dominant_reason(verdict[f"{layer}_reason"]),
        })
    per_layer = pd.DataFrame(per_layer_rows)

    # ---- presentational waterfall (intersection cascade) -----------------
    alive = pd.Series(True, index=verdict.index)
    waterfall_rows = []
    for layer in LAYER_ORDER:
        entering = int(alive.sum())
        this_pass = verdict[f"{layer}_pass"].fillna(False).to_numpy(bool)
        dropped_mask = alive.to_numpy() & (~this_pass)
        dominant = _dominant_reason(verdict.loc[dropped_mask, f"{layer}_reason"])
        alive = alive & pd.Series(this_pass, index=verdict.index)
        surviving = int(alive.sum())
        waterfall_rows.append({
            "layer": layer,
            "name": LAYER_NAME[layer],
            "in": entering,
            "out": surviving,
            "killed": entering - surviving,
            "dominant_failure": dominant,
        })
    waterfall = pd.DataFrame(waterfall_rows)

    survivors = list(verdict.loc[verdict["survivor"], "id"])
    return {
        "verdict": verdict,
        "per_layer": per_layer,
        "waterfall": waterfall,
        "survivors": survivors,
        "n_total": n_total,
    }


def format_waterfall(waterfall: pd.DataFrame, n_total: int) -> str:
    lines = []
    lines.append(f"Population entering the gate: {n_total} candidates")
    lines.append("")
    header = f"{'Layer':<5}{'Description':<36}{'In':>5}{'Out':>5}{'Killed':>8}  Dominant failure reason"
    lines.append(header)
    lines.append("-" * len(header))
    for _, r in waterfall.iterrows():
        lines.append(
            f"{r['layer']:<5}{r['name']:<36}{r['in']:>5}{r['out']:>5}"
            f"{r['killed']:>8}  {r['dominant_failure'] or '-'}")
    return "\n".join(lines)


def format_per_layer(per_layer: pd.DataFrame) -> str:
    lines = []
    header = (f"{'Layer':<5}{'Description':<36}{'Pass':>6}{'Fail':>6}"
              f"{'Rate':>8}  Dominant failure reason")
    lines.append(header)
    lines.append("-" * len(header))
    for _, r in per_layer.iterrows():
        lines.append(
            f"{r['layer']:<5}{r['name']:<36}{r['passed']:>6}{r['failed']:>6}"
            f"{r['pass_rate']*100:>7.1f}%  {r['dominant_failure'] or '-'}")
    return "\n".join(lines)
