"""Layer 4 -- Monte Carlo and the Deflated Sharpe Ratio.

Three tests, the third being the point of the layer:

  * Block-permutation test (no-skill null).  Holding the position path fixed, we
    shuffle the *order* of ~1-day blocks of market returns -- destroying any
    alignment between positions and what follows while *preserving* the returns'
    autocorrelation and volatility clustering -- recompute the net Sharpe,
    and repeat.  The p-value is the share of permuted Sharpes at least as large as
    the observed one.  Permuting blocks rather than i.i.d. shuffling keeps the null
    realistic: a full shuffle erases volatility clustering and makes the p-value
    anti-conservative.

  * Moving-block bootstrap.  Resampling the net-return series in ~1-day blocks
    (preserving short-horizon autocorrelation) gives a distribution of maximum
    drawdown and of terminal wealth, so we can report a bootstrapped 95th-pct
    drawdown rather than a single in-sample number.

  * Deflated Sharpe Ratio (Bailey & Lopez de Prado).  With N = total instances
    tried (48), the expected maximum Sharpe under the null is non-trivial; the
    DSR is the probability the strategy's true Sharpe beats that deflated
    benchmark.  This is the multiple-testing correction: 48 trials make ordinary
    Sharpes flattering by construction.

An instance passes only if the permutation p-value is below alpha AND the DSR
clears the pre-registered threshold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import stats as stats_mod
from . import common


def _permutation_pvalue(net: np.ndarray, pos: np.ndarray, cost: np.ndarray,
                        r: np.ndarray, B: int, L: int, rng, chunk: int = 250) -> float:
    """Block-permutation no-skill p-value.

    We shuffle the *order* of non-overlapping length-`L` blocks of market returns
    (preserving within-block autocorrelation and volatility clustering) while
    holding the position path fixed, and measure how often the permuted Sharpe
    matches or beats the observed one.  A full i.i.d. shuffle would destroy the
    returns' serial structure, narrowing the null and making the p-value
    anti-conservative; permuting blocks keeps the null realistic.
    """
    obs = stats_mod.sharpe_per_obs(net)
    T = len(r)
    nblk = max(1, T // L)
    T2 = nblk * L                                     # drop the final partial block
    rb = r[:T2].reshape(nblk, L)
    pos2, cost2 = pos[:T2], cost[:T2]
    ge, done = 0, 0
    while done < B:
        c = min(chunk, B - done)
        order = np.argsort(rng.random((c, nblk)), axis=1)     # random block order
        r_perm = rb[order].reshape(c, T2)                     # (c, T2)
        net_perm = pos2[None, :] * r_perm - cost2[None, :]
        mu = net_perm.mean(axis=1)
        sd = net_perm.std(axis=1, ddof=1)
        sd[sd == 0] = np.nan
        sh = mu / sd
        ge += int(np.nansum(sh >= obs))
        done += c
    return (1 + ge) / (B + 1)


def _block_bootstrap(net: np.ndarray, B: int, L: int, rng, chunk: int = 250):
    T = len(net)
    nblk = int(np.ceil(T / L))
    max_start = T - L
    dd_samples = np.empty(B)
    term_samples = np.empty(B)
    arangeL = np.arange(L)
    done = 0
    while done < B:
        c = min(chunk, B - done)
        starts = rng.integers(0, max_start + 1, size=(c, nblk))
        idx = (starts[:, :, None] + arangeL[None, None, :]).reshape(c, nblk * L)[:, :T]
        samp = net[idx]                               # (c, T)
        eq = np.cumprod(1.0 + samp, axis=1)
        peak = np.maximum.accumulate(eq, axis=1)
        dd = (eq / peak - 1.0).min(axis=1)
        dd_samples[done:done + c] = -dd
        term_samples[done:done + c] = eq[:, -1] - 1.0
        done += c
    return {
        "dd_p50": float(np.percentile(dd_samples, 50)),
        "dd_p95": float(np.percentile(dd_samples, 95)),
        "prob_positive": float(np.mean(term_samples > 0)),
    }


def run(pop, data: dict, cfg: dict, cost_model: str = "realized_spread") -> pd.DataFrame:
    c = cfg["layer4"]
    rng = np.random.default_rng(c["random_seed"])
    n_trials = len(pop)

    # --- pass A: net returns, moments, permutation, bootstrap ---------------
    cache = {}
    sr_list = []
    for inst in pop:
        df = data[inst.symbol]
        ppy = df.attrs.get("bars_per_year", 252 * 24)
        sig = __import__("alphafactory.families", fromlist=["get"]).get(inst.family).signal(df, inst.params)
        r = df["ret"].to_numpy(float)
        spread = df["spread_frac"].to_numpy(float)
        pos = np.empty_like(sig, float)
        pos[0] = 0.0
        pos[1:] = sig[:-1]
        prev = np.empty_like(pos)
        prev[0] = 0.0
        prev[1:] = pos[:-1]
        turn = np.abs(pos - prev)
        from .. import costs as costs_mod
        cost = costs_mod.trade_costs(turn, spread, cost_model,
                                     df.attrs.get("asset_class", "index"))
        net = pos * r - cost
        m = stats_mod.summarize(net, ppy)
        perm_p = _permutation_pvalue(net, pos, cost, r, c["n_permutations"], c["block_length"], rng)
        boot = _block_bootstrap(net, c["n_bootstrap"], c["block_length"], rng)
        cache[inst.id] = {"net": net, "ppy": ppy, "m": m, "perm_p": perm_p, "boot": boot}
        sr_list.append(m["sharpe_per_obs"])

    var_sr = float(np.var(np.array(sr_list), ddof=1))

    # --- pass B: Deflated Sharpe Ratio (needs Var_SR across all trials) -----
    rows = []
    for inst in pop:
        e = cache[inst.id]
        m = e["m"]
        dsr, sr_star = stats_mod.deflated_sharpe_ratio(
            m["sharpe_per_obs"], m["n"], m["skew"], m["kurtosis"], var_sr, n_trials)
        ok_perm = e["perm_p"] < c["permutation_alpha"]
        ok_dsr = dsr >= c["min_dsr"]
        passed = bool(ok_perm and ok_dsr)
        if passed:
            reason = ""
        elif not ok_dsr and not ok_perm:
            reason = "no skill (permutation) and fails DSR"
        elif not ok_dsr:
            reason = "fails Deflated Sharpe (multiple testing)"
        else:
            reason = "permutation p-value not significant"
        rows.append({
            "id": inst.id, "layer": "L4", "passed": passed, "reason": reason,
            "sharpe": m["sharpe"], "perm_pvalue": e["perm_p"], "dsr": dsr,
            "dsr_benchmark_sr": sr_star, "boot_dd_p95": e["boot"]["dd_p95"],
            "boot_prob_positive": e["boot"]["prob_positive"],
        })
    out = pd.DataFrame(rows)
    out.attrs["var_sr_across_trials"] = var_sr
    out.attrs["n_trials"] = n_trials
    return out
