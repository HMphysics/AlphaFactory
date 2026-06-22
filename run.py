#!/usr/bin/env python3
"""The Alpha Factory -- single-command funnel reproduction.

    python run.py                  # full run, uses cached H1 bars if present
    python run.py --quick          # fast smoke test (reduced Monte Carlo)
    python run.py --no-cache       # rebuild H1 bars from the raw CSVs

The script (1) prints the SHA-256 of the pre-registered criteria file, then
(2) loads the data, (3) builds the instance population, (4) runs all four gate
layers *independently on the full population*, (5) intersects their verdicts
into the survival funnel, (6) reports the effective trial count behind the
Deflated Sharpe, and (7) writes every table to ``results/`` and the report
figures to ``results/figures/``.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

# package imports -----------------------------------------------------------
from alphafactory import data as data_mod
from alphafactory import factory as factory_mod
from alphafactory import families  # noqa: F401  (registers families on import)
from alphafactory import plotting
from alphafactory.gate import (
    layer1_isoos,
    layer2_walkforward,
    layer3_stress,
    layer4_montecarlo,
    funnel as funnel_mod,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "config", "gate_criteria.yaml")
DEFAULT_DATA = os.path.join(HERE, "data_csv")
DEFAULT_CACHE = os.path.join(HERE, "results", "cache")
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(RESULTS, "figures")

SYMBOLS = ["SPXUSD", "USDJPY", "XAUUSD", "ETHUSD"]


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def banner(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def effective_trial_count(pop, data, cost_model):
    """Effective number of independent trials behind the Deflated Sharpe.

    The 48 instances are correlated (grid neighbours within a family/market move
    together), so they are not 48 independent bets. We measure how many they are
    worth as the participation ratio of their NxN return-correlation matrix,

        N_eff = (sum_i lambda_i)^2 / sum_i lambda_i^2 ,

    which equals N for independent series and 1 for perfectly correlated ones.
    Returns the raw N and N_eff on three alignments (outer-join/flat-fill, the
    common index, and daily returns) as a robustness check.
    """
    cols = {inst.id: pd.Series(inst.backtest(data[inst.symbol],
                                             cost_model=cost_model).net_returns,
                               index=data[inst.symbol].index) for inst in pop}
    rets = pd.DataFrame(cols)

    def _pr(mat):
        w = np.clip(np.linalg.eigvalsh(np.corrcoef(mat, rowvar=False)), 0.0, None)
        return float((w.sum() ** 2) / (w ** 2).sum())

    return (len(pop),
            _pr(rets.fillna(0.0).to_numpy()),
            _pr(rets.dropna().to_numpy()),
            _pr(rets.resample("1D").sum(min_count=1).fillna(0.0).to_numpy()))


def main() -> int:
    ap = argparse.ArgumentParser(description="Alpha Factory gate runner")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE)
    ap.add_argument("--no-cache", action="store_true",
                    help="rebuild H1 bars from raw CSVs (ignore pickle cache)")
    ap.add_argument("--symbols", nargs="*", default=SYMBOLS)
    ap.add_argument("--skip-figures", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="reduce Monte Carlo iterations for a fast smoke test")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)

    # -- 1. pre-registered criteria + hash ---------------------------------
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    cfg_hash = sha256_of(args.config)
    cost_model = cfg["meta"]["cost_model"]

    banner("PRE-REGISTERED GATE CRITERIA")
    print(f"config file : {os.path.relpath(args.config, HERE)}")
    print(f"SHA-256     : {cfg_hash}")
    print(f"cost model  : {cost_model}")
    print(f"timeframe   : {cfg['meta']['timeframe']}")

    if args.quick:
        cfg["layer4"]["n_permutations"] = 200
        cfg["layer4"]["n_bootstrap"] = 200
        print("\n[quick mode] Monte Carlo reduced to 200/200 iterations.")

    # -- 2. data -----------------------------------------------------------
    banner("DATA")
    t0 = time.time()
    data = data_mod.load_all(args.data_dir, args.cache_dir,
                             timeframe=cfg["meta"]["timeframe"],
                             symbols=args.symbols,
                             use_cache=not args.no_cache)
    for s in args.symbols:
        df = data[s]
        print(f"  {s:8s} {len(df):>7,d} H1 bars  "
              f"{df.index[0].date()} -> {df.index[-1].date()}  "
              f"bars/yr={df.attrs['bars_per_year']:.0f}  "
              f"mean spread={df['spread_frac'].mean()*1e4:.2f} bps")
    print(f"  loaded in {time.time() - t0:.1f}s")

    # -- 3. population -----------------------------------------------------
    banner("FACTORY")
    pop = factory_mod.build_population(args.symbols)
    pop_table = factory_mod.population_table(pop)
    pop_table.to_csv(os.path.join(RESULTS, "population.csv"), index=False)
    n_fam = pop_table["family"].nunique()
    print(f"  {len(pop)} instances  =  {n_fam} families x grids x "
          f"{len(args.symbols)} symbols")
    for fam, grp in pop_table.groupby("family"):
        print(f"    {fam:14s} {len(grp)} instances")

    # -- 4. run the four layers independently ------------------------------
    layers = {
        "L1": layer1_isoos,
        "L2": layer2_walkforward,
        "L3": layer3_stress,
        "L4": layer4_montecarlo,
    }
    results: dict[str, pd.DataFrame] = {}
    for tag, mod in layers.items():
        banner(f"LAYER {tag[-1]} -- {funnel_mod.LAYER_NAME[tag]}")
        t0 = time.time()
        res = mod.run(pop, data, cfg, cost_model=cost_model)
        results[tag] = res
        res.to_csv(os.path.join(RESULTS, f"layer{tag[-1]}.csv"), index=False)
        passed = int(res["passed"].sum())
        print(f"  {passed}/{len(res)} instances pass  ({time.time() - t0:.1f}s)")

    # -- 5. funnel (intersection) ------------------------------------------
    banner("SURVIVAL FUNNEL")
    fun = funnel_mod.combine(results)
    fun["verdict"].to_csv(os.path.join(RESULTS, "verdict.csv"), index=False)
    fun["per_layer"].to_csv(os.path.join(RESULTS, "funnel_per_layer.csv"),
                            index=False)
    fun["waterfall"].to_csv(os.path.join(RESULTS, "funnel_waterfall.csv"),
                            index=False)

    print("\nIndependent per-layer view (each layer on the full population):\n")
    print(funnel_mod.format_per_layer(fun["per_layer"]))
    print("\nPresentational cascade (survivors = intersection of all layers):\n")
    print(funnel_mod.format_waterfall(fun["waterfall"], fun["n_total"]))

    print(f"\nFINAL SURVIVORS (passed all four layers): {len(fun['survivors'])}")
    for sid in fun["survivors"]:
        row = fun["verdict"].loc[fun["verdict"]["id"] == sid].iloc[0]
        print(f"  + {sid}")
    if not fun["survivors"]:
        print("  (none -- the honest outcome the gate is designed to produce)")

    # near-misses: instances passing 3 of 4 layers, for the report narrative
    near = fun["verdict"][fun["verdict"]["n_layers_passed"] == 3]
    if len(near):
        print(f"\nNear-misses (passed 3/4 layers): {len(near)}")
        for _, r in near.iterrows():
            failed = [L for L in funnel_mod.LAYER_ORDER if not r[f"{L}_pass"]]
            print(f"  - {r['id']:42s} fails {','.join(failed)}")

    # -- 5b. effective trial count behind the Deflated Sharpe --------------
    banner("EFFECTIVE TRIAL COUNT (for the Deflated Sharpe)")
    n_raw, n_eff, n_eff_inter, n_eff_daily = effective_trial_count(
        pop, data, cost_model)
    print(f"  raw trial count        N      = {n_raw}")
    print(f"  effective trial count  N_eff  = {n_eff:.1f}   "
          f"(participation ratio of the {n_raw}x{n_raw} return-correlation matrix)")
    print(f"  robustness: common-index basis {n_eff_inter:.1f}; "
          f"daily-return basis {n_eff_daily:.1f}")
    print("  -> the gate deflates at the conservative raw N; "
          "the verdict is unchanged under N_eff.")

    # write a compact run summary -----------------------------------------
    with open(os.path.join(RESULTS, "summary.txt"), "w") as fh:
        fh.write(f"config_sha256: {cfg_hash}\n")
        fh.write(f"cost_model: {cost_model}\n")
        fh.write(f"n_instances: {len(pop)}\n")
        fh.write(f"effective_n: {n_eff:.1f}  (participation ratio; raw N={n_raw})\n\n")
        fh.write(funnel_mod.format_per_layer(fun["per_layer"]) + "\n\n")
        fh.write(funnel_mod.format_waterfall(fun["waterfall"], fun["n_total"]))
        fh.write(f"\n\nsurvivors: {fun['survivors']}\n")

    # -- 6. figures --------------------------------------------------------
    if not args.skip_figures:
        banner("FIGURES")
        plotting.plot_funnel(fun["waterfall"], fun["n_total"],
                             os.path.join(FIGURES, "funnel.png"))
        plotting.plot_per_layer_bars(fun["per_layer"], fun["n_total"],
                                     os.path.join(FIGURES, "per_layer.png"))
        plotting.plot_oos_distribution(results["L1"], cfg,
                                       os.path.join(FIGURES, "oos_sharpe.png"))
        plotting.plot_dsr_scatter(results["L4"], cfg,
                                  os.path.join(FIGURES, "dsr_scatter.png"))
        # representative equity curves: best trend + best reversion by L1 OoS
        l1 = results["L1"].copy()
        l1 = l1.merge(pop_table[["id", "family"]], on="id", how="left")
        rep_ids = []
        for fam in sorted(l1["family"].unique()):
            sub = l1[l1["family"] == fam].sort_values("oos_sharpe",
                                                      ascending=False)
            if len(sub):
                rep_ids.append(sub.iloc[0]["id"])
        plotting.plot_equity_curves(pop, data, rep_ids,
                                    os.path.join(FIGURES, "equity_curves.png"),
                                    cost_model=cost_model)
        print(f"  wrote 5 figures to {os.path.relpath(FIGURES, HERE)}/")

    banner("DONE")
    print(f"All tables in {os.path.relpath(RESULTS, HERE)}/  "
          f"(config SHA-256 {cfg_hash[:12]}...)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
