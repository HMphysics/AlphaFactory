# The Alpha Factory

A miniature strategy factory and a **validation gate**, built for
the Clarion Capital take-home. The point of the exercise is not to find a winning
strategy — it is to build a machine that manufactures candidates and then judges
them honestly. On the supplied data, with pre-registered criteria, **48 candidates
enter the gate and 0 survive.** That is the intended, defensible outcome.

---

## Quick start

```bash
pip install -r requirements.txt

# put the four supplied CSVs in ./data_csv/  (see "Data" below), then:
python run.py
```

`run.py` is the single command that reproduces the entire funnel. It prints the
SHA-256 of the pre-registered criteria, loads and resamples the data, builds the
48-instance population, runs all four gate layers **independently on the full
population**, intersects their verdicts into the survival funnel, writes every
table to `results/`, and renders the report figures to `results/figures/`.

Useful flags:

```bash
python run.py --quick        # reduced Monte Carlo (200 iters) — fast smoke test
python run.py --no-cache     # rebuild H1 bars from the raw CSVs (ignore cache)
python run.py --symbols SPXUSD XAUUSD    # run a subset of markets
python run.py --skip-figures             # tables only
```

A full run takes a few minutes (Layer 4's permutation + bootstrap dominates);
`--quick` finishes in well under a minute and gives the same verdict.

---

## Data

Place the four MetaTrader M1 CSVs in `./data_csv/` (the loader matches them by
glob, so the exact MetaTrader filenames are fine):

```
data_csv/
  SPXUSD_M1_*.csv      # S&P 500 index
  USDJPY*_M1_*.csv     # USD/JPY
  XAUUSD*_M1_*.csv     # gold
  ETHUSD_M1_*.csv      # ETH/USD
```

They are not committed (each is ~150 MB). The loader resamples M1 → **H1**,
drops no-trade hours rather than forward-filling (preserving real gap risk),
computes a per-bar realised spread, and caches the result as a pickle in
`results/cache/` keyed on the source file's size and mtime, so subsequent runs
are instant.

---

## Architecture — factory, backtester, and gate are separable

```
alphafactory/
  data.py              # M1 CSV -> clean H1 bars (+ spread_frac, ret), disk cache
  families.py          # the two root ideas + a registry; add a 3rd = register()
  factory.py           # expands families x grids x symbols -> 48 Instances
  costs.py             # realised-spread + commission cost model
  backtest.py          # vectorized, look-ahead-free engine (one-bar exec lag)
  stats.py             # Sharpe, drawdown, PSR, expected-max-Sharpe, DSR
  plotting.py          # report figures
  gate/
    common.py          # shared helpers (full net returns, masks, splits)
    layer1_isoos.py    # L1: IS/OoS split + +/-20% parameter robustness
    layer2_walkforward.py  # L2: rolling re-optimization (walk-forward)
    layer3_stress.py   # L3: worst real windows (incl. COVID) + synthetic shock
    layer4_montecarlo.py   # L4: permutation test, block bootstrap, Deflated Sharpe
    funnel.py          # intersects the four verdicts -> survival funnel
config/
  gate_criteria.yaml   # ALL pass/fail thresholds, fixed before running
run.py                 # one-command reproduction of the whole funnel
report/                # report.tex + report.pdf (<=6 pages) + figures
results/               # written by run.py (tables, figures, cache)
```

* **The factory** (`families.py` + `factory.py`) knows nothing about the gate.
  Each family is a subclass exposing `signal(df, params)`, a parameter `grid`,
  and parameter `bounds`. Two opposed families ship: `voltarget_mom`
  (vol-targeted time-series momentum -- trend; free params `lookback`,
  `vol_win`) and `bollinger_revert` (Bollinger-band mean reversion with a
  take-profit at the mean and a trend-strength gate -- reversal; free params
  `n`, `k`). **Adding a third root idea is a single `families.register()` call**
  -- the population builder and all four gate layers pick it up with no other edits.

* **The backtester** (`backtest.py`) is a pure function of `(returns, spreads,
  signal)`. Signals use information only up to each bar's close; the engine
  applies a strict one-bar execution lag and charges costs on turnover. It has no
  dependency on the factory or the gate.

* **The gate** (`gate/`) consumes a population and a data dict. Each layer is an
  independent module exposing `run(pop, data, cfg, cost_model)` and is executed
  on the **full population**; `funnel.py` then takes the **intersection** of the
  four verdicts. The funnel is reported both as an independent per-layer view and
  as a presentational L1→L2→L3→L4 cascade.

---

## Pre-registration (the honesty contract)

Every numeric threshold lives in `config/gate_criteria.yaml` and is fixed
**before** the gate runs. `run.py` prints the file's SHA-256 so the criteria are
timestamped against the output; any change yields a different hash and is
disclosed in the report. The report documents the verdict, the per-layer
findings, the Deflated Sharpe analysis (the multiple-testing correction is the
whole point), and the limitations — see `report/report.pdf`.

---

## What `run.py` writes to `results/`

| file | contents |
|------|----------|
| `population.csv` | the 48 instances (family, symbol, params) |
| `layer1.csv` … `layer4.csv` | per-instance verdict + metrics for each layer |
| `verdict.csv` | merged per-instance verdict across all four layers |
| `funnel_per_layer.csv` | independent pass/fail counts per layer |
| `funnel_waterfall.csv` | the intersection cascade |
| `summary.txt` | the config hash + both funnel views + survivor list |
| `figures/*.png` | the figures used in the report |


See `DECISIONS.md` for the full chronological record of design decisions.
