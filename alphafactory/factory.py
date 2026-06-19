"""The strategy factory: expand root ideas into a candidate population.

An *instance* is a fully specified candidate strategy = (family, symbol,
parameter pair).  The population is the Cartesian product of every family's grid
with every symbol.  With two families of six grid points each over four symbols
this yields 48 candidates (>= the required 30).

Because the factory only iterates `families.all_families()` x `symbols`, adding a
third root idea (one `register(...)` call) automatically enlarges the population
with zero further changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import backtest as bt
from . import families as fam_mod


def _fmt_params(p: dict) -> str:
    parts = []
    for k, v in p.items():
        parts.append(f"{k}{int(v)}" if float(v).is_integer() else f"{k}{v:g}")
    return "_".join(parts)


@dataclass
class Instance:
    family: str
    symbol: str
    params: dict
    id: str = field(default="")

    def __post_init__(self):
        if not self.id:
            self.id = f"{self.family}|{self.symbol}|{_fmt_params(self.params)}"

    def signal(self, df: pd.DataFrame) -> np.ndarray:
        return fam_mod.get(self.family).signal(df, self.params)

    def backtest(self, df: pd.DataFrame, cost_model: str = "realized_spread"):
        return bt.run_backtest(df, self.signal(df), cost_model=cost_model)


def build_population(symbols: list[str]) -> list[Instance]:
    pop: list[Instance] = []
    for fname, fam in fam_mod.all_families().items():
        for p in fam.grid:
            for s in symbols:
                pop.append(Instance(family=fname, symbol=s, params=dict(p)))
    return pop


def population_table(pop: list[Instance]) -> pd.DataFrame:
    return pd.DataFrame([
        {"id": i.id, "family": i.family, "symbol": i.symbol, **i.params} for i in pop
    ])
