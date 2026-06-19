"""Shared helpers for the four gate layers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import backtest as bt
from .. import families as fam_mod
from .. import stats as stats_mod


def full_net_returns(df: pd.DataFrame, family: str, params: dict,
                     cost_model: str = "realized_spread") -> np.ndarray:
    """Net per-bar returns for (family, params) over the whole series.

    Signals are computed on the full series (rolling windows only ever look
    back, so this introduces no lookahead) and the engine applies the one-bar
    execution lag.  Slicing afterwards keeps position continuity across any
    sub-period boundary.
    """
    sig = fam_mod.get(family).signal(df, params)
    return bt.backtest_returns_only(
        df["ret"].to_numpy(float), df["spread_frac"].to_numpy(float),
        sig, df.attrs.get("asset_class", "index"), cost_model)


def sharpe_on_mask(net: np.ndarray, mask: np.ndarray, ppy: float) -> float:
    r = net[mask]
    return stats_mod.sharpe_annual(r, ppy)


def chronological_split(n: int, is_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    cut = int(round(n * is_fraction))
    is_mask = np.zeros(n, bool)
    oos_mask = np.zeros(n, bool)
    is_mask[:cut] = True
    oos_mask[cut:] = True
    return is_mask, oos_mask
