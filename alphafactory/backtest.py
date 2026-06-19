"""Vectorized backtest engine.

Execution convention (lookahead-free):
  * a strategy emits a target position `signal_t in {-1, 0, +1}` using information
    available **at the close of bar t** (closed bars only);
  * that position is acted on with a one-bar lag, i.e. it earns the close-to-close
    return of bar t+1.  Concretely `pos = signal.shift(1)`;
  * costs are charged on the turnover |pos_t - pos_{t-1}| using the realized
    spread at the fill bar.

Net per-bar return:  ret_t = pos_t * r_t - cost_t.

The engine takes a *signal array* (already computed by a strategy family on the
price series) so that the family, the backtester and the gate stay cleanly
separable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import costs as costs_mod
from . import stats as stats_mod


@dataclass
class BacktestResult:
    net_returns: np.ndarray          # per-bar net returns, aligned to index[1:]
    gross_returns: np.ndarray
    positions: np.ndarray
    turnover: np.ndarray
    index: pd.DatetimeIndex
    periods_per_year: float
    metrics: dict = field(default_factory=dict)

    @property
    def n_trades(self) -> int:
        # a "trade" = a change in position magnitude (open/close/flip counts moves)
        return int(np.count_nonzero(self.turnover))

    @property
    def avg_turnover(self) -> float:
        return float(np.mean(self.turnover))


def run_backtest(price_df: pd.DataFrame, signal: np.ndarray, *,
                 cost_model: str = "realized_spread") -> BacktestResult:
    """Backtest a signal on a resampled OHLC frame.

    `signal` must be length == len(price_df), with values in {-1, 0, +1} and use
    only information up to and including each bar's close (the engine applies the
    one-bar execution lag itself).
    """
    r = price_df["ret"].to_numpy(float)          # close-to-close returns at bar t
    spread = price_df["spread_frac"].to_numpy(float)
    asset_class = price_df.attrs.get("asset_class", "index")
    ppy = price_df.attrs.get("bars_per_year", 252 * 24)

    sig = np.asarray(signal, float)
    pos = np.empty_like(sig)
    pos[0] = 0.0
    pos[1:] = sig[:-1]                            # one-bar execution lag
    prev = np.empty_like(pos)
    prev[0] = 0.0
    prev[1:] = pos[:-1]
    turnover = np.abs(pos - prev)

    cost = costs_mod.trade_costs(turnover, spread, cost_model, asset_class)
    gross = pos * r
    net = gross - cost

    res = BacktestResult(
        net_returns=net, gross_returns=gross, positions=pos, turnover=turnover,
        index=price_df.index, periods_per_year=ppy,
    )
    m = stats_mod.summarize(net, ppy)
    m["n_trades"] = res.n_trades
    m["avg_turnover"] = res.avg_turnover
    m["exposure"] = float(np.mean(np.abs(pos)))
    m["gross_sharpe"] = stats_mod.sharpe_annual(gross, ppy)
    res.metrics = m
    return res


def backtest_returns_only(r: np.ndarray, spread: np.ndarray, signal: np.ndarray,
                          asset_class: str, model: str = "realized_spread") -> np.ndarray:
    """Fast path returning only the net-return array (used in hot loops)."""
    sig = np.asarray(signal, float)
    pos = np.empty_like(sig)
    pos[0] = 0.0
    pos[1:] = sig[:-1]
    prev = np.empty_like(pos)
    prev[0] = 0.0
    prev[1:] = pos[:-1]
    turnover = np.abs(pos - prev)
    cost = costs_mod.trade_costs(turnover, spread, model, asset_class)
    return pos * r - cost
