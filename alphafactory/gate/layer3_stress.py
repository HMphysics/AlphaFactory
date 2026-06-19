"""Layer 3 -- Stress replay.

Each candidate is replayed on its worst real episodes and on a synthetic shock,
all defined in advance:

  * Real windows -- a fixed COVID window (Feb-Apr 2020, always included if it
    overlaps the data) plus the two worst rolling `worst_window_days` windows OF
    THE STRATEGY ITSELF, located from the instance's own equity curve.  Using the
    strategy's own worst windows (rather than the market's worst draw-downs) tests
    each candidate on its genuine worst case in *either* direction -- a mean-
    reversion or short book can bleed during a sharp rally, which a market-drawdown
    window would entirely miss.  "Acceptable behaviour" is numeric: over each window
    the return must stay above a floor and the within-window drawdown below a cap.

  * Synthetic shock -- a volatility-ONLY replay.  We rebuild each symbol's price
    path with its return *deviations* scaled by `vol_mult` while holding the mean
    return fixed: ``r_s = mean + mult * (r - mean)``.  This doubles the volatility
    without doubling the drift.  A raw ``mult * r`` would also double the trend,
    which flatters trend-following (a bigger, more persistent move is *easier* to
    ride); scaling only the deviations is a genuine "twice as turbulent" path.  We
    require the full-sample Sharpe on it to stay non-negative.

An instance passes only if every real-window floor holds AND the synthetic
criterion holds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import stats as stats_mod
from . import common


def _worst_windows(level: pd.Series, win_days: int, n: int):
    """The `n` worst non-overlapping rolling `win_days`-day windows of a
    datetime-indexed level series (a price OR a strategy equity curve), ranked by
    cumulative return over the window (most negative first)."""
    daily = level.resample("1D").last().dropna()
    v = daily.to_numpy(float)
    idx = daily.index
    if len(v) <= win_days + 1:
        return []
    roll = v[win_days:] / v[:-win_days] - 1.0
    order = np.argsort(roll)
    used = np.zeros(len(roll), bool)
    picked = []
    for j in order:
        if used[j]:
            continue
        picked.append((idx[j], idx[j + win_days]))
        lo = max(0, j - win_days)
        hi = min(len(roll), j + win_days + 1)
        used[lo:hi] = True
        if len(picked) >= n:
            break
    return picked


def _covid_window(df: pd.DataFrame, c: dict):
    cs, ce = pd.Timestamp(c["covid_start"]), pd.Timestamp(c["covid_end"])
    if (ce >= df.index[0]) and (cs <= df.index[-1]):
        return [("covid", cs, ce)]
    return []


def _vol_scaled_df(df: pd.DataFrame, mult: float) -> pd.DataFrame:
    """Volatility-only synthetic shock: scale the return *deviations* by `mult`
    while keeping the mean return, rebuild the price path, and scale each bar's
    intrabar range by the same factor.  Doubles volatility, not drift."""
    r = df["ret"].to_numpy(float)
    mu = float(np.nanmean(r))
    r_s = np.clip(mu + mult * (r - mu), -0.99, None)   # vol up, drift unchanged
    close0 = float(df["close"].iloc[0])
    synth_close = close0 * np.cumprod(1.0 + r_s)

    out = df.copy()
    c0 = df["close"].to_numpy(float)
    for col, floor in (("high", 0.01), ("low", 0.01), ("open", 0.01)):
        if col in df.columns:
            frac = df[col].to_numpy(float) / c0 - 1.0          # intrabar excursion
            out[col] = synth_close * np.clip(1.0 + frac * mult, floor, None)
    out["close"] = synth_close
    out["ret"] = r_s
    out.attrs.update(df.attrs)
    return out


def run(pop, data: dict, cfg: dict, cost_model: str = "realized_spread") -> pd.DataFrame:
    c = cfg["layer3"]
    covid_cache, synth_cache = {}, {}
    for sym, df in data.items():
        covid_cache[sym] = _covid_window(df, c)
        synth_cache[sym] = _vol_scaled_df(df, c["vol_mult"])

    rows = []
    for inst in pop:
        df = data[inst.symbol]
        ppy = df.attrs.get("bars_per_year", 252 * 24)
        net = common.full_net_returns(df, inst.family, inst.params, cost_model)

        # stress windows = fixed COVID + the strategy's OWN worst windows
        equity = pd.Series(np.cumprod(1.0 + net), index=df.index)
        own = _worst_windows(equity, c["worst_window_days"], c["n_worst_windows"])
        windows = covid_cache[inst.symbol] + [
            (f"worst{i}", s, e) for i, (s, e) in enumerate(own, 1)]

        worst_ret, worst_dd, failed_win = np.inf, 0.0, ""
        per_window = {}
        for label, s, e in windows:
            mask = np.asarray((df.index >= s) & (df.index <= e))
            seg = net[mask]
            if seg.size == 0:
                continue
            wret = float(np.prod(1.0 + seg) - 1.0)
            wdd = stats_mod.max_drawdown(seg)
            per_window[label] = (wret, wdd)
            if wret < worst_ret:
                worst_ret = wret
            if wdd > worst_dd:
                worst_dd = wdd
            if (wret < c["stress_min_return"] or wdd > c["stress_max_drawdown"]) and not failed_win:
                failed_win = label

        synth = synth_cache[inst.symbol]
        synth_net = common.full_net_returns(synth, inst.family, inst.params, cost_model)
        synth_sharpe = stats_mod.sharpe_annual(synth_net, ppy)

        ok_real = (worst_ret >= c["stress_min_return"]) and (worst_dd <= c["stress_max_drawdown"])
        ok_synth = synth_sharpe >= c["synthetic_min_sharpe"]
        passed = bool(ok_real and ok_synth)
        if passed:
            reason = ""
        elif not ok_real:
            reason = f"blows up in stress window ({failed_win or 'real'})"
        else:
            reason = "fails under 2x volatility"
        rows.append({
            "id": inst.id, "layer": "L3", "passed": passed, "reason": reason,
            "worst_window_return": (None if worst_ret == np.inf else worst_ret),
            "worst_window_dd": worst_dd, "synth_sharpe": synth_sharpe,
            "n_windows": len(per_window),
        })
    return pd.DataFrame(rows)
