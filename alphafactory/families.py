"""Strategy families ("root ideas").

Each family is a small object exposing:
  * `name`                  -- identifier;
  * `params`                -- the (at most two) free, optimizable parameters;
  * `grid`                  -- the economically-sensible default grid of pairs;
  * `signal(df, p)`         -- target position on closed bars: a continuous risk-
                               scaled exposure or {-1, 0, +1}; the backtester sizes
                               P&L and costs by the position magnitude, so either
                               convention is supported;
  * `bounds`                -- (min, is_int) per parameter for validity / perturbation.

Adding a third root idea is intentionally trivial: subclass `Family`, implement
`signal`, set `grid`/`bounds`, and call `register(...)`.  The factory, backtester
and gate never name a specific family, so nothing downstream changes.

The two shipped families are deliberately *opposed* market hypotheses, so a
population that "works" cannot be a single bet wearing two hats.

Family A -- VOL-TARGETED TIME-SERIES MOMENTUM (continuation).
    Hypothesis: information diffuses slowly and flows herd, so returns are
    positively autocorrelated over multi-day horizons; volatility targeting is the
    most replicated way to *improve* a trend signal's risk-adjusted return, cutting
    size into turbulence and adding it back in calm regimes.  Direction is the sign
    of the `lookback`-bar return; the position is then scaled by (trailing typical
    volatility / current `vol_win`-bar volatility).  Both volatility estimates are
    rolling (backward-only -> no look-ahead); the normalising horizon and the
    leverage cap are fixed by design.  Free params: `lookback`, `vol_win` (bars).

Family B -- FILTERED MEAN REVERSION (Bollinger band, mean-exit, trend gate).
    Hypothesis: over short horizons liquidity demand and over-reaction push price
    away from a local fair value to which it reverts -- but naive fading dies by
    fighting genuine trends, so a *complete* reversion needs an exit rule and a
    trend filter.  We fade a deviation of `k` standard deviations from the
    `n`-bar mean (short if stretched up, long if stretched down), **take profit
    when price crosses back through that mean**, and stay flat between trades.  A
    fixed longer-horizon z-score gates entries: we do **not** fade while a strong
    trend is underway.  Free params: `n` (look-back), `k` (entry threshold in std).
    Everything else fixed: mean-exit, the trend-gate window and threshold, the
    symmetric band, flat-between-trades.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class Family:
    name: str = "base"
    params: tuple[str, ...] = ()
    grid: list[dict] = []
    bounds: dict[str, tuple[float, bool]] = {}     # name -> (min_value, is_int)

    def signal(self, df: pd.DataFrame, p: dict) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    # -- shared helpers ----------------------------------------------------
    def valid(self, p: dict) -> bool:
        for k, (lo, is_int) in self.bounds.items():
            v = p[k]
            if is_int and int(v) != v:
                return False
            if v < lo:
                return False
        return True

    def perturb(self, p: dict, frac: float = 0.20) -> list[dict]:
        """One-at-a-time +/- `frac` perturbations of each free parameter.

        Integer parameters are rounded; values are clipped to their lower bound.
        Invalid / duplicate neighbours are removed, as is the base point itself.
        """
        neigh: list[dict] = []
        for k in self.params:
            lo, is_int = self.bounds[k]
            for s in (1 - frac, 1 + frac):
                v = p[k] * s
                if is_int:
                    v = int(round(v))
                v = max(v, lo)
                q = dict(p)
                q[k] = v
                neigh.append(q)
        out, seen = [], {tuple(sorted(p.items()))}
        for q in neigh:
            key = tuple(sorted(q.items()))
            if key in seen or not self._constraint_ok(q):
                continue
            seen.add(key)
            out.append(q)
        return out

    def _constraint_ok(self, p: dict) -> bool:
        return self.valid(p)


class VolTargetMomentum(Family):
    name = "voltarget_mom"
    params = ("lookback", "vol_win")
    bounds = {"lookback": (2, True), "vol_win": (2, True)}
    grid = [{"lookback": lb, "vol_win": vw}
            for lb in (48, 240) for vw in (24, 72, 168)]

    NORM_WIN = 24 * 63          # ~3 months of H1 bars: trailing "typical" vol (fixed)
    MAX_LEV = 2.0               # leverage cap, fixed by design

    def signal(self, df: pd.DataFrame, p: dict) -> np.ndarray:
        c = df["close"]
        lookback = int(p["lookback"])
        vol_win = int(p["vol_win"])
        ret = df["ret"]

        direction = np.sign((c - c.shift(lookback)).to_numpy())     # +1 / -1 / 0

        short_vol = ret.rolling(vol_win, min_periods=vol_win).std()
        norm_vol = ret.rolling(self.NORM_WIN,
                               min_periods=self.NORM_WIN // 2).std()
        # target the asset's own trailing vol: scale up in calm, down in turbulence
        scale = (norm_vol / short_vol.replace(0.0, np.nan)).to_numpy()
        scale = np.clip(scale, 0.0, self.MAX_LEV)

        pos = direction * scale
        warmup = np.isnan(scale) | (np.arange(len(c)) < lookback)
        pos[warmup] = 0.0
        return pos


class BollingerRevert(Family):
    name = "bollinger_revert"
    params = ("n", "k")
    bounds = {"n": (3, True), "k": (0.25, False)}
    grid = [{"n": n, "k": k} for n in (24, 96) for k in (1.0, 1.5, 2.0)]

    TREND_WIN = 480            # ~1 month of H1 bars: trend-strength reference (fixed)
    TREND_GATE = 1.0           # don't fade while |long-horizon z| exceeds this (fixed)

    def signal(self, df: pd.DataFrame, p: dict) -> np.ndarray:
        c = df["close"]
        n = int(p["n"])
        k = float(p["k"])

        mu = c.rolling(n, min_periods=n).mean()
        sd = c.rolling(n, min_periods=n).std(ddof=0)
        z = ((c - mu) / sd.replace(0.0, np.nan)).to_numpy()

        # longer-horizon z = trend strength; fade only when the trend is not strong
        mu_l = c.rolling(self.TREND_WIN, min_periods=self.TREND_WIN).mean()
        sd_l = c.rolling(self.TREND_WIN, min_periods=self.TREND_WIN).std(ddof=0)
        z_long = ((c - mu_l) / sd_l.replace(0.0, np.nan)).to_numpy()
        calm = np.abs(z_long) < self.TREND_GATE

        # event-driven, mean-exit state machine (vectorised via forward-fill)
        raw = np.full(len(c), np.nan)
        raw[(z < -k) & calm] = 1.0       # stretched down -> fade long
        raw[(z > k) & calm] = -1.0       # stretched up   -> fade short
        zprev = np.concatenate([[np.nan], z[:-1]])
        raw[(z * zprev) < 0] = 0.0       # price crossed back through the mean -> exit
        sig = pd.Series(raw).ffill().fillna(0.0).to_numpy(copy=True)

        warm = np.isnan(z) | np.isnan(z_long)
        sig[warm] = 0.0                  # flat until both windows are valid
        return sig


# --------------------------------------------------------------------------
# Registry.  `register()` makes a third root idea a one-line addition.
# --------------------------------------------------------------------------
_REGISTRY: dict[str, Family] = {}


def register(fam: Family) -> None:
    _REGISTRY[fam.name] = fam


def get(name: str) -> Family:
    return _REGISTRY[name]


def all_families() -> dict[str, Family]:
    return dict(_REGISTRY)


register(VolTargetMomentum())
register(BollingerRevert())
