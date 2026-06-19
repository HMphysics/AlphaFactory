"""Performance statistics and the multiple-testing machinery (PSR / DSR).

All Sharpe ratios come in two flavours:
  * `sharpe` (annualised) for human-readable reporting;
  * the *per-observation* Sharpe used inside the Probabilistic / Deflated Sharpe
    formulas, which are defined on the raw return frequency and the raw sample
    size `n` -- annualising before plugging into those formulas is a common and
    serious error, so we keep them separate.

Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014): given `N` independent
trials, the expected maximum Sharpe under the null of no skill is

    E[max SR] = sqrt(Var_SR) * ((1-g) * Z^-1(1 - 1/N) + g * Z^-1(1 - 1/(N e)))

with g the Euler-Mascheroni constant.  The DSR is the Probabilistic Sharpe Ratio
evaluated against that deflated benchmark, i.e. the probability that the true
(per-observation) Sharpe exceeds what the best of N noise draws would produce.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

EULER_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------------------
# Basic return-series statistics
# ---------------------------------------------------------------------------
def sharpe_annual(returns: np.ndarray, periods_per_year: float) -> float:
    r = np.asarray(returns, float)
    sd = r.std(ddof=1)
    if not np.isfinite(sd) or sd == 0 or len(r) < 2:
        return 0.0
    return float(r.mean() / sd * np.sqrt(periods_per_year))


def sharpe_per_obs(returns: np.ndarray) -> float:
    """Non-annualised (per-observation) Sharpe, for the PSR/DSR formulas."""
    r = np.asarray(returns, float)
    sd = r.std(ddof=1)
    if not np.isfinite(sd) or sd == 0 or len(r) < 2:
        return 0.0
    return float(r.mean() / sd)


def equity_curve(returns: np.ndarray) -> np.ndarray:
    return np.cumprod(1.0 + np.asarray(returns, float))


def max_drawdown(returns: np.ndarray) -> float:
    """Maximum drawdown (as a positive fraction) of the compounded curve."""
    eq = equity_curve(returns)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    return float(-dd.min()) if len(dd) else 0.0


def cagr(returns: np.ndarray, periods_per_year: float) -> float:
    r = np.asarray(returns, float)
    if len(r) == 0:
        return 0.0
    total = float(np.prod(1.0 + r))
    if total <= 0:
        return -1.0
    years = len(r) / periods_per_year
    return total ** (1.0 / years) - 1.0 if years > 0 else 0.0


def summarize(returns: np.ndarray, periods_per_year: float) -> dict:
    r = np.asarray(returns, float)
    md = max_drawdown(r)
    sh = sharpe_annual(r, periods_per_year)
    total = float(np.prod(1.0 + r) - 1.0) if len(r) else 0.0
    out = {
        "n": int(len(r)),
        "sharpe": sh,
        "sharpe_per_obs": sharpe_per_obs(r),
        "cagr": cagr(r, periods_per_year),
        "total_return": total,
        "max_drawdown": md,
        "calmar": (cagr(r, periods_per_year) / md) if md > 0 else 0.0,
        "vol_annual": float(r.std(ddof=1) * np.sqrt(periods_per_year)) if len(r) > 1 else 0.0,
        "skew": float(stats.skew(r)) if len(r) > 2 else 0.0,
        "kurtosis": float(stats.kurtosis(r, fisher=False)) if len(r) > 3 else 3.0,
    }
    return out


# ---------------------------------------------------------------------------
# Probabilistic & Deflated Sharpe Ratio
# ---------------------------------------------------------------------------
def probabilistic_sharpe_ratio(sr_per_obs: float, n: int, skew: float,
                               kurtosis: float, sr_benchmark: float = 0.0) -> float:
    """PSR: P(true per-obs Sharpe > benchmark), accounting for skew & kurtosis.

    `kurtosis` is the *non-excess* kurtosis (normal == 3).
    """
    if n < 2:
        return 0.5
    denom = 1.0 - skew * sr_per_obs + (kurtosis - 1.0) / 4.0 * sr_per_obs ** 2
    denom = max(denom, 1e-12)
    z = (sr_per_obs - sr_benchmark) * np.sqrt(n - 1) / np.sqrt(denom)
    return float(stats.norm.cdf(z))


def expected_max_sharpe(var_sr_across_trials: float, n_trials: int) -> float:
    """E[max SR] across `n_trials` independent strategies under the null.

    `var_sr_across_trials` is the variance of the *per-observation* Sharpe
    estimates across the trials.
    """
    if n_trials < 2 or var_sr_across_trials <= 0:
        return 0.0
    sigma = np.sqrt(var_sr_across_trials)
    a = stats.norm.ppf(1.0 - 1.0 / n_trials)
    b = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(sigma * ((1.0 - EULER_GAMMA) * a + EULER_GAMMA * b))


def deflated_sharpe_ratio(sr_per_obs: float, n: int, skew: float, kurtosis: float,
                          var_sr_across_trials: float, n_trials: int) -> tuple[float, float]:
    """Return (DSR, deflated benchmark SR*)."""
    sr_star = expected_max_sharpe(var_sr_across_trials, n_trials)
    dsr = probabilistic_sharpe_ratio(sr_per_obs, n, skew, kurtosis, sr_benchmark=sr_star)
    return dsr, sr_star
