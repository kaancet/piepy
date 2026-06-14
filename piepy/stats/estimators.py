"""Point estimates with uncertainty.

Every estimator returns an :class:`Estimate` (value + confidence interval + n), so plotting
and per-paradigm summaries get one consistent, error-bar-ready object. Pure numpy/scipy;
analytic CIs where they exist (Wilson for proportions, t for means) so the common
psychophysics path needs no bootstrap, with :func:`bootstrap_ci` as the general fallback.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats as sps

__all__ = ["Estimate", "proportion_ci", "mean_ci", "median_ci", "bootstrap_ci"]

_NAN = float("nan")


@dataclass(frozen=True)
class Estimate:
    """A scalar estimate with a confidence interval.

    Attributes:
        value: the point estimate.
        ci_low / ci_high: confidence-interval bounds.
        n: sample size the estimate is based on.
        method: how it was computed (for provenance, e.g. ``"proportion[wilson]"``).
    """

    value: float
    ci_low: float
    ci_high: float
    n: int
    method: str

    @property
    def err(self) -> tuple[float, float]:
        """``(-err, +err)`` magnitudes, ready for asymmetric matplotlib/behaviz errorbars."""
        return (self.value - self.ci_low, self.ci_high - self.value)


def _z(confidence: float) -> float:
    return float(sps.norm.ppf(1 - (1 - confidence) / 2))


def proportion_ci(
    successes: int, n: int, *, confidence: float = 0.95, method: str = "wilson"
) -> Estimate:
    """CI for a binomial proportion (e.g. hit rate).

    Defaults to the Wilson score interval, which stays inside ``[0, 1]`` and behaves well at
    extreme rates (0% / 100%) and small n -- exactly the psychophysics regime where the naive
    normal interval breaks.
    """
    successes, n = int(successes), int(n)
    if n <= 0:
        return Estimate(_NAN, _NAN, _NAN, 0, f"proportion[{method}]")
    p = successes / n
    z = _z(confidence)
    if method == "wilson":
        denom = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        half = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
        lo, hi = center - half, center + half
    elif method == "normal":
        half = z * np.sqrt(p * (1 - p) / n)
        lo, hi = p - half, p + half
    else:
        raise ValueError(
            f"Unknown proportion CI method {method!r}; use 'wilson' or 'normal'."
        )
    return Estimate(
        p, max(0.0, float(lo)), min(1.0, float(hi)), n, f"proportion[{method}]"
    )


def mean_ci(x: ArrayLike, *, confidence: float = 0.95) -> Estimate:
    """Mean with a Student-t confidence interval (NaNs dropped)."""
    a = np.asarray(x, dtype=float)
    a = a[~np.isnan(a)]
    n = a.size
    if n == 0:
        return Estimate(_NAN, _NAN, _NAN, 0, "mean")
    m = float(a.mean())
    if n == 1:
        return Estimate(m, m, m, 1, "mean")
    half = float(sps.sem(a) * sps.t.ppf((1 + confidence) / 2, n - 1))
    return Estimate(m, m - half, m + half, n, "mean")


def bootstrap_ci(
    x: ArrayLike,
    statistic=np.mean,
    *,
    confidence: float = 0.95,
    n_boot: int = 2000,
    seed: int | None = None,
    label: str = "bootstrap",
) -> Estimate:
    """Percentile bootstrap CI for any vectorized ``statistic`` (NaNs dropped).

    ``statistic`` must accept an ``axis`` argument (``np.mean``, ``np.median``, ``np.std`` ...),
    which lets the whole resample be computed in one vectorized call -- fast even at large
    ``n_boot``.
    """
    a = np.asarray(x, dtype=float)
    a = a[~np.isnan(a)]
    n = a.size
    if n == 0:
        return Estimate(_NAN, _NAN, _NAN, 0, label)
    point = float(statistic(a))
    rng = np.random.default_rng(seed)
    resamples = a[rng.integers(0, n, size=(n_boot, n))]  # (n_boot, n)
    boot = statistic(resamples, axis=1)
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(boot, [alpha, 1 - alpha])
    return Estimate(point, float(lo), float(hi), n, label)


def median_ci(
    x: ArrayLike, *, confidence: float = 0.95, n_boot: int = 2000, seed: int | None = None
) -> Estimate:
    """Median with a bootstrap confidence interval."""
    return bootstrap_ci(
        x, np.median, confidence=confidence, n_boot=n_boot, seed=seed, label="median"
    )
