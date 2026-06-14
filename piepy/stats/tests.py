"""Statistical tests with a consistent result type.

``compare(x1, x2, method=...)`` is the workhorse for two-group comparisons (reaction times,
rates, ... typically obtained via :func:`piepy.stats.group_arrays`). It returns a
:class:`TestResult` carrying the statistic, p-value, group sizes, and an effect size.

The 2-D distribution tests (``ks_2d``, ``energy_2d``) are thin wrappers returning ``TestResult``;
their heavy math currently lives in ``core/statistics.py`` and will physically relocate here when
that module's plotter consumers are migrated (Phase 4). ``mantel_haenszel`` (a k-table omnibus
test for equality of psychometric functions) takes a dict of contingency tables rather than two
samples, so it keeps its native result shape and is simply re-exported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats as sps

from ..core.statistics import energy_stat_2d, ks2s_2d
from ..core.statistics import mantel_haenzsel as mantel_haenszel  # corrected spelling

__all__ = ["TestResult", "compare", "ks_2d", "energy_2d", "mantel_haenszel"]

_NAN = float("nan")
_PAIRED = {"wilcoxon", "ttest_paired"}


@dataclass(frozen=True)
class TestResult:
    """The outcome of a statistical comparison."""

    statistic: float
    pvalue: float
    n1: int
    n2: int
    method: str
    effect_size: float | None = None

    @property
    def significant(self) -> bool:
        """Convenience: p < 0.05 (and not NaN)."""
        return self.pvalue == self.pvalue and self.pvalue < 0.05


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = a.size, b.size
    pooled = np.sqrt(
        ((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2)
    )
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else _NAN


def _paired_d(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    sd = np.std(d, ddof=1)
    return float(d.mean() / sd) if sd > 0 else _NAN


def compare(
    x1: ArrayLike,
    x2: ArrayLike,
    *,
    method: str = "mannu",
    alternative: str = "two-sided",
) -> TestResult:
    """Compare two samples.

    Args:
        x1, x2: the two samples (NaNs dropped; for paired tests they must be equal length and
            rows with a NaN in either are dropped pairwise).
        method: ``"mannu"`` (Mann-Whitney U, default), ``"ttest"`` (Student's, equal var),
            ``"welch"`` (unequal var), ``"wilcoxon"`` (paired signed-rank), or
            ``"ttest_paired"``.
        alternative: ``"two-sided"`` / ``"less"`` / ``"greater"``.

    Returns:
        TestResult with the test statistic, p-value, group sizes, and an effect size
        (rank-biserial for Mann-Whitney, Cohen's d for the t-tests, paired d for the paired tests).
    """
    if method in _PAIRED:
        a = np.asarray(x1, dtype=float)
        b = np.asarray(x2, dtype=float)
        if a.shape != b.shape:
            raise ValueError(
                f"{method!r} is paired; x1 and x2 must have the same length "
                f"({a.size} vs {b.size})."
            )
        mask = ~(np.isnan(a) | np.isnan(b))
        a, b = a[mask], b[mask]
        n1 = n2 = int(a.size)
    else:
        a = np.asarray(x1, dtype=float)
        b = np.asarray(x2, dtype=float)
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        n1, n2 = int(a.size), int(b.size)

    if n1 == 0 or n2 == 0:
        return TestResult(_NAN, _NAN, n1, n2, method, None)

    if method == "mannu":
        res = sps.mannwhitneyu(a, b, alternative=alternative)
        stat, p = float(res.statistic), float(res.pvalue)
        effect = 1.0 - 2.0 * stat / (n1 * n2)  # rank-biserial correlation
    elif method in ("ttest", "welch"):
        res = sps.ttest_ind(a, b, equal_var=(method == "ttest"), alternative=alternative)
        stat, p = float(res.statistic), float(res.pvalue)
        effect = _cohens_d(a, b)
    elif method == "wilcoxon":
        res = sps.wilcoxon(a, b, alternative=alternative)
        stat, p = float(res.statistic), float(res.pvalue)
        effect = _paired_d(a, b)
    elif method == "ttest_paired":
        res = sps.ttest_rel(a, b, alternative=alternative)
        stat, p = float(res.statistic), float(res.pvalue)
        effect = _paired_d(a, b)
    else:
        raise ValueError(
            f"Unknown method {method!r}; use 'mannu', 'ttest', 'welch', "
            "'wilcoxon', or 'ttest_paired'."
        )
    return TestResult(stat, p, n1, n2, method, float(effect))


def ks_2d(a: ArrayLike, b: ArrayLike, *, n_boot: int | None = None) -> TestResult:
    """Two-dimensional two-sample Kolmogorov-Smirnov test (each sample is shape ``(n, 2)``)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    p, d = ks2s_2d(a, b, nboot=n_boot)
    return TestResult(float(d), float(p), len(a), len(b), "ks_2d")


def energy_2d(
    a: ArrayLike, b: ArrayLike, *, n_boot: int = 1000, method: str = "log"
) -> TestResult:
    """Two-dimensional two-sample energy-distance test (each sample is shape ``(n, 2)``)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    p, energy, _ = energy_stat_2d(a, b, nboot=n_boot, method=method)
    return TestResult(float(energy), float(p), len(a), len(b), "energy_2d")
