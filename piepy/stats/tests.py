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
from itertools import combinations

import numpy as np
import polars as pl
from typing import Literal
from numpy.typing import ArrayLike
from scipy import stats as sps
from scipy.spatial.distance import pdist, cdist

__all__ = ["TestResult", "compare", "compare_by_x", "ks_2d", "energy_2d", "mantel_haenzsel"]

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
    pooled = np.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2))
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
            ``"welch"`` (unequal var), ``"wilcoxon"`` (paired signed-rank), ``"ttest_paired"``, or
            ``"fisher"`` (2x2 Fisher exact for two **binary** 0/1 samples -- two-proportion test).
        alternative: ``"two-sided"`` / ``"less"`` / ``"greater"``.

    Returns:
        TestResult with the test statistic, p-value, group sizes, and an effect size (rank-biserial
        for Mann-Whitney, Cohen's d for the t-tests, paired d for the paired tests, odds ratio for
        ``"fisher"`` -- whose ``statistic`` is also the odds ratio).
    """
    if method in _PAIRED:
        a = np.asarray(x1, dtype=float)
        b = np.asarray(x2, dtype=float)
        if a.shape != b.shape:
            raise ValueError(f"{method!r} is paired; x1 and x2 must have the same length ({a.size} vs {b.size}).")
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
    elif method == "fisher":
        if not (np.isin(a, (0.0, 1.0)).all() and np.isin(b, (0.0, 1.0)).all()):
            raise ValueError("method='fisher' needs binary (0/1) samples.")
        k1, k2 = int(a.sum()), int(b.sum())
        odds, p = sps.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]], alternative=alternative)
        stat = effect = float(odds)  # odds ratio is both the statistic and the effect size
    else:
        raise ValueError(
            f"Unknown method {method!r}; use 'mannu', 'ttest', 'welch', 'wilcoxon', 'fisher', or 'ttest_paired'."
        )
    return TestResult(stat, p, n1, n2, method, float(effect))


def compare_groups(
    df: pl.DataFrame,
    *,
    comparing: str,
    value: str,
    success: object | None = None,
    subject: str | None = None,
    method: str = "auto",
    alternative: str = "two-sided",
    correction: str | None = None,
) -> pl.DataFrame:
    """Pairwise :func:`compare` of the ``comparing`` groups over the whole frame -- a tidy frame.

    Tests **every pair** of ``comparing`` levels (use this when the comparison does not need to be
    done per stimulus level; for that, see :func:`compare_by_x`). Pass ``success`` to test a
    **proportion** (``value`` binarized to ``value == success``); omit it for the raw continuous
    ``value``. Pass ``subject`` for a **paired** test across subjects: each subject is summarized
    within each group (its rate when ``success`` is set, else its mean ``value``), subjects are
    matched across the pair, and a paired test runs on those matched values (``method="auto"`` then
    defaults to ``"wilcoxon"``).

    Args:
        comparing: the grouping column to compare (e.g. ``opto``); every pair of its levels is tested.
        value: the per-trial column being compared (``outcome`` for a rate, ``reaction_time`` ...).
        success: if given, ``value`` is binarized to ``value == success`` (a proportion).
        subject: a subject column (e.g. ``animalid``) -> matched per-subject paired test (no
            pseudoreplication). Required for ``"wilcoxon"``/``"ttest_paired"``.
        method: ``"auto"`` picks ``"wilcoxon"`` when ``subject`` is set, else ``"fisher"`` (with
            ``success``) or ``"mannu"``; or name any :func:`compare` method.
        alternative: passed through to :func:`compare`.
        correction: multiple-comparison adjustment across the returned pairs
            (``"bonferroni"`` / ``"holm"`` / ``"bh"``); ``None`` -> none.

    Returns:
        pl.DataFrame, one row per group pair: ``["group_a", "group_b", "statistic", "pvalue",
        "pvalue_corrected", "effect_size", "n_a", "n_b", "significant", "method", "correction"]``.
        ``significant`` uses ``pvalue_corrected`` (== ``pvalue`` when ``correction`` is None).
    """
    for col in (comparing, value, *((subject,) if subject else ())):
        if col not in df.columns:
            raise ValueError(f"compare_groups: column {col!r} not in the dataframe.")
    m = _resolve_method(method, subject, success)
    rows = _pairwise(df, comparing, value, success, subject, m, alternative)
    if not rows:
        return pl.DataFrame()
    return _finalize(pl.DataFrame(rows), correction).sort(["group_a", "group_b"])


def compare_by_x(
    df: pl.DataFrame,
    *,
    x: str | None = None,
    comparing: str,
    value: str,
    success: object | None = None,
    subject: str | None = None,
    method: str = "auto",
    alternative: str = "two-sided",
    correction: str | None = None,
) -> pl.DataFrame:
    """:func:`compare_groups` repeated once per level of ``x`` (e.g. per contrast).

    Same arguments as :func:`compare_groups` plus ``x`` (the level column); the result gains a
    leading ``x`` column. ``correction`` is applied **across all** tests from all ``x`` levels (not
    per level), so the per-level blocks are computed uncorrected and adjusted together at the end.

    Returns:
        pl.DataFrame, one row per ``(x level, group pair)`` with the same columns as
        :func:`compare_groups` plus ``x``.
    """
    if x is None:
        return compare_groups(
            df,
            comparing=comparing,
            value=value,
            success=success,
            subject=subject,
            method=method,
            alternative=alternative,
            correction=correction,
        )

    if x not in df.columns:
        raise ValueError(f"compare_by_x: column {x!r} not in the dataframe.")
    parts = []
    for xkey, xsub in df.group_by(x, maintain_order=True):
        xval = xkey[0] if isinstance(xkey, tuple) else xkey
        block = compare_groups(
            xsub,
            comparing=comparing,
            value=value,
            success=success,
            subject=subject,
            method=method,
            alternative=alternative,
            correction=None,  # correct once, globally, after stacking all x-levels
        )
        if not block.is_empty():
            parts.append(block.with_columns(pl.lit(xval).alias(x)))
    if not parts:
        return pl.DataFrame()
    out = _finalize(pl.concat(parts), correction)
    return out.select([x, *[c for c in out.columns if c != x]]).sort([x, "group_a", "group_b"])


def _resolve_method(method: str, subject: str | None, success: object | None) -> str:
    """Resolve ``method="auto"`` and reject paired methods that lack a ``subject`` to match on."""
    m = method
    if m == "auto":
        m = "wilcoxon" if subject else ("fisher" if success is not None else "mannu")
    if m in _PAIRED and subject is None:
        raise ValueError(f"method={m!r} is paired; pass subject=... to match samples across groups.")
    return m


def _pairwise(df, comparing, value, success, subject, method, alternative) -> list[dict]:
    """Run :func:`compare` on every pair of ``comparing`` levels; one row dict per pair."""
    groups = sorted(df[comparing].unique().drop_nulls().to_list())
    rows = []
    for a_lvl, b_lvl in combinations(groups, 2):
        if subject is not None:
            a_arr, b_arr = _paired_subject_arrays(df, comparing, subject, value, success, a_lvl, b_lvl)
        else:
            a_arr = _level_array(df, comparing, value, success, a_lvl)
            b_arr = _level_array(df, comparing, value, success, b_lvl)
        res = compare(a_arr, b_arr, method=method, alternative=alternative)
        rows.append(
            {
                "group_a": a_lvl,
                "group_b": b_lvl,
                "statistic": res.statistic,
                "pvalue": res.pvalue,
                "effect_size": res.effect_size,
                "n_a": res.n1,
                "n_b": res.n2,
                "method": method,
            }
        )
    return rows


def _finalize(out: pl.DataFrame, correction: str | None) -> pl.DataFrame:
    """Add ``pvalue_corrected`` / ``significant`` / ``correction`` columns (significant uses the
    corrected p; == raw p when ``correction`` is None)."""
    pvals = out["pvalue"].to_numpy()
    corrected = _correct(pvals, correction) if correction else pvals
    return out.with_columns(
        pl.Series("pvalue_corrected", corrected),
        pl.Series("significant", corrected < 0.05),
        pl.lit(correction).alias("correction"),
    )


def _correct(pvals: np.ndarray, method: str) -> np.ndarray:
    """Multiple-comparison adjust a vector of p-values (NaNs pass through untouched)."""
    p = np.asarray(pvals, dtype=float)
    out = p.copy()
    finite = np.isfinite(p)
    q = p[finite]
    m = q.size
    if m == 0:
        return out

    if method in ("bonferroni", "bonf"):
        adj = np.minimum(q * m, 1.0)
    elif method == "holm":
        order = np.argsort(q)
        adj = np.empty(m)
        adj[order] = np.minimum(np.maximum.accumulate((m - np.arange(m)) * q[order]), 1.0)
    elif method in ("bh", "fdr", "fdr_bh"):
        order = np.argsort(q)
        ranked = q[order] * m / np.arange(1, m + 1)
        adj = np.empty(m)
        adj[order] = np.minimum(np.minimum.accumulate(ranked[::-1])[::-1], 1.0)
    else:
        raise ValueError(f"unknown correction {method!r}; use 'bonferroni', 'holm', or 'bh'.")

    out[finite] = adj
    return out


def _level_array(sub, comparing, value, success, lvl) -> np.ndarray:
    """Per-trial values for one ``comparing`` level (binarized to 0/1 when ``success`` is set)."""
    col = sub.filter(pl.col(comparing) == lvl)[value].drop_nulls()
    if success is not None:
        return (col == success).cast(pl.Int8).to_numpy().astype(float)
    return col.to_numpy().astype(float)


def _paired_subject_arrays(sub, comparing, subject, value, success, a_lvl, b_lvl):
    """Matched per-subject summaries for two groups (rate if ``success`` else mean), inner-joined."""
    stat = (pl.col(value) == success).cast(pl.Float64).mean() if success is not None else pl.col(value).mean()

    def per_subject(lvl):
        return sub.filter(pl.col(comparing) == lvl).group_by(subject).agg(stat.alias("stat"))

    merged = per_subject(a_lvl).join(per_subject(b_lvl), on=subject, suffix="_b")
    return merged["stat"].to_numpy().astype(float), merged["stat_b"].to_numpy().astype(float)


def ks_2d(a: ArrayLike, b: ArrayLike, *, n_boot: int | None = None) -> TestResult:
    """Two-dimensional two-sample Kolmogorov-Smirnov test (each sample is shape ``(n, 2)``)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    p, d = ks2s_2d(a, b, nboot=n_boot)
    return TestResult(float(d), float(p), len(a), len(b), "ks_2d")


def energy_2d(a: ArrayLike, b: ArrayLike, *, n_boot: int = 1000, method: str = "log") -> TestResult:
    """Two-dimensional two-sample energy-distance test (each sample is shape ``(n, 2)``)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    p, energy, _ = energy_stat_2d(a, b, nboot=n_boot, method=method)
    return TestResult(float(energy), float(p), len(a), len(b), "energy_2d")


# =============================
# =============================
def ks2s_2d(data1: ArrayLike, data2: ArrayLike, nboot: int | None = None) -> tuple[float, float]:
    """Two-dimensional Kolmogorov-Smirnov test on two samples.
    Adapted from: https://github.com/syrte/ndtest
    Notes
    -----
    This is the two-sided K-S test. Small p-values means that the two samples are significantly different.
    Note that the p-value is only an approximation as the analytic distribution is unkonwn. The approximation
    is accurate enough when N > ~20 and p-value < ~0.20 or so. When p-value > 0.20, the value may not be accurate,
    but it certainly implies that the two samples are not significantly different.

    Args:
        data1 (ArrayLike): shape (n1,2) Data of sample one
        data2 (ArrayLike): shape (n2,2) Data of sample two (n1 and n2 can be different)
        nboot (int | None, optional): Number of bootstrap resample to estimate the p-value. A large number is expected.
        If None, an approximate analytic estimate will be used. Defaults to None.

    Returns:
        tuple[float,float]: Two-tailed p-value, KS statistic
    """

    def quadct(x, y, xx, yy):
        n = len(xx)
        ix1, ix2 = xx <= x, yy <= y
        a = np.sum(ix1 & ix2) / n
        b = np.sum(ix1 & ~ix2) / n
        c = np.sum(~ix1 & ix2) / n
        d = 1 - a - b - c
        return a, b, c, d

    def maxdist(x1, y1, x2, y2):
        n1 = len(x1)
        D1 = np.empty((n1, 4))
        for i in range(n1):
            a1, b1, c1, d1 = quadct(x1[i], y1[i], x1, y1)
            a2, b2, c2, d2 = quadct(x1[i], y1[i], x2, y2)
            D1[i] = [a1 - a2, b1 - b2, c1 - c2, d1 - d2]

        # re-assign the point to maximize difference,
        # the discrepancy is significant for N < ~50
        D1[:, 0] -= 1 / n1

        dmin, dmax = -D1.min(), D1.max() + 1 / n1
        return max(dmin, dmax)

    def avgmaxdist(x1, y1, x2, y2):
        D1 = maxdist(x1, y1, x2, y2)
        D2 = maxdist(x2, y2, x1, y1)
        return (D1 + D2) / 2

    n1, n2 = len(data1), len(data2)

    x1, x2 = data1[:, 0], data2[:, 0]
    y1, y2 = data1[:, 1], data2[:, 1]
    D = avgmaxdist(x1, y1, x2, y2)

    if nboot is None:
        sqen = np.sqrt(n1 * n2 / (n1 + n2))
        r1 = sps.pearsonr(x1, y1)[0]
        r2 = sps.pearsonr(x2, y2)[0]
        r = np.sqrt(1 - 0.5 * (r1**2 + r2**2))
        d = D * sqen / (1 + r * (0.25 - 0.75 / sqen))
        p = sps.kstwobign.sf(d)
    else:
        n = n1 + n2
        x = np.concatenate([x1, x2])
        y = np.concatenate([y1, y2])
        d = np.empty(nboot, "f")
        for i in range(nboot):
            idx = np.random.choice(n, n, replace=True)
            ix1, ix2 = idx[:n1], idx[n1:]
            # ix1 = random.choice(n, n1, replace=True)
            # ix2 = random.choice(n, n2, replace=True)
            d[i] = avgmaxdist(x[ix1], y[ix1], x[ix2], y[ix2])
        p = np.sum(d > D).astype("f") / nboot

    return p, D


def energy_stat_2d(
    data1: ArrayLike,
    data2: ArrayLike,
    nboot: int = 1000,
    replace: bool = False,
    method: Literal["log", "gaussian", "linear"] = "log",
) -> tuple[float, float, float]:
    """Energy distance statistics test.
    Adapted from: https://github.com/syrte/ndtest

    Args:
        data1 (ArrayLike): shape (n1,2) Data of sample one
        data2 (ArrayLike): shape (n2,2) Data of sample two (n1 and n2 can be different)
        nboot (int, optional): Number of bootstrap resample to estimate the p-value. A large number is expected. Defaults to 1000.
        replace (bool, optional): Sample with replacement. Defaults to False.
        method (Literal["log","gaussian","linear"], optional): . Defaults to 'log'.

    Returns:
        tuple[float, float, float]: p-value, energy,
    """

    def energy(x, y, method="log"):
        dx, dy, dxy = pdist(x), pdist(y), cdist(x, y)
        n, m = len(x), len(y)
        if method == "log":
            dx, dy, dxy = np.log(dx), np.log(dy), np.log(dxy)
        elif method == "gaussian":
            raise NotImplementedError
        elif method == "linear":
            pass
        else:
            raise ValueError
        z = dxy.sum() / (n * m) - dx.sum() / n**2 - dy.sum() / m**2
        # z = ((n*m)/(n+m)) * z # ref. SR
        return z

    n, N = len(data1), len(data1) + len(data2)
    stack = np.vstack((data1, data2))
    stack = (stack - stack.mean(0)) / stack.std(0)
    if replace:
        rand = lambda x: np.random.randint(x, size=x)  # noqa: E731
    else:
        rand = np.random.permutation

    en = energy(stack[:n], stack[n:], method)
    en_boot = np.zeros(nboot, "f")
    for i in range(nboot):
        idx = rand(N)
        en_boot[i] = energy(stack[idx[:n]], stack[idx[n:]], method)

    # if fitting:
    #     param = genextreme.fit(en_boot)
    #     p = genextreme.sf(en, *param)
    #     return p, en, param
    # else:
    p = (en_boot >= en).sum() / nboot
    return p, en, en_boot


def mantel_haenzsel(data, stats="Q"):
    """Garcia-Perez, M.A., & Nuñez-Anton, V. (2018). Nonparametric tests for
    equality of psychometric functions. Behavior Research Methods,
    https://doi.org/10.3758/s13428-017-0989-0"""
    # error handling
    if not isinstance(data, dict):
        raise TypeError("data argument needs a dict, got a {0}".format(type(data)))

    num_tables = len(data)
    if isinstance(stats, str):
        if stats != "Q" and stats != "G":
            raise ValueError("stats string either takes Q or G not {0}".format(stats))
    else:
        if num_tables < 2:
            raise ValueError("The split Mantel-Haenszel test cannot be used if there is only one table")
        elif len(stats) != 2:
            raise ValueError("stats vector should have 2 elements, got {0}".format(len(stats)))
        elif any(elem <= 0 for elem in stats):
            raise ValueError("Elements in stats vector should be positive, got {0}".format(stats))
        elif sum(stats) != num_tables:
            raise ValueError("stats vector does not add up to ({0}) length of data({1})".format(stats, num_tables))

    table_ids = list(data.keys())
    cr = np.array([np.nan] * num_tables)
    cc = np.array([np.nan] * num_tables)

    for it, key in enumerate(data.keys()):
        # fill in row and column sizes for each data table
        cr[it] = data[key].shape[0]
        cc[it] = data[key].shape[1]
        # check dimensions of tables
        if data[key].ndim > 2 or cr[it] < 2 or cc[it] < 2:
            raise ValueError("Invalid table at {0}, not a two-way table".format(key))
        # check if any table element is 0 or fractional
        elif (data[key] < 0).any(axis=None) or not (data[key] % 1 == 0).all(axis=None):
            raise ValueError("Invalid table at {0}, contains negative or fractional values".format(key))

    if np.sum(np.add(cr, -cr[0])) != 0:
        raise ValueError("Invalid Data, tables do not have the same number of rows")

    if np.sum(np.add(cc, -cc[0])) != 0:
        raise ValueError("Invalid Data, tables do not have the same number of columns")

    cc = cc.astype(int)
    cr = cr.astype(int)

    # compute selected statistic
    if isinstance(stats, str):
        kase = 1 if stats == "G" else 2
    else:
        kase = 3

    # generalized Berry-Mielke test
    if kase == 1:
        pass
        # N_k = np.array([np.nan] * num_tables)
        # Tmean = np.array([np.nan] * num_tables)
        # Tvar  = np.array([np.nan] * num_tables)
        # Tskew = np.array([np.nan] * num_tables)
        # Tstat = np.array([np.nan] * num_tables)
        # Zstat = np.array([np.nan] * num_tables)
        # Gstat = np.array([np.nan] * num_tables)
        # c_k   = np.array([np.nan] * num_tables)
        # Ncols = np.ones(num_tables,dtype=bool)
        # Nrows = np.ones(num_tables,dtype=bool)
        # Ntbls = np.ones(num_tables,dtype=bool)
        # sngl_row = np.zeros(num_tables,dtype=bool)
        # sngl_col = np.zeros(num_tables,dtype=bool)
        # sngl_perm = np.zeros(num_tables,dtype=bool)
        # data_used = {}

        # warn1 = 'All tables were used'
        # warn2 = 'All usable tables were used with their numbers of rows'
        # warn3 = 'All usable tables were used with their numbers of columns'
        # warn4 = 'For all usable tables, gamma_T >= 0.5'

        # for it,key in enumerate(data.keys()):
        #     in_data = data[it]
        #     row = np.sum(in_data,1).to_numpy()
        #     col = np.sum(in_data,0).to_numpy()

        #     N = np.sum(row);
        #     N_k[it] = N
        #     # check for actual number of rows and columns
        #     table = in_data.loc[row>0,col>0]

        #     row = row[row>0]
        #     col = col[col>0]
        #     I = len(row)
        #     J = len(col)

        #     if I==1:
        #         sngl_row[it] = True
        #     if J==1:
        #         sngl_co[it] = True

        #     if table.shape[0] > 1 and table.shape[0] != in_data.shape[0]:
        #         Nrows[it] = False

        #     if table.shape[1] > 1 and table.shape[1] != in_data.shape[1]:
        #         Ncols[it] = False

        #     # check for Ix2 table with equal row marginal frequencies and a column marginal frequency of 1
        #     if J == 1 or (J == 2 and all(row == row[0]) and any(col == 1)):
        #         Ntbls[it] = False;
        #         sngl_perm[it] = True

        #     # check for 2xJ table with equal column marginal frequencies and a row marginal frequency of 1
        #     if I == 1 or (I == 2 and all(col == col[0]) and any(row == 1)):
        #         Ntbls[it] = False
        #         sngl_perm[it] = True

        #     if Ntbls[it]:
        #         data_used[key] = table
        #         # compute moments
        #         N_ = np.empty((6,2))
        #         N_[:] = np.nan
        #         R_m = np.empty((I,6))
        #         R_m[:] = np.nan
        #         C_m = np.empty((J,6))
        #         C_m[:] = np.nan
        #         R = np.empty((6,6))
        #         R[:] = np.nan
        #         C = np.empty((6,6))
        #         C[:] = np.nan

        #         for m in range(4):
        #             N_[m,0] = np.prod(N-3:N-4+m)
        #         for m in range(6):
        #             N_[m,1] = np.prod(N-5:N-6+m)

        #         R_m[:,0] = row
        #         for m in range(1,6):
        #             R_m[:,m] = R_m[:,m-1] * (row - m +1)

        #         for m in range(4):
        #             R_m[m,1] = np.sum(R_m[:,m] / (row ** 2))

        #         R[2,2] = I * (I-1)
        #         R[3,2] = (I-1) * (N-I)
        #         R[4,2] = (N-I)**2 + 2 * N -I np.sum(row**2)

        #         for m in range(6):
        #             R[m,3] = np.sum(R_m[:,m] / (row**3))

        #         for m in range(2,6):
        #             R[m,4] = np.sum(R_m[:,m-2] * (N-row-I+1)/(row**2))

        #         for m

        # for m=3:6, R(m,4) = sum(R_m(:,m-2).*(N-row-I+1)./(row.^2)); end
        # for m=2:5, R(m,5) = (I-1)*R(m-1,1); end
        # R(3,6) = I*(I-1)*(I-2);
        # R(4,6) = (I-1)*(I-2)*(N-I);
        # R(5,6) = (I-2)*R(4,2);

    # generalized Mantel-Haenzsel test
    elif kase == 2:
        description = "Generalized Mantel-Haenszel test in {0} populations with {1} response categories".format(
            cr[0], cc[0]
        )

        R = {k: None for k in table_ids}
        C = {k: None for k in table_ids}
        N = np.array([np.nan] * num_tables)
        O = data.copy()
        data_used = data.copy()
        E = {k: None for k in table_ids}
        V = {k: None for k in table_ids}
        row = np.zeros((cr[0], 1), dtype=int)
        col = np.zeros((1, cc[0]), dtype=int)

        # check for empty rows or columns accross tables
        for it, key in enumerate(data.keys()):
            R[key] = np.sum(data[key], axis=1).reshape(cr[0], -1)
            C[key] = np.sum(data[key], axis=0).reshape(-1, cc[0])
            N[it] = np.sum(C[key])
            E[key] = np.matmul(R[key], C[key]) / N[it]
            row += R[key]
            col += C[key]

        I = np.sum(row > 0)
        J = np.sum(col > 0)

        warn_msg1 = ""
        warn_msg2 = ""
        qgmh_stat = None
        df = None
        p_value = None

        if I == 1 or J == 1:
            warn_msg1 = "Data cannot be used: all but one column or one row are empty"
        else:
            if I < cr[0] or J < cc[0]:
                warn_msg1 = "Tables used with {0} rows and {1} columns".format(I, J)
            else:
                warn_msg1 = "Tables used with all their rows and columns"

            if num_tables == 1:
                warn_msg2 = "Q_GMH is only an adjusted Pearsons statistic when K = 1"

            sum1 = np.zeros((1, (I - 1) * (J - 1)))
            sum2 = np.zeros(((I - 1) * (J - 1), (I - 1) * (J - 1)))

            for it, key in enumerate(table_ids):
                R[key] = R[key][row > 0]
                R[key] = R[key][0 : I - 1]
                R[key] = R[key].reshape(len(R[key]), -1)
                C[key] = C[key][col > 0]
                C[key] = C[key][0 : J - 1]
                C[key] = C[key].reshape(-1, len(C[key]))

                E[key] = E[key][row[:, 0] > 0, :]  # row filter
                E[key] = E[key][:, col[0, :] > 0]  # column
                E[key] = np.reshape(E[key][0 : I - 1, 0 : J - 1], (1, (I - 1) * (J - 1)), order="F")

                O[key] = O[key][row[:, 0] > 0, :]  # row filter
                O[key] = O[key][:, col[0, :] > 0]  # column
                data_used[key] = O[key]

                O[key] = np.reshape(O[key][0 : I - 1, 0 : J - 1], (1, (I - 1) * (J - 1)), order="F")

                sum1 += np.subtract(O[key], E[key])
                if N[it] > 1:
                    V[key] = np.kron(
                        N[it] * np.diag(C[key][0]) - np.matmul(np.transpose(C[key]), C[key]),
                        N[it] * np.diag(R[key]) - np.matmul(R[key], np.transpose(R[key])),
                    ) / (N[it] * N[it] * (N[it] - 1))
                else:
                    V[key] = np.zeros(((I - 1) * (J - 1), (I - 1) * (J - 1)))

                sum2 = np.add(sum2, V[key])

            if np.linalg.matrix_rank(sum2) == sum2.shape[0]:
                temp = np.linalg.lstsq(sum2.T, sum1.T)[0]
                qgmh_stat = np.matmul(temp.T, sum1.T)
                df = (I - 1) * (J - 1)
                p_value = sps.chi2.sf(qgmh_stat, df)

        output = {
            "NumTables": num_tables,
            "SampleSizes": N,
            "Warning1": warn_msg1,
            "warning2": warn_msg2,
            "Q_GMH": qgmh_stat,
            "deg_free": df,
            "p_value": p_value,
        }

    # split Mantel-Haenszel test
    elif kase == 3:
        description = "Split Mantel-Haenszel test in {0} populations with {1} response categories".format(cr[0], cc[0])

        R = {k: None for k in table_ids}
        C = {k: None for k in table_ids}
        N = np.array([np.nan] * num_tables)
        O = data.copy()
        data_used = data.copy()
        E = {k: None for k in table_ids}
        V = {k: None for k in table_ids}
        qgmh_stat = np.array([np.nan] * 2)
        df = np.array([np.nan] * 2)
        warn_msg1 = {0: None, 1: None}
        cual = {0: "first split", 1: "second split"}
        first = [0, stats[0]]
        last = [stats[0], num_tables]
        for split in range(2):
            row = np.zeros((cr[0], 1), dtype=int)
            col = np.zeros((1, cc[0]), dtype=int)
            for it in range(first[split], last[split]):
                key = list(data.keys())[it]
                R[key] = np.sum(data[key], axis=1).reshape(cr[0], -1)
                C[key] = np.sum(data[key], axis=0).reshape(-1, cc[0])
                N[it] = np.sum(C[key])
                E[key] = np.matmul(R[key], C[key]) / N[it]
                row += R[key]
                col += C[key]

            I = np.sum(row > 0)
            J = np.sum(col > 0)

            if I == 1 or J == 1:
                warn_msg1[split] = "Tables in {0} cannot be used: all but one column or one row are empty".format(
                    cual[split]
                )
            else:
                if I < cr[0] or J < cc[0]:
                    warn_msg1[split] = "Tables in {0} used with {1} rows and {2} columns".format(cual[split], I, J)
                else:
                    warn_msg1[split] = "Tables in {0} used with all their rows and columns".format(cual[split])
                warn_msg2 = ""

                if num_tables == 1:
                    warn_msg2 = "S-Q_GMH is only the sum of adjusted Pearsons statistics when K = 2"

                sum1 = np.zeros((1, (I - 1) * (J - 1)))
                sum2 = np.zeros(((I - 1) * (J - 1), (I - 1) * (J - 1)))

                for it in range(first[split], last[split]):
                    key = list(data.keys())[it]
                    R[key] = R[key][row > 0]
                    R[key] = R[key][0 : I - 1]
                    R[key] = R[key].reshape(len(R[key]), -1)
                    C[key] = C[key][col > 0]
                    C[key] = C[key][0 : J - 1]
                    C[key] = C[key].reshape(-1, len(C[key]))

                    E[key] = E[key][row[:, 0] > 0, :]  # row filter
                    E[key] = E[key][:, col[0, :] > 0]  # column
                    E[key] = np.reshape(E[key][0 : I - 1, 0 : J - 1], (1, (I - 1) * (J - 1)), order="F")

                    O[key] = O[key][row[:, 0] > 0, :]  # row filter
                    O[key] = O[key][:, col[0, :] > 0]  # column
                    data_used[key] = O[key]

                    O[key] = np.reshape(O[key][0 : I - 1, 0 : J - 1], (1, (I - 1) * (J - 1)), order="F")

                    sum1 += np.subtract(O[key], E[key])
                    if N[it] > 1:
                        V[key] = np.kron(
                            N[it] * np.diag(C[key][0]) - np.matmul(np.transpose(C[key]), C[key]),
                            N[it] * np.diag(R[key]) - np.matmul(R[key], np.transpose(R[key])),
                        ) / (N[it] * N[it] * (N[it] - 1))
                    else:
                        V[key] = np.zeros(((I - 1) * (J - 1), (I - 1) * (J - 1)))

                    sum2 = np.add(sum2, V[key])

                if np.linalg.matrix_rank(sum2) == sum2.shape[0]:
                    temp = np.linalg.lstsq(sum2.T, sum1.T)[0]
                    qgmh_stat[split] = np.matmul(temp.T, sum1.T)
                    df[split] = (I - 1) * (J - 1)

        s_qgmh_stat = np.nansum(qgmh_stat)
        s_df = np.nansum(df)
        p_value = None
        if not np.isnan(s_qgmh_stat).all():
            p_value = sps.chi2.sf(s_qgmh_stat, s_df)

        output = {
            "description": description,
            "NumTables": num_tables,
            "In_Split1": [*range(0, stats[0])],
            "In_Split2": [*range(stats[0], num_tables)],
            "SampleSizes": N,
            "Warning1": warn_msg1,
            "warning2": warn_msg2,
            "Components": qgmh_stat,
            "S-Q_GMH": s_qgmh_stat,
            "deg_free": s_df,
            "p_value": p_value,
        }

    return output, data_used
