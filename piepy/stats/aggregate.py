"""Group-and-estimate: the analysis workhorse.

``aggregate`` turns a trial table into a **tidy** per-condition estimate frame -- one row per
``(group x metric)`` with ``value``, ``sem``, ``ci_low``, ``ci_high``, ``n`` -- which is what the
plotting (behaviz) and statistics pipelines consume. It is experiment-agnostic: you name the
grouping columns and the metrics; nothing about wheels/outcomes is hardcoded.

It generalizes the old ``WheelGroupedAggregator``, with three deliberate improvements:

* a **pure function** (no set_outcomes/set_data state);
* **vectorized** confidence intervals -- Wilson for proportions and the Student-t interval for
  means are pure polars/numpy expressions; the median CI uses the analytic order-statistic
  (binomial) interval by default -- so there is no per-group Python bootstrap loop;
* **tidy** output that composes with grouped/faceted plotting and with :func:`group_arrays`
  for feeding the statistical tests.

Metrics are small specs::

    aggregate(df, group=["signed_contrast"], rate="is_hit")                 # shorthand
    aggregate(df, group=["opto"], metrics=[Rate("outcome", rate_of="hit"),  # composable
                                           Median("reaction_time"),
                                           Count()])
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import stats as sps

from .estimators import bootstrap_ci

__all__ = [
    "Rate",
    "Mean",
    "Median",
    "Count",
    "aggregate",
    "subject_average",
    "group_arrays",
]

_TIDY_COLS = ("metric", "value", "sem", "ci_low", "ci_high", "n")


# --------------------------------------------------------------------------- #
# Metric specs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Rate:
    """Proportion of something, with a Wilson confidence interval.

    If ``rate_of`` is None the column is treated as boolean / 0-1; otherwise the rate is the
    fraction of (non-null) rows equal to ``rate_of``.
    """

    column: str
    rate_of: object | None = None
    name: str | None = None

    @property
    def columns(self) -> list[str]:
        return [self.column]

    @property
    def label(self) -> str:
        if self.name:
            return self.name
        tag = self.column if self.rate_of is None else f"{self.column}={self.rate_of}"
        return f"rate[{tag}]"

    def compute(
        self, df: pl.DataFrame, group: list[str], confidence: float
    ) -> pl.DataFrame:
        succ = (
            pl.col(self.column).cast(pl.Float64)
            if self.rate_of is None
            else (pl.col(self.column) == self.rate_of).cast(pl.Float64)
        )

        df = df.filter(pl.col("outcome") != "early")

        agg = df.group_by(group).agg(
            pl.col(self.column).is_not_null().sum().cast(pl.Int64).alias("n"),
            succ.sum().alias("k"),
        )
        z = float(sps.norm.ppf(1 - (1 - confidence) / 2))
        n, k = pl.col("n"), pl.col("k")
        p = k / n
        denom = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        half = (z / denom) * (p * (1 - p) / n + z**2 / (4 * n**2)).sqrt()
        ok = n > 0
        # the Wilson interval always contains p analytically; clamp so it brackets p exactly
        # too (kills ~1e-16 float overshoot at p=0/1 that would otherwise break errorbar plots).
        lo = pl.min_horizontal((center - half).clip(0, 1), p)
        hi = pl.max_horizontal((center + half).clip(0, 1), p)
        return agg.select(
            *group,
            pl.lit(self.label).alias("metric"),
            pl.when(ok).then(p).otherwise(None).cast(pl.Float64).alias("value"),
            # standard error of a proportion: sqrt(p*(1-p)/n)
            pl.when(ok)
            .then((p * (1 - p) / n).sqrt())
            .otherwise(None)
            .cast(pl.Float64)
            .alias("sem"),
            pl.when(ok).then(lo).otherwise(None).cast(pl.Float64).alias("ci_low"),
            pl.when(ok).then(hi).otherwise(None).cast(pl.Float64).alias("ci_high"),
            n.alias("n"),
        )


@dataclass(frozen=True)
class Mean:
    """Mean with a Student-t confidence interval."""

    column: str
    name: str | None = None

    @property
    def columns(self) -> list[str]:
        return [self.column]

    @property
    def label(self) -> str:
        return self.name or f"mean[{self.column}]"

    def compute(
        self, df: pl.DataFrame, group: list[str], confidence: float
    ) -> pl.DataFrame:
        agg = df.group_by(group).agg(
            pl.col(self.column).drop_nulls().mean().alias("value"),
            pl.col(self.column).drop_nulls().std().alias("_sd"),
            pl.col(self.column).drop_nulls().len().cast(pl.Int64).alias("n"),
        )
        n = agg["n"].to_numpy()
        sd = agg["_sd"].to_numpy().astype(float)
        val = agg["value"].to_numpy().astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            sem = np.where(n >= 2, sd / np.sqrt(n), np.nan)  # standard error of the mean
            tcrit = sps.t.ppf(1 - (1 - confidence) / 2, np.maximum(n - 1, 1))
            half = np.where(n >= 2, tcrit * sd / np.sqrt(n), np.nan)
        return agg.select(
            *group,
            pl.lit(self.label).alias("metric"),
            pl.col("value").cast(pl.Float64),
            pl.Series("sem", sem).cast(pl.Float64),
            pl.Series("ci_low", val - half).cast(pl.Float64),
            pl.Series("ci_high", val + half).cast(pl.Float64),
            pl.col("n"),
        )


@dataclass(frozen=True)
class Median:
    """Median with a confidence interval.

    ``ci="order"`` (default) uses the distribution-free order-statistic (binomial) interval:
    exact, deterministic and vectorized across groups. ``ci="bootstrap"`` uses a percentile
    bootstrap per group (seedable, a little slower).
    """

    column: str
    ci: str = "order"
    name: str | None = None
    seed: int | None = None

    @property
    def columns(self) -> list[str]:
        return [self.column]

    @property
    def label(self) -> str:
        return self.name or f"median[{self.column}]"

    def compute(
        self, df: pl.DataFrame, group: list[str], confidence: float
    ) -> pl.DataFrame:
        agg = df.group_by(group).agg(
            pl.col(self.column).drop_nulls().sort().alias("_vals"),
            pl.col(self.column).drop_nulls().median().alias("value"),
            pl.col(self.column).drop_nulls().len().cast(pl.Int64).alias("n"),
        )
        if self.ci == "order":
            return self._order_ci(agg, group, confidence)
        if self.ci == "bootstrap":
            return self._bootstrap_ci(agg, group, confidence)
        raise ValueError(f"Median.ci must be 'order' or 'bootstrap', got {self.ci!r}")

    def _order_ci(self, agg, group, confidence):
        n = agg["n"].to_numpy()
        side = (1 - confidence) / 2
        with np.errstate(invalid="ignore"):
            lo = np.nan_to_num(sps.binom.ppf(side, n, 0.5), nan=0.0).astype(int)
        last = np.maximum(n - 1, 0)
        lo = np.clip(lo, 0, last)
        hi = np.clip(n - 1 - lo, 0, last)
        agg = agg.with_columns(pl.Series("_lo", lo), pl.Series("_hi", hi))
        ok = pl.col("n") > 0
        return agg.select(
            *group,
            pl.lit(self.label).alias("metric"),
            pl.col("value").cast(pl.Float64),
            pl.lit(None, dtype=pl.Float64).alias(
                "sem"
            ),  # no standard error for the median; CI only
            pl.when(ok)
            .then(pl.col("_vals").list.get(pl.col("_lo"), null_on_oob=True))
            .otherwise(None)
            .cast(pl.Float64)
            .alias("ci_low"),
            pl.when(ok)
            .then(pl.col("_vals").list.get(pl.col("_hi"), null_on_oob=True))
            .otherwise(None)
            .cast(pl.Float64)
            .alias("ci_high"),
            pl.col("n"),
        )

    def _bootstrap_ci(self, agg, group, confidence):
        lows, highs = [], []
        for vals in agg["_vals"].to_list():
            if vals:
                est = bootstrap_ci(vals, np.median, confidence=confidence, seed=self.seed)
                lows.append(est.ci_low)
                highs.append(est.ci_high)
            else:
                lows.append(None)
                highs.append(None)
        return agg.select(
            *group,
            pl.lit(self.label).alias("metric"),
            pl.col("value").cast(pl.Float64),
            pl.lit(None, dtype=pl.Float64).alias(
                "sem"
            ),  # no standard error for the median; CI only
            pl.Series("ci_low", lows, dtype=pl.Float64),
            pl.Series("ci_high", highs, dtype=pl.Float64),
            pl.col("n"),
        )


@dataclass(frozen=True)
class Count:
    """Number of (non-null over the group) rows. No confidence interval."""

    name: str = "count"

    @property
    def columns(self) -> list[str]:
        return []

    @property
    def label(self) -> str:
        return self.name

    def compute(
        self, df: pl.DataFrame, group: list[str], confidence: float
    ) -> pl.DataFrame:
        return (
            df.group_by(group)
            .agg(pl.len().cast(pl.Int64).alias("n"))
            .select(
                *group,
                pl.lit(self.label).alias("metric"),
                pl.col("n").cast(pl.Float64).alias("value"),
                pl.lit(None, dtype=pl.Float64).alias("sem"),
                pl.lit(None, dtype=pl.Float64).alias("ci_low"),
                pl.lit(None, dtype=pl.Float64).alias("ci_high"),
                pl.col("n"),
            )
        )


Metric = Rate | Mean | Median | Count


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def _as_list(x) -> list[str]:
    return [x] if isinstance(x, str) else list(x)


def _resolve_metrics(metrics, rate, value, stat, rate_of) -> list[Metric]:
    if metrics is not None:
        return list(metrics)
    out: list[Metric] = []
    if rate is not None:
        if rate_of is None:
            raise ValueError("A rate_of argument is needed to calculate the rate")
        out.append(Rate(rate, rate_of=rate_of))
    if value is not None:
        if stat == "median":
            out.append(Median(value))
        elif stat == "mean":
            out.append(Mean(value))
        else:
            raise ValueError(f"stat must be 'median' or 'mean', got {stat!r}")
    if not out:
        raise ValueError("Nothing to aggregate: pass metrics=[...] or rate=/value=.")
    return out


def aggregate(
    df: pl.DataFrame,
    *,
    group: str | list[str],
    metrics: list[Metric] | None = None,
    rate: str | None = None,
    value: str | None = None,
    stat: str = "median",
    rate_of: object | None = None,
    confidence: float = 0.95,
    points: bool = False,
    sort: bool = True,
) -> pl.DataFrame:
    """Per-group estimate(s) with confidence intervals, as a tidy frame.

    Args:
        df: the trial table.
        group: column(s) to group by (animal, condition, contrast, ...).
        metrics: list of metric specs (:class:`Rate`/:class:`Mean`/:class:`Median`/:class:`Count`).
        rate / value / stat / rate_of: shorthands for a single metric when ``metrics`` is None
            (``rate="is_hit"`` -> ``Rate``; ``value="reaction_time", stat="median"`` -> ``Median``).
        confidence: CI level.
        sort: sort the result by ``group`` then ``metric``.
        points: also attach a ``points`` list column of the raw values behind each row, for the
            value metrics (:class:`Mean`/:class:`Median`); ``Rate``/``Count`` rows leave it null.

    Returns:
        pl.DataFrame with columns ``[*group, "metric", "value", "sem", "ci_low", "ci_high", "n"]``
        (plus ``points`` when ``points=True``), one row per group x metric. ``sem`` is the standard
        error (proportion SE for :class:`Rate`, sd/sqrt(n) for :class:`Mean`; null for
        :class:`Median`/:class:`Count`, which have no standard SEM). Empty groups yield NaN/None
        estimates rather than errors.
    """
    group = _as_list(group)
    resolved = _resolve_metrics(metrics, rate, value, stat, rate_of)

    missing = [c for c in group if c not in df.columns]
    for m in resolved:
        missing += [c for c in m.columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"aggregate: column(s) {sorted(set(missing))} not in the dataframe (have: {df.columns})."
        )

    parts = []
    for m in resolved:
        part = m.compute(df, group, confidence)
        if points and isinstance(
            m, (Mean, Median)
        ):  # raw distribution only for value metrics
            pts = df.group_by(group).agg(pl.col(m.column).drop_nulls().alias("points"))
            part = part.join(pts, on=group)
        parts.append(part)

    out = pl.concat(parts, how="diagonal_relaxed")
    keep = [*group, *_TIDY_COLS, *(["points"] if "points" in out.columns else [])]
    out = out.select(keep)
    if sort:
        out = out.sort([*group, "metric"])
    return out


def subject_average(
    df: pl.DataFrame,
    *,
    x: str | None = None,
    subject: str,
    rate: str | None = None,
    rate_of: object | None = None,
    value: str | None = None,
    stat: str = "median",
    compare: str | None = None,
    confidence: float = 0.95,
    points: bool = False,
) -> pl.DataFrame:
    """Subject-averaged estimate per ``x`` -- the two-stage (hierarchical) estimate.

    Stage 1: summarize each ``subject`` at every ``x`` -- their **rate** (pass ``rate``/``rate_of``)
    or a **continuous summary** (pass ``value`` with ``stat="median"``/``"mean"``, e.g. each
    subject's median reaction time). Stage 2: the **mean of those per-subject values** with a
    Student-t interval *across subjects*. This weights every subject equally (no pseudoreplication
    from animals with more trials) -- the correct cohort estimate, unlike pooling all trials with a
    single :func:`aggregate`. Give exactly one of ``rate`` / ``value`` (same rule as
    :func:`aggregate`).

    Returns the same tidy schema as :func:`aggregate` (``[x(, compare), metric, value, sem, ci_low,
    ci_high, n]`` where ``n`` is the **number of subjects** and ``sem`` is the SEM across subjects),
    so it drops into the same downstream
    plotting/fitting. Note: because ``n`` is subjects (not trials), fit such curves with
    least-squares, not binomial MLE.
    """
    g = ([x] if x else []) + ([compare] if compare else [])
    # stage 1: one estimate per (x[, compare], subject) -- rate or median/mean of `value`.
    per_subject = aggregate(
        df,
        group=g + [subject],
        rate=rate,
        rate_of=rate_of,
        value=value,
        stat=stat,
        sort=False,
    )
    # stage 2: mean of the per-subject estimates across subjects, with a t-CI.
    return aggregate(
        per_subject,
        group=g,
        value="value",
        stat="mean",
        points=points,
        confidence=confidence,
    )


def group_arrays(df: pl.DataFrame, *, group: str | list[str], value: str) -> dict:
    """Per-group raw value arrays, for feeding the statistical tests (``compare``).

    Returns ``{group_key: np.ndarray}`` (the key is a scalar for a single group column, else a
    tuple), with nulls dropped within each group.
    """
    group = _as_list(group)
    missing = [c for c in (*group, value) if c not in df.columns]
    if missing:
        raise ValueError(f"group_arrays: column(s) {missing} not in the dataframe.")

    agg = df.group_by(group).agg(pl.col(value).drop_nulls().alias("_vals"))
    result: dict = {}
    for row in agg.iter_rows(named=True):
        key = tuple(row[g] for g in group)
        if len(group) == 1:
            key = key[0]
        result[key] = np.asarray(row["_vals"], dtype=float)
    return result
