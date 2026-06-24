"""The plotting contract: every behavior plot is a simple function that runs three steps.

Phase 4 retires the hand-rolled matplotlib/bokeh plotters in favour of thin, declarative plot
**functions** that lean on the rest of the refactored stack. A plot is just::

    def psychometric(data, *, color=None, ax=None, **kwargs) -> PlotResult:
        df  = _resolve(data)            # 0. accept Run / Session / Hub / DataFrame
        _need(df, ["signed_contrast"], plot="psychometric")  # 0b. validate inputs
        agg = aggregate(df, ...)        # 1. group & aggregate  -> piepy.stats.aggregate
        stats = fit(...) / compare(...) # 2. statistical test/fit -> piepy.stats / piepy.fitting
        figure = bv.plot_...(...)       # 3. draw -> behaviz (lazy import inside the function)
        return PlotResult(data=agg, stats=stats, figure=figure)

No base class, no registry -- each plot is a function you call (``psychometric(run)``) that returns
a :class:`PlotResult` carrying all three outputs. To add your own plot, drop a function with this
shape in ``piepy/viz/plots/`` (or your own module importing these helpers) and export it; nothing
else is wired.

Two shared seams every plot reuses:

* :func:`_resolve` -- accept a ``Run`` / ``Session`` / ``Hub`` / ``DataFrame`` interchangeably, so
  callers never have to reach for ``.data.data`` / ``concatenate_runs()`` / ``.data`` themselves.
* :func:`_need` -- raise a structured :class:`~piepy.core.errors.SchemaError` (naming the missing
  column + how to get it) instead of letting a bare polars ``KeyError`` surface mid-draw.

behaviz is imported lazily *inside* each plot, so importing ``piepy.viz`` never requires it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from piepy.core.errors import SchemaError

__all__ = ["PlotResult"]


@dataclass
class PlotResult:
    """What every plot function returns: the aggregated data, the test/fit, and the figure.

    Keeping all three together means a caller (notebook, dashboard, report) can re-use the numbers
    behind a figure without recomputing them, and can compose figures however they like. Until a
    plot's behaviz drawing is filled in, :attr:`figure` is ``None`` while :attr:`data`/:attr:`stats`
    are already populated -- so the one-liner is useful today and grows a figure later.
    """

    data: pl.DataFrame  # tidy per-condition estimates from step 1 (aggregate)
    stats: Any = None  # TestResult / FitResult / dict / None from step 2 (test)
    figure: Any = None  # the behaviz figure (or axis) from step 3 (render)


def _resolve(data: Any) -> pl.DataFrame:
    """Return the trial/cohort table behind any of the things a plot accepts.

    Accepts (duck-typed, no imports of the core classes -> no cycle):

    * ``pl.DataFrame``            -> returned as-is.
    * ``Session`` (has ``concatenate_runs``) -> its runs stacked on one session clock.
    * ``Hub``     (``.data`` is a DataFrame) -> the cohort table.
    * ``Run``     (``.data.data`` is a DataFrame) -> that run's trial table.
    """
    if isinstance(data, pl.DataFrame):
        return data
    if hasattr(data, "concatenate_runs"):  # Session
        return data.concatenate_runs()
    inner = getattr(data, "data", None)
    if isinstance(inner, pl.DataFrame):  # Hub.data
        return inner
    if isinstance(getattr(inner, "data", None), pl.DataFrame):  # Run.data (RunData).data
        return inner.data
    raise SchemaError(
        f"Cannot plot a {type(data).__name__}.",
        where=type(data).__name__,
        fix="Pass a polars DataFrame, or a parsed Run / Session / Hub.",
        hint="Run needs analyze_run(); Hub needs initialize() before it has .data.",
    )


def _need(df: pl.DataFrame, cols: list[str], *, plot: str, hint: str | None = None) -> None:
    """Raise a structured error if ``df`` is missing any of ``cols`` (a plot's required inputs)."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SchemaError(
            f"{plot}: missing column(s) {missing}.",
            where=f"columns present: {df.columns}",
            fix=hint or "Make sure the run was augmented (analyze_run) before plotting.",
        )
