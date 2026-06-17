"""The plotting contract: every behavior plot is a simple function that runs three steps.

Phase 4 retires the ~12k LOC of hand-rolled matplotlib/bokeh plotters in favour of thin,
declarative plot **functions** that lean on the rest of the refactored stack. A plot is just::

    def psychometric(df, *, color=None, ax=None, **kwargs) -> PlotResult:
        agg = aggregate(df, ...)        # 1. group & aggregate  -> piepy.stats.aggregate
        stats = fit(...) / compare(...) # 2. statistical test/fit -> piepy.stats / piepy.fitting
        figure = ...                    # 3. plot + significance -> behaviz (you fill this in)
        return PlotResult(data=agg, stats=stats, figure=figure)

No base class, no instances -- each plot is a one-liner to call (``psychometric(df)``) and returns
a :class:`PlotResult` carrying all three outputs. Steps 1 and 2 are wired against
``piepy.stats``/``piepy.fitting``; step 3 (the behaviz drawing) is **left for you to fill in** and
currently leaves ``figure=None``. Colours come from :mod:`piepy.viz.colors`, passed into behaviz's
``color=`` kwargs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

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
