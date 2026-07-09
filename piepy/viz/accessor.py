"""``obj.viz.<plot>()``: small accessor that binds a Run/Session/Hub to the plot functions.

every method just forwards to the matching free function in ``plots`` with
the bound object as ``data`` (``_resolve`` turns it into a trial table). It does two convenience
things: supplies **scope-appropriate defaults** (a ``Hub`` subject-averages, a ``Session`` pools, so
you never type ``average_over=``), and applies an optional ``filter=`` that subsets the rows before
plotting.

To add a plot: write the free function in ``plots.py``, then add a one-line delegating method here
(passing ``self._data(filter)`` as its ``data``).
"""

from __future__ import annotations

import polars as pl

from . import plots
from .base import _resolve


class Viz:
    """Plotting bound to one object: ``obj.viz.psychometric(...)``.

    Args:
        obj: the Run / Session / Hub the plots will read from.
        subject: the subject column to average over for cohort-scope objects (a Hub passes
            ``"animalid"``); ``None`` for single-session scope, which pools trials.
    """

    def __init__(self, obj, *, subject: str | None = None) -> None:
        self._obj = obj
        self._subject = subject

    def _data(self, filterer: dict | None):
        """The data handed to a plot, optionally row-filtered.

        ``filter`` maps ``column -> value`` (or a list of values); rows are kept where the column is
        in the given value(s), and multiple keys are ANDed. Without a filter the bound object is
        passed straight through, so the plot resolves it itself (and a Session still pools lazily).
        """
        if not filterer:
            return self._obj
        df = _resolve(self._obj)
        for col, val in filterer.items():
            allowed = list(val) if isinstance(val, (list, tuple, set)) else [val]
            df = df.filter(pl.col(col).is_in(allowed))
        return df

    def psychometric(self, *, filterer: dict | None = None, **kwargs):
        """See :func:`piepy.viz.plots.psychometric`. Defaults ``average_over`` to this scope's
        subject column (Hub subject-averages, Session pools); ``filter`` subsets rows first."""
        kwargs.setdefault("average_over", self._subject)
        return plots.psychometric(self._data(filterer), **kwargs)

    def reaction_time_cloud(self, *, filterer: dict | None = None, **kwargs):
        """See :func:`piepy.viz.plots.reaction_time_cloud`. Defaults ``average_over`` to this scope's
        subject column (so a Hub subject-averages, a Session pools)."""
        kwargs.setdefault("average_over", self._subject)
        return plots.reaction_time_cloud(self._data(filterer), **kwargs)

    def reaction_time_dist(self, *, filterer: dict | None = None, **kwargs):
        """See :func:`piepy.viz.plots.reaction_time_dist`. Defaults ``average_over`` to this scope's
        subject column (so a Hub subject-averages, a Session pools)."""
        kwargs.setdefault("average_over", self._subject)
        return plots.reaction_time_dist(self._data(filterer), **kwargs)
