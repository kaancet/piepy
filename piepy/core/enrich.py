"""Declarative ``enrich`` builder.

A paradigm's ``enrich`` hook (see :mod:`piepy.core.registry`) turns a parsed session into the
cohort-ready trial table by joining per-run columns onto :meth:`Session.concatenate_runs`.
``build_enrich`` owns the boilerplate -- the concat, the per-run loop, ``stat_*`` prefixing and
the ``run_no`` join -- so a paradigm supplies only *what* to add: a ``run_stats`` callable,
simple ``meta``/``opts`` key maps, and an optional ``per_run`` callable for derived columns::

    enrich = build_enrich("mytask", run_stats=get_run_stats,
                          opts_map={"task": "controller"}, meta_map={"level": "level"})
    register_paradigm("mytask", trial_handler_cls=..., enrich=enrich)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import polars as pl

__all__ = ["build_enrich"]


def build_enrich(
    paradigm: str,
    *,
    run_stats: Callable | None = None,
    meta_map: Mapping[str, str] | None = None,
    opts_map: Mapping[str, str] | None = None,
    per_run: Callable | None = None,
) -> Callable:
    """Return an ``enrich(session) -> pl.DataFrame`` for ``register_paradigm(enrich=...)``.

    Args:
        paradigm: stamped on the concatenated base table.
        run_stats: ``run_df -> dict``; each item becomes a ``stat_<k>`` column.
        meta_map: ``{column: meta_key}`` pulled from ``run.meta``.
        opts_map: ``{column: opts_key}`` pulled from ``run.meta["opts"]``.
        per_run: ``(run, run_df, session) -> dict`` for derived/computed columns (escape hatch).

    The per-run dicts are stacked and left-joined onto ``concatenate_runs`` on ``run_no``.
    """
    meta_map = dict(meta_map or {})
    opts_map = dict(opts_map or {})

    def enrich(session) -> pl.DataFrame:
        base = session.concatenate_runs(paradigm=paradigm)
        if base.is_empty():
            return base
        rows = []
        for run_no, run in enumerate(session.runs, start=1):
            d = run.data.data if run.data is not None else None
            if d is None or d.is_empty():
                continue
            meta = run.meta or {}
            opts = meta.get("opts") or {}
            row = {"run_no": run_no}
            if run_stats is not None:
                row.update({f"stat_{k}": v for k, v in run_stats(d).items()})
            row.update({col: meta.get(key) for col, key in meta_map.items()})
            row.update({col: opts.get(key) for col, key in opts_map.items()})
            if per_run is not None:
                row.update(per_run(run, d, session))
            rows.append(row)
        if not rows:
            return base
        add = pl.DataFrame(rows).with_columns(pl.col("run_no").cast(pl.UInt32))
        return base.join(add, on="run_no", how="left")

    return enrich
