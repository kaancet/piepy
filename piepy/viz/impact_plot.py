"""Impact plot: how a metric shifts across a manipulation axis, with paired subjects + tests.

Same resolve -> aggregate -> test -> draw -> PlotResult shape as :mod:`piepy.viz.plots`, so it is
task-agnostic: the manipulation axis (``x``), the metric (a rate or a continuous ``value``) and the
subject column are all arguments. The classic use is opto vs non-opto hit-rate/reaction-time change,
but nothing about wheels/opto is hardcoded -- any trial table with a discrete manipulation column
plots.

One panel per call (pass ``ax=`` to place it in a facet grid, exactly like the other plots). It
draws, per level of ``x``:

* one connecting line per subject (paired across the manipulation levels),
* the across-subject mean (or median) with an SEM bar,
* a pairwise significance bracket between every pair of ``x`` levels.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from piepy.stats import aggregate, compare_groups, subject_average
from .base import PlotResult, _need, _resolve


def impact(
    data,
    *,
    x: str = "opto_pattern",
    outcome: str = "outcome",
    success: object = "hit",
    value: str | None = None,
    stat: str = "mean",
    average_over: str = "animalid",
    compare: str | None = None,
    test: bool = True,
    ax=None,
    **style,
) -> PlotResult:
    """A metric across a discrete manipulation axis: paired per-subject lines + summary + tests.

    Args:
        data: a Run / Session / Hub / trial-table DataFrame (``_resolve`` handles all four).
        x: the manipulation column on the x-axis (discrete, e.g. ``opto_pattern``: -1/0/1).
        outcome / success: when ``value`` is None, the metric is the **rate** of ``outcome ==
            success`` (pass ``success=None`` if ``outcome`` is already 0/1).
        value: a continuous column (e.g. ``reaction_time``) to summarize instead of a rate; its
            ``stat`` ("mean"/"median") becomes the metric.
        subject: the paired unit -> one connecting line per subject, and the across-subject summary
            weights every subject equally (no pseudoreplication).
        compare: an extra column to split on -> one colored set per level (behaviz ``hue=``).
        test: draw a pairwise significance bracket between every pair of ``x`` levels (paired across
            ``subject``); set False to skip. The tidy test frame is returned in ``stats``.
        ax / **style: forwarded to behaviz (``color=`` / ``palette=`` ride in ``**style``).

    Returns:
        PlotResult(data=per-(x, subject) estimate frame, stats=pairwise test frame|None,
        figure=(fig, ax)).
    """
    import behaviz as bv

    spec = style.pop("spec", None)
    if spec is None:
        spec = bv.load_preset(style.pop("preset", "impact"))

    style_overrides = bv.split_styles(
        style,
        components=("line", "errorbar", "scatter"),
        defaults={
            "line": {"linewidth": 2, "alpha": 0.5, "marker": "o", "markersize": 5},
            "errorbar": {
                "linewidth": 0,
                "elinewidth": 5,
                "marker": "_",
                "markersize": 12,
                "color": "#000000",
                "alpha": 0.6,
            },
            "scatter": {"linewidths": 0, "s": 50},
        },
    )

    df = _resolve(data)
    _need(
        df,
        [
            x,
            average_over,
            *([outcome] if value is None else [value]),
            *([compare] if compare else []),
        ],
        plot="impact",
    )

    if spec.x.ticks is None:
        spec = spec.with_xticks(df[x].drop_nulls().unique().sort().to_list())

    metric = (
        {"rate": outcome, "rate_of": success}
        if value is None
        else {"value": value, "stat": stat}
    )

    base_df = df.filter(pl.col("contrast") == 0.0)
    data_df = df.filter(pl.col("contrast") != 0.0)

    # per-subject value at each manipulation level (the paired lines)
    per_subject = aggregate(
        data_df, group=[x, average_over, *([compare] if compare else [])], **metric
    ).sort([x, average_over])

    # across-subject summary at each level (equal weight per subject) -- mean/sem via subject_average
    summary = subject_average(
        data_df, x=x, subject=average_over, compare=compare, **metric
    ).sort([x, *([compare] if compare else [])])

    grp = {"hue": compare} if compare else {}

    # one connecting line per subject (group= draws a separate line per subject)
    fig, ax = bv.plot_line(
        data=per_subject,
        x=x,
        y="value",
        group=average_over,
        spec=spec,
        ax=ax,
        **({"hue": compare} if compare else {"hue": average_over}),
        **style_overrides["line"],
    )
    fig, ax = bv.plot_scatter(
        data=per_subject,
        x=x,
        y="value",
        spec=spec,
        ax=ax,
        **({"hue": compare} if compare else {"hue": average_over}),
        **style_overrides["scatter"],
    )

    # across-subject summary point + SEM bar
    err = np.vstack([summary["sem"].to_numpy(), summary["sem"].to_numpy()])
    fig, ax = bv.plot_errorbar(
        data=summary,
        x=x,
        y="value",
        yerr=err,
        spec=spec,
        ax=ax,
        **grp,
        **style_overrides["errorbar"],
    )

    if not base_df.is_empty():
        per_sub_base = aggregate(base_df, group=[average_over], **metric)
        vals = per_sub_base["value"].drop_nulls().to_numpy()
        if vals.size:
            b_mean = float(np.mean(vals))
            b_sem = (
                float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
            )

            bv.plot_horizontal(
                b_mean,
                linestyle="--",
                color="#000000",
                ax=ax,
                spec=spec,
                zorder=1,
            )
            bv.plot_fill_between(
                summary[x].to_list(),
                b_mean - b_sem,
                b_mean + b_sem,
                color="#525252",
                linewidth=0,
                alpha=0.4,
                ax=ax,
                spec=spec,
                zorder=1,
            )

    # pairwise significance between manipulation levels (paired across subjects)
    test_res = None
    if test and df[x].drop_nulls().n_unique() >= 2:
        test_res = compare_groups(
            df,
            comparing=x,
            value=(outcome if value is None else value),
            success=(success if value is None else None),
            subject=average_over,
        )
        y_top = (
            float(summary["value"].max()) if summary["value"].drop_nulls().len() else 1.0
        )
        for i, (a, b, p) in enumerate(
            test_res.select(["group_a", "group_b", "pvalue"]).iter_rows()
        ):
            _, ax = bv.plot_pval(p, [a, b], y_top * (1.05 + 0.08 * i), spec=spec, ax=ax)

    return PlotResult(data=per_subject, stats=test_res, figure=(fig, ax))
