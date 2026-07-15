"""Thin behaviz wrappers: each plot summarizes -> tests/fits -> draws, returning a PlotResult.

A plot function owns no maths and no drawing of its own. It calls the refactored stack
(``piepy.stats`` to shape, ``piepy.fitting`` to fit) and behaviz to draw, then bundles all three
outputs in a :class:`PlotResult`. They are task-agnostic: column names are arguments, so any trial
table with the right channels plots without a per-task function.
"""

from __future__ import annotations

import numpy as np
import polars as pl


from piepy.stats import aggregate, compare_groups, compare_by_x, subject_average
from piepy.fitting import fit
from .base import PlotResult, _need, _resolve


def psychometric(
    data,
    *,
    x: str = "contrast",
    outcome: str = "outcome",
    success: object = "hit",
    compare: str | None = None,
    average_over: str | None = None,
    fit_curve: bool = True,
    model: str = "logistic",
    ax=None,
    **style,
) -> PlotResult:
    """Success-rate vs stimulus level: Wilson-CI points + an optional fitted curve.

    Args:
        data: a Run / Session / Hub / trial-table DataFrame (``_resolve`` handles all four).
        x: stimulus-level column to put on the x-axis (e.g. ``signed_contrast``).
        outcome / success: the rate at each level is the fraction of ``outcome == success``
            (pass ``success=None`` if ``outcome`` is already a 0/1 column).
        compare: a column to split on -> one colored curve per level (behaviz ``hue=``), a fit per
            level, and a per-x significance frame in ``stats``. ``None`` -> a single series.
        average_over: a subject column (e.g. ``animalid``) -> plot the **subject-averaged** rate
            (each subject's rate, then the mean across subjects with a t-CI -- no pseudoreplication).
            Usually you don't pass this by hand: ``hub.viz.psychometric()`` sets it for you.
        fit_curve: set False to skip the fit.
        model: psychometric model for the fitted curve ("logistic"/"weibull"/"erf").
        ax / **style: forwarded to behaviz. Colour rides in ``**style`` -- ``color=`` for a single
            series, ``palette=`` for the ``compare`` levels (behaviz's own kwargs; no separate args).
    Returns:
        PlotResult(data=per-level estimate frame, stats=per-x test frame|None, figure=(fig, ax)).
    """
    # behaviz is the drawing backend; import it lazily so importing piepy.viz never requires it.
    import behaviz as bv

    # `spec` is the saved look (axes, limits, labels) for psychometric plots, loaded from ~/.behaviz.
    spec = bv.load_preset(style.pop("preset", "psychometric"))

    # Pull styling for each drawn component out of **style (so callers can pass e.g.
    # error_markersize=12 / fit_linewidth=1) and merge over these defaults.
    style_overrides = bv.split_styles(
        style,
        components=("errorbar", "line"),
        defaults={
            "errorbar": {
                "linewidth": 0,
                "marker": "o",
                "markersize": 10,
                "markeredgewidth": 0.3,
                "markeredgecolor": "#FFFFFF",
                "dodge": "none",
            },
            "line": {
                "linewidth": 2.5,
            },
        },
    )

    # resolve the input to a trial table and validate the columns we will touch
    # Run -> run.data.data, Session -> concatenate_runs(), Hub -> .data, df -> df
    df = _resolve(data)

    _need(
        df,
        [
            x,
            outcome,
            *([compare] if compare else []),
            *([average_over] if average_over else []),
        ],
        plot="psychometric",
    )  # structured error naming any missing column

    spec = spec.with_xticks(df[x].drop_nulls().unique().sort().to_list())

    # aggregate to one tidy estimate per x-level (per `compare` level).
    group = [x, compare] if compare else x
    if average_over:
        # Two-stage / hierarchical: each subject's rate, then the mean across subjects (t-CI).
        # This weights subjects equally.
        agg = subject_average(
            df, x=x, subject=average_over, rate=outcome, rate_of=success, compare=compare
        )
    else:
        # Single-stage: pool all trials at each level, Wilson CI on the counts.
        agg = aggregate(df, group=group, rate=outcome, rate_of=success).sort(group)

    y = agg["value"].to_numpy()
    # behaviz wants error as (2, N) positive magnitudes; the agg gives absolute Wilson bounds.
    err = np.vstack([y - agg["ci_low"].to_numpy(), agg["ci_high"].to_numpy() - y])

    # split a compare column onto behaviz's `hue`; colour (color=/palette=) rides in **style
    grp = {"hue": compare} if compare else {}
    fig, ax = bv.plot_errorbar(
        data=agg,
        x=x,
        y="value",
        yerr=err,
        spec=spec,
        ax=ax,
        **grp,
        **style_overrides["errorbar"],
    )

    # (only when comparing) a significance test at each x-level, drawn as p-value stars
    test_res = None
    if compare is not None:
        if df[compare].drop_nulls().n_unique() < 2:
            raise ValueError(f"{compare} column has less than 2 values")
        else:
            test_res = compare_by_x(
                df,
                x=x,
                subject=average_over,
                comparing=compare,
                value=outcome,
                success=success,
            )
            for xc, p in test_res.select([x, "pvalue"]).to_numpy():
                _, ax = bv.plot_pval(
                    p,
                    [xc, xc],
                    spec.y.lim[1],  # y-loc
                    spec=spec,
                    ax=ax,
                )

    # fit a curve and draw it.
    if fit_curve:
        # When subject-averaged, `n` is a subject count (not binomial trials) -> least-squares
        def fit_n(frame):
            return None if average_over else frame["n"].to_numpy()

        if compare is None:
            f = fit(model, agg[x].to_numpy(), agg["value"].to_numpy(), n=fit_n(agg))
            curve = pl.DataFrame(dict(zip([x, "value"], f.curve())))
        else:
            # one fit per compare level, stacked long-form so behaviz can hue the curves to match
            parts = []
            for key, sub in agg.group_by(compare, maintain_order=True):
                lvl = key[0] if isinstance(key, tuple) else key
                f = fit(model, sub[x].to_numpy(), sub["value"].to_numpy(), n=fit_n(sub))
                xx, yy = f.curve()
                parts.append(pl.DataFrame({x: xx, "value": yy, compare: lvl}))
            curve = pl.concat(parts)
        fig, ax = bv.plot_line(
            data=curve, x=x, y="value", spec=spec, ax=ax, **grp, **style_overrides["line"]
        )

    return PlotResult(data=agg, stats=test_res, figure=(fig, ax))


def reaction_time_cloud(
    data,
    *,
    x: str = "contrast",
    value: str = "reaction_time",
    compare: str | None = None,
    bin_width: float = 10,  # ms
    average_over: str | None = None,
    ax=None,
    **style,
) -> PlotResult:
    """Reaction time vs stimulus level as a raincloud per x-level (per ``compare`` level).

    Same resolve -> aggregate -> test -> draw shape as :func:`psychometric`, but the summary is the
    median ``value`` per level with its raw points drawn as a cloud; ``compare`` overlays groups and
    adds a per-x test.
    """
    import behaviz as bv

    spec = bv.load_preset(style.pop("preset", "reaction_time_cloud"))

    df = _resolve(data)

    _need(
        df,
        [
            x,
            value,
            *([compare] if compare else []),
            *([average_over] if average_over else []),
        ],
        plot="reaction time cloud",
    )  # structured error naming any missing column

    spec = spec.with_xticks(df[x].drop_nulls().unique().sort().to_list())
    vals = df[value].drop_nulls()
    if vals.len():  # empty / all-null value column -> keep the preset's ylim
        spec = spec.with_ylim(lo=vals.min() - 10, hi=vals.max() + 10)

    # aggregate to one tidy estimate per x-level (per `compare` level).
    group = [x, compare] if compare else x
    if average_over:
        # Two-stage / hierarchical: each subject's rate, then the mean across subjects (t-CI).
        # This weights subjects equally.
        agg = subject_average(
            df,
            x=x,
            subject=average_over,
            value=value,
            stat="median",
            compare=compare,
            points=True,
        )
    else:
        # Single-stage: pool all trials at each level, Wilson CI on the counts.
        agg = aggregate(df, group=group, value=value, stat="median", points=True).sort(
            group
        )

    grp = {"hue": compare} if compare else {}
    draw = agg.filter(pl.col("points").list.len() > 0)
    if draw.is_empty():
        raise ValueError(f"reaction time cloud: no non-null {value!r} values to plot.")
    fig, ax = bv.plot_raincloud(
        data=draw,
        x=x,
        ys="points",
        bin_width=bin_width,
        cloud_side="left",
        spec=spec,
        ax=ax,
        **grp,
        **style,
    )

    test_res = None
    if compare is not None:
        test_res = compare_by_x(
            df, x=x, subject=average_over, comparing=compare, value=value
        )
        for xc, p in test_res.select([x, "pvalue"]).to_numpy():
            _, ax = bv.plot_pval(
                p,
                [xc, xc],
                spec.y.lim[1],  # y-loc
                spec=spec,
                ax=ax,
            )

    return PlotResult(data=agg, stats=test_res, figure=(fig, ax))


def reaction_time_dist(
    data,
    *,
    value: str = "reaction_time",
    comparing: str | None = None,
    bin_width: float = 10,  # ms
    average_over: str | None = None,
    ax=None,
    **style,
) -> PlotResult:
    """Reaction-time distribution: a histogram + median line(s). With ``comparing`` it overlays one
    histogram per group and tests the groups pairwise; without it, a single distribution (no test).
    """
    import behaviz as bv

    spec = bv.load_preset(style.pop("preset", "reaction_distribution"))

    df = _resolve(data)

    _need(
        df,
        [
            value,
            *([comparing] if comparing else []),
            *([average_over] if average_over else []),
        ],
        plot="reaction time distribution",
    )  # structured error naming any missing column

    # one row per group (or a single row when there is no `comparing`), each carrying the median
    # `value` plus the raw `points` the histogram is drawn from.
    if comparing:
        if average_over:
            # Two-stage / hierarchical: per-subject median, then averaged across subjects.
            agg = subject_average(
                df,
                subject=average_over,
                value=value,
                stat="median",
                compare=comparing,
                points=True,
            ).sort(comparing)
        else:
            agg = aggregate(
                df, group=comparing, value=value, stat="median", points=True
            ).sort(comparing)
    else:
        # single distribution over the whole frame (group by a constant, then drop it). With no
        # `comparing` there is nothing to subject-average across, so `average_over` is ignored here
        # (subject_average(compare=None) would group by an empty key set and raise).
        agg = aggregate(
            df.with_columns(pl.lit("all").alias("_grp")),
            group="_grp",
            value=value,
            stat="median",
            points=True,
        ).drop("_grp")

    grp = {"hue": comparing} if comparing else {}
    # behaviz's histogram reduces over each group's points; an empty group blows up inside it
    # (min() of an empty array), so only hand it the groups that actually have values.
    draw = agg.filter(pl.col("points").list.len() > 0)
    if draw.is_empty():
        raise ValueError(
            f"reaction time distribution: no non-null {value!r} values to plot."
        )
    fig, ax = bv.plot_hist1d(
        data=draw, values="points", bin_width=bin_width, spec=spec, ax=ax, **grp, **style
    )

    # median line(s) + label -- the group value when comparing, else the median itself
    for row in agg.iter_rows(named=True):
        v = row["value"]
        if v is None:  # a group with no (non-null) values -> no median to mark
            continue
        fig, ax = bv.plot_vertical(x=v, spec=spec, ax=ax, color="#990000")
        label = str(row[comparing]) if comparing else f"{v:.0f}"
        _, ax = bv.plot_text(
            v + 3,
            5,
            label,
            ax=ax,
            spec=spec,
            rotation=90,
            ha="left",
            va="bottom",
            color="#000000",
        )

    # pairwise comparison only makes sense with >= 2 groups
    test_res = None
    if comparing is not None and df[comparing].drop_nulls().n_unique() >= 2:
        test_res = compare_groups(
            df, comparing=comparing, value=value, subject=average_over
        )
        medians = dict(
            zip(agg[comparing].to_list(), agg["value"].to_list())
        )  # group -> its median
        ii = 0
        # iter_rows keeps native dtypes (a mixed to_numpy would coerce string groups + float p)
        for xc1, xc2, p in test_res.select(["group_a", "group_b", "pvalue"]).iter_rows():
            xi1, xi2 = medians.get(xc1), medians.get(xc2)
            if (
                xi1 is None or xi2 is None
            ):  # a group with no median -> nowhere to anchor the bracket
                continue
            _, ax = bv.plot_pval(p, [xi1, xi2], 10 + (ii * 2), spec=spec, ax=ax)
            ii += 1

    return PlotResult(data=agg, stats=test_res, figure=(fig, ax))
