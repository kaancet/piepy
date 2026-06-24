"""Thin behaviz wrappers: each plot summarizes -> tests/fits -> draws, returning a PlotResult.

A plot function owns no maths and no drawing of its own. It calls the refactored stack
(``piepy.stats`` to shape, ``piepy.fitting`` to fit) and behaviz to draw, then bundles all three
outputs in a :class:`PlotResult`. They are task-agnostic: column names are arguments, so any trial
table with the right channels plots without a per-task function.
"""

from __future__ import annotations

import numpy as np
import polars as pl


from piepy.stats import aggregate, compare_by_x, subject_average
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
    model: str = "logistic",
    palette: tuple[str] | None = None,
    fit_curve: bool = True,
    color: str | None = None,
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
        model: psychometric model for the fitted curve ("logistic"/"weibull"/"erf").
        fit_curve: set False to skip the fit.
        palette: tuple of hex colors for the ``compare`` levels.
        color / ax / **style: forwarded to behaviz (``color`` for the single-series case).

    Returns:
        PlotResult(data=per-level estimate frame, stats=per-x test frame|None, figure=(fig, ax)).
    """
    # behaviz is the drawing backend; import it lazily so importing piepy.viz never requires it.
    import behaviz as bv

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
    # `spec` is the saved look (axes, limits, labels) for psychometric plots, loaded from ~/.behaviz.
    spec = bv.load_preset("psychometric")

    # resolve the input to a trial table and validate the columns we will touch
    # Run -> run.data.data, Session -> concatenate_runs(), Hub -> .data, df -> df
    df = _resolve(data)
    spec = spec.with_xticks(df[x].drop_nulls().unique().sort().to_list())

    _need(
        df,
        [x, outcome, *([compare] if compare else []), *([average_over] if average_over else [])],
        plot="psychometric",
    )  # structured error naming any missing column

    # aggregate to one tidy estimate per x-level (per `compare` level).
    group = [x, compare] if compare else x
    if average_over:
        # Two-stage / hierarchical: each subject's rate, then the mean across subjects (t-CI).
        # This weights subjects equally.
        agg = subject_average(df, x=x, subject=average_over, rate=outcome, rate_of=success, compare=compare)
    else:
        # Single-stage: pool all trials at each level, Wilson CI on the counts.
        agg = aggregate(df, group=group, rate=outcome, rate_of=success).sort(group)

    y = agg["value"].to_numpy()
    # behaviz wants error as (2, N) positive magnitudes; the agg gives absolute Wilson bounds.
    err = np.vstack([y - agg["ci_low"].to_numpy(), agg["ci_high"].to_numpy() - y])

    # `grp` is the grouping/colour kwargs: hue+palette when comparing, else a single colour
    grp = {"hue": compare, "palette": palette} if compare else {"color": color}
    fig, ax = bv.plot_errorbar(
        data=agg, x=x, y="value", err=err, spec=spec, ax=ax, **grp, **style_overrides["errorbar"]
    )

    # (only when comparing) a significance test at each x-level, drawn as p-value stars
    test_res = None
    if compare is not None:
        test_res = compare_by_x(df, x=x, subject=average_over, comparing=compare, value=outcome, success=success)
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
        fig, ax = bv.plot_line(data=curve, x=x, y="value", spec=spec, ax=ax, **grp, **style_overrides["line"])

    return PlotResult(data=agg, stats=test_res, figure=(fig, ax))


def reaction_time_cloud(
    data,
    *,
    x: str = "contrast",
    value: str = "reaction_time",
    compare: str | None = None,
    bin_width: float = 10,  # ms
    average_over: str | None = None,
    palette: tuple[str] | None = None,
    color: str | None = None,
    ax=None,
    **style,
) -> PlotResult:
    """"""
    import behaviz as bv

    spec = bv.load_preset("reaction_time_cloud")

    df = _resolve(data)
    spec = spec.with_xticks(df[x].drop_nulls().unique().sort().to_list())
    spec = spec.with_ylim(lo=df[value].min() - 10, hi=df[value].max() + 10)

    _need(
        df,
        [x, value, *([compare] if compare else []), *([average_over] if average_over else [])],
        plot="reaction time cloud",
    )  # structured error naming any missing column

    # aggregate to one tidy estimate per x-level (per `compare` level).
    group = [x, compare] if compare else x
    if average_over:
        # Two-stage / hierarchical: each subject's rate, then the mean across subjects (t-CI).
        # This weights subjects equally.
        agg = subject_average(df, x=x, subject=average_over, value=value, stat="median", compare=compare, points=True)
    else:
        # Single-stage: pool all trials at each level, Wilson CI on the counts.
        agg = aggregate(df, group=group, value=value, stat="median", points=True).sort(group)

    grp = {"hue": compare, "palette": palette} if compare else {"color": color}

    fig, ax = bv.plot_raincloud(
        data=agg, x=x, ys="points", bin_width=bin_width, cloud_side="left", spec=spec, ax=ax, **grp, **style
    )

    test_res = None
    if compare is not None:
        test_res = compare_by_x(df, x=x, subject=average_over, comparing=compare, value=value)
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
    palette: tuple[str] | None = None,
    color: str | None = None,
    ax=None,
    **style,
) -> PlotResult:
    """"""

    import behaviz as bv

    spec = bv.load_preset("reaction_distribution")

    df = _resolve(data)

    _need(
        df,
        [value, *([comparing] if comparing else []), *([average_over] if average_over else [])],
        plot="reaction time distribution",
    )  # structured error naming any missing column

    if average_over:
        # Two-stage / hierarchical: each subject's rate, then the mean across subjects (t-CI).
        # This weights subjects equally.
        agg = subject_average(df, subject=average_over, value=value, stat="median", compare=comparing, points=True)
    else:
        # Single-stage: pool all trials at each level, Wilson CI on the counts.
        agg = aggregate(df, group=comparing, value=value, stat="median", points=True).sort(comparing)

    grp = {"hue": comparing, "palette": palette} if comparing else {"color": color}

    fig, ax = bv.plot_hist1d(data=agg, values="points", bin_width=bin_width, spec=spec, ax=ax, **grp, **style)

    # draw median lines
    for comp_val, v in agg.select([comparing, "value"]).to_numpy():
        fig, ax = bv.plot_vertical(x=v, spec=spec, ax=ax, color="#990000")
        _, ax = bv.plot_text(
            v + 3,
            5,
            comp_val,
            ax=ax,
            spec=spec,
            rotation=90,
            ha="left",
            va="bottom",
            color="#000000",
        )

    # compare medians
    test_res = None
    if comparing is not None:
        test_res = compare_by_x(df, comparing=comparing, value=value, subject=average_over)
        for ii, (xc1, xc2, p) in enumerate(test_res.select(["group_a", "group_b", "pvalue"]).to_numpy()):
            xi1 = agg.filter(pl.col(comparing) == xc1)[0, "value"]
            xi2 = agg.filter(pl.col(comparing) == xc2)[0, "value"]
            _, ax = bv.plot_pval(
                p,
                [xi1, xi2],
                10 + (ii * 2),
                spec=spec,
                ax=ax,
            )

    return PlotResult(data=agg, stats=test_res, figure=(fig, ax))
