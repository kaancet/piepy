import itertools
import numpy as np
import polars as pl
from typing import Literal
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from scipy.stats import (
    wilcoxon,  # noqa: F401
    ttest_rel,
    shapiro,
)

from ....core.data_functions import make_subsets
from ....core.statistics import mean_confidence_interval
from ...plotting_utils import set_style, make_named_axes, override_plots, pval_plotter
from ....psychophysics.wheel.detection.wheelDetectionGroupedAggregator import (
    WheelDetectionGroupedAggregator,
)
from ....psychophysics.wheel.detection.wheelDetectionExperimentHub import (
    generate_unique_session_id,
)


ANIMAL_COLORS = {
    "KC139": "#332288",
    "KC141": "#117733",
    "KC142": "#DDCC77",
    "KC143": "#AA4499",
    "KC144": "#882255",
    "KC145": "#88CCEE",
    "KC146": "#275D6D",
    "KC147": "#F57A6C",
    "KC148": "#ADFA9A",
    "KC149": "#A45414",
    "KC150": "#0000FF",
    "KC151": "#00FF11",
    "KC152": "#FFAA33",
}


X_AXIS_LABEL = {-1: "Non\nOpto", 0: "Opto", 1: "Opto\nOff"}


def _plot_single_panel(
    ax: plt.Axes,
    panel_data: pl.DataFrame,
    plot_of: str,
    plot_with: Literal["sem", "conf", "iqr"] = "sem",
    mpl_kwargs: dict = {},
    **kwargs,
) -> plt.Axes:
    """Plots a single panel in the impact plot figure, which usually seperated by contrast and/or stimulus type

    Args:
        ax (plt.Axes): Axes to be plotted on
        panel_data (pl.DataFrame): Data to be plotted. Should be filterediltered to have only one contrast and stimulus type
        plot_of (str): column name that corresponds to the plotted valu, e.g. "hit_rate", "median_reaction_time" etc.
        plot_with (Literal["sem","conf","iqr"], optional): Distribution measurement of the datapoints, which will be drawn as a shaded region . Defaults to "sem".
        mpl_kwargs (dict | {}, optional): kwargs for styling matplotlib plots. Defaults to {}.

    Returns:
        plt.Axes: Plotted axis
    """
    min_trial_count = kwargs.get("min_trial_count", 15)
    # plotting single sessions
    for filt_tup in make_subsets(panel_data, ["session_no"]):
        filt_df = filt_tup[-1]

        if not filt_df.is_empty():
            x_axis = filt_df["opto_pattern"].unique().sort().to_numpy()

            if plot_of == "hit_rate":
                if len(x_axis) == 1 and filt_df[0, "contrast"] == 0:
                    # if only single opto_pattern value(-1) aggegate the values to have baseline
                    y_axis = filt_df["hit_count"].sum() / filt_df["count"].sum()
                else:
                    y_axis = filt_df["hit_rate"].to_numpy()
                y_axis = 100 * y_axis
            elif "time" in plot_of:
                y_axis = np.array(
                    [
                        val if t_count >= min_trial_count else np.nan
                        for val, t_count in zip(
                            filt_df[plot_of].to_numpy(), filt_df["hit_count"].to_list()
                        )
                    ]
                )

            # individual sessions
            # add a very little random jitter
            jit = np.random.randn(len(x_axis)) / 60
            ax._plot(
                x_axis + jit,
                y_axis,
                c=kwargs.get("color", ANIMAL_COLORS[filt_df[0, "animalid"]]),
                label=filt_df[0, "animalid"],
                marker="o",
                markersize=10,
                markeredgewidth=1,
                linewidth=2,
                mpl_kwargs=mpl_kwargs,
                zorder=1,
            )

    # plotting average and propagated error
    if "time" in plot_of:
        panel_data = panel_data.filter(pl.col("hit_count") >= min_trial_count)

    avg_df = (
        panel_data.group_by(["opto_pattern"])
        .agg(
            [
                pl.col(plot_of).mean().alias(f"mean_{plot_of}"),
                pl.col(plot_of).median().alias(f"median_{plot_of}"),
                (pl.col(plot_of).std() / pl.col(plot_of).len().sqrt()).alias(
                    f"sem_{plot_of}"
                ),
                pl.col(plot_of)
                .map_elements(
                    mean_confidence_interval, return_dtype=pl.List(pl.Float64)
                )
                .alias(f"conf_{plot_of}"),
                pl.concat_list(
                    [
                        pl.col(plot_of).median() - pl.col(plot_of).quantile(0.25),
                        pl.col(plot_of).quantile(0.75) - pl.col(plot_of).median(),
                    ]
                ).alias(f"iqr_{plot_of}"),
            ]
        )
        .sort(["opto_pattern"])
    )
    # above mean_confidence_interval function returns the mean and CI, get only the CI
    avg_df = avg_df.with_columns(pl.col(f"conf_{plot_of}").list.get(1))

    # *100 if hit_rate
    if plot_of == "hit_rate":
        avg_df = avg_df.with_columns(
            mean_hit_rate=100 * pl.col("mean_hit_rate"),
            median_hit_rate=100 * pl.col("median_hit_rate"),
            sem_hit_rate=100 * pl.col("sem_hit_rate"),
            conf_hit_rate=100 * pl.col("conf_hit_rate"),
            iqr_hit_rate=100 * pl.col("iqr_hit_rate"),
        )

    if plot_with == "iqr":
        _plotting = avg_df[f"median_{plot_of}"].to_numpy()
    else:
        _plotting = avg_df[f"mean_{plot_of}"].to_numpy()

    # mean or median
    ax._scatter(
        avg_df["opto_pattern"].to_numpy(),
        _plotting,
        s=(plt.rcParams["lines.markersize"] ** 2),
        c="k",
        marker="_",
        zorder=3,
    )

    # shaded CI
    bars = np.array(avg_df[f"{plot_with}_{plot_of}"].to_numpy()).transpose()
    ax.errorbar(
        avg_df["opto_pattern"].to_numpy(),
        _plotting,
        bars,
        markersize=0,
        markeredgewidth=1,
        elinewidth=5,
        linewidth=0,
        c="k",
        alpha=0.5,
        zorder=2,
    )
    return ax


def plot_hit_rate_change(
    data: pl.DataFrame,
    plot_with: Literal["sem", "conf", "iqr"] = "sem",
    stim_side: Literal["ipsi", "contra"] = "contra",
    p_test: Literal["wilcoxon", "paired_t", "auto"] = "auto",
    mpl_kwargs: dict | None = None,
    **kwargs,
) -> plt.Figure:
    """Plots the change in hit-rates between opto and non-opto

    Args:
        data (pl.DataFrame): Experimental data
        plot_with (Literal["sem","conf","iqr"], optional): Distribution measurement of the datapoints, which will be drawn as a shaded region . Defaults to "sem".
        stim_side (Literal["contra","ipsi"], optional): Which side of stimulus presentation to plot. Defaults to "contra".
        p_test (Literal["wilcoxon","paired_t"], optional): Statistical test between opto and non-opto. Defaults to "wilcoxon".
        mpl_kwargs (dict | None, optional): kwargs for styling matplotlib plots. Defaults to None.

    Returns:
        plt.Figure: Figure of the plot
    """
    override_plots()
    set_style(kwargs.get("style", "presentation"))
    if mpl_kwargs is None:
        mpl_kwargs = {}

    # if provided dataframe has no session no column create session_no column
    if "session_no" not in data.columns:
        _animalids = data["animalid"].to_list()
        _baredates = data["baredate"].to_list()
        vec_func = np.vectorize(generate_unique_session_id)
        unique_ids = vec_func(_animalids, _baredates)
        data = data.with_columns(pl.Series("session_no", unique_ids))

    aggregator = WheelDetectionGroupedAggregator()
    aggregator.set_data(data=data)
    aggregator.group_data(
        group_by=[
            "animalid",
            "session_no",
            "stim_type",
            "stim_side",
            "contrast",
            "opto_pattern",
        ]
    )
    aggregator.calculate_hit_rates()
    aggregator.calculate_opto_pvalues()

    plot_data = aggregator.grouped_data.drop_nulls("contrast")

    if stim_side == "contra":
        plot_data = plot_data.filter(pl.col("stim_side") != "ipsi")
    elif stim_side == "ipsi":
        plot_data = plot_data.filter(pl.col("stim_side") != "contra")

    fig, axes_dict = make_named_axes(
        data=plot_data,
        col_names=["stim_type", "contrast"],
        figsize=mpl_kwargs.pop("figsize", (12, 12)),
    )

    # calculate the baseline from all the zero contrast non-optos
    zero_nonopto_df = plot_data.filter(
        (pl.col("contrast") == 0) & (pl.col("opto_pattern") == -1)
    )
    baseline_avg = 100 * zero_nonopto_df["hit_rate"].mean()
    baseline_err = (
        100
        * (zero_nonopto_df["hit_rate"].std())
        / np.sqrt(zero_nonopto_df["hit_rate"].len())
    )
    # plot zero only
    zero_opto_df = plot_data.filter((pl.col("contrast") == 0))
    for s in zero_opto_df["stim_type"].drop_nulls().unique().sort().to_list():
        ax_key = f"{s}_{0.0}"
        ax = axes_dict[ax_key]
        ax = _plot_single_panel(
            ax,
            zero_opto_df,
            plot_of="hit_rate",
            plot_with=plot_with,
            mpl_kwargs=mpl_kwargs,
            **kwargs,
        )
        ax.set_ylabel(f"{s}\nHit rate (%)")

    # nonzero data
    nonzero_data = plot_data.filter(pl.col("contrast") != 0)
    for filt_tup in make_subsets(nonzero_data, ["stim_type", "contrast"]):
        filt_df = filt_tup[-1]
        filt_contrast = filt_tup[1]
        filt_stimtype = filt_tup[0]
        ax_key = f"{filt_stimtype}_{filt_contrast}"

        ax = axes_dict[ax_key]
        ax = _plot_single_panel(
            ax,
            filt_df,
            plot_of="hit_rate",
            plot_with=plot_with,
            mpl_kwargs=mpl_kwargs,
            **kwargs,
        )

        opto_pattern_df = (
            filt_df.group_by(["opto_pattern"]).agg([pl.col("*")]).sort("opto_pattern")
        )
        uniq_opto_pattern = (
            opto_pattern_df["opto_pattern"].drop_nulls().unique().sort().to_list()
        )
        if len(uniq_opto_pattern) > 1:
            for i, j in list(
                itertools.combinations([x for x in range(len(uniq_opto_pattern))], 2)
            ):
                hr_1 = opto_pattern_df[i, "hit_rate"].to_list()
                hr_2 = opto_pattern_df[j, "hit_rate"].to_list()

                if p_test == "auto":
                    _is_norm1 = shapiro(hr_1)
                    _is_norm2 = shapiro(hr_2)
                    if (
                        _is_norm1.pvalue < 0.05 and _is_norm2.pvalue < 0.05
                    ):  # both are normal
                        print(
                            f"p1={_is_norm1.pvalue:4} and p2={_is_norm2.pvalue:.4}, data appears normal, doing paired t-test"
                        )
                        p_test = "paired_t"
                    else:
                        print(
                            f"p1={_is_norm1.pvalue:4} and p2={_is_norm2.pvalue:.4}, data doesn't appear normal, doing wilcoxon t-test"
                        )
                        p_test = "wilcoxon"

                if p_test == "wilcoxon":
                    res = wilcoxon(hr_1, hr_2, nan_policy="omit")
                elif p_test == "paired_t":
                    res = ttest_rel(hr_1, hr_2, nan_policy="omit")
                p = res.pvalue
                print(filt_stimtype, filt_contrast, res, flush=True)
                ax = pval_plotter(
                    ax, p, pos=[uniq_opto_pattern[i], uniq_opto_pattern[j]], loc=100
                )

            ax.tick_params(
                axis="y", labelsize=0, length=0, width=0, which="major", color="k"
            )

    # make them all look pretty
    for k in axes_dict.keys():
        ax = axes_dict[k]

        ax.axhline(y=baseline_avg, linestyle="--", c="k", alpha=0.4, zorder=1)
        ax.axhspan(
            baseline_avg + baseline_err,
            baseline_avg - baseline_err,
            color="gray",
            alpha=0.2,
            linewidth=0,
            zorder=1,
        )

        ax.set_ylim([0, 110])
        ax.set_yticks([0, 25, 50, 75, 100])
        _x_ticks = filt_df["opto_pattern"].unique().sort().to_list()
        ax.set_xlabel(f"c={k.split('_')[-1]}", labelpad=10)
        ax.set_xticks(_x_ticks)
        ax.set_xticklabels([X_AXIS_LABEL[i] for i in _x_ticks])
        ax.tick_params(
            axis="x",
            length=10,
            which="major",
            color="k",
        )
        ax.grid(True, axis="y", alpha=0.4)

        ax.spines["bottom"].set_position(("outward", 20))
        ax.spines["bottom"].set_linewidth(2)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    plt.subplots_adjust(
        hspace=mpl_kwargs.pop("hspace", 0.5), wspace=mpl_kwargs.pop("wspace", 0.25)
    )
    return fig


def plot_reaction_time_change(
    data: pl.DataFrame,
    reaction_of: Literal["reaction", "response"] = "reaction",
    stim_side: Literal["contra", "ipsi"] = "contra",
    plot_with: Literal["sem", "conf"] = "sem",
    p_test: Literal["wilcoxon", "paired_t"] = "wilcoxon",
    mpl_kwargs: dict | None = None,
    **kwargs,
) -> plt.Figure:
    """Plots the change in reaction times between opto and non-opto

    Args:
        data (pl.DataFrame): Experimental data
        reaction_of (Literal["reaction","response"], optional): Plot reaction or response time. Defaults to "reaction".
        stim_side (Literal["contra","ipsi"], optional): Which side of stimulus presentation to plot. Defaults to "contra".
        plot_with (Literal["sem","conf","iqr"], optional): Distribution measurement of the datapoints, which will be drawn as a shaded region . Defaults to "sem".
        p_test (Literal["wilcoxon","paired_t"], optional): Statistical test between opto and non-opto. Defaults to "wilcoxon".
        mpl_kwargs (dict | None, optional): kwargs for styling matplotlib plots. Defaults to None.

    Returns:
        plt.Figure: Figure of the plot
    """

    override_plots()
    set_style(kwargs.get("style", "presentation"))
    if mpl_kwargs is None:
        mpl_kwargs = {}

    # if provided dataframe has no session no column create session_no column
    if "session_no" not in data.columns:
        _animalids = data["animalid"].to_list()
        _baredates = data["baredate"].to_list()
        vec_func = np.vectorize(generate_unique_session_id)
        unique_ids = vec_func(_animalids, _baredates)
        data = data.with_columns(pl.Series("session_no", unique_ids))

    aggregator = WheelDetectionGroupedAggregator()
    aggregator.set_data(data=data)
    aggregator.group_data(
        group_by=[
            "animalid",
            "session_no",
            "stim_type",
            "stim_side",
            "contrast",
            "opto_pattern",
        ]
    )
    aggregator.calculate_hit_rates()
    aggregator.calculate_opto_pvalues()

    plot_data = aggregator.grouped_data.drop_nulls("contrast")

    if stim_side == "contra":
        plot_data = plot_data.filter(pl.col("stim_side") != "ipsi")
    elif stim_side == "ipsi":
        plot_data = plot_data.filter(pl.col("stim_side") != "contra")

    fig, axes_dict = make_named_axes(
        data=plot_data,
        col_names=["stim_type", "contrast"],
        figsize=mpl_kwargs.pop("figsize", (12, 12)),
    )

    if not kwargs.get("include_misses", False):
        reaction_of = "hit_" + reaction_of + "_times"
    plot_of = "median_" + reaction_of

    # nonzero data
    nonzero_data = plot_data.filter(pl.col("contrast") != 0)
    for filt_tup in make_subsets(nonzero_data, ["stim_type", "contrast"]):
        filt_df = filt_tup[-1]
        filt_contrast = filt_tup[1]
        filt_stimtype = filt_tup[0]
        ax_key = f"{filt_stimtype}_{filt_contrast}"

        ax = axes_dict[ax_key]
        ax = _plot_single_panel(
            ax,
            filt_df,
            plot_of=plot_of,
            plot_with=plot_with,
            mpl_kwargs=mpl_kwargs,
            **kwargs,
        )

        opto_pattern_df = (
            filt_df.group_by(["opto_pattern"]).agg([pl.col("*")]).sort("opto_pattern")
        )
        uniq_opto_pattern = (
            opto_pattern_df["opto_pattern"].drop_nulls().unique().sort().to_list()
        )
        if len(uniq_opto_pattern) > 1:
            for i, j in list(
                itertools.combinations([x for x in range(len(uniq_opto_pattern))], 2)
            ):
                rt_1 = opto_pattern_df[i, f"{plot_of}"].to_numpy()
                rt_2 = opto_pattern_df[j, f"{plot_of}"].to_numpy()

                if p_test == "auto":
                    _is_norm1 = shapiro(rt_1)
                    _is_norm2 = shapiro(rt_2)
                    if (
                        _is_norm1.pvalue < 0.05 and _is_norm2.pvalue < 0.05
                    ):  # both are normal
                        print(
                            f"p1={_is_norm1.pvalue:4} and p2={_is_norm2.pvalue:.4}, data appears normal, doing paired t-test"
                        )
                        p_test = "paired_t"
                    else:
                        print(
                            f"p1={_is_norm1.pvalue:4} and p2={_is_norm2.pvalue:.4}, data doesn't appear normal, doing wilcoxon t-test"
                        )
                        p_test = "wilcoxon"

                if p_test == "wilcoxon":
                    res = wilcoxon(rt_1, rt_2, nan_policy="omit")
                elif p_test == "paired_t":
                    res = ttest_rel(rt_1, rt_2, nan_policy="omit")
                p = res.pvalue
                print(filt_stimtype, filt_contrast, res)
                ax = pval_plotter(
                    ax, p, pos=[uniq_opto_pattern[i], uniq_opto_pattern[j]], loc=900
                )
            ax.tick_params(
                axis="y", labelsize=0, length=0, width=0, which="major", color="k"
            )

    # make them all look pretty
    for i, k in enumerate(axes_dict.keys()):
        ax = axes_dict[k]

        ax.set_ylim([100, 1000])
        ax.set_yticks([200, 400, 600, 800, 1000])

        if i == 0:
            _yl = plot_of.split("_")
            ax.set_ylabel(f"{' '.join(_yl).capitalize()} (ms)")

        _x_ticks = filt_df["opto_pattern"].unique().sort().to_list()
        ax.set_xlabel(f"c={k.split('_')[-1]}", labelpad=10)
        ax.set_xticks(_x_ticks)
        ax.set_xticklabels([X_AXIS_LABEL[i] for i in _x_ticks])
        ax.tick_params(
            axis="x",
            length=10,
            which="major",
            color="k",
        )

        ax.spines["bottom"].set_position(("outward", 20))
        ax.spines["bottom"].set_linewidth(2)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        ax.set_yscale("symlog")
        minor_locs = [200, 400, 600, 800]
        ax.yaxis.set_minor_locator(plt.FixedLocator(minor_locs))
        ax.yaxis.set_minor_formatter(plt.FormatStrFormatter("%d"))
        ax.yaxis.set_major_locator(ticker.FixedLocator([100, 1000, 10000]))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
        ax.set_ylim([150, 1000])
        ax.grid(True, axis="y", which="both", alpha=0.4)

    plt.subplots_adjust(
        hspace=mpl_kwargs.pop("hspace", 0.5), wspace=mpl_kwargs.pop("wspace", 0.25)
    )
    return fig
