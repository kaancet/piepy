import itertools
import numpy as np
import polars as pl
from typing import Literal
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  # noqa: F401
from matplotlib.colors import Normalize, BoundaryNorm, LinearSegmentedColormap
from collections import defaultdict
from scipy import stats
from statsmodels.stats.multitest import multipletests

from ....core.data_functions import make_subsets, get_baseline_hr
from ....core.statistics import bootstrap_confidence_interval
from ...plotting_utils import (
    set_style,
    make_linear_axis,
    make_dot_cloud,
    override_plots,
    pval_plotter,
)
from ...colors.color import Color
from ....psychophysics.wheel.detection.wheelDetectionGroupedAggregator import (
    WheelDetectionGroupedAggregator,
)


@np.vectorize
def lenient_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """returns the numerator if denominator is zero, otherwise does regular division

    Args:
        num (Number | np.ndarray): Numerator
        den (Number | np.ndarray): Denominator

    Returns:
        Number | np.ndarray: Division result
    """
    return float(den and num / den or num)


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

# values from allen ccf_2022, using the shoelace algorithm to calculate the area from area boundary coordinates
AREA_SIZE = {
    "V1": 4.002,
    "HVA": 2.925,
    "dorsal": 1.428,
    "ventralPM": 1.496,
    "LM": 0.571,
    "AL": 0.389,
    "RL": 0.583,
    "PM": 0.719,
    "AM": 0.456,
}
# "LI":0.207
# "cortex" : 6.927,

# hierarchy scores from steinmetz?
AREA_SCORE = {}


X_AXIS_LABEL_STYPE = {
    0: "0.04cpd_8Hz",
    1: "0.16cpd_0.5Hz",
}

X_AXIS_LABEL_CNO = {0: "Non\nCNO", 1: "CNO"}


# TODO: axis positioning is a bit hardcoded
def plot_delta_effect_contrast(
    data: pl.DataFrame,
    effect_on: Literal["hit_rate", "reaction", "response"],
    effect_metric: Literal["delta", "BSI", "BSI_base"] = "delta",
    plot_with: Literal["sem", "conf", "iqr"] = "sem",
    p_test: Literal["wilcoxon", "t_test", "auto"] = "wilcoxon",
    ax: plt.Axes | None = None,
    include_misses: bool = False,
    trial_count_identifier: Literal["dot_color", "dot_size"] = "dot_color",
    polarity: Literal[-1, 1] = 1,
    mpl_kwargs: dict | None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Plots a connected scatter plot of the delta values between opto and non-opto conditions for different contrasts.
    Each connected dot-pair corresponds to a session and the delta in different contrast conditions

    Args:
        data (pl.DataFrame): Experimental data
        effect_on (Literal["hit_rate","reaction","response"]): The behavioral readout to plot.
        plot_with (Literal["sem","conf","iqr"], optional): The distibution measurement. Defaults to "sem"
        ax (plt.Axes | None, optional): Precreated axis to plot to. Defaults to None.
        include_misses (bool, optional): Whether or not to include miss trials. Does nothing if 'effect_on'="hit_rate". Defaults to False. Doesn't make sense if effect_on is "hit_rate".

        polarity (Literal[-1,1], optional): Polarity of the metric. 1 is nonopto - opto, -1 is opto - nonopto. Defaults to 1.
        mpl_kwargs (dict | None, optional): kwargs for styling matplotlib plots. Defaults to None.

    Returns:
        tuple[plt.Figure, plt.Axes]: Figure and axes of plot
    """

    if effect_on != "hit_rate":
        if not include_misses:
            effect_on = f"hit_{effect_on}"
        effect_on = f"median_{effect_on}_times"

    polarity = float(polarity)
    clr = Color(task="detection")
    override_plots()
    set_style(kwargs.get("style", "presentation"))
    if mpl_kwargs is None:
        mpl_kwargs = {}

    if ax is None:
        fig = plt.figure(figsize=mpl_kwargs.pop("figsize", (8, 8)))
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = ax.get_figure()

    # colormap stuff
    cmap_name = kwargs.get("dot_cmap", "Greens")
    cmap = plt.get_cmap(cmap_name)

    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # force the first color entry to be grey
    cmaplist[0] = (0.5, 0.5, 0.5, 1.0)

    trial_bounds = kwargs.get("trial_bounds", (0, 1, 5, 20, 50, 100))
    # create the new map
    cmap = LinearSegmentedColormap.from_list("Custom cmap", cmaplist, len(trial_bounds))
    normalizer = BoundaryNorm(trial_bounds, len(trial_bounds))
    im = cm.ScalarMappable(norm=normalizer, cmap=cmap)

    aggregator = WheelDetectionGroupedAggregator()
    aggregator.set_data(data=data)
    aggregator.group_data(
        group_by=[
            "animalid",
            "session_id",
            "stim_type",
            "stim_side",
            "contrast",
            "opto_pattern",
        ]
    )
    aggregator.calculate_hit_rates()
    aggregator.calculate_opto_pvalues()

    plot_data = aggregator.grouped_data.drop_nulls("contrast").filter(pl.col("stim_side") != "ipsi")
    lin_axis_dict = make_linear_axis(plot_data, "signed_contrast")
    _lin_axis = [
        float(lin_axis_dict[c]) if c is not None else None for c in plot_data["contrast"].to_list()
    ]
    plot_data = plot_data.with_columns(pl.Series("linear_axis", _lin_axis))

    for stim_tup in make_subsets(plot_data, ["stim_type"], start_enumerate=-1):
        pos_ofs = stim_tup[0]
        stim_type = stim_tup[1]
        stim_df = stim_tup[-1]
        diff_values = defaultdict(list)
        trial_count_values = defaultdict(list)
        x_pos_values = []
        animalids = defaultdict(list)
        for filt_tup in make_subsets(stim_df, ["session_id"]):
            filt_df = filt_tup[-1]
            session_baseline_hr = get_baseline_hr(filt_df)
            if not filt_df.is_empty():
                q = (
                    filt_df.group_by(["opto_pattern", "contrast"])
                    .agg(
                        [
                            pl.col(effect_on).get(0),
                            pl.col("hit_count").get(0),
                            pl.col("count").get(0),
                        ]
                    )
                    .sort(["opto_pattern", "contrast"])
                )

                vals_nonopto = q.filter(pl.col("opto_pattern") == -1)[effect_on].to_numpy()
                contrast_nonopto = q.filter(pl.col("opto_pattern") == -1)["contrast"].to_list()

                opto_patterns = q["opto_pattern"].sort().unique().to_numpy()
                opto_patterns = [o for o in opto_patterns if o != -1]
                for o in opto_patterns:
                    vals_opto = q.filter(pl.col("opto_pattern") == o)[effect_on].to_numpy()
                    contrast_opto = q.filter(pl.col("opto_pattern") == 0)["contrast"].to_list()
                    hit_count_opto = q.filter(pl.col("opto_pattern") == o)["hit_count"].to_numpy()

                    trial_count_opto = q.filter(pl.col("opto_pattern") == o)["count"].to_numpy()

                    paired_idx = [i for i, c in enumerate(contrast_nonopto) if c in contrast_opto]
                    vals_nonopto = vals_nonopto[paired_idx]

                    if effect_on == "hit_rate":
                        if effect_metric == "delta":
                            # simple delta
                            temp = vals_nonopto[: len(vals_opto)] - vals_opto
                        elif effect_metric == "BSI":
                            # BSI: (nonopto - opto) / nonopto
                            # what percentage is delta hit rates of nonopto
                            _top = vals_nonopto[: len(vals_opto)] - vals_opto
                            temp = lenient_div(_top, vals_nonopto[: len(vals_opto)])

                        elif effect_metric == "BSI_base":
                            # BSI_base: (|nonopto - baseline| - |opto-baseline|) / (|nonopto - baseline|)
                            # with baseline normalization
                            _opto = np.abs(vals_opto - session_baseline_hr)
                            _nonopto = np.abs(vals_nonopto[: len(vals_opto)] - session_baseline_hr)
                            _top = _nonopto - _opto
                            temp = lenient_div(_top, _nonopto)

                        _count_vals = trial_count_opto

                    if "time" in effect_on:
                        _count_vals = hit_count_opto
                        temp = vals_nonopto[: len(vals_opto)] - vals_opto
                        # vals_diff = [
                        #     t if h_c >= min_trial_count else None
                        #     for t, h_c in zip(temp, hit_count_opto)
                        # ]
                    # else:
                    #     vals_diff = temp
                    # vals_diff = [
                    #     v * polarity if v is not None else v for v in vals_diff
                    # ]
                    vals_diff = polarity * temp
                    diff_values[o].append(vals_diff)
                    trial_count_values[o].append(_count_vals)
                    animalids[o].append(filt_df[0, "animalid"])
                    x_pos_values.append(paired_idx)

        for k, v in diff_values.items():
            data_mat = np.array(v)
            count_mat = np.array(trial_count_values[k])
            data_mat_x = np.zeros_like(data_mat)
            for j in range(data_mat.shape[1]):
                if not np.all(np.isnan(data_mat[:, j])):
                    cloud_pos = x_pos_values[0][j] + pos_ofs * kwargs.get("cloud_offset", 0.25)

                    x_points = make_dot_cloud(
                        data_mat[:, j],
                        cloud_pos,
                        bin_width=kwargs.get("bin_width", 0.05),
                        width=kwargs.get("cloud_width", 0.2),
                    )

                    # add a very little random jitter
                    jit = np.random.randn(len(x_points)) / 25
                    x_points += jit
                    data_mat_x[:, j] = x_points
                    y_vals = data_mat[:, j]
                    if effect_on == "hit_rate":
                        y_vals = 100 * y_vals

                    if trial_count_identifier == "dot_color":
                        ax._scatter(
                            x_points,
                            y_vals,
                            s=50,
                            c=count_mat[:, j],
                            zorder=1,
                            mpl_kwargs=mpl_kwargs,
                            cmap=cmap,
                            edgecolors=clr.stim_keys[f"{stim_df[0, 'stim_type']}_{int(k)}"][
                                "color"
                            ],
                            norm=normalizer,
                        )
                    elif trial_count_identifier == "dot_size":
                        ax._scatter(
                            x_points,
                            y_vals,
                            c=clr.stim_keys[f"{stim_df[0, 'stim_type']}_{int(k)}"]["color"],
                            zorder=1,
                            mpl_kwargs=mpl_kwargs,
                        )

                    # check for normality and select the test accordingly
                    if p_test == "auto":
                        _is_norm = stats.shapiro(data_mat[:, j])
                        if _is_norm.pvalue < 0.05:  # is normal
                            print(
                                f"p1={_is_norm.pvalue:4}, data appears normal, doing paired t-test"
                            )
                            p_test = "t_test"
                        else:
                            print(
                                f"p1={_is_norm.pvalue:4}, data doesn't appear to normal, doing wilcoxon test"
                            )
                            p_test = "wilcoxon"

                    # apply statistical test
                    if p_test == "wilcoxon":
                        res = stats.wilcoxon(data_mat[:, j], nan_policy="omit")
                    elif p_test == "t_test":
                        res = stats.ttest_1samp(data_mat[:, j], 0, nan_policy="omit")

                    p = res.pvalue
                    print(stim_type, res, flush=True)
                    ax = pval_plotter(
                        ax,
                        p,
                        [cloud_pos, cloud_pos],
                        loc=polarity * 100,
                        color=clr.stim_keys[f"{stim_df[0, 'stim_type']}_{int(k)}"]["color"],
                    )

            if plot_with == "conf":
                means, ci_plus, ci_neg = np.apply_along_axis(
                    bootstrap_confidence_interval,
                    axis=0,
                    arr=data_mat,
                    statistic=np.nanmean,
                )
            elif plot_with == "sem":
                means = np.nanmean(data_mat, axis=0)
                ci_plus = stats.sem(data_mat, axis=0, nan_policy="omit")
                ci_neg = stats.sem(data_mat, axis=0, nan_policy="omit")
            elif plot_with == "iqr":
                means = np.nanmean(data_mat, axis=0)
                ci_plus = stats.iqr(data_mat, axis=0, nan_policy="omit")
                ci_neg = stats.iqr(data_mat, axis=0, nan_policy="omit")

            if effect_on == "hit_rate":
                means *= 100
                ci_plus *= 100
                ci_neg *= 100

            # shaded 95% CI
            ax.errorbar(
                np.unique(_lin_axis),
                means,
                yerr=(ci_plus, ci_neg),
                color=clr.stim_keys[f"{stim_df[0, 'stim_type']}_{int(k)}"]["color"],
                alpha=0.3,
                zorder=2,
                linewidth=0,
                elinewidth=10,
                markersize=0,
            )

            # means
            ax.scatter(
                np.unique(_lin_axis),
                means,
                s=(plt.rcParams["lines.markersize"] ** 2),
                c=clr.stim_keys[f"{stim_df[0, 'stim_type']}_{int(k)}"]["color"],
                marker="_",
                linewidths=3,
                edgecolors="w",
                zorder=3,
            )

            # plotting connecting lines
            for i in range(data_mat_x.shape[0]):
                y_vals = data_mat[i, :]
                if effect_on == "hit_rate":
                    y_vals = 100 * y_vals
                ax.plot(
                    data_mat_x[i, :],
                    y_vals,
                    linewidth=2,
                    alpha=0.2,
                    zorder=1,
                    # c=clr.stim_keys[f"{stim_df[0, 'stim_type']}_{int(k)}"]["color"],
                    c=ANIMAL_COLORS[animalids[k][i]],
                    marker="o",
                    mfc=ANIMAL_COLORS[animalids[k][i]],
                )

    _x_axis = np.array([0, 12.5, 50])
    _x_axis_pos = np.mean(np.array(x_pos_values), axis=0, dtype=int)
    ax.set_xticks(_x_axis_pos)
    ax.set_xticklabels(_x_axis[_x_axis_pos])

    if "time" in effect_on:
        # ax.set_yscale("symlog",linthresh=500)
        # ax.set_yscale('asinh',linear_width=200, base=10)
        # major_ticks = [0]
        # ax.set_yticks(major_ticks)

        # minor_locator = plt.LogLocator(base=10.0, subs=np.arange(200, 1000,200), numticks=5)
        # ax.yaxis.set_minor_locator(minor_locator)

        # negative_minor_ticks = [-t for t in minor_locator.tick_values(10, 1000) if t > 0]  # Mirror positive minor ticks
        # all_minor_ticks = sorted(negative_minor_ticks + minor_locator.tick_values(10, 1000).tolist())  # Combine
        # ax.set_yticks(all_minor_ticks, minor=True)  # Apply minor ticks symmetrically

        # ax.yaxis.set_minor_formatter(plt.FormatStrFormatter("%d"))
        # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
        # ax.set_ylim([-400,600])
        _y_ = [-600, -400, -200, 0, 200, 400, 600]
        y_ticks = [_t * polarity for _t in _y_]
        ax.set_yticks(y_ticks)
        ax.set_ylim([min(y_ticks) - 10, max(y_ticks) + 10])

    else:
        _y_ = [0, 25, 50, 75, 100]
        y_ticks = [_t * polarity for _t in _y_]
        ax.set_yticks(y_ticks)
        ax.set_ylim([min(y_ticks) - 10, max(y_ticks) + 10])
    ax.set_xlabel("Stimulus contrast (%)")
    ylab = " ".join(effect_on.split("_")).capitalize()
    ax.set_ylabel(rf"$\Delta${ylab} (%)")

    # add colorbar
    cbar_ax = fig.add_axes([0.95, 0, 0.05, 0.8])
    fig.colorbar(im, cax=cbar_ax)

    return fig, ax


# TODO: axis positioning is a bit hardcoded
def plot_delta_effect_stimtype(
    data: pl.DataFrame,
    effect_on: Literal["hit_rate", "reaction", "response"],
    contrast: float,
    do_contrast_type: bool = False,
    effect_metric: Literal["delta", "BSI", "BSI_base"] = "delta",
    plot_with: Literal["sem", "conf", "iqr"] = "sem",
    p_test: Literal["wilcoxon", "paired_t", "auto"] = "auto",
    ax: plt.Axes | None = None,
    include_misses: bool = False,
    polarity: Literal[-1, 1] = 1,
    trial_count_identifier: Literal["dot_color", "dot_size"] = "dot_color",
    mpl_kwargs: dict | None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Plots a connected scatter plot of the delta values between opto and non-opto conditions for different stimulus types.
    Each connected dot-pair corresponds to a session and the delta in different stimulus presentation conditions

    Args:
        data (pl.DataFrame): Experimental data
        effect_on (Literal["hit_rate","reaction","response"]): The behavioral readout to plot.
        plot_with (Literal["sem","conf","iqr"], optional): The distibution measurement. Defaults to "sem"
        ax (plt.Axes | None, optional): Precreated axis to plot to. Defaults to None.
        include_misses (bool, optional): Whether or not to include miss trials. Defaults to False. Doesn't make sense if effect_on is "hit_rate".
        polarity (Literal[-1,1], optional): Polarity of subtraction. 1 is nonopto - opto, -1 is opto - nonopto. Defaults to 1.
        mpl_kwargs (dict | None, optional): kwargs for styling matplotlib plots. Defaults to None.

    Returns:
        tuple[plt.Figure, plt.Axes]: Figure and axes of plot
    """
    if effect_on != "hit_rate":
        if not include_misses:
            effect_on = f"hit_{effect_on}"
        effect_on = f"median_{effect_on}_times"
        bin_width = kwargs.get("bin_width", 50)
    else:
        bin_width = kwargs.get("bin_width", 0.05)

    clr = Color(task="detection")
    override_plots()
    set_style(kwargs.get("style", "presentation"))
    if mpl_kwargs is None:
        mpl_kwargs = {}
    polarity = float(polarity)
    if ax is None:
        fig = plt.figure(figsize=mpl_kwargs.pop("figsize", (8, 8)))
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = ax.get_figure()

    # colormap stuff
    cmap_name = kwargs.get("dot_cmap", "Greens")
    cmap = plt.get_cmap(cmap_name)

    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # force the first color entry to be grey
    cmaplist[0] = (0.5, 0.5, 0.5, 1.0)

    trial_bounds = kwargs.get("trial_bounds", (0, 1, 5, 20, 50, 100))
    # create the new map
    cmap = LinearSegmentedColormap.from_list("Custom cmap", cmaplist, len(trial_bounds))
    normalizer = BoundaryNorm(trial_bounds, len(trial_bounds))
    im = cm.ScalarMappable(norm=normalizer, cmap=cmap)

    aggregator = WheelDetectionGroupedAggregator()
    aggregator.set_data(data=data)
    if do_contrast_type:
        _cont = "contrast_type"
    else:
        _cont = "contrast"
    aggregator.group_data(
        group_by=[
            "animalid",
            "session_id",
            "stim_type",
            "stim_side",
            _cont,
            "opto_pattern",
        ]
    )
    aggregator.calculate_hit_rates()
    aggregator.calculate_opto_pvalues()

    plot_data = aggregator.grouped_data.drop_nulls(_cont).filter(pl.col("stim_side") != "ipsi")

    if do_contrast_type:
        if contrast == 0.5:
            plot_data = plot_data.filter(pl.col("contrast_type") == "easy")
        elif contrast == 0.125:
            plot_data = plot_data.filter(pl.col("contrast_type") == "hard")
    else:
        plot_data = plot_data.filter(pl.col("contrast") == contrast)

    diff_values = np.zeros(
        (
            plot_data["session_id"].n_unique(),
            plot_data["stim_type"].n_unique(),
            plot_data["opto_pattern"].n_unique() - 1,
        )
    )
    diff_values[:] = np.nan

    trial_count_values = np.zeros_like(diff_values)
    trial_count_values[:] = np.nan
    for filt_tup in make_subsets(plot_data, ["session_id"], start_enumerate=0):
        filt_df = filt_tup[-1]
        # session_baseline_hr = get_baseline_hr(filt_df)
        if not filt_df.is_empty():
            q = (
                filt_df.group_by(["opto_pattern", "stim_type"])
                .agg(
                    pl.col(effect_on).get(0),
                    pl.col("count").get(0),
                    pl.col("hit_count").get(0),
                )
                .sort(["opto_pattern", "stim_type"])
            )

            vals_nonopto = q.filter(pl.col("opto_pattern") == -1)[effect_on].to_numpy()
            stype_nonopto = q.filter(pl.col("opto_pattern") == -1)["stim_type"].to_list()

            opto_patterns = q["opto_pattern"].sort().unique().to_numpy()
            opto_patterns = [o for o in opto_patterns if o != -1]
            for o in opto_patterns:
                vals_opto = q.filter(pl.col("opto_pattern") == o)[effect_on].to_numpy()
                stype_opto = q.filter(pl.col("opto_pattern") == 0)["stim_type"].to_list()

                hit_count_opto = q.filter(pl.col("opto_pattern") == o)["hit_count"].to_numpy()

                trial_count_opto = q.filter(pl.col("opto_pattern") == o)["count"].to_numpy()

                e1 = np.zeros_like(vals_nonopto)
                e1[:] = np.nan

                paired_idx = [i for i, c in enumerate(stype_nonopto) if c in stype_opto]

                e1[paired_idx] = vals_opto[paired_idx]

                if effect_on == "hit_rate":
                    if effect_metric == "delta":
                        # simple delta
                        temp = vals_nonopto - e1
                    elif effect_metric == "BSI":
                        # BSI: (nonopto - opto) / nonopto
                        # what percentage is delta hit rates of nonopto
                        _top = vals_nonopto - e1
                        temp = lenient_div(_top, vals_nonopto)

                    # elif effect_metric == "BSI_base":
                    #     # BSI_base: (|nonopto - baseline| - |opto-baseline|) / (|nonopto - baseline|)
                    #     # with baseline normalization
                    #     _opto = np.abs(e1 - session_baseline_hr)
                    #     _nonopto = np.abs(vals_nonopto - session_baseline_hr)
                    #     _top = _nonopto - _opto
                    #     temp = lenient_div(_top, _nonopto)

                    _count_vals = trial_count_opto

                if "time" in effect_on:
                    _count_vals = hit_count_opto
                    temp = vals_nonopto - e1

                vals_diff = polarity * temp
                print(filt_tup[1], vals_diff)
                diff_values[filt_tup[0], :, o] = vals_diff
                trial_count_values[filt_tup[0], :, o] = _count_vals

    # colormap stuff(rounding to nearest 5 base)
    # cmap = kwargs.get("dot_cmap", "viridis")
    # normalizer = Normalize(5*round(np.min(trial_count_values)/5),
    #                        5*round(np.max(trial_count_values)/5))
    # im = cm.ScalarMappable(norm=normalizer, cmap=cmap)

    s_type = ["0.04cpd_8.0Hz", "0.16cpd_0.5Hz"]
    x_pos_values = [0, 1]
    for k in range(diff_values.shape[2]):
        data_mat = diff_values[:, :, k]
        count_mat = trial_count_values[:, :, k]
        data_mat_x = np.zeros_like(data_mat)
        for j in range(data_mat.shape[1]):
            _st = [s_type[j] for _ in data_mat[:, j]]
            cloud_pos = x_pos_values[j]
            colors = [clr.stim_keys[f"{s}_{int(o)}"]["color"] for s in _st]

            x_points = make_dot_cloud(
                data_mat[:, j],
                cloud_pos,
                bin_width=bin_width,
                width=kwargs.get("cloud_width", 0.05),
            )
            data_mat_x[:, j] = x_points
            y_vals = data_mat[:, j]
            if effect_on == "hit_rate":
                y_vals = 100 * y_vals

            if trial_count_identifier == "dot_color":
                ax._scatter(
                    x_points,
                    y_vals,
                    s=50,
                    c=count_mat[:, j],
                    zorder=2,
                    mpl_kwargs=mpl_kwargs,
                    cmap=cmap,
                    norm=normalizer,
                )
            elif trial_count_identifier == "dot_size":
                ax._scatter(
                    x_points,
                    y_vals,
                    c=colors,
                    zorder=2,
                    mpl_kwargs=mpl_kwargs,
                )

            for p1, p2 in list(itertools.combinations([x for x in range(data_mat.shape[1])], 2)):
                d1 = data_mat[:, p1]
                d2 = data_mat[:, p2]

                if p_test == "auto":
                    _is_norm1 = stats.shapiro(d1)
                    _is_norm2 = stats.shapiro(d2)
                    if _is_norm1.pvalue < 0.05 and _is_norm2.pvalue < 0.05:  # is normal
                        print(
                            f"p1={_is_norm1.pvalue:4} and p2={_is_norm2.pvalue}, data appears normal, doing paired t-test"
                        )
                        p_test = "paired_t"
                    else:
                        print(
                            f"p1={_is_norm1.pvalue:4} and p2={_is_norm2.pvalue}, data doesn't appear to normal, doing wilcoxon test"
                        )
                        p_test = "wilcoxon"

                if p_test == "wilcoxon":
                    res = stats.wilcoxon(d1, d2, nan_policy="omit")
                elif p_test == "paired_t":
                    res = stats.ttest_rel(d1, d2, nan_policy="omit")
                p = res.pvalue
                print(p)
                ax = pval_plotter(ax, p, [p1, p2], loc=polarity * 100)

        # plot the means and shaded regions
        if plot_with == "conf":
            means, ci_plus, ci_neg = np.apply_along_axis(
                bootstrap_confidence_interval,
                axis=0,
                arr=diff_values,
                statistic=np.nanmean,
            )
        elif plot_with == "sem":
            means = np.nanmean(data_mat, axis=0)
            ci_plus = stats.sem(data_mat, axis=0, nan_policy="omit")
            ci_neg = stats.sem(data_mat, axis=0, nan_policy="omit")
        elif plot_with == "iqr":
            means = np.nanmean(data_mat, axis=0)
            ci_plus = stats.iqr(data_mat, axis=0, nan_policy="omit")
            ci_neg = stats.iqr(data_mat, axis=0, nan_policy="omit")

        if effect_on == "hit_rate":
            means *= 100
            ci_plus *= 100
            ci_neg *= 100

        # shaded 95% CI
        ax.errorbar(
            np.unique(x_pos_values),
            means,
            yerr=(ci_plus, ci_neg),
            # color=clr.stim_keys[f"{stim_df[0,'stim_type']}_{int(o)}"]["color"],
            alpha=0.3,
            zorder=2,
            linewidth=0,
            elinewidth=10,
            markersize=0,
        )

        # # means
        ax.scatter(
            np.unique(x_pos_values),
            means,
            s=(plt.rcParams["lines.markersize"] ** 2),
            # c = clr.stim_keys[f"{stim_df[0,'stim_type']}_{int(o)}"]["color"],
            marker="_",
            linewidths=3,
            edgecolors="w",
            zorder=3,
        )

        for i in range(data_mat_x.shape[0]):
            y_vals = data_mat[i, :]
            if effect_on == "hit_rate":
                y_vals = 100 * y_vals
            ax.plot(data_mat_x[i, :], y_vals, linewidth=2, alpha=0.2, c="#bfbfbf", zorder=0)

    ax.set_xticks(x_pos_values)
    ax.set_xticklabels([X_AXIS_LABEL_STYPE[i] for i in x_pos_values])

    if "time" in effect_on:
        _y_ = np.arange(-200, 800, 200)
        y_ticks = [_t * -1 * polarity for _t in _y_]
    else:
        _y_ = [0, 25, 50, 75, 100]
        y_ticks = [_t * polarity for _t in _y_]

    ax.set_yticks(y_ticks)
    ax.set_ylim([min(y_ticks) - 10, max(y_ticks) + 10])
    ax.set_xlabel("Stimulus contrast (%)")
    ylab = " ".join(effect_on.split("_")).capitalize()
    ax.set_ylabel(rf"$\Delta${ylab} (%)")

    # add colorbar
    cbar_ax = fig.add_axes([0.95, 0, 0.05, 0.8])
    fig.colorbar(im, cax=cbar_ax)

    return fig, ax


def plot_delta_effect_areas(
    data: pl.DataFrame,
    effect_on: Literal["hit_rate", "reaction", "response"],
    contrast: float,
    areas: list[str] = [
        "V1",
        "HVA",
        "dorsal",
        "ventralPM",
        "LM",
        "AL",
        "RL",
        "PM",
        "AM",
    ],
    stim_type: Literal["0.04cpd_8.0Hz", "0.16cpd_0.5Hz"] = "0.4cpd_8.0Hz",
    effect_metric: Literal["delta", "BSI", "BSI_base"] = "delta",
    plot_with: Literal["sem", "conf", "iqr"] = "sem",
    p_test: Literal["wilcoxon", "paired_t", "auto"] = "auto",
    ax: plt.Axes | None = None,
    include_misses: bool = False,
    trial_count_identifier: Literal["dot_color", "dot_size"] = "dot_color",
    polarity: Literal[-1, 1] = 1,
    mpl_kwargs: dict | None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Plots a connected scatter plot of the delta values between opto and non-opto conditions for different areas.
    Each connected dot-sequence corresponds to an animal (one session)

    Args:
        data (pl.DataFrame): Experimental data
        effect_on (Literal["hit_rate","reaction","response"]): The behavioral readout to plot.
        contrast (float): contrast value to plot
        areas (list[str]): list of areas to be plotted(in that order)
        plot_with (Literal["sem","conf","iqr"], optional): The distibution measurement. Defaults to "sem"
        ax (plt.Axes | None, optional): Precreated axis to plot to. Defaults to None.
        include_misses (bool, optional): Whether or not to include miss trials. Defaults to False. Doesn't make sense if effect_on is "hit_rate".
        polarity (Literal[-1,1], optional): Polarity of subtraction. 1 is nonopto - opto, -1 is opto - nonopto. Defaults to 1.
        mpl_kwargs (dict | None, optional): kwargs for styling matplotlib plots. Defaults to None.

    Returns:
        tuple[plt.Figure, plt.Axes]: Figure and axes of plot
    """

    if effect_on != "hit_rate":
        if not include_misses:
            effect_on = f"hit_{effect_on}"
        effect_on = f"median_{effect_on}_times"
        bin_width = kwargs.get("bin_width", 50)
    else:
        bin_width = kwargs.get("bin_width", 0.05)

    override_plots()
    set_style(kwargs.get("style", "presentation"))
    if mpl_kwargs is None:
        mpl_kwargs = {}
    polarity = float(polarity)
    if ax is None:
        fig = plt.figure(figsize=mpl_kwargs.pop("figsize", (8, 8)))
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = ax.get_figure()

    # colormap stuff
    cmap_name = kwargs.get("dot_cmap", "Greens")
    cmap = plt.get_cmap(cmap_name)

    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # force the first color entry to be grey
    cmaplist[0] = (0.5, 0.5, 0.5, 1.0)

    trial_bounds = kwargs.get("trial_bounds", (0, 1, 5, 20, 50, 100))
    # create the new map
    cmap = LinearSegmentedColormap.from_list("Custom cmap", cmaplist, len(trial_bounds))
    normalizer = BoundaryNorm(trial_bounds, len(trial_bounds))
    im = cm.ScalarMappable(norm=normalizer, cmap=cmap)

    aggregator = WheelDetectionGroupedAggregator()
    aggregator.set_data(data=data)
    aggregator.group_data(
        group_by=[
            "animalid",
            "area",
            "stim_type",
            "stim_side",
            "contrast",
            "opto_pattern",
        ]
    )
    aggregator.calculate_hit_rates()
    aggregator.calculate_opto_pvalues()

    # manually add baselines by joining
    catch_trials = aggregator.grouped_data.filter(
        (pl.col("contrast") == 0) & (pl.col("opto_pattern") == -1)
    ).select(["animalid", "area", "stim_type", "contrast", "opto_pattern", "hit_count", "count"])
    # get baseline_hr
    catch_trials = catch_trials.with_columns(
        (pl.col("hit_count").sum() / pl.col("count").sum()).alias("baseline_hr")
    )

    _data = aggregator.grouped_data.join(
        catch_trials.select(["animalid", "area", "stim_type", "baseline_hr"]),
        how="inner",
        left_on=["animalid", "area", "stim_type"],
        right_on=["animalid", "area", "stim_type"],
    )

    plot_data = _data.drop_nulls("contrast").filter(pl.col("stim_side") != "ipsi")
    plot_data = plot_data.filter(pl.col("contrast") == contrast)
    plot_data = plot_data.filter(pl.col("stim_type") == stim_type)
    plot_data = plot_data.filter(pl.col("area").is_in(areas))

    n_animals = plot_data.n_unique("animalid")

    diff_mat = np.zeros((n_animals, len(areas)))
    diff_mat[:] = np.nan
    count_mat = np.zeros_like(diff_mat)
    count_mat[:] = np.nan
    animal_ids = []

    for filt_tup in make_subsets(plot_data, ["animalid"], start_enumerate=0):
        filt_df = filt_tup[-1]
        a_id = filt_tup[1]
        row_idx = filt_tup[0]
        if not filt_df.is_empty():
            # order the areas in given order
            with pl.StringCache():
                pl.Series(areas).cast(pl.Categorical)
                filt_df = filt_df.with_columns(pl.col("area").cast(pl.Categorical))
                filt_df = filt_df.sort("area")

            q = (
                filt_df.group_by(["opto_pattern", "area"])
                .agg(
                    pl.col(effect_on).get(0),
                    pl.col("count").get(0),
                    pl.col("hit_count").get(0),
                    pl.col("baseline_hr").get(0),
                )
                .sort(["opto_pattern", "area"])
            )

            diff_vals = np.zeros((len(areas)))
            count_vals = np.zeros_like(diff_vals)
            diff_vals[:] = np.nan
            count_vals[:] = np.nan
            # loop for areas because there might be missing sessions for some animals
            for j, a in enumerate(areas):
                _area = q.filter(pl.col("area") == a)

                if not _area.is_empty():
                    _nonopto = _area.filter((pl.col("opto_pattern") == -1))[0, effect_on]
                    _opto = _area.filter(pl.col("opto_pattern") == 0)[0, effect_on]

                    _opto_hit_count = _area.filter(pl.col("opto_pattern") == 0)[0, "hit_count"]

                    _opto_trial_count = _area.filter(pl.col("opto_pattern") == 0)[0, "count"]

                    _baseline = _area.filter(pl.col("opto_pattern") == 0)[0, "baseline_hr"]

                    if effect_on == "hit_rate":
                        if effect_metric == "delta":
                            # simple delta
                            temp = _nonopto - _opto
                        elif effect_metric == "BSI":
                            # BSI: (nonopto - opto) / nonopto
                            # what percentage is delta hit rates of nonopto
                            _top = _nonopto - _opto
                            temp = lenient_div(_top, _nonopto)

                        elif effect_metric == "BSI_base":
                            # BSI_base: (|nonopto - baseline| - |opto-baseline|) / (|nonopto - baseline|)
                            # with baseline normalization
                            _base_opto = np.abs(_opto - _baseline)
                            _base_nonopto = np.abs(_nonopto - _baseline)
                            _top = _base_nonopto - _base_opto
                            temp = lenient_div(_top, _base_nonopto)

                        _count_vals = _opto_trial_count

                    if "time" in effect_on:
                        _count_vals = _opto_hit_count
                        if _opto is not None:
                            temp = _opto - _nonopto
                        else:
                            temp = np.nan

                    diff_vals[j] = polarity * temp
                    count_vals[j] = _count_vals

            diff_mat[row_idx, :] = diff_vals
            count_mat[row_idx, :] = count_vals
            animal_ids.append(a_id)
    x_axis = np.arange(len(areas))

    # plot the scatters
    clouded_x_values = np.zeros_like(diff_mat)
    clouded_x_values[:] = np.nan
    uncorrected_p_vals = []
    for c in range(diff_mat.shape[1]):
        area_vals = diff_mat[:, c]
        area_counts = count_mat[:, c]
        x_points = make_dot_cloud(
            area_vals,
            x_axis[c],
            bin_width=bin_width,
            width=kwargs.get("cloud_width", 0.05),
        )
        clouded_x_values[:, c] = x_points

        if effect_on == "hit_rate":
            area_vals = 100 * area_vals

        if trial_count_identifier == "dot_color":
            ax._scatter(
                x_points,
                area_vals,
                s=50,
                c=area_counts,
                zorder=3,
                mpl_kwargs=mpl_kwargs,
                cmap=cmap,
                norm=normalizer,
            )
        elif trial_count_identifier == "dot_size":
            ax._scatter(
                x_points,
                area_vals,
                c="#C7C7C7",
                zorder=3,
                mpl_kwargs=mpl_kwargs,
            )

        # check for normality and select the test accordingly
        if p_test == "auto":
            _is_norm = stats.shapiro(area_vals)
            if _is_norm.pvalue < 0.05:  # is normal
                print(f"p1={_is_norm.pvalue:4}, data appears normal, doing paired t-test")
                p_test = "t_test"
            else:
                print(f"p1={_is_norm.pvalue:4}, data doesn't appear to normal, doing wilcoxon test")
                p_test = "wilcoxon"

        # apply statistical test
        if p_test == "wilcoxon":
            res = stats.wilcoxon(area_vals, nan_policy="omit")
        elif p_test == "t_test":
            res = stats.ttest_1samp(area_vals, 0, nan_policy="omit")

        p = res.pvalue
        print(areas[c], res, flush=True)
        uncorrected_p_vals.append(p)

    # plot p in between
    _, corrected_p_vals, _, _ = multipletests(uncorrected_p_vals, 0.05, "holm")
    for ii, p_corrected in enumerate(corrected_p_vals):
        print(p_corrected)
        ax = pval_plotter(
            ax,
            p_corrected,
            [x_axis[ii], x_axis[ii]],
            loc=polarity * 100,
            color="k",
        )

    uncorrected_p_vals2 = []
    for p_idx1 in range(diff_mat.shape[1] - 1):
        for p_idx2 in range(p_idx1 + 1, diff_mat.shape[1]):
            area_1 = diff_mat[:, p_idx1]
            area_2 = diff_mat[:, p_idx2]
            res_compare = stats.wilcoxon(area_1, area_2, nan_policy="omit")
            print(f"{areas[p_idx1]}-{areas[p_idx2]}: {res_compare}", flush=True)
            uncorrected_p_vals2.append([p_idx1, p_idx2, res_compare.pvalue])

    uncorrected_p_vals2 = np.array(uncorrected_p_vals2)
    _, corrected_p_vals2, _, _ = multipletests(uncorrected_p_vals2[:, 2], 0.05, "holm")
    for ii, p_corrected2 in enumerate(corrected_p_vals2):
        print(p_corrected2)
        ax = pval_plotter(
            ax,
            p_corrected2,
            [
                x_axis[int(uncorrected_p_vals2[ii, 0])],
                x_axis[int(uncorrected_p_vals2[ii, 1])],
            ],
            loc=polarity * 100,
            color="k",
        )

    # plot the connected lines
    for i, r in enumerate(diff_mat):
        x_vals = clouded_x_values[i, :]

        if effect_on == "hit_rate":
            r = 100 * r

        ax._plot(
            x_vals,
            r,
            zorder=1,
            markersize=0,
            c=kwargs.get("color", ANIMAL_COLORS[animal_ids[i]]),
            mpl_kwargs=mpl_kwargs,
        )

    # plot the means and shaded regions
    if plot_with == "conf":
        means, ci_plus, ci_neg = np.apply_along_axis(
            bootstrap_confidence_interval, axis=0, arr=diff_mat, statistic=np.nanmean
        )
    elif plot_with == "sem":
        means = np.nanmean(diff_mat, axis=0)
        ci_plus = stats.sem(diff_mat, axis=0, nan_policy="omit")
        ci_neg = stats.sem(diff_mat, axis=0, nan_policy="omit")
    elif plot_with == "iqr":
        means = np.nanmean(diff_mat, axis=0)
        ci_plus = stats.iqr(diff_mat, axis=0, nan_policy="omit")
        ci_neg = stats.iqr(diff_mat, axis=0, nan_policy="omit")

    if effect_on == "hit_rate":
        means *= 100
        ci_plus *= 100
        ci_neg *= 100

    ax.scatter(
        x_axis,
        means,
        s=(plt.rcParams["lines.markersize"] ** 2),
        c="k",
        marker="_",
        linewidths=3,
        edgecolors="w",
        zorder=3,
    )

    # shaded 95% CI
    ax.errorbar(
        x_axis,
        means,
        yerr=(ci_plus, ci_neg),
        color="k",
        alpha=0.3,
        zorder=2,
        linewidth=0,
        elinewidth=10,
        markersize=0,
    )

    ax.set_xticks(x_axis)
    ax.set_xticklabels(areas)

    if "time" in effect_on:
        # ax.set_yscale("symlog",linthresh=500)
        # # ax.set_yscale('asinh',linear_width=200, base=10)
        # major_ticks = [-100,0,100]
        # ax.set_yticks(major_ticks)

        # minor_locator = plt.LogLocator(base=10.0, subs=np.arange(50, 500,200), numticks=5)
        # ax.yaxis.set_minor_locator(minor_locator)

        # negative_minor_ticks = [-t for t in minor_locator.tick_values(10, 5000) if t > 0]  # Mirror positive minor ticks
        # all_minor_ticks = sorted(negative_minor_ticks + minor_locator.tick_values(10, 1000).tolist())  # Combine
        # ax.set_yticks(all_minor_ticks, minor=True)  # Apply minor ticks symmetrically

        # ax.yaxis.set_minor_formatter(plt.FormatStrFormatter("%d"))
        # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
        # ax.set_ylim([-400,600])
        _y_ = [-600, -400, -200, 0, 200, 400, 600]
        y_ticks = [_t * polarity for _t in _y_]
        ax.set_yticks(y_ticks)
        ax.set_ylim([min(y_ticks) - 10, max(y_ticks) + 10])

    else:
        _y_ = [0, 25, 50, 75, 100]
        y_ticks = [_t * polarity for _t in _y_]
        ax.set_yticks(y_ticks)
        ax.set_ylim([min(y_ticks) - 10, max(y_ticks) + 10])
    ax.set_xlabel("Stimulus contrast (%)")
    ylab = " ".join(effect_on.split("_")).capitalize()
    if effect_metric == "BSI":
        ax.set_ylabel("SI")
    else:
        ax.set_ylabel(rf"$\Delta${ylab} (%)")

    # add colorbar
    cbar_ax = fig.add_axes([0.95, 0, 0.05, 0.8])
    fig.colorbar(im, cax=cbar_ax)

    return fig, ax


def plot_delta_effect_CNO(
    data: pl.DataFrame,
    effect_on: Literal["hit_rate", "reaction", "response"],
    contrast: float,
    stim_type: str,
    effect_metric: Literal["delta", "BSI", "BSI_base"] = "delta",
    plot_with: Literal["sem", "conf", "iqr"] = "sem",
    p_test: Literal["wilcoxon", "paired_t", "auto"] = "auto",
    ax: plt.Axes | None = None,
    include_misses: bool = False,
    trial_count_identifier: Literal["dot_color", "dot_size"] = "dot_color",
    polarity: Literal[-1, 1] = 1,
    mpl_kwargs: dict | None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Plots a connected scatter plot of the delta values between non_CNO and CNO conditions.
    Each connected dot-pair corresponds to two sessions of a given animal, without and with CNO, respectively

    Args:
        data (pl.DataFrame): Experimental data
        effect_on (Literal["hit_rate","reaction","response"]): The behavioral readout to plot.
        contrast (float): contrast value to plot
        stim_type (str): stimulus type to plot, e.g. 0.04cpd_8.0Hz
        plot_with (Literal["sem","conf","iqr"], optional): The distibution measurement. Defaults to "sem"
        ax (plt.Axes | None, optional): Precreated axis to plot to. Defaults to None.
        include_misses (bool, optional): Whether or not to include miss trials. Defaults to False. Doesn't make sense if effect_on is "hit_rate".
        polarity (Literal[-1,1], optional): Polarity of subtraction. 1 is nonopto - opto, -1 is opto - nonopto. Defaults to 1.
        mpl_kwargs (dict | None, optional): kwargs for styling matplotlib plots. Defaults to None.

    Returns:
        tuple[plt.Figure, plt.Axes]: Figure and axes of plot
    """
    if effect_on != "hit_rate":
        if not include_misses:
            effect_on = f"hit_{effect_on}"
        effect_on = f"median_{effect_on}_times"
        bin_width = kwargs.get("bin_width", 50)
    else:
        bin_width = kwargs.get("bin_width", 0.05)

    override_plots()
    set_style(kwargs.get("style", "presentation"))
    if mpl_kwargs is None:
        mpl_kwargs = {}
    polarity = float(polarity)
    if ax is None:
        fig = plt.figure(figsize=mpl_kwargs.pop("figsize", (8, 8)))
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = ax.get_figure()

    # colormap stuff
    cmap_name = kwargs.get("dot_cmap", "Greens")
    cmap = plt.get_cmap(cmap_name)

    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # force the first color entry to be grey
    cmaplist[0] = (0.5, 0.5, 0.5, 1.0)

    trial_bounds = kwargs.get("trial_bounds", (0, 1, 5, 20, 50, 100))
    # create the new map
    cmap = LinearSegmentedColormap.from_list("Custom cmap", cmaplist, len(trial_bounds))
    normalizer = BoundaryNorm(trial_bounds, len(trial_bounds))

    im = cm.ScalarMappable(norm=normalizer, cmap=cmap)

    aggregator = WheelDetectionGroupedAggregator()
    aggregator.set_data(data=data)
    aggregator.group_data(
        group_by=[
            "animalid",
            "area",
            "stim_type",
            "stim_side",
            "contrast",
            "opto_pattern",
            "isCNO",
        ]
    )
    aggregator.calculate_hit_rates()
    aggregator.calculate_opto_pvalues()

    plot_data = aggregator.grouped_data.drop_nulls("contrast").filter(pl.col("stim_side") != "ipsi")
    plot_data = plot_data.filter(
        (pl.col("contrast") == contrast) & (pl.col("stim_type") == stim_type)
    )

    diff_values = np.zeros((plot_data["animalid"].n_unique(), 2))  # nonCNO and CNO
    trial_counts = np.zeros_like(diff_values)

    diff_values[:] = np.nan
    trial_counts[:] = np.nan
    animal_ids = []
    for filt_tup in make_subsets(plot_data, ["animalid"], start_enumerate=0):
        filt_df = filt_tup[-1]
        a_id = filt_tup[1]
        if not filt_df.is_empty():
            q = (
                filt_df.group_by(["isCNO", "opto_pattern"])
                .agg(
                    [
                        pl.col(effect_on).get(0),
                        pl.col("count").get(0),
                        pl.col("hit_count").get(0),
                    ]
                )
                .sort(["isCNO", "opto_pattern"])
            )
            _noncno = q.filter(pl.col("isCNO") == 0).sort("opto_pattern")
            _cno = q.filter(pl.col("isCNO") == 1).sort("opto_pattern")
            if effect_on == "hit_rate":
                # trial counts
                _count_noncno = _noncno[0, "count"]
                _count_cno = _cno[0, "count"]
                if effect_metric == "delta":
                    # delta non_CNO
                    diff_noncno = _noncno[0, effect_on] - _noncno[1, effect_on]

                    # delta CNO
                    diff_cno = _cno[0, effect_on] - _cno[1, effect_on]

                elif effect_metric == "BSI":
                    # BSI: (nonopto - opto) / nonopto
                    # what percentage is delta hit rates of nonopto
                    # BSI non_CNO
                    _top = _noncno[0, effect_on] - _noncno[1, effect_on]
                    diff_noncno = lenient_div(_top, _noncno[0, effect_on])

                    # BSI CNO
                    _top = _cno[0, effect_on] - _cno[1, effect_on]
                    diff_cno = lenient_div(_top, _cno[0, effect_on])

            elif "time" in effect_on:
                # hit trial counts
                _count_noncno = _noncno[0, "hit_count"]
                _count_cno = _cno[0, "hit_count"]
                # delta non_CNO
                diff_noncno = _noncno[0, effect_on] - _noncno[1, effect_on]

                # delta CNO
                diff_cno = _cno[0, effect_on] - _cno[1, effect_on]

            diff_noncno = polarity * diff_noncno
            diff_cno = polarity * diff_cno
            diff_values[filt_tup[0], :] = [diff_noncno, diff_cno]
            trial_counts[filt_tup[0], :] = [_count_noncno, _count_noncno]
            animal_ids.append(a_id)

    x_axis = np.arange(diff_values.shape[1])

    # plot the scatters
    clouded_x_values = np.zeros_like(diff_values)
    clouded_x_values[:] = np.nan
    for c in range(diff_values.shape[1]):
        cno_vals = diff_values[:, c]
        trial_vals = trial_counts[:, c]
        x_points = make_dot_cloud(
            cno_vals,
            x_axis[c],
            bin_width=bin_width,
            width=kwargs.get("cloud_width", 0.05),
        )
        clouded_x_values[:, c] = x_points

        if effect_on == "hit_rate":
            cno_vals = 100 * cno_vals

        if trial_count_identifier == "dot_color":
            ax._scatter(
                x_points,
                cno_vals,
                s=50,
                c=trial_vals,
                zorder=3,
                mpl_kwargs=mpl_kwargs,
                cmap=cmap,
                norm=normalizer,
            )
        elif trial_count_identifier == "dot_size":
            ax._scatter(
                x_points,
                cno_vals,
                c="#C7C7C7",
                zorder=3,
                mpl_kwargs=mpl_kwargs,
            )

    # plot the pvasl
    for p1, p2 in list(itertools.combinations([x for x in range(diff_values.shape[1])], 2)):
        d1 = diff_values[:, p1]
        d2 = diff_values[:, p2]
        if p_test == "auto":
            _is_norm1 = stats.shapiro(d1)
            _is_norm2 = stats.shapiro(d2)
            if _is_norm1.pvalue < 0.05 and _is_norm2.pvalue < 0.05:  # is normal
                print(
                    f"p1={_is_norm1.pvalue:4} and p2={_is_norm2.pvalue}, data appears normal, doing paired t-test"
                )
                p_test = "t_test"
            else:
                print(
                    f"p1={_is_norm1.pvalue:4} and p2={_is_norm2.pvalue}, data doesn't appear to normal, doing wilcoxon test"
                )
                p_test = "wilcoxon"

        if p_test == "wilcoxon":
            res = stats.wilcoxon(d1, d2, nan_policy="omit")
        elif p_test == "paired_t":
            res = stats.ttest_rel(d1, d2, nan_policy="omit")
        p = res.pvalue
        ax = pval_plotter(ax, p, [p1, p2], loc=polarity * 100)

    # plot the connected lines
    for i, r in enumerate(diff_values):
        x_vals = clouded_x_values[i, :]

        if effect_on == "hit_rate":
            r = 100 * r

        ax._plot(
            x_vals,
            r,
            zorder=1,
            markersize=0,
            c=kwargs.get("color", ANIMAL_COLORS[animal_ids[i]]),
            mpl_kwargs=mpl_kwargs,
        )

    # plot the means and shaded regions
    if plot_with == "conf":
        means, ci_plus, ci_neg = np.apply_along_axis(
            bootstrap_confidence_interval, axis=0, arr=diff_values, statistic=np.nanmean
        )
    elif plot_with == "sem":
        means = np.nanmean(diff_values, axis=0)
        ci_plus = stats.sem(diff_values, axis=0, nan_policy="omit")
        ci_neg = stats.sem(diff_values, axis=0, nan_policy="omit")
    elif plot_with == "iqr":
        means = np.nanmean(diff_values, axis=0)
        ci_plus = stats.iqr(diff_values, axis=0, nan_policy="omit")
        ci_neg = stats.iqr(diff_values, axis=0, nan_policy="omit")

    if effect_on == "hit_rate":
        means *= 100
        ci_plus *= 100
        ci_neg *= 100

    ax.scatter(
        x_axis,
        means,
        s=(plt.rcParams["lines.markersize"] ** 2),
        c="k",
        marker="_",
        linewidths=3,
        edgecolors="w",
        zorder=3,
    )

    # shaded 95% CI
    ax.errorbar(
        x_axis,
        means,
        yerr=(ci_plus, ci_neg),
        color="k",
        alpha=0.3,
        zorder=2,
        linewidth=0,
        elinewidth=10,
        markersize=0,
    )

    ax.set_xticks(x_axis)
    ax.set_xticklabels([X_AXIS_LABEL_CNO[i] for i in x_axis])

    if "time" in effect_on:
        # ax.set_yscale("symlog",linthresh=500)
        # ax.set_yscale('asinh',linear_width=200, base=10)
        # major_ticks = [0]
        # ax.set_yticks(major_ticks)

        # minor_locator = plt.LogLocator(base=10.0, subs=np.arange(200, 1000,200), numticks=5)
        # ax.yaxis.set_minor_locator(minor_locator)

        # negative_minor_ticks = [-t for t in minor_locator.tick_values(10, 1000) if t > 0]  # Mirror positive minor ticks
        # all_minor_ticks = sorted(negative_minor_ticks + minor_locator.tick_values(10, 1000).tolist())  # Combine
        # ax.set_yticks(all_minor_ticks, minor=True)  # Apply minor ticks symmetrically

        # ax.yaxis.set_minor_formatter(plt.FormatStrFormatter("%d"))
        # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
        # ax.set_ylim([-400,600])
        _y_ = [-600, -400, -200, 0, 200, 400, 600]
        y_ticks = [_t * polarity for _t in _y_]
        ax.set_yticks(y_ticks)
        ax.set_ylim([min(y_ticks) - 10, max(y_ticks) + 10])

    else:
        _y_ = [0, 25, 50, 75, 100]
        y_ticks = [_t * polarity for _t in _y_]
        ax.set_yticks(y_ticks)
        ax.set_ylim([min(y_ticks) - 10, max(y_ticks) + 10])
    ax.set_xlabel("Stimulus contrast (%)")
    ylab = " ".join(effect_on.split("_")).capitalize()
    ax.set_ylabel(rf"$\Delta${ylab} (%)")

    # add colorbar
    cbar_ax = fig.add_axes([0.95, 0, 0.05, 0.8])
    fig.colorbar(im, cax=cbar_ax, spacing="uniform")  # uniform or proprtional

    return fig, ax
