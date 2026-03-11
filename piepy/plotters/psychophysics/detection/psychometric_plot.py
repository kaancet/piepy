import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from ...colors.color import Color
from ...plotting_utils import (
    set_style,
    make_linear_axis,
    make_label,
    override_plots,
    pval_plotter,
)
from ....psychophysics.fit_funcs import mle_fit, neg_likelihood, weibull, erf_psycho
from ....core.data_functions import make_subsets
from ....psychophysics.wheel.detection.wheelDetectionGroupedAggregator import (
    WheelDetectionGroupedAggregator,
)


def plot_psychometric(
    data: pl.DataFrame,
    ax: plt.Axes = None,
    make_fit: bool = False,
    combine_sides: bool = False,
    log_x: bool = False,
    baseline_normalize: bool = False,
    mpl_kwargs: dict | None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Plots the hit rates with 95% confidence intervals

    Args:
        data (pl.DataFrame): Data to be plotted, can be single or multiple sessions
        ax (plt.Axes, optional): An axes object to place to plot,default is None, which creates the axes
        make_fit (bool, optional): Boolean to eaither fit a weibull function or just connect the points. Defaults to False
        combine_sides (bool, optional): Combines the ipsi and contra trials
        baseline_normalize (bool, optional): Baseline normalization of hit rates, subtracts the baseline and normalizes the
        mpl_kwargs (dict | None, optional): kwargs for styling matplotlib plots. Defaults to None.
    Returns:
        tuple[plt.Figure,plt.Axes]: Plotted figure and axes objects
    """
    override_plots()
    set_style(kwargs.get("style", "presentation"))
    if mpl_kwargs is None:
        mpl_kwargs = {}

    if ax is None:
        fig = plt.figure(figsize=mpl_kwargs.pop("figsize", (8, 8)))
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = ax.get_figure()

    if combine_sides:
        _grouper = ["stim_type", "contrast", "opto_pattern", "isCatch"]
        _subsetter = ["stimkey"]
        _axer = "contrast"
    else:
        _grouper = ["stim_type", "stim_side", "contrast", "opto_pattern", "isCatch"]
        _subsetter = ["stimkey", "stim_side"]
        _axer = "signed_contrast"

    clr = Color(task="detection")
    analyzer = WheelDetectionGroupedAggregator()
    analyzer.set_data(data=data)
    analyzer.group_data(group_by=_grouper)
    analyzer.calculate_hit_rates()
    analyzer.calculate_baseline_norm_hit_rates()
    analyzer.calculate_opto_pvalues()

    nonearly_data = analyzer.grouped_data.drop_nulls("contrast")

    lin_axis_dict = make_linear_axis(nonearly_data, _axer)
    _lin_axis = [float(lin_axis_dict[c]) if c is not None else None for c in nonearly_data[_axer].to_list()]
    nonearly_data = nonearly_data.with_columns(pl.Series("linear_axis", _lin_axis))
    catch_data = nonearly_data.filter(pl.col("isCatch"))
    noncatch_data = nonearly_data.filter(~pl.col("isCatch"))

    for filt_tup in make_subsets(noncatch_data, _subsetter, start_enumerate=0):
        i = filt_tup[0]
        filt_df = filt_tup[-1]
        filt_key = filt_tup[1]

        if not filt_df.is_empty():
            contrast_label = filt_df[_axer].to_numpy() * 100
            lin_ax = filt_df["linear_axis"].to_numpy()
            xax = contrast_label if log_x else lin_ax

            confs = np.round(100 * filt_df["hit_rate_confs"].to_numpy().transpose(), 3)
            count = filt_df["count"].to_numpy()
            if baseline_normalize:
                hr = np.round(100 * filt_df["base_norm_hit_rate"].to_numpy().flatten(), 3)
            else:
                hr = np.round(100 * filt_df["hit_rate"].to_numpy().flatten(), 3)

            stim_label = filt_df["stim_label"].unique().to_numpy()
            p_val = filt_df["p_hit_rate"].to_numpy()

            if make_fit:
                _fit_data = np.vstack((xax, count, hr / 100))
                params, likelihood = mle_fit(
                    _fit_data,
                    P_model="weibull",
                    side="contra" if combine_sides else filt_tup[2],
                    nfits=10,
                )
                if log_x:
                    #TODO: FIX THIS SHIT
                    x_fit1 = np.linspace(1, xax[0], 50)
                    x_fit2 = np.linspace(xax[0], xax[-1], 50)
                    x_fit = np.concatenate((x_fit1, x_fit2[1:]))
                else:
                    x_fit = np.linspace(0, xax[np.argmax(np.abs(xax))], 100)
                y_fit = weibull(params, x_fit)
                ax._plot(
                    x_fit,
                    y_fit * 100,
                    color=clr.stim_keys[filt_key]["color"],
                    linewidth=plt.rcParams["lines.linewidth"],
                    mpl_kwargs=mpl_kwargs,
                )

                # mark the 50
                _threshold = x_fit[np.argmin(np.abs(y_fit - 0.5))]
                if not combine_sides and filt_tup[2] == "ipsi":
                    # np.interp needs increasing order
                    _contrast_val = np.interp(_threshold, xax[::-1], contrast_label[::-1])
                else:
                    _contrast_val = np.interp(_threshold, xax, contrast_label)
                ax.plot([_threshold, _threshold], [0, 50], "k", linewidth=0.5, linestyle=":")
                ax.text(_threshold + 0.05, 50, round(_contrast_val, 3))

            ax._errorbar(
                x=xax,
                y=hr,
                yerr=confs,
                marker="o",
                label=f"{stim_label[0]}{make_label(contrast_label, count)}",
                color=clr.stim_keys[filt_key]["color"],
                linewidth=plt.rcParams["lines.linewidth"] * 2 if not make_fit else 0,
                elinewidth=plt.rcParams["lines.linewidth"],
                mpl_kwargs=mpl_kwargs,
            )

            if not np.all(p_val[:, 0] == -1):
                _p = p_val[:, 0]
                for j, p in enumerate(_p):
                    ax = pval_plotter(
                        ax,
                        p,
                        pos=[xax[j], xax[j]],
                        loc=102 + i,
                        tail_height=0,
                        color=clr.stim_keys[filt_key]["color"],
                    )
    # baseline
    if len(catch_data) and not baseline_normalize:
        cnt = catch_data["count"].to_numpy()
        base_hr = np.sum(catch_data["hit_count"].to_numpy()) / np.sum(cnt)
        base_conf = 1.96 * np.sqrt((base_hr * (1.0 - base_hr)) / np.sum(cnt))
        ax._errorbar(
            0.1 if log_x else 0,
            100 * base_hr,
            100 * base_conf,
            marker="o",
            label=f"Catch Trials{make_label([0], cnt)}",
            color="#909090",
            mpl_kwargs=mpl_kwargs,
        )
        ax.axhline(100 * base_hr, color="k", linestyle=":", linewidth=2, alpha=0.7)

    if log_x:
        pass
        ax.set_xscale("symlog")
        ax.set_xlim([-0.1, 110])
    else:
        x_ticks = nonearly_data["linear_axis"].unique().sort().to_numpy()
        x_labels = nonearly_data[_axer].unique().sort().to_numpy()
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.set_xlim([x_ticks[0] - 0.5, x_ticks[-1] + 0.5])
    ax.set_ylim([0, 110])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Stimulus Contrast (%)")
    x_lab = "hit rate (%)"
    if baseline_normalize:
        xlab = "baseline norm. " + x_lab
        ax.set_ylabel(xlab)
    # ax.legend(loc='center left',bbox_to_anchor=(1,0.5),frameon=False)
    return fig, ax
