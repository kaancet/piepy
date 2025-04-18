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
from ....core.data_functions import make_subsets
from ....psychophysics.wheel.detection.wheelDetectionGroupedAggregator import (
    WheelDetectionGroupedAggregator,
)


def plot_psychometric(
    data: pl.DataFrame,
    ax: plt.Axes = None,
    mpl_kwargs: dict | None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Plots the hit rates with 95% confidence intervals

    Args:
        data (pl.DataFrame): Data to be plotted, can be single or multiple sessions
        ax (plt.Axes, optional): An axes object to place to plot,default is None, which creates the axes
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

    clr = Color(task="detection")
    analyzer = WheelDetectionGroupedAggregator()
    analyzer.set_data(data=data)
    analyzer.group_data(group_by=["stim_type", "stim_side", "contrast", "opto_pattern"])
    analyzer.calculate_hit_rates()
    analyzer.calculate_opto_pvalues()

    nonearly_data = analyzer.grouped_data.drop_nulls("contrast")

    lin_axis_dict = make_linear_axis(nonearly_data, "signed_contrast")
    _lin_axis = [
        float(lin_axis_dict[c]) if c is not None else None
        for c in nonearly_data["signed_contrast"].to_list()
    ]
    nonearly_data = nonearly_data.with_columns(pl.Series("linear_axis", _lin_axis))

    for filt_tup in make_subsets(
        nonearly_data, ["stimkey", "stim_side"], start_enumerate=0
    ):
        i = filt_tup[0]
        filt_df = filt_tup[-1]
        filt_key = filt_tup[1]
        if not filt_df.is_empty():
            # don't plot nonopto catch(baseline) here, we'll do it later
            if filt_tup[2] == "catch" and filt_df[0, "opto_pattern"] == -1:
                continue

            contrast_label = filt_df["signed_contrast"].to_numpy()
            lin_ax = filt_df["linear_axis"].to_numpy()
            confs = 100 * filt_df["hit_rate_confs"].to_numpy().transpose()
            count = filt_df["count"].to_numpy()
            hr = 100 * filt_df["hit_rate"].to_numpy().flatten()
            stim_label = filt_df["stim_label"].unique().to_numpy()
            p_val = filt_df["p_hit_rate"].to_numpy()

            ax._errorbar(
                x=lin_ax,
                y=hr,
                yerr=confs,
                marker="o",
                label=f"{stim_label[0]}{make_label(contrast_label, count)}",
                color=clr.stim_keys[filt_key]["color"],
                linewidth=plt.rcParams["lines.linewidth"] * 2,
                elinewidth=plt.rcParams["lines.linewidth"],
                linestyle=clr.stim_keys[filt_key]["linestyle"],
                mpl_kwargs=mpl_kwargs,
            )
            if not np.all(p_val[:, 0] == -1):
                _p = p_val[:, 0]
                for j, p in enumerate(_p):
                    ax = pval_plotter(
                        ax,
                        p,
                        pos=[lin_ax[j], lin_ax[j]],
                        loc=102 + i,
                        tail_height=0,
                        color=clr.stim_keys[filt_key]["color"],
                    )

    # baseline
    baseline = nonearly_data.filter(
        (pl.col("stim_side") == "catch") & (pl.col("opto_pattern") == -1)
    )  # noqa: E712
    if len(baseline):
        cnt = baseline["count"].to_numpy()
        base_hr = np.sum(baseline["hit_count"].to_numpy()) / np.sum(cnt)
        base_conf = 1.96 * np.sqrt((base_hr * (1.0 - base_hr)) / np.sum(cnt))
        ax._errorbar(
            0,
            100 * base_hr,
            100 * base_conf,
            marker="o",
            label=f"Catch Trials{make_label([0], cnt)}",
            color="#909090",
            mpl_kwargs=mpl_kwargs,
        )
        ax.axhline(100 * base_hr, color="k", linestyle=":", linewidth=2, alpha=0.7)

    x_ticks = nonearly_data["linear_axis"].unique().sort().to_numpy()
    x_labels = nonearly_data["signed_contrast"].unique().sort().to_numpy()
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_xlim([x_ticks[0] - 0.5, x_ticks[-1] + 0.5])
    ax.set_ylim([0, 110])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Stimulus Contrast (%)")
    ax.set_ylabel("Hit Rate (%)")
    # ax.legend(loc='center left',bbox_to_anchor=(1,0.5),frameon=False)
    return fig, ax
