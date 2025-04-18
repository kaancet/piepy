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
from ....psychophysics.wheel.discrimination.wheelDiscriminationGroupedAggregator import (
    WheelDiscriminationGroupedAggregator,
)


def plot_psychometric(
    data: pl.DataFrame,
    discrim_of: str,
    ax: plt.Axes = None,
    mpl_kwargs: dict = None,
    **kwargs,
) -> plt.Axes:
    """Plots the hit rates with 95% confidence intervals

    Parameters:
    data (pl.DataFrame) : run data
    ax (plt.axes) : An axes object to place to plot,default is None, which creates the axes

    Returns:
    plt.axes: Axes object
    """
    override_plots()
    set_style(kwargs.get("style", "presentation"))
    if mpl_kwargs is None:
        mpl_kwargs = {}

    if ax is None:
        fig = plt.figure(figsize=mpl_kwargs.pop("figsize", (8, 8)))
        ax = fig.add_subplot(1, 1, 1)

    clr = Color(task="discrimination")

    signed_diff = f"diff_{discrim_of}"
    analyzer = WheelDiscriminationGroupedAggregator()
    analyzer.set_data(data=data)
    analyzer.group_data(
        group_by=["stim_type", "target_side", signed_diff, "opto_pattern"]
    )
    analyzer.calculate_proportion()
    analyzer.calculate_opto_pvalues()

    plot_data = analyzer.grouped_data

    lin_axis_dict = make_linear_axis(plot_data, signed_diff)
    _lin_axis = [
        float(lin_axis_dict[c]) if c is not None else None
        for c in plot_data[signed_diff].to_list()
    ]
    plot_data = plot_data.with_columns(pl.Series("linear_axis", _lin_axis))

    for filt_tup in make_subsets(
        plot_data, ["stimkey", "target_side"], start_enumerate=0
    ):
        i = filt_tup[0]
        filt_df = filt_tup[-1]
        filt_key = filt_tup[1]
        if not filt_df.is_empty():
            lin_ax = filt_df["linear_axis"].to_numpy()
            confs = 100 * filt_df["right_choice_confs"].to_numpy().transpose()

            prob_r = 100 * filt_df["right_choice_prob"].to_numpy().flatten()
            stim_label = filt_df["stim_label"].unique().to_numpy()
            p_val = filt_df["p_right_choice_prob"].to_numpy()

            ax._errorbar(
                x=lin_ax,
                y=prob_r,
                yerr=confs,
                marker="o",
                label=f"{stim_label[0]}",
                # color=clr.stim_keys[filt_key]["color"],
                linewidth=plt.rcParams["lines.linewidth"] * 2,
                elinewidth=plt.rcParams["lines.linewidth"],
                # linestyle=clr.stim_keys[filt_key]["linestyle"],
                mpl_kwargs=mpl_kwargs,
            )
            if not np.all(p_val[:, 0] == -1):
                _p = p_val[:, 0]
                for j, p in enumerate(_p):
                    ax = pval_plotter(
                        ax, p, pos=[lin_ax[j], lin_ax[j]], loc=102 + i, tail_height=0
                    )

    x_ticks = plot_data["linear_axis"].unique().sort().to_numpy()
    x_labels = plot_data[signed_diff].unique().sort().to_numpy()
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_xlim([x_ticks[0] - 0.5, x_ticks[-1] + 0.5])
    ax.set_ylim([0, 110])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel(f"Stimulus {discrim_of} difference (contra - ipsi)")
    ax.set_ylabel("Probability of choosing right (%)")
    ax.grid()
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
    return ax
