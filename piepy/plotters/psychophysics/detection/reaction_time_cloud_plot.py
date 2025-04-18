import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from ...colors.color import Color
from ...plotting_utils import (
    set_style, 
    make_linear_axis, 
    override_plots, 
    pval_plotter,
    make_dot_cloud)
from ....core.data_functions import make_subsets
from ....core.statistics import nonparametric_pvalues
from ....psychophysics.wheel.detection.wheelDetectionGroupedAggregator  import WheelDetectionGroupedAggregator


def plot_reaction_time_cloud(
    data: pl.DataFrame,
    ax: plt.Axes = None,
    reaction_of: str = "reaction_times",
    hit_only: bool = True,
    bin_width:float = 50,
    cloud_width: float = 0.33,
    include_zero: bool = False,
    mpl_kwargs: dict| None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """ Plots the reaction(or response) times as a bee swarm plot, with medians and bootstrapped 95% CI's

    Args:
        data (pl.DataFrame): Data to be plotted, can be single or multiple sessions
        ax (plt.Axes, optional): An axes object to place to plot,default is None, which creates the axes
        reaction_of (str, optional): Which time readout to plot. "reaction_times" or "response_times". Defaults to "reaction_times".
        hit_only (bool, optional): Plot only hit trials. Defaults to True.
        bin_width (float, optional) : width of bins in ms, Defaults to 50.
        cloud_width (float, optional): width of the swarm plot. Defaults to 0.33.
        include_zero (bool, optional): Whether to include catch trials. Defaults to False.
        mpl_kwargs (dict | None, optional): kwargs for styling matplotlib plots. Defaults to None.

    Raises:
        ValueError: Invalid column name to plot

    Returns:
        tuple[plt.Figure,plt.Axes]: Plotted figure and axes objects
    """
    if mpl_kwargs is None:
        mpl_kwargs = {}
    set_style(kwargs.get("style", "presentation"))
    override_plots()    

    if ax is None:
        fig = plt.figure(figsize=mpl_kwargs.pop("figsize", (8, 8)))
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = ax.get_figure()

    clr = Color(task="detection")
    analyzer = WheelDetectionGroupedAggregator()
    analyzer.set_data(data=data)
    analyzer.group_data(group_by=["stim_type", "stim_side", "contrast", "opto_pattern"])
    grouped_nonearly_data = analyzer.grouped_data.drop_nulls("contrast")

    if hit_only:
        reaction_of = "hit_" + reaction_of

    if reaction_of not in grouped_nonearly_data.columns:
        raise ValueError(
            f"{reaction_of} is an invalid column name to plot reaction of, try {[c for c in grouped_nonearly_data.columns if 'time' in c]}"
        )

    # add the linear axis
    lin_axis_dict = make_linear_axis(grouped_nonearly_data, "signed_contrast")
    _lin_axis = [
        float(lin_axis_dict[c]) if c is not None else None
        for c in grouped_nonearly_data["signed_contrast"].to_list()
    ]
    grouped_nonearly_data = grouped_nonearly_data.with_columns(
        pl.Series("linear_axis", _lin_axis)
    )

    if not include_zero:
        grouped_nonearly_data = grouped_nonearly_data.filter(pl.col("contrast") != 0)

    for filt_tup in make_subsets(grouped_nonearly_data, ["stim_side", "contrast"]):
        filt_df = filt_tup[-1]
        if not filt_df.is_empty():
            lin_ax = filt_df[0, "linear_axis"]
            for stimkey_filt_tup in make_subsets(filt_df, ["stimkey"]):
                stim_filt_df = stimkey_filt_tup[-1]
                stim_filt_key = stimkey_filt_tup[0]
                if not stim_filt_df.is_empty():
                    times = stim_filt_df[0, reaction_of].to_numpy()
                    times = times[~np.isnan(times)]
                    if len(times):
                        x_dots = make_dot_cloud(
                            times, center=lin_ax, bin_width=bin_width, width=cloud_width
                        )
                        # add a little bit jitter to x_axis, because sparse points look like a rope...
                        jit = cloud_width * 00.1  # arbitrary
                        x_dots = x_dots + np.random.uniform(
                            low=-jit, high=jit, size=x_dots.shape[0]
                        )
                        
                        medi = stim_filt_df[0,f"median_{reaction_of}"]
                        medi_conf = stim_filt_df[0,f"median_{reaction_of}_confs"].to_numpy().reshape(-1,1)
                        
                        # medians
                        ax.scatter(lin_ax,
                                   medi,
                                   s = (plt.rcParams["lines.markersize"] ** 2),
                                   c = clr.stim_keys[stim_filt_key]["color"],
                                   marker="_",
                                   linewidths = 3,
                                   edgecolors = "w",
                                   zorder=3,)
                        
                        # individual dots
                        ax._scatter(
                            x_dots,
                            times,
                            s=(plt.rcParams["lines.markersize"] ** 2) / 2,
                            color=clr.stim_keys[stim_filt_key]["color"],
                            linewidths = 0.3,
                            edgecolors = "w",
                            label=(
                                filt_df[0, "stim_label"]
                                if filt_tup[1] == "contra" and filt_tup[2] == 12.5
                                else "_"
                            ),
                            alpha=0.5,
                            zorder=1,
                            mpl_kwargs=mpl_kwargs,
                        )
                        
                        # shaded CI's
                        ax.errorbar(
                            lin_ax,
                            medi,
                            yerr = medi_conf if len(medi_conf) else 0,
                            color = clr.stim_keys[stim_filt_key]["color"],
                            alpha=0.3,
                            zorder=2,
                            elinewidth=cloud_width*110,
                            markersize=0,
                        )

            if len(filt_df) >= 2:
                times_non_opto = filt_df[0, reaction_of].to_numpy()
                for k in range(1, len(filt_df)):
                    times_opto = filt_df[k, reaction_of].to_numpy()
                    if len(times_opto):
                        p = nonparametric_pvalues(times_non_opto, times_opto)
                        
                        ax = pval_plotter(ax,p,
                                          pos=[lin_ax,lin_ax],
                                          loc=1020 + 2 * k,
                                          tail_height=0,
                                          color=clr.stim_keys[filt_df[k, "stimkey"]]["color"])
                        
    # mid line
    ax.plot([0, 0], ax.get_ylim(), color="gray", linewidth=2, alpha=0.5)

    # miss line
    ax.axhline(1000, color="r", linewidth=1.5, linestyle=":")

    ax.set_xlabel("Stimulus Contrast (%)")
    _yl = reaction_of.split("_")
    ax.set_ylabel(f"{' '.join(_yl).capitalize()} (ms)")
    x_ticks = grouped_nonearly_data["linear_axis"].unique().sort().to_numpy()
    x_labels = grouped_nonearly_data["signed_contrast"].unique().sort().to_numpy()
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    
    # ax.set_yticks([200,400,600,800,1000])
    
    ax.set_yscale("symlog")
    minor_locs = [200, 400, 600, 800]
    ax.yaxis.set_minor_locator(plt.FixedLocator(minor_locs))
    ax.yaxis.set_minor_formatter(plt.FormatStrFormatter("%d"))
    ax.yaxis.set_major_locator(ticker.FixedLocator([100, 1000, 10000]))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    ax.set_ylim([100,1000])
    # ax.xaxis.set_major_locator(ticker.FixedLocator(list(cpos.values())))
    # ax.xaxis.set_major_formatter(ticker.FixedFormatter([i for i in cpos.keys()]))
    ax.legend(loc="center left", bbox_to_anchor=(0.98, 0.5), frameon=False)
    return fig, ax
