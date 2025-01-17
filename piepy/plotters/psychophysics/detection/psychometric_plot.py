import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from ...color import Color
from ...plotting_utils import set_style,make_linear_axis,make_label
from ....core.data_functions import make_subsets
from ....psychophysics.detection.wheelDetection.wheelDetectionGroupedAnalyzer import WheelDetectionGroupedAnalyzer


def plot_psychometric(
    data: pl.DataFrame,
    ax: plt.Axes=None,
    mpl_kwargs: dict=None,
    **kwargs,
) -> plt.Axes:
    """Plots the hit rates with 95% confidence intervals

    Parameters:
    data (pl.DataFrame) : run data
    ax (plt.axes) : An axes object to place to plot,default is None, which creates the axes

    Returns:
    plt.axes: Axes object
    """
    set_style(kwargs.get("style","presentation"))
    if mpl_kwargs is None:
        mpl_kwargs = {}

    if ax is None:
        fig = plt.figure(figsize=mpl_kwargs.pop("figsize", (8, 8)))
        ax = fig.add_subplot(1, 1, 1)
    
    clr = Color()
    analyzer = WheelDetectionGroupedAnalyzer(data,**kwargs)
    nonearly_data = analyzer.grouped_data.drop_nulls("contrast")
    
    lin_axis_dict = make_linear_axis(nonearly_data,"signed_contrast")
    _lin_axis = [float(lin_axis_dict[c]) if c is not None else None for c in nonearly_data["signed_contrast"].to_list()]
    nonearly_data = nonearly_data.with_columns(pl.Series("linear_axis",_lin_axis))
    
    for filt_tup in make_subsets(nonearly_data, ["stimkey", "stim_side"]):
        filt_df = filt_tup[-1]
        filt_key = filt_tup[0]
        if not filt_df.is_empty():
            # don't plot nonopto catch(baseline) here, we'll do it later
            if filt_tup[1] == "catch" and not filt_df[0, "opto"]:
                continue
            
            contrast_label = filt_df["signed_contrast"].to_numpy()
            lin_ax = filt_df["linear_axis"].to_numpy()
            confs = 100 * filt_df["confs"].to_numpy()
            count = filt_df["count"].to_numpy()
            hr = 100 * filt_df["hit_rate"].to_numpy()
            stim_label = filt_df["stim_label"].unique().to_numpy()
            p_val = filt_df["p_hit_rate"].to_numpy()
            
            ax.errorbar(
                    lin_ax,
                    hr,
                    confs,
                    marker="o",
                    label=f"{stim_label[0]}{make_label(contrast_label,count)}",
                    color=clr.stim_keys[filt_key]["color"],
                    linewidth=plt.rcParams["lines.linewidth"] * 2,
                    elinewidth=plt.rcParams["lines.linewidth"],
                    linestyle=clr.stim_keys[filt_key]["linestyle"],
                    **mpl_kwargs,
                )
            
            for i,p in enumerate(p_val):
                stars = ""
                if p < 0.0001:
                    stars = "****"
                elif 0.0001 <= p < 0.001:
                    stars = "***"
                elif 0.001 <= p < 0.01:
                    stars = "**"
                elif 0.01 <= p < 0.05:
                    stars = "*"
                ax.text(
                    lin_ax[i], 102 + 2 * i, stars, color=clr.stim_keys[filt_key]["color"]
                )
                
    # baseline
    baseline = nonearly_data.filter((pl.col("stim_side") == "catch") & (pl.col("opto") == False))  # noqa: E712
    if len(baseline):
        cnt = baseline["count"].to_numpy()
        base_hr = np.sum(baseline["hit_count"].to_numpy()) / np.sum(cnt)
        base_conf = 1.96 * np.sqrt((base_hr * (1.0 - base_hr)) / np.sum(cnt))
        ax.errorbar(
            0,
            100 * base_hr,
            100 * base_conf,
            marker="o",
            label=f"Catch Trials{make_label([0],cnt)}",
            color="#909090",
            **mpl_kwargs,
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
    ax.grid()
    # ax.legend(loc='center left',bbox_to_anchor=(1,0.5),frameon=False)
    return ax