import numpy as np
import polars as pl
import matplotlib.pyplot as plt


from ...color import Color
from ...plotting_utils import set_style,make_linear_axis
from ....core.data_functions import make_subsets
from ....psychophysics.detection.wheelDetection.wheelDetectionGroupedAnalyzer import WheelDetectionGroupedAnalyzer

#TODO: change nbins to bin width
def make_dot_cloud(
    y_points:np.ndarray,
    center:float=0,
    bin_width:float=50, #ms
    width:float=0.5
    ) ->np.ndarray:
    """ Turns the data points into a cloud by dispersing them horizontally depending on their distribution,
    The more points in a bin, the wider the dispersion
    Returns x-coordinates of the dispersed points"""
    
    bin_edges = np.arange(np.nanmin(y_points), np.nanmax(y_points)+bin_width, bin_width)
        
    # Get upper bounds of bins
    counts, bin_edges = np.histogram(y_points, bins=bin_edges)
    
    # get the indices that correspond to points inside the bin edges
    idx_in_bin = []
    for ymin, ymax in zip(bin_edges[:-1], bin_edges[1:]):
        i = np.nonzero((y_points >= ymin) * (y_points < ymax))[0]
        idx_in_bin.append(i)
        
    x_coords = np.zeros(len(y_points))
    dx = width / (np.nanmax(counts) // 2)
    
    for i in idx_in_bin:
        _points = y_points[i]  # value of points that fall into the bin
        # if less then 2, leave untouched, will put it in the mid line
        if len(i) > 1:
            j = len(i) % 2
            i = i[np.argsort(_points)]
            # if even numbers of points, j will be 0, which will allocate the points equally to left and right
            # if odd, j will be 1, then, below lines will leave idx 0 at the midline and start from idx 1
            a = i[j::2]
            b = i[j + 1 :: 2]
            x_coords[a] = (0.5 + j / 3 + np.arange(len(a))) * dx
            x_coords[b] = (0.5 + j / 3 + np.arange(len(b))) * -dx
            
    return x_coords + center


def plot_reaction_time_cloud(
    data:pl.DataFrame,
    ax:plt.Axes=None,
    reaction_of:str="reaction_time",
    hit_only:bool=True,
    cloud_width:float=0.33,
    include_zero:bool=False,
    mpl_kwargs:dict=None,
    **kwargs
) -> plt.Axes:
    """ """
    
    set_style(kwargs.get("style","presentation"))
    if mpl_kwargs is None:
        mpl_kwargs = {}

    if ax is None:
        fig = plt.figure(figsize=mpl_kwargs.pop("figsize", (8, 8)))
        ax = fig.add_subplot(1, 1, 1)
        
    clr = Color()
    analyzer = WheelDetectionGroupedAnalyzer(data, **kwargs)
    grouped_nonearly_data = analyzer.grouped_data.drop_nulls("contrast")
    
    if hit_only:
        reaction_of = "hit_" + reaction_of
    
    if reaction_of not in grouped_nonearly_data.columns:
        raise ValueError(f"{reaction_of} is an invalid column name to plot reaction of, try {[c for c in grouped_nonearly_data.columns if 'time' in c]}")
    
    # add the linear axis
    lin_axis_dict = make_linear_axis(grouped_nonearly_data,"signed_contrast")
    _lin_axis = [float(lin_axis_dict[c]) if c is not None else None for c in grouped_nonearly_data["signed_contrast"].to_list()]
    grouped_nonearly_data = grouped_nonearly_data.with_columns(pl.Series("linear_axis",_lin_axis))
    
    if not include_zero:
        grouped_nonearly_data = grouped_nonearly_data.filter(pl.col("contrast") != 0)

    for filt_tup in make_subsets(grouped_nonearly_data, ["stim_side", "contrast"]):
        filt_df = filt_tup[-1]
        if not filt_df.is_empty():
            lin_ax = filt_df[0,"linear_axis"]
            for stimkey_filt_tup in make_subsets(filt_df,["stimkey"]):
                stim_filt_df = stimkey_filt_tup[-1]
                stim_filt_key = stimkey_filt_tup[0]
                if not stim_filt_df.is_empty():
                    times = stim_filt_df[0, reaction_of].to_numpy()
                    times = times[~np.isnan(times)]
                    # median = np.nanmedian(times)
            
                    x_dots = make_dot_cloud(times, 
                                            center=lin_ax, 
                                            bin_width=50,
                                            width=cloud_width
                                            )
                    # add a little bit jitter to x_axis, because sparse points look like a rope...
                    jit = cloud_width*00.1 #arbitrary
                    x_dots = x_dots + np.random.uniform(low=-jit, high=jit, size=x_dots.shape[0])
                    
                    ax.scatter(
                        x_dots,
                        times,
                        s=(plt.rcParams["lines.markersize"] ** 2) / 2,
                        color=clr.stim_keys[stim_filt_key]["color"],
                        label=(
                                filt_df[0, "stim_label"]
                                if filt_tup[1] == "contra" and filt_tup[2] == 12.5
                                else "_"
                            ),
                        **mpl_kwargs,
                    )
            
            if len(filt_df) >= 2:
                times_non_opto = filt_df[0, reaction_of].to_numpy()
                for k in range(1, len(filt_df)):
                    times_opto = filt_df[k, reaction_of].to_numpy()
                    if len(times_opto):
                        p = analyzer.get_pvalues_nonparametric(times_non_opto,times_opto)
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
                            lin_ax, 1020 + 2 * k, stars, color=clr.stim_keys[filt_df[k,"stimkey"]]["color"]
                        )
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
    # ax.set_yscale("symlog")
    # minor_locs = [200, 400, 600, 800, 2000, 4000, 6000, 8000]
    # ax.yaxis.set_minor_locator(plt.FixedLocator(minor_locs))
    # ax.yaxis.set_minor_formatter(plt.FormatStrFormatter("%d"))
    # ax.yaxis.set_major_locator(ticker.FixedLocator([100, 1000, 10000]))
    # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    # ax.xaxis.set_major_locator(ticker.FixedLocator(list(cpos.values())))
    # ax.xaxis.set_major_formatter(ticker.FixedFormatter([i for i in cpos.keys()]))
    ax.legend(loc='center left',bbox_to_anchor=(0.98,0.5),frameon=False)
    return ax

