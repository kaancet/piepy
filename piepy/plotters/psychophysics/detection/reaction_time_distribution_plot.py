import numpy as np
import polars as pl
import matplotlib.pyplot as plt


from ...color import Color
from ...plotting_utils import set_style
from ....core.data_functions import make_subsets


def bin_times(time_arr, bin_width=50, bins: np.ndarray = None):
    """Counts the response times in bins(ms)"""
    if bins is None:
        bins = np.arange(
            np.min(time_arr) - bin_width, np.max(time_arr) + bin_width, bin_width
        )

    return np.histogram(time_arr, bins)


def plot_reaction_time_distribution(
    data:pl.DataFrame,
    ax:plt.Axes=None,
    reaction_of:str="reaction_time",
    bin_width:float=50, #ms
    include_zero:bool=False,
    mpl_kwargs:dict=None,
    **kwargs
) -> plt.Axes:
    """ """
    set_style(kwargs.get("style","presentation"))
    if mpl_kwargs is None:
        mpl_kwargs = {}

    if ax is None:
        fig = plt.figure(figsize=mpl_kwargs.pop("figsize", (5, 5)))
        ax = fig.add_subplot(1, 1, 1)
        
    clr = Color()
    
    if reaction_of not in ["reaction_time","response_time"]:
        raise ValueError(f"{reaction_of} is an invalid column name to plot reaction of, can be {['reaction_time','response_time']}")
    
    if not include_zero:
        # for some reason filtering only for 0 contrast also filters out null values, this is a workaround...
        _ne_contrast = data.filter((pl.col("outcome")!="early") & (pl.col("contrast")!=0)) # non early contrast trials(excludes 0-150 ms responses too)
        _early = data.filter((pl.col("outcome")=="early") & (pl.col("contrast").is_null())) # early trials (still excludes 0-150ms)
        _early_after_stim = data.filter((pl.col("outcome")=="early") & (pl.col("contrast")!=0)) # 0-150 ms trials
        plot_data = pl.concat([_ne_contrast,_early,_early_after_stim]).sort("trial_no")
        
    # plot the early first
    early_data = plot_data.filter(pl.col("outcome")=="early")
    if not early_data.is_empty(): 
        early_times = early_data[reaction_of].sort().to_numpy()
        bin_edges_early = np.arange(np.nanmin(early_times), np.nanmax(early_times)+bin_width, bin_width)
        early_counts, _ = np.histogram(early_times, bins=bin_edges_early)    
        
        ax.step(bin_edges_early[1:], early_counts, where="pre",color="#9c9c9c",**mpl_kwargs)
    
    nonearly_data = plot_data.filter(pl.col("outcome")!="early")
    for filt_tup in make_subsets(nonearly_data,["stimkey"]):
        filt_df = filt_tup[-1]
        filt_key = filt_tup[0]
        if not filt_df.is_empty():
            times = filt_df[reaction_of].sort().to_numpy()
            
            # prep bin edges
            bin_edges = np.arange(np.nanmin(times), 1000, bin_width)
            bin_edges_miss = np.arange(1000, np.nanmax(times), bin_width)
            bin_edges = np.hstack((bin_edges, bin_edges_miss))
            
            counts, _ = np.histogram(times, bins=bin_edges)
            ax.step(bin_edges[:-1], counts, 
                    color=clr.stim_keys[filt_key]["color"],
                    **mpl_kwargs)

            # zero line
            ax.axvline(x=0, color="k", linewidth=1)
            ax.set_xlim([None, 1100])
            ax.set_xticks([-1000, -500, 0, 500, 1000])
            
            ax.set_ylabel("Trial count")
            ax.set_xlabel("Time from Stimulus Onset(ms)")
    return ax