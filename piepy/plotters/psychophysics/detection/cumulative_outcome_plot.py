import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from ...plotting_utils import set_style,override_plots


def get_cumulative(time_data_arr: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Gets the cumulative distribution"""
    sorted_times = np.sort(time_data_arr)
    counts, _ = np.histogram(sorted_times, bins=bin_edges)
    pdf = counts / np.sum(counts)
    cum_sum = np.cumsum(pdf)
    return cum_sum


def plot_cumulative_outcome(data:pl.DataFrame,
                            ax:plt.Axes = None,
                            reaction_of:str="reaction_time",
                            bin_width:int=10, # in ms
                            include_zero:bool=False,
                            mpl_kwargs:dict|None=None,
                            **kwargs) -> tuple[plt.Figure,plt.Axes]:
    """ Plots  cumulative distribution of trial outcomes, wrt stimulus onset

    Args:
        data (pl.DataFrame): Data to be plotted, can be single or multiple sessions
        ax (plt.Axes, optional): An axes object to place to plot,default is None, which creates the axes
        reaction_of (str, optional): _description_. Defaults to "reaction_time".
        bin_width (int, optional): width of time bins in ms. Defaults to 10.
        mpl_kwargs (dict | None, optional): kwargs for styling matplotlib plots. Defaults to None.

    Returns:
        tuple[plt.Figure,plt.Axes]: Plotted figure and axes objects
    """
    if mpl_kwargs is None:
        mpl_kwargs = {}
    
    if ax is None:
        fig = plt.figure(figsize=mpl_kwargs.pop("figsize", (8, 8)))
        ax = fig.add_subplot(1, 1, 1)
    set_style(kwargs.get("style","presentation"))
    override_plots()
    if not include_zero:
        # for some reason filtering only for 0 contrast also filters out null values, this is a workaround...
        _ne_contrast = data.filter((pl.col("outcome")!="early") & (pl.col("contrast")!=0)) # non early contrast trials(excludes 0-150 ms responses too)
        _early = data.filter((pl.col("outcome")=="early") & (pl.col("contrast").is_null())) # early trials (still excludes 0-150ms)
        _early_after_stim = data.filter((pl.col("outcome")=="early") & (pl.col("contrast")!=0)) # 0-150 ms trials
        plot_data = pl.concat([_ne_contrast,_early,_early_after_stim]).sort("trial_no")
    
    if not plot_data.is_empty():
        react_times = plot_data[reaction_of].sort().to_numpy()
        
        bin_edges_early = np.arange(np.nanmin(react_times), 0, bin_width)
        bin_edges = np.arange(0, 1000, bin_width)
        bin_edges_miss = np.arange(1000, np.nanmax(react_times), bin_width)
        
        bin_edges = np.hstack((bin_edges_early, bin_edges, bin_edges_miss))
        cum_sum = get_cumulative(react_times,bin_edges)
        
        hit_start = np.where(bin_edges >= 150)[0][0]
        hit_end = np.where(bin_edges >= 1000)[0][0]

        ax._step(bin_edges[0:hit_start], cum_sum[0:hit_start], color="#9c9c9c",where="pre", mpl_kwargs=mpl_kwargs)
        ax._step(bin_edges[hit_end - 1 : -1], cum_sum[hit_end - 1 :], color="#CA0000",where="pre", mpl_kwargs=mpl_kwargs)
        ax._step(bin_edges[hit_start:hit_end], cum_sum[hit_start:hit_end], color="#039612",where="pre", mpl_kwargs=mpl_kwargs)

        ax.set_ylabel("Fraction of Trials")
        ax.set_xlabel("Time from Stimulus Onset(ms)")
    
    return fig,ax
