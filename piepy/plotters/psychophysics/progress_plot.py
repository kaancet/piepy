import numpy as np
import polars as pl
from typing import Literal
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from ...core.data_functions import make_subsets
from ..plotting_utils import set_style, override_plots
from ..colors.color import Color


def plot_performance(
    data: pl.DataFrame,
    ax: plt.Axes|None = None,
    plot_in_time: bool = False,
    seperate_by: list[str] = ["stimkey"],
    rolling_window: int = 0,
    mpl_kwargs: dict|None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """ Plots the accuracy of subset of trials through the run 
    
    Args:
        data: dataframe of trials to plot
        ax: axes object to plot on
        plot_in_time: flag to either plot the x-axis in time (mins) or trial count
        seperate_by: column names to seperate the data by (e.g. stimkey, contrast)
        rolling_window: width of the window to do rolling average
        mpl_kwargs: keyword arguments to be used in matplotlib function calls
        
    Returns:
        tuple[plt.Figure, plt.Axes]: Figure and axes of plot
    """

    
    if mpl_kwargs is None:
        mpl_kwargs = {}
        
    clr_obj = Color(task=kwargs.get("task_color","detection"))
    set_style(kwargs.get("style", "presentation"))
    override_plots()

    if ax is None:
        fig = plt.figure(figsize=mpl_kwargs.pop("figsize", (15, 8)))
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = ax.get_figure()

    def _get_perf_(
        arr: np.ndarray,
    ) -> np.ndarray:
        """Returns the hit rates as an array"""
        arr = arr.astype(float)
        # convert -1(earlies) to np.nan if they exist
        arr[arr == -1] = np.nan
        # cumulative count of hits ignoring nans (1 hit, 0 miss)
        _hits = np.nancumsum(arr)
        # cumulative counts of all not nans (non earlies in this case)
        _non_earlies = np.cumsum(~np.isnan(arr))
        return (_hits / _non_earlies) * 100

    data = data.with_columns((pl.col("t_trialend") / 60_000).alias("trial_time"))

    for filt_tup in make_subsets(data, seperate_by):
        filt_df = filt_tup[-1]
        if "stimkey" in seperate_by and "contrast" in seperate_by:
            clr = {"color": clr_obj.make_key_mixed_color(filt_tup[0], str(filt_tup[1]))}
        elif len(seperate_by) == 1 and "stimkey" in seperate_by:
            clr = {"color": clr_obj.stim_keys[filt_tup[0]]["color"]}
        elif len(seperate_by) == 1 and "contrast" in seperate_by:
            clr = {"color": clr_obj.contrast_keys[str(filt_tup[0])]["color"]}
        else:
            clr = {}

        acc = _get_perf_(filt_df["state_outcome"].to_numpy())
        if plot_in_time:
            x = filt_df["trial_time"].to_numpy()
        else:
            x = filt_df["trial_no"].to_numpy()

        filt_df = filt_df.with_columns(
            pl.Series("hit_rate", acc)
            .rolling_mean(rolling_window, center=True)
            .alias("hit_rate")
        )
        y = filt_df["hit_rate"].to_numpy()

        ax._plot(x, y, label=f"{filt_tup[:-1]}", **clr, mpl_kwargs=mpl_kwargs)

    ax.set_ylim([0, 110])
    ax.set_xlabel("Time (mins)" if plot_in_time else "Trial no.")
    ax.set_ylabel("Hit rate (%)")
    ax.legend(loc="center left", frameon=False, fontsize=5)
    return fig, ax


def plot_reactiontime(
    data: pl.DataFrame,
    ax: plt.Axes = None,
    reaction_of: Literal["reaction_time","response_time"] = "response_time",
    include_miss: bool = False,
    include_zero: bool = False,
    plot_in_time: bool = False,
    seperate_by: list = ["stimkey"],
    rolling_window: int = 0,
    mpl_kwargs: dict = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """ Plots the change of reaction time of different type of trials through the run 
    
    Args:
        data: dataframe of trials to plot
        ax: axes object to plot on
        reaction_of: column name of which type of reaction time to use (e.g. reaction_time or response_time)
        include_zero: Include zero contrast trials or not
        plot_in_time: flag to either plot the x-axis in time (mins) or trial count
        seperate_by: column names to seperate the data by (e.g. stimkey, contrast)
        rolling_window: width of the window to do rolling average
        mpl_kwargs: keyword arguments to be used in matplotlib function calls
        
    Returns:
        tuple[plt.Figure, plt.Axes]: Figure and axes of plot
    """

    if mpl_kwargs is None:
        mpl_kwargs = {}
        
    clr_obj = Color(task=kwargs.get("task_color","detection"))
    set_style(kwargs.get("style", "presentation"))
    override_plots()

    if ax is None:
        fig = plt.figure(figsize=mpl_kwargs.pop("figsize", (15, 8)))
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = ax.get_figure()

    if reaction_of not in data.columns:
        raise ValueError(f"{reaction_of} not in data columns!")

    data = data.with_columns((pl.col("t_trialend") / 60_000).alias("trial_time"))
    if not include_zero:
        # for some reason filtering only for 0 contrast also filters out null values, this is a workaround...
        _ne_contrast = data.filter(
            (pl.col("outcome") != "early") & (pl.col("contrast") != 0)
        )  # non early contrast trials(excludes 0-150 ms responses too)
        _early = data.filter(
            (pl.col("outcome") == "early") & (pl.col("contrast").is_null())
        )  # early trials (still excludes 0-150ms)
        _early_after_stim = data.filter(
            (pl.col("outcome") == "early") & (pl.col("contrast") != 0)
        )  # 0-150 ms trials
        plot_data = pl.concat([_ne_contrast, _early, _early_after_stim]).sort("trial_no")

    plot_data = plot_data.filter(pl.col("outcome") != "early")
    for filt_tup in make_subsets(plot_data, seperate_by):
        filt_df = filt_tup[-1]
        if "stimkey" in seperate_by and "contrast" in seperate_by:
            clr = {"color": clr_obj.make_key_mixed_color(filt_tup[0], str(filt_tup[1]))}
        elif len(seperate_by) == 1 and "stimkey" in seperate_by:
            clr = {"color": clr_obj.stim_keys[filt_tup[0]]["color"]}
        elif len(seperate_by) == 1 and "contrast" in seperate_by:
            clr = {"color": clr_obj.contrast_keys[str(filt_tup[0])]["color"]}
        else:
            clr = {}

        if plot_in_time:
            x = filt_df["trial_time"].to_numpy()
        else:
            x = filt_df["trial_no"].to_numpy()

        filt_df = filt_df.with_columns(
            pl.col(reaction_of).rolling_mean(rolling_window).alias(f"roll_{reaction_of}")
        )
        y = filt_df[f"roll_{reaction_of}"].to_numpy()

        ax._plot(x, y, label=f"{filt_tup[:-1]}", **clr, mpl_kwargs=mpl_kwargs)

    ax.set_xlabel("Time (mins)" if plot_in_time else "Trial no.")
    # parse the axis label
    ax.set_ylabel(f'{reaction_of.replace("_", " ").capitalize()} (ms)')

    ax.set_yscale("symlog")
    minor_locs = [200, 400, 600, 800]
    ax.set_yticks([200,400,600,800,1000])
    ax.set_ylim([190,1000])
    ax.yaxis.set_minor_locator(plt.FixedLocator(minor_locs))
    ax.yaxis.set_minor_formatter(plt.FormatStrFormatter("%d"))
    ax.yaxis.set_major_locator(ticker.FixedLocator([100, 1000, 10000]))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    ax.legend(loc="center left", frameon=False, fontsize=5)
    
    return fig, ax


def plot_lick(
    data: pl.DataFrame,
    ax: plt.Axes = None,
    plot_in_time: bool = False,
    mpl_kwargs: dict = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """ Plots the cumulative amount of licks throughout the run
    
    Args:
        data: dataframe of trials to plot
        ax: axes object to plot on
        plot_in_time: flag to either plot the x-axis in time (mins) or trial count
        rolling_window: width of the window to do rolling average
        mpl_kwargs: keyword arguments to be used in matplotlib function calls
        
    Returns:
        tuple[plt.Figure, plt.Axes]: Figure and axes of plot
    """
    if mpl_kwargs is None:
        mpl_kwargs = {"color":"#00BBFF"}
    
    set_style(kwargs.get("style", "presentation"))    
    override_plots()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=mpl_kwargs.pop("figsize", (15, 8)))
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = ax.get_figure()

    if plot_in_time:
        _x_label = "Time (mins)"
        x = data["lick"].explode().drop_nulls().to_numpy() / 60_000
    else:
        _x_label = "Trial No"
        licks = data["lick"].to_list()
        trial_nos = data["trial_no"].to_list()
        listed_trial_nos = [
            [t] * len(ll) for t, ll in zip(trial_nos, licks) if ll is not None
        ]
        df = data.with_columns(pl.Series("lick_trial_nos", listed_trial_nos))
        x = df["lick_trial_nos"].explode().to_numpy()

    y = np.arange(1, len(x) + 1)
    ax._plot(x, y, mpl_kwargs=mpl_kwargs)

    ax.set_xlabel(_x_label)
    ax.set_ylabel("Lick count")
    
    return fig, ax
