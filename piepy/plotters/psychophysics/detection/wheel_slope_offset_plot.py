import itertools
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from sklearn.neighbors import KernelDensity

from ...colors.color import Color
from ...plotting_utils import set_style, override_plots, pval_plotter
from ....core.data_functions import make_subsets
from ....core.statistics import mean_confidence_interval,ks2s_2d


def plot_wheel_slope_and_offset_xyerror(
    data:pl.DataFrame,
    ax:plt.Axes|None=None,
    seperate_by:list[str] = ["stimkey","contrast"],
    include_misses:bool=False,
    include_zero:bool=False,
    mpl_kwargs:dict=None,
    **kwargs
) -> tuple[plt.Figure,plt.Axes]:
    """Plots the slope vs onset distributions of trials with 95%CI ranges
    Seperates the data by one variable and plots it on a single axes
    This is to see how the wheel movement dynamics change with different experimental conditions

    Args:
        data (pl.DataFrame): Data to be plotted, can be single or multiple sessions
        ax (plt.Axes, optional): An axes object to place to plot,default is None, which creates the axes
        seperate_by (list[str], optional): Conditions to seperate the data by. Defaults to ["stimkey","contrast"].
        include_misses (bool, optional): Whether to plot miss trials. Defaults to False.
        include_zero (bool, optional): Whether to include catch trials. Defaults to False.
        mpl_kwargs (dict | None, optional): kwargs for styling matplotlib plots. Defaults to None.

    Returns:
        tuple[plt.Figure,plt.Axes]: Plotted figure and axes
    """
    if mpl_kwargs is None:
        mpl_kwargs = {}
        
    clr_obj = Color(task="detection")
    set_style(kwargs.get("style","presentation"))
    override_plots()

    if ax is None:
        fig = plt.figure(figsize=mpl_kwargs.pop("figsize", (8, 8)))
        ax = fig.add_subplot(1, 1, 1)
    
    if include_misses:
        plot_data = data.filter(pl.col("outcome") != "early")
    else:
        plot_data = data.filter(pl.col("outcome") == "hit")
        
    if not include_zero:
        # for some reason filtering only for 0 contrast also filters out null values, this is a workaround...
        _ne_contrast = data.filter((pl.col("outcome")!="early") & (pl.col("contrast")!=0)) # non early contrast trials(excludes 0-150 ms responses too)
        _early = data.filter((pl.col("outcome")=="early") & (pl.col("contrast").is_null())) # early trials (still excludes 0-150ms)
        _early_after_stim = data.filter((pl.col("outcome")=="early") & (pl.col("contrast")!=0)) # 0-150 ms trials
        plot_data = pl.concat([_ne_contrast,_early,_early_after_stim]).sort("trial_no")
        
    _temp_dict = {}
    for filt_tup in make_subsets(plot_data,seperate_by):
        filt_df = filt_tup[-1]
        
        filt_df = filt_df.filter(pl.col("outcome")=="hit")
        mov_onset = filt_df["reaction_time"].to_numpy()
        mov_peak_speed = filt_df["peak_speed"].to_numpy()
        
        onset_mean,onset_ci = mean_confidence_interval(mov_onset)
        peak_speed_mean,peak_speed_ci = mean_confidence_interval(mov_peak_speed)
        
        _key = f"{[k for k in filt_tup[:-1]]}".strip("[").strip("]")
        _temp_dict[_key] = np.hstack((mov_peak_speed.reshape(-1,1),
                                      mov_onset.reshape(-1,1)))

        if "stimkey" in seperate_by and "contrast" in seperate_by:
            clr = {"color": clr_obj.make_key_mixed_color(filt_tup[0], str(filt_tup[1]))}
        elif len(seperate_by) == 1 and "stimkey" in seperate_by:
            clr = {"color": clr_obj.stim_keys[filt_tup[0]]["color"]}
        elif len(seperate_by) == 1 and "contrast" in seperate_by:
            clr = {"color": clr_obj.contrast_keys[str(filt_tup[0])]["color"]}
        else:
            clr = {}
        
        ax._scatter(mov_peak_speed,mov_onset,
                   c=clr["color"],
                   alpha=0.2,
                   mpl_kwargs=mpl_kwargs)

        ax.errorbar(peak_speed_mean,
                    onset_mean,
                    xerr=peak_speed_ci,
                    yerr=onset_ci,
                    c=clr["color"],
                    marker='o', 
                    markersize=10,
                    label=f"{[k for k in filt_tup[:-1]]}")
        
    p_vals = {} 
    d_vals = {}
    for i,keys in enumerate(list(itertools.combinations(list(_temp_dict.keys()), 2))):
        k1,k2 = keys
        data1 = _temp_dict[k1]
        data2 = _temp_dict[k2]
        p,d = ks2s_2d(data1,data2,nboot=100)
        p_vals[f"{k1}={k2}"] = p
        d_vals[f"{k1}={k2}"] = d
        
        x1,_ = mean_confidence_interval(data1[:,0])
        x2,_ = mean_confidence_interval(data2[:,0])
        
        y2,_ = mean_confidence_interval(data2[:,1])
        ax = pval_plotter(ax,p,
                          pos=[x1,x2],
                          loc=700+i*10,
                          tail_height=0.001)
    
    ax.set_xlabel("Slope (rad/s)")
    ax.set_ylabel("Onset (ms)")
    ax.legend(frameon=False)
    ax.set_xlim()
    return fig, ax, p_vals, d_vals


def plot_wheel_slope_and_offset_ellipse(
    data:pl.DataFrame,
    seperate_by:list[str] = ["contrast"],
    include_misses:bool=False,
    include_zero:bool=False,
    mpl_kwargs:dict=None,
    **kwargs
) -> tuple[plt.Figure,plt.Axes]:
    """Plots the slope vs onset distributions of trials with 95%CI ranges
    Seperates the data by one variable and plots it on a single axes
    This is to see how the wheel movement dynamics change with different experimental conditions
    NOTE: It is better to call this function after filtering other conditions

    Args:
        data (pl.DataFrame): Data to be plotted, can be single or multiple sessions
        seperate_by (list[str], optional): Conditions to seperate the data by. Defaults to ["stimkey","contrast"].
        include_misses (bool, optional): Whether to plot miss trials. Defaults to False.
        include_zero (bool, optional): Whether to include catch trials. Defaults to False.
        mpl_kwargs (dict | None, optional): kwargs for styling matplotlib plots. Defaults to None.

    Returns:
        tuple[plt.Figure,plt.Axes]: Plotted figure and axes
    """
    if mpl_kwargs is None:
        mpl_kwargs = {}
        
    clr = Color(task="detection")
    set_style(kwargs.get("style","presentation"))
    override_plots()

    fig = plt.figure(figsize=mpl_kwargs.pop("figsize", (8, 8)))
    gs = fig.add_gridspec(2, 2,  width_ratios=(5, 1), height_ratios=(1, 5),
                            left=0.1, right=0.9, bottom=0.1, top=0.9,
                            wspace=0.05, hspace=0.05)
    ax = fig.add_subplot(gs[1, 0])
    ax_kdex = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_kdey = fig.add_subplot(gs[1, 1], sharey=ax)
    
    if include_misses:
        plot_data = data.filter(pl.col("outcome") != "early")
    else:
        plot_data = data.filter(pl.col("outcome") == "hit")
    
    if not include_zero:
        plot_data = plot_data.filter(pl.col("contrast") != 0)
        
    for filt_tup in make_subsets(plot_data,seperate_by):
        filt_df = filt_tup[-1]
        sep = filt_tup[0]
        
        filt_df = filt_df.filter(pl.col("outcome")=="hit")
        mov_onset = filt_df["reaction_time"].to_numpy()
        mov_peak_speed = filt_df["peak_speed"].to_numpy()

        ax._scatter(mov_peak_speed,mov_onset,
                   c=clr.contrast_keys[str(sep)]["color"],
                   alpha=0.5,
                   mpl_kwargs=mpl_kwargs)
        
        x_d = np.linspace(np.nanmin(mov_peak_speed)-1, np.nanmax(mov_peak_speed)+1, 100)
        kde = KernelDensity(bandwidth="scott", kernel='gaussian')
        kde.fit(np.array(mov_peak_speed)[:, None])
        logprob_slope = kde.score_samples(x_d[:, None])
        
        ax_kdex._plot(x_d,np.exp(logprob_slope),color=clr.contrast_keys[str(sep)]["color"],mpl_kwargs=mpl_kwargs)
        
        y_d = np.linspace(np.nanmin(mov_onset)-1, np.nanmax(mov_onset)+1, 100)
        kde = KernelDensity(bandwidth="scott", kernel='gaussian')
        kde.fit(np.array(mov_onset)[:, None])
        logprob_onset = kde.score_samples(y_d[:, None])
        
        ax_kdey._plot(np.exp(logprob_onset),y_d,color=clr.contrast_keys[str(sep)]["color"],mpl_kwargs=mpl_kwargs)
        
        contrast_points = np.array([mov_peak_speed,mov_onset])
        # Calculate the eigenvectors and eigenvalues
        covariance = np.cov(contrast_points)

        # Get the coordinates of the data mean
        avg_0 = np.mean(contrast_points[0])
        avg_1 = np.mean(contrast_points[1])
        
        pearson = covariance[0, 1]/np.sqrt(covariance[0, 0] * covariance[1, 1])
        
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, 
                                  height=ell_radius_y * 2, 
                                  facecolor=clr.contrast_keys[str(sep)]["color"],
                                  alpha=0.7,
                                  label=sep)
        
        scale_x = np.sqrt(covariance[0, 0] * 2.4477)
        scale_y = np.sqrt(covariance[1, 1] * 2.4477)

        transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(avg_0, avg_1)
        ellipse.set_transform(transf + ax.transData)
        ax.add_patch(ellipse)

    ax.set_xlabel("Slope (rad/ms)")
    ax.set_ylabel("Onset (ms)")
    ax.legend(frameon=False)
    ax.set_xlim()
    #[np.nanmin(mov_peak_speed),np.nanmax(mov_peak_speed)]
    ax_kdex.set_axis_off()
    ax_kdey.set_axis_off()
    return fig
    
    
def plot_all_wheel_slope_and_offset_ellipse(
    data: pl.DataFrame,
    seperate_by:list[str] = ["contrast"],
    include_misses:bool=False,
    include_zero:bool=False,
    time_reset:str="t_vstimstart_rig",
    mpl_kwargs:dict=None,
    **kwargs
) -> plt.Figure:
    """tuple[plt.Figure,plt.Axes]: Plotted figure and axes

    Args:
        data (pl.DataFrame): Data to be plotted, can be single or multiple sessions
        seperate_by (list[str], optional): Conditions to seperate the data by. Defaults to ["stimkey","contrast"].
        include_misses (bool, optional): Whether to plot miss trials. Defaults to False.
        include_zero (bool, optional): Whether to include catch trials. Defaults to False.
        time_reset (str, optional): _description_. Defaults to "t_vstimstart_rig".
        mpl_kwargs (dict | None, optional): kwargs for styling matplotlib plots. Defaults to None.

    Returns:
        plt.Axes: Plotted axes object
    """
    figs = []
    set_style(kwargs.get("style","presentation"))
    for filt_tup in make_subsets(data, ["opto_pattern", "stim_type"]):
        filt_df = filt_tup[-1]


        f = plot_wheel_slope_and_offset_ellipse(data=filt_df,
                                                seperate_by=seperate_by,
                                                include_misses=include_misses,
                                                include_zero=include_zero,
                                                time_reset=time_reset,
                                                mpl_kwargs=mpl_kwargs,
                                                **kwargs)
        f.suptitle(f"{filt_tup[1]}_{filt_tup[0]}")
        figs.append(f)
    return figs
