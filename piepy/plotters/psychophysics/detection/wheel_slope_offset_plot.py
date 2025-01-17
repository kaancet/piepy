import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from sklearn.neighbors import KernelDensity

from ...color import Color
from ....psychophysics.wheelTrace import WheelTrace
from ...plotting_utils import set_style
from ....core.data_functions import make_subsets


def plot_wheel_slope_and_offset(
    data:pl.DataFrame,
    seperate_by:list[str] = ["contrast"],
    include_misses:bool=False,
    include_zero:bool=False,
    time_reset:str="t_vstimstart_rig",
    mpl_kwargs:dict=None,
    **kwargs
) -> tuple[plt.Figure,plt.Axes]:
    """ 
    Plots the slope vs onset distributions of trials with 95%CI ranges
    Seperates the data by one variable and plots it on a single axes
    This is to see how the wheel movement dynamics change with different experimental conditions
    NOTE: It is better to call this function after filtering other conditions"""
    
    clr = Color()
    trace = WheelTrace()
    trace_interp_freq = kwargs.get("interp_freq", 5)
    set_style(kwargs.get("style","presentation"))
    if mpl_kwargs is None:
        mpl_kwargs = {}
        

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
        mov_onset = []
        mov_slope = []
        for i, trial in enumerate(filt_df.iter_rows(named=True)):
            wheel_t = np.array(trial["wheel_t"])
            wheel_tick = np.array(trial["wheel_pos"])
            reset_time_point = trial[time_reset]
            _,_, t_interp, tick_interp = trace.reset_and_interpolate(wheel_t, 
                                                                     wheel_tick, 
                                                                     reset_time_point, 
                                                                     trace_interp_freq)
            
            # convert the interpolation
            pos_interp = trace.cm_to_rad(trace.ticks_to_cm(tick_interp))
            
            mov_dict = trace.get_movements(
                t_interp,
                pos_interp,
                freq=trace_interp_freq,
                pos_thresh=kwargs.get("pos_thresh", 0.00015),
                t_thresh=kwargs.get("t_thresh", 0.5),
            )
            
            wheel_react = trial["reaction_time"]
            if wheel_react is not None:
                for i in range(len(mov_dict["onsets"])):
                    _on = mov_dict["onsets"][i,1]
                    _off = mov_dict["offsets"][i,1]
                    if wheel_react < _off and wheel_react >= _on:
                        y0 = pos_interp[int(mov_dict["onsets"][i,0])]
                        y1 = pos_interp[int(mov_dict["offsets"][i,0])]
                        slope = (y1 - y0) / (_off - _on)
                        break
                    else:
                        pass
                
                # uncertainty_idx = _on/slope
                mov_onset.append(_on)
                mov_slope.append(slope*1000)
                
        
        ax.scatter(mov_slope,mov_onset,
                   c=clr.contrast_keys[str(sep)]["color"],
                   alpha=0.5,
                   **mpl_kwargs)
        
        x_d = np.linspace(np.nanmin(mov_slope)-2, np.nanmax(mov_slope)+2, 100)
        kde = KernelDensity(bandwidth=np.abs(np.nanmin(mov_slope)/2), kernel='gaussian')
        kde.fit(np.array(mov_slope)[:, None])
        logprob_slope = kde.score_samples(x_d[:, None])
        
        ax_kdex.plot(x_d,np.exp(logprob_slope),color=clr.contrast_keys[str(sep)]["color"])
        
        y_d = np.linspace(np.nanmin(mov_onset)-2, np.nanmax(mov_onset)+2, 100)
        kde = KernelDensity(bandwidth=np.nanmin(mov_onset)/2, kernel='gaussian')
        kde.fit(np.array(mov_onset)[:, None])
        logprob_onset = kde.score_samples(y_d[:, None])
        
        ax_kdey.plot(np.exp(logprob_onset),y_d,color=clr.contrast_keys[str(sep)]["color"])
        
        contrast_points = np.array([mov_slope,mov_onset])
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

    ax.set_xlabel("Slope (A.U)")
    ax.set_ylabel("Onset (ms)")
    ax.legend(frameon=False)
    
    ax_kdex.set_axis_off()
    ax_kdey.set_axis_off()
    return fig
    
    
def plot_all_wheel_slope_and_offsets(
    data: pl.DataFrame,
    include_misses:bool=False,
    include_zero:bool=False,
    seperate_by:list[str] = ["contrast"],
    time_reset:str="t_vstimstart_rig",
    mpl_kwargs:dict=None,
    **kwargs
) -> plt.Figure:
    """ Runs through opto patterns and stimulus types to plot them in seperate figures"""
    figs = []

    for filt_tup in make_subsets(data, ["opto_pattern", "stim_type"]):
        filt_df = filt_tup[-1]


        f = plot_wheel_slope_and_offset(data=filt_df,
                                         seperate_by=seperate_by,
                                         include_misses=include_misses,
                                         include_zero=include_zero,
                                         time_reset=time_reset,
                                         mpl_kwargs=mpl_kwargs,
                                         **kwargs)
        f.suptitle(f"{filt_tup[1]}_{filt_tup[0]}")
        figs.append(f)
    return figs
