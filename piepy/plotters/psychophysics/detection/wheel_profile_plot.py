import numpy as np
import polars as pl
from scipy import stats
import matplotlib.pyplot as plt

from ...colors.color import Color
from ....psychophysics.wheel.wheelTrace import WheelTrace
from ...plotting_utils import set_style, override_plots
from ....core.data_functions import make_subsets


def plot_wheel_profile(
    data: pl.DataFrame,
    ax: plt.Axes = None,
    seperate_by: list[str] = ["contrast"],
    time_reset: str = "t_vstimstart_rig",
    plot_speed: bool = True,
    include_misses: bool = False,
    mpl_kwargs: dict = None,
    **kwargs,
) -> plt.Axes:
    """Plots the profile of the wheel (speed or position)
    Seperates the data by one variable and plots it on a single axes
    This is to see how the average wheel trajectory profile looks like for a given condition
    NOTE: It is better to call this function after filtering other conditions

    Args:
        data (pl.DataFrame): Data to be plotted, can be single or multiple sessions
        ax (plt.Axes, optional): An axes object to place to plot,default is None, which creates the axes
        seperate_by (list[str], optional): Conditions to seperate the data by. Defaults to ["contrast"].
        time_reset (str, optional): Time anchor point to reset the time frames of wheel trajectories. Defaults to "t_vstimstart_rig".
        plot_speed (bool, optional): If true, plot speed, else plot position. Defaults to True.
        include_misses (bool, optional): Whether to plot miss trials. Defaults to False.
        mpl_kwargs (dict | None, optional): kwargs for styling matplotlib plots. Defaults to None.

    Returns:
        plt.Axes: Plotted axes object
    """

    if mpl_kwargs is None:
        mpl_kwargs = {}

    clr = Color(task="detection")
    override_plots()

    trace = WheelTrace()
    trace_interp_freq = kwargs.get("interp_freq", 5)

    time_lims = kwargs.pop("time_lims", None)
    if time_lims is None:
        time_lims = [-200, 1500]

    traj_lims = kwargs.pop("traj_lims", None)
    if traj_lims is None:
        traj_lims = [None, None]

    if ax is None:
        fig = plt.figure(figsize=mpl_kwargs.pop("figsize", (8, 8)))
        ax = fig.add_subplot(1, 1, 1)

    if include_misses:
        plot_data = data.filter(pl.col("outcome") != "early")
    else:
        plot_data = data.filter(pl.col("outcome") == "hit")

    all_thresh = []
    for filt_tup in make_subsets(plot_data, seperate_by):
        filt_df = filt_tup[-1]
        sep = filt_tup[0]

        _longest_trace_len = 0
        trials_wheel_list = []
        _rig_response_rad_list = []
        for i, trial in enumerate(filt_df.iter_rows(named=True)):
            wheel_t = np.array(trial["wheel_t"])
            wheel_tick = np.array(trial["wheel_pos"])
            reset_time_point = trial[time_reset]

            _, _, t_interp, tick_interp = trace.reset_and_interpolate(
                wheel_t, wheel_tick, reset_time_point, trace_interp_freq
            )
            # convert the interpolation to rad
            interp_pos = trace.cm_to_rad(trace.ticks_to_cm(tick_interp))

            speed = (
                np.abs(trace.get_filtered_velocity(interp_pos, trace_interp_freq))
                * 1000
            )  # rad/s
            idx_in_time_range = np.where(
                (t_interp >= time_lims[0]) & (t_interp <= time_lims[1])
            )

            if len(idx_in_time_range):
                time_window = t_interp[idx_in_time_range]
                if plot_speed:
                    speed_window = speed[idx_in_time_range]
                    y_in_rads = speed_window
                    if trial["rig_response_tick"] is not None:
                        speed_thresh = trial["rig_response_tick"] / (
                            trial["median_loop_time"] * 5
                        )  # 5 is wheelbuffer
                        speed_thresh = (
                            trace.cm_to_rad(trace.ticks_to_cm(speed_thresh)) * 1000
                        )  # rad/s
                        _rig_response_rad_list.append(speed_thresh)
                else:
                    y_in_rads = interp_pos[idx_in_time_range]
                    if trial["rig_response_tick"] is not None:
                        pos_thresh = trace.cm_to_rad(
                            trace.ticks_to_cm(trial["rig_response_tick"])
                        )
                        _rig_response_rad_list.append(pos_thresh)

                trials_wheel_list.append(y_in_rads.tolist())
                # adjusting the longest
                if len(y_in_rads) >= _longest_trace_len:
                    _longest_trace_len = len(y_in_rads)
                    _longest_time = time_window

        # make a matrix to avreage over the rows, pad with None until reaching longest trace
        all_traces_mat = np.array(
            [xi + [None] * (_longest_trace_len - len(xi)) for xi in trials_wheel_list],
            dtype=float,
        )

        #
        all_thresh.extend(_rig_response_rad_list)

        avg = np.nanmean(all_traces_mat, axis=0)
        sem = stats.sem(all_traces_mat, axis=0, nan_policy="omit")

        ax.fill_between(
            _longest_time,
            avg - sem,
            avg + sem,
            color=clr.contrast_keys[str(sep)]["color"],
            alpha=0.2,
            linewidth=0,
        )
        ax._plot(
            _longest_time, avg, **clr.contrast_keys[str(sep)], mpl_kwargs=mpl_kwargs
        )

    thresh_mean = np.mean(all_thresh)
    ax.axhline(thresh_mean, color="#147800", linewidth=0.5, alpha=0.8)

    # anchor line(stimstart init start)
    ax.axvline(0, color="k", linewidth=1, alpha=0.6)

    # stim end
    ax.axvline(1000, color="k", linewidth=1, alpha=0.6)

    ax.set_xlim(time_lims[0] - 10, time_lims[1] + 10)
    # ax.set_ylim(traj_lims[0], traj_lims[1])
    ax.set_ylim([0, 8])
    if plot_speed:
        ax.set_ylabel("Wheel\nspeed (rad/s)")
    else:
        ax.set_ylabel("Wheel\nmovement (rad)")
    ax.set_xlabel("Time from stim onset (ms)")
    return ax


def plot_all_wheel_profiles(
    data: pl.DataFrame,
    seperate_by: list[str] = ["contrast"],
    time_reset: str = "t_vstimstart_rig",
    plot_speed: bool = True,
    include_misses: bool = False,
    mpl_kwargs: dict = None,
    **kwargs,
) -> plt.Figure:
    """Runs through opto patterns and stimulus types to plot them on seperate axes

    Args:
        data (pl.DataFrame): Data to be plotted, can be single or multiple sessions
        seperate_by (list[str], optional): Conditions to seperate the data by. Defaults to ["contrast"].
        time_reset (str, optional): Time anchor point to reset the time frames of wheel trajectories. Defaults to "t_vstimstart_rig".
        plot_speed (bool, optional): If true, plot speed, else plot position. Defaults to True.
        include_misses (bool, optional): Whether to plot miss trials. Defaults to False.
        mpl_kwargs (dict | None, optional): kwargs for styling matplotlib plots. Defaults to None.

    Returns:
        plt.Figure: Plotted figure object
    """
    set_style(kwargs.get("style", "presentation"))
    uniq_opto = data["opto_pattern"].drop_nulls().unique().sort().to_list()
    n_opto = len(uniq_opto)

    uniq_stim = data["stim_type"].drop_nulls().unique().sort().to_list()
    n_stim = len(uniq_stim)

    # opto patterns are on the columns
    # stim types are on the rows
    fig, axes = plt.subplots(
        nrows=n_stim,
        ncols=n_opto,
        constrained_layout=True,
        figsize=mpl_kwargs.pop("figsize", (20, 15)),
    )

    # if single axes
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    _target_id = uniq_opto[0]
    _stim_name = uniq_stim[0]
    ax_idx = 0
    for filt_tup in make_subsets(data, ["opto_pattern", "stim_type"]):
        filt_df = filt_tup[-1]
        if filt_tup[0] != _target_id or filt_tup[1] != _stim_name:
            ax_idx += 1
            # make the target match the current opto_pattern and stim type in concurrent loops
            _target_id = filt_tup[0]
            _stim_name = filt_tup[1]

        ax = axes[ax_idx]

        ax = plot_wheel_profile(
            data=filt_df,
            ax=ax,
            seperate_by=seperate_by,
            include_misses=include_misses,
            plot_speed=plot_speed,
            time_reset=time_reset,
            mpl_kwargs=mpl_kwargs,
            **kwargs,
        )

        ax.set_title(f"{filt_tup[1]}_{filt_tup[0]}")

    return fig


def plot_wheel_heatmap(
    data: pl.DataFrame,
    ax: plt.Axes = None,
    seperate_by: list[str] = ["contrast"],
    time_reset: str = "t_vstimstart_rig",
    plot_speed: bool = True,
    mpl_kwargs: dict = None,
    **kwargs,
) -> plt.Axes:
    if mpl_kwargs is None:
        mpl_kwargs = {}

    clr = Color(task="detection")
    override_plots()

    trace = WheelTrace()
    trace_interp_freq = kwargs.get("interp_freq", 5)

    time_lims = kwargs.pop("time_lims", None)
    if time_lims is None:
        time_lims = [-200, 1100]

    traj_lims = kwargs.pop("traj_lims", None)
    if traj_lims is None:
        traj_lims = [None, None]

    if ax is None:
        fig = plt.figure(figsize=mpl_kwargs.pop("figsize", (8, 8)))
        ax = fig.add_subplot(1, 1, 1)

    plot_data = data.filter(pl.col("outcome") != "early")

    for filt_tup in make_subsets(plot_data, seperate_by):
        filt_df = filt_tup[-1]
        sep = filt_tup[0]

        _longest_trace_len = 0
        trials_wheel_list = []
        _rig_response_rad_list = []
        for i, trial in enumerate(filt_df.iter_rows(named=True)):
            wheel_t = np.array(trial["wheel_t"])
            wheel_tick = np.array(trial["wheel_pos"])
            reset_time_point = trial[time_reset]

            _, _, t_interp, tick_interp = trace.reset_and_interpolate(
                wheel_t, wheel_tick, reset_time_point, trace_interp_freq
            )
            # convert the interpolation to rad
            interp_pos = trace.cm_to_rad(trace.ticks_to_cm(tick_interp))

            speed = (
                np.abs(trace.get_filtered_velocity(interp_pos, trace_interp_freq))
                * 1000
            )  # rad/s
            idx_in_time_range = np.where(
                (t_interp >= time_lims[0]) & (t_interp <= time_lims[1])
            )

            if len(idx_in_time_range):
                time_window = t_interp[idx_in_time_range]
                if plot_speed:
                    speed_window = speed[idx_in_time_range]
                    y_in_rads = speed_window
                    if trial["rig_response_tick"] is not None:
                        speed_thresh = trial["rig_response_tick"] / (
                            trial["median_loop_time"] * 5
                        )  # 5 is wheelbuffer
                        speed_thresh = (
                            trace.cm_to_rad(trace.ticks_to_cm(speed_thresh)) * 1000
                        )  # rad/s
                        _rig_response_rad_list.append(speed_thresh)
                else:
                    y_in_rads = interp_pos[idx_in_time_range]
                    if trial["rig_response_tick"] is not None:
                        pos_thresh = trace.cm_to_rad(
                            trace.ticks_to_cm(trial["rig_response_tick"])
                        )
                        _rig_response_rad_list.append(pos_thresh)

                trials_wheel_list.append(y_in_rads.tolist())
                # adjusting the longest
                if len(y_in_rads) >= _longest_trace_len:
                    _longest_trace_len = len(y_in_rads)
                    _longest_time = time_window

            # make a matrix to avreage over the rows, pad with None until reaching longest trace
            all_traces_mat = np.array(
                [
                    xi + [None] * (_longest_trace_len - len(xi))
                    for xi in trials_wheel_list
                ],
                dtype=float,
            )
