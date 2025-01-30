import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from ....psychophysics.wheel.wheelTrace import WheelTrace


def plot_trial(
    trial_row: pl.DataFrame, ax: plt.Axes = None, mpl_kwargs: dict = None, **kwargs
) -> plt.Axes:
    """ """
    if ax is None:
        fig = plt.figure(figsize=kwargs.pop("figsize", (15, 8)))
        ax = fig.add_subplot(1, 1, 1)
    
    if len(trial_row) != 1:
        raise ValueError(f"trial_row needs to be of size 1, got {len(trial_row)}")

    _trial = trial_row.to_dict(as_series=False)
    _trial = {k: v[0] for k, v in _trial.items()}
    interp_freq = kwargs.get("interp_freq", 5)
    trace = WheelTrace()
    
    # look for a column that has t_*start_rig
    _start_name = [
        s for s in _trial.keys() if s.startswith("t_") and s.endswith("start_rig")
    ][0]
    
    # timeframe reset value
    if _trial[_start_name] is not None:
        reset_time = _trial[_start_name]
    else:
        if _trial["outcome"] == "early":
            reset_time = _trial["t_trialinit"] + _trial["duration_blank"]
        else:
            reset_time = _trial["t_vstimstart"]
    
    # wheel
    wheel_t = np.array(_trial["wheel_t"])
    wheel_tick = np.array(_trial["wheel_pos"])
    
    if not len(wheel_t):
        print("NO WHEEL MOVEMENT IN TRIAL")
        return None
    
    reset_t,reset_tick, t_interp, tick_interp = trace.reset_and_interpolate(wheel_t, 
                                                                            wheel_tick, 
                                                                            reset_time, 
                                                                            interp_freq)

    # get radians
    wheel_pos_rad = trace.cm_to_rad(trace.ticks_to_cm(reset_tick))
    # convert the interpolation
    interp_pos = trace.cm_to_rad(trace.ticks_to_cm(tick_interp))
    
    # get speed
    speed = np.abs(trace.get_filtered_velocity(interp_pos,interp_freq)) * 1000
    
    # plot interp
    ax.plot(t_interp, interp_pos)

    # plot data
    ax.plot(reset_t, wheel_pos_rad)

    # get the epoch time points, order them and reset the timeframe
    epochs_time_points = [
        (c, v) for c, v in _trial.items() if c.startswith("t_") and v is not None
    ]

    idx_sorted = np.argsort([v[1] for v in epochs_time_points])
    sorted_epoch_time_points = [epochs_time_points[i] for i in idx_sorted]
    reset_epoch_time_points = [(n, v - reset_time) for n, v in sorted_epoch_time_points]

    # plot epoch start/end
    y_max = np.max(wheel_pos_rad)
    for name, t_val in reset_epoch_time_points:
        ax.axvline(t_val, color="k", ymax=y_max + 0.1)
        ax.scatter(t_val, y_max + 0.2, marker="v", c="k")
        name = name.split("_")[1]
        ax.text(t_val, y_max + 0.25, name, fontdict={"fontsize": 20})

    # plot movements
    mov_dict = trace.get_movements(
        t_interp,
        interp_pos,
        freq=interp_freq,
        pos_thresh=kwargs.get("pos_thresh", 0.0005),
        t_thresh=kwargs.get("t_thresh", 1),
        min_gap=kwargs.get("min_gap", 30),
        pos_thresh_onset=kwargs.get("pos_thresh_onset", 1.5),
        min_dur=kwargs.get("min_dur", 20),
    )

    for i in range(len(mov_dict["onsets"])):
        _t = mov_dict["onsets"][i]
        ax.scatter(_t[1], interp_pos[int(_t[0])], color="b")
        _e = mov_dict["offsets"][i]
        ax.scatter(_e[1], interp_pos[int(_e[0])], color="r")
        
    ax_speed = ax.twinx()
    ax_speed.plot(t_interp,speed,c='r')
        
    _resp = _trial["rig_response_time"]
    ax.axvline(_resp)
    for i in range(len(mov_dict["onsets"])):
        _on = mov_dict["onsets"][i, 1]
        _off = mov_dict["offsets"][i, 1]
        if _resp < _off and _resp >= _on:
            # this is the movement that registered the animals answer
            ax.axvline(_on)
            break
        # sometimes the response is in between two movements
        elif _resp >= _off and _resp <= _off + 100:
            ax.axvline(_on)
            break
