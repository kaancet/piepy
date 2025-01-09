import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from ...psychophysics.wheelTrace import WheelTrace


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

    # wheel
    wheel_t = np.array(_trial["wheel_t"])
    wheel_pos = np.array(_trial["wheel_pos"])

    trace = WheelTrace()
    trace.init_interpolator(wheel_t, wheel_pos)
    interp_freq = kwargs.get("interp_freq", 5)
    t_interp, tick_interp = trace.interpolate_trace(wheel_t, wheel_pos, interp_freq)

    # plot interp
    pos_interp = trace.cm_to_rad(trace.ticks_to_cm(tick_interp))
    ax.plot(t_interp, pos_interp)

    # plot data
    pos_data = trace.cm_to_rad(trace.ticks_to_cm(wheel_pos))
    ax.plot(wheel_t, pos_data)

    # look for a column that has t_*start_rig
    _start_name = [
        s for s in _trial.keys() if s.startswith("t_") and s.endswith("start_rig")
    ][0]

    # timeframe reset value
    if _trial[_start_name] is not None:
        reset_time = _trial[_start_name]
    else:
        reset_time = _trial["t_trialstart"]

    # get the epoch time points, order them and reset the timeframe
    epochs_time_points = [
        (c, v) for c, v in _trial.items() if c.startswith("t_") and v is not None
    ]

    idx_sorted = np.argsort([v[1] for v in epochs_time_points])
    sorted_epoch_time_points = [epochs_time_points[i] for i in idx_sorted]
    reset_epoch_time_points = [(n, v - reset_time) for n, v in sorted_epoch_time_points]

    # plot epoch start/end
    y_max = np.max(pos_data)
    for name, t_val in reset_epoch_time_points:
        ax.axvline(t_val, color="k", ymax=y_max + 0.1)
        ax.scatter(t_val, y_max + 0.2, marker="v", c="k")
        name = name.split("_")[1]
        ax.text(t_val, y_max + 0.25, name, fontdict={"fontsize": 20})

    # plot movements
    mov_dict = trace.get_movements(
        t_interp,
        pos_interp,
        freq=interp_freq,
        pos_thresh=kwargs.get("pos_thresh", 0.02),
        t_thresh=kwargs.get("t_thresh", 0.5),
    )

    for i in range(len(mov_dict["onsets"])):
        _t = mov_dict["onsets"][i]
        ax.scatter(_t[1], pos_interp[int(_t[0])], color="b")
        _e = mov_dict["offsets"][i]
        ax.scatter(_e[1], pos_interp[int(_e[0])], color="r")
