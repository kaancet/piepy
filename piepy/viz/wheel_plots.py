from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats
from collections import defaultdict

from piepy.psychophysics.wheelTrace import WheelTrace

from .base import PlotResult, _need, _resolve


def _aligned_trial_traces(
    trace, wheel_t, wheel_pos, reset_times, *, time_range, common_t, plot_speed, interp_freq
) -> list:
    """Each trial's speed (or position) resampled onto ``common_t``; out-of-range -> NaN, empties skipped."""
    out = []
    for t, pos, reset_t in zip(wheel_t, wheel_pos, reset_times):
        if reset_t is None or t is None or not len(t):
            continue
        processed = trace.load(t=t, pos=pos).process(reset_time=reset_t, freq=interp_freq)
        idx = np.where((processed["t"] >= time_range[0]) & (processed["t"] <= time_range[1]))[0]
        if not idx.size:
            continue
        y = np.abs(processed["velocity"][idx]) if plot_speed else processed["pos"][idx]
        out.append(np.interp(common_t, processed["t"][idx], y, left=np.nan, right=np.nan))
    return out


def _mean_trace(traces) -> np.ndarray:
    """NaN-ignoring mean over a list of aligned traces; None if the list is empty."""
    return np.nanmean(np.vstack(traces), axis=0) if traces else None


def wheel_profile(
    data,
    separate_by: str = "contrast",
    *,
    time_reset: str = "t_vstimstart_rig",
    time_range: list[float] | None = None,
    plot_speed: bool = True,
    average_over: str | None = None,
    interp_freq: int = 5,
    ax=None,
    **style,
) -> PlotResult:
    """Average wheel trace over time, one curve per ``separate_by`` level (with a SEM band).

    Each trial's wheel trace is zeroed at ``time_reset`` (stim onset), clipped to ``time_range`` (ms)
    and resampled onto a common time axis. The curve is the mean over trials, the band its SEM.

    ``average_over`` (e.g. ``animalid`` / ``session``) switches to a **two-stage** average: first the
    mean trace per subject, then the mean of those per-subject mean traces. The band is then the SEM
    **across subjects** -- every subject weighted equally, no pseudoreplication.

    Args:
        data: a Run / Session / Hub / trial-table DataFrame (``_resolve`` handles all four).
        separate_by: column whose levels each get their own curve (e.g. ``contrast``).
        time_reset: per-trial timestamp column each trace is zeroed on (default rig stim onset).
        time_range: ``[start, end]`` ms around the reset; defaults to ``[-200, 1500]``.
        plot_speed: True -> ``|velocity|``; False -> position.
        average_over: subject column for the two-stage average; ``None`` -> pool all trials.
        interp_freq: resample rate passed to :class:`WheelTrace` and used for the common time grid.
        ax / **style: forwarded to behaviz (``fill_*`` / ``line_*`` component overrides).

    Returns:
        PlotResult(data=<per-level mean/SEM trace frame>, figure=(fig, ax)).
    """
    import behaviz as bv

    spec = bv.load_preset(style.pop("preset", "reaction_distribution"))

    style_overrides = bv.split_styles(
        style,
        components=("fill", "line"),
        defaults={
            "fill": {
                "color": "#090909",
                "alpha": 0.3,
                "linewidth": 0,
            },
            "line": {
                "linewidth": 2,
                "color": "#090909",
            },
        },
    )

    df = _resolve(data)

    _need(
        df,
        ["wheel_t", "wheel_pos", separate_by, time_reset, *([average_over] if average_over else [])],
        plot="wheel profile plot",
    )  # structured error naming any missing column

    if time_range is None:
        time_range = [-200, 1500]

    # check if time reset has non null
    if not df[time_reset].drop_nulls().len():
        raise ValueError(f"The column {time_reset} used for time_reset has only null values!")

    trace = WheelTrace()
    common_t = np.arange(time_range[0], time_range[1], 1 / interp_freq)

    # to pass to mean func
    kw = dict(time_range=time_range, common_t=common_t, plot_speed=plot_speed, interp_freq=interp_freq)

    group_cols = [separate_by, average_over] if average_over else [separate_by]
    grouped_df = (
        df.group_by(group_cols)
        .agg(pl.col(time_reset), pl.col("wheel_t"), pl.col("wheel_pos"))
        .drop_nulls(group_cols)
        .sort(group_cols)
    )

    # rows to average within each level: individual trials (pooled) or per-subject mean traces (2-stage)
    level_rows: dict[object, list] = defaultdict(list)
    for r in grouped_df.iter_rows(named=True):
        trials = _aligned_trial_traces(trace, r["wheel_t"], r["wheel_pos"], r[time_reset], **kw)
        if average_over:
            # stage 1: this subject's mean trace at this level
            subj_mean = _mean_trace(trials)
            if subj_mean is not None:
                level_rows[r[separate_by]].append(subj_mean)
        else:
            level_rows[r[separate_by]].extend(trials)

    # stage 2: mean + SEM across the rows (trials, or subjects when averaging) of each level
    plot_dict = defaultdict(list)
    for level in sorted(level_rows):
        rows = level_rows[level]
        if not rows:  # a level whose every trial fell outside time_range / had a null reset
            continue
        mat = np.vstack(rows)
        avg = np.nanmean(mat, axis=0)
        sem = stats.sem(mat, axis=0, nan_policy="omit") if mat.shape[0] > 1 else np.full(avg.shape, np.nan)
        plot_dict["t"].extend(common_t)
        plot_dict["y"].extend(avg)
        plot_dict["sem_lo"].extend(avg - sem)
        plot_dict["sem_hi"].extend(avg + sem)
        plot_dict[separate_by].extend([level] * len(common_t))

    if not plot_dict:
        raise ValueError("wheel profile: no wheel data fell inside time_range after resetting.")

    plot_df = pl.DataFrame(plot_dict)
    fig, ax = bv.plot_fill_between(
        data=plot_df,
        x="t",
        y1="sem_lo",
        y2="sem_hi",
        hue=separate_by,
        group=separate_by,
        ax=ax,
        spec=spec,
        **style_overrides["fill"],
    )
    fig, ax = bv.plot_line(
        data=plot_df,
        x="t",
        y="y",
        hue=separate_by,
        group=separate_by,
        ax=ax,
        spec=spec,
        **style_overrides["line"],
    )
    return PlotResult(data=plot_df, figure=(fig, ax))


def wheel_heatmap(
    data,
    sort_by: list[str],
    *,
    time_reset: str = "t_vstimstart_rig",
    time_range: list[float] | None = None,
    plot_speed: bool = True,
    normalise: bool = True,
    interp_freq: int = 5,
    ax=None,
    **style,
) -> PlotResult:
    """_summary_

    Args:
        data (_type_): _description_
        sort_by (list[str]):
        time_reset (str, optional): _description_. Defaults to "t_vstimstart_rig".
        time_range (list[float] | None, optional): _description_. Defaults to None.
        plot_speed (bool, optional): _description_. Defaults to True.
        interp_freq (int, optional): _description_. Defaults to 5.
        ax (_type_, optional): _description_. Defaults to None.

    Returns:
        PlotResult: _description_
    """

    import behaviz as bv

    spec = bv.load_preset(style.pop("preset", "wheel_heatmap"))

    style_overrides = bv.split_styles(
        style,
        components=("image", "bar"),
        defaults={
            "image": {
                "cmap": "inferno",
            },
            "bar": {
                "color": "#090909",
            },
        },
    )

    df = _resolve(data)

    _need(
        df,
        ["wheel_t", "wheel_pos", *sort_by, time_reset],
        plot="wheel heat-map plot",
    )  # structured error naming any missing column

    if time_range is None:
        time_range = [-200, 1500]

    # check if time reset has non null
    if not df[time_reset].drop_nulls().len():
        raise ValueError(f"The column {time_reset} used for time_reset has only null values!")

    trace = WheelTrace()
    common_t = np.arange(time_range[0], time_range[1], 1 / interp_freq)

    # to pass to mean func
    kw = dict(time_range=time_range, common_t=common_t, plot_speed=plot_speed, interp_freq=interp_freq)

    # order by sort_by
    ordered_df = df.sort(sort_by, descending=[True, False]).drop_nulls(sort_by[0])
    # create the
    trials = np.array(
        _aligned_trial_traces(trace, ordered_df["wheel_t"], ordered_df["wheel_pos"], ordered_df[time_reset], **kw)
    )

    label_suffix = ""
    if normalise:
        trials_max = np.nanmax(trials)
        trials = trials / trials_max
        label_suffix = "normalised"

    f, ax = bv.plot_image(
        trials,
        origin="upper",
        extent=[
            common_t[0],
            common_t[-1],
            0,
            len(ordered_df),
        ],
        colorbar=f"{label_suffix} speed (rad/s)" if plot_speed else f"{label_suffix} pos (rad)",
        ax=ax,
        spec=spec,
        **style_overrides["image"],
    )

    bottom = 0
    uniques = ordered_df[sort_by[0]].unique(maintain_order=True).to_list()[::-1]
    unique_counts = ordered_df[sort_by[0]].unique_counts().to_list()[::-1]
    for u_n, u_s in zip(uniques, unique_counts):
        f, ax = bv.plot_bar(
            time_range[-1], u_s - 1, bottom=bottom + 1, width=10, ax=ax, spec=spec, **style_overrides["bar"]
        )

        f, ax = bv.plot_text(
            time_range[-1] + 10,
            bottom + u_s // 2,
            f"{u_n}",
            ha="left",
            va="center",
            color="#010101",
            ax=ax,
            spec=spec,
        )
        bottom += u_s

    return PlotResult(data=ordered_df, figure=(f, ax))
