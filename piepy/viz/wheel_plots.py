from __future__ import annotations
from typing import Literal
import numpy as np
import polars as pl
from scipy import stats
from collections import defaultdict

from piepy.psychophysics.wheelTrace import WheelTrace, match_response_movement

from .base import PlotResult, _need, _resolve

# same detector settings the trial parser uses on wheelDetectionTrial.py
DEFAULT_MOVEMENT_PARAMS = {
    "pos_thresh": 0.025,
    "t_thresh": 20,
    "min_dur": 40,
    "min_gap": 40,
    "pos_thresh_onset": 0.0092,
}


def _with_resp(df, response_col):
    """Add ``_resp``: ``response_col``, else the rig response time with the state time as fallback
    (what the trial parser matches against the wheel trace)."""
    expr = pl.col(response_col) if response_col else pl.coalesce("rig_response_time", "state_response_time")
    return df.with_columns(expr.alias("_resp"))


def _process_and_match(trace, t, pos, reset_t, resp_t, *, interp_freq, movement_params):
    """Process one trial and match the movement that produced its response -> (processed, RTResult)."""
    processed = trace.load(t=t, pos=pos).process(reset_time=reset_t, freq=interp_freq, units="rad", **movement_params)
    return processed, match_response_movement(processed["movements"], resp_t)


def _aligned_trial_traces(
    trace: WheelTrace,
    wheel_t: np.ndarray,
    wheel_pos: np.ndarray,
    reset_times: np.ndarray,
    common_t: np.ndarray,
    *,
    time_range: list | None = None,
    plot_speed: bool = True,
    interp_freq: float = 1,
    resp_times: str | None = None,
    movement_params: dict | None = None,
    exclude_anticipatory: bool = True,
) -> np.ndarray:
    """Resample every trial's wheel speed (or position) onto one shared time axis.

    Each trial is processed on its own (interpolated onto a regular grid, velocity computed,
    movements detected), optionally re-zeroed in time, cropped to ``time_range`` and interpolated
    onto ``common_t``. The result is a trials-by-time matrix: row ``i`` is input trial ``i``.

    Two alignment modes:
    1. **Stimulus-aligned** (``resp_times is None``): time 0 is ``reset_t`` for each trial (usually
      stimulus onset).
    2. **Movement-aligned** (``resp_times`` given): the movement that produced the response is
      found (``match_response_movement``, using ``movement_params``) and time 0 becomes that
      movement's onset. Position traces also have the position at onset subtracted, so every
      trial starts at 0. Trials with no response, no matching movement, or (by default) an
      anticipatory onset are dropped.

    Args:
        trace: One reusable ``WheelTrace`` object. It holds no data of its own here and each trial's arrays
            are loaded into it with ``trace.load(t, pos)`` before processing.
        wheel_t: Per-trial raw wheel timestamps (ms), one array per trial.
        wheel_pos: Per-trial raw wheel positions (encoder ticks), same length and order as
            ``wheel_t``.
        reset_times: Per-trial time (ms, same clock as ``wheel_t``) that is set to 0 before any
            movement alignment, e.g. ``t_vstimstart_rig``. A ``None`` entry skips that trial.
        common_t: 1-D array of ms values every trial is interpolated onto, usually
            ``np.arange(start, end, 1 / interp_freq)``.
        time_range: ``[start, end]`` in ms, inclusive, relative to time 0 (stimulus onset, or
                    movement onset when movement-aligned). Samples outside it are cropped away. Defaults to None.
                    If None, uses the range of common_t.
        plot_speed: ``True`` returns absolute speed (``abs(velocity)``, rad per ms.
            ``False`` returns position in rad (relative to the position at
            movement onset when movement-aligned). Defaults to True.
        interp_freq: Samples per ms of the grid used when interpolating each trace
            (``freq`` in ``WheelTrace.process``). Defaults to 1.
        resp_times: Per-trial response times (ms, same clock as ``reset_times``), same order as
            ``wheel_t``. ``None`` selects stimulus alignment; any list selects movement alignment.
            A ``None`` entry means "no response" and the trial is dropped.
        movement_params: Keyword arguments for the movement detector
            (``pos_thresh``, ``t_thresh``, ``min_dur``, ``min_gap``, ``pos_thresh_onset``).
            Only used when ``resp_times`` is given.
        exclude_anticipatory: Drop trials whose matched movement starts before the minimum
            plausible reaction time (150 ms, or before the stimulus). Only used when ``resp_times``
            is given. Defaults to ``True``.

    Returns:
        Matrix of shape ``(len(wheel_t), len(common_t))``; row ``i`` belongs to input trial ``i``.
        Entries are NaN (missing value) wherever that trial's recording does not cover the time
        point, so use NaN-ignoring statistics (``np.nanmean``). A trial that is skipped leaves an
        entire row of NaN: ``reset_t`` or the wheel data is missing or empty, no samples fall
        inside ``time_range``, or, in movement mode, no clean response movement is matched. The
        matrix therefore keeps the input's length and order, and dropped trials are the rows
        where ``np.isnan(out).all(axis=1)``. It does not say why a trial was dropped.
    """
    out = np.full((len(wheel_t), len(common_t)), np.nan)

    if time_range is None:
        time_range = [common_t[0], common_t[-1]]

    for i, (t, pos, reset_t) in enumerate(zip(wheel_t, wheel_pos, reset_times)):
        if reset_t is None or t is None or not len(t):
            continue
        shift = base = 0.0
        if resp_times is None:
            processed = trace.load(t=t, pos=pos).process(reset_time=reset_t, freq=interp_freq)
        else:
            processed, rt = _process_and_match(
                trace, t, pos, reset_t, resp_times[i],
                interp_freq=interp_freq, movement_params=movement_params,
            )  # fmt: skip
            if rt.reaction_time is None or (exclude_anticipatory and rt.anticipatory):
                continue
            shift = rt.reaction_time
            base = np.interp(shift, processed["t"], processed["pos"])
        times = processed["t"] - shift
        idx = np.where((times >= time_range[0]) & (times <= time_range[1]))[0]
        if not idx.size:
            continue
        y = np.abs(processed["velocity"][idx]) if plot_speed else processed["pos"][idx] - base
        out[i] = np.interp(common_t, times[idx], y, left=np.nan, right=np.nan)
    return out


def wheel_profile_compute(
    data: pl.DataFrame,
    separate_by: str = "contrast",
    *,
    time_reset: str = "t_vstimstart_rig",
    time_range: list = None,
    plot_speed: bool = True,
    average_over: str | None = None,
    interp_freq: float = 1,
    align: Literal["stim", "movement"] = "stim",
    movement_params: dict | None = None,
    response_col: str | None = None,
    exclude_anticipatory: bool = True,
) -> pl.DataFrame:
    """Processes every trace, mean+SEM per ``separate_by``. Returns a dataframe
    [t, y, sem_lo, sem_hi, n, <separate_by>] (n = trials contributing at that t).

    Args:
        data: A polars dataframe that contains the data
        separate_by: Name of the column in the ``data`` to group by and separate the trials. Defaults to ``contrast``.
        time_reset: Time point (ms, same clock as ``wheel_t``) that is set to 0 before any
            movement alignment, e.g. ``t_vstimstart_rig``. Defaults to ``t_vstimstart_rig``.
        time_range: ``[start, end]`` in ms, inclusive, relative to time 0 (stimulus onset, or
                    movement onset when movement-aligned). Samples outside it are cropped away. Defaults to None.
                    If None, uses the range ``[-200, 1500]``(stim alignment) or ``[-200, 800]``(movement alignment).
        plot_speed: ``True`` returns absolute speed (``abs(velocity)``, rad per ms.
                    ``False`` returns position in rad (relative to the position at
                    movement onset when movement-aligned). Defaults to True.
        average_over: Name of the column to per-value average the data over, i.e. ``animalid`` for animals.
                    Defaults to ``None``. If ``None`` trials are pooled
        interp_freq: Samples per ms of the grid used when interpolating each trace
            (``freq`` in ``WheelTrace.process``). Defaults to 1.
        align: Whether to align the traces to stimulus or movement onset. Defaults to ``stim``.
        movement_params: Keyword arguments for the movement detector
            (``pos_thresh``, ``t_thresh``, ``min_dur``, ``min_gap``, ``pos_thresh_onset``).
            Only used when ``resp_times`` is given.
        response_col: Name of the column that contains response times (ms, same clock as ``time_reset``), same order as
                    ``wheel_t``. ``None`` selects stimulus alignment; any list selects movement alignment.
                    A ``None`` entry means "no response" and the trial is dropped.
        exclude_anticipatory: Drop trials whose matched movement starts before the minimum
                    plausible reaction time (150 ms, or before the stimulus). Only used when ``resp_times``
                    is given. Defaults to ``True``.

    Returns:
        Polars data frame of shape ``(len(separate_by), 6)``:
        [t, y, sem_lo, sem_hi, n, <separate_by>] (n = trials contributing at that t)
    """
    if align not in ("stim", "movement"):
        raise ValueError(f"align must be 'stim' or 'movement', got {align!r}")
    by_movement = align == "movement"
    df = _resolve(data)
    _need(
        df,
        [
            "wheel_t",
            "wheel_pos",
            separate_by,
            time_reset,
            *([average_over] if average_over else []),
            *([response_col or "state_response_time"] if by_movement else []),
        ],
        plot="wheel profile plot",
    )
    if by_movement:
        df = _with_resp(df, response_col)
    if time_range is None:
        time_range = [-200, 800] if by_movement else [-200, 1500]
    if not df[time_reset].drop_nulls().len():
        raise ValueError(f"The column {time_reset} used for time_reset has only null values!")

    trace = WheelTrace()
    common_t = np.arange(time_range[0], time_range[1], 1 / interp_freq)
    kw = dict(
        time_range=time_range,
        common_t=common_t,
        plot_speed=plot_speed,
        interp_freq=interp_freq,
    )

    group_cols = [separate_by, average_over] if average_over else [separate_by]
    grouped_df = (
        df.group_by(group_cols)
        .agg(
            pl.col(time_reset),
            pl.col("wheel_t"),
            pl.col("wheel_pos"),
            *([pl.col("_resp")] if by_movement else []),
        )
        .drop_nulls(group_cols)
        .sort(group_cols)
    )

    level_rows = defaultdict(list)
    for r in grouped_df.iter_rows(named=True):
        if by_movement:
            kw["resp_times"] = r["_resp"]
            kw["movement_params"] = movement_params or DEFAULT_MOVEMENT_PARAMS
            kw["exclude_anticipatory"] = exclude_anticipatory
        trials = _aligned_trial_traces(trace, r["wheel_t"], r["wheel_pos"], r[time_reset], **kw)
        trials = trials[~np.isnan(trials).all(axis=1)]  # drop skipped trials (all-NaN rows)
        if average_over:
            subj_mean = np.nanmean(trials, axis=0) if len(trials) else None
            if subj_mean is not None:
                level_rows[r[separate_by]].append(subj_mean)
        else:
            level_rows[r[separate_by]].extend(trials)

    plot_dict = defaultdict(list)
    for level in sorted(level_rows):
        rows = level_rows[level]
        if not rows:
            continue
        mat = np.vstack(rows)
        avg = np.nanmean(mat, axis=0)
        sem = stats.sem(mat, axis=0, nan_policy="omit") if mat.shape[0] > 1 else np.full(avg.shape, np.nan)
        plot_dict["t"].extend(common_t)
        plot_dict["y"].extend(avg)
        plot_dict["sem_lo"].extend(avg - sem)
        plot_dict["sem_hi"].extend(avg + sem)
        plot_dict["n"].extend((~np.isnan(mat)).sum(axis=0))
        plot_dict[separate_by].extend([level] * len(common_t))

    if not plot_dict:
        raise ValueError("wheel profile: no wheel data fell inside time_range after resetting.")
    return pl.DataFrame(plot_dict)


def wheel_profile_draw(
    plot_df: pl.DataFrame,
    separate_by: str = "contrast",
    *,
    ax=None,
    **style,
):
    """The drawing half of wheel_profile: fill + line from the precomputed mean/SEM frame.
    Separated from compute, perhaps for a live dashboard

    Args:
        plot_df: Dataframe from ``wheel_profile_compute`` in the shape ``(len(separate_by),6)``
        separate_by: Name of the column in the ``data`` to group by and separate the trials. Defaults to ``contrast``.

    Returns:
        A figure and axes that has the drawn plot
    """
    import behaviz as bv

    spec = bv.load_preset(style.pop("preset", "reaction_distribution"))
    style_overrides = bv.split_styles(
        style,
        components=("fill", "line"),
        defaults={
            "fill": {"color": "#090909", "alpha": 0.3, "linewidth": 0},
            "line": {"linewidth": 2, "color": "#090909"},
        },
    )
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
    return fig, ax


def wheel_profile(
    data: pl.DataFrame,
    separate_by: str = "contrast",
    *,
    time_reset: str = "t_vstimstart_rig",
    time_range: list | None = None,
    plot_speed: bool = True,
    average_over: str | None = None,
    interp_freq: float = 1,
    align: str = "stim",
    movement_params: dict | None = None,
    response_col: str | None = None,
    exclude_anticipatory: bool = True,
    ax=None,
    **style,
) -> PlotResult:
    plot_df = wheel_profile_compute(
        data,
        separate_by,
        time_reset=time_reset,
        time_range=time_range,
        plot_speed=plot_speed,
        average_over=average_over,
        interp_freq=interp_freq,
        align=align,
        movement_params=movement_params,
        response_col=response_col,
        exclude_anticipatory=exclude_anticipatory,
    )
    fig, ax = wheel_profile_draw(plot_df, separate_by, ax=ax, **style)
    return PlotResult(data=plot_df, figure=(fig, ax))


def wheel_onsets_compute(
    data: pl.DataFrame,
    separate_by: str = "contrast",
    *,
    time_reset: str = "t_vstimstart_rig",
    interp_freq: float = 1,
    response_col: str | None = None,
    movement_params: dict | None = None,
    exclude_anticipatory: bool = True,
) -> pl.DataFrame:
    """Calculates the onset, which is the start (ms from ``time_reset``) of the movement that produced the response.

    Args:
        data: A polars dataframe that contains the data
        separate_by: Name of the column in the ``data`` to group by and separate the trials. Defaults to ``contrast``.
        time_reset: Time point (ms, same clock as ``wheel_t``) that is set to 0 before any
            movement alignment, e.g. ``t_vstimstart_rig``. Defaults to ``t_vstimstart_rig``.
        interp_freq: Samples per ms of the grid used when interpolating each trace
                    (``freq`` in ``WheelTrace.process``). Defaults to 1.
        response_col: Name of the column that contains response times (ms, same clock as ``time_reset``), same order as
                    ``wheel_t``. ``None`` selects stimulus alignment; any list selects movement alignment.
                    A ``None`` entry means "no response" and the trial is dropped.
        movement_params: Keyword arguments for the movement detector
                    (``pos_thresh``, ``t_thresh``, ``min_dur``, ``min_gap``, ``pos_thresh_onset``).
                    Only used when ``resp_times`` is given.
        exclude_anticipatory: Drop trials whose matched movement starts before the minimum
                    plausible reaction time (150 ms, or before the stimulus). Only used when ``resp_times``
                    is given. Defaults to ``True``.

    Returns:
        A dataframe with the shape ``(len(trials), 2)``, with columns [``separate_by``, onset].
        Trials without a response, without a matching movement, or (default) anticipatory are left out.
    """
    df = _resolve(data)
    _need(
        df,
        ["wheel_t", "wheel_pos", separate_by, time_reset, response_col or "state_response_time"],
        plot="wheel onsets",
    )
    df = _with_resp(df, response_col)
    trace = WheelTrace()
    mp = movement_params or DEFAULT_MOVEMENT_PARAMS
    rows = []
    for r in df.drop_nulls([separate_by, time_reset]).iter_rows(named=True):
        if r["wheel_t"] is None or not len(r["wheel_t"]):
            continue
        _, rt = _process_and_match(
            trace, r["wheel_t"], r["wheel_pos"], r[time_reset], r["_resp"],
            interp_freq=interp_freq, movement_params=mp,
        )  # fmt: skip
        if rt.reaction_time is None or (exclude_anticipatory and rt.anticipatory):
            continue
        rows.append({separate_by: r[separate_by], "onset": rt.reaction_time})
    if not rows:
        raise ValueError("wheel onsets: no trial had a matched response movement.")
    return pl.DataFrame(rows)


def onset_ecdf_compute(
    onsets: pl.DataFrame,
    separate_by: str = "contrast",
) -> pl.DataFrame:
    """Cumulative distribution function plot of onset times, per ``separate_by`` value.
    For each ``separate_by`` level, ``fraction`` at ``onset`` = share of that level's trials whose movement started
    at or before that time.

    Args:
        onsets: Onsets dataframe acquired from ``wheel_onsets_compute``.
        separate_by: Name of the column in the ``data`` to group by and separate the trials. Defaults to ``contrast``.

    Returns:
        [onset, fraction, <separate_by>].
    """
    return (
        onsets.sort(separate_by, "onset")
        .with_columns((pl.int_range(1, pl.len() + 1).over(separate_by) / pl.len().over(separate_by)).alias("fraction"))
        .select("onset", "fraction", separate_by)
    )


def onset_ecdf_draw(
    ecdf_df: pl.DataFrame,
    separate_by: str = "contrast",
    *,
    ax=None,
    **style,
):
    import behaviz as bv

    spec = bv.load_preset(style.pop("preset", "reaction_distribution"))
    style_overrides = bv.split_styles(
        style,
        components=("line",),
        defaults={"line": {"linewidth": 2, "color": "#090909", "drawstyle": "steps-post"}},
    )
    fig, ax = bv.plot_line(
        data=ecdf_df,
        x="onset",
        y="fraction",
        hue=separate_by,
        group=separate_by,
        ax=ax,
        spec=spec,
        **style_overrides["line"],
    )
    return fig, ax


def onset_ecdf(
    data: pl.DataFrame,
    separate_by: str = "contrast",
    *,
    time_reset: str = "t_vstimstart_rig",
    response_col: str | None = None,
    movement_params: dict | None = None,
    exclude_anticipatory: bool = True,
    interp_freq: float = 1,
    ax=None,
    **style,
) -> PlotResult:
    """Cumulative fraction of trials whose response movement has started, vs time from ``time_reset``."""
    onsets = wheel_onsets_compute(
        data,
        separate_by,
        time_reset=time_reset,
        response_col=response_col,
        movement_params=movement_params,
        exclude_anticipatory=exclude_anticipatory,
        interp_freq=interp_freq,
    )
    ecdf_df = onset_ecdf_compute(onsets, separate_by)
    fig, ax = onset_ecdf_draw(ecdf_df, separate_by, ax=ax, **style)
    return PlotResult(data=ecdf_df, figure=(fig, ax))


def wheel_heatmap(
    data,
    sort_by: list[str],
    *,
    time_reset: str = "t_vstimstart_rig",
    time_range: list[float] | None = None,
    plot_speed: bool = True,
    normalise: bool = True,
    interp_freq: int = 1,
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
        interp_freq (int, optional): _description_. Defaults to 1.
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
    kw = dict(
        time_range=time_range,
        common_t=common_t,
        plot_speed=plot_speed,
        interp_freq=interp_freq,
    )

    # order by sort_by
    ordered_df = df.sort(sort_by, descending=[True, False]).drop_nulls(sort_by[0])
    # create the
    # one row per ordered_df row; skipped trials stay as blank (all-NaN) rows
    trials = _aligned_trial_traces(
        trace,
        ordered_df["wheel_t"],
        ordered_df["wheel_pos"],
        ordered_df[time_reset],
        **kw,
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
        colorbar=(f"{label_suffix} speed (rad/s)" if plot_speed else f"{label_suffix} pos (rad)"),
        ax=ax,
        spec=spec,
        **style_overrides["image"],
    )

    bottom = 0
    uniques = ordered_df[sort_by[0]].unique(maintain_order=True).to_list()[::-1]
    unique_counts = ordered_df[sort_by[0]].unique_counts().to_list()[::-1]
    for u_n, u_s in zip(uniques, unique_counts):
        f, ax = bv.plot_bar(
            time_range[-1],
            u_s - 1,
            bottom=bottom + 1,
            width=10,
            ax=ax,
            spec=spec,
            **style_overrides["bar"],
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
