"""Whole-session view of a wheel-detection session, from its temporaldata streams.

Per-trial-normalised wheel traces over the session clock, with licks and rewards below and the
trials as coloured hbars above.
"""

from __future__ import annotations

import numpy as np

from piepy.psychophysics.tasks.wheel_detection.wheelDetectionStreams import (
    WheelDetectionStreams,
)
from ...psychophysics.wheelTrace import WheelTrace
from ..base import PlotResult

_OUTCOME_COLOR = {
    "hit": "#1B8A3A",
    "miss": "#990000",
    "early": "#DD7703",
    "catch": "#4C4CC7",
}


def session_stream(
    data, *, slice_times: list[float] | None = None, ax=None, **style
) -> PlotResult:
    """Plot a whole detection session: wheel traces (normalised per trial) + licks/rewards + trials.

    Args:
        data: a Session or a session-clock trial-table DataFrame (passed to WheelDetectionStreams).
    """
    import behaviz as bv

    streams = WheelDetectionStreams(data).build()

    if slice_times:
        sliced = streams.slice(slice_times[0], slice_times[1], reset_origin=False)
    else:
        sliced = streams

    wt = np.asarray(sliced.wheel.timestamps, float)
    wp = np.asarray(sliced.wheel.position, float)
    starts = np.asarray(sliced.trials.start, float)
    ends = np.asarray(sliced.trials.end, float)
    outcomes = np.asarray(sliced.trials.outcome)
    # stim_starts = np.asarray(sliced.stim.start, float)
    # stim_ends = np.asarray(sliced.stim.end, float)

    fig = None
    # wheel trace, min-max normalised within each trial so amplitudes are comparable
    trace = WheelTrace()
    for s, e in zip(starts, ends):
        m = (wt >= s) & (wt < e)
        trial_wt = wt[m]
        trial_wp = wp[m]
        if trial_wt.size < 2:
            continue

        res = trace.load(trial_wt, trial_wp).process(
            reset_time=trial_wt[0], freq=3, units="rad"
        )

        fig, ax = bv.plot_line(
            res["t"] + trial_wt[0],
            np.abs(res["velocity"] * 1000),
            ax=ax,
            color="#090909",
            linewidth=1,
        )

    # licks and rewards on their own lanes below the traces
    if "licks" in sliced.keys():
        lk = np.asarray(sliced.licks.timestamps, float)
        fig, ax = bv.plot_scatter(
            lk, np.full_like(lk, -0.1), ax=ax, marker="|", s=50, color="#1EB9E4"
        )
    if "reward" in sliced.keys():
        rw = np.asarray(sliced.reward.timestamps, float)
        fig, ax = bv.plot_scatter(
            rw, np.full_like(rw, 0), ax=ax, marker="v", s=50, color="#1B8A3A"
        )

    # trials as hbars above, coloured by outcome
    for s, e, o in zip(starts, ends, outcomes):
        fig, ax = bv.plot_hbar(
            1.15,
            e - s,
            left=s,
            height=0.1,
            ax=ax,
            color=_OUTCOME_COLOR.get(o, "#ACACAC"),
            alpha=0.6,
        )

    return PlotResult(data=sliced, figure=(fig, ax))
