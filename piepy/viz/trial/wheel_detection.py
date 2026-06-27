"""Trial timeline for a single wheel-detection trial: wheel trace + event markers."""

from __future__ import annotations

import numpy as np
import polars as pl

from ..base import PlotResult, _resolve
from ...psychophysics.wheelTrace import WheelTrace


def trial_snapshot(data, trial_no: int, *, ax=None, **style) -> PlotResult:
    """Wheel position vs time for one trial, aligned to stimulus onset, with event markers.

    Two clocks, both zeroed at stimulus onset: state events use ``t_vstimstart``; the rig streams
    (wheel/lick/reward) use ``t_vstimstart_rig``. Early trials fall back to cue + blank.
    """
    import behaviz as bv

    style_overrides = bv.split_styles(
        style,
        components=("wheel", "lick"),
        defaults={
            "wheel": {"linewidth": 2, "color": "#090909"},
            "lick": {
                "marker": "|",
                "s": 250,
                "color": "#1EB9E4",
            },
        },
    )

    spec = bv.load_preset("trial_snapshot")

    row = _resolve(data).filter(pl.col("trial_no") == trial_no).row(0, named=True)
    title = f"{row['trial_no']}-{row['outcome']}"
    spec = spec.with_title(title)

    # always look for rig times first, if not use state times
    z_state = (
        row["t_vstimstart"] if row["t_vstimstart"] is not None else row["t_trialinit"] + (row["duration_blank"] or 0)
    )
    z_rig = row["t_vstimstart_rig"] if row["t_vstimstart_rig"] is not None else z_state

    # wheel trace (raw rig time -> shift onto the stim-onset frame)
    wheel_t = np.array(row["wheel_t"], float)
    wheel_pos = np.array(row["wheel_pos"], float)

    trace = WheelTrace(t=wheel_t, pos=wheel_pos)
    res = trace.process(reset_time=z_rig, freq=5, units="rad")

    fig, ax = bv.plot_line(res["t"], np.abs(res["velocity"] * 1000), spec=spec, ax=ax, **style_overrides["wheel"])

    # state-clock event markers, drawn only when present
    for label, col in [
        ("trial start", "t_trialstart"),
        ("cue", "t_trialinit"),
        ("stim on", "t_vstimstart"),
        ("stim off", "t_vstimend"),
        ("trial end", "t_trialend"),
    ]:
        if row[col] is not None:
            _, ax = bv.plot_vertical([row[col] - z_state], spec=spec, ax=ax, color="#4C4C4C")

            fig, ax = bv.plot_text(
                row[col] - z_state + 3,
                9,
                label,
                ax=ax,
                spec=spec,
                rotation=90,
                ha="left",
                va="bottom",
                color="#4C4C4C",
            )

    # reaction / response are already measured from stimulus onset
    for label, col in [("reaction", "reaction_time"), ("response", "response_time")]:
        if row[col] is not None:
            _, ax = bv.plot_vertical([row[col]], spec=spec, ax=ax, color="#990000")
            fig, ax = bv.plot_text(
                row[col] + 3,
                9,
                label,
                ax=ax,
                spec=spec,
                rotation=90,
                ha="left",
                va="bottom",
                color="#990000",
            )

    # licks (all times) and reward (only its first element is a time), rig clock
    if row["lick"]:
        licks = np.array(row["lick"], float) - z_rig
        _, ax = bv.plot_scatter(licks, np.zeros_like(licks), spec=spec, ax=ax, label="lick", **style_overrides["lick"])
    if row["reward"]:
        _, ax = bv.plot_scatter([row["reward"][0] - z_rig], [0], spec=spec, ax=ax, label="reward")

    # pre-stim phases as a stacked horizontal bar (state clock)
    t0 = row["t_trialstart"] - z_state
    for label, col, color in [("quiescence", "duration_quiescence", "#ACACAC"), ("blank", "duration_blank", "#343434")]:
        if row[col] is not None:
            _, ax = bv.plot_hbar(8, row[col], left=t0, height=2, spec=spec, ax=ax, color=color, alpha=0.3)
            _, ax = bv.plot_text(
                t0 + row[col] / 2, 8, label, spec=spec, ax=ax, ha="center", va="center", color="#000000"
            )
            t0 += row[col]

    return PlotResult(data=row, figure=(fig, ax))
