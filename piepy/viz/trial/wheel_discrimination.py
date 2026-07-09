"""Trial timeline for a single wheel-discrimination trial: wheel trace + event markers."""

from __future__ import annotations

import numpy as np
import polars as pl

from ..base import PlotResult, _need, _resolve
from ...psychophysics.wheelTrace import WheelTrace


def trial_snapshot(data, trial_no: int, *, ax=None, **style) -> PlotResult:
    """Wheel speed vs time for one discrimination trial, aligned to stimulus onset, with markers.

    Two clocks, both zeroed at stimulus onset: state events use ``t_vstimstart``; the rig streams
    (wheel/lick/reward) use ``t_vstimstart_rig`` (falling back to the state onset). Unlike detection,
    every discrimination trial has a stimulus, so the phases drawn after onset are the open-loop
    (wait) and response windows -- shown only when those columns are present.
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

    df = _resolve(data)
    _need(
        df,
        ["trial_no", "outcome", "t_trialstart", "t_vstimstart", "t_vstimend", "t_trialend", "wheel_t", "wheel_pos"],
        plot="trial snapshot",
    )  # a missing column errors cleanly here rather than as a KeyError mid-draw
    sel = df.filter(pl.col("trial_no") == trial_no)
    if sel.is_empty():
        raise ValueError(f"trial_no {trial_no} not found in the data.")
    row = sel.row(0, named=True)

    side = row.get("target_side")
    title = f"{row['trial_no']}-{row['outcome']}" + (f" ({side})" if side else "")
    spec = spec.with_title(title)

    # both clocks zeroed at stim onset (state vs rig); fall back to trial start if onset is missing
    z_state = row["t_vstimstart"] if row["t_vstimstart"] is not None else row["t_trialstart"]
    z_rig = row["t_vstimstart_rig"] if row.get("t_vstimstart_rig") is not None else z_state

    # wheel trace (raw rig time -> shift onto the stim-onset frame)
    wheel_t = np.array(row["wheel_t"], float)
    wheel_pos = np.array(row["wheel_pos"], float)
    res = WheelTrace(t=wheel_t, pos=wheel_pos).process(reset_time=z_rig, freq=5, units="rad")

    fig, ax = bv.plot_line(res["t"], np.abs(res["velocity"] * 1000), spec=spec, ax=ax, **style_overrides["wheel"])

    # state-clock event markers, drawn only when present
    for label, col in [
        ("trial start", "t_trialstart"),
        ("stim on", "t_vstimstart"),
        ("stim off", "t_vstimend"),
        ("trial end", "t_trialend"),
    ]:
        if row.get(col) is not None:
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
        if row.get(col) is not None:
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
    if row.get("lick"):
        licks = np.array(row["lick"], float) - z_rig
        _, ax = bv.plot_scatter(licks, np.zeros_like(licks), spec=spec, ax=ax, label="lick", **style_overrides["lick"])
    if row.get("reward"):
        _, ax = bv.plot_scatter([row["reward"][0] - z_rig], [0], spec=spec, ax=ax, label="reward")

    # post-stim phases as a stacked horizontal bar from stim onset (state clock); drawn when the
    # window columns are present (added by the session enrich step).
    t0 = 0.0
    for label, col, color in [("open loop", "wait_window", "#ACACAC"), ("response", "response_window", "#343434")]:
        w = row.get(col)
        if w is not None:
            _, ax = bv.plot_hbar(8, w, left=t0, height=2, spec=spec, ax=ax, color=color, alpha=0.3)
            _, ax = bv.plot_text(t0 + w / 2, 8, label, spec=spec, ax=ax, ha="center", va="center", color="#000000")
            t0 += w

    return PlotResult(data=row, figure=(fig, ax))
