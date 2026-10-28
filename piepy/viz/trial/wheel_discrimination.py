"""Trial timeline for a single wheel-discrimination trial: wheel trace + event markers."""

from __future__ import annotations

import numpy as np
import polars as pl

from ..base import PlotResult, _need, _resolve
from ...psychophysics.wheelTrace import WheelTrace


def trial_snapshot(
    data,
    trial_no: int,
    *,
    show_movements: bool = False,
    movement_params: dict | None = None,
    annotation_offset: float = 3,
    ax=None,
    **style,
) -> PlotResult:
    """Wheel speed vs time for one discrimination trial, aligned to stimulus onset, with markers.

    Two clocks, both zeroed at stimulus onset: state events use ``t_vstimstart``; the rig streams
    (wheel/lick/reward) use ``t_vstimstart_rig`` (falling back to the state onset). Unlike detection,
    every discrimination trial has a stimulus, so the phases drawn after onset are the open-loop
    (wait) and response windows -- shown only when those columns are present.
    """
    import behaviz as bv

    style_overrides = bv.split_styles(
        style,
        components=("wheel", "lick", "reward"),
        defaults={
            "wheel": {"linewidth": 2, "color": "#090909"},
            "lick": {
                "marker": "|",
                "s": 250,
                "color": "#1EB9E4",
            },
            "reward": {
                "color": "#007DB7",
                "s": 100,
            },
        },
    )

    spec = bv.load_preset("trial_snapshot")

    df = _resolve(data)
    _need(
        df,
        [
            "trial_no",
            "outcome",
            "t_trialstart",
            "t_vstimstart",
            "t_vstimend",
            "t_trialend",
            "wheel_t",
            "wheel_pos",
        ],
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
    z_state = (
        row["t_vstimstart"] if row["t_vstimstart"] is not None else row["t_trialstart"]
    )
    z_rig = (
        row["t_vstimstart_rig"] if row.get("t_vstimstart_rig") is not None else z_state
    )

    # wheel trace (raw rig time -> shift onto the stim-onset frame)
    wheel_t = np.array(row["wheel_t"], float)
    wheel_pos = np.array(row["wheel_pos"], float)

    if movement_params is None:
        movement_params = {
            "pos_thresh": 0.025,
            "t_thresh": 20,
            "min_dur": 40,
            "min_gap": 40,
            "pos_thresh_onset": 0.005,
        }
    res = WheelTrace(t=wheel_t, pos=wheel_pos).process(
        reset_time=z_rig, freq=5, units="rad", **movement_params
    )
    pos = res["pos"]

    fig, ax = bv.plot_line(res["t"], pos, spec=spec, ax=ax, **style_overrides["wheel"])

    if show_movements:
        for on_idx, off_idx in zip(
            res["movements"]["onsets"][:, 0].astype(int),
            res["movements"]["offsets"][:, 0].astype(int),
        ):
            fig, ax = bv.plot_line(
                res["t"][on_idx:off_idx],
                pos[on_idx:off_idx],
                spec=spec,
                ax=ax,
                linewidth=2,
                color="#223399",
            )
            # start
            fig, ax = bv.plot_scatter(
                [res["t"][on_idx]],
                [pos[on_idx]],
                color="#1A902F",
                marker=">",
                spec=spec,
                ax=ax,
            )

            # end
            fig, ax = bv.plot_scatter(
                [res["t"][off_idx]],
                [pos[off_idx]],
                color="#C70C02",
                marker="<",
                spec=spec,
                ax=ax,
            )

    # state-clock event markers, drawn only when present
    for label, col in [
        ("trial start", "t_trialstart"),
        ("stim on", "t_vstimstart"),
        ("stim off", "t_vstimend"),
        ("trial end", "t_trialend"),
    ]:
        if row.get(col) is not None:
            _, ax = bv.plot_vertical(
                [row[col] - z_state], spec=spec, ax=ax, color="#4C4C4C"
            )
            fig, ax = bv.plot_text(
                row[col] - z_state + 15,
                annotation_offset,
                f"{label}",
                label,
                ax=ax,
                spec=spec,
                rotation=90,
                ha="left",
                va="center",
                color="#4C4C4C",
                backgroundcolor="#FFFFFF",
                alpha=0.3,
                zorder=1,
            )

    # reaction / response are already measured from stimulus onset
    for label, col in [("reaction", "reaction_time"), ("response", "response_time")]:
        if row.get(col) is not None:
            _, ax = bv.plot_vertical([row[col]], spec=spec, ax=ax, color="#990000")
            fig, ax = bv.plot_text(
                row[col] - 5,
                annotation_offset,
                f"{label}:{round(row[col], 1)}ms",
                ax=ax,
                spec=spec,
                rotation=90,
                ha="right",
                va="bottom",
                color="#990000",
                alpha=0.5,
            )

    # licks (all times) and reward (only its first element is a time), rig clock
    if row.get("lick"):
        licks = np.array(row["lick"], float) - z_rig
        _, ax = bv.plot_scatter(
            licks,
            np.ones_like(licks) * annotation_offset,
            spec=spec,
            ax=ax,
            label="lick",
            **style_overrides["lick"],
        )
    if row.get("reward"):
        _, ax_pos = bv.plot_scatter(
            [row["reward"][0] - z_rig],
            [annotation_offset],
            spec=spec,
            ax=ax,
            label="reward",
            **style_overrides["reward"],
        )

    # post-stim phases as a stacked horizontal bar from stim onset (state clock); drawn when the
    # window columns are present (added by the session enrich step).
    t0 = 0.0
    for label, col, color in [
        ("open loop", "wait_window", "#ACACAC"),
        ("response", "response_window", "#343434"),
    ]:
        w = row.get(col)
        if w is not None:
            _, ax = bv.plot_hbar(
                annotation_offset,
                w,
                left=t0,
                height=3,
                spec=spec,
                ax=ax,
                color=color,
                linewidth=0,
                alpha=0.3,
            )
            _, ax = bv.plot_text(
                t0 + w / 2,
                annotation_offset,
                f"{label}\n{int(row[col])}ms",
                spec=spec,
                ax=ax,
                ha="center",
                va="center",
                color="#000000",
                alpha=0.3,
            )
            t0 += w

    return PlotResult(data=row, figure=(fig, ax))
