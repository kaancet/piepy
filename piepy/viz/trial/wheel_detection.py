"""Trial timeline for a single wheel-detection trial: wheel trace + event markers."""

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
    **style,
) -> PlotResult:
    """Wheel position vs time and wheel speed vs time for one trial, aligned to stimulus onset, with event markers.

    Two clocks, both zeroed at stimulus onset: state events use ``t_vstimstart``; the rig streams
    (wheel/lick/reward) use ``t_vstimstart_rig``. Early trials fall back to cue + blank.
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
                "color": "#00BFFF",
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
            "t_trialinit",
            "t_vstimstart",
            "t_vstimend",
            "t_trialend",
            "t_vstimstart_rig",
            "duration_quiescence",
            "duration_blank",
            "reaction_time",
            "response_time",
            "lick",
            "reward",
            "wheel_t",
            "wheel_pos",
        ],
        plot="trial snapshot",
    )  # a missing column errors cleanly here rather than as a KeyError mid-draw
    sel = df.filter(pl.col("trial_no") == trial_no)
    if sel.is_empty():
        raise ValueError(f"trial_no {trial_no} not found in the data.")
    row = sel.row(0, named=True)
    title = f"{row['trial_no']}-{row['outcome']}"

    # always look for rig times first, if not use state times
    z_state = (
        row["t_vstimstart"]
        if row["t_vstimstart"] is not None
        else (row["t_trialinit"] or 0) + (row["duration_blank"] or 0)
    )
    z_rig = row["t_vstimstart_rig"] if row["t_vstimstart_rig"] is not None else z_state

    # wheel trace (raw rig time -> shift onto the stim-onset frame)
    wheel_t = np.array(row["wheel_t"], float)
    wheel_pos = np.array(row["wheel_pos"], float)

    trace = WheelTrace(t=wheel_t, pos=wheel_pos)

    if movement_params is None:
        movement_params = {
            "pos_thresh": 0.025,
            "t_thresh": 20,
            "min_dur": 40,
            "min_gap": 40,
            "pos_thresh_onset": 0.005,
        }
    res = trace.process(
        reset_time=z_rig,
        freq=5,
        units="rad",
        **movement_params,
    )
    pos = res["pos"]
    speed = np.abs(res["velocity"] * 1000)

    with bv.grid(
        nrows=2, ncols=1, spec=spec, height_ratios=[1, 1], sharex=True, suptitle=title
    ) as axs:
        fig, ax_pos = bv.plot_line(
            res["t"], pos, ax=axs[0, 0], **style_overrides["wheel"]
        )

        fig, ax_speed = bv.plot_line(
            res["t"], speed, ax=axs[1, 0], **style_overrides["wheel"]
        )

        if show_movements:
            for on_idx, off_idx in zip(
                res["movements"]["onsets"][:, 0].astype(int),
                res["movements"]["offsets"][:, 0].astype(int),
            ):
                fig, ax_pos = bv.plot_line(
                    res["t"][on_idx:off_idx],
                    pos[on_idx:off_idx],
                    spec=spec,
                    ax=ax_pos,
                    linewidth=2,
                    color="#223399",
                )
                # start
                fig, ax_pos = bv.plot_scatter(
                    [res["t"][on_idx]],
                    [pos[on_idx]],
                    color="#1A902F",
                    marker=">",
                    spec=spec,
                    ax=ax_pos,
                )

                # end
                fig, ax_pos = bv.plot_scatter(
                    [res["t"][off_idx]],
                    [pos[off_idx]],
                    color="#C70C02",
                    marker="<",
                    spec=spec,
                    ax=ax_pos,
                )

                fig, ax_speed = bv.plot_line(
                    res["t"][on_idx:off_idx],
                    speed[on_idx:off_idx],
                    spec=spec,
                    ax=ax_speed,
                    linewidth=2,
                    color="#223399",
                )

                # start
                fig, ax_speed = bv.plot_scatter(
                    [res["t"][on_idx]],
                    [speed[on_idx]],
                    color="#1A902F",
                    marker=">",
                    spec=spec,
                    ax=ax_speed,
                )

                # end
                fig, ax_speed = bv.plot_scatter(
                    [res["t"][off_idx]],
                    [speed[off_idx]],
                    color="#C70C02",
                    marker="<",
                    spec=spec,
                    ax=ax_speed,
                )

    # state-clock event markers, drawn only when present
    for label, col in [
        ("trial start", "t_trialstart"),
        ("cue", "t_trialinit"),
        ("stim on", "t_vstimstart"),
        ("stim off", "t_vstimend"),
        ("trial end", "t_trialend"),
    ]:
        if row[col] is not None:
            _, ax_pos = bv.plot_vertical(
                [row[col] - z_state], spec=spec, ax=ax_pos, color="#4C4C4C"
            )

            fig, ax_pos = bv.plot_text(
                row[col] - z_state + 15,
                annotation_offset,
                f"{label}",
                ax=ax_pos,
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
    for label, col in [("RT", "reaction_time"), ("resp", "state_response_time")]:
        if row[col] is not None:
            _, ax_pos = bv.plot_vertical(
                [row[col]], spec=spec, ax=ax_pos, color="#990000"
            )
            _, ax_speed = bv.plot_vertical(
                [row[col]], spec=spec, ax=ax_speed, color="#990000"
            )

            fig, ax_speed = bv.plot_text(
                row[col] - 5,
                annotation_offset,
                f"{label}:{round(row[col], 1)}ms",
                ax=ax_speed,
                spec=spec,
                rotation=90,
                ha="right",
                va="bottom",
                color="#990000",
                alpha=0.5,
            )

    # licks (all times) and reward (only its first element is a time), rig clock
    if row["lick"]:
        licks = np.array(row["lick"], float) - z_rig
        _, ax_pos = bv.plot_scatter(
            licks,
            np.ones_like(licks) * annotation_offset,
            spec=spec,
            ax=ax_pos,
            label="lick",
            **style_overrides["lick"],
        )
    if row["reward"]:
        _, ax_pos = bv.plot_scatter(
            [row["reward"][0] - z_rig],
            [annotation_offset],
            spec=spec,
            ax=ax_pos,
            label="reward",
            **style_overrides["reward"],
        )

    # pre-stim phases as a stacked horizontal bar (state clock)
    t0 = row["t_trialstart"] - z_state
    for label, col, color in [
        ("quiescence", "duration_quiescence", "#ACACAC"),
        ("blank", "duration_blank", "#343434"),
    ]:
        if row[col] is not None:
            _, ax_pos = bv.plot_hbar(
                annotation_offset,
                row[col],
                left=t0,
                height=3,
                spec=spec,
                ax=ax_pos,
                color=color,
                linewidth=0,
                alpha=0.3,
            )
            _, ax_pos = bv.plot_text(
                t0 + row[col] / 2,
                annotation_offset,
                f"{label}\n{int(row[col])}ms",
                spec=spec,
                ax=ax_pos,
                ha="center",
                va="center",
                color="#000000",
                alpha=0.3,
            )
            t0 += row[col]

    ax_pos.set_ylabel("Wheel position (rad)", fontsize=spec.x.fontsize)
    ax_pos.tick_params(axis="x", labelbottom=False)
    ax_speed.set_ylabel("Wheel speed (rad/s)", fontsize=spec.x.fontsize)
    ax_pos.grid(False)
    return PlotResult(data=row, figure=(fig, ax_pos, ax_speed))
