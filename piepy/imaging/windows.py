"""Trial -> uniform frame window: the pure input to trial averaging (no IO).

``frame_windows`` turns a trial table into one fixed-length frame window per trial, read from
the ``<mode>_frame_ids`` column that ``Trial.set_frame_endpoints`` writes ([first, last] camera
frame index within the trial's epoch). It replaces ``OnePAnalysis.make_frame_matrix`` +
``CamDataAnalysis.set_minimum_dur`` -- same logic, but a pure function returning a small dataclass
so the averaging reducer can be tested in isolation.

Windows are **front-aligned**: frames for trial i are ``range(starts[i], starts[i] + count)``, with
the first ``pre`` frames being the pre-stim baseline. All trials share one ``count`` (the minimum,
or a fixed ``duration``), so their blocks stack into a ``(count, H, W)`` average.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class FrameWindows:
    """One fixed-length frame window per trial (front-aligned; baseline = the first ``pre`` frames).

    Attributes:
        starts: int[N], first frame index of each trial's block (already shifted back by ``pre``).
        count: frames per trial (uniform across all trials).
        pre: number of pre-stim baseline frames at the start of each block.
        trial_no: int[N], the trial numbers kept (for canonical ordering / provenance).
        groups: object[N] condition key per trial (scalar for a single group column, tuple for
            several), or ``None`` when no grouping was requested.
    """

    starts: np.ndarray
    count: int
    pre: int
    trial_no: np.ndarray
    groups: np.ndarray | None = None

    @property
    def n(self) -> int:
        return int(self.starts.size)


def frame_windows(
    trials_df: pl.DataFrame,
    *,
    mode: str = "onepcam",
    pre_t: float = 0.0,
    post_t: float = 0.0,
    duration: float | None = None,
    frame_t: float,
    group: str | list[str] | None = None,
) -> FrameWindows:
    """Build one uniform frame window per trial from ``<mode>_frame_ids``.

    Args:
        trials_df: the (per-run) trial table; must have ``trial_no`` and ``<mode>_frame_ids``.
        mode: imaging channel, e.g. ``"onepcam"`` -> reads ``onepcam_frame_ids``.
        pre_t / post_t: baseline / tail to add before the first and after the last epoch frame (ms).
        duration: fixed window length in ms; ``None`` -> the shortest trial's length (min-duration).
        frame_t: mean frame period in ms (``pre``/``post``/``duration`` are converted with this).
        group: column(s) to attach as a per-trial condition key (one averaged movie per group).

    Returns:
        FrameWindows. Trials shorter than ``count`` are dropped; the rest are front-aligned to
        ``count`` frames (the tail is trimmed), matching the old ``set_minimum_dur``.

    Raises:
        ValueError: bad params, a missing column, no trials with frames, an empty result, or a
            ``pre`` window that reaches before the first recorded frame.
    """
    if frame_t <= 0:
        raise ValueError(f"frame_t must be > 0 ms, got {frame_t}.")
    if pre_t < 0 or post_t < 0:
        raise ValueError(f"pre_t/post_t must be >= 0 ms, got pre_t={pre_t}, post_t={post_t}.")

    col = f"{mode}_frame_ids"
    for c in (col, "trial_no", *([group] if isinstance(group, str) else (group or []))):
        if c not in trials_df.columns:
            raise ValueError(f"frame_windows: column {c!r} not in the trial table.")

    # keep trials that actually have a [first, last] frame pair
    sub = trials_df.filter(pl.col(col).is_not_null() & (pl.col(col).list.len() == 2))
    if sub.is_empty():
        raise ValueError(f"No trials with {col} frames to average.")

    start = sub[col].list.get(0).to_numpy().astype(np.int64)
    end = sub[col].list.get(1).to_numpy().astype(np.int64)
    trial_no = sub["trial_no"].to_numpy().astype(np.int64)

    pre = int(round(pre_t / frame_t))
    post = int(round(post_t / frame_t))
    starts = start - pre  # baseline frames prepended
    raw_count = (end - start) + pre + post  # block length (end-exclusive, matches the old matrix)

    if duration is None:
        count = int(raw_count.min())  # min-duration: shortest trial sets the length
    else:
        count = int(round(duration / frame_t))
        if count <= 0:
            raise ValueError(f"duration must give a positive frame count, got {count}.")

    keep = raw_count >= count  # drop trials shorter than the window (front-align, trim the tail)
    starts, trial_no = starts[keep], trial_no[keep]
    if starts.size == 0:
        raise ValueError(f"No trial reaches the requested window of {count} frames.")
    if starts.min() < 0:
        raise ValueError(
            f"pre window ({pre} frames) reaches before the first recorded frame (min start {starts.min()})."
        )

    groups = None
    if group is not None:
        gcols = [group] if isinstance(group, str) else list(group)
        rows = sub.select(gcols).rows()
        keys = [r[0] if len(gcols) == 1 else r for r in rows]
        arr = np.empty(len(keys), dtype=object)  # element-wise so tuple keys stay 1-D, not (N, k)
        arr[:] = keys
        groups = arr[keep]

    return FrameWindows(starts=starts, count=count, pre=pre, trial_no=trial_no, groups=groups)
