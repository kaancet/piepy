"""Average the imaging frames of many trials, in a way that is safe to split across workers.

The average of N trials is: add up all their frames, then divide by N. Both steps here are built
so that splitting the work into pieces gives *exactly* the same answer as doing it all at once:

* Each trial is added up on its own. Adding the same trials in a different order, or in separate
  groups, gives the same total. So it does not matter how the trials are divided between workers.
* Camera frames are whole numbers. Adding whole numbers is exact (no rounding), so the running
  total is identical no matter how many pieces the trials are split into. The division that turns
  the total into an average happens once, at the very end.
* dF/F (:func:`dff`) also happens once, on the finished average. It must never run on part of the
  trials, because it divides by a baseline that only makes sense for the whole average.

This is why parallel and single-worker runs return the same result, bit for bit, for integer
camera data.
"""

from __future__ import annotations

from functools import partial

import numpy as np

from .executor import LocalExecutor
from .windows import FrameWindows


def partial_sum(stack, windows: FrameWindows, idx: np.ndarray, *, channel: int = 0) -> dict:
    """Add up the frames of the trials listed in ``idx``, kept separate per condition.

    Reads only the trials in ``idx`` (a subset of all trials), so several of these can run on
    different workers over different subsets. For each condition it keeps a running total of the
    frames and a count of how many trials went into it.

    Args:
        stack: the frame source; ``stack[frame_indices]`` returns those frames as an array shaped
            ``(n_frames, H, W)`` or ``(n_frames, channels, H, W)``.
        windows: the per-trial frame windows from :func:`piepy.imaging.windows.frame_windows`.
        idx: indices (into ``windows``) of the trials to add up in this call.
        channel: which channel to read when the frames have a channel axis.

    Returns:
        ``{condition: [running_total, trial_count]}``. ``running_total`` is an integer array for
        integer frames (kept exact) or a float array otherwise, shaped ``(window, H, W)``.
        ``condition`` is ``None`` when the windows have no grouping.
    """
    totals: dict = {}
    counts: dict = {}
    for i in idx:
        i = int(i)
        frames = np.arange(windows.starts[i], windows.starts[i] + windows.count)
        block = np.asarray(stack[frames])
        if block.ndim == 4:  # (frames, channels, H, W) -> keep one channel
            block = block[:, channel]

        key = None if windows.groups is None else windows.groups[i]
        if key not in totals:
            # integer frames -> integer running total, which stays exact when added up.
            exact = np.issubdtype(block.dtype, np.integer)
            totals[key] = np.zeros(block.shape, dtype=np.int64 if exact else np.float64)
            counts[key] = 0
        totals[key] += block
        counts[key] += 1

    return {key: [totals[key], counts[key]] for key in totals}


def combine(partials: list[dict]) -> dict:
    """Add the running totals from every group of trials together and turn them into averages.

    Args:
        partials: the ``{condition: [running_total, trial_count]}`` results from :func:`partial_sum`
            (one per group of trials).

    Returns:
        ``{condition: average}`` where ``average`` is the total divided by the number of trials for
        that condition, as a float array shaped ``(window, H, W)``.
    """
    totals: dict = {}
    counts: dict = {}
    for part in partials:
        for key, (total, count) in part.items():
            if key not in totals:
                totals[key] = np.array(total)  # copy so the inputs are left untouched
                counts[key] = count
            else:
                totals[key] += total
                counts[key] += count

    return {key: totals[key] / counts[key] for key in totals}


def trial_average(
    stack,
    windows: FrameWindows,
    *,
    executor=None,
    n_pieces: int = 1,
    downsample: int = 1,
    channel: int = 0,
) -> dict:
    """Average the trial frames, one averaged movie per condition.

    Splits the trials into ``n_pieces`` groups, adds each group up (optionally on separate workers),
    then combines the totals into one average per condition. The number of pieces only changes how
    the work is divided, not the result.

    Args:
        stack: the frame source (see :func:`partial_sum`).
        windows: the per-trial frame windows.
        executor: where the groups run; defaults to :class:`~piepy.imaging.executor.LocalExecutor`
            (all in this process).
        n_pieces: how many groups to split the trials into.
        downsample: shrink the averaged movie by this factor in height and width. Done once at the
            end, after averaging (shrinking then averaging and averaging then shrinking give the
            same result, so this is safe to defer).
        channel: which channel to read when frames have a channel axis.

    Returns:
        ``{condition: averaged_movie}``, each shaped ``(window, H//downsample, W//downsample)``.
    """
    executor = executor or LocalExecutor()

    # split the trials into contiguous groups (a fixed trial order keeps the result reproducible);
    # drop empty groups when there are more pieces than trials.
    groups = [g for g in np.array_split(np.arange(windows.n), n_pieces) if g.size]

    partials = executor.map(partial(partial_sum, stack, windows, channel=channel), groups)
    means = combine(partials)

    if downsample > 1:
        from .onep.retinoutils import downsample_movie  # heavy import, only when needed

        means = {key: downsample_movie(mean, block_size=downsample) for key, mean in means.items()}
    return means


def dff(mean: np.ndarray, pre: int, *, eps: float = 0.0) -> np.ndarray:
    """Turn an averaged movie into dF/F0: how much each frame changes from the pre-stimulus level.

    ``F0`` (the baseline) is the average of the first ``pre`` frames, which are the frames before
    the stimulus. Every frame becomes ``(frame - F0) / F0``. Run this once, on the finished average
    from :func:`trial_average` -- never on part of the trials, because the baseline only makes sense
    for the whole average.

    Args:
        mean: the averaged movie, shaped ``(window, H, W)``.
        pre: number of baseline frames at the start of the movie (0 uses the whole movie as the
            baseline).
        eps: small number added to the baseline to avoid dividing by zero. Leave at 0 unless the
            baseline can be near zero.

    Returns:
        float32 dF/F0 movie, same shape as ``mean``.
    """
    base = mean[:pre] if pre else mean
    f0 = base.mean(axis=0)
    return ((mean - f0) / (f0 + eps)).astype(np.float32)
