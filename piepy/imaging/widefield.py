"""Run widefield trial averaging on a run: trials + frames in, one averaged movie per condition out.

This ties the pieces together:

    trial table  ->  frame windows (windows.py)  ->  average (average.py)  ->  dF/F

:func:`analyze_widefield` takes an already-loaded frame source and the frame period, so it can be
tested on its own. :func:`widefield_from_run` is the thin wrapper that opens the frames and reads
the frame period from a parsed run.

Note: the frame numbers in the trial table (``<mode>_frame_ids``) count frames within one run, and
each run has its own image files, so average one run at a time. To combine runs, add their running
totals with :func:`piepy.imaging.average.combine`.
"""

from __future__ import annotations

import os
from datetime import datetime
from os.path import join as pjoin

import numpy as np
import polars as pl

from .average import dff, trial_average
from .windows import frame_windows


def analyze_widefield(
    trials_df: pl.DataFrame,
    stack,
    *,
    frame_t: float,
    conditions: str | list[str] | None = None,
    mode: str = "onepcam",
    pre_t: float = 100.0,
    post_t: float = 0.0,
    duration: float | None = None,
    downsample: int = 1,
    executor=None,
    n_pieces: int = 1,
    eps: float = 0.0,
) -> dict:
    """Average the imaging frames of a run's trials and return dF/F, one movie per condition.

    Args:
        trials_df: the run's trial table (needs ``trial_no`` and ``<mode>_frame_ids``).
        stack: the frame source (``stack[frame_indices]`` returns those frames).
        frame_t: mean frame period in ms (from :func:`run_frame_period_ms`).
        conditions: column(s) to average separately (e.g. ``"contrast"``); ``None`` averages all
            trials together.
        mode / pre_t / post_t / duration: passed to :func:`piepy.imaging.windows.frame_windows`.
        downsample / executor / n_pieces: passed to :func:`piepy.imaging.average.trial_average`.
        eps: divide-by-zero guard for dF/F.

    Returns:
        ``{condition: dff_movie}`` (``{None: movie}`` when ``conditions`` is ``None``).
    """
    windows = frame_windows(
        trials_df,
        mode=mode,
        group=conditions,
        pre_t=pre_t,
        post_t=post_t,
        duration=duration,
        frame_t=frame_t,
    )
    means = trial_average(
        stack, windows, executor=executor, n_pieces=n_pieces, downsample=downsample
    )
    return {key: dff(mean, windows.pre, eps=eps) for key, mean in means.items()}


def widefield_from_run(
    run, *, mode: str = "onepcam", timestamp_precision: float = 1, **kwargs
) -> dict:
    """Open a run's frames and frame period, then average -- see :func:`analyze_widefield`.

    Reads the trial table from ``run.data.data`` and the image folder from ``run.paths.<mode>``.
    """
    from .onep.stacks import load_stack

    folder = getattr(run.paths, mode)
    if folder is None:
        raise ValueError(f"This run has no {mode} image folder.")
    stack = load_stack(folder, nchannels=1)
    frame_t = run_frame_period_ms(folder, timestamp_precision=timestamp_precision)
    return analyze_widefield(run.data.data, stack, frame_t=frame_t, mode=mode, **kwargs)


def run_frame_period_ms(folder: str, timestamp_precision: float) -> float:
    """Read the camera log in ``folder`` and return the mean time between frames, in ms.
    timestamp_precision controls the the timing precision of camlogs: 1 ms, 1000 s, 0.001 us and so on
    """
    from ..core.parsers import parse_labcams_log

    logs = [f for f in os.listdir(folder) if f.endswith("log")]
    if len(logs) != 1:
        raise IOError(f"Expected exactly one camera log in {folder}, found {len(logs)}.")
    camlog, comments, _ = parse_labcams_log(pjoin(folder, logs[0]))
    return frame_period_ms(camlog["timestamp"].to_numpy(), timestamp_precision, comments)


def frame_period_ms(timestamps, timestamp_precision, comments) -> float:
    """Mean time between camera frames, in milliseconds.

    Uses the gaps between frame timestamps. If those are all zero (some rigs don't log real times),
    falls back to the total recording time from the log's comment lines divided by the frame count.

    Args:
        timestamps: one timestamp per frame.
        timestamp_precision: the timing precision of timestamps in the camlog, to convert to ms here:
        comments: the camera log's comment lines (used only for the fallback).
    """
    ts = np.asarray(timestamps, dtype=float)
    
    # patch for now:
    drops = np.diff(ts) < 0
    # Count cumulative wraps and align with the original array size
    cumulative_wraps = np.insert(drops, 0, False).cumsum()

    # Apply the progressive offset (1,000,000 per accumulated wrap)
    ts += cumulative_wraps * 1000000
        
    ts = ts * timestamp_precision
    avg = float(np.nanmean(np.diff(ts))) if ts.size > 1 else 0.0

    if avg == 0.0:
        # no real per-frame times: use total time (last comment - first comment) / number of frames
        marks = [c for c in comments if "# [" in c]
        start = datetime.strptime(marks[0].split("]")[0][-8:], "%H:%M:%S")
        end = datetime.strptime(marks[-1].split("]")[0][-8:], "%H:%M:%S")
        per_ms = ((end - start).seconds / len(timestamps)) * 1000
    else:
        per_ms = avg
    return per_ms


def to_display_uint16(movie: np.ndarray) -> np.ndarray:
    """Rescale a movie to the full 16-bit range for viewing only.

    This changes the values (a per-movie stretch to fill 0..65535), so use it only to save a movie
    for looking at -- never for the dF/F you analyse.
    """
    lo = float(np.nanmin(movie))
    hi = float(np.nanmax(movie))
    if hi == lo:
        return np.zeros(movie.shape, dtype=np.uint16)
    return ((movie - lo) / (hi - lo) * 65535).astype(np.uint16)

def save_averages(results: dict, save_dir: str) -> list[str]:
    """Save each condition's averaged movie as a float32 tiff. Returns the paths written."""
    import tifffile as tf

    os.makedirs(save_dir, exist_ok=True)
    paths = []
    for key, movie in results.items():
        name = "avg.tif" if key is None else f"avg_{key}.tif"
        path = pjoin(save_dir, name)
        tf.imwrite(path, np.asarray(movie, dtype=np.float32))
        paths.append(path)
    return paths
