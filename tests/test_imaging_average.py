"""Trial averaging (step 2): splitting the trials into pieces must not change the result.

Data-free, runs in CI. Uses a fake frame source so no image files are needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from piepy.imaging.average import combine, partial_sum, trial_average
from piepy.imaging.executor import LocalExecutor, ProcessExecutor
from piepy.imaging.windows import FrameWindows


class FakeStack:
    """Stands in for a real frame source: ``stack[frame_indices]`` returns those frames.

    Frame ``f`` is filled with the value ``f`` everywhere, so a window's average is easy to check.
    """

    def __init__(self, n_frames=200, h=4, w=5, channels=None):
        base = np.arange(n_frames, dtype=np.uint16)[:, None, None] * np.ones(
            (1, h, w), dtype=np.uint16
        )
        self.frames = base if channels is None else base[:, None].repeat(channels, axis=1)

    def __getitem__(self, idx):
        return self.frames[idx]

    @property
    def shape(self):
        return self.frames.shape


def _windows(starts, count, *, pre=0, groups=None):
    starts = np.asarray(starts, dtype=np.int64)
    g = None
    if groups is not None:
        g = np.empty(len(groups), dtype=object)
        g[:] = groups
    return FrameWindows(
        starts=starts,
        count=count,
        pre=pre,
        trial_no=np.arange(1, len(starts) + 1),
        groups=g,
    )


def test_pieces_do_not_change_result_exact():
    stack = FakeStack()
    w = _windows([0, 10, 25, 40, 55, 70, 90], count=8)  # 7 trials
    one = trial_average(stack, w, n_pieces=1)
    for k in (2, 3, 7, 10):  # 10 > 7 trials: extra empty pieces are dropped
        many = trial_average(stack, w, n_pieces=k)
        assert np.array_equal(
            one[None], many[None]
        )  # identical, bit for bit (integer frames)


def test_overlapping_windows_still_exact():
    stack = FakeStack()
    w = _windows([0, 3, 6, 9], count=10)  # windows overlap each other
    assert np.array_equal(
        trial_average(stack, w, n_pieces=1)[None],
        trial_average(stack, w, n_pieces=3)[None],
    )


def test_average_value_is_correct():
    stack = FakeStack()
    starts, count = [0, 10, 20], 5
    w = _windows(starts, count)
    got = trial_average(stack, w)[None]
    want = np.mean([stack.frames[s : s + count] for s in starts], axis=0)
    assert np.allclose(got, want)
    # frame k holds the average of (start + k) over trials
    assert np.allclose(got[:, 0, 0], [(0 + 10 + 20) / 3 + k for k in range(count)])


def test_grouped_conditions():
    stack = FakeStack()
    w = _windows([0, 10, 20, 30], count=5, groups=["a", "a", "b", "b"])
    out = trial_average(stack, w, n_pieces=3)
    assert set(out) == {"a", "b"}
    assert np.allclose(
        out["a"], np.mean([stack.frames[0:5], stack.frames[10:15]], axis=0)
    )
    assert np.allclose(
        out["b"], np.mean([stack.frames[20:25], stack.frames[30:35]], axis=0)
    )
    # grouping is unaffected by how the trials are split
    ref = trial_average(stack, w, n_pieces=1)
    assert np.array_equal(out["a"], ref["a"]) and np.array_equal(out["b"], ref["b"])


def test_channel_axis_selected():
    flat = FakeStack()
    multi = FakeStack(channels=3)  # (frames, channels, H, W)
    w = _windows([0, 10, 20], count=6)
    assert np.array_equal(
        trial_average(flat, w)[None], trial_average(multi, w, channel=0)[None]
    )


def test_combine_order_does_not_matter():
    stack = FakeStack()
    w = _windows([0, 10, 20, 30, 40], count=6)
    p = [partial_sum(stack, w, g) for g in ([0, 1], [2], [3, 4])]
    forward = combine(p)[None]
    backward = combine(p[::-1])[None]
    assert np.array_equal(forward, backward)


def test_downsample_after_averaging():
    stack = FakeStack(h=8, w=8)
    w = _windows([0, 10, 20, 30], count=5)
    small = trial_average(stack, w, downsample=2)[None]
    assert small.shape == (5, 4, 4)
    # shrinking then averaging equals averaging then shrinking, for any number of pieces
    assert np.array_equal(small, trial_average(stack, w, n_pieces=3, downsample=2)[None])


def test_local_executor_runs_in_order():
    assert LocalExecutor().map(lambda x: x * 2, [1, 2, 3]) == [2, 4, 6]


def test_process_executor_matches_serial():
    # a plain array is a valid frame source (fancy indexing) and is picklable, so it can be sent
    # to worker processes; the parallel result must equal the single-process one, bit for bit.
    frames = np.arange(200, dtype=np.uint16)[:, None, None] * np.ones(
        (1, 4, 5), dtype=np.uint16
    )
    w = _windows([0, 10, 20, 30, 40, 50, 60], count=8)
    serial = trial_average(frames, w, n_pieces=1)[None]
    parallel = trial_average(frames, w, executor=ProcessExecutor(2), n_pieces=4)[None]
    assert np.array_equal(serial, parallel)


def test_float_frames_use_float_total():
    class FloatStack(FakeStack):
        def __init__(self):
            super().__init__()
            self.frames = self.frames.astype(np.float32)

    stack = FloatStack()
    w = _windows([0, 10, 20], count=5)
    assert np.allclose(
        trial_average(stack, w)[None], trial_average(stack, w, n_pieces=3)[None]
    )


@pytest.mark.parametrize("n_pieces", [1, 2, 4])
def test_partial_sum_counts(n_pieces):
    stack = FakeStack()
    w = _windows([0, 10, 20, 30], count=5)
    groups = np.array_split(np.arange(w.n), n_pieces)
    total = 0
    for g in groups:
        if g.size:
            total += partial_sum(stack, w, g)[None][
                1
            ]  # trial count for the ungrouped case
    assert total == w.n
