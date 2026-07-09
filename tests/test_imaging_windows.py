"""Pure trial->window builder + dF/F (step 1 of the imaging plan). Data-free, runs in CI."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from piepy.imaging.average import dff
from piepy.imaging.windows import frame_windows

FRAME_T = 10.0  # ms/frame -> pre_t/post_t/duration convert 1:1 over 10


def _make(rows):
    out = []
    for r in rows:
        d = {"trial_no": r["trial_no"], "onepcam_frame_ids": r["frame_ids"]}
        d.update({k: v for k, v in r.items() if k not in ("trial_no", "frame_ids")})
        out.append(d)
    return pl.DataFrame(out, schema_overrides={"onepcam_frame_ids": pl.List(pl.Int64)})


def test_basic_starts_count_pre():
    df = _make([{"trial_no": 1, "frame_ids": [100, 160]}, {"trial_no": 2, "frame_ids": [300, 350]}])
    w = frame_windows(df, pre_t=100, post_t=0, frame_t=FRAME_T)  # pre=10 frames
    # raw counts: (160-100)+10 = 70, (350-300)+10 = 60 -> min-duration = 60
    assert w.count == 60
    assert w.pre == 10
    assert np.array_equal(w.starts, [90, 290])  # start - pre
    assert np.array_equal(w.trial_no, [1, 2])
    assert w.groups is None


def test_post_extends_length():
    df = _make([{"trial_no": 1, "frame_ids": [100, 160]}])
    w = frame_windows(df, pre_t=0, post_t=50, frame_t=FRAME_T)  # post=5
    assert w.count == (160 - 100) + 0 + 5
    assert w.starts[0] == 100


def test_fixed_duration_and_drop_short():
    df = _make(
        [
            {"trial_no": 1, "frame_ids": [100, 160]},  # raw 60
            {"trial_no": 2, "frame_ids": [300, 355]},  # raw 55
            {"trial_no": 3, "frame_ids": [10, 20]},  # raw 10 -> dropped
        ]
    )
    w = frame_windows(df, pre_t=0, post_t=0, duration=550, frame_t=FRAME_T)  # 55 frames
    assert w.count == 55
    assert np.array_equal(w.trial_no, [1, 2])  # trial 3 too short, dropped
    assert np.array_equal(w.starts, [100, 300])


def test_group_scalar_and_multi():
    df = _make(
        [
            {"trial_no": 1, "frame_ids": [100, 160], "contrast": 50, "opto": 0},
            {"trial_no": 2, "frame_ids": [300, 360], "contrast": 25, "opto": 1},
        ]
    )
    w1 = frame_windows(df, pre_t=0, frame_t=FRAME_T, group="contrast")
    assert list(w1.groups) == [50, 25]
    w2 = frame_windows(df, pre_t=0, frame_t=FRAME_T, group=["contrast", "opto"])
    assert list(w2.groups) == [(50, 0), (25, 1)]


def test_drops_null_frame_ids():
    df = _make([{"trial_no": 1, "frame_ids": None}, {"trial_no": 2, "frame_ids": [300, 360]}])
    w = frame_windows(df, pre_t=0, frame_t=FRAME_T)
    assert np.array_equal(w.trial_no, [2])


def test_negative_pre_raises():
    df = _make([{"trial_no": 1, "frame_ids": [5, 60]}])
    with pytest.raises(ValueError, match="before the first recorded frame"):
        frame_windows(df, pre_t=100, frame_t=FRAME_T)  # pre=10 > start 5


def test_missing_column_and_empty():
    with pytest.raises(ValueError, match="not in the trial table"):
        frame_windows(pl.DataFrame({"trial_no": [1]}), frame_t=FRAME_T)
    df = _make([{"trial_no": 1, "frame_ids": None}])
    with pytest.raises(ValueError, match="No trials with"):
        frame_windows(df, frame_t=FRAME_T)


def test_bad_frame_t():
    df = _make([{"trial_no": 1, "frame_ids": [100, 160]}])
    with pytest.raises(ValueError, match="frame_t must be"):
        frame_windows(df, frame_t=0)


def test_dff_baseline_subtract_divide():
    mean = np.stack([np.full((2, 2), 100.0)] * 2 + [np.full((2, 2), 110.0)] * 2)  # (4,2,2)
    out = dff(mean, pre=2)  # F0 = 100
    assert out.dtype == np.float32
    assert np.allclose(out[:2], 0.0)
    assert np.allclose(out[2:], 0.1)  # (110-100)/100


def test_dff_pre_zero_uses_whole_block():
    mean = np.stack([np.full((1, 1), 4.0), np.full((1, 1), 8.0)])  # F0 = mean = 6
    out = dff(mean, pre=0)
    assert np.allclose(out.ravel(), [(4 - 6) / 6, (8 - 6) / 6])
