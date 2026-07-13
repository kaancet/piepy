"""Widefield wiring (step 4): trials + frames -> dF/F per condition, plus the small helpers.

Data-free. The file-reading wrappers (widefield_from_run / run_frame_period_ms) need real image
folders, so they are exercised on real data elsewhere; here we test the parts that don't read files.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from piepy.imaging.widefield import analyze_widefield, frame_period_ms, to_display_uint16

FRAME_T = 10.0  # ms/frame


def _frames(n=200, h=4, w=5):
    return np.arange(n, dtype=np.uint16)[:, None, None] * np.ones(
        (1, h, w), dtype=np.uint16
    )


def _trials(rows):
    return pl.DataFrame(rows, schema_overrides={"onepcam_frame_ids": pl.List(pl.Int64)})


def test_analyze_widefield_single_condition():
    frames = _frames()
    df = _trials(
        [
            {"trial_no": 1, "onepcam_frame_ids": [100, 160]},
            {"trial_no": 2, "onepcam_frame_ids": [30, 90]},
        ]
    )
    out = analyze_widefield(
        df, frames, frame_t=FRAME_T, pre_t=100, post_t=0
    )  # pre = 10 frames
    assert set(out) == {None}
    movie = out[None]
    assert movie.dtype == np.float32
    # dF/F is measured against the average of the 'pre' baseline frames, so the baseline window
    # averages to ~0 (individual baseline frames still vary around it).
    assert np.allclose(movie[:10].mean(axis=0), 0.0, atol=1e-6)


def test_analyze_widefield_per_condition_and_pieces_agree():
    frames = _frames()
    df = _trials(
        [
            {"trial_no": 1, "onepcam_frame_ids": [10, 60], "contrast": 50},
            {"trial_no": 2, "onepcam_frame_ids": [70, 120], "contrast": 50},
            {"trial_no": 3, "onepcam_frame_ids": [130, 180], "contrast": 25},
        ]
    )
    one = analyze_widefield(
        df, frames, frame_t=FRAME_T, conditions="contrast", pre_t=0, n_pieces=1
    )
    many = analyze_widefield(
        df, frames, frame_t=FRAME_T, conditions="contrast", pre_t=0, n_pieces=3
    )
    assert set(one) == {50, 25}
    for key in one:
        assert np.array_equal(
            one[key], many[key]
        )  # splitting into pieces changes nothing


def test_frame_period_from_gaps_seconds():
    # gaps of 0.01 s -> 10 ms
    assert frame_period_ms(
        [0.0, 0.01, 0.02, 0.03], timestamp_precision=1000, comments=[]
    ) == pytest.approx(10.0)


def test_frame_period_from_tick_gaps():
    # gaps of 100 (>= 1) are 10-microsecond ticks -> 100/10000 s = 0.01 s -> 10 ms
    assert frame_period_ms(
        [0.0, 10000.0, 20000.0], timestamp_precision=0.001, comments=[]
    ) == pytest.approx(10.0)


def test_frame_period_fallback_to_comments():
    # timestamps all zero -> use total time from comments / frame count: 10 s over 5 frames = 2000 ms
    comments = ["# [12:00:00] start of recording", "# [12:00:10] end of recording"]
    assert frame_period_ms(
        [0.0] * 5, timestamp_precision=1, comments=comments
    ) == pytest.approx(2000.0)


def test_to_display_uint16_spans_full_range():
    movie = np.array([[[0.0]], [[1.0]], [[2.0]]])  # 3 frames
    out = to_display_uint16(movie)
    assert out.dtype == np.uint16
    assert out.min() == 0 and out.max() == 65535


def test_to_display_uint16_flat_movie():
    assert np.all(to_display_uint16(np.full((2, 2, 2), 5.0)) == 0)
