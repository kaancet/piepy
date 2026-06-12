"""Unit tests for the canonical data contract (piepy.core.schema).

Data-free and fast: these run in CI and pin down the identity scheme, the schema-aligned
concat, and the session-clock math with synthetic frames.
"""

from __future__ import annotations

import datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from piepy.core.schema import (
    align_and_concat,
    attach_run_identity,
    concat_session_runs,
    run_uid,
    session_uid,
    stable_uid,
)


# --------------------------------------------------------------------------- #
# identity
# --------------------------------------------------------------------------- #
def test_stable_uid_is_deterministic_and_distinct():
    assert stable_uid("a", "b") == stable_uid("a", "b")  # reproducible
    assert stable_uid("a", "b") != stable_uid("b", "a")  # order matters
    assert stable_uid("a", 1) != stable_uid("a", 2)


def test_session_uid_no_same_day_collision():
    # the bug in the old hash(date, animal) scheme: these two collided.
    a = session_uid("220427_KC141_ISI__1P_KC")
    b = session_uid("220427_KC141_ISI_ket__1P_KC")
    assert a != b


def test_run_uid_varies_by_run():
    s = "240731_KC150_detect__no_cam_KC"
    assert run_uid(s, 1, "run00_x") != run_uid(s, 2, "run01_x")
    assert run_uid(s, 1, "run00_x") == run_uid(s, 1, "run00_x")


def test_attach_run_identity_front_and_no_duplicate():
    df = pl.DataFrame({"trial_no": [1, 2], "animalid": ["KC150", "KC150"]})
    out = attach_run_identity(
        df,
        sessiondir="240731_KC150_detect__no_cam_KC",
        run_no=1,
        run_name="run00_x",
        paradigm="wheel_detection",
        animalid="SHOULD_NOT_OVERWRITE",
        baredate="240731",
    )
    # identity columns are at the front, in contract order
    assert out.columns[:4] == ["session_uid", "run_uid", "run_no", "paradigm"]
    # existing animalid is kept, not overwritten
    assert out["animalid"].to_list() == ["KC150", "KC150"]
    # date derived from baredate
    assert out["date"].to_list() == [datetime.date(2024, 7, 31)] * 2
    assert out["paradigm"].unique().to_list() == ["wheel_detection"]


# --------------------------------------------------------------------------- #
# align_and_concat
# --------------------------------------------------------------------------- #
def test_align_and_concat_drops_none_and_empty():
    a = pl.DataFrame({"x": [1]})
    assert align_and_concat([None, pl.DataFrame(), a]).equals(a)
    assert align_and_concat([None, pl.DataFrame()]).is_empty()


def test_align_and_concat_unions_columns_with_nulls_and_order():
    a = pl.DataFrame({"x": [1], "y": [10]})
    b = pl.DataFrame({"x": [2], "z": ["q"]})
    out = align_and_concat([a, b])
    assert out.columns == ["x", "y", "z"]  # first-seen union order
    assert out["y"].to_list() == [10, None]
    assert out["z"].to_list() == [None, "q"]


def test_align_and_concat_coerces_dtype_supertype():
    a = pl.DataFrame({"x": pl.Series([1], dtype=pl.Int64)})
    b = pl.DataFrame({"x": pl.Series([2.5], dtype=pl.Float64)})
    out = align_and_concat([a, b])
    assert out["x"].dtype == pl.Float64
    assert out["x"].to_list() == [1.0, 2.5]


# --------------------------------------------------------------------------- #
# concat_session_runs (session clock)
# --------------------------------------------------------------------------- #
def _make_run(t_start, t_end, wheel_t, lick, reward, wheel_pos):
    return pl.DataFrame(
        {
            "trial_no": list(range(1, len(t_start) + 1)),
            "t_trialstart": pl.Series(t_start, dtype=pl.UInt64),
            "t_trialend": pl.Series(t_end, dtype=pl.UInt64),
            "wheel_t": wheel_t,  # pure time list
            "lick": lick,  # pure time list
            "reward": reward,  # [time, value] list
            "wheel_pos": wheel_pos,  # NOT a time -> must not be offset
        }
    )


@pytest.fixture
def two_runs():
    run1 = _make_run(
        t_start=[100, 300],
        t_end=[200, 400],  # max end = 400 -> run2 offset
        wheel_t=[[101.0, 150.0], [310.0]],
        lick=[[120.0], []],
        reward=[[180.0, 31.0], []],
        wheel_pos=[[-5, -6], [7]],
    )
    run2 = _make_run(
        t_start=[10, 20],
        t_end=[15, 25],
        wheel_t=[[11.0], [21.0, 22.0]],
        lick=[[12.0], []],
        reward=[[14.0, 9.0], []],
        wheel_pos=[[1], [2, 3]],
    )
    return run1, run2


def test_concat_offsets_and_numbering(two_runs):
    run1, run2 = two_runs
    out = concat_session_runs([run1, run2])

    # one row per trial across both runs
    assert out.height == 4
    # cumulative session trial numbering
    assert out["session_trial_no"].to_list() == [1, 2, 3, 4]
    # offsets: run1 = 0, run2 = max(t_trialend of run1) = 400
    assert out["run_time_offset"].to_list() == [0, 0, 400, 400]


def test_concat_scalar_session_clock(two_runs):
    run1, run2 = two_runs
    out = concat_session_runs([run1, run2])
    # run1 unchanged (offset 0), run2 shifted by 400
    assert out["t_trialstart_session"].to_list() == [100, 300, 410, 420]
    assert out["t_trialend_session"].to_list() == [200, 400, 415, 425]
    # originals are NEVER modified
    assert out["t_trialstart"].to_list() == [100, 300, 10, 20]


def test_concat_list_session_clock(two_runs):
    run1, run2 = two_runs
    out = concat_session_runs([run1, run2]).sort("session_trial_no")
    # pure-time lists: every element offset (run2 by 400)
    assert out["wheel_t_session"].to_list() == [
        [101.0, 150.0],
        [310.0],
        [411.0],
        [421.0, 422.0],
    ]
    assert out["lick_session"].to_list() == [[120.0], [], [412.0], []]
    # reward = [time, value]: only element 0 (time) is offset, value kept
    assert out["reward_session"].to_list() == [[180.0, 31.0], [], [414.0, 9.0], []]


def test_concat_does_not_offset_non_time_lists(two_runs):
    run1, run2 = two_runs
    out = concat_session_runs([run1, run2])
    # wheel_pos is positions, not times -> no *_session column created
    assert "wheel_pos_session" not in out.columns
    # and the original positions are untouched
    assert out["wheel_pos"].to_list() == [[-5, -6], [7], [1], [2, 3]]


def test_concat_single_run_is_degenerate():
    run = _make_run([5], [9], [[6.0]], [[7.0]], [[8.0, 2.0]], [[1]])
    out = concat_session_runs([run])
    assert out["run_time_offset"].to_list() == [0]
    assert out["session_trial_no"].to_list() == [1]
    assert out["t_trialstart_session"].to_list() == [5]
    # session clock equals the originals when offset is 0
    assert_frame_equal(
        out.select("t_trialstart").rename({"t_trialstart": "v"}).cast(pl.Int64),
        out.select("t_trialstart_session").rename({"t_trialstart_session": "v"}),
    )
