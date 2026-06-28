"""SessionStreams builds a temporaldata Data from the session-clock columns; trial = a slice.

The base owns the universal trials domain + a declarative `series` map; a task subclass
(WheelDetectionStreams) adds the rig-clock streams with the rig->state shift.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from piepy.tasks.wheel_detection.wheelDetectionStreams import WheelDetectionStreams
from piepy.temporal import SessionStreams, trial_slice


def _session_df() -> pl.DataFrame:
    """Two trials on the session clock. Rig stim onset is 5 ms behind the state onset, so the
    rig streams (wheel/lick/reward) must be shifted +5 to land on the state clock."""
    return pl.DataFrame(
        {
            "trial_no": [1, 2],
            "outcome": ["hit", "miss"],
            "t_trialstart_session": [0.0, 1000.0],
            "t_trialend_session": [800.0, 1800.0],
            "t_vstimstart_session": [300.0, 1300.0],  # state clock
            "t_vstimstart_rig_session": [295.0, 1295.0],  # rig clock (5 ms behind) -> offset = +5
            "t_vstimend_session": [600.0, 1600.0],
            "wheel_t_session": [[295.0, 345.0], [1295.0, 1345.0]],  # rig clock
            "wheel_pos": [[0.0, 2.0], [0.0, -1.0]],
            "lick_session": [[400.0], []],  # rig clock
            "reward_session": [[395.0, 1.5], []],  # [time, amount], rig clock
        }
    )


def test_wheel_subclass_builds_and_shifts_rig_to_state_clock():
    data = WheelDetectionStreams(_session_df()).build()
    assert set(data.keys()) >= {"trials", "wheel", "licks", "reward", "stim"}

    # trials domain carries metadata
    assert np.array_equal(np.asarray(data.trials.trial_no), [1, 2])
    assert list(np.asarray(data.trials.outcome)) == ["hit", "miss"]

    # wheel rig times (295,345,...) shifted +5 -> 300, 350, ...
    assert np.allclose(np.asarray(data.wheel.timestamps), [300.0, 350.0, 1300.0, 1350.0])
    assert np.allclose(np.asarray(data.wheel.position), [0.0, 2.0, 0.0, -1.0])
    # reward = leading element only, 395 + 5 = 400; lick 400 + 5 = 405
    assert np.allclose(np.asarray(data.reward.timestamps), [400.0])
    assert np.allclose(np.asarray(data.licks.timestamps), [405.0])


def test_trial_is_a_slice_relative_time():
    data = WheelDetectionStreams(_session_df()).build()
    t2 = trial_slice(data, 2)  # window 1000..1800 -> relative time
    assert np.allclose(np.asarray(t2.wheel.timestamps), [300.0, 350.0])  # 1300,1350 -> rel start


def test_base_declarative_series_builds_irregular_from_session_list():
    # a task can expose a simple extra column with one line: series = {"pulse": "irregular"}
    class Toy(SessionStreams):
        series = {"pulse": "irregular"}

    df = _session_df().with_columns(pl.Series("pulse_session", [[310.0, 320.0], [1310.0]]))
    data = Toy(df).build()
    assert "pulse" in data.keys()
    assert np.allclose(np.asarray(data.pulse.timestamps), [310.0, 320.0, 1310.0])


def test_missing_session_columns_errors():
    bare = pl.DataFrame({"trial_no": [1], "wheel_t_session": [[1.0]], "wheel_pos": [[0.0]]})
    try:
        WheelDetectionStreams(bare)
    except ValueError as e:
        assert "session-clock columns" in str(e)
    else:
        raise AssertionError("expected ValueError for missing session-clock columns")
