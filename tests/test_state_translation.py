"""Tests for vectorized state-transition translation (Plan 5, F06)."""

import polars as pl
import pytest

from piepy.core.errors import StateMachineError


def _sm_frame(old_new_pairs):
    """Build a minimal statemachine DataFrame from (old, new) int pairs."""
    return pl.DataFrame(
        {
            "oldState": [o for o, _ in old_new_pairs],
            "newState": [n for _, n in old_new_pairs],
            "elapsed": list(range(len(old_new_pairs))),
            "cycle": [1] * len(old_new_pairs),
            "stateElapsed": [0] * len(old_new_pairs),
        }
    )


TRANSFORM = {
    "0->1": "trialstart",
    "1->2": "stimstart",
    "2->3": "hit",
    "3->0": "trialend",
}


class _FakeRun:
    """Minimal stand-in for Run to test translate_state_changes."""

    state_transitions = TRANSFORM
    paths = None

    def __init__(self, sm):
        self.rawdata = {"statemachine": sm}


def test_fully_mapped_frame():
    from piepy.core.run import Run

    sm = _sm_frame([(0, 1), (1, 2), (2, 3), (3, 0)])
    fake = _FakeRun(sm)
    Run.translate_state_changes(fake, TRANSFORM)
    assert fake.rawdata["statemachine"]["transition"].to_list() == [
        "trialstart",
        "stimstart",
        "hit",
        "trialend",
    ]
    assert "trialNo" in fake.rawdata["statemachine"].columns


def test_unmapped_pair_raises_statemachine_error():
    from piepy.core.run import Run

    sm = _sm_frame([(0, 1), (1, 7)])  # 1->7 not in map
    fake = _FakeRun(sm)
    with pytest.raises(StateMachineError, match="1->7"):
        Run.translate_state_changes(fake, TRANSFORM)


def test_vectorized_matches_expected_values():
    """Transition column values identical to what the old row-by-row lambda produced."""
    from piepy.core.run import Run

    pairs = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 1), (1, 2), (2, 3), (3, 0)]
    sm = _sm_frame(pairs)
    sm = sm.with_columns(pl.Series("cycle", [1, 1, 1, 1, 2, 2, 2, 2]))
    fake = _FakeRun(sm)
    Run.translate_state_changes(fake, TRANSFORM)
    expected = ["trialstart", "stimstart", "hit", "trialend"] * 2
    assert fake.rawdata["statemachine"]["transition"].to_list() == expected
