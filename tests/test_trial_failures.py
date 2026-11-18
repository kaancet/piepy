"""M1/M2: a trial with no screen pulse must not crash frame lookup; a bad trial is skipped
(or raised with its number under strict). Data-free -- bare Run / TrialHandler.
"""

from __future__ import annotations

import patito as pt
import polars as pl
import pytest

from piepy.core.errors import TrialParseError
from piepy.core.run import Run
from piepy.core.trial import TrialHandler


class _M(pt.Model):
    trial_no: int


class _Handler:
    trial_model = _M

    def get_trial(self, t, rawdata):
        if t == 2:
            raise IndexError("boom")
        return {"trial_no": [t]}


def _run():
    r = object.__new__(Run)
    r.rawdata = {"statemachine": pl.DataFrame({"trialNo": [1, 2, 3]})}
    r.trial_handler = _Handler()
    r.meta = {"sessiondir": "250618_M1_detect__no_cam_KC"}
    r.state_transitions = {}
    return r


def test_set_frame_endpoints_none_epoch_does_not_crash():
    h = object.__new__(TrialHandler)
    h.data = {"imaging": pl.DataFrame({"duinotime": [1.0, 2.0], "value": [1, 2]})}
    h._trial = {}
    h.set_frame_endpoints("imaging", [None, None])
    assert h._trial["imaging_frame_ids"] == [None]


def test_get_trials_skips_and_reports():
    r = _run()
    df = r.get_trials()
    assert df["trial_no"].to_list() == [1, 3]
    assert list(r.failed_trials) == [2]
    assert r._provenance()["skipped_trials"] == [2]


def test_get_trials_strict_names_trial():
    with pytest.raises(TrialParseError, match="Trial 2 failed"):
        _run().get_trials(strict=True)


def test_get_trials_all_failed_raises():
    r = _run()
    r.rawdata = {"statemachine": pl.DataFrame({"trialNo": [2]})}
    with pytest.raises(TrialParseError, match="No trial could be parsed"):
        r.get_trials()
