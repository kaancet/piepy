"""The channel-agnostic StimPy primitives on base TrialHandler (set_trial windowing +
state_table/transition_time/rig_event) are enough to parse a NON-visual task -- statemachine +
a 'lick' rig channel, no vstim/screen. This is the reuse surface a non-wheel get_trial builds on.
"""

from __future__ import annotations

import polars as pl

from piepy.core.trial import TrialHandler


def _rawdata():
    return {
        "statemachine": pl.DataFrame(
            {
                "trialNo": [1, 1, 1, 2, 2, 2],
                "transition": [
                    "trialstart",
                    "stimstart",
                    "trialend",
                    "trialstart",
                    "stimstart",
                    "trialend",
                ],
                "elapsed": [0, 10, 50, 100, 110, 150],
            }
        ),
        "lick": pl.DataFrame({"duinotime": [12, 40, 120], "value": [1, 1, 1]}),
    }


def test_set_trial_windows_channels_and_exposes_state():
    h = TrialHandler()
    h.init_trial()  # get_trial does this first; required before set_trial
    assert h.set_trial(1, _rawdata()) is True
    # trial endpoints come from the statemachine, no visual channels needed
    assert h._trial["t_trialstart"] == 0
    assert h._trial["t_trialend"] == 50
    # state events via the public accessors
    assert h.transition_time("stimstart") == 10
    assert h.transition_time("not_a_transition") is None
    assert h.state_table().filter(pl.col("transition") == "trialstart").height == 1


def test_rig_event_is_windowed_per_trial():
    h = TrialHandler()
    h.init_trial()
    h.set_trial(1, _rawdata())  # window [0, 50] -> licks at 12, 40
    assert len(h.rig_event("lick")) == 2
    h.init_trial()
    h.set_trial(2, _rawdata())  # window [100, 150] -> lick at 120
    assert len(h.rig_event("lick")) == 1
    assert h.rig_event("reward") is None  # absent channel -> None, not a crash


def test_set_trial_skips_none_channel():
    """A rawdata entry that is None should not crash set_trial."""
    raw = _rawdata()
    raw["broken_channel"] = None
    h = TrialHandler()
    h.init_trial()
    assert h.set_trial(1, raw) is True
    assert "broken_channel" not in h.data


def test_set_trial_skips_non_vstim_presenttime_channel():
    """A channel with 'presentTime' column that isn't 'vstim' should be skipped."""
    raw = _rawdata()
    raw["photo"] = pl.DataFrame({"presentTime": [0.01, 0.02], "code": [1, 1]})
    h = TrialHandler()
    h.init_trial()
    assert h.set_trial(1, raw) is True
    assert "photo" not in h.data
