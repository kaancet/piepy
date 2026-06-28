"""Provenance stamping: a parse records the resolved state-transition map (readable sidecar +
a short hash column), so a saved table is traceable. Data-free -- bare Run, real methods.
"""

from __future__ import annotations

from types import SimpleNamespace

import polars as pl

from piepy.core.run import Run


def _bare_run(transitions, sessiondir="250618_M1_detect__no_cam_KC"):
    r = object.__new__(Run)  # skip __init__ (needs real paths)
    r.state_transitions = transitions
    r.meta = {"sessiondir": sessiondir}
    return r


def test_provenance_hash_is_deterministic_and_order_independent():
    p1 = _bare_run({"0->1": "trialstart", "1->2": "stimstart"})._provenance()
    p2 = _bare_run({"1->2": "stimstart", "0->1": "trialstart"})._provenance()  # reordered
    assert p1["state_transitions_hash"] == p2["state_transitions_hash"]
    assert len(p1["state_transitions_hash"]) == 12
    assert p1["state_transitions"] == {"0->1": "trialstart", "1->2": "stimstart"}
    assert p1["piepy_version"]  # non-empty
    assert "parsed_at" in p1
    assert p1["paradigm"] == "wheel_detection"  # parsed from the session dir


def test_different_map_gives_different_hash():
    a = _bare_run({"0->1": "trialstart"})._provenance()["state_transitions_hash"]
    b = _bare_run({"0->1": "cuestart"})._provenance()["state_transitions_hash"]
    assert a != b


def test_stamp_provenance_adds_one_hash_column():
    r = _bare_run({"0->1": "trialstart"})
    r.data = SimpleNamespace(data=pl.DataFrame({"trial_no": [1, 2, 3]}))
    r._stamp_provenance()
    col = r.data.data["state_transitions_hash"]
    assert col.n_unique() == 1
    assert col[0] == r.provenance["state_transitions_hash"]
