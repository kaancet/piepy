"""Provenance stamping: a parse records the resolved state-transition map (readable sidecar +
a short hash column), so a saved table is traceable. Data-free -- bare Run, real methods.
"""

from __future__ import annotations

from types import SimpleNamespace

import polars as pl

from piepy.core.run import Run, RunMeta


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


def test_is_saved_current_detects_stale_parser_version(tmp_path):
    import json

    r = _bare_run({"0->1": "trialstart"})
    r.paths = SimpleNamespace(data=[str(tmp_path / "runData.parquet")])
    assert not r.is_saved_current()  # nothing saved
    (tmp_path / "runData.parquet").touch()
    assert not r.is_saved_current()  # no sidecar
    sidecar = tmp_path / "runProvenance.json"
    sidecar.write_text(json.dumps(r._provenance()))
    assert r.is_saved_current()
    sidecar.write_text(json.dumps({**r._provenance(), "parser_version": 1}))
    assert not r.is_saved_current()  # old parser


def test_get_prot_start_time_from_run_name_and_level(tmp_path):
    f = tmp_path / "run00_164133_detectionTask_level2_x.prot"
    f.write_text("a = 1\n")
    d = RunMeta.get_prot(str(f))
    assert d["run_start_time"] == "1641" and d["level"] == 2 and d["run_name"].startswith("run00_164133")
    g = tmp_path / "run01_090502_level_x.prot"  # "level" without digits used to raise UnboundLocalError
    g.write_text("a = 1\n")
    d = RunMeta.get_prot(str(g))
    assert d["run_start_time"] == "0905" and d["level"] is None
