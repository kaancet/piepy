"""Tests for the paradigm registry and the generic Hub (Phase 2.5).

Registry + Hub plumbing are data-free (CI); a single needs_data test exercises the detection
enrich hook on a real session.
"""

from __future__ import annotations

import datetime

import polars as pl
import pytest

from piepy.core.registry import (
    get_paradigm,
    get_session_class,
    register_paradigm,
    registered_paradigms,
)


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
class _Dummy:
    pass


def test_register_and_resolve():
    register_paradigm("dummy_reg", _Dummy)
    assert get_session_class("dummy_reg") is _Dummy
    assert get_paradigm("dummy_reg").session_cls is _Dummy
    assert _Dummy.paradigm == "dummy_reg"  # name stamped on the class
    assert "dummy_reg" in registered_paradigms()


def test_register_as_decorator():
    @register_paradigm("dummy_deco")
    class Sess:
        pass

    assert get_session_class("dummy_deco") is Sess


def test_unknown_paradigm_raises():
    with pytest.raises(ValueError, match="No paradigm registered"):
        get_paradigm("does_not_exist_xyz")


# --------------------------------------------------------------------------- #
# Hub plumbing
# --------------------------------------------------------------------------- #
def test_combine_session_data_sorts_and_numbers():
    from piepy.core.hub import _combine_session_data

    a = pl.DataFrame(
        {"date": [datetime.date(2024, 1, 2)], "animalid": ["A"], "run_no": [1], "x": [1]}
    )
    b = pl.DataFrame(
        {"date": [datetime.date(2024, 1, 1)], "animalid": ["A"], "run_no": [1], "x": [2]}
    )
    out = _combine_session_data([a, pl.DataFrame(), b])
    assert out.columns[0] == "total_trial_no"
    assert out["total_trial_no"].to_list() == [1, 2]
    assert out["x"].to_list() == [2, 1]  # sorted by date ascending


def test_analyze_one_catches_failures():
    """_analyze_one wraps both construction AND analyze(); returns error string on failure."""
    from piepy.core.hub import _analyze_one

    frame, err = _analyze_one(("wheel_detection", False, "nonexistent_session_dir"))
    assert frame.is_empty()
    assert err is not None
    assert "nonexistent_session_dir" in err


def test_analyze_one_returns_none_err_on_success(monkeypatch):
    """On success, error is None."""
    from piepy.core.hub import _analyze_one

    class FakeSession:
        def __init__(self, name):
            pass

        def analyze(self, load_flag=False):
            return pl.DataFrame({"x": [1]})

    class FakeSpec:
        session_cls = FakeSession

    monkeypatch.setattr("piepy.core.hub.get_paradigm", lambda p: FakeSpec())
    frame, err = _analyze_one(("fake", False, "some_session"))
    assert not frame.is_empty()
    assert err is None


def test_gather_sessions_sequential_collects_failures(monkeypatch):
    """Sequential gather collects failures and still returns successful frames."""
    from piepy.core.hub import Hub

    call_count = {"n": 0}

    def fake_analyze_one(args):
        call_count["n"] += 1
        paradigm, load_flag, sessiondir = args
        if "bad" in sessiondir:
            return pl.DataFrame(), f"{sessiondir} : ValueError — broke"
        return (
            pl.DataFrame({"date": ["2024-01-01"], "animalid": ["A"], "run_no": [1]}),
            None,
        )

    monkeypatch.setattr("piepy.core.hub._analyze_one", fake_analyze_one)
    monkeypatch.setattr("piepy.core.hub.cfg.multiprocess", {"enable": False, "cores": 1})

    hub = Hub("wheel_detection")
    hub.gather_sessions(["good_session", "bad_session", "good_session2"])

    assert call_count["n"] == 3
    assert hub.data is not None
    assert hub.data.height == 2


def test_save_defaults_to_analysis_path(monkeypatch, tmp_path):
    """save(None) uses cfg.paths['analysis'][0], not a nonexistent session_path column."""
    from piepy.core.hub import Hub

    monkeypatch.setattr("piepy.core.hub.cfg.paths", {"analysis": [str(tmp_path / "out")]})

    hub = Hub("wheel_detection")
    hub.data = pl.DataFrame({"baredate": ["20240101"], "animalid": ["A"], "x": [1]})
    hub.save()
    saved = list((tmp_path / "out").glob("*.parquet"))
    assert len(saved) == 1


@pytest.mark.needs_data
def test_hub_one_session_enriches_real_detection(redirect_analysis):
    from piepy.core.hub import Hub

    out = Hub("wheel_detection")._one_session("230106_KC144_detect__no_cam_KC")
    if out.is_empty():
        pytest.skip("230106 detection session not available locally")
    # enrich hook added the cohort columns on top of the canonical identity
    assert any(c.startswith("stat_") for c in out.columns)
    assert {"session_id", "signed_contrast", "session_uid", "run_no", "paradigm"} <= set(
        out.columns
    )
    assert out["paradigm"].unique().to_list() == ["wheel_detection"]
