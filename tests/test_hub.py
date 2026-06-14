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
    assert get_paradigm("dummy_reg").enrich is None
    assert "dummy_reg" in registered_paradigms()


def test_register_with_enrich_as_direct_call():
    register_paradigm("dummy_enrich", _Dummy, enrich=lambda s: s)
    assert get_paradigm("dummy_enrich").enrich is not None


def test_register_as_decorator():
    @register_paradigm("dummy_deco")
    class Sess:
        pass

    assert get_session_class("dummy_deco") is Sess


def test_builtin_detection_lazy_loads_with_enrich():
    # detection is a builtin: get_paradigm imports its module on demand, no manual import
    spec = get_paradigm("detection")
    assert spec.session_cls.__name__ == "WheelDetectionSession"
    assert spec.enrich is not None  # detection registers an enrich hook


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


def test_hub_filters_session_list_by_paradigm():
    from piepy.core.hub import Hub

    hub = Hub("detection")
    kept = hub._filter_session_list(
        [
            "240810_KC150_detect__no_cam_KC",
            "250217_VB101_discrim_opto120_V1__no_cam_VO",
            "garbage-name",
        ]
    )
    assert kept == ["240810_KC150_detect__no_cam_KC"]


@pytest.mark.needs_data
def test_hub_one_session_enriches_real_detection(redirect_analysis):
    from piepy.core.hub import Hub

    out = Hub("detection")._one_session("230106_KC144_detect__no_cam_KC")
    if out.is_empty():
        pytest.skip("230106 detection session not available locally")
    # enrich hook added the cohort columns on top of the canonical identity
    assert any(c.startswith("stat_") for c in out.columns)
    assert {"session_id", "signed_contrast", "session_uid", "run_no", "paradigm"} <= set(
        out.columns
    )
    assert out["paradigm"].unique().to_list() == ["detection"]
