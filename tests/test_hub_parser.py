"""Tests for WheelDetectionHub.parse_session_name after migrating it onto the Phase-2 parser.

Data-free (parses session-name strings only) -> runs in CI. Asserts the corrected behaviour:
the old positional ``area = parts[4]`` bug (which read "no" out of "no_cam") is fixed.
"""

from __future__ import annotations

from piepy.psychophysics.wheel.detection.wheelDetectionHub import WheelDetectionHub

parse = WheelDetectionHub.parse_session_name


def test_no_camera_session_area_is_none_not_no():
    d = parse("/data/presentation/240810_KC150_detect__no_cam_KC")
    assert d["sessiondir"] == "240810_KC150_detect__no_cam_KC"
    assert d["animalid"] == "KC150"
    assert d["date"] == "240810"
    assert d["user"] == "KC"
    assert d["paradigm"] == "detection"
    assert d["imaging"] is None
    assert d["opto_power"] is None
    assert d["area"] is None  # was the spurious "no" under the old positional parser
    assert d["isCNO"] is False
    assert d["session_path"] == "/data/presentation/240810_KC150_detect__no_cam_KC"


def test_opto_imaging_area_session():
    d = parse("240312_KC147_detect_opto120_V1__1P_KC")
    assert d["paradigm"] == "detection"
    assert d["opto_power"] == 1.2
    assert d["area"] == "V1"
    assert d["imaging"] == "1P"
    assert d["user"] == "KC"


def test_cno_area():
    d = parse("240327_KC149_detect_ALCNO__2P_KC")
    assert d["area"] == "AL"
    assert d["isCNO"] is True
    assert d["imaging"] == "2P"


def test_returns_all_expected_keys():
    d = parse("240810_KC150_detect__no_cam_KC")
    assert set(d) == {
        "session_path",
        "sessiondir",
        "date",
        "animalid",
        "user",
        "opto_power",
        "imaging",
        "paradigm",
        "area",
        "isCNO",
    }
