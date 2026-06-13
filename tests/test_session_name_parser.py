"""Unit tests for the pluggable session-name parser (Phase 2, chunk 1). Data-free -> CI."""

from __future__ import annotations

import datetime

import pytest

from piepy.core.errors import MalformedSessionError
from piepy.core.paths import (
    DefaultSessionNameParser,
    SessionName,
    parse_session_name,
    set_session_name_parser,
)


def test_behaviour_no_camera():
    s = parse_session_name("240810_KC150_detect__no_cam_KC")
    assert (s.baredate, s.animalid, s.paradigm) == ("240810", "KC150", "detection")
    assert s.date == datetime.date(2024, 8, 10)
    assert s.extra["imaging"] is None
    assert s.extra["user"] == "KC"
    assert s.extra["opto_power"] is None


def test_opto_imaging_area():
    s = parse_session_name("240312_KC147_detect_opto120_V1__1P_KC")
    assert s.paradigm == "detection"
    assert s.extra["opto_power"] == 1.2
    assert s.extra["area"] == "V1"
    assert s.extra["imaging"] == "1P"
    assert s.extra["user"] == "KC"


def test_discrimination():
    s = parse_session_name("250217_VB101_discrim_opto120_V1__no_cam_VO")
    assert s.paradigm == "discrimination"
    assert s.extra == {
        "user": "VO",
        "imaging": None,
        "opto_power": 1.2,
        "area": "V1",
        "isCNO": False,
    }


def test_accepts_full_path():
    s = parse_session_name("/data/presentation/240810_KC150_detect__no_cam_KC/")
    assert s.sessiondir == "240810_KC150_detect__no_cam_KC"
    assert s.animalid == "KC150"


def test_cno_area():
    s = parse_session_name("240327_KC149_detect_ALCNO__2P_KC")
    assert s.extra["area"] == "AL"
    assert s.extra["isCNO"] is True


@pytest.mark.parametrize("bad", ["garbage", "KC150_detect", "ABCDEF_KC150_detect__1P_KC"])
def test_malformed_names_raise_actionable_error(bad):
    with pytest.raises(MalformedSessionError) as exc:
        parse_session_name(bad)
    # the error carries a 'fix' the user can act on
    assert "fix:" in str(exc.value)


def test_invalid_date_raises():
    with pytest.raises(MalformedSessionError):
        parse_session_name("249999_KC150_detect__no_cam_KC")  # month 99


def test_parser_is_pluggable():
    class MyParser:
        def __call__(self, sessiondir: str) -> SessionName:
            return SessionName(sessiondir, "000000", "ZZ", datetime.date(2000, 1, 1))

    default = DefaultSessionNameParser()
    try:
        set_session_name_parser(MyParser())
        s = parse_session_name("whatever_string")
        assert s.animalid == "ZZ"
    finally:
        set_session_name_parser(default)  # restore for other tests
