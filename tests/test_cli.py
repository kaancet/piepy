"""CLI arg-wiring smoke test: every subcommand parses and binds its handler.

No data is touched -- argv is parsed into a namespace and the fields the handlers
rely on are asserted. Guards against a broken subparser or a renamed flag.
"""

import pytest

from piepy.cli import (
    build_parser,
    cmd_dashboard,
    cmd_hub,
    cmd_session,
    cmd_training_report,
    cmd_widefield,
)


def test_session_parses():
    o = build_parser().parse_args(
        ["session", "240810_KC150_detect__no_cam_KC", "-p", "wheel_detection"]
    )
    assert o.command == "session" and o.func is cmd_session
    assert o.paradigm == "wheel_detection"
    assert o.load is False and o.plot is True  # defaults


def test_hub_parses_multiple_animals():
    o = build_parser().parse_args(
        ["hub", "KC150", "KC143", "-p", "wheel_detection", "--load"]
    )
    assert o.func is cmd_hub
    assert o.animalids == ["KC150", "KC143"] and o.load is True


def test_training_report_parses():
    o = build_parser().parse_args(["training-report", "KC150", "-p", "wheel_detection"])
    assert o.func is cmd_training_report and o.animalids == ["KC150"]


def test_widefield_parses():
    o = build_parser().parse_args(
        [
            "widefield",
            "240810_KC150_detect_1P__onepcam_KC",
            "-p",
            "wheel_detection",
            "--tpre",
            "0.5",
            "--tpost",
            "1.5",
            "--downsample",
            "4",
            "--precision",
            "0.001",
        ]
    )
    assert o.func is cmd_widefield
    assert o.tpre == 0.5 and o.tpost == 1.5
    assert o.downsample == 4 and o.precision == 0.001
    assert o.load is False  # default


def test_widefield_defaults():
    o = build_parser().parse_args(["widefield", "sd", "-p", "wheel_detection"])
    assert o.tpre == 0.0 and o.tpost == 0.0
    assert o.downsample == 1 and o.precision == 1e-6


def test_dashboard_parses():
    o = build_parser().parse_args(["dashboard"])
    assert o.func is cmd_dashboard


def test_shared_flags_on_each_subcommand():
    o = build_parser().parse_args(
        ["session", "x", "-p", "p", "--no-plot", "--output", "/tmp/out", "--no-verbose"]
    )
    assert o.plot is False and o.output == "/tmp/out" and o.verbose is False


def test_missing_subcommand_errors():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
