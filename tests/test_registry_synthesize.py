"""Wiring-only paradigm registration: register a task with just a TrialHandler + a
state-transition map (+ optional RunData) and get a working Session/Run synthesized for it,
with no Run/Session boilerplate. Data-free -- inspects the synthesized classes, does not parse.
"""

from __future__ import annotations

import json

from piepy.core.paths import parse_session_name
from piepy.core.registry import (
    get_paradigm,
    get_session_class,
    load_scheme,
    register_paradigm,
    registered_paradigms,
)
from piepy.core.run import Run, RunData
from piepy.core.session import Session
from piepy.core.trial import TrialHandler

TRANSITIONS = {"0->1": "trialstart", "1->2": "stimstart", "2->0": "trialend"}


class _ToyHandler(TrialHandler):
    def get_trial(self, trial_no, rawdata):  # pragma: no cover - not parsed here
        return None


class _ToyRunData(RunData):
    pass


def test_wiring_only_registration_synthesizes_session():
    cls = register_paradigm(
        "toytask",
        trial_handler_cls=_ToyHandler,
        state_transitions=TRANSITIONS,
        rundata_cls=_ToyRunData,
    )

    # returns + registers a Session subclass named after the paradigm
    assert cls is get_session_class("toytask")
    assert issubclass(cls, Session)
    assert cls.__name__ == "ToytaskSession"
    assert cls.paradigm == "toytask"

    # its Run is wired to the parts we passed (the base Session/Run read these attrs)
    run_cls = cls.run_cls
    assert issubclass(run_cls, Run)
    assert run_cls.trial_handler_cls is _ToyHandler
    assert run_cls.rundata_cls is _ToyRunData
    assert run_cls.state_transitions == TRANSITIONS

    # registry sees it; the stored spec has no per-paradigm enrich hook
    assert "toytask" in registered_paradigms()
    assert get_paradigm("toytask").enrich is None


def test_rundata_defaults_to_base_when_omitted():
    cls = register_paradigm(
        "toytask_norundata",
        trial_handler_cls=_ToyHandler,
        state_transitions=TRANSITIONS,
    )
    assert cls.run_cls.rundata_cls is RunData
    # snake_case paradigm -> CamelCase class name
    assert cls.__name__ == "ToytaskNorundataSession"


def test_explicit_session_cls_takes_precedence():
    class _CustomSession(Session):
        run_cls = Run

    cls = register_paradigm(
        "toytask_custom",
        _CustomSession,
        trial_handler_cls=_ToyHandler,  # ignored: explicit session_cls wins
    )
    assert cls is _CustomSession
    assert get_session_class("toytask_custom") is _CustomSession


# --------------------------------------------------------------------------- #
# scheme.json from the ~/.piepy/paradigms drop-in store (transitions + naming_template)
# --------------------------------------------------------------------------- #
def _write_scheme(tmp_path, name, *, transitions=None, template=None):
    d = tmp_path / name
    d.mkdir()
    scheme = {}
    if transitions is not None:
        scheme["state_transitions"] = transitions
    if template is not None:
        scheme["naming_template"] = template
    (d / "scheme.json").write_text(json.dumps(scheme))


def test_config_exposes_paradigms_path():
    from piepy.core.config import config

    assert "paradigms" in config.paths


def test_load_scheme_from_json(tmp_path, monkeypatch):
    from piepy.core.config import config

    monkeypatch.setitem(config.paths, "paradigms", [str(tmp_path)])
    _write_scheme(tmp_path, "jsontask", transitions=TRANSITIONS)

    assert load_scheme("jsontask")["state_transitions"] == TRANSITIONS
    assert load_scheme("absent") is None


def test_wiring_only_picks_up_json_transitions_when_omitted(tmp_path, monkeypatch):
    from piepy.core.config import config

    monkeypatch.setitem(config.paths, "paradigms", [str(tmp_path)])
    _write_scheme(tmp_path, "jsonwire", transitions=TRANSITIONS)

    cls = register_paradigm("jsonwire", trial_handler_cls=_ToyHandler)  # no kwarg
    assert cls.run_cls.state_transitions == TRANSITIONS


def test_explicit_transitions_win_over_json(tmp_path, monkeypatch):
    from piepy.core.config import config

    monkeypatch.setitem(config.paths, "paradigms", [str(tmp_path)])
    _write_scheme(tmp_path, "bothwire", transitions={"0->1": "from_json"})

    inline = {"0->1": "trialstart"}
    cls = register_paradigm("bothwire", trial_handler_cls=_ToyHandler, state_transitions=inline)
    assert cls.run_cls.state_transitions == inline


def test_naming_template_registers_a_session_name_scheme(tmp_path, monkeypatch):
    from piepy.core.config import config

    monkeypatch.setitem(config.paths, "paradigms", [str(tmp_path)])
    template = r"(?P<date>\d{6})_(?P<animalid>[^_]+)_mytaskparse__(?P<user>[^_]+)"
    _write_scheme(tmp_path, "mytaskparse", transitions=TRANSITIONS, template=template)

    register_paradigm("mytaskparse", trial_handler_cls=_ToyHandler)

    # the previously-unparseable name now resolves, paradigm inferred from the scheme owner
    parsed = parse_session_name("250618_M001_mytaskparse__bob")
    assert parsed.paradigm == "mytaskparse"
    assert parsed.animalid == "M001"
    assert parsed.baredate == "250618"
    assert parsed.extra["user"] == "bob"

    # a name that doesn't match still falls back to the in-house default scheme
    default = parse_session_name("240810_KC150_detect__no_cam_KC")
    assert default.paradigm == "detection"
