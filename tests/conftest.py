"""Shared pytest fixtures and helpers for the piepy test-suite.

The golden-master parse tests run real sessions through the analysis pipeline and
compare the resulting trial table against a stored snapshot, so we can refactor
aggressively and immediately see if parsed output drifts.

Design notes
------------
* Sessions are resolved through ``~/.piepy/config.json`` (see ``golden_sessions.toml``).
  Missing sessions => the test is skipped, never failed.
* Re-parsing a session normally re-saves ``runData.parquet`` into the configured
  ``analysis`` dir. The ``redirect_analysis`` fixture points that dir at a temp
  location so tests never overwrite the user's real analysis output.
* Snapshots live under ``tests/_snapshots`` (gitignored, override with
  ``PIEPY_GOLDEN_DIR``). Generate/refresh them with ``pytest --update-golden``.
"""

from __future__ import annotations

import importlib
import os
import tomllib
from pathlib import Path

import pytest

HERE = Path(__file__).parent
SESSIONS_TOML = HERE / "golden_sessions.toml"
DEFAULT_SNAPSHOT_DIR = HERE / "_snapshots"

# paradigm name -> (module path, Session class name)
PARADIGMS: dict[str, tuple[str, str]] = {
    "wheel_detection": (
        "piepy.tasks.wheel_detection.wheelDetectionSession",
        "WheelDetectionSession",
    ),
    "wheel_discrimination": (
        "piepy.tasks.wheel_discrimination.wheelDiscriminationSession",
        "WheelDiscriminationSession",
    ),
    "visual": (
        "piepy.tasks.sensory.visual.visualSession",
        "VisualSession",
    ),
}


class SessionUnavailable(Exception):
    """Raised when a configured golden session cannot be found locally."""


def load_cases() -> list[tuple[str, str]]:
    """Flatten golden_sessions.toml into ``(paradigm, session_dir)`` tuples."""
    if not SESSIONS_TOML.exists():
        return []
    with SESSIONS_TOML.open("rb") as fh:
        data = tomllib.load(fh)
    cases: list[tuple[str, str]] = []
    for paradigm, entries in data.items():
        for entry in entries:
            cases.append((paradigm, entry["session"]))
    return cases


def build_session(paradigm: str, session_dir: str):
    """Construct the Session for ``paradigm`` (forces a fresh re-parse)."""
    if paradigm not in PARADIGMS:
        raise ValueError(f"Unknown paradigm {paradigm!r}; known: {sorted(PARADIGMS)}")
    from piepy.core.errors import PathfindingError

    mod_name, cls_name = PARADIGMS[paradigm]
    session_cls = getattr(importlib.import_module(mod_name), cls_name)
    try:
        sess = session_cls(session_dir)
        sess.analyze(load_flag=False)
        return sess

    except (FileNotFoundError, PathfindingError) as exc:
        # not found locally, ambiguous, or malformed -> can't resolve here, so skip not fail.
        raise SessionUnavailable(
            f"{paradigm} session {session_dir!r} not resolvable locally: {exc}"
        ) from exc


# --------------------------------------------------------------------------- #
# pytest wiring
# --------------------------------------------------------------------------- #
def pytest_addoption(parser):
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="(Re)generate golden snapshot parquet files instead of comparing.",
    )


def pytest_generate_tests(metafunc):
    """Parametrize any test that asks for ``paradigm`` + ``session_dir``."""
    if {"paradigm", "session_dir"} <= set(metafunc.fixturenames):
        cases = load_cases()
        metafunc.parametrize(
            ("paradigm", "session_dir"),
            cases,
            ids=[f"{p}:{s}" for p, s in cases],
        )


@pytest.fixture(scope="session")
def update_golden(request) -> bool:
    return bool(request.config.getoption("--update-golden"))


@pytest.fixture(scope="session")
def snapshot_dir() -> Path:
    d = Path(os.environ.get("PIEPY_GOLDEN_DIR", DEFAULT_SNAPSHOT_DIR))
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(autouse=True)
def _quiet_piepy(monkeypatch):
    """Silence piepy's verbose logging/progress bars during tests."""
    from piepy.core.config import config

    monkeypatch.setattr(config, "verbose", False, raising=False)


@pytest.fixture
def redirect_analysis(tmp_path, monkeypatch):
    """Redirect the configured 'analysis' (save) dir to a temp location so a
    re-parse during tests never overwrites the user's real analysis output."""
    from piepy.core.config import config

    monkeypatch.setitem(config.paths, "analysis", [str(tmp_path / "analysis")])
    yield tmp_path


@pytest.fixture
def parsed_session(paradigm, session_dir, redirect_analysis):
    """A freshly-parsed Session, or a skip if the data is not present locally."""
    try:
        return build_session(paradigm, session_dir)
    except SessionUnavailable as exc:
        pytest.skip(str(exc))
