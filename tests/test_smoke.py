"""Data-free smoke tests.

These need no session data, so they run in CI and guard against the most common
refactor breakage: an import that silently stops working. Keep this list current
as modules move during the refactor.
"""

from __future__ import annotations

import importlib

import pytest

CORE_MODULES = [
    "piepy",
    "piepy.core.config",
    "piepy.core.run",
    "piepy.core.session",
    "piepy.core.trial",
    "piepy.core.schema",
    "piepy.core.registry",
    "piepy.core.errors",
    "piepy.core.paths",
    "piepy.stats",
    "piepy.fitting",
    "piepy.simulations",
    "piepy.core.hub",
    "piepy.core.mouse",
    "piepy.core.parsers",
    "piepy.core.log_repair_functions",
    "piepy.core.data_functions",
    "piepy.psychophysics.opto",
    "piepy.psychophysics.tasks.wheel_detection.wheelDetectionSession",
    "piepy.psychophysics.tasks.wheel_detection.wheelDetectionTrial",
    "piepy.psychophysics.tasks.wheel_discrimination.wheelDiscriminationSession",
    "piepy.psychophysics.wheelTrace",
    "piepy.viz",
    "piepy.temporal",
    "piepy.imaging.widefield",
    "piepy.cli",
]


@pytest.mark.parametrize("module_name", CORE_MODULES)
def test_module_imports(module_name):
    importlib.import_module(module_name)


def test_config_singleton_has_paths():
    from piepy.core.config import config

    assert isinstance(config.paths, dict)
    assert "analysis" in config.paths


def test_paradigm_registry_resolves():
    """Every builtin paradigm must resolve to a Session class via the registry."""
    from piepy.core.registry import get_session_class, registered_paradigms

    for paradigm in registered_paradigms():
        cls = get_session_class(paradigm)
        assert cls is not None, f"{paradigm}: registry returned None"
