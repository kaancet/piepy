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
    "piepy.core.pathfinder",
    "piepy.core.run",
    "piepy.core.session",
    "piepy.core.trial",
    "piepy.core.hub",
    "piepy.core.statistics",
    "piepy.core.parsers",
    "piepy.core.log_repair_functions",
    "piepy.core.data_functions",
    "piepy.psychophysics.fit_funcs",
    "piepy.psychophysics.wheel.detection.wheelDetectionSession",
    "piepy.psychophysics.wheel.detection.wheelDetectionTrial",
    "piepy.psychophysics.wheel.discrimination.wheelDiscriminationSession",
    "piepy.psychophysics.wheel.wheelTrace",
]


@pytest.mark.parametrize("module_name", CORE_MODULES)
def test_module_imports(module_name):
    importlib.import_module(module_name)


def test_config_singleton_has_paths():
    from piepy.core.config import config

    assert isinstance(config.paths, dict)
    assert "analysis" in config.paths


def test_paradigm_registry_resolves():
    """Every paradigm declared for the golden tests must point at a real class."""
    from conftest import PARADIGMS

    for paradigm, (mod_name, cls_name) in PARADIGMS.items():
        mod = importlib.import_module(mod_name)
        assert hasattr(mod, cls_name), f"{paradigm}: {mod_name}.{cls_name} missing"
