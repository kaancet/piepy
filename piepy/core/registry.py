"""Paradigm registry: the single source of truth mapping a paradigm to its analysis code.

A paradigm registers its Session class (and an optional ``enrich`` hook that turns a parsed
session into the cohort-ready trial table) with :func:`register_paradigm`, used as a decorator
or a direct call::

    @register_paradigm("detection", enrich=_enrich_detection)
    class WheelDetectionSession(Session): ...

    # or
    register_paradigm("detection", WheelDetectionSession, enrich=_enrich_detection)

Both the generic :class:`piepy.core.hub.Hub` and :class:`piepy.core.mouse.Mouse` resolve their
Session class through here, so there is one place that knows paradigm -> code (replacing the old
string-munged dynamic import and Mouse's hard-coded dict).

Builtin paradigms are imported on demand (:data:`_BUILTIN_MODULES`) so their decorators run;
external paradigms register by importing their module before use.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass

__all__ = [
    "ParadigmSpec",
    "register_paradigm",
    "get_paradigm",
    "get_session_class",
    "registered_paradigms",
]


@dataclass(frozen=True)
class ParadigmSpec:
    """What the analysis pipeline needs to know about a paradigm."""

    paradigm: str
    session_cls: type
    enrich: Callable | None = None  # enrich(session) -> pl.DataFrame (cohort-ready table)


_REGISTRY: dict[str, ParadigmSpec] = {}

# builtin paradigms, imported lazily so their @register_paradigm decorators run on first lookup
_BUILTIN_MODULES: dict[str, str] = {
    "detection": "piepy.psychophysics.wheel.detection.wheelDetectionSession",
    "discrimination": "piepy.psychophysics.wheel.discrimination.wheelDiscriminationSession",
}


def register_paradigm(paradigm: str, session_cls: type | None = None, *, enrich=None):
    """Register a paradigm's Session class (+ optional enrich hook). Decorator or direct call."""

    def _apply(cls: type) -> type:
        _REGISTRY[paradigm] = ParadigmSpec(paradigm, cls, enrich)
        return cls

    return _apply(session_cls) if session_cls is not None else _apply


def get_paradigm(paradigm: str) -> ParadigmSpec:
    """Resolve a paradigm to its :class:`ParadigmSpec`, importing a builtin on demand."""
    if paradigm not in _REGISTRY and paradigm in _BUILTIN_MODULES:
        importlib.import_module(_BUILTIN_MODULES[paradigm])
    try:
        return _REGISTRY[paradigm]
    except KeyError:
        known = sorted(set(_REGISTRY) | set(_BUILTIN_MODULES))
        raise ValueError(
            f"No paradigm registered as {paradigm!r}; known: {known}. "
            "Register one with register_paradigm(...) (decorator or direct call)."
        ) from None


def get_session_class(paradigm: str) -> type:
    """The Session class for a paradigm."""
    return get_paradigm(paradigm).session_cls


def registered_paradigms() -> list[str]:
    """Paradigms registered so far (plus the builtins available on demand)."""
    return sorted(set(_REGISTRY) | set(_BUILTIN_MODULES))
