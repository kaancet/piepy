"""Paradigm registry: the single source of truth mapping a paradigm to its analysis code.

A paradigm registers its Session class with :func:`register_paradigm`, used as a decorator or a
direct call. The Session's ``analyze()`` (``concatenate_runs`` + ``enrich``) produces the
cohort-ready table; a paradigm overrides ``Session.enrich`` to add per-run/cohort columns::

    @register_paradigm("wheel_detection")
    class WheelDetectionSession(Session): ...

    # or
    register_paradigm("wheel_detection", WheelDetectionSession)

A paradigm that needs no custom per-run flow can skip the Session/Run entirely and register the
*parts* -- a TrialHandler, a state-transition map, and (optionally) a RunData -- letting the
registry synthesize a generic Run + Session::

    register_paradigm("mytask", trial_handler_cls=MyTrialHandler,
                      state_transitions={"0->1": "trialstart", ...}, rundata_cls=MyRunData)

Both the generic :class:`piepy.core.hub.Hub` and :class:`piepy.core.mouse.Mouse` resolve their
Session class through here, so there is one place that knows paradigm -> code (replacing the old
string-munged dynamic import and Mouse's hard-coded dict).

Builtin paradigms are imported on demand (:data:`_BUILTIN_MODULES`) so their decorators run;
external paradigms register by importing their module before use.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass

__all__ = [
    "ParadigmSpec",
    "register_paradigm",
    "get_paradigm",
    "get_session_class",
    "registered_paradigms",
    "load_scheme",
]


@dataclass(frozen=True)
class ParadigmSpec:
    """What the analysis pipeline needs to know about a paradigm."""

    paradigm: str
    session_cls: (
        type  # its .analyze() returns the cohort-ready table (concatenate_runs + enrich)
    )


_REGISTRY: dict[str, ParadigmSpec] = {}

# builtin paradigms, imported lazily so their @register_paradigm decorators run on first lookup
_BUILTIN_MODULES: dict[str, str] = {
    "wheel_detection": "piepy.psychophysics.tasks.wheel_detection.wheelDetectionSession",
    "wheel_discrimination": "piepy.psychophysics.tasks.wheel_discrimination.wheelDiscriminationSession",
    "visual": "piepy.sensory.visual.visualSession",
}


def register_paradigm(
    paradigm: str,
    session_cls: type | None = None,
    *,
    trial_handler_cls: type | None = None,
    rundata_cls: type | None = None,
    state_transitions: dict | None = None,
):
    """Register a paradigm. Three call styles:

    * decorator on a Session subclass:  ``@register_paradigm("x")``
    * direct with a Session subclass:   ``register_paradigm("x", XSession)``
    * wiring-only (no Run/Session needed)::

        register_paradigm("x", trial_handler_cls=XHandler,
                          state_transitions={...}, rundata_cls=XRunData)

    The wiring-only form synthesizes a generic Run + Session from the parts, so a new paradigm
    needs only a ``Trial`` schema, a ``TrialHandler``, and a state-transition map -- no
    Run/Session boilerplate. An explicit ``session_cls`` always wins (for paradigms that need
    custom per-run hooks, e.g. a custom ``Session.enrich`` for cohort columns).
    """

    def _store(cls: type) -> type:
        cls.paradigm = (
            paradigm  # so Session.analyze()/concatenate_runs() know their own name
        )
        _REGISTRY[paradigm] = ParadigmSpec(paradigm, cls)
        return cls

    if session_cls is not None:
        return _store(session_cls)
    if trial_handler_cls is not None:
        # pull the paradigm's scheme.json (~/.piepy/paradigms/<name>/scheme.json) for the
        # naming template + state-transition map; an explicit kwarg still wins for transitions.
        scheme = load_scheme(paradigm) or {}
        if state_transitions is None:
            state_transitions = scheme.get("state_transitions")
        _validate_transitions(paradigm, trial_handler_cls, state_transitions)
        if scheme.get("naming_template"):
            from .paths.parser import register_scheme

            register_scheme(paradigm, scheme["naming_template"])
        return _store(
            _build_session_cls(
                paradigm, trial_handler_cls, rundata_cls, state_transitions
            )
        )
    return _store  # decorator form: @register_paradigm("x")


def _validate_transitions(
    paradigm: str, trial_handler_cls: type, state_transitions: dict | None
) -> None:
    """Check the state-transition map produces every name the handler declares it needs.

    Fails at registration with a clear message instead of a deep ``StateMachineError`` once
    parsing hits an unmapped transition. No-op when the handler declares no requirements.
    """
    required = getattr(trial_handler_cls, "required_transitions", None) or frozenset()
    if not required:
        return
    produced = set((state_transitions or {}).values())
    missing = set(required) - produced
    if missing:
        raise ValueError(
            f"Paradigm {paradigm!r}: state-transition map is missing transition name(s) "
            f"{sorted(missing)} required by {trial_handler_cls.__name__}. Add them to "
            f"<paradigms_path>/{paradigm}/scheme.json (or the state_transitions= map)."
        )


def load_scheme(paradigm: str) -> dict | None:
    """Load a paradigm's ``scheme.json`` from the ``paradigms`` config path.

    Looks for ``<paradigms_path>/<paradigm>/scheme.json`` (default
    ``~/.piepy/paradigms/<paradigm>/scheme.json``), a JSON object with optional keys
    ``naming_template`` (a regex defining the session-name scheme; must capture ``date`` and
    ``animalid``) and ``state_transitions`` (the ``'<old>-><new>' -> name`` map). Returns
    ``None`` when no file is present, so callers fall back to in-code values.
    """
    import json
    from pathlib import Path

    from .config import config as cfg

    for root in cfg.paths.get("paradigms") or []:
        path = Path(root) / paradigm / "scheme.json"
        if path.exists():
            with path.open() as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError(
                    f"{path} must hold a JSON object with 'naming_template' and/or "
                    f"'state_transitions'; got {type(data).__name__}."
                )
            return data
    return None


def _build_session_cls(
    paradigm: str,
    trial_handler_cls: type,
    rundata_cls: type | None,
    state_transitions: dict | None,
) -> type:
    """Synthesize a generic Session (and its Run) wired to the given paradigm parts.

    Real named classes (``FooRun``/``FooSession``), not a metaclass trick, so tracebacks and
    ``repr`` stay readable. The wiring lives on the Run as plain class attributes that the base
    ``Session``/``Run`` already read.
    """
    from .run import Run, RunData
    from .session import Session

    name = (
        "".join(w.capitalize() for w in paradigm.replace("_", " ").split()) or "Paradigm"
    )
    run_cls = type(
        f"{name}Run",
        (Run,),
        {
            "trial_handler_cls": trial_handler_cls,
            "rundata_cls": rundata_cls or RunData,
            "state_transitions": dict(state_transitions or {}),
        },
    )
    return type(f"{name}Session", (Session,), {"run_cls": run_cls, "paradigm": paradigm})


def get_paradigm(paradigm: str) -> ParadigmSpec:
    """Resolve a paradigm to its :class:`ParadigmSpec`.

    On a registry miss: import a builtin module on demand, else try to discover a drop-in
    paradigm (``<paradigms_path>/<name>/handler.py``) and import it so its register call runs.
    This is what lets a spawned Hub worker rebuild a paradigm from just its name.
    """
    if paradigm not in _REGISTRY:
        if paradigm in _BUILTIN_MODULES:
            importlib.import_module(_BUILTIN_MODULES[paradigm])
        else:
            _discover_paradigm(paradigm)
    try:
        return _REGISTRY[paradigm]
    except KeyError:
        known = sorted(set(_REGISTRY) | set(_BUILTIN_MODULES))
        raise ValueError(
            f"No paradigm registered as {paradigm!r}; known: {known}. "
            "Register one with register_paradigm(...), or drop a handler.py + scheme.json "
            "into <paradigms_path>/<name>/."
        ) from None


def _discover_paradigm(paradigm: str) -> None:
    """Import a drop-in paradigm's ``handler.py`` (runs its register call) from the store.

    Looks for ``<root>/<paradigm>/handler.py`` for each root in ``config.paths['paradigms']``
    (default ``~/.piepy/paradigms``). Loaded by file path, so a handler is a single self-
    contained module. Silent no-op when none is found (the caller then raises a clear error).
    """
    import importlib.util
    from pathlib import Path

    from .config import config as cfg

    for root in cfg.paths.get("paradigms") or []:
        handler = Path(root) / paradigm / "handler.py"
        if not handler.exists():
            continue
        mod_name = f"piepy_paradigm_{paradigm}"
        spec = importlib.util.spec_from_file_location(mod_name, handler)
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module  # cache so re-resolution is cheap
        spec.loader.exec_module(module)  # runs module-level register_paradigm(...)
        return


def get_session_class(paradigm: str) -> type:
    """The Session class for a paradigm."""
    return get_paradigm(paradigm).session_cls


def registered_paradigms() -> list[str]:
    """Paradigms registered so far (plus the builtins available on demand)."""
    return sorted(set(_REGISTRY) | set(_BUILTIN_MODULES))
