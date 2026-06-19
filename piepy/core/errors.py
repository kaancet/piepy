"""Structured, actionable errors for piepy.

Every user-facing failure should raise a :class:`PiepyError` (or subclass) that says three
things plainly: WHAT went wrong, WHERE, and HOW to fix it. The goal is that a user who hits
an error can remedy it without reading piepy's source.

Example::

    raise MalformedSessionError(
        "Found 2 .stimlog files directly in the session directory.",
        where=session_dir,
        fix="Each run must live in its own run<NN> sub-directory. Move each "
            ".stimlog/.riglog pair into its own run folder, or remove the extra one.",
    )

renders (via ``str(exc)``) as::

    piepy malformed-session error: Found 2 .stimlog files directly in the session directory.
      where:  /data/presentation/240810_KC150_detect__no_cam_KC
      fix:    Each run must live in its own run<NN> sub-directory. Move each .stimlog/.riglog
              pair into its own run folder, or remove the extra one.

Design:
* One base, ``PiepyError(Exception)`` -- so it IS caught by ``except Exception`` (the existing
  exceptions in ``exceptions.py`` subclass ``BaseException``, which is a bug to be migrated).
* A small category taxonomy (config / pathfinding / parsing / schema), each with a short tag.
* "Smart" subclasses (e.g. :class:`MissingConfigKeyError`) pre-fill the fix text so call
  sites stay short and messages stay consistent.
"""

from __future__ import annotations

import textwrap

# width to wrap the message body at; continuation lines align under the field value
_WRAP = 92
_LABEL_W = 7
_CONT_INDENT = " " * (2 + _LABEL_W)


class PiepyError(Exception):
    """Base for all piepy errors; carries structured remediation context.

    Args:
        problem: one-line statement of what went wrong.
        where: the file / directory / session / identifier the problem concerns.
        fix: a concrete action the user can take to remedy it.
        hint: optional extra pointer (a likely cause, a doc reference).
    """

    #: short category tag shown in the header line; override in subclasses
    category: str = "error"

    def __init__(
        self,
        problem: str,
        *,
        where: object | None = None,
        fix: str | None = None,
        hint: str | None = None,
    ) -> None:
        self.problem = problem
        self.where = where
        self.fix = fix
        self.hint = hint
        super().__init__(self.render())

    def _field(self, label: str, value: object) -> str:
        body = textwrap.fill(
            str(value),
            width=_WRAP,
            initial_indent="",
            subsequent_indent=_CONT_INDENT,
        )
        return f"  {label:<{_LABEL_W}}{body}"

    def render(self) -> str:
        """The full multi-line message (used as the exception string)."""
        header = "piepy error" if self.category == "error" else f"piepy {self.category} error"
        lines = [f"{header}: {self.problem}"]
        if self.where is not None:
            lines.append(self._field("where:", self.where))
        if self.fix is not None:
            lines.append(self._field("fix:", self.fix))
        if self.hint is not None:
            lines.append(self._field("hint:", self.hint))
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
class ConfigError(PiepyError):
    category = "config"


class MissingConfigKeyError(ConfigError):
    """A required key is absent from ~/.piepy/config.json."""

    def __init__(self, key: str, *, config_path: str = "~/.piepy/config.json") -> None:
        super().__init__(
            f"Required config key {key!r} is missing.",
            where=config_path,
            fix=f'Add "{key}" under the "paths" section of your config, e.g. "{key}": ["/path/to/{key}"], then re-run.',
        )


# --------------------------------------------------------------------------- #
# Pathfinding / session discovery  (Phase 2's primary customers)
# --------------------------------------------------------------------------- #
class PathfindingError(PiepyError):
    category = "pathfinding"


class SessionNotFoundError(PathfindingError):
    """A session directory could not be located in any configured data root."""


class AmbiguousSessionError(PathfindingError):
    """The same session resolves in more than one data root (e.g. presentation + training)."""


class MalformedSessionError(PathfindingError):
    """A session's on-disk layout violates the expected structure."""


# --------------------------------------------------------------------------- #
# Parsing & data contract
# --------------------------------------------------------------------------- #
class ParsingError(PiepyError):
    category = "parsing"


class OptoPatternError(ParsingError):
    """An optogenetics silencing-pattern directory is missing/invalid, or its image ids are
    misnamed (don't match the logged ``opto_pattern`` values)."""


class SchemaError(PiepyError):
    category = "schema"


# --------------------------------------------------------------------------- #
# Placeholders migrated from the old core/exceptions.py (2026-06-13).
#
# These were flat BaseException subclasses; they now ride on PiepyError so they're caught by
# `except Exception` and gain structured rendering. They are intentionally thin for now -- as
# the refactor proceeds each will either be given proper problem/where/fix context at its
# raise site, or removed. Do not add logic here; rebuild them where they're raised.
# --------------------------------------------------------------------------- #
class WrongSessionTypeError(ParsingError):
    pass


class PrefProtMismatchError(ParsingError):
    pass


class NoRigReactionTimeError(ParsingError):
    pass


class StateMachineError(ParsingError):
    pass


class LogTypeMissingError(ParsingError):
    pass


class FrameLoggingError(ParsingError):
    pass


class ScreenPulseError(ParsingError):
    pass


class VstimLoggingError(ParsingError):
    pass


class PathSettingError(PathfindingError):
    pass
