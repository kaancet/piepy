"""Pluggable session-name parsing.

A session directory name encodes metadata. The in-house scheme is::

    <YYMMDD>_<animalid>_<paradigm>[_<opto>][_<area>]__<imaging>_<user>
    240810_KC150_detect__no_cam_KC          -> behaviour, no camera
    240312_KC147_detect_opto120_V1__1P_KC   -> opto 1.2, area V1, 1P imaging

The double underscore ``__`` separates the *task* part from the *rig/imaging* part; the
default parser keys off that delimiter and recognises tokens (opto120, 1P, ...) rather than
fixed positions, so it is robust to optional fields.

Parsing is **pluggable in code**: piepy ships :class:`DefaultSessionNameParser` for the
in-house scheme, and any callable ``str -> SessionName`` can replace it via
:func:`set_session_name_parser` for a different lab/convention.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import date as date_cls
from datetime import datetime
from typing import Protocol, runtime_checkable

from ..errors import MalformedSessionError


@dataclass(frozen=True)
class SessionName:
    """Structured result of parsing a session directory name.

    ``baredate``/``animalid``/``date`` are the required identity fields the data contract
    needs; scheme-specific fields (paradigm, imaging, opto_power, area, user, ...) live in
    ``extra`` so different conventions can carry different metadata.
    """

    sessiondir: str
    baredate: str
    animalid: str
    date: date_cls
    paradigm: str | None = None
    extra: dict = field(default_factory=dict)


@runtime_checkable
class SessionNameParser(Protocol):
    """Anything that turns a session directory name into a :class:`SessionName`."""

    def __call__(self, sessiondir: str) -> SessionName: ...


class DefaultSessionNameParser:
    """Parser for the in-house ``date_animal_paradigm[_opto][_area]__imaging_user`` scheme."""

    _DATE = re.compile(r"^\d{6}$")
    _PARADIGMS = {
        "detect": "detection",
        "detection": "detection",
        "discrim": "discrimination",
        "discrimination": "discrimination",
    }

    def __call__(self, sessiondir: str) -> SessionName:
        name = os.path.basename(str(sessiondir).rstrip("/\\"))
        head, sep, tail = name.partition("__")
        head_parts = [p for p in head.split("_") if p != ""]

        if len(head_parts) < 2 or not self._DATE.match(head_parts[0]):
            raise MalformedSessionError(
                f"Session name {name!r} must start with <YYMMDD>_<animalid>.",
                where=sessiondir,
                fix="Rename the directory to begin with a 6-digit date and an animal id "
                "(e.g. 240810_KC150_detect__no_cam_KC), or install a custom session-name "
                "parser via set_session_name_parser() for your lab's scheme.",
            )

        baredate, animalid = head_parts[0], head_parts[1]
        try:
            parsed_date = datetime.strptime(baredate, "%y%m%d").date()
        except ValueError:
            raise MalformedSessionError(
                f"Session date {baredate!r} is not a valid YYMMDD date.",
                where=sessiondir,
                fix="Correct the leading 6 digits of the session name to a real date (YYMMDD).",
            )

        task_tokens = head_parts[2:]  # paradigm, then optional opto/area
        rig_tokens = tail.split("_") if sep else []

        paradigm, opto_power, area, is_cno = self._parse_task_tokens(task_tokens)
        imaging = next((t for t in rig_tokens if t in ("1P", "2P")), None)
        user = rig_tokens[-1] if rig_tokens else None

        extra = {
            "user": user,
            "imaging": imaging,  # None for behaviour-only / no_cam
            "opto_power": opto_power,
            "area": area,
            "isCNO": is_cno,
        }
        return SessionName(
            sessiondir=name,
            baredate=baredate,
            animalid=animalid,
            date=parsed_date,
            paradigm=paradigm,
            extra=extra,
        )

    def _parse_task_tokens(self, tokens: list[str]):
        """Recognise paradigm / opto power / area / CNO from the task tokens by content."""
        paradigm = opto_power = area = None
        is_cno = False
        for tok in tokens:
            low = tok.lower()
            if low in self._PARADIGMS:
                paradigm = self._PARADIGMS[low]
            elif tok.startswith("opto") and tok[4:].isdigit():
                opto_power = int(tok[4:]) / 100
            elif tok.upper().endswith("CNO"):
                area = tok[:-3] or None
                is_cno = True
            elif paradigm is not None and area is None:
                # first unclassified token after the paradigm is the recording area
                area = tok
        return paradigm, opto_power, area, is_cno


# --------------------------------------------------------------------------- #
# Pluggable module-level default
# --------------------------------------------------------------------------- #
_active_parser: SessionNameParser = DefaultSessionNameParser()


def get_session_name_parser() -> SessionNameParser:
    """Return the currently-installed session-name parser."""
    return _active_parser


def set_session_name_parser(parser: SessionNameParser) -> None:
    """Install a custom session-name parser globally (the pluggable hook)."""
    global _active_parser
    _active_parser = parser


def parse_session_name(sessiondir: str) -> SessionName:
    """Parse a session directory name with the currently-installed parser."""
    return _active_parser(sessiondir)
