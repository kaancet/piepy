"""Pluggable session-name parsing.

A session directory name encodes metadata. The in-house scheme is::

    <YYMMDD>_<animalid>_<paradigm>[_<opto>][_<area>]__<imaging>_<user>
    240810_KC150_detect__no_cam_KC          -> behaviour, no camera
    240312_KC147_detect_opto120_V1__1P_KC   -> opto 1.2, area V1, 1P imaging

The double underscore ``__`` separates the *task* part from the *rig/imaging* part; the
default parser keys off that delimiter and recognises tokens (opto120, 1P, ...) rather than
fixed positions, so it is robust to optional fields.

Parsing resolves in three layers: an explicit override installed via
:func:`set_session_name_parser` (any callable ``str -> SessionName``) wins; otherwise the
per-paradigm ``naming_template`` regexes registered from each paradigm's ``scheme.json``
(:class:`SchemeNameParser`) are tried; finally the in-house :class:`DefaultSessionNameParser`.
A new task therefore declares its session-name scheme as data (a regex in ``scheme.json``),
not code.
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


class SchemeNameParser:
    """Parser driven by per-paradigm ``naming_template`` regexes (from each scheme.json).

    Each registered template is a regex with named groups and must capture ``date`` (YYMMDD)
    and ``animalid``. The paradigm comes from the scheme's owner (the folder it was loaded
    from), so the template need not capture it; any other named group lands in ``extra``.
    """

    def __init__(self) -> None:
        self._schemes: dict[str, re.Pattern] = {}

    def register(self, paradigm: str, naming_template: str) -> None:
        rx = re.compile(naming_template)
        missing = {"date", "animalid"} - set(rx.groupindex)
        if missing:
            raise ValueError(
                f"naming_template for {paradigm!r} must capture named group(s) "
                f"{sorted(missing)}, e.g. (?P<date>\\d{{6}})_(?P<animalid>[^_]+)_..."
            )
        self._schemes[paradigm] = rx

    def __call__(self, sessiondir: str) -> SessionName:
        name = os.path.basename(str(sessiondir).rstrip("/\\"))
        matches = [(p, m) for p, rx in self._schemes.items() if (m := rx.fullmatch(name))]
        if not matches:
            raise MalformedSessionError(
                f"No registered paradigm scheme matched {name!r}.",
                where=sessiondir,
                fix="Add/repair naming_template in the paradigm's scheme.json, or rely on the default scheme.",
            )
        if len(matches) > 1:
            raise MalformedSessionError(
                f"Session name {name!r} matched multiple paradigm schemes: {sorted(p for p, _ in matches)}.",
                where=sessiondir,
                fix="Make the naming_template regexes mutually exclusive (e.g. pin the paradigm token literally).",
            )
        paradigm, m = matches[0]
        gd = m.groupdict()
        baredate = gd["date"]
        try:
            parsed_date = datetime.strptime(baredate, "%y%m%d").date()
        except ValueError:
            raise MalformedSessionError(
                f"Session date {baredate!r} is not a valid YYMMDD date.",
                where=sessiondir,
                fix="Fix the 'date' capture in the scheme (must be 6-digit YYMMDD).",
            ) from None
        extra = {k: v for k, v in gd.items() if k not in ("date", "animalid")}
        return SessionName(
            sessiondir=name,
            baredate=baredate,
            animalid=gd["animalid"],
            date=parsed_date,
            paradigm=paradigm,
            extra=extra,
        )


# --------------------------------------------------------------------------- #
# Pluggable module-level parsing
# --------------------------------------------------------------------------- #
_scheme_parser = SchemeNameParser()  # per-paradigm naming_template regexes
_default_parser = DefaultSessionNameParser()  # in-house fallback scheme
_override_parser: SessionNameParser | None = None  # full override hook


def register_scheme(paradigm: str, naming_template: str) -> None:
    """Register a paradigm's ``naming_template`` regex (called from scheme.json loading)."""
    _scheme_parser.register(paradigm, naming_template)


def get_session_name_parser() -> SessionNameParser:
    """Return the active parser (the override if installed, else the default scheme)."""
    return _override_parser or _default_parser


def set_session_name_parser(parser: SessionNameParser) -> None:
    """Install a custom session-name parser globally, fully overriding scheme + default."""
    global _override_parser
    _override_parser = parser


def parse_session_name(sessiondir: str) -> SessionName:
    """Parse a session directory name.

    Order: an explicit override (:func:`set_session_name_parser`) wins; otherwise try the
    registered per-paradigm ``naming_template`` schemes, then fall back to the in-house default.
    """
    if _override_parser is not None:
        return _override_parser(sessiondir)
    try:
        return _scheme_parser(sessiondir)
    except MalformedSessionError:
        return _default_parser(sessiondir)
