"""Session discovery: resolve a session name into a structured manifest.

:class:`SessionLocator` replaces the glob/os.walk guesswork in the old ``pathfinder.py`` with
explicit rules and actionable errors:

* the session must resolve in exactly one configured data root
  (``SessionNotFoundError`` for none, ``AmbiguousSessionError`` for more than one);
* runs follow the layout rule **"flat is fine only for a single run"**:
  one loose ``.stimlog``/``.riglog`` pair in the session dir = a single run; two or more loose
  pairs = ``MalformedSessionError``; multiple runs must live in ``run<NN>`` sub-directories.

The result is a :class:`SessionManifest` of structured :class:`RunArtifacts`. Each
``RunArtifacts`` is also the per-run "paths" object the rest of the pipeline consumes: it
exposes ``stimlog``/``riglog``/``prot``/``prefs``, the camera logs, ``opto_pattern``, and the
``save``/``data`` output paths -- the same attribute surface the old ``Paths`` had, so
``run.py``/``trial.py`` are untouched by the swap.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import natsort

from ..config import config as cfg
from ..errors import (
    AmbiguousSessionError,
    MalformedSessionError,
    SessionNotFoundError,
)
from .parser import SessionName, SessionNameParser, get_session_name_parser

# a run sub-directory starts with run<digits>, optionally followed by a descriptive suffix
# (e.g. "run00", "run01_171107_15_detectionTask_...").
_RUN_DIR = re.compile(r"^run\d+", re.IGNORECASE)

# data-root kinds that hold the behavioural stim/rig logs, searched for the session dir
_BEHAVIOUR_KINDS = ("presentation", "training")
# camera roots that may hold a matching session dir with per-run camlogs
_CAM_KINDS = ("onepcam", "eyecam", "facecam")
# auxiliary roots resolved per session (cameras + the opto silencing patterns)
_AUX_KINDS = (*_CAM_KINDS, "opto_pattern")


@dataclass
class RunArtifacts:
    """The files that make up a single run; also the per-run 'paths' object."""

    run_no: int  # 1-based, in run order
    run_dir: str  # directory holding this run's logs
    run_name: str | None  # stem of the .prot (or .stimlog) file
    stimlog: str | None
    riglog: str | None
    prefs: str | None
    prot: str | None
    save_dirs: list[str] = field(
        default_factory=list
    )  # analysis output dir(s) for this run
    # auxiliary artifacts (None when the session has no such data)
    onepcam: str | None = None
    onepcamlog: str | None = None
    eyecam: str | None = None
    eyecamlog: str | None = None
    facecam: str | None = None
    facecamlog: str | None = None
    opto_pattern: str | None = None

    # --- compatibility with the old Paths attribute surface consumed by run.py ---
    @property
    def save(self) -> list[str]:
        return self.save_dirs

    @property
    def data(self) -> list[str]:
        return [os.path.join(d, "runData.parquet") for d in self.save_dirs]


@dataclass
class SessionManifest:
    """Everything the locator discovered about one session."""

    sessiondir: str  # the directory NAME (e.g. 240810_KC150_detect__no_cam_KC)
    session_path: str  # full path to the resolved session directory
    root_kind: str  # which data-root kind it was found in (presentation/training)
    name: SessionName  # parsed metadata
    runs: list[RunArtifacts]

    @property
    def run_count(self) -> int:
        return len(self.runs)


def _files_with_suffix(directory: str, suffix: str) -> list[str]:
    """Full paths of files in ``directory`` ending with ``suffix`` (sorted)."""
    return natsort.natsorted(
        os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(suffix)
    )


def _one_or_none(directory: str, suffix: str) -> str | None:
    found = _files_with_suffix(directory, suffix)
    return found[0] if found else None


def _stem(path: str | None) -> str | None:
    return os.path.basename(path).rsplit(".", 1)[0] if path else None


def _run_dirs(session_path: str) -> list[str]:
    """The run directories of a session: run<NN> sub-dirs, or the session dir itself (flat)."""
    subdirs = natsort.natsorted(
        os.path.join(session_path, d)
        for d in os.listdir(session_path)
        if os.path.isdir(os.path.join(session_path, d)) and _RUN_DIR.match(d)
    )
    return subdirs if subdirs else [session_path]


def _roots_from_config(kinds: tuple[str, ...]) -> list[tuple[str, str]]:
    """(kind, root_dir) pairs for the given config kinds, skipping unconfigured ones."""
    pairs: list[tuple[str, str]] = []
    for kind in kinds:
        for root in cfg.paths.get(kind) or []:
            pairs.append((kind, root))
    return pairs


class SessionLocator:
    """Resolves a session directory name into a :class:`SessionManifest`.

    Args:
        behaviour_roots: ``(kind, dir)`` pairs to search for the session. Defaults to the
            ``presentation`` and ``training`` entries in the config.
        analysis_roots: directories under which analysis output is saved. Defaults to the
            config ``analysis`` entry.
        aux_roots: ``{kind: [dirs]}`` for the camera / opto-pattern roots. Defaults to the
            corresponding config entries; pass ``{}`` to disable auxiliary discovery (tests).
        name_parser: session-name parser. Defaults to the currently-installed global parser.
    """

    def __init__(
        self,
        *,
        behaviour_roots: list[tuple[str, str]] | None = None,
        analysis_roots: list[str] | None = None,
        aux_roots: dict[str, list[str]] | None = None,
        name_parser: SessionNameParser | None = None,
    ) -> None:
        self.behaviour_roots = (
            behaviour_roots
            if behaviour_roots is not None
            else _roots_from_config(_BEHAVIOUR_KINDS)
        )
        self.analysis_roots = (
            analysis_roots
            if analysis_roots is not None
            else (cfg.paths.get("analysis") or [])
        )
        self.aux_roots = (
            aux_roots
            if aux_roots is not None
            else {k: (cfg.paths.get(k) or []) for k in _AUX_KINDS}
        )
        self._name_parser = name_parser

    def locate(self, sessiondir: str) -> SessionManifest:
        """Find ``sessiondir`` and return its manifest, or raise a structured error."""
        root_kind, session_path = self._resolve(sessiondir)
        parser = self._name_parser or get_session_name_parser()
        runs = self._enumerate_runs(sessiondir, session_path)
        self._attach_aux(sessiondir, runs)
        return SessionManifest(
            sessiondir=sessiondir,
            session_path=session_path,
            root_kind=root_kind,
            name=parser(sessiondir),
            runs=runs,
        )

    def _resolve(self, sessiondir: str) -> tuple[str, str]:
        """Locate the session dir in exactly one behaviour root."""
        matches = [
            (kind, os.path.join(root, sessiondir))
            for kind, root in self.behaviour_roots
            if os.path.isdir(os.path.join(root, sessiondir))
        ]
        if not matches:
            searched = (
                ", ".join(root for _, root in self.behaviour_roots)
                or "(no roots configured)"
            )
            raise SessionNotFoundError(
                f"Session {sessiondir!r} was not found in any data root.",
                where=searched,
                fix="Check the session name for typos, or that its 'paths' (presentation / "
                "training) are set correctly in ~/.piepy/config.json.",
            )
        if len(matches) > 1:
            locations = ", ".join(path for _, path in matches)
            raise AmbiguousSessionError(
                f"Session {sessiondir!r} exists in more than one data root.",
                where=locations,
                fix="Keep the session in exactly one data root (e.g. remove the duplicate "
                "from either presentation or training).",
            )
        return matches[0]

    def _enumerate_runs(self, sessiondir: str, session_path: str) -> list[RunArtifacts]:
        """Apply the layout rule and collect each run's behavioural artifacts."""
        runs: list[RunArtifacts] = []
        for run_no, run_dir in enumerate(_run_dirs(session_path), start=1):
            stimlogs = _files_with_suffix(run_dir, ".stimlog")
            if len(stimlogs) == 0:
                raise MalformedSessionError(
                    f"No .stimlog file found in {os.path.basename(run_dir)!r}.",
                    where=run_dir,
                    fix="A run must contain a .stimlog and matching .riglog. Add them, or "
                    "remove the empty directory.",
                )
            if len(stimlogs) > 1:
                raise MalformedSessionError(
                    f"Found {len(stimlogs)} .stimlog files where a single run was expected.",
                    where=run_dir,
                    fix="Each run must contain exactly one .stimlog/.riglog pair. Put multiple "
                    "runs in separate run<NN> sub-directories (run00, run01, ...).",
                )
            prot = _one_or_none(run_dir, ".prot")
            runs.append(
                RunArtifacts(
                    run_no=run_no,
                    run_dir=run_dir,
                    run_name=_stem(prot or stimlogs[0]),
                    stimlog=stimlogs[0],
                    riglog=_one_or_none(run_dir, ".riglog"),
                    prefs=_one_or_none(run_dir, ".prefs"),
                    prot=prot,
                    save_dirs=self._save_dirs(sessiondir, run_dir, session_path),
                )
            )
        return runs

    def _save_dirs(self, sessiondir: str, run_dir: str, session_path: str) -> list[str]:
        """Analysis output dir(s) for a run, mirroring the run sub-dir layout under analysis."""
        rel = "" if run_dir == session_path else os.path.basename(run_dir)
        return [
            os.path.join(root, sessiondir, rel) if rel else os.path.join(root, sessiondir)
            for root in self.analysis_roots
        ]

    def _attach_aux(self, sessiondir: str, runs: list[RunArtifacts]) -> None:
        """Attach camera logs (per run) and the opto-pattern dir (session-level) if present."""
        for kind in _CAM_KINDS:
            cam_path = self._aux_session_path(kind, sessiondir)
            if cam_path is None:
                continue
            cam_run_dirs = _run_dirs(cam_path)
            for i, run in enumerate(runs):
                if i < len(cam_run_dirs):
                    setattr(run, kind, cam_run_dirs[i])
                    setattr(run, f"{kind}log", _one_or_none(cam_run_dirs[i], "camlog"))

        opto_path = self._aux_session_path("opto_pattern", sessiondir)
        for run in runs:
            run.opto_pattern = opto_path

    def _aux_session_path(self, kind: str, sessiondir: str) -> str | None:
        """First matching ``<aux_root>/<sessiondir>`` directory for an aux kind, or None."""
        for root in self.aux_roots.get(kind, []):
            candidate = os.path.join(root, sessiondir)
            if os.path.isdir(candidate):
                return candidate
        return None
