"""The generic, paradigm-agnostic session-gathering hub.

``Hub(paradigm)`` gathers a list of sessions into one cohort trial table. It is experiment-
agnostic: it resolves the paradigm's Session class (and optional ``enrich`` hook) through the
:mod:`piepy.core.registry`, runs each session, and stacks the results with
:func:`piepy.core.schema.align_and_concat`. There are no more per-experiment Hub subclasses --
experiment specifics live in the Session class + its registered enrich hook.

    hub = Hub("wheel_detection")
    hub.initialize(session_list, load_sessions=True)
    hub.data   # the cohort trial table
"""

from __future__ import annotations

import hashlib
import multiprocessing
import os

import natsort
import polars as pl
from datetime import datetime as dt

from .config import config as cfg
from .io import display
from .registry import get_paradigm
from .schema import align_and_concat
from tqdm.auto import tqdm

__all__ = ["Hub", "generate_unique_session_id"]


def generate_unique_session_id(
    baredate: str, animalid: str, *args, digit_len: int = 7
) -> int:
    """A deterministic legacy session id from date+animal (used by the detection plotters).

    NOTE: this assumes baredate+animalid is unique. For the canonical, collision-free id use
    ``piepy.core.schema.session_uid`` (hash of the full sessiondir).
    """
    combined = "|".join(sorted([baredate, animalid, *args]))
    return int(hashlib.sha256(combined.encode("utf-8")).hexdigest(), 16) % 10**digit_len


def _analyze_one(args: tuple) -> tuple[pl.DataFrame, str | None]:
    """Worker: analyze one session into its cohort-ready frame.

    Returns ``(frame, None)`` on success or ``(empty_frame, error_string)`` on failure.
    Takes ``(paradigm, load_flag, sessiondir)`` -- only picklable strings/flags cross the
    process boundary.
    """
    paradigm, load_flag, sessiondir = args
    name = os.path.basename(str(sessiondir).rstrip("/\\"))

    # Mute per-session output in workers so it doesn't flood the caller
    cfg.verbose = False
    try:
        spec = get_paradigm(paradigm)
        session = spec.session_cls(name)

        return session.analyze(load_flag=load_flag), None

    except Exception as exc:  # noqa: BLE001 - one bad session shouldn't sink the gather
        return pl.DataFrame(), f"{name} : {type(exc).__name__} — {exc}"


def _combine_session_data(frames: list[pl.DataFrame]) -> pl.DataFrame:
    """Align + stack per-session frames into the cohort table, sorted with a cumulative count."""
    data = align_and_concat(frames)
    if data.is_empty():
        return data
    sort_cols = [c for c in ("date", "animalid", "run_no") if c in data.columns]
    if sort_cols:
        data = data.sort(sort_cols)
    data = data.with_columns(
        pl.int_range(1, data.height + 1, dtype=pl.Int64).alias("total_trial_no")
    )
    return data.select(
        ["total_trial_no", *(c for c in data.columns if c != "total_trial_no")]
    )


class Hub:
    """Gathers many sessions of one paradigm into a single cohort trial table."""

    def __init__(self, paradigm: str) -> None:
        """
        Args:
            paradigm: a registered paradigm, e.g. ``"detection"`` or ``"discrimination"``.
        """
        self.paradigm = paradigm
        self.spec = get_paradigm(paradigm)  # validates + lazily loads builtins
        self.data: pl.DataFrame | None = None
        self.load_flag = False
        display(f"Hub set to paradigm {paradigm!r}", color="cyan")

    @property
    def viz(self):
        """Plotting bound to this cohort: ``hub.viz.psychometric(...)``.

        Cohort scope -> defaults to subject-averaging over ``animalid`` (override per call, e.g.
        ``hub.viz.psychometric(average_over="mouse")``).
        """
        from piepy.viz import Viz

        return Viz(self, subject="animalid")

    def initialize(self, data: pl.DataFrame | list, load_sessions: bool = False) -> None:
        """Initialize from a previously-gathered DataFrame, or gather from a session list."""
        if isinstance(data, pl.DataFrame):
            self.data = data.filter(pl.col("paradigm") == self.paradigm)
            if self.data.is_empty():
                display(
                    f">>> WARNING <<< No trials match paradigm {self.paradigm!r}; data is empty!",
                    color="red",
                )
            id_col = (
                "session_path" if "session_path" in self.data.columns else "session_uid"
            )
            self.session_list = self.data[id_col].unique(maintain_order=True).to_list()
        else:
            self.session_list = natsort.natsorted(data)
            self.gather_sessions(self.session_list, load_sessions=load_sessions)

    def gather_sessions(
        self, session_list: list, load_sessions: bool = False
    ) -> pl.DataFrame:
        """Analyze each session in parallel and stack into the cohort table."""
        self.load_flag = load_sessions

        work = [(self.paradigm, self.load_flag, s) for s in session_list]

        cores = cfg.multiprocess.get("cores", 1)
        use_mp = cfg.multiprocess.get("enable", False) and cores > 1 and len(work) > 1

        if use_mp:
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(processes=cores) as pool:
                results = list(
                    tqdm(
                        pool.imap(_analyze_one, work),
                        total=len(work),
                        desc="Gathering sessions",
                        unit="session",
                    )
                )
        else:
            results = [
                _analyze_one(w)
                for w in tqdm(work, desc="Gathering sessions", unit="session")
            ]

        frames = []
        failures = []
        for frame, err in results:
            if err is not None:
                failures.append(err)
            elif not frame.is_empty():
                frames.append(frame)

        if failures:
            display(
                f"\n>> WARNING << {len(failures)} session(s) failed during gather:",
                color="yellow",
            )
            for f in failures:
                display(f"  {f}", color="yellow")

        self.data = _combine_session_data(frames)
        return self.data

    def _one_session(self, sessiondir: str) -> pl.DataFrame:
        """Analyze a single session into its cohort-ready table (empty frame on failure).

        Kept for direct single-process use; the parallel path uses the module-level
        :func:`_analyze_one` so nothing on ``self`` is pickled to the workers.
        """
        frame, err = _analyze_one((self.paradigm, self.load_flag, sessiondir))
        if err is not None:
            display(f">> WARNING << {err}", color="yellow")
        return frame

    def save(self, saveloc: str | None = None) -> None:
        """Save the cohort data as a dated parquet."""
        date_first, date_last = self.data[0, "baredate"], self.data[-1, "baredate"]
        animals = ",".join(self.data["animalid"].unique().sort().to_list())
        savename = f"{date_first}-{date_last}_{animals}_{dt.strftime(dt.today(), '%y%m%d')}.parquet"
        if saveloc is None:
            saveloc = cfg.paths["analysis"][0]
        os.makedirs(saveloc, exist_ok=True)
        self.data.write_parquet(f"{saveloc}/{savename}")
        display(f"Saved at {saveloc}/{savename}", color="green")
