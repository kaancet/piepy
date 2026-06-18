"""The generic, paradigm-agnostic session-gathering hub.

``Hub(paradigm)`` gathers a list of sessions into one cohort trial table. It is experiment-
agnostic: it resolves the paradigm's Session class (and optional ``enrich`` hook) through the
:mod:`piepy.core.registry`, runs each session, and stacks the results with
:func:`piepy.core.schema.align_and_concat`. There are no more per-experiment Hub subclasses --
experiment specifics live in the Session class + its registered enrich hook.

    hub = Hub("detection")
    hub.initialize(session_list, load_sessions=True)
    hub.data   # the cohort trial table
"""

from __future__ import annotations

import hashlib
import os

import natsort
import polars as pl
from datetime import datetime as dt
from multiprocessing import Pool, set_start_method

from .config import config as cfg
from .io import display
from .paths import parse_session_name
from .registry import get_paradigm
from .schema import align_and_concat

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


def _analyze_one(args: tuple) -> pl.DataFrame:
    """Worker: analyze one session into its cohort-ready frame (empty frame on failure).

    Takes ``(paradigm, load_flag, sessiondir)`` -- only picklable strings/flags cross the
    process boundary. Each worker resolves the paradigm spec itself via :func:`get_paradigm`
    (re-importing/discovering as needed under ``spawn``), so a dynamically-synthesized Session
    class never needs to be pickled to the worker.
    """
    paradigm, load_flag, sessiondir = args
    name = os.path.basename(str(sessiondir).rstrip("/\\"))
    try:
        spec = get_paradigm(paradigm)
        session = spec.session_cls(name, load_flag=load_flag)
    except Exception as exc:  # noqa: BLE001 - one bad session shouldn't sink the gather
        print(f" >> WARNING << {name} not analyzed ({exc}); skipping...", flush=True)
        return pl.DataFrame()
    if spec.enrich is not None:
        return spec.enrich(session)
    return session.concatenate_runs(paradigm)


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
            self.session_list = natsort.natsorted(self._filter_session_list(data))
            self.gather_sessions(self.session_list, load_sessions=load_sessions)

    def _filter_session_list(self, session_list: list) -> list:
        """Keep sessions whose parsed paradigm matches this hub's (skipping unparseable names)."""
        kept = []
        for s in session_list:
            try:
                if parse_session_name(s).paradigm == self.paradigm:
                    kept.append(s)
            except (
                Exception
            ):  # noqa: BLE001 - an unparseable name just isn't this paradigm
                continue
        return kept

    def gather_sessions(
        self, session_list: list, load_sessions: bool = False
    ) -> pl.DataFrame:
        """Analyze each session in parallel and stack into the cohort table."""
        self.load_flag = load_sessions
        try:
            set_start_method("spawn")
        except RuntimeError:
            pass
        with Pool(processes=cfg.multiprocess["cores"]) as pool:
            frames = pool.map(
                _analyze_one,
                [(self.paradigm, self.load_flag, s) for s in session_list],
            )
        self.data = _combine_session_data(frames)
        return self.data

    def _one_session(self, sessiondir: str) -> pl.DataFrame:
        """Analyze a single session into its cohort-ready table (empty frame on failure).

        Kept for direct single-process use; the parallel path uses the module-level
        :func:`_analyze_one` so nothing on ``self`` is pickled to the workers.
        """
        return _analyze_one((self.paradigm, self.load_flag, sessiondir))

    def save(self, saveloc: str | None = None) -> None:
        """Save the cohort data as a dated parquet."""
        date_first, date_last = self.data[0, "baredate"], self.data[-1, "baredate"]
        animals = ",".join(self.data["animalid"].unique().sort().to_list())
        savename = f"{date_first}-{date_last}_{animals}_{dt.strftime(dt.today(), '%y%m%d')}.parquet"
        if saveloc is None:
            saveloc = self.data[-1, "session_path"].replace("presentation", "analysis")
        self.data.write_parquet(f"{saveloc}/{savename}")
        display(f"Saved at {saveloc}", color="green")
