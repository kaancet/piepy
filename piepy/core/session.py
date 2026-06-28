import time

import polars as pl

from .io import display
from .run import Run
from .utils import timeit

from .paths import SessionLocator
from .schema import attach_run_identity, concat_session_runs


class Session:
    # paradigm wiring -- a subclass sets ``run_cls`` to its Run; the rest is generic.
    run_cls = Run

    def __init__(self, sessiondir: str):
        """A base Session object, reads and aggregates the recorded data which can then be used in user specific
        analysis pipelines

        Args:
            sessiondir (str): directory of the session inside the presentation folder(e.g. 200619_KC033_wheel_KC)
        """
        start = time.time()
        self.sessiondir = sessiondir
        self.runs = []

        # resolve the session and its runs (raises a structured pathfinding error on failure)
        self.manifest = SessionLocator().locate(self.sessiondir)
        self.run_count = self.manifest.run_count

        self.init_session_runs()
        display(f"Done! t={(time.time() - start):.2f} s")

    @property
    def viz(self):
        """Plotting bound to this session: ``session.viz.psychometric(...)`` (pools its runs)."""
        from piepy.viz import Viz

        return Viz(self)

    def init_session_runs(self) -> None:
        """Build, parse (or load), and collect every run in the session.

        Generic across paradigms: the run type comes from ``run_cls`` (and its
        ``state_transitions`` / ``trial_handler_cls`` / ``rundata_cls``). A paradigm customizes
        parsing through hooks on its Run (``repair_rawdata`` / ``augment_data`` /
        ``compute_stats``), not by re-implementing this loop.
        """
        for i, run_paths in enumerate(self.manifest.runs, start=1):
            run = self.run_cls(run_paths, run_no=i)
            run.set_meta()
            run.get_rawdata()
            self.runs.append(run)

    def analyze(self, paradigm: str | None = None, load_flag: bool = False, save_mat: bool = False) -> pl.DataFrame:
        """The analysis-ready trial table for this session

        This is what users (and the Hub) call. ``concatenate_runs`` is the structural step (stack
        runs on one clock)

        Args:
            paradigm (str | None, optional): _description_. Defaults to None.
            load_flag (bool, optional):  flag to either load previously parsed data or to parse it again. Defaults to False
            save_mat (bool, optional):   flag to make the parser also output a .mat file to be used in MATLAB scripts. Defaults to False

        Returns:
            pl.DataFrame: Concatenated session data
        """
        for r in self.runs:
            if r.is_run_saved() and load_flag:
                display(f"Loading from {r.paths.save}")
                r.load_run()
            else:
                r.analyze_run()
                r.data.add_metadata_columns(r.meta)
                r.save_run(save_mat)

        return self.concatenate_runs(paradigm)

    def concatenate_runs(self, paradigm: str | None = None) -> pl.DataFrame:
        """Concatenate this session's runs into one trial table on a session-wide clock.

        Additive: this does NOT modify the per-run data and writes nothing to disk. The
        returned DataFrame carries the canonical identity columns (``session_uid``,
        ``run_uid``, ``run_no``, ``paradigm``, ...), a session clock (``run_time_offset``
        plus ``*_session`` copies of every absolute-time column, originals untouched), and
        a cumulative ``session_trial_no``.

        Args:
            paradigm: paradigm label stamped on every row. Falls back to ``self.paradigm``
                if that attribute exists, otherwise ``None``.

        Returns:
            pl.DataFrame: the concatenated session trial table (empty if no runs have data).
        """
        if not self.runs:
            raise ValueError("No runs to concatenate; call init_session_runs() first.")

        paradigm = paradigm if paradigm is not None else getattr(self, "paradigm", None)

        frames = []
        for run_no, run in enumerate(self.runs, start=1):
            data = run.data.data if run.data is not None else None
            if data is None:
                continue
            meta = run.meta or {}
            frames.append(
                attach_run_identity(
                    data,
                    sessiondir=self.sessiondir,
                    run_no=run_no,
                    run_name=meta.get("run_name", f"run{run_no}"),
                    paradigm=paradigm,
                )
            )
        return concat_session_runs(frames)

    @timeit("Saving...")
    def save_session(self) -> None:
        """Saves the session data, meta and stats"""
        for run in self.runs:
            run.save_run(self.save_mat)

    @timeit("Loading...")
    def load_session(self) -> None:
        """Helper method to loop through the runs and load data and stats"""
        for run in self.runs:
            run.load_run()

    ####
    # DATABASE RELATED, NOT USED AT THE MOMENT
    ###

    # def save_to_db(self, db_dict: dict) -> None:
    #     """Checks if an entry for the session already exists and saves/updates accordingly"""
    #     if not self.db_interface.exists({"sessionId": self.meta.session_id}, "sessions"):
    #         self.db_interface.add_entry(db_dict, "sessions")
    #         self.db_interface.update_entry(
    #             {"id": self.meta.animalid},
    #             {"nSessions": self.current_session_no},
    #             "animals",
    #         )
    #     else:
    #         self.db_interface.update_entry(
    #             {"sessionId": self.meta.session_id}, db_dict, "sessions"
    #         )
    #         display(
    #             f"Session with id {self.meta.session_id} is already in database, updated the entry"
    #         )

    # def get_latest_trial_count(self):
    #     """Gets the last trial count from"""
    #     prev_trials = self.db_interface.get_entries({"id": self.meta.animalid}, "trials")
    #     try:
    #         return int(prev_trials["total_trial_no"].iloc[-1])
    #     except:
    #         return 0

    # def overall_session_no(self) -> int:
    #     """Gets the session number of the session"""
    #     mouse_entry = self.db_interface.get_entries(
    #         {"id": self.meta.animalid}, table_name="animals"
    #     )
    #     if len(mouse_entry):
    #         last_session_no = mouse_entry["nSessions"].iloc[0]
    #     else:
    #         display(f"No entry for mouse {self.meta.animalid} in animals table!")
    #         last_session_no = 0

    #     current_session_no = last_session_no + 1
    #     return current_session_no

    # def remove_session_db(self):
    #     """ """
    #     pass
