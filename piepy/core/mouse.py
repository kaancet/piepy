import os
import glob
import natsort
import numpy as np
import polars as pl
from tqdm import tqdm
from os.path import join as pjoin
from collections import namedtuple
from datetime import datetime as dt

from ..core.config import config as cfg
from .utils import timeit
from .io import display
from .paths import parse_session_name
from .schema import align_and_concat


def list_animal_sessions(animalid: str) -> pl.DataFrame:
    """Every presentation+training session dir for one animal, as a table.

    Columns: ``animalid, sessiondir, date, exp_type, fullpath, paradigm``. The
    ``paradigm`` label comes from the canonical session-name parser (``None`` when a
    name doesn't parse), so it matches the registry names (``wheel_detection`` etc.).
    Shared by :class:`Mouse` and the ``hub``/``training-report`` CLI commands.
    """
    presentation = cfg.paths["presentation"][0]
    training = cfg.paths["training"][0]
    experiment_sessions = glob.glob(f"{presentation}/*{animalid}*/")
    training_sessions = glob.glob(f"{training}/*{animalid}*__no_cam_*/")
    all_sessions = natsort.natsorted(experiment_sessions + training_sessions)
    all_sessions = [s for s in all_sessions if not s.endswith(f"_skip{os.sep}")]

    types, dates, sessions, paradigms = [], [], [], []
    for sesh in all_sessions:
        s = sesh.split(os.sep)[-2]
        dates.append(dt.strptime(s.split("_")[0], "%y%m%d"))
        if "training" in sesh:
            types.append("training")
        elif "1P" in s:
            types.append("opto" if "opto" in s else "1P")
        elif "2P" in s:
            types.append("2P")
        elif "opto" in sesh:
            types.append("opto")
        else:
            types.append(None)
        sessions.append(s)
        try:
            paradigms.append(parse_session_name(s).paradigm)
        except Exception:  # noqa: BLE001 - an unparseable name just has no paradigm
            paradigms.append(None)

    return pl.DataFrame(
        {
            "animalid": [animalid] * len(all_sessions),
            "sessiondir": sessions,
            "date": dates,
            "exp_type": types,
            "fullpath": all_sessions,
            "paradigm": paradigms,
        }
    ).sort("date")


class MouseMeta:
    def __init__(self) -> None:
        pass


class MouseData:
    def __init__(self) -> None:
        self.summary_data = None
        self.cumul_data = None

    def set_paradigm(self, paradigm: str) -> None:
        """ """
        self.paradigm = paradigm

    def append(self, cumul_data_list: list, summary_data_list: list) -> None:
        """Appends new data to the existing data"""
        assert len(cumul_data_list) == len(
            summary_data_list["date"]
        ), "Cumulative and summary data has to be the same length!!"

        # summary: append the new summary rows onto whatever we already have
        tmp = pl.DataFrame(summary_data_list)
        self.summary_data = align_and_concat([self.summary_data, tmp])

        # cumulative: drop the derived running counter (re-added below), then align + stack
        # all incoming session frames in one schema-aligned pass
        cumul_existing = self.cumul_data
        if cumul_existing is not None and "cumul_trial_no" in cumul_existing.columns:
            cumul_existing = cumul_existing.drop("cumul_trial_no")
        self.cumul_data = align_and_concat([cumul_existing, *cumul_data_list])

        # sort both by date
        self.cumul_data = self.cumul_data.sort(["date", "trial_no"])
        self.summary_data = self.summary_data.sort("date")
        self.cumul_data = self.cumul_data.with_row_index("cumul_trial_no", offset=1)

    def save(self, save_path: str) -> None:
        """Saves the data in the given location"""
        if self.cumul_data is not None:
            cumul_save_name = pjoin(
                save_path, f"{self.paradigm}BehaviorData.parquet"
            ).replace("\\", "/")
            summary_save_name = pjoin(
                save_path, f"{self.paradigm}BehaviorDataSummary.csv"
            ).replace("\\", "/")

            # cast sf and tf to str
            summary_save_data = self.summary_data.with_columns(
                [
                    (
                        "[" + pl.col("sf").cast(pl.List(pl.Utf8)).list.join(", ") + "]"
                    ).alias("sf"),
                    (
                        "[" + pl.col("tf").cast(pl.List(pl.Utf8)).list.join(", ") + "]"
                    ).alias("tf"),
                ]
            )

            summary_save_data.write_csv(summary_save_name)
            # cumul data is lazy df, so sink it instead of directly writing it
            self.cumul_data.write_parquet(cumul_save_name, compression="lz4")
        else:
            display("No data to save...")

    def load(self, load_path: str) -> None:
        """Loads the data, both the cumulative and summary from directory"""
        # this loads the most recent found data
        # load_and_add and last_saved will enter here
        self.cumul_data = pl.read_parquet(
            pjoin(load_path, f"{self.paradigm}BehaviorData.parquet")
        )

        summary_data = pl.read_csv(
            pjoin(load_path, f"{self.paradigm}BehaviorDataSummary.csv")
        )
        # sf and tf needs to be converted back to lists
        self.summary_data = summary_data.with_columns(
            [
                pl.col("sf")
                .str.replace_all("[", "", literal=True)
                .str.replace_all("]", "", literal=True)
                .str.split(",")
                .apply(lambda x: [float(i) for i in x])
                .alias("sf"),
                pl.col("tf")
                .str.replace_all("[", "", literal=True)
                .str.replace_all("]", "", literal=True)
                .str.split(",")
                .apply(lambda x: [float(i) for i in x])
                .alias("tf"),
                pl.col("dt_date").str.to_date(),
            ]
        )


class Mouse:
    """Analyzes the training progression of animals through multiple sessions
    animalid:  id of the animal to be analyzed(e.g. KC033)
    paradigm:  task paradigm, e.g. detection, multiSense, etc...
    """

    def __init__(
        self, animalid: str, paradigm: str = None, dateinterval: list | str = None
    ) -> None:
        self.animalid = animalid
        self.init_data_paths()
        self.data = MouseData()
        self.all_sessions = self.get_sessions()
        # set paradigm also filters the sessions list to only desired paradigm sessions
        self.set_paradigm(paradigm)
        if dateinterval is not None:
            self.filter_dates(dateinterval)

        self.load_modes = ["no_load", "reanalyze", "load_and_add", "last_saved"]

    def set_paradigm(self, paradigm: str) -> None:
        """Sets the paradigm of which the read sessions will be analysed in, eg. detection task"""
        self.paradigm = paradigm
        if self.paradigm is not None:
            # set session parser
            self.session_parser = self.get_session_class(self.paradigm)
            # filter sessions list
            self.session_list = self.all_sessions.filter(
                pl.col("paradigm") == self.paradigm
            )
            # set paradigm in the data class
            self.data.set_paradigm(self.paradigm)
            display(f"Set the data analysis paradigm to {self.paradigm}", color="cyan")

    def filter_dates(self, date_interval: list) -> None:
        """Filters the"""
        # dateinterval is a list of two date strings e.g. ['200127','200131']
        if isinstance(date_interval, str):
            date_interval = [date_interval]
            # add current day as end date
            date_interval.append(dt.today().strftime("%y%m%d"))
        elif isinstance(date_interval, list):
            assert (
                len(date_interval) <= 2
            ), f"You need to provide a single start(1) or start and end dates(2), got {len(date_interval)} dates"
            if len(date_interval) == 1:
                date_interval.append(dt.today().strftime("%y%m%d"))
        else:
            raise ValueError("Got an unexpected type for dates!")

        startdate = dt.strptime(date_interval[0], "%y%m%d")
        enddate = dt.strptime(date_interval[1], "%y%m%d")

        self.session_list = self.session_list.filter(
            (pl.col("date") >= startdate) & (pl.col("date") <= enddate)
        )

        display("Retreiving between {0} - {1}".format(startdate, enddate))

    def init_data_paths(self) -> None:
        """Initializes data paths"""

        paths = cfg.paths
        # excepy for analysis, take the first pathway present in config
        # only take the necessary things from config
        config = {
            n: p
            for n, p in paths.items()
            if n in ["analysis", "presentation", "training", "colors", "database"]
        }
        tmp_dict = {name: path[0] for name, path in config.items()}
        tmp_paths = namedtuple("Paths", list(tmp_dict.keys()))
        self.paths = tmp_paths(**tmp_dict)

    def get_sessions(self) -> pl.DataFrame:
        """Session-list table for this animal (see :func:`list_animal_sessions`)."""
        return list_animal_sessions(self.animalid)

    @timeit("Gathering behavior data...")
    def gather_data(self, load_type: str = None) -> None:
        """Gathers the data from all the sessions in the session list"""
        if self.paradigm is None:
            display(
                "No paradigm set to analyze the data, do that first by using set_paradigm method",
                color="orange",
            )
            return None

        if load_type not in self.load_modes:
            raise ValueError(
                f"{load_type} is not a valid loading mode, try one of: {self.load_modes}"
            )

        if load_type is None:
            load_type = "last_saved"

        missing_sessions = self.get_unanalyzed_sessions(load_type)
        if len(missing_sessions) == len(self.session_list):
            # no_load and reanalyze will enter here for sure
            # load_and_add will enter here if only there is no data to load
            session_counter = 0
        else:
            # this loads the most recent found data
            # load_and_add and last_saved will enter here
            self.load()
            session_counter = self.data.summary_data[-1, "session_no"]
        summary_to_append: dict = {}  # column name -> list of per-session values
        cumul_to_append = []

        # analyze each missing session, resilient to individual session failures
        pbar = tqdm(missing_sessions)
        self.faulty_sessions = []
        for i, row in enumerate(missing_sessions.iter_rows()):
            sessiondir, exp_type = row[1], row[3]
            pbar.set_description(
                f"Analyzing {sessiondir} [{i + 1}/{len(missing_sessions)}]"
            )

            try:
                _single_session = self.session_parser(sessiondir)
                # analyze() parses each run then concatenates them onto one session clock
                # (the Session already knows its own paradigm, set at registration)
                session_data = _single_session.analyze(load_flag=(load_type != "no_load"))
            except Exception as exc:
                display(
                    f" >>> WARNING <<< Could not analyze {sessiondir}: {exc}",
                    color="yellow",
                )
                self.faulty_sessions.append(sessiondir)
                pbar.update()
                continue

            if session_data.is_empty():
                display(
                    f" >>> WARNING <<< No data for session {sessiondir}", color="yellow"
                )
                self.faulty_sessions.append(sessiondir)
                pbar.update()
                continue

            # session-level meta/stats come from the runs (first run for session-level fields)
            meta = _single_session.runs[0].meta or {}
            stats = _single_session.runs[0].stats or {}
            opts = meta.get("opts", {}) or {}
            rig = meta.get("rig")

            summary_temp = {
                "date": meta.get("baredate"),
                "dt_date": meta.get("date"),
                "blank_time": opts.get("openStimDuration"),
                "response_window": opts.get("closedStimDuration"),
                "level": int(meta["level"]) if meta.get("level") is not None else -1,
                "session_no": session_counter + 1,
                **stats,
                "task": opts.get("controller"),
                "sf": (
                    session_data["sf"].unique().drop_nulls().to_list()
                    if "sf" in session_data.columns
                    else []
                ),
                "tf": (
                    session_data["tf"].unique().drop_nulls().to_list()
                    if "tf" in session_data.columns
                    else []
                ),
                "rig": rig.get("name") if isinstance(rig, dict) else rig,
            }

            session_data = session_data.with_columns(
                pl.lit(session_counter + 1).alias("session_no"),
                pl.lit(exp_type).alias("session_type"),
            )
            cumul_to_append.append(session_data)
            for k, v in summary_temp.items():
                summary_to_append.setdefault(k, []).append(v)

            session_counter += 1
            pbar.update()

        if len(summary_to_append):
            self.data.append(cumul_to_append, summary_to_append)
            display("Appended new data!", color="cyan")

    def save(self) -> None:
        """Saves the behavior data"""
        latest_session = self.session_list[-1, "sessiondir"]

        savepath = pjoin(self.paths.analysis, latest_session).replace("\\", os.sep)

        self.data.save(savepath)
        display(f"{self.paradigm} behavior data saved in {savepath}", color="green")

        # deleting the old data
        if not self.data.summary_data.is_empty():
            if self.saved_dir is not None:
                # remove the old data
                if self.saved_dir != latest_session:
                    del_target = pjoin(self.paths.analysis, self.saved_dir)
                    display(f"Deleting the old data in {del_target}", color="red")
                    os.remove(pjoin(del_target, f"{self.paradigm}BehaviorData.parquet"))
                    os.remove(
                        pjoin(del_target, f"{self.paradigm}BehaviorDataSummary.csv")
                    )

    def load(self):
        """Loads"""
        load_path = pjoin(self.paths.analysis, self.saved_dir)
        self.data.load(load_path)

    def isSaved(self) -> bool:
        """Finds the session folder that has the saved behavior data"""
        cumul_data_saved_loc = glob.glob(
            f"{self.paths.analysis}/*{self.animalid}*/{self.paradigm}BehaviorData*.parquet"
        )
        summary_data_saved_loc = glob.glob(
            f"{self.paths.analysis}/*{self.animalid}*/{self.paradigm}BehaviorDataSummary.csv"
        )

        if len(cumul_data_saved_loc) > 1 and len(summary_data_saved_loc) > 1:
            display(
                f"There should be only single _trainingData.parquet (most recent one) found {summary_data_saved_loc}, using the last one...",
                color="yellow",
            )
            cumul_data_saved_loc = cumul_data_saved_loc[:1]
            summary_data_saved_loc = summary_data_saved_loc[:1]

        if len(cumul_data_saved_loc) == 1 and len(summary_data_saved_loc) == 1:
            # check if the location is same for both data, should be the case
            cumul_dir = cumul_data_saved_loc[0].split(os.sep)[-2]
            summary_dir = summary_data_saved_loc[0].split(os.sep)[-2]
            if cumul_dir == summary_dir:
                self.saved_dir = cumul_dir
                return True
            else:
                raise FileExistsError(
                    f"Location of the cumulative data {self.cumul_file_loc} is not the same with summary data {self.summary_file_loc}"
                )
        elif len(cumul_data_saved_loc) == 0 and len(summary_data_saved_loc) == 0:
            self.saved_dir = None
            return False
        else:
            raise RuntimeError(
                "!! This should not happen! Saving of behavior data is messed up !!"
            )

    def get_unanalyzed_sessions(self, load_type: str) -> pl.DataFrame:
        """Returns the list of sessions that have not been added to the behavior analysis"""
        is_saved = self.isSaved()
        if load_type == "load_and_add":
            if is_saved:
                display(f"Found behavior data at {self.saved_dir}", color="cyan")
                reverse_session_list = (
                    self.session_list.reverse()
                )  # this will have sessions listed from new to old for ease of search
                _until = np.where(
                    reverse_session_list["sessiondir"].to_numpy() == self.saved_dir
                )[0][0]
                missing_sessions = reverse_session_list[:_until]
                missing_sessions = (
                    missing_sessions.reverse()
                )  # reverse again to have sessions added from old to new(chronological order)
                display(
                    f"Adding {len(missing_sessions)} missing sessions to last analysis data",
                    color="cyan",
                )
            else:
                display("No behavior data present, creating new")
                missing_sessions = self.session_list

        elif load_type == "reanalyze" or load_type == "no_load":
            missing_sessions = self.session_list
        elif load_type == "last_saved":
            if not is_saved:
                # no saved file found!
                raise FileNotFoundError(
                    "Can't do load, there is no saved behavioral analysis files for cumulative (*.parquet) and summary (*.csv) data"
                )
            # returns an empty list, no new session will be analyzed
            missing_sessions = pl.DataFrame()
        return missing_sessions

    @staticmethod
    def get_session_class(session_type: str):
        """Return the Session class for a paradigm (e.g. 'wheel_detection') via the registry."""
        from .registry import get_session_class

        return get_session_class(session_type)
