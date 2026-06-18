import os
import re
import sys
from datetime import datetime as dt
from os.path import exists as pexists
from os.path import join as pjoin

import numpy as np
import patito as pt
import polars as pl
import scipy.io as sio
from tqdm import tqdm

from .config import config
from .errors import StateMachineError, WrongSessionTypeError
from .io import display, load_json_dict, save_dict_json
from .parsers import (
    parse_labcams_log,
    parse_preference,
    parse_protocol,
    parse_stimpy_log,
    parse_stimpygithub_log,
)
from .log_repair_functions import (
    add_total_iStim,
    extract_trial_count,
    stitch_logs,
    extrapolate_time,
)
from .paths import RunArtifacts as Paths
from .paths import parse_session_name
from .trial import TrialHandler

STATE_TRANSITION_KEYS = {}


class RunMeta:
    def __init__(self) -> None:
        pass

    @classmethod
    def get_meta(cls, path: Paths) -> dict:
        """A wrapper function that calls

        Args:
            path: Paths object that has the

        Returns:
            dict: metadata as a dict
        """
        # derive the session dir name by walking up past any run<NN> component of the prot path
        _temp = path.prot.split(os.sep)
        _cntr = -1
        while True:
            sessiondir = _temp[_cntr]
            if not sessiondir.startswith("run"):
                break
            _cntr -= 1

        # identity fields come from the pluggable session-name parser (not positional split)
        parsed = parse_session_name(sessiondir)
        _general = {
            "sessiondir": sessiondir,
            "baredate": parsed.baredate,
            "date": parsed.date,
            "nicedate": dt.strftime(parsed.date, "%d %b %y"),
            "animalid": parsed.animalid,
            "user_id": parsed.extra.get("user"),
            "imaging_mode": parsed.extra.get("imaging"),
        }

        _prot = cls.get_prot(path.prot)
        _pref = cls.get_pref(path.prefs)
        return {**_general, **_prot, **_pref}

    @staticmethod
    def get_prot(prot_path: str) -> dict:
        """Gets the options and parameters from prot file

        Args:
            prot_path: The path to protocol file

        Returns:
            dict: protocol file as a dict
        """
        prot_dict = {}
        prot_dict["run_name"] = prot_path.split(os.sep)[-1].split(".")[0]
        opts, params, _ = parse_protocol(prot_path)

        # get the level if exists
        if prot_path.find("level") != -1:
            match = re.search(r"level(\d+)", prot_path)
            if match:
                lvl = int(match.group(1))
        else:
            lvl = None  # should be experiments
        prot_dict["level"] = lvl

        # get the run start time
        os_stat = os.stat(prot_path)
        if sys.platform == "darwin":
            create_epoch = os_stat.st_birthtime
        elif sys.platform == "win32":
            create_epoch = os_stat.st_ctime
        prot_dict["run_start_time"] = dt.fromtimestamp(create_epoch).strftime("%H%M")
        prot_dict["opts"] = opts
        prot_dict["params"] = params
        return prot_dict

    @staticmethod
    def get_pref(pref_path: str) -> dict:
        """Returns the parsed preference file

        Args:
            pref_path: The path to preference file

        Returns
            dict: preferenc file as a dict
        """
        return parse_preference(pref_path)


class RunData:
    def __init__(self, data: pl.DataFrame = None) -> None:
        self.set_data(data)

    def set_data(self, data: pl.DataFrame) -> None:
        """Sets the data of the

        Args:
            data: The dataframe that has the trials
        """
        self.data = data

    def add_metadata_columns(self, metadata: dict) -> None:
        """Adds some metadata to run data for ease of manipulation

        Args:
            metadata: Adds metadata columns to the dataframe
        """
        # animal id and baredate as str
        self.data = self.data.with_columns(
            [
                pl.lit(metadata["animalid"]).alias("animalid"),
                pl.lit(metadata["baredate"]).alias("baredate"),
            ]
        )

        # datetime date
        self.data = self.data.with_columns(
            pl.col("baredate").str.strptime(pl.Date, format="%y%m%d").cast(pl.Date).alias("date")
        )

    def save_data(self, save_path: str, save_mat: bool = False) -> None:
        """Saves the run data as .parquet (and .mat file if desired)

        Args:
            save_path: The path to save the dataframe as a parquet
            save_mat: Flag to save the dataframe as a .mat file
        """
        data_save_path = pjoin(save_path, "runData.parquet")
        self.data.write_parquet(data_save_path)
        if save_mat:
            self.save_as_mat(save_path)
            display(f"Saved .mat file at {save_path}", color="green")

    def load_data(self, load_path: str) -> pl.DataFrame:
        """Loads the data from J:/analysis/<exp_folder> as a pandas data frame

        Args:
            load_path: The path to load the data from
        """
        # data = pd.read_csv(self.paths.data)
        data = pl.read_parquet(load_path)
        self.set_data(data)

    def save_as_mat(self, save_path: str) -> None:
        """Helper method to convert the data into a .mat file

        Args:
            save_path: The path to save the .mat file to
        """
        datafile = pjoin(save_path, "sessionData.mat")

        save_dict = {name: col.values for name, col in self.data.stim_data.items()}
        sio.savemat(datafile, save_dict)
        display(f"Saved .mat file at {datafile}")


class Run:
    rundata_cls = RunData
    trial_handler_cls = TrialHandler
    state_transitions: dict = {}

    def __init__(self, paths: Paths) -> None:
        self.meta = None
        self.stats = None
        self.paths = paths
        self.comments = None
        # initialize the logger(only log at one analysis location, currently arbitrary)
        # self.logger = Logger(log_path=self.paths.save[0])
        self.data = self.rundata_cls()
        self.trial_handler = self.trial_handler_cls()

    def __repr__(self):
        _controller = ""
        if self.meta is not None:
            _controller = self.meta["opts"]["controller"]
        _dat = ""
        if self.data.data is not None and self.stats is not None:
            _dat = f" - {len(self.data.data)} trials"
        return f"{_controller}{_dat}"

    def set_meta(self) -> None:
        """Sets the run meta"""
        self.meta = RunMeta().get_meta(self.paths)

    def create_save_paths(self) -> None:
        """Creates save paths"""
        # create save paths
        for s_path in self.paths.save:
            if not pexists(s_path):
                os.makedirs(s_path)

    def get_rawdata(self, transform_dict: dict | None = None) -> None:
        """Reads the data from various logs and does some repairs/fixes for standardization

        Args:
            transform_dict (dict | None): Maps numbered state transitions (2->3) to named ones, e.g.stimstart
            Falls back to this Run's ``state_transitions`` when None.
        """
        if transform_dict is None:
            transform_dict = self.state_transitions or None

        self.read_run_data()
        self.translate_state_changes(transform_dict)

        self.rawdata = extract_trial_count(self.rawdata)
        # add total iStim just in case
        self.rawdata = add_total_iStim(self.rawdata)
        self.repair_rawdata()

    def repair_rawdata(self) -> None:
        """Hook for paradigm-specific rawdata fixes after the standard read. No-op by default."""
        pass

    def analyze_run(self) -> None:
        """Parse rawdata into the trial table, then run the paradigm hooks.

        Subclasses customize via ``augment_data`` (derived columns) and ``compute_stats``
        rather than overriding this loop.
        """
        run_data = self.get_trials()

        # set the data object
        self.data.set_data(run_data)

        self.augment_data()
        self.stats = self.compute_stats()

    def augment_data(self) -> None:
        """Hook to add paradigm-specific derived columns after set_data. No-op by default."""
        pass

    def compute_stats(self) -> dict | None:
        """Hook returning a per-run summary-stats dict (persisted with the data). None by default."""
        return None

    def get_trials(self) -> pt.DataFrame:
        """Gathers all the data, validates them and returns a dataframe

        Returns:
            pt.DataFrame: Table of trials
        """
        data_to_append = None
        trial_nos = np.unique(self.rawdata["statemachine"]["trialNo"])
        pbar = tqdm(trial_nos, desc="Extracting trial data:", disable=not config.verbose)
        for t in pbar:
            _trial = self.trial_handler.get_trial(int(t), self.rawdata)
            if _trial is not None:
                if data_to_append is None:
                    data_to_append = _trial
                else:
                    for k, v in _trial.items():
                        data_to_append[k].extend(v)

            pbar.update()

        return (
            pt.DataFrame(data_to_append)
            .set_model(self.trial_handler.trial_model)
            .derive()
            .drop()
            .cast()
            .fill_null(strategy="defaults")
            .validate()
        )

    @staticmethod
    def read_combine_logs(stimlog_path: str | list[str], riglog_path: str | list[str]) -> tuple[dict, dict]:
        """Reads the logs and combines them if multiple logs of same type exist in the run directory

        Args:
            stimlog_path (str | list[str]): path to .stimlog file
            riglog_path (str | list[str]): path to .riglog file

        Returns:
            tuple[dict, dict]: Rawdata dictionary and comments dictionary
        """
        if isinstance(stimlog_path, list) and isinstance(riglog_path, list):
            assert len(stimlog_path) == len(riglog_path), (
                f"The number stimlog files need to be equal to amount of riglog files {len(stimlog_path)}=/={len(riglog_path)}"
            )

            stim_data_all = []
            rig_data_all = []
            stim_comments = []
            rig_comments = []
            for i, s_log in enumerate(stimlog_path):
                try:
                    temp_slog, temp_scomm = parse_stimpy_log(s_log)
                except Exception:
                    # probably not the right stimpy version, try github
                    temp_slog, temp_scomm = parse_stimpygithub_log(s_log)
                temp_rlog, temp_rcomm = parse_stimpy_log(riglog_path[i])
                stim_data_all.append(temp_slog)
                rig_data_all.append(temp_rlog)
                stim_comments.extend(temp_scomm)
                rig_comments.extend(temp_rcomm)

            stim_data = stitch_logs(stim_data_all, isStimlog=True)  # stimlog
            rig_data = stitch_logs(rig_data_all, isStimlog=False)  # riglog
        else:
            stim_data, stim_comments = parse_stimpy_log(stimlog_path)
            rig_data, rig_comments = parse_stimpy_log(riglog_path)

        rawdata = {**stim_data, **rig_data}
        comments = {"stimlog": stim_comments, "riglog": rig_comments}
        return rawdata, comments

    def read_run_data(self) -> None:
        """Reads the data from concatanated riglog and stimlog files, and if exists, from camlog files"""
        # stimlog and camlog
        rawdata, self.comments = self.read_combine_logs(self.paths.stimlog, self.paths.riglog)
        self.rawdata = extrapolate_time(rawdata)

        # sometimes screen has an extra '0' cvalue entry in the beginning, omit that entry:
        if len(self.rawdata["screen"]):
            if self.rawdata["screen"][0, "value"] == 0:
                self.rawdata["screen"] = self.rawdata["screen"].slice(1)

        if self.paths.onepcam is not None and pexists(self.paths.onepcamlog):
            self.rawdata["onepcam_log"], self.comments["onepcam"], _ = parse_labcams_log(self.paths.onepcamlog)

        # try eyecam and facecam either way
        if self.paths.eyecam is not None and pexists(self.paths.eyecamlog):
            self.rawdata["eyecam_log"], self.comments["eyecam"], _ = parse_labcams_log(self.paths.eyecamlog)

        if self.paths.facecam is not None and pexists(self.paths.facecamlog):
            self.rawdata["facecam_log"], self.comments["facecam"], _ = parse_labcams_log(self.paths.facecamlog)

        display("Read rawdata")

    def translate_state_changes(self, transform_dict: dict = None) -> None:
        """Checks if state data exists and translated the state transitions according to defined translation dictionary
        This function needs the translate transition to be defined beforehand

        Args:
            transform_dict (dict): The dictionary that maps the numbered state transitions (2->3) to named transitions (stimstart)
        """
        if transform_dict is None:
            display(
                ">> WARNING! << No state transofrmation dictionary provided, using a generic one. It is very likely this will cause issues",
                color="yellow",
            )
            transform_dict = {
                "0->1": "trialstart",
                "1->2": "stimstart",
                "2->3": "stimend",
                "3->0": "trialend",
            }

        def translate_transition(old_state: str, new_state: str) -> str:
            _key = f"{int(old_state)}->{int(new_state)}"
            return transform_dict[_key]

        # do the translation
        try:
            self.rawdata["statemachine"] = self.rawdata["statemachine"].with_columns(
                pl.struct(["oldState", "newState"])
                .map_elements(
                    lambda x: translate_transition(x["oldState"], x["newState"]),
                    return_dtype=str,
                )
                .alias("transition")
            )
        except WrongSessionTypeError:
            raise WrongSessionTypeError(
                "Unable to translate state changes to valid transitions. Make sure you are using the correct session type to analyze your data!"
            )

        if self.rawdata["statemachine"]["transition"].is_null().any():
            raise StateMachineError(
                """There are untranslated state changes! Are you sure you're using the correct state transition keys?
                                    I got"""
            )

        # rename cycle to 'trialNo for semantic reasons
        self.rawdata["statemachine"] = self.rawdata["statemachine"].rename({"cycle": "trialNo"})

    def is_run_saved(self) -> bool:
        """Checks if data already exists

        Returns:
            bool: True if saved ata is found, else False
        """
        loadable = False
        for d_path in self.paths.data:
            if pexists(d_path):
                loadable = True
                display(f"Found saved data: {d_path}", color="cyan")
                break
            else:
                display(f"{d_path} does not exist...", color="yellow")
        return loadable

    def save_run(self, save_mat: bool = False) -> None:
        """Saves the run data

        Args:
            save_mat (bool, optional): Flag to save the dataframe as a .mat file. Defaults to False
        """
        if self.data is not None and self.data.data is not None:
            for s_path in self.paths.save:
                if not pexists(s_path):
                    os.makedirs(s_path)
                self.data.save_data(s_path, save_mat)
                display(f"Saved session data to {s_path}", color="green")
        self._save_stats()

    def _save_stats(self) -> None:
        """Persist the per-run stats dict (if the paradigm produced one)."""
        if self.stats is None:
            return
        for s_path in self.paths.save:
            save_dict_json(pjoin(s_path, "sessionStats.json"), self.stats)

    def load_run(self) -> None:
        """Loads the saved run data (and stats, if present)."""
        for d_path in self.paths.data:
            if pexists(d_path):
                self.data.load_data(d_path)
                display(f"Loaded session data from {d_path}", color="green")
                break

    def _load_stats(self) -> None:
        """Load the per-run stats dict if a paradigm saved one."""
        for s_path in self.paths.save:
            stat_path = pjoin(s_path, "sessionStats.json")
            if pexists(stat_path):
                self.stats = load_json_dict(stat_path)
                break
