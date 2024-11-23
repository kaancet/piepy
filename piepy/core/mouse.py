import os
import glob
import natsort
import argparse
import importlib
import numpy as np
import polars as pl
import pandas as pd
from tqdm import tqdm
from os.path import join as pjoin
from collections import namedtuple
from datetime import datetime as dt
from os.path import dirname, abspath, normpath

from ..core.config import config as cfg
from .utils import timeit
from .io import display
from ..gsheet_functions import GSheet


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
        ), f"Cumulative and summary data has to be the same length!!"

        tmp = pl.DataFrame(summary_data_list)
        if self.summary_data is not None:
            if self.summary_data.dtypes != tmp.dtypes:
                self.summary_data = self.summary_data.with_columns(
                    [
                        pl.col(n).cast(t, strict=False)
                        for n, t in zip(self.summary_data.columns, tmp.dtypes)
                        if t != pl.Null
                    ]
                )
            self.summary_data = pl.concat([self.summary_data, tmp])
        else:
            self.summary_data = tmp

        if self.cumul_data is None:
            # if no cumulative data exists, make one from the first in list and iterate over the rest
            self.cumul_data = cumul_data_list[0]
            cumul_data_list = cumul_data_list[1:]

        if "cumul_trial_no" in self.cumul_data.columns:
            self.cumul_data = self.cumul_data.drop("cumul_trial_no")

        for new_cumul in cumul_data_list:
            # if there are columns that are not in df add them with None
            for c in new_cumul.columns:
                if c not in self.cumul_data.columns:
                    self.cumul_data = self.cumul_data.with_columns(pl.lit(None).alias(c))

            # sorting the columns
            new_cumul = new_cumul.select(self.cumul_data.columns)
            # fixing column datatypes
            if self.cumul_data.dtypes != new_cumul.dtypes:
                try:
                    self.cumul_data = self.cumul_data.with_columns(
                        [
                            pl.col(n).cast(t)
                            for n, t in zip(self.cumul_data.columns, new_cumul.dtypes)
                            if t != pl.Null
                        ]
                    )
                except:
                    print("jlsdiobjsdf")
            try:
                self.cumul_data = pl.concat([self.cumul_data, new_cumul])
            except pl.SchemaError:
                raise pl.SchemaError("WEIRDNESS WITH COLUMNS")

        # sort both by date
        self.cumul_data = self.cumul_data.sort(["date", "trial_no"])
        self.summary_data = self.summary_data.sort("date")
        self.cumul_data = self.cumul_data.with_row_count("cumul_trial_no", offset=1)

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

        self.read_googlesheet()
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
        else:
            assert (
                len(date_interval) <= 2
            ), f"You need to provide a single start(1) or start and end dates(2), got {len(date_interval)} dates"

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
            if n
            in ["analysis", "presentation", "training", "gsheet", "colors", "database"]
        }
        tmp_dict = {name: path[0] for name, path in config.items()}
        tmp_paths = namedtuple("Paths", list(tmp_dict.keys()))
        self.paths = tmp_paths(**tmp_dict)

    def get_sessions(self) -> pl.DataFrame:
        """Create a session list dataframe"""
        experiment_sessions = glob.glob(f"{self.paths.presentation}/*{self.animalid}*/")
        training_sessions = glob.glob(
            f"{self.paths.training}/*{self.animalid}*__no_cam_*/"
        )
        tmp = experiment_sessions + training_sessions
        all_sessions = natsort.natsorted(tmp, reverse=False)

        types = []
        dates = []
        sessions = []
        for sesh in all_sessions:
            s = sesh.split(os.sep)[-2]
            dates.append(dt.strptime(s.split("_")[0], "%y%m%d"))

            if "training" in sesh:
                types.append("training")
            else:
                if "1P" in s:
                    if "opto" in s:
                        types.append("opto")
                    else:
                        types.append("1P")
                elif "2P" in s:
                    types.append("2P")
                elif "opto" in sesh:
                    types.append("opto")
                else:
                    types.append(None)

            sessions.append(s)

        session_list = pl.DataFrame(
            data={
                "animalid": [self.animalid] * len(all_sessions),
                "sessiondir": sessions,
                "date": dates,
                "exp_type": types,
                "fullpath": all_sessions,
            }
        )

        session_list = session_list.with_columns(
            pl.when(pl.col("sessiondir").str.contains("detect"))
            .then(pl.lit("detection"))
            .otherwise(None)
            .alias("paradigm")
        )
        session_list = session_list.sort("date")

        return session_list

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
        summary_to_append = []
        cumul_to_append = []

        # we're analyzing the individual sessions here in this below loop
        pbar = tqdm(missing_sessions)
        for i, row in enumerate(missing_sessions.iter_rows()):
            # last_saved will not enter here as missing sessions will be []
            pbar.set_description(f"Analyzing {row[1]} [{i+1}/{len(missing_sessions)}]")

            if load_type == "no_load":
                detect_session = self.session_parser(
                    row[1], load_flag=False, skip_google=True
                )
            else:
                # reanalyze load type should enter here
                detect_session = self.session_parser(
                    row[1], load_flag=True, skip_google=True
                )

            session_data = detect_session.data.data
            gsheet_dict = self.get_gsheet_row(
                detect_session.meta.baredate,
                cols=[
                    "paradigm",
                    "supp water [µl]",
                    "user",
                    "time [hh:mm]",
                    "rig water [µl]",
                ],
            )

            if len(session_data):
                # add behavior related fields as a dictiionary
                meta = detect_session.get_meta()
                summary_temp = {}
                summary_temp["date"] = meta.baredate
                summary_temp["blank_time"] = meta.openStimDuration
                summary_temp["response_window"] = meta.closedStimDuration
                try:
                    summary_temp["level"] = int(meta.level)
                except:
                    summary_temp["level"] = -1
                summary_temp["session_no"] = session_counter + 1

                # put data from session stats
                for k in detect_session.stats.__slots__:
                    summary_temp[k] = getattr(detect_session.stats, k, None)

                # put values from session meta data
                summary_temp["weight"] = meta.weight
                summary_temp["task"] = meta.controller
                summary_temp["sf"] = meta.sf_values
                summary_temp["tf"] = meta.tf_values
                summary_temp["rig"] = meta.rig
                summary_temp = {**summary_temp, **gsheet_dict}

                session_data = session_data.with_columns(
                    [
                        (pl.lit(session_counter + 1)).alias("session_no"),
                        (pl.lit(row[3])).alias("session_type"),
                    ]
                )

                cumul_to_append.append(session_data)
                if i == 0:
                    summary_to_append = {k: [v] for k, v in summary_temp.items()}
                else:
                    for k, v in summary_temp.items():
                        summary_to_append[k].append(v)

                session_counter += 1
            else:
                display(f" >>> WARNING << NO DATA FOR SESSION {row[1]}", color="yellow")
                continue
            pbar.update()

        if len(summary_to_append):
            self.data.append(cumul_to_append, summary_to_append)
            display("Appended new data!", color="cyan")

    # TODO:
    def get_gsheet_row(self, date: str, cols: list = None) -> pl.DataFrame:
        sheet_stats = {}
        if cols is None:
            cols = ["weight [g]", "supp water [µl]", "user", "time [hh:mm]"]
        # current date data
        row = self.gsheet_df[self.gsheet_df["Date [YYMMDD]"] == date]
        for c in cols:
            key = c.split("[")[0].strip(" ")  # get rid of units in column names
            if len(row):
                sheet_stats[key] = row[c].values[0]
            else:
                sheet_stats[key] = None
        return sheet_stats

    # TODO:
    def read_googlesheet(self) -> None:
        """Reads all the entries from the googlesheet with the current animal id"""
        logsheet = GSheet("Mouse Database_new")
        # below, 2 is the log2021 sheet ID
        temp_df = logsheet.read_sheet(2)

        temp_df = temp_df[temp_df["Mouse ID"] == self.animalid]

        # convert decimal "," to "." and date string to datetime and drop na
        temp_df["weight [g]"] = temp_df["weight [g]"].apply(
            lambda x: str(x).replace(",", ".")
        )
        temp_df["weight [g]"] = pd.to_numeric(temp_df["weight [g]"], errors="coerce")
        temp_df["supp water [µl]"] = pd.to_numeric(
            temp_df["supp water [µl]"], errors="coerce"
        ).fillna(0)

        temp_df["Date [YYMMDD]"] = temp_df["Date [YYMMDD]"].apply(lambda x: str(x))
        temp_df["Date_dt"] = pd.to_datetime(temp_df["Date [YYMMDD]"], format="%y%m%d")

        self.gsheet_df = temp_df
        # self.gsheet_df = pl.from_pandas(data=temp_df)

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
                    f"Can't do load, there is no saved behavioral analysis files for cumulative (*.parquet) and summary (*.csv) data"
                )
            # returns an empty list, no new session will be analyzed
            missing_sessions = pl.DataFrame()
        return missing_sessions

    @staticmethod
    def get_session_class(session_type: str) -> None:
        """Initializes the relevant session parser"""
        session_class_name = f"{session_type}Session"
        if session_type == "detection":
            session_class_name = session_class_name[0].upper() + session_class_name[1:]
            session_class_name = f"wheel{session_class_name}"

        # __file__ is core.mouse.py
        mod_path = normpath(
            pjoin(
                abspath(dirname(dirname(__file__))),
                session_type,
                session_class_name + ".py",
            )
        )
        if os.path.exists(mod_path):
            mod = importlib.import_module(f"piepy.{session_type}.{session_class_name}")
            session_class_name = (
                session_class_name[0].upper() + session_class_name[1:]
            )  # uppercasing the first letter for class name
            return getattr(mod, session_class_name)
        else:
            raise ModuleNotFoundError(f"No module found at {mod_path}")


def main():
    load_help_str = """ Gathers the data from all the sessions in the session list
        load_type:    string to set how to load\n
        \n'last_saved' = loads only the last analyzed data, doesn't analyze and add new sessions since last analysis\n
        \n'load_and_add' = loads all the data and adds new sessions to the loaded data\n
        \n'reanalyze' = loads the session data and reanalyzes the behavior data from that\n
        \n'no_load' = doesn't load anything reanalyzes the sessions data from scratch\n
    """
    parser = argparse.ArgumentParser(description="Mouse Behavior Data Parsing Tool")

    parser.add_argument("id", metavar="animalid", type=str, help="Animal ID (e.g. KC133)")
    parser.add_argument(
        "-p",
        "--paradigm",
        metavar="paradigm",
        type=str,
        help="Behavior paradigm(e.g. detection)",
    )
    parser.add_argument("-l", "--load", metavar="load_type", type=str, help=load_help_str)
    parser.add_argument(
        "-d",
        "--date",
        metavar="dateinterval",
        type=str,
        default=None,
        help="Analysis start date (e.g. 231124)",
    )
    parser.add_argument(
        "-s",
        "--save",
        metavar="save_behavior",
        action=argparse.BooleanOptionalAction,
        type=str,
        default=True,
        help="Save behavior data or not",
    )

    """
    mouseparse -p detection -l no_load -d 231124 KC133
    """

    opts = parser.parse_args()

    display(f"Reading {opts.paradigm} Behavior for {opts.id}")
    m = Mouse(animalid=opts.id, paradigm=opts.paradigm, dateinterval=opts.date)
    m.gather_data(load_type=opts.load)
    if opts.save:
        m.save()


if __name__ == "__main__":
    main()
