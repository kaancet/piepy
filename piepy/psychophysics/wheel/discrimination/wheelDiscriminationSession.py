import time
import os
from os.path import join as pjoin
import polars as pl


from ....core.run import RunData, Run
from ....core.pathfinder import Paths
from ....core.session import Session
from ....core.io import display, load_json_dict, save_dict_json
from ....core.log_repair_functions import fix_first_line_state_logging
from .wheelDiscriminationTrial import WheelDiscriminationTrialHandler


STATE_TRANSITION_KEYS = {
    "0->1": "trialstart",
    "1->2": "stimstart",
    "2->3": "responsestart",
    "3->4": "correct",
    "3->5": "incorrect",
    "3->6": "catch",
    "4->6": "stimendcorrect",
    "5->6": "stimendincorrect",
    "6->0": "trialend",
}


class WheelDiscriminationRunData(RunData):
    def __init__(self, data=None):
        super().__init__(data)

    def set_data(self, data: pl.DataFrame) -> None:
        """Sets the data of the session and augments it

        Args:
            data (pl.Dataframe): Dataframe to initialize the run data
        """
        if data is not None:
            super().set_data(data)
            self.add_qolumns()

    def add_qolumns(self) -> None:
        """Adds some quality of life (qol) columns"""

        # add a stim_side column for ease of access
        self.data = self.data.with_columns(
            pl.when(pl.col("target_pos").list.get(0) > 0)
            .then(pl.lit("contra"))
            .when(pl.col("target_pos").list.get(0) < 0)
            .then(pl.lit("ipsi"))
            .otherwise(None)
            .alias("target_side")
        )

        # right choice
        self.data = self.data.with_columns(
            pl.col("state_outcome")
            .cast(pl.Boolean)
            .xor(pl.col("correct_side").cast(pl.Boolean))
            .not_()
            .cast(pl.Int64)
            .alias("right_choice")
        )

        # add response_time columns
        self.data = self.data.with_columns(
            pl.col("state_response_time").alias("response_time")
        )

        # round sf and tf
        self.data = self.data.with_columns(
            [
                pl.col(_sf).round(2).alias(_sf)
                for _sf in self.data.columns
                if _sf.endswith("_sf")
            ]
        )
        self.data = self.data.with_columns(
            [
                pl.col(_tf).round(2).alias(_tf)
                for _tf in self.data.columns
                if _tf.endswith("_tf")
            ]
        )

    def add_stim_diff_and_type(self, discrim_of: str) -> None:
        """_summary_

        Args:
            discrim_of (str): _description_
        """

        if discrim_of == "width":
            u = "deg"
        elif discrim_of == "sf":
            u = "cpd"
        elif discrim_of == "tf":
            u = "Hz"
        elif discrim_of == "contrast":
            u = "%"
        else:
            u = "NA"

        self.data = self.data.with_columns(pl.lit(discrim_of).alias("discriminating"))

        self.data = self.data.with_columns(
            pl.when(pl.col("target_side") == "contra")
            .then(pl.col(f"target_{discrim_of}") - pl.col(f"distract_{discrim_of}"))
            .otherwise(
                pl.col(f"distract_{discrim_of}") - pl.col(f"target_{discrim_of}")
            )
            .alias(f"diff_{discrim_of}")
        )

        # adds string stimtype
        self.data = self.data.with_columns(
            (
                pl.col(f"target_{discrim_of}").cast(pl.Utf8)
                + f"{u}_"
                + pl.col(f"distract_{discrim_of}").cast(pl.Utf8)
                + f"{u}"
            ).alias("stim_type")
        )

    def add_pattern_related_columns(self, pattern_path: str) -> None:
        """_Adds columns related to the silencing pattern if they exist

        Args:
            pattern_path (str): path to the opto_pattern

        Raises:
            KeyError: _description_
            ValueError: _description_
        """
        if len(self.data["opto"].unique()) == 1:
            # Regular training sessions
            # add the pattern name depending on pattern id
            self.data = self.data.with_columns(pl.lit(None).alias("opto_region"))
            # add 'stimkey' from sftf
            self.data = self.data.with_columns(
                (pl.col("stim_type") + "_-1").alias("stimkey")
            )
            # add stim_label for legends and stuff
            self.data = self.data.with_columns(
                (pl.col("stim_type")).alias("stim_label")
            )
        else:
            if pattern_path is not None and os.path.exists(pattern_path):
                pattern_names = {}
                for im in os.listdir(pattern_path):
                    if im.endswith(".tif"):
                        pattern_id = int(im[:-4].split("_")[-1])
                        if pattern_id == -1:
                            pattern_names[pattern_id] = "nonopto"
                        else:
                            name = im[:-4].split("_")[-2]
                            pattern_names[pattern_id] = name

                try:
                    # add the pattern name depending on pattern id
                    self.data = self.data.with_columns(
                        pl.struct(["opto_pattern", "state_outcome"])
                        .map_elements(
                            lambda x: (
                                pattern_names[x["opto_pattern"]]
                                if x["state_outcome"] != -1
                                else None
                            ),
                            return_dtype=str,
                        )
                        .alias("opto_region")
                    )
                except KeyError:
                    raise KeyError(
                        "Opto pattern not set correctly. You need to change the number at the end of the opto pattern image file to an integer (0,-1,1,..)!"
                    )
            else:
                raise ValueError(f"{pattern_path} is not a valid path")

            # add 'stimkey' from sftf
            self.data = self.data.with_columns(
                (
                    pl.col("stim_type")
                    + "_"
                    + pl.col("opto_pattern").cast(int).cast(str)
                ).alias("stimkey")
            )
            # add stim_label for legends and stuff
            self.data = self.data.with_columns(
                (pl.col("stim_type") + "_" + pl.col("opto_region").cast(str)).alias(
                    "stim_label"
                )
            )


class WheelDiscriminationRun(Run):
    def __init__(self, path: Paths) -> None:
        super().__init__(path)
        self.data = WheelDiscriminationRunData()
        self.trial_handler = WheelDiscriminationTrialHandler()

    def __repr__(self):
        _base = super().__repr__()
        return _base

    def get_rawdata(self, transform_dict: dict) -> None:
        """Reads the data from various logs and does some repairs/fixes for standardization

        Args:
            transform_dict (dict): The dictionary that maps the numbered state transitions (2->3) to named transitions (stimstart)
        """
        super().get_rawdata(transform_dict)

        self.rawdata = fix_first_line_state_logging(self.rawdata)

    def analyze_run(self, discrim_of: str) -> None:
        """Main loop to extract data from rawdata, should be overwritten in child classes

        Args:
            discrim_of (str):
        """

        super().analyze_run()

        self.data.add_stim_diff_and_type(discrim_of=discrim_of)

        self.data.add_pattern_related_columns(self.paths.opto_pattern)

        self.stats = get_run_stats(self.data.data)

    def save_run(self) -> None:
        """Saves the run data, meta and stats"""
        super().save_run()

        for s_path in self.paths.save:
            save_dict_json(pjoin(s_path, "sessionStats.json"), self.stats)

    def load_run(self) -> None:
        """Loads the run data and stats if exists"""
        super().load_run()

        for s_path in self.paths.save:
            stat_path = pjoin(s_path, "sessionStats.json")
            if os.path.exists(stat_path):
                self.stats = load_json_dict(stat_path)
                break


class WheelDiscriminationSession(Session):
    def __init__(
        self,
        sessiondir,
        load_flag: bool,
        save_mat: bool = False,
        skip_google: bool = True,
    ):
        start = time.time()
        super().__init__(sessiondir, load_flag, save_mat)

        # initialize runs : read and parse or load the data
        self.init_session_runs(skip_google)

        end = time.time()
        display(f"Done! t={(end - start):.2f} s")

    def __repr__(self):
        r = f"Discrimination Session {self.sessiondir}"
        return r

    def init_session_runs(self, skip_google: bool = True) -> None:
        """Initializes runs in a session"""
        for r in range(self.run_count):
            _path = Paths(self.paths.all_paths, r)
            # the run itself
            _run = WheelDiscriminationRun(_path)
            _run.set_meta(skip_google)
            _run.get_rawdata(STATE_TRANSITION_KEYS)
            if _run.is_run_saved() and self.load_flag:
                display(f"Loading from {_run.paths.save}")
                _run.load_run()
            else:
                discrim_of = _run.meta["opts"]["AttendVectorName"]
                _run.analyze_run(discrim_of)
                _run.data.add_metadata_columns(_run.meta)
                _run.save_run()

            self.runs.append(_run)
            self.metas.append(_run.meta)
            self.stats.append(_run.stats)


def get_run_stats(data: pl.DataFrame) -> dict:
    """Gets run stats from run dataframe"""
    stats_dict = {}
    correct_data = data.filter(pl.col("outcome") == "correct")
    miss_data = data.filter(pl.col("outcome") == "incorrect")
    nonopto_data = data.filter(pl.col("opto") == 0)
    opto_data = data.filter(pl.col("opto") == 1)

    # counts #
    stats_dict["total_trial_count"] = len(data)
    stats_dict["correct_trial_count"] = len(correct_data)
    stats_dict["miss_trial_count"] = len(miss_data)
    stats_dict["opto_trial_count"] = len(opto_data)
    stats_dict["opto_ratio"] = round(
        100 * stats_dict["opto_trial_count"] / stats_dict["total_trial_count"], 3
    )

    # rates #
    nonopto_correct_count = len(nonopto_data.filter(pl.col("outcome") == "hit"))
    stats_dict["nonopto_hit_rate"] = round(
        100 * nonopto_correct_count / len(nonopto_data), 3
    )

    stats_dict["correct_rate"] = round(
        100 * stats_dict["correct_trial_count"] / stats_dict["total_trial_count"], 3
    )

    # median response time #
    stats_dict["median_response_latency "] = round(
        nonopto_data.filter(pl.col("outcome") == "correct")[
            "state_response_time"
        ].median(),
        3,
    )

    return stats_dict
