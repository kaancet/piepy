import time
import os
import numpy as np
from PIL import Image
import tifffile as tf
import polars as pl
import scipy.stats as st
from tabulate import tabulate
from os.path import join as pjoin

from ....core.io import display, load_json_dict, save_dict_json
from ....core.run import RunData, Run
from ....core.pathfinder import Paths
from ....core.session import Session
from .wheelDetectionTrial import WheelDetectionTrialHandler


STATE_TRANSITION_KEYS = {
    "0->1": "trialstart",
    "1->2": "cuestart",
    "2->3": "stimstart",
    "2->5": "early",
    "3->4": "hit",
    "3->5": "miss",
    "3->6": "catch",
    "4->6": "stimendcorrect",
    "5->6": "stimendincorrect",
    "6->0": "trialend",
}


class WheelDetectionRunData(RunData):
    def __init__(self, data: pl.DataFrame = None) -> None:
        super().__init__(data)

    def set_data(self, data: pl.DataFrame) -> None:
        """Sets the data of the session and augments it"""
        if data is not None:
            super().set_data(data)
            self.add_qolumns()

    def set_outcome(self, outcome_type: str = "state") -> None:
        """Sets the outcome column to the selected column"""
        display(f"Setting outcome to {outcome_type}")
        col_name = f"{outcome_type}_outcome"
        if "outcome" not in self.data.columns:
            self.data = self.data.with_columns(pl.col(col_name).alias("outcome"))
        else:
            if col_name in self.data.columns:
                self.data = self.data.with_columns(pl.col(col_name).alias("outcome"))
            else:
                raise ValueError(
                    f"{outcome_type} is not a valid outcome type!!! Try 'pos', 'speed' or 'state'."
                )

    def compare_outcomes(self) -> None:
        """Compares the different outcome types and prints a small summary table"""
        out_cols = [c for c in self.data.columns if "_outcome" in c]
        q = (
            self.data.group_by(out_cols)
            .agg([pl.count().alias("count")])
            .sort(["state_outcome"])
        )
        tmp = q.to_pandas()
        print(tabulate(tmp, headers=q.columns))

    def add_qolumns(self) -> None:
        """Adds some quality of life (qol) columns"""

        # add a stim_side column for ease of access
        self.data = self.data.with_columns(
            pl.when(pl.col("stim_pos") > 0)
            .then(pl.lit("contra"))
            .when(pl.col("stim_pos") < 0)
            .then(pl.lit("ipsi"))
            .when(pl.col("stim_pos") == 0)
            .then(pl.lit("catch"))
            .otherwise(None)
            .alias("stim_side")
        )

        # round sf and tf
        self.data = self.data.with_columns(
            [
                (pl.col("sf").round(2).alias("sf")),
                (pl.col("tf").round(1).alias("tf")),
            ]
        )

        # adds string stimtype
        self.data = self.data.with_columns(
            (
                pl.col("sf").round(2).cast(str) + "cpd_" + pl.col("tf").cast(str) + "Hz"
            ).alias("stim_type")
        )

        # add signed contrast
        self.data = self.data.with_columns(
            pl.when(pl.col("stim_side") == "ipsi")
            .then((pl.col("contrast") * -1))
            .otherwise(pl.col("contrast"))
            .alias("signed_contrast")
        )

        # add easy/hard contrast type groups
        self.data = self.data.with_columns(
            pl.when(pl.col("contrast") >= 25)
            .then(pl.lit("easy"))
            .when((pl.col("contrast") < 25) & (pl.col("contrast") > 0))
            .then(pl.lit("hard"))
            .when(pl.col("contrast") == 0)
            .then(pl.lit("catch"))
            .otherwise(None)
            .alias("contrast_type")
        )

    def add_pattern_related_columns(self) -> None:
        """Adds columns related to the silencing pattern if they exist"""
        if len(self.data["opto"].unique()) == 1:
            # Regular training sessions
            # add the pattern name depending on pattern id
            self.data = self.data.with_columns(pl.lit(None).alias("opto_region"))
            # add 'stimkey' from sftf
            self.data = self.data.with_columns(
                (pl.col("stim_type") + "_-1").alias("stimkey")
            )
            # add stim_label for legends and stuff
            self.data = self.data.with_columns((pl.col("stim_type")).alias("stim_label"))
        else:
            if isinstance(self.pattern_names, dict):
                try:
                    # add the pattern name depending on pattern id
                    self.data = self.data.with_columns(
                        pl.struct(["opto_pattern", "state_outcome"])
                        .map_elements(
                            lambda x: (
                                self.pattern_names[x["opto_pattern"]]
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
            elif isinstance(self.pattern_names, str):
                display(f"{self.paths.opto_pattern} NO OPTO PATTERN DIRECTORY!!")
                self.data = self.data.with_columns(
                    pl.struct(["opto_pattern", "state_outcome"])
                    .apply(
                        lambda x: self.pattern_names if x["state_outcome"] != -1 else None
                    )
                    .alias("opto_region")
                )
            else:
                raise ValueError(f"Weird pattern name: {self.pattern_names}")

            # add 'stimkey' from sftf
            self.data = self.data.with_columns(
                (pl.col("stim_type") + "_" + pl.col("opto_pattern").cast(str)).alias(
                    "stimkey"
                )
            )
            # add stim_label for legends and stuff
            self.data = self.data.with_columns(
                (pl.col("stim_type") + "_" + pl.col("opto_region").cast(str)).alias(
                    "stim_label"
                )
            )

    @staticmethod
    def get_pattern_size(pattern_img: np.ndarray) -> float:
        """Gets the pattern size using the x and y vertices from"""
        pass

    def get_opto_images(self, get_from: str) -> None:
        """Reads the related run images(window, pattern, etc)
        Returns a dict with images and also a dict that"""
        if get_from is not None and os.path.exists(get_from):
            self.sesh_imgs = {}
            self.pattern_names = {}
            self.sesh_patterns = {}
            for im in os.listdir(get_from):
                if im.endswith(".tif"):
                    pattern_id = int(im[:-4].split("_")[-1])
                    read_img = tf.imread(pjoin(get_from, im))
                    if pattern_id == -1:
                        self.sesh_imgs["window"] = read_img
                        self.pattern_names[pattern_id] = "nonopto"
                    else:
                        name = im[:-4].split("_")[-2]
                        self.pattern_names[pattern_id] = name
                        self.sesh_imgs[name] = read_img
                elif im.endswith(".bmp"):
                    pattern_id = int(im.split("_")[0])
                    name = im.split("_")[1]
                    read_bmp = np.array(Image.open(pjoin(get_from, im)).convert("L"))
                    self.sesh_patterns[name] = read_bmp[::-1, ::-1]
        else:
            self.sesh_imgs = None
            self.sesh_patterns = None
            self.pattern_names = None  # '*'+self.paths.pattern.split('_')[4]+'*'


class WheelDetectionRun(Run):
    def __init__(self, path: Paths) -> None:
        super().__init__(path)
        self.data = WheelDetectionRunData()
        self.trial_handler = WheelDetectionTrialHandler()

    def __repr__(self):
        _base = super().__repr__()
        _stats = ""
        if self.stats is not None:
            _stats = (
                f"- HR={self.stats['hit_rate']}% - FA={self.stats['false_alarm_rate']}"
            )
        return _base + _stats

    def analyze_run(self, transform_dict: dict) -> None:
        """Main loop to extract data from rawdata"""
        super().analyze_run(transform_dict)

        # get related images(silencing patterns etc)
        self.data.get_opto_images(self.paths.opto_pattern)
        self.data.add_pattern_related_columns()

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


class WheelDetectionSession(Session):
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
        display(f"Done! t={(end-start):.2f} s")

    def __repr__(self):
        r = f"Detection Session {self.sessiondir}"
        return r

    def init_session_runs(self, skip_google: bool = True) -> None:
        """Initializes runs in a session"""
        for r in range(self.run_count):
            _path = Paths(self.paths.all_paths, r)
            # the run itself
            _run = WheelDetectionRun(_path)
            _run.set_meta(skip_google)
            if _run.is_run_saved() and self.load_flag:
                display(f"Loading from {_run.paths.save}")
                _run.load_run()
            else:
                _run.analyze_run(STATE_TRANSITION_KEYS)
                _run.data.add_metadata_columns(_run.meta)
                _run.save_run()

            self.runs.append(_run)
            self.metas.append(_run.meta)
            self.stats.append(_run.stats)


def get_run_stats(data: pl.DataFrame) -> dict:
    """Gets run stats from run dataframe"""
    stats_dict = {}
    early_data = data.filter((pl.col("outcome") == "early"))
    stim_data = data.filter((pl.col("outcome") != "early") & (pl.col("isCatch") == 0))
    catch_data = data.filter((pl.col("outcome") != "early") & (pl.col("isCatch") == 1))
    correct_data = stim_data.filter(pl.col("outcome") == "hit")
    miss_data = stim_data.filter(pl.col("outcome") == "miss")
    nonopto_data = stim_data.filter(pl.col("opto") == 0)
    opto_data = stim_data.filter(pl.col("opto") == 1)

    # counts #
    stats_dict["total_trial_count"] = len(data)
    stats_dict["early_trial_count"] = len(early_data)
    stats_dict["stim_trial_count"] = len(stim_data)
    stats_dict["correct_trial_count"] = len(correct_data)
    stats_dict["miss_trial_count"] = len(miss_data)
    stats_dict["catch_trial_count"] = len(catch_data)
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
    stats_dict["hit_rate"] = round(
        100 * stats_dict["correct_trial_count"] / stats_dict["stim_trial_count"], 3
    )
    stats_dict["false_alarm_rate"] = round(
        100 * stats_dict["early_trial_count"] / stats_dict["total_trial_count"], 3
    )
    stats_dict["nogo_rate"] = round(
        100 * stats_dict["miss_trial_count"] / stats_dict["stim_trial_count"], 3
    )

    # median response time #
    stats_dict["median_response_latency "] = round(
        nonopto_data.filter(pl.col("outcome") == "hit")["state_response_time"].median(), 3
    )

    # d prime(?) #
    stats_dict["d_prime"] = st.norm.ppf(stats_dict["hit_rate"] / 100) - st.norm.ppf(
        stats_dict["false_alarm_rate"] / 100
    )

    ## performance on easy trials
    easy_data = nonopto_data.filter(pl.col("contrast").is_in([100, 50]))
    stats_dict["easy_trial_count"] = len(easy_data)
    easy_correct_count = len(easy_data.filter(pl.col("outcome") == "hit"))
    if stats_dict["easy_trial_count"]:
        stats_dict["easy_hit_rate"] = round(
            100 * easy_correct_count / stats_dict["easy_trial_count"], 3
        )
        stats_dict["easy_median_response_latency"] = round(
            easy_data.filter(pl.col("outcome") == "hit")["state_response_time"].median(),
            3,
        )
    else:
        stats_dict["easy_hit_rate"] = -1
        stats_dict["easy_median_response_latency"] = -1

    return stats_dict


# @timeit("Getting rolling averages...")
# def get_running_stats(data_in: pl.DataFrame, window_size: int = 20) -> pl.DataFrame:
#     """Gets the running statistics of certain columns"""
#     data_in = data_in.with_columns(
#         pl.col("response_latency")
#         .rolling_median(window_size)
#         .alias("running_response_latency")
#     )
#     # answers
#     outcomes = {"correct": 1, "nogo": 0, "early": -1}

#     for k, v in outcomes.items():
#         key = "fraction_" + k
#         data_arr = data_in["state_outcome"].to_numpy()
#         data_in[key] = get_fraction(data_arr, fraction_of=v)

#     return data_in


# import multiprocessing
# from multiprocessing import Pool
# from tqdm.contrib.concurrent import process_map
# def get_all_trials(self) -> pl.DataFrame | None:
#     """ """

#     trial_nos = np.unique(self.rawdata["statemachine"]["trialNo"])
#     trial_nos = [int(t) for t in trial_nos]

#     if cfg.multiprocess["enable"]:
#         mngr = multiprocessing.Manager()
#         self.queue = mngr.Queue(-1)
#         self._list_data = mngr.list()

#         listener = multiprocessing.Process(
#             target=self.logger.listener_process, args=(self.queue,)
#         )
#         listener.start()

#         process_map(
#             self._get_trial,
#             trial_nos,
#             max_workers=cfg.multiprocess["cores"],
#             chunksize=100,
#         )

#     else:
#         self.logger.listener_configurer()
#         self._list_data = []
#         pbar = tqdm(
#             trial_nos,
#             desc="Extracting trial data:",
#             leave=True,
#             position=0,
#             disable=not config.verbose,
#         )
#         for t in pbar:
#             self._get_trial(t)

#     # convert list of dicts to dict of lists
#     data_to_frame = {k: [v] for k, v in self._list_data[0].items()}
#     for i in range(1, len(self._list_data)):
#         for k, v in self._list_data[i].items():
#             data_to_frame[k].append(v)

#     r_data = pl.DataFrame(data_to_frame)
#     # order by trial no
#     r_data = r_data.sort("trial_no")

#     # add contrast titration boolean
#     uniq_stims = nonan_unique(r_data["contrast"].to_numpy())
#     isTitrated = 0
#     if len(uniq_stims) > len(self.meta.contrastVector):
#         isTitrated = 1
#     r_data = r_data.with_columns(
#         [pl.lit(isTitrated).cast(pl.Boolean).alias("isTitrated")]
#     )

#     if r_data.is_empty():
#         self.logger.error("THERE IS NO SESSION DATA !!!", cml=True)
#         return None
#     else:
#         return r_data

# def _get_trial(self, trial_no: int) -> pl.DataFrame | None:
#     """Main loop that parses the rawdata into a polars dataframe where each row corresponds to a trial"""
#     # self.logger.attach_to_queue(self.log_queue)
#     if cfg.multiprocess["enable"]:
#         self.logger.worker_configurer(self.queue)

#     _trial = WheelDetectionTrial(
#         trial_no=trial_no, meta=self.meta, logger=self.logger
#     )
#     # get the data slice using state changes
#     if _trial.set_data_slices(self.rawdata):
#         _data = _trial.trial_data_from_logs()

#         if _data["state_outcome"] is not None:
#             self._list_data.append(_data)
