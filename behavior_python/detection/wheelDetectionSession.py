import time
from PIL import Image
import tifffile as tf
import scipy.stats as st
from os.path import join as pjoin
from behavior_python.core.session import Session, SessionData, SessionMeta
from .wheelDetectionTrial import *
from tabulate import tabulate


class WheelDetectionData(SessionData):
    def __init__(self, data_paths, data: pl.DataFrame = None) -> None:
        super().__init__()
        self.set_paths(data_paths)
        self.set_data(data)

    def set_data(self, data: pl.DataFrame, isgrating: bool = False) -> None:
        if data is not None:
            super().set_data(data)
            self.pattern_imgs, self.patterns, pattern_names = self.get_session_images()
            self.enhance_data(pattern_names, isgrating)

    def set_outcome(self, outcome_type: str = "state") -> None:
        display(f"Setting outcome to {outcome_type}")
        col_name = f"{outcome_type}_outcome"
        if "outcome" not in self.data.columns:
            self.data = self.data.with_columns(pl.col(col_name).alias("outcome"))
        else:
            if col_name in self.data.columns:
                tmp = self.data[col_name]
                self.data.replace("outcome", tmp)
            else:
                raise ValueError(
                    f"{outcome_type} is not a valid outcome type!!! Try 'wheel' or 'state'."
                )

    def compare_outcomes(self) -> None:
        """Compares the different outcome types and prints a small summary table"""
        out_cols = [c for c in self.data.columns if "_outcome" in c]
        q = (
            self.data.groupby(out_cols)
            .agg([pl.count().alias("count")])
            .sort(["state_outcome"])
        )
        tmp = q.to_pandas()
        print(tabulate(tmp, headers=q.columns))

    def enhance_data(self, pattern_names: dict, isgrating: bool = False) -> None:
        # add a isgrating column
        if "is_grating" not in self.data.columns:
            self.data = pl.concat(
                [
                    self.data,
                    pl.DataFrame({"is_grating": [bool(isgrating)] * len(self.data)}),
                ],
                how="horizontal",
            )
            # add stim type
            self.data = self.data.with_columns(
                (
                    pl.col("spatial_freq").round(2).cast(str)
                    + "cpd_"
                    + pl.col("temporal_freq").cast(str)
                    + "Hz"
                ).alias("stim_type")
            )
            self.data = self.data.with_columns(
                pl.when(pl.col("stim_side") == "ipsi")
                .then((pl.col("contrast") * -1))
                .otherwise(pl.col("contrast"))
                .alias("signed_contrast")
            )

            if len(self.data["opto"].unique().to_numpy()) == 1:
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
                if isinstance(pattern_names, dict):
                    try:
                        # add the pattern name depending on pattern id
                        self.data = self.data.with_columns(
                            pl.struct(["opto_pattern", "state_outcome"])
                            .apply(
                                lambda x: (
                                    pattern_names[x["opto_pattern"]]
                                    if x["state_outcome"] != -1
                                    else None
                                )
                            )
                            .alias("opto_region")
                        )
                    except:
                        raise KeyError(
                            f"Opto pattern not set correctly. You need to change the number at the end of the opto pattern image file to 0!"
                        )
                elif isinstance(pattern_names, str):
                    display(
                        f"{self.data_paths.patternPath} NO OPTO PATTERN DIRECTORY!!"
                    )
                    self.data = self.data.with_columns(
                        pl.struct(["opto_pattern", "state_outcome"])
                        .apply(
                            lambda x: (
                                pattern_names if x["state_outcome"] != -1 else None
                            )
                        )
                        .alias("opto_region")
                    )
                else:
                    raise ValueError(f"Weird pattern name: {pattern_names}")
                # add 'stimkey' from sftf
                self.data = self.data.with_columns(
                    (
                        pl.col("stim_type")
                        + "_"
                        + pl.col("opto_pattern").cast(pl.Int8, strict=False).cast(str)
                    ).alias("stimkey")
                )
                # add stim_label for legends and stuff
                self.data = self.data.with_columns(
                    (pl.col("stim_type") + "_" + pl.col("opto_region")).alias(
                        "stim_label"
                    )
                )

    def get_session_images(self):
        """Reads the related session images(window, pattern,etc)
        Returns a dict with images and also a dict that"""
        if os.path.exists(self.data_paths.patternPath):
            sesh_imgs = {}
            pattern_names = {}
            sesh_patterns = {}
            for im in os.listdir(self.data_paths.patternPath):
                if im.endswith(".tif"):
                    pattern_id = int(im[:-4].split("_")[-1])
                    read_img = tf.imread(pjoin(self.data_paths.patternPath, im))
                    if pattern_id == -1:
                        sesh_imgs["window"] = read_img
                        pattern_names[pattern_id] = "nonopto"
                    else:
                        name = im[:-4].split("_")[-2]
                        pattern_names[pattern_id] = name
                        sesh_imgs[name] = read_img
                elif im.endswith(".bmp"):
                    pattern_id = int(im.split("_")[0])
                    name = im.split("_")[1]
                    read_bmp = np.array(
                        Image.open(pjoin(self.data_paths.patternPath, im))
                    )
                    sesh_patterns[name] = read_bmp[::-1, ::-1]
        else:
            sesh_imgs = None
            pattern_names = None
            sesh_patterns = None
            pattern_names = "*" + self.data_paths.patternPath.split("_")[4] + "*"
        return sesh_imgs, sesh_patterns, pattern_names


class WheelDetectionStats:
    __slots__ = [
        "all_count",
        "early_count",
        "stim_count",
        "correct_count",
        "miss_count",
        "all_correct_percent",
        "hit_rate",
        "easy_hit_rate",
        "false_alarm",
        "nogo_percent",
        "median_response_time",
        "d_prime",
    ]

    def __init__(
        self, dict_in: dict = None, data_in: WheelDetectionData = None
    ) -> None:
        if data_in is not None:
            self.init_from_data(data_in)
        elif dict_in is not None:
            self.init_from_dict(dict_in)

    def __repr__(self):
        rep = """"""
        for k in self.__slots__:
            rep += f"""{k} = {getattr(self,k,None)}\n"""
        return rep

    def init_from_data(self, data_in: WheelDetectionData):
        data = data_in.data
        early_data = data.filter((pl.col("outcome") == -1) & (pl.col("isCatch") == 0))
        stim_data = data.filter((pl.col("outcome") != -1) & (pl.col("isCatch") == 0))
        catch_data = data.filter(pl.col("isCatch") == 1)
        correct_data = data.filter(pl.col("outcome") == 1)
        miss_data = data.filter(pl.col("outcome") == 0)

        # counts
        self.all_count = len(data)
        self.early_count = len(early_data)
        self.stim_count = len(stim_data)
        self.correct_count = len(correct_data)
        self.miss_count = len(miss_data)

        # percents
        self.all_correct_percent = round(100 * self.correct_count / self.all_count, 3)
        self.hit_rate = round(100 * self.correct_count / self.stim_count, 3)
        self.false_alarm = round(
            100 * self.early_count / (self.early_count + self.correct_count), 3
        )
        self.nogo_percent = round(100 * self.miss_count / self.stim_count, 3)

        ## performance on easy trials
        easy_data = data.filter(
            pl.col("contrast").is_in([100, 50])
        )  # earlies can't be easy or hard
        easy_correct_count = len(easy_data.filter(pl.col("outcome") == 1))
        self.easy_hit_rate = round(100 * easy_correct_count / len(easy_data), 3)

        # median response time
        self.median_response_time = round(
            stim_data.filter(pl.col("outcome") == 1)["response_latency"].median(), 3
        )

        # d prime(?)
        self.d_prime = st.norm.ppf(self.hit_rate / 100) - st.norm.ppf(
            self.false_alarm / 100
        )

        # if self.all_trials >= 200:
        #     data200 = data[:200]
        # else:
        #     data200 = data

        # data200_answered = len(data200[data200['answer']!=0])
        # data200_nogo = len(data200[data200['answer']==0])

    def init_from_dict(self, dict_in: dict):
        for k, v in dict_in.items():
            setattr(self, k, v)

    def get_dict(self) -> dict:
        return {key: getattr(self, key, None) for key in self.__slots__}


class WheelDetectionSession(Session):
    def __init__(self, sessiondir, *args, **kwargs):
        super().__init__(sessiondir, *args, **kwargs)

        start = time.time()
        self.trial_list = []
        # add specific data paths
        self.data_paths.metaPath = pjoin(self.data_paths.savePath, "sessionMeta.json")
        self.data_paths.statPath = pjoin(self.data_paths.savePath, "sessionStats.json")
        self.data = WheelDetectionData(self.data_paths)

        if self.isSaved() and self.load_flag:
            self.logger.info(f"Loading from {self.data_paths.savePath}", cml=True)
            self.load_session()
            self.data.set_outcome(kwargs.get("outcome_type", "state"))

        else:
            self.set_meta(**kwargs)
            self.read_data()

            if len(self.rawdata) == 1:
                # this will be the case in opto silencing experiments as there is single runs in sessions
                self.rawdata = self.rawdata[0]
                self.comments = self.comments[0]
            session_data = self.get_session_data()
            # session_data = get_running_stats(session_data)

            g = "grating" in self.data_paths.stimlog

            self.data.set_data(session_data, g)
            self.data.set_outcome(kwargs.get("outcome_type", "state"))
            self.stats = WheelDetectionStats(data_in=self.data)

            if self.meta.water_consumed is not None:
                self.meta.water_per_reward = (
                    self.meta.water_consumed / self.stats.correct_count
                )
            else:
                self.logger.warning("CONSUMED REWARD NOT ENTERED IN GOOGLE SHEET")
                self.meta.water_per_reward = -1

            # add some metadata to the dataframe
            self.data.data = self.data.data.with_columns(
                [
                    pl.lit(self.meta.animalid).alias("animalid"),
                    pl.lit(self.meta.baredate).alias("baredate"),
                ]
            )

            self.data.data = self.data.data.with_columns(
                pl.col("baredate")
                .str.strptime(pl.Date, format="%y%m%d")
                .cast(pl.Date)
                .alias("date")
            )
            self.save_session()
            self.logger.info(f"Saving data to {self.data_paths.savePath}", cml=True)

        end = time.time()
        self.logger.info(f"Done! t={(end-start):.2f} s", cml=True)

    def __repr__(self):
        r = f"Detection Session {self.sessiondir}"
        return r

    def set_meta(self, **kwargs):
        self.meta = SessionMeta(prot_file=self.data_paths.prot, **kwargs)
        self.meta.logversion = self.logversion
        self.meta.set_rig(self.data_paths.prefs)

        self.meta.contrastVector = [
            float(i)
            for i in self.meta.contrastVector.strip("] [").strip(" ").split(",")
        ]

        if hasattr(self.meta, "easyContrast"):
            self.meta.easyContrast = [
                float(i)
                for i in self.meta.easyContrast.strip("] [").strip(" ").split(",")
            ]

        if hasattr(self.meta, "stimRegion"):
            self.meta.stimRegion = [
                float(i)
                for i in self.meta.stimRegion.strip("] [").strip(" ").split(",")
            ]

    @timeit("Saving...")
    def save_session(self) -> None:
        """Saves the session data, meta and stats"""
        self.data.save_data(self.save_mat)

        save_dict_json(self.data_paths.metaPath, self.meta.__dict__)
        self.logger.info("Saved session metadata")

        save_dict_json(self.data_paths.statPath, self.stats.get_dict())
        self.logger.info("Saved session stats")

    @timeit("Loaded all data")
    def load_session(self):
        """Loads the saved session data"""
        meta = load_json_dict(self.data_paths.metaPath)
        self.meta = SessionMeta(init_dict=meta)
        self.logger.info("Loaded session metadata")

        self.data.load_data()
        self.logger.info("Loaded session data")

        stats = load_json_dict(self.data_paths.statPath)
        self.stats = WheelDetectionStats(dict_in=stats)
        self.logger.info("Loaded Session stats")

    def translate_transition(self, oldState, newState) -> dict:
        """
        A function to be called that add the meaning of state transitions into the state DataFrame
        """
        curr_key = "{0}->{1}".format(int(oldState), int(newState))
        state_keys = {
            "0->1": "trialstart",
            "1->2": "cuestart",
            "2->3": "stimstart",
            "2->5": "early",
            "3->4": "hit",
            "3->5": "miss",
            "3->6": "catch",
            "6->0": "trialend",
            "4->6": "stimendcorrect",
            "5->6": "stimendincorrect",
        }

        return state_keys[curr_key]

    def get_session_data(self) -> pl.DataFrame:
        data_to_append = []
        self.rawdata["statemachine"] = self.rawdata["statemachine"].with_columns(
            pl.struct(["oldState", "newState"])
            .apply(lambda x: self.translate_transition(x["oldState"], x["newState"]))
            .alias("transition")
        )
        self.states = self.rawdata["statemachine"]

        self.logger.info(f"Setting global indexing keys for {self.logversion} logging")

        if self.states.shape[0] == 0:
            self.logger.critical(
                "NO STATE MACHINE TO ANALYZE. LOGGING PROBLEMATIC. SOLVE THIS ISSUE FAST!!",
                cml=True,
            )
            return None

        trials = np.unique(self.states["cycle"])
        if len(trials) == 1 and len(self.states) > 6 and self.logversion == "stimpy":
            # this is a failsafe for some early stimpy data where trial count has not been incremented
            self.extract_trial_count()
            trials = np.unique(self.states[self.column_keys["trialNo"]])

        pbar = tqdm(trials, desc="Extracting trial data:", leave=True, position=0)
        for t in pbar:
            temp_trial = WheelDetectionTrial(int(t), meta=self.meta, logger=self.logger)
            temp_trial.get_data_slices(self.rawdata)
            trial_row = temp_trial.trial_data_from_logs()
            if trial_row["state_outcome"] is not None:
                self.trial_list.append(temp_trial)
                if t == 1:
                    data_to_append = {k: [v] for k, v in trial_row.items()}
                else:
                    for k, v in trial_row.items():
                        data_to_append[k].append(v)
            else:
                if t != len(trials):
                    self.logger.critical("NO TRIAL DATA")
                else:
                    self.logger.info("Discarded the last trial...")

            pbar.update()

        self.logger.set_msg_prefix("session")
        session_data = pl.DataFrame(data_to_append)

        # add a stim_side column for ease of access
        session_data = session_data.with_columns(
            pl.when(pl.col("stim_pos") > 0)
            .then(pl.lit("contra"))
            .when(pl.col("stim_pos") < 0)
            .then(pl.lit("ipsi"))
            .when(pl.col("stim_pos") == 0)
            .then(pl.lit("catch"))
            .otherwise(None)
            .alias("stim_side")
        )

        # add easy/hard contrast type groups
        session_data = session_data.with_columns(
            pl.when(pl.col("contrast") >= 25)
            .then(pl.lit("easy"))
            .when((pl.col("contrast") < 25) & (pl.col("contrast") > 0))
            .then(pl.lit("hard"))
            .when(pl.col("contrast") == 0)
            .then(pl.lit("catch"))
            .otherwise(None)
            .alias("contrast_type")
        )

        # add contrast titration boolean
        uniq_stims = nonan_unique(session_data["contrast"].to_numpy())
        isTitrated = 0
        if len(uniq_stims) > len(self.meta.contrastVector):
            isTitrated = 1
        session_data = session_data.with_columns(
            [pl.lit(isTitrated).cast(pl.Boolean).alias("isTitrated")]
        )

        # round sf and tf
        session_data = session_data.with_columns(
            [
                (pl.col("spatial_freq").round(2).alias("spatial_freq")),
                (pl.col("temporal_freq").round(1).alias("temporal_freq")),
            ]
        )

        if session_data.is_empty():
            self.logger.error("THERE IS NO SESSION DATA !!!", cml=True)
            return None
        else:
            return session_data


@timeit("Getting rolling averages...")
def get_running_stats(data_in: pd.DataFrame, window_size: int = 20) -> pd.DataFrame:
    """Gets the running statistics of certain columns"""

    data_in = data_in.with_columns(
        pl.col("response_latency")
        .rolling_median(window_size)
        .alias("running_response_latency")
    )
    # answers
    outcomes = {"correct": 1, "nogo": 0, "early": -1}

    for k, v in outcomes.items():
        key = "fraction_" + k
        data_arr = data_in["state_outcome"].to_numpy()
        data_in[key] = get_fraction(data_arr, fraction_of=v)

    return data_in


def main():
    from argparse import ArgumentParser
    import cProfile, pstats
    from io import StringIO

    parser = ArgumentParser(description="Wheel Detection Session Analysis")

    parser.add_argument(
        "expname",
        metavar="expname",
        type=str,
        help="Experiment filename (e.g. 200325_KC020_wheel_KC)",
    )
    parser.add_argument(
        "-l",
        "--load",
        metavar="load_flag",
        default=True,
        type=str,
        help="Flag for loading existing data",
    )

    opts = parser.parse_args()
    expname = opts.expname
    load_flag = opts.load

    profiler = cProfile.Profile()
    profiler.enable()
    print(load_flag)
    w = WheelDetectionSession(sessiondir=expname, load_flag=False)

    profiler.disable()
    s = StringIO
    stats = pstats.Stats(profiler, stream=s)
    stats.strip_dirs()
    stats.sort_stats("cumtime")
    stats.dump_stats(w.data_paths.analysisPath + os.sep + "profile.prof")
    # stats.print_stats()


if __name__ == "__main__":
    main()
