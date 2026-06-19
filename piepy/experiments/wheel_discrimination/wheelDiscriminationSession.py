import polars as pl

from piepy.core.run import Run
from piepy.core.session import Session
from piepy.core.registry import register_paradigm
from piepy.core.log_repair_functions import fix_first_line_state_logging
from piepy.psychophysics.opto import OptoPattern
from piepy.psychophysics.psychophysicalRunData import PsychophysicalRunData
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


class WheelDiscriminationRunData(OptoPattern, PsychophysicalRunData):
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
        self.data = self.data.with_columns(pl.col("state_response_time").alias("response_time"))

        # round sf and tf
        self.data = self.data.with_columns(
            [pl.col(_sf).round(2).alias(_sf) for _sf in self.data.columns if _sf.endswith("_sf")]
        )
        self.data = self.data.with_columns(
            [pl.col(_tf).round(2).alias(_tf) for _tf in self.data.columns if _tf.endswith("_tf")]
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
            .otherwise(pl.col(f"distract_{discrim_of}") - pl.col(f"target_{discrim_of}"))
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


class WheelDiscriminationRun(Run):
    rundata_cls = WheelDiscriminationRunData
    trial_handler_cls = WheelDiscriminationTrialHandler
    state_transitions = STATE_TRANSITION_KEYS

    def repair_rawdata(self) -> None:
        """Discrimination-specific rawdata fixes after the standard read."""
        self.rawdata = fix_first_line_state_logging(self.rawdata)
        self.rawdata["vstim"] = self.transform_header(self.rawdata["vstim"])

    def transform_header(self, df: pl.DataFrame) -> pl.DataFrame:
        """Changes the vstim header

        Args:
            in_df (pl.DataFrame): vstim dataframe

        Returns:
            pl.DataFrame: _description_
        """
        header = df.columns.copy()
        distract_name = self.meta["opts"]["DistractVectorName"]
        if distract_name == "c":
            distract_name = "contrast"

        realtf_cols = [rc for rc in header if "realtf" in rc]

        if len(realtf_cols) != 0:
            # get all the columns with _r and _l
            l_headers = [c for c in header if "_l" in c if "pos" not in c]
            lr_headers = [(i, c.split("_")[0]) for i, c in enumerate(header) if c in l_headers]

            for j, head_tup in enumerate(lr_headers):
                h_pos, h_name = head_tup
                if h_name == "contrast":
                    header[h_pos] = "width_l"
                    header[h_pos + 1] = "width_r"
                elif h_name == "tf":
                    header[h_pos] = "contrast_l"
                    header[h_pos + 1] = "contrast_r"
                elif h_name == "realtf":
                    header[h_pos] = "tf_l"
                    header[h_pos + 1] = "tf_r"

            df = df.rename({h: header[i] for i, h in enumerate(df.columns)})

            return df

    def augment_data(self) -> None:
        discrim_of = self.meta["opts"]["AttendVectorName"]
        self.data.add_stim_diff_and_type(discrim_of=discrim_of)
        self.data.add_pattern_related_columns(self.paths.opto_pattern)

    def compute_stats(self) -> dict:
        return get_run_stats(self.data.data)


class WheelDiscriminationSession(Session):
    run_cls = WheelDiscriminationRun

    def __repr__(self):
        return f"Discrimination Session {self.sessiondir}"


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
    stats_dict["opto_ratio"] = round(100 * stats_dict["opto_trial_count"] / stats_dict["total_trial_count"], 3)

    # rates #
    nonopto_correct_count = len(nonopto_data.filter(pl.col("outcome") == "hit"))
    stats_dict["nonopto_hit_rate"] = round(100 * nonopto_correct_count / len(nonopto_data), 3)

    stats_dict["correct_rate"] = round(100 * stats_dict["correct_trial_count"] / stats_dict["total_trial_count"], 3)

    # median response time #
    stats_dict["median_response_latency "] = round(
        nonopto_data.filter(pl.col("outcome") == "correct")["state_response_time"].median(),
        3,
    )

    return stats_dict


# default enrich (None) -> the generic Hub uses Session.concatenate_runs; a richer detection-style
# enrich hook can be added here later if discrimination needs cohort stat_* columns.
register_paradigm("discrimination", WheelDiscriminationSession)
