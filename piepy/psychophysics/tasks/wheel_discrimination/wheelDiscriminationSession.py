import polars as pl

from piepy.core.run import Run
from piepy.core.session import Session
from piepy.core.registry import register_paradigm
from piepy.core.hub import generate_unique_session_id
from piepy.core.utils import safe_ratio, safe_median
from piepy.core.log_repair_functions import fix_first_line_state_logging
from piepy.psychophysics.opto import add_opto_pattern_columns
from piepy.psychophysics.transforms import add_runno
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


def add_choice_descriptors(df: pl.DataFrame) -> pl.DataFrame:
    """Choice columns: ``target_side``, ``right_choice``, and a ``response_time`` copy."""
    df = df.with_columns(
        pl.when(pl.col("target_pos").list.get(0) > 0)
        .then(pl.lit("contra"))
        .when(pl.col("target_pos").list.get(0) < 0)
        .then(pl.lit("ipsi"))
        .otherwise(None)
        .alias("target_side")
    )
    df = df.with_columns(
        pl.col("state_outcome")
        .cast(pl.Boolean)
        .xor(pl.col("correct_side").cast(pl.Boolean))
        .not_()
        .cast(pl.Int64)
        .alias("right_choice")
    )
    return df.with_columns(pl.col("state_response_time").alias("response_time"))


def add_stim_diff_and_type(df: pl.DataFrame, discrim_of: str) -> pl.DataFrame:
    """Add the discriminated-feature difference + a ``stim_type`` key (needs ``discrim_of``)."""
    unit = {"width": "deg", "sf": "cpd", "tf": "Hz", "contrast": "%"}.get(
        discrim_of, "NA"
    )
    df = df.with_columns(pl.lit(discrim_of).alias("discriminating"))
    df = df.with_columns(
        pl.when(pl.col("target_side") == "contra")
        .then(pl.col(f"target_{discrim_of}") - pl.col(f"distract_{discrim_of}"))
        .otherwise(pl.col(f"distract_{discrim_of}") - pl.col(f"target_{discrim_of}"))
        .alias(f"diff_{discrim_of}")
    )
    return df.with_columns(
        (
            pl.col(f"target_{discrim_of}").cast(pl.Utf8)
            + f"{unit}_"
            + pl.col(f"distract_{discrim_of}").cast(pl.Utf8)
            + f"{unit}"
        ).alias("stim_type")
    )


class WheelDiscriminationRun(Run):
    trial_handler_cls = WheelDiscriminationTrialHandler
    state_transitions = STATE_TRANSITION_KEYS

    def __repr__(self):
        _base = super().__repr__()
        _stats = ""
        if self.stats is not None:
            cr = self.stats.get("correct_rate")
            _stats = f"- CR={cr}%"
        return _base + _stats

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
        attend_name = self.meta["opts"]["AttendVectorName"]  # noqa: F841
        distract_name = self.meta["opts"]["DistractVectorName"]
        if distract_name == "c":
            realtf_cols = [rc for rc in header if "realtf" in rc]

            if len(realtf_cols) != 0:
                # get all the columns with _r and _l
                l_headers = [c for c in header if "_l" in c if "pos" not in c]
                lr_headers = [
                    (i, c.split("_")[0]) for i, c in enumerate(header) if c in l_headers
                ]

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
        elif distract_name == "ori":
            # hardcoded column for ori
            new_header = [
                "code",
                "presentTime",
                "iTrial",
                "photo",
                "width_l",
                "width_r",
                "posx_l",
                "posx_r",
                "sf_l",
                "sf_r",
                "ori_l",
                "ori_r",
                "tf_l",
                "tf_r",
                "correct",
                "reward",
                "fraction_r",
                "prob",
            ]

            # later added columns
            _extra_cols = header[len(new_header) :]
            new_header = new_header + _extra_cols
            df = df.rename({h: new_header[i] for i, h in enumerate(df.columns)})

        return df

    def augment_data(self) -> None:
        # all discrimination column derivation, in order; context (attended feature, opto path)
        discrim_of = self.meta["opts"]["AttendVectorName"]
        d = self.data.data
        d = add_runno(d, self.run_no)
        d = add_choice_descriptors(d)
        d = add_stim_diff_and_type(d, discrim_of=discrim_of)
        d = add_opto_pattern_columns(d, self.paths.opto_pattern)
        self.data.data = d

    def enrich_data(self) -> pl.DataFrame:
        """Join per-run discrimination stats + session metadata onto the concatenated table.

        One row per run: ``stat_*`` from ``get_run_stats``, a few meta/opts fields, and the derived
        columns from ``_detection_per_run`` -- left-joined on ``run_no``. (``df`` is already
        concatenated by :meth:`Session.analyze`; this never concatenates.)
        """
        d = self.data.data

        if d is None or d.is_empty():
            return

        meta = self.meta or {}
        opts = meta.get("opts") or {}

        _enrich = {
            "run_no": self.run_no,
            **{f"stat_{k}": v for k, v in get_run_stats(d).items()},
            "level": meta.get("level"),
            "run_start_time": meta.get("run_start_time"),
            "task": opts.get("controller"),
            "opto_ratio": opts.get("optoRatio"),
            "wait_window": opts.get("openStimDuration"),
            "response_window": opts.get("closedStimDuration"),
            **_discrimination_per_run(self, d, self),
        }

        add = pl.DataFrame([_enrich]).with_columns(pl.col("run_no").cast(pl.UInt32))
        self.data.data = d.join(add, on="run_no", how="left")

    def compute_stats(self) -> dict:
        return get_run_stats(self.data.data)


class WheelDiscriminationSession(Session):
    run_cls = WheelDiscriminationRun

    def __repr__(self):
        return f"Discrimination Session {self.sessiondir}"

    def analyze(self, load_flag: bool = False, save_mat: bool = False) -> pl.DataFrame:
        """The analysis-ready trial table for this session, paradigm is already set for discrimination

        This is what users (and the Hub) call. ``concatenate_runs`` is the structural step (stack
        runs on one clock)

        Args:
            load_flag (bool, optional):  flag to either load previously parsed data or to parse it again. Defaults to False
            save_mat (bool, optional):   flag to make the parser also output a .mat file to be used in MATLAB scripts. Defaults to False

        Returns:
            pl.DataFrame: Concatenated session data
        """
        return super().analyze(
            "wheel_discrimination", load_flag=load_flag, save_mat=save_mat
        )


def get_run_stats(data: pl.DataFrame) -> dict:
    """Per-run summary statistics for wheel-discrimination.

    Safe against edge cases: empty subsets return None for rates/medians.
    Uses outcome == "correct" (not "hit") for discrimination.
    """
    stats_dict = {}
    correct_data = data.filter(pl.col("outcome") == "correct")
    miss_data = data.filter(pl.col("outcome") == "incorrect")
    nonopto_data = data.filter(pl.col("opto") == 0)
    opto_data = data.filter(pl.col("opto") == 1)

    total = len(data)

    # counts #
    stats_dict["total_trial_count"] = total
    stats_dict["correct_trial_count"] = len(correct_data)
    stats_dict["miss_trial_count"] = len(miss_data)
    stats_dict["opto_trial_count"] = len(opto_data)
    stats_dict["opto_ratio"] = safe_ratio(len(opto_data), total)

    # rates #
    nonopto_correct_count = len(nonopto_data.filter(pl.col("outcome") == "correct"))
    stats_dict["nonopto_correct_rate"] = safe_ratio(
        nonopto_correct_count, len(nonopto_data)
    )

    stats_dict["correct_rate"] = safe_ratio(len(correct_data), total)

    # median response time #
    correct_nonopto = nonopto_data.filter(pl.col("outcome") == "correct")
    stats_dict["median_response_time"] = safe_median(
        correct_nonopto["state_response_time"]
        if len(correct_nonopto)
        else pl.Series(dtype=pl.Float64)
    )

    return stats_dict


def _discrimination_per_run(run, d, session) -> dict:
    """Derived/computed cohort columns for detection (the non-boilerplate part of enrich)."""
    meta = run.meta or {}
    opts = meta.get("opts") or {}  # a pandas DataFrame (from parse_protocol); not a dict
    rig = meta.get("rig")
    contrast_vector = (
        opts.get("contrastVector", []) or []
    )  # noqa: BLE001 - width may be absent / shaped differently

    return {
        "opto_targets": d["opto_pattern"].unique().len() - 1,
        "stimulus_count": d["stim_type"].drop_nulls().unique().len(),
        "stim_combination": "+".join(
            d["stim_type"].unique().sort().drop_nulls().to_list()
        ),
        "rig": rig.get("name") if isinstance(rig, dict) else rig,
        "session_id": generate_unique_session_id(
            meta.get("baredate", ""), meta.get("animalid", "")
        ),
        "area": meta.get("area"),
        "opto_power": meta.get("opto_power"),
        "imaging": meta.get("imaging"),
        "user": meta.get("user"),
        "isCNO": meta.get("isCNO"),
        "contrast_vector": list(contrast_vector),
    }


# default enrich (None) -> the generic Hub uses Session.concatenate_runs; a richer detection-style
# enrich hook can be added here later if discrimination needs cohort stat_* columns.
register_paradigm("wheel_discrimination", WheelDiscriminationSession)
