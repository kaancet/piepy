import polars as pl
import scipy.stats as st

from piepy.core.run import Run
from piepy.core.session import Session
from piepy.core.registry import register_paradigm
from piepy.core.hub import generate_unique_session_id
from piepy.core.utils import safe_ratio, safe_median
from piepy.psychophysics.opto import add_opto_pattern_columns
from piepy.psychophysics.transforms import (
    add_runno,
    add_rig_response_time,
    add_stim_side,
    add_sftf_descriptor,
)
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


# Here are more task specific augmenters
def add_contrast_descriptors(df: pl.DataFrame) -> pl.DataFrame:
    """Contrast columns: ``signed_contrast`` (by side) and easy/hard/catch ``contrast_type``."""
    df = df.with_columns(
        pl.when(pl.col("stim_side") == "ipsi")
        .then(pl.col("contrast") * -1)
        .otherwise(pl.col("contrast"))
        .alias("signed_contrast")
    )
    return df.with_columns(
        pl.when(pl.col("contrast") >= 0.25)
        .then(pl.lit("easy"))
        .when((pl.col("contrast") < 0.25) & (pl.col("contrast") > 0))
        .then(pl.lit("hard"))
        .when(pl.col("contrast") == 0)
        .then(pl.lit("catch"))
        .otherwise(None)
        .alias("contrast_type")
    )


class WheelDetectionRun(Run):
    trial_handler_cls = WheelDetectionTrialHandler
    state_transitions = STATE_TRANSITION_KEYS

    def __repr__(self):
        _base = super().__repr__()
        _stats = ""
        if self.stats is not None:
            hr = self.stats.get("hit_rate")
            fa = self.stats.get("false_alarm_rate")
            _stats = f"- HR={hr}% - FA={fa}"
        return _base + _stats

    def augment_data(self) -> None:
        # all detection column derivation, in order (contrast needs stim_side; opto needs the
        # per-run pattern path on self) -- composed from pure transforms.
        d = self.data.data
        d = add_runno(d, self.run_no)
        d = add_stim_side(d)
        d = add_contrast_descriptors(d)
        d = add_sftf_descriptor(d)
        d = add_rig_response_time(d)
        d = add_opto_pattern_columns(d, self.paths.opto_pattern)
        self.data.data = d

    def enrich_data(self) -> pl.DataFrame:
        """Join per-run detection stats + session metadata onto the concatenated table.

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
            **_detection_per_run(self, d, self),
        }

        add = pl.DataFrame([_enrich]).with_columns(pl.col("run_no").cast(pl.UInt32))
        self.data.data = d.join(add, on="run_no", how="left")

    def compute_stats(self) -> dict:
        return get_run_stats(self.data.data)


class WheelDetectionSession(Session):
    run_cls = WheelDetectionRun

    def __repr__(self):
        return f"Detection Session {self.sessiondir}"


def get_run_stats(data: pl.DataFrame) -> dict:
    """Gets run stats from run dataframe

    Args:
        data (pl.DataFrame): Data of the session to calculate the summary stats of

    Returns:
        dict: Summary statistics as a dictionary
    """
    stats_dict = {}
    early_data = data.filter((pl.col("outcome") == "early"))
    stim_data = data.filter((pl.col("outcome") != "early") & (pl.col("isCatch") == 0))
    catch_data = data.filter((pl.col("outcome") != "early") & (pl.col("isCatch") == 1))
    correct_data = stim_data.filter(pl.col("outcome") == "hit")
    miss_data = stim_data.filter(pl.col("outcome") == "miss")
    nonopto_data = stim_data.filter(pl.col("opto") == 0)
    opto_data = stim_data.filter(pl.col("opto") == 1)

    total = len(data)

    # counts #
    stats_dict["total_trial_count"] = total
    stats_dict["early_trial_count"] = len(early_data)
    stats_dict["stim_trial_count"] = len(stim_data)
    stats_dict["correct_trial_count"] = len(correct_data)
    stats_dict["miss_trial_count"] = len(miss_data)
    stats_dict["catch_trial_count"] = len(catch_data)
    stats_dict["opto_trial_count"] = len(opto_data)
    stats_dict["opto_ratio"] = safe_ratio(len(opto_data), total)

    # rates #
    nonopto_correct_count = len(nonopto_data.filter(pl.col("outcome") == "hit"))
    stats_dict["nonopto_hit_rate"] = safe_ratio(nonopto_correct_count, len(nonopto_data))

    stats_dict["correct_rate"] = safe_ratio(len(correct_data), total)
    stats_dict["hit_rate"] = safe_ratio(len(correct_data), len(stim_data))
    stats_dict["false_alarm_rate"] = safe_ratio(len(early_data), total)
    stats_dict["nogo_rate"] = safe_ratio(len(miss_data), len(stim_data))

    # median response time #
    hit_nonopto = nonopto_data.filter(pl.col("outcome") == "hit")
    stats_dict["median_response_time"] = safe_median(
        hit_nonopto["state_response_time"]
        if len(hit_nonopto)
        else pl.Series(dtype=pl.Float64)
    )

    # median reaction time
    stats_dict["median_reaction_time"] = safe_median(
        hit_nonopto["reaction_time"] if len(hit_nonopto) else pl.Series(dtype=pl.Float64)
    )

    # d prime(?) #
    # d_prime with 1/(2N) correction to keep it finite at 0% and 100%
    hr = stats_dict["hit_rate"]
    far = stats_dict["false_alarm_rate"]
    if hr is not None and far is not None and len(stim_data) > 0:
        n_stim = len(stim_data)
        clipped_hr = max(1 / (2 * n_stim), min(1 - 1 / (2 * n_stim), hr / 100))
        clipped_far = max(1 / (2 * total), min(1 - 1 / (2 * total), far / 100))
        stats_dict["d_prime"] = st.norm.ppf(clipped_hr) - st.norm.ppf(clipped_far)
    else:
        stats_dict["d_prime"] = None

    ## performance on easy trials
    easy_data = nonopto_data.filter(pl.col("contrast").is_in([1.0, 0.5]))
    stats_dict["easy_trial_count"] = len(easy_data)
    easy_correct = easy_data.filter(pl.col("outcome") == "hit")
    if stats_dict["easy_trial_count"]:
        stats_dict["easy_hit_rate"] = safe_ratio(
            len(easy_correct), stats_dict["easy_trial_count"]
        )
        stats_dict["easy_median_response_time"] = safe_median(
            easy_correct["state_response_time"]
            if len(easy_correct)
            else pl.Series(dtype=pl.Float64)
        )
        stats_dict["easy_median_reaction_time"] = safe_median(
            easy_correct["reaction_time"]
            if len(easy_correct)
            else pl.Series(dtype=pl.Float64)
        )
    else:
        stats_dict["easy_hit_rate"] = -1
        stats_dict["easy_median_response_time"] = -1

    return stats_dict


def _detection_per_run(run, d, session) -> dict:
    """Derived/computed cohort columns for detection (the non-boilerplate part of enrich)."""
    meta = run.meta or {}
    opts = meta.get("opts") or {}
    params = meta.get("params")  # a pandas DataFrame (from parse_protocol); not a dict
    rig = meta.get("rig")
    contrast_vector = opts.get("contrastVector", []) or []
    n_uniq_contrast = d["contrast"].drop_nulls().unique().len()
    try:
        stim_size = float(params["width"][0])
    except Exception:  # noqa: BLE001 - width may be absent / shaped differently
        stim_size = None
    return {
        "opto_targets": d["opto_pattern"].unique().len() - 1,
        "stimulus_count": d["stim_type"].drop_nulls().unique().len(),
        "stim_combination": "+".join(
            d["stim_type"].unique().sort().drop_nulls().to_list()
        ),
        "isTitrated": n_uniq_contrast > len(contrast_vector),
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
        "stim_size": stim_size,
        "sf_values": (
            d["sf"].drop_nulls().unique().to_list() if "sf" in d.columns else []
        ),
        "tf_values": (
            d["tf"].drop_nulls().unique().to_list() if "tf" in d.columns else []
        ),
    }


register_paradigm("wheel_detection", WheelDetectionSession)
