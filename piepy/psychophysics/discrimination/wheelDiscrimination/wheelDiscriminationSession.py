import time
import scipy.stats as st
from os.path import join as pjoin

from ....core.run import *
from ....core.pathfinder import *
from ....core.config import config
from ....core.session import Session
from ....core.log_repair_functions import *
from .wheelDiscriminationTrial import WheelDiscriminationTrialHandler


STATE_TRANSITION_KEYS = {
    "0->1": "trialstart",
    "1->2": "stimstart",
    "2->3": "responsestart",
    "2->5": "",
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


class WheelDiscriminationRun(Run):
    def __init__(self, path: Paths) -> None:
        super().__init__(path)
        self.data = WheelDiscriminationRunData()
        self.trial_handler = WheelDiscriminationTrialHandler()

    def __repr__(self):
        _base = super().__repr__()
        return _base

    def analyze_run(self, transform_dict: dict) -> None:
        """ """
        super().analyze_run(transform_dict)

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
        display(f"Done! t={(end-start):.2f} s")

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
