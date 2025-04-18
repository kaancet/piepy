import time
import polars as pl
from ...core.io import display
from ...core.run import Run, RunData, RunMeta
from ...core.pathfinder import Paths
from ...core.session import Session
from ...core.log_repair_functions import extract_trial_count, add_total_iStim
from .visualTrial import VisualTrialHandler


STATE_TRANSITION_KEYS = {
    "0->1": "trialstart",
    "1->2": "stimstart",
    "2->0": "stimtrialend",
    "2->3": "stimend",
    "3->0": "trialend",
}


class VisualRunData(RunData):
    def __init__(self, data: pl.DataFrame = None):
        super().__init__(data)


class VisualRun(Run):
    def __init__(self, paths: Paths):
        super().__init__(paths)
        self.data = VisualRunData()
        self.trial_handler = VisualTrialHandler()

    def read_run_data(self) -> None:
        """Add experiment specific modifiactins to read data in this method"""
        super().read_run_data()

        # sometimes iTrial starts from 0, shift all to start from 1
        if self.rawdata["vstim"]["iTrial"].drop_nulls()[0] == 0:
            self.rawdata["vstim"] = self.rawdata["vstim"].with_columns(
                (pl.col("iTrial") + 1).alias("iTrial")
            )

    def analyze_run(self, transform_dict: dict) -> None:
        """Main loop to extract data from rawdata

        Args:
            transform_dict (dict): Dictionary that has state transition and state name dictionary (like the one defined above)
        """
        self.read_run_data()
        self.translate_state_changes(transform_dict)

        self.rawdata = extract_trial_count(self.rawdata)
        # add total iStim just in case
        self.rawdata = add_total_iStim(self.rawdata)

        ## adding fake timing differences to trialstart and end, because the previous way of doing things are fucked
        self.rawdata["statemachine"] = self.rawdata["statemachine"].with_columns(
            pl.when(pl.col("transition") == "trialstart")
            .then(pl.col("elapsed") + 300)
            .when(pl.col("transition") == "trialend")
            .then(pl.col("elapsed") + 200)
            .otherwise(pl.col("elapsed"))
            .alias("elapsed")
        )

        run_data = self.get_trials()

        # set the data object
        self.data.set_data(run_data)


class VisualSession(Session):
    def __init__(self, sessiondir, load_flag=False, save_mat=False):
        start = time.time()
        super().__init__(sessiondir, load_flag, save_mat)

        # initialize runs : read and parse or load the data
        self.init_session_runs()

        end = time.time()
        display(f"Done! t={(end - start):.2f} s")

    def init_session_runs(self, skip_google=True):
        """Initializes runs in a session

        Args:
            skip_google (bool, optional): Whether to skip reading data from google sheet. Defaults to True.
        """
        for r in range(self.run_count):
            _path = Paths(self.paths.all_paths, r)
            # meta data
            _meta = RunMeta.get_meta(_path, skip_google)

            # the run itself
            _run = VisualRun(_path)
            if _run.is_run_saved() and self.load_flag:
                display(f"Loading from {_run.paths.save}")
                _run.load_run()
            else:
                _run.analyze_run(STATE_TRANSITION_KEYS)
                _run.data.add_metadata_columns(_meta)
                _run.save_run()

            self.runs.append(_run)
            self.metas.append(_meta)
