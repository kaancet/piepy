import polars as pl

from ....core.run import Run
from ....core.session import Session
from ....core.registry import register_paradigm
from .visualTrial import VisualTrialHandler

STATE_TRANSITION_KEYS = {
    "0->1": "trialstart",
    "1->2": "stimstart",
    "2->0": "stimtrialend",
    "2->3": "stimend",
    "3->0": "trialend",
}


class VisualRun(Run):
    # Paradigm wiring: the base Run builds the handler, reads the rawdata, and runs the parse loop;
    # this subclass only supplies the two visual-specific fixes below.
    trial_handler_cls = VisualTrialHandler
    state_transitions = STATE_TRANSITION_KEYS

    def read_run_data(self) -> None:
        """Standard read, plus a visual-specific fix: some logs number trials from 0, not 1."""
        super().read_run_data()
        if self.rawdata["vstim"]["iTrial"].drop_nulls()[0] == 0:
            self.rawdata["vstim"] = self.rawdata["vstim"].with_columns(
                (pl.col("iTrial") + 1).alias("iTrial")
            )

    def repair_rawdata(self) -> None:
        """Shift the trial start/end times by fixed offsets so the downstream timing lines up.

        This runs after the state transitions have been named, so it can match "trialstart" and
        "trialend". The offsets are a quirk of the visual rig's logging, kept from the original
        pipeline.
        """
        self.rawdata["statemachine"] = self.rawdata["statemachine"].with_columns(
            pl.when(pl.col("transition") == "trialstart")
            .then(pl.col("elapsed") + 300)
            .when(pl.col("transition") == "trialend")
            .then(pl.col("elapsed") + 200)
            .otherwise(pl.col("elapsed"))
            .alias("elapsed")
        )
    
    def analyze_run(self):
        super().analyze_run()
        self._extract_list_columns()


    def _extract_list_columns(self) -> pl.DataFrame:
        """Extracts the element in single element list columns"""
        # all list-typed columns
        list_cols = [c for c in self.data.data.columns if self.data.data.schema[c].base_type() == pl.List]

        # check lengths in a single pass
        max_lens = self.data.data.select(pl.col(c).list.len().max().alias(c) for c in list_cols)

        # keep only columns where the longest list is 1 element
        single_element_cols = [c for c in list_cols if max_lens[c].item() == 1]

        self.data.data = self.data.data.with_columns(pl.col(c).list.first() for c in single_element_cols)


class VisualSession(Session):
    run_cls = VisualRun

    def analyze(self, load_flag: bool = False, save_mat: bool = False) -> pl.DataFrame:
        """Parse (or load) every run and return the concatenated visual trial table.

        Args:
            load_flag: reuse a previous parse if one is saved, instead of parsing again.
            save_mat: also write a MATLAB ``.mat`` copy.
        """
        return super().analyze("visual", load_flag=load_flag, save_mat=save_mat)


register_paradigm("visual", VisualSession)
