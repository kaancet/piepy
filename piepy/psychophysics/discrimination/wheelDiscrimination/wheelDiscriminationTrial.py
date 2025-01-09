import polars as pl
import patito as pt
from typing import Literal

from ...wheelTrace import WheelTrace
from ....sensory.visual.visualTrial import VisualTrial, VisualTrialHandler
from ...psychophysicalTrial import PsychophysicalTrial, PsychophysicalTrialHandler

OUTCOMES = {0: "incorrect", 1: "correct"}


class WheelDiscriminationTrial(VisualTrial, PsychophysicalTrial):
    outcome: Literal["incorrect", "correct"]
    wheel_t: list[float] = pt.Field(default=[], dtype=pl.List(pl.Float64))
    wheel_pos: list[int] = pt.Field(default=[], dtype=pl.List(pl.Int64))


class WheelDiscriminationTrialHandler(VisualTrialHandler, PsychophysicalTrialHandler):
    def __init__(self):
        super().__init__()
        self._trial = {k: None for k in WheelDiscriminationTrial.columns}
        self.was_screen_off = True  # flag for not having OFF pulse in screen data
        self.set_model(WheelDiscriminationTrial)

    def get_trial(
        self, trial_no: int, rawdata: dict, return_as="dict"
    ) -> pt.DataFrame | dict | list | None:
        """Main function that is called from outside, sets the trial, validates data type and returns it"""
        self.init_trial()
        is_trial_set = self.set_trial(trial_no, rawdata)

        if not is_trial_set:
            return None
        else:
            self.set_screen_events()  # should return a 2x2 matrix, first column is timings for screen ON and OFF.
            self.sync_timeframes()  # syncs the state and vstim log times, using screen ONSET

            # NOTE: sometimes due to state machine logic, the end of trial will be end of stimulus
            # this causes a trial to have a single screen event (ON) to be parsed into a given trial
            # to remedy this, we check the screen data after syncing the timeframes of rig(arduoino) and statemachine(python)
            if not self.was_screen_off:
                self.recheck_screen_events(rawdata["screen"])
            self.set_vstim_properties()  # should be run after sync_timeframes

            self.set_state_events()  # should be run after sync_timeframes, needs the corrected time columns
            self.set_outcome()
            self.set_licks()
            self.set_reward()
            self.set_opto()
            self.set_wheel_traces(self._trial["t_vstimstart_rig"])

            return self._update_and_return(return_as)

    def set_state_events(self) -> None:
        """Goes over the transitions to set state based timings and also sets the state_outcome"""

        self._trial["t_stimstart"] = self.data["state"].filter(
            pl.col("transition") == "stimstart"
        )[0, "corrected_elapsed"]

        correct = self.data["state"].filter(pl.col("transition") == "correct")
        if len(correct):
            self._trial["state_outcome"] = 1
            self._trial["state_response_time"] = correct[0, "stateElapsed"]

        incorrect = self.data["state"].filter(pl.col("transition") == "incorrect")
        if len(incorrect):
            self._trial["state_outcome"] = 0
            self._trial["state_response_time"] = incorrect[0, "stateElapsed"]

        stim_end = self.data["state"].filter(pl.col("transition").str.contains("stimend"))
        if len(stim_end):
            self._trial["t_stimend"] = stim_end[0, "corrected_elapsed"]

        trial_end = self.data["state"].filter(
            pl.col("transition").str.contains("trialend")
        )
        if len(trial_end):
            self._trial["t_trialend"] = trial_end[0, "corrected_elapsed"]

    def set_vstim_properties(self):
        """Overwrites the visualTrialHandler method to extract the relevant vstim properties"""
        super().set_vstim_properties()

        self._trial["prob"] = self._trial["prob"][0][0]
        self._trial["fraction_r"] = self._trial["fraction_r"][0][0]

    def set_wheel_traces(self, reset_time_point: float) -> None:
        """Sets the wheel"""
        wheel_array = self._get_rig_event("position")
        if wheel_array is not None:
            t = wheel_array[:, 0]
            pos = wheel_array[:, 1]
            WheelTrace.init_interpolator(t, pos)
            self._trial["wheel_t"] = [
                WheelTrace().reset_time_frame(t, reset_time_point).tolist()
            ]
            self._trial["wheel_pos"] = [WheelTrace().reset_position(pos, 0).tolist()]

    def set_outcome(self) -> None:
        """Sets the trial outcome by using the integer state outcome value"""
        if self._trial["state_outcome"] is not None:
            self._trial["outcome"] = OUTCOMES[self._trial["state_outcome"]]
