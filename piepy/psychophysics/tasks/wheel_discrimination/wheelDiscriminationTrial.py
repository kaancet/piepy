import polars as pl
import patito as pt
from typing import Literal

from piepy.psychophysics.wheelTrace import WheelTrace, match_response_movement
from piepy.sensory.visual.visualTrial import VisualTrial, VisualTrialHandler
from piepy.psychophysics.psychophysicalTrial import (
    PsychophysicalTrial,
    PsychophysicalTrialHandler,
)

OUTCOMES = {0: "incorrect", 1: "correct"}

_GAP_TOL_MS = 100.0
_MIN_RT_MS = 150.0


class WheelDiscriminationTrial(VisualTrial, PsychophysicalTrial):
    outcome: Literal["incorrect", "correct"]
    wheel_t: list[float] = pt.Field(default=[], dtype=pl.List(pl.Float64))
    wheel_pos: list[int] = pt.Field(default=[], dtype=pl.List(pl.Int64))
    reaction_time: float | None = pt.Field(default=None, dtype=pl.Float64)
    # provenance of reaction_time: "contain" | "gap" | "none" | None (not computed)
    reaction_time_source: str | None = pt.Field(default=None, dtype=pl.Utf8)
    anticipatory: bool | None = pt.Field(default=None, dtype=pl.Boolean)


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
            # start with rig time, if not use state time
            wheel_reset = (
                self._trial["t_vstimstart_rig"]
                if self._trial["t_vstimstart_rig"]
                else self._trial["t_vstimstart"]
            )
            self.set_wheel_traces(wheel_reset)

            return self._update_and_return(return_as)

    def set_state_events(self) -> None:
        """Goes over the transitions to set state based timings and also sets the state_outcome"""

        self._trial["t_vstimstart"] = self.data["state"].filter(
            pl.col("transition") == "stimstart"
        )[0, "corrected_elapsed"]

        openloop_start = self.data["state"].filter(
            pl.col("transition") == "responsestart"
        )

        correct = self.data["state"].filter(pl.col("transition") == "correct")
        if len(correct):
            self._trial["state_outcome"] = 1
            self._trial["state_response_time"] = correct[0, "stateElapsed"]

        incorrect = self.data["state"].filter(pl.col("transition") == "incorrect")
        if len(incorrect):
            self._trial["state_outcome"] = 0
            self._trial["state_response_time"] = incorrect[0, "stateElapsed"]

        self._trial["state_response_time"] += openloop_start[0, "stateElapsed"]

        stim_end = self.data["state"].filter(pl.col("transition").str.contains("stimend"))
        if len(stim_end):
            self._trial["t_vstimend"] = stim_end[0, "corrected_elapsed"]

        trial_end = self.data["state"].filter(
            pl.col("transition").str.contains("trialend")
        )
        if len(trial_end):
            self._trial["t_trialend"] = trial_end[0, "corrected_elapsed"]

    def set_vstim_properties(self):
        """Overwrites the visualTrialHandler method to extract the relevant vstim properties"""
        super().set_vstim_properties()
        # only look at columns that have "_l", ASSUMING there will be an "_r" counterpart of it
        columns_to_modify = [
            k.strip("_l") for k in self._trial.keys() if k.endswith("_l")
        ]
        self._trial["prob"] = self._trial["prob"][0][0]
        self._trial["fraction_r"] = self._trial["fraction_r"][0][0]
        if "opto_pattern" in self._trial.keys():
            if self._trial["opto_pattern"] is not None:
                self._trial["opto_pattern"] = int(self._trial["opto_pattern"][0][0])
            else:
                self._trial["opto_pattern"] = -1
        else:
            self._trial["opto_pattern"] = -1

        _correct = self._trial.pop("correct")[0][0]  # right if 1, left if 0
        self._trial["correct_side"] = int(_correct)
        _side = "_r" if _correct else "_l"
        _other_side = "_l" if _correct else "_r"  # right if 1, left if 0
        for col in columns_to_modify:
            _targ = self._trial.pop(col + _side)  # this is the target side
            _dist = self._trial.pop(col + _other_side)
            if col == "posx":
                col = "pos"
            else:
                _targ = _targ[0][0]
                _dist = _dist[0][0]

            self._trial[f"target_{col}"] = _targ
            self._trial[f"distract_{col}"] = _dist

    def set_wheel_traces(self, reset_time_point: float) -> None:
        """Sets the wheel"""
        wheel_array = self._get_rig_event("position")
        if wheel_array is None or not len(wheel_array):
            return

        trace = WheelTrace(wheel_array[:, 0], wheel_array[:, 1])
        self._trial["wheel_t"] = [trace.t.tolist()]
        self._trial["wheel_pos"] = [trace.pos.tolist()]

        res = trace.process(
            reset_time_point,
            freq=5,
            units="rad",
            pos_thresh=0.00015,  # rads, 0.02 for ticks
            t_thresh=0.5,
        )

        # hardware response time preferred, else the state-machine time
        resp = self._trial.get("rig_response_time")
        if resp is None:
            resp = self._trial["state_response_time"]
        rt = match_response_movement(
            res["movements"], resp, gap_tol=_GAP_TOL_MS, min_rt=_MIN_RT_MS
        )
        self._trial["reaction_time"] = rt.reaction_time
        self._trial["reaction_time_source"] = rt.source
        self._trial["anticipatory"] = rt.anticipatory

    def set_outcome(self) -> None:
        """Sets the trial outcome by using the integer state outcome value"""
        if self._trial["state_outcome"] is not None:
            self._trial["outcome"] = OUTCOMES[self._trial["state_outcome"]]
