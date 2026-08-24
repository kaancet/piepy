import polars as pl
import patito as pt
import numpy as np
from typing import Literal

from piepy.core.errors import StateMachineError, VstimLoggingError  # noqa: F401
from piepy.core.utils import unique_except
from piepy.sensory.visual.visualTrial import VisualTrial, VisualTrialHandler
from piepy.psychophysics.psychophysicalTrial import (
    PsychophysicalTrial,
    PsychophysicalTrialHandler,
)
from piepy.psychophysics.wheelTrace import WheelTrace, match_response_movement

OUTCOMES = {-1: "early", 1: "hit", 0: "miss"}

# reaction-time matching tolerances (ms, in the stimulus-reset frame)
_GAP_TOL_MS = 100.0  # response may land just after a (clipped) movement's offset
_MIN_RT_MS = 150.0  # onsets earlier than this are anticipatory, not stimulus-driven
_SPEED_SCALE = 1000.0  # peak_speed reporting scale (rad/s -> reported units)


class WheelDetectionTrial(VisualTrial, PsychophysicalTrial):
    outcome: Literal["early", "hit", "miss", "catch"]
    wheel_t: list[float] = pt.Field(default=[], dtype=pl.List(pl.Float64))
    wheel_pos: list[int] = pt.Field(default=[], dtype=pl.List(pl.Int64))
    reaction_time: float | None = pt.Field(default=None, dtype=pl.Float64)
    peak_speed: float | None = pt.Field(default=None, dtype=pl.Float64)
    # provenance of reaction_time: "contain" | "gap" | "inferred" | "none" | None (not computed)
    reaction_time_source: str | None = pt.Field(default=None, dtype=pl.Utf8)
    anticipatory: bool | None = pt.Field(default=None, dtype=pl.Boolean)


class WheelDetectionTrialHandler(VisualTrialHandler, PsychophysicalTrialHandler):
    def __init__(self):
        super().__init__()
        self._trial = {k: None for k in WheelDetectionTrial.columns}
        self.is_early = False
        self.set_model(WheelDetectionTrial)

    def get_trial(
        self, trial_no: int, rawdata: dict, return_as="dict"
    ) -> pt.DataFrame | dict | list | None:
        """Main function that is called from outside, sets the trial, validates data type and returns it

        Args:
            trial_no (int): Trial number
            rawdata (dict): rawdata dictionary that will be used to extract the desired trial (trial_no)
            return_as (str, optional): Return type string. Defaults to "dict".

        Returns:
            pt.DataFrame | dict | list | None: returned data
        """
        self.was_screen_off = True  # flag for not having OFF pulse in screen data
        self.init_trial()
        _is_trial_set = self.set_trial(trial_no, rawdata)
        self.is_early = self.check_early()

        if not _is_trial_set:
            return None

        if not self.is_early:
            self.set_screen_events()  # should return a 2x2 matrix, first column is timings for screen ON and OFF.
        self.sync_timeframes()  # syncs the state and vstim log times, using screen ONSET

        # NOTE: sometimes due to state machine logic, the end of trial will be end of stimulus
        # this causes a trial to have a single screen event (ON) to be parsed into a given trial
        # to remedy this, we check the screen data after syncing the timeframes of rig(arduoino) and statemachine(python)
        if not self.was_screen_off:
            self.recheck_screen_events(rawdata["screen"])

        # should be run after sync_timeframes, needs the corrected time columns
        if not self.set_state_events():
            return None
        self.set_vstim_properties()  # should be run after sync_timeframes

        self.adjust_rig_response()
        if not self.is_early:
            if self._trial["t_vstimstart_rig"] is not None:
                self.set_wheel_traces(self._trial["t_vstimstart_rig"])
            else:
                self.set_wheel_traces(self._trial["t_vstimstart"])
        else:
            self.set_wheel_traces(
                self._trial["t_trialinit"] + self._trial["duration_blank"]
            )  # would be stimulus start

        self.set_licks()
        self.set_reward()
        self.set_opto()
        self.set_outcome()
        return self._update_and_return(return_as)

    def check_early(self) -> bool:
        """Checks if the trial is early

        Returns:
            bool: True if trial was early, False otherwise
        """
        _ret = False
        if "early" in self.data["state"]["transition"].to_list():
            _ret = True
        else:
            if "hit" in self.data["state"]["transition"].to_list():
                _name = "hit"
            elif "miss" in self.data["state"]["transition"].to_list():
                _name = "miss"
            elif "catch" in self.data["state"]["transition"].to_list():
                _name = "catch"
            else:
                raise ValueError("ijbasdjsdobwdfibwdefiubweiubwef")

            _resp = self.data["state"].filter(pl.col("transition") == _name)[
                0, "stateElapsed"
            ]
            if _resp <= 150 and _name != "catch":
                _ret = True
        return _ret

    def set_state_events(self) -> bool:
        """Goes over the transitions to set state based timings and also sets the state_outcome

        Returns:
            bool: True if state set correctly, False if not
        """
        # iscatch?
        _states_set = True
        catch = self.data["state"].filter(pl.col("transition") == "catch")
        if len(catch):
            self._trial["isCatch"] = True
            catch_resp_time = catch[0, "stateElapsed"]
            if catch_resp_time < 1000:
                self._trial["state_outcome"] = 1
            else:
                self._trial["state_outcome"] = 0

            self._trial["t_vstimend"] = catch[0, "corrected_elapsed"]
            self._trial["state_response_time"] = catch_resp_time
        else:
            self._trial["isCatch"] = False

        # trial init and blank duration
        cue = self.data["state"].filter(pl.col("transition") == "cuestart")
        if len(cue):
            self._trial["t_trialinit"] = cue[0, "corrected_elapsed"]
            self._trial["duration_quiescence"] = cue[0, "stateElapsed"]
            try:
                temp_blank = cue[0, "blankDuration"]
            except Exception:
                temp_blank = cue[0, "trialType"]  # old logging for some sessions
            self._trial["duration_blank"] = temp_blank

        # early
        early = self.data["state"].filter(pl.col("transition") == "early")
        if len(early):
            self._trial["state_outcome"] = -1
            self._trial["state_response_time"] = (
                early[0, "stateElapsed"] - self._trial["duration_blank"]
            )

        # stimulus start
        else:
            self._trial["t_vstimstart"] = self.data["state"].filter(
                pl.col("transition") == "stimstart"
            )[0, "corrected_elapsed"]

            # hit
            hit = self.data["state"].filter(pl.col("transition") == "hit")
            if len(hit):
                self._trial["state_outcome"] = 1
                self._trial["state_response_time"] = hit[0, "stateElapsed"]

            # miss
            miss = self.data["state"].filter(pl.col("transition") == "miss")
            if len(miss):
                temp_resp = miss[0, "stateElapsed"]
                if temp_resp < 150:
                    # this is actually early, higher threshold here compared to stimpy because of logging lag
                    self._trial["state_outcome"] = -1
                    self._trial["state_response_time"] = temp_resp
                elif 200 < temp_resp < 1000:
                    # This should not happen
                    # DISCARD TRIAL
                    _states_set = False
                    print(
                        f"[TRIAL-{self._trial['trial_no']}] A miss trial that has a state_response_time of {temp_resp} is not allowed!!"
                    )
                else:
                    # actual miss >= 1050
                    self._trial["state_outcome"] = 0
                    self._trial["state_response_time"] = temp_resp

            # stimulus end
            stim_end = self.data["state"].filter(
                pl.col("transition").str.contains("stimend")
            )
            if len(stim_end):
                self._trial["t_vstimend"] = stim_end[0, "corrected_elapsed"]

        trial_end = self.data["state"].filter(
            pl.col("transition").str.contains("trialend")
        )
        if len(trial_end):
            self._trial["t_trialend"] = trial_end[0, "corrected_elapsed"]

        return _states_set

    def set_vstim_properties(self) -> None:
        """Overwrites the visualTrialHandler method to extract the relevant vstim properties,
        Also converts some properties to be scalars instead of lists(this is experiment specific)
        """
        super().set_vstim_properties()
        columns_to_modify = [
            k.strip("_l") for k in self._trial.keys() if k.endswith("_l")
        ]
        self._trial.pop("correct")

        # set the opto pattern
        if (
            "opto_pattern" in self._trial.keys()
            and self._trial["opto_pattern"] is not None
        ):
            self._trial["opto_pattern"] = int(self._trial["opto_pattern"][0][0])
        else:
            self._trial["opto_pattern"] = -1

        if self.is_early:
            for c in columns_to_modify:
                self._trial.pop(c + "_l")
                self._trial.pop(c + "_r")
                self._trial[c] = None

            self._trial["prob"] = None
            self._trial["stim_pos"] = self._trial.pop("posx")
            self._trial["median_loop_time"] = None
        else:
            if (
                self._trial["contrast_l"][0][0] == 0
                and self._trial["contrast_r"][0][0] == 0
                and self._trial["opto_pattern"] == -1
            ):
                self._trial["isCatch"] = True
            _correct = (
                0
                if self._trial["contrast_l"][0][0] > self._trial["contrast_r"][0][0]
                else 1
            )
            _side = "_r" if _correct else "_l"  # right if 1, left if 0
            _other_side = "_l" if _correct else "_r"  # right if 1, left if 0
            for c in columns_to_modify:
                _temp = self._trial.pop(c + _side)[0][0]  # this is the target side
                _ = self._trial.pop(c + _other_side)  # this is the non-target side
                self._trial[c] = _temp

            self._trial["prob"] = self._trial["prob"][0][0]
            self._trial["stim_pos"] = int(self._trial.pop("posx"))
            self._trial["stim_pos"] = (
                0 if self._trial["contrast"] == 0 else self._trial["stim_pos"]
            )
            self._trial["median_loop_time"] = round(
                float(np.median(np.diff(self._trial["corrected_presentTime"]))), 3
            )

        if (
            "rig_react_diff" in self._trial.keys()
            and self._trial["rig_react_diff"] is not None
        ):
            _tiks = self._trial.pop("rig_react_diff")[0]
            _idx = next((i for i, x in enumerate(_tiks) if x != -1), None)
            self._trial["rig_response_tick"] = (
                int(abs(_tiks[_idx])) if _idx is not None else None
            )
        else:
            _ = self._trial.pop("rig_react_diff", None)
            self._trial["rig_response_tick"] = None

    def adjust_rig_response(self) -> None:
        """A specialized method to change the rig_reaction_t and rig_reaction_diff from list to float"""
        self._trial["rig_response_time"] = None
        if "rig_react_t" in self._trial.keys():
            if not self.is_early:
                _time_temp = unique_except(self._trial["rig_react_t"][0], [-1])
                if len(_time_temp) == 1:
                    if self._trial["t_vstimstart_rig"] is None:
                        print(
                            "NO RIG VSTIM TIME IN A NON_EARLY TRIAL THIS SHOULD NOT HAPPEN, USING STATE TIME"
                        )
                        self._trial["t_vstimstart_rig"] = int(self._trial["t_vstimstart"])
                        self._trial["t_vstimend_rig"] = int(self._trial["t_vstimend"])

                    self._trial["rig_response_time"] = float(
                        _time_temp[0] * 1000 - self._trial["t_vstimstart_rig"]
                    )
                elif len(_time_temp) == 0:
                    self._trial["rig_response_time"] = None
                else:
                    raise VstimLoggingError("Reaction time logging is weird!")
        self._trial.pop("rig_react_t", None)

    def set_outcome(self) -> None:
        """Sets the trial outcome by using the integer state outcome value"""
        self._trial["outcome"] = OUTCOMES[self._trial["state_outcome"]]

        # designate too early reaction times as early
        if self._trial["outcome"] == "hit":
            r_time = self._trial.get("reaction_time", None)
            if r_time is not None and r_time <= 100:
                self._trial["outcome"] = "early"

    def set_wheel_traces(self, reset_time_point: float) -> None:
        """Sets the wheel traces and wheel reaction time from the traces

        Args:
            reset_time_point (bool): Time point to reset the wheel trajectory time values
        """
        wheel_array = self._get_rig_event("position")
        trace = WheelTrace()
        if wheel_array is None or not len(wheel_array):
            return

        trace = WheelTrace(wheel_array[:, 0], wheel_array[:, 1])
        self._trial["wheel_t"] = [trace.t.tolist()]
        self._trial["wheel_pos"] = [trace.pos.tolist()]

        res = trace.process(
            reset_time_point,
            freq=5,
            units="rad",
            pos_thresh=0.0003,  # rads, 0.02 for ticks
            t_thresh=1,
            min_dur=20,
            min_gap=30,
        )
        t_interp = res["t"]
        mov_dict = res["movements"]

        # match the response to the movement that produced it (hardware time preferred, else the
        # state-machine time). The matcher prefers the *containing* movement over an earlier one
        # that merely ends just before the response, and flags anticipatory onsets.
        resp = self._trial["rig_response_time"]
        if resp is None:
            resp = self._trial["state_response_time"]
        rt = match_response_movement(
            mov_dict, resp, gap_tol=_GAP_TOL_MS, min_rt=_MIN_RT_MS
        )

        self._trial["reaction_time"] = rt.reaction_time
        self._trial["peak_speed"] = (
            None if rt.peak_speed is None else rt.peak_speed * _SPEED_SCALE
        )
        self._trial["reaction_time_source"] = rt.source
        self._trial["anticipatory"] = rt.anticipatory

        if (
            self._trial["state_outcome"] == 1
            and rt.reaction_time is None
            and len(mov_dict["onsets"])
        ):
            # hit, but the response time matched no movement (even the rig time can be logged late):
            # last resort -- the first movement whose peak speed lands after the min-RT floor.
            # Marked "inferred"; the measured rig_response_time is left untouched.
            peak_times = t_interp[mov_dict["speed_peaks"][:, 0].astype(int)]
            plausible = np.where(peak_times > _MIN_RT_MS)[0]
            if plausible.size:
                j = int(plausible[0])
                onset = float(mov_dict["onsets"][j, 1])
                self._trial["reaction_time"] = onset
                self._trial["peak_speed"] = float(
                    mov_dict["speed_peaks"][j, 1] * _SPEED_SCALE
                )
                self._trial["reaction_time_source"] = "inferred"
                self._trial["anticipatory"] = bool(onset < _MIN_RT_MS)

    def set_opto(self) -> None:
        """Failsafe"""
        super().set_opto()

        if self._trial["opto"]:
            _vstim = self.data["vstim"]
            # look for opto in vstim
            if "opto" in _vstim.columns:
                o_len = _vstim["opto"].unique().drop_nulls().len()
                if o_len == 2:
                    self._trial["opto"] = False
                    self._trial["opto_pulse"] = [[]]
                elif o_len > 2:
                    raise ValueError(
                        f"[TRIAL-{self._trial['trial_no']}] Opto logging error!"
                    )
            else:
                # if no opto in vstim, then it is not an opto trial
                self._trial["opto"] = False
                self._trial["opto_pulse"] = [[]]
