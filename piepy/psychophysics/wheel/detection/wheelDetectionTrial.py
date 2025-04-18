import polars as pl
import patito as pt
import numpy as np
from typing import Literal

from ....core.exceptions import StateMachineError, VstimLoggingError  # noqa: F401
from ....core.utils import unique_except
from ....sensory.visual.visualTrial import VisualTrial, VisualTrialHandler
from ...psychophysicalTrial import PsychophysicalTrial, PsychophysicalTrialHandler
from ..wheelTrace import WheelTrace


OUTCOMES = {-1: "early", 1: "hit", 0: "miss"}


class WheelDetectionTrial(VisualTrial, PsychophysicalTrial):
    outcome: Literal["early", "hit", "miss", "catch"]
    wheel_t: list[float] = pt.Field(default=[], dtype=pl.List(pl.Float64))
    wheel_pos: list[int] = pt.Field(default=[], dtype=pl.List(pl.Int64))
    reaction_time: float | None = pt.Field(default=None, dtype=pl.Float64)


class WheelDetectionTrialHandler(VisualTrialHandler, PsychophysicalTrialHandler):
    def __init__(self):
        super().__init__()
        self._trial = {k: None for k in WheelDetectionTrial.columns}
        self.is_early = False
        self.was_screen_off = True  # flag for not having OFF pulse in screen data
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
        self.init_trial()
        _is_trial_set = self.set_trial(trial_no, rawdata)
        self.is_early = self.check_early()

        if not _is_trial_set:
            return None

        self.set_screen_events()  # should return a 2x2 matrix, first column is timings for screen ON and OFF.
        self.sync_timeframes()  # syncs the state and vstim log times, using screen ONSET

        # NOTE: sometimes due to state machine logic, the end of trial will be end of stimulus
        # this causes a trial to have a single screen event (ON) to be parsed into a given trial
        # to remedy this, we check the screen data after syncing the timeframes of rig(arduoino) and statemachine(python)
        if not self.was_screen_off:
            self.recheck_screen_events(rawdata["screen"])

        _states_set = (
            self.set_state_events()
        )  # should be run after sync_timeframes, needs the corrected time columns
        if not _states_set:
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

        self.set_outcome()
        self.set_licks()
        self.set_reward()
        self.set_opto()
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
                elif 150 < temp_resp < 1000:
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
                        self._trial["t_vstimstart_rig"] = int(
                            self._trial["t_vstimstart"]
                        )
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

    def set_wheel_traces(self, reset_time_point: float) -> None:
        """Sets the wheel traces and wheel reaction time from the traces

        Args:
            reset_time_point (bool): Time point to reset the wheel trajectory time values
        """
        wheel_array = self._get_rig_event("position")
        trace = WheelTrace()
        if wheel_array is not None and len(wheel_array):
            t = wheel_array[:, 0]
            pos = wheel_array[:, 1]

            # check for timing recording errors, sometimes t is not monotonically increasing
            t, pos = trace.fix_trace_timing(t, pos)

            self._trial["wheel_t"] = [t.tolist()]
            self._trial["wheel_pos"] = [pos.tolist()]

            _, _, t_interp, tick_interp = trace.reset_and_interpolate(
                t, pos, reset_time_point, 5
            )

            pos_interp = trace.cm_to_rad(trace.ticks_to_cm(tick_interp))

            mov_dict = trace.get_movements(
                t_interp,
                pos_interp,
                freq=5,
                pos_thresh=0.0003,  # rads, 0.02 for ticks
                t_thresh=1,
                min_dur=20,
                min_gap=30,
            )

            self._trial["reaction_time"] = None
            self._trial["peak_speed"] = None
            _resp = self._trial["state_response_time"]
            for i in range(len(mov_dict["onsets"])):
                _on = mov_dict["onsets"][i, 1]
                _off = mov_dict["offsets"][i, 1]
                if _resp < _off and _resp >= _on:
                    # this is the movement that registered the animals answer
                    self._trial["reaction_time"] = float(_on)
                    self._trial["peak_speed"] = float(
                        mov_dict["speed_peaks"][i, 1] * 1000
                    )
                    break
                # sometimes the response is in between two movements
                elif _resp >= _off and _resp <= _off + 100:
                    self._trial["reaction_time"] = float(_on)
                    self._trial["peak_speed"] = float(
                        mov_dict["speed_peaks"][i, 1] * 1000
                    )
                    break

            if (
                self._trial["state_outcome"] == 1
                and self._trial["reaction_time"] is None
            ):
                # sometimes the even the rig response time is not recorded on time, so
                # we get the peak speed time as response
                _temp = mov_dict["speed_peaks"][:, 0].astype(int)
                velo_times = t_interp[_temp]  # time of highest speed
                speed_after_150 = np.where(velo_times > 150)[0]
                if not len(speed_after_150):
                    # very rarely the reaction time is too early
                    # don't do anything
                    pass
                else:
                    idx_val = speed_after_150[0]
                    self._trial["rig_response_time"] = float(velo_times[idx_val])
                    self._trial["reaction_time"] = float(
                        mov_dict["onsets"][:, 1][idx_val]
                    )
                    self._trial["peak_speed"] = float(
                        mov_dict["speed_peaks"][:, 1][idx_val] * 1000
                    )
                    print(self._trial["trial_no"])
                    print(f"old:{_resp}")
                    print(f"new:{float(velo_times[idx_val])}")
