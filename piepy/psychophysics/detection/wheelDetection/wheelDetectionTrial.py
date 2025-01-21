import polars as pl
import patito as pt
import numpy as np
from typing import Literal

from ....core.exceptions import StateMachineError, VstimLoggingError
from ....core.utils import unique_except
from ....sensory.visual.visualTrial import VisualTrial, VisualTrialHandler
from ...psychophysicalTrial import PsychophysicalTrial, PsychophysicalTrialHandler
from ...wheelTrace import WheelTrace


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
        self.was_screen_off = True  # flag for not having OFF pulse in screen data
        self.set_model(WheelDetectionTrial)

    def get_trial(
        self, trial_no: int, rawdata: dict, return_as="dict"
    ) -> pt.DataFrame | dict | list | None:
        """Main function that is called from outside, sets the trial, validates data type and returns it"""
        self.init_trial()
        _is_trial_set = self.set_trial(trial_no, rawdata)
        _is_trial_early = "early" in self.data["state"]["transition"].to_list()

        if not _is_trial_set:
            return None
        else:
            self.set_screen_events()  # should return a 2x2 matrix, first column is timings for screen ON and OFF.
            self.sync_timeframes()  # syncs the state and vstim log times, using screen ONSET

            # NOTE: sometimes due to state machine logic, the end of trial will be end of stimulus
            # this causes a trial to have a single screen event (ON) to be parsed into a given trial
            # to remedy this, we check the screen data after syncing the timeframes of rig(arduoino) and statemachine(python)
            if not self.was_screen_off:
                self.recheck_screen_events(rawdata["screen"])

            self.set_state_events()  # should be run after sync_timeframes, needs the corrected time columns
            if self._trial["state_outcome"] == -1:
                _is_trial_early = True
            self.set_vstim_properties(
                _is_trial_early
            )  # should be run after sync_timeframes

            # else:
            #     # here we manually set corrected columns,
            #     # not pretty, but necessary to keep different trial types independent and modular
            #     self.data["state"] = self.data["state"].with_columns(
            #         pl.col("elapsed").alias("corrected_elapsed")
            #     )
            #     self.data["vstim"] = self.data["vstim"].with_columns(
            #         pl.col("presentTime").alias("corrected_presentTime")
            #     )

            if not _is_trial_early:
                if self._trial["t_vstimstart_rig"] is not None:
                    self.set_wheel_traces(self._trial["t_vstimstart_rig"])
                else:
                    self.set_wheel_traces(self._trial["t_vstimstart"])
            else:
                self.set_wheel_traces(
                    self._trial["t_trialinit"] + self._trial["duration_blank"]
                )  # would be stimulus start

            self.adjust_rig_response()
            self.set_outcome()
            self.set_licks()
            self.set_reward()
            self.set_opto()
            return self._update_and_return(return_as)

    def set_state_events(self) -> None:
        """Goes over the transitions to set state based timings and also sets the state_outcome"""
        # iscatch?
        catch = self.data["state"].filter(pl.col("transition") == "catch")
        if len(catch):
            self._trial["isCatch"] = True
            self._trial["t_vstimend"] = catch[0, "corrected_elapsed"]
            self._trial["state_outcome"] = None
            self._trial["state_response_time"] = catch[0, "stateElapsed"]
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
                if temp_resp <= 200:
                    # this is actually early, higher threshold here compared to stimpy because of logging lag
                    self._trial["state_outcome"] = -1
                    self._trial["state_response_time"] = temp_resp
                elif 200 < temp_resp < 1000:
                    # This should not happen
                    # DISCARD TRIAL
                    raise StateMachineError(
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

    def set_vstim_properties(self, is_early_trial: bool) -> None:
        """Overwrites the visualTrialHandler method to extract the relevant vstim properties,
        Also converts some properties to be scalars instead of lists(this is experiment specific)
        """
        super().set_vstim_properties()
        columns_to_modify = ["contrast", "sf", "tf", "posx"]
        if is_early_trial:
            self._trial.pop("correct")
            for c in columns_to_modify:
                self._trial.pop(c + "_l")
                self._trial.pop(c + "_r")
                self._trial[c] = None

            self._trial["prob"] = None
            self._trial["stim_pos"] = self._trial.pop("posx")
            self._trial["median_loop_time"] = None
        else:
            _correct = self._trial.pop("correct")[0][0]
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

        if "rig_react_diff" in self._trial.keys():
            _tiks = self._trial["rig_react_diff"][0]
            _idx = next((i for i, x in enumerate(_tiks) if x != -1), None)
            self._trial["rig_response_tick"] = (
                int(abs(_tiks[_idx])) if _idx is not None else None
            )
        else:
            self._trial["rig_response_tick"] = None

        if (
            "opto_pattern" in self._trial.keys()
            and self._trial["opto_pattern"] is not None
        ):
            self._trial["opto_pattern"] = int(self._trial["opto_pattern"][0][0])
        else:
            self._trial["opto_pattern"] = -1

    def adjust_rig_response(self) -> None:
        """A specialized method to change the rig_reaction_t and rig_reaction_diff from list to float"""
        if "rig_react_t" in self._trial.keys():
            if "early" not in self.data["state"]["transition"].to_list():
                _time_temp = unique_except(self._trial["rig_react_t"][0], [-1])
                if len(_time_temp) == 1:
                    self._trial["rig_response_time"] = float(
                        _time_temp[0] * 1000 - self._trial["t_vstimstart_rig"]
                    )
                elif len(_time_temp) == 0:
                    self._trial["rig_response_time"] = None
                else:
                    raise VstimLoggingError("Reaction time logging is weird!")

                _diff_temp = unique_except(self._trial["rig_react_diff"][0], [-1])
                if len(_diff_temp) == 1:
                    self._trial["rig_response_diff"] = float(_diff_temp[0])
                elif len(_diff_temp) == 0:
                    self._trial["rig_response_diff"] = None
                else:
                    raise VstimLoggingError("Reaction time logging is weird!")
        else:
            self._trial["rig_response_time"] = None

    def set_outcome(self) -> None:
        """Sets the trial outcome by using the integer state outcome value"""
        if self._trial["state_outcome"] is not None:
            self._trial["outcome"] = OUTCOMES[self._trial["state_outcome"]]
        else:
            self._trial["outcome"] = "catch"

    def set_wheel_traces(self, reset_time_point: float) -> None:
        """Sets the wheel traces and wheel reaction time from the traces"""
        wheel_array = self._get_rig_event("position")
        trace = WheelTrace()
        if wheel_array is not None and len(wheel_array):
            t = wheel_array[:, 0]
            pos = wheel_array[:, 1]

            # check for timing recording errors, sometimes t is not monotonically increasing
            while not np.all(np.diff(t) > 0):
                # find and delete that fucker
                _idx = np.where(np.diff(t) < 0)[0]
                t = np.delete(t, _idx)
                pos = np.delete(pos, _idx)

            self._trial["wheel_t"] = [t]
            self._trial["wheel_pos"] = [pos]

            _, _, t_interp, tick_interp = trace.reset_and_interpolate(
                t, pos, reset_time_point, 5
            )

            pos_interp = trace.cm_to_rad(trace.ticks_to_cm(tick_interp))

            mov_dict = trace.get_movements(
                t_interp,
                pos_interp,
                freq=5,
                pos_thresh=0.00015,  # rads, 0.02 for ticks
                t_thresh=0.5,
            )

            self._trial["reaction_time"] = None
            if "_rig_response_time" in self._trial.keys():
                _resp = self._trial["rig_response_time"]
            else:
                _resp = self._trial["state_response_time"]
            for i in range(len(mov_dict["onsets"])):
                _on = mov_dict["onsets"][i, 1]
                _off = mov_dict["offsets"][i, 1]
                if _resp < _off and _resp >= _on:
                    # this is the movement that registered the animals answer
                    self._trial["reaction_time"] = float(_on)
                    break
                # sometimes the response is in between two movements
                if _resp >= _off and _resp <= _off + 100:
                    self._trial["reaction_time"] = float(_on)
                    break


# class WheelDetectionTrial(Trial):
#     def __init__(self, trial_no: int, meta, logger) -> None:
#         super().__init__(trial_no, meta, logger)

#     def correct_timeframes(self, rawdata: dict) -> None:
#         """Corrects the timing of statemachine and vstim using the stimulus onset in screen log"""

#         state = self.data["state"]
#         screen = rawdata["screen"].filter(
#             (pl.col("duinotime") >= self.t_trialstart)
#             & (pl.col("duinotime") <= self.t_trialend)
#         )
#         # use trial no to get vstim data slice
#         vstim = rawdata["vstim"].filter(pl.col("iTrial") == self.trial_no)

#         if screen.is_empty():
#             # no screen events to sync timeframes, can happen during training but shouldn't happen during experiments!
#             self.vstim_offset = 0.0
#             self.state_offset = 0.0
#         else:
#             # correction can only happen in stim trials
#             if "stimstart" in state["transition"].to_list():
#                 rig_stim_onset = screen[0, "duinotime"]
#                 state_stim_onset = state.filter(pl.col("transition") == "stimstart")[
#                     0, "elapsed"
#                 ]

#                 try:
#                     vstim_onset = vstim.filter(pl.col("photo") == True)[0, "presentTime"]
#                 except:
#                     vstim_onset = vstim.filter(pl.col("presentTime") >= rig_stim_onset)[
#                         0, "presentTime"
#                     ]
#                     self.logger.error(f"No photosensor logged in vstim!!")
#                 vstim_onset *= 1000  # ms
#                 self.vstim_offset = vstim_onset - rig_stim_onset
#                 self.state_offset = state_stim_onset - rig_stim_onset
#             else:
#                 # no stimulus, meaning early trials
#                 self.vstim_offset = 0.0
#                 self.state_offset = 0.0

#         # if offset is negative, that means the rig_onset time happened after the python timing
#         self.data["state"] = state.with_columns(
#             (pl.col("elapsed") - self.state_offset).alias("corrected_elapsed")
#         )
#         self.data["vstim"] = vstim.with_columns(
#             (pl.col("presentTime") - self.vstim_offset).alias("corrected_presentTime")
#         )
#         self.data["screen"] = screen

#         # correct the trialstart and end times
#         self.t_trialstart = self.data["state"].filter(
#             pl.col("transition") == "trialstart"
#         )[0, "corrected_elapsed"]
#         self.t_trialend = self.data["state"].filter(pl.col("transition") == "trialend")[
#             0, "corrected_elapsed"
#         ]

#     def set_data_slices(self, rawdata: dict) -> bool:
#         return super().set_data_slices(rawdata)

#     def get_vstim_properties(self) -> dict:
#         """Extracts the necessary properties from vstim data"""
#         ignore = ["iTrial", "photo", "code", "presentTime"]

#         vstim = self.data["vstim"]
#         vstim = vstim.drop_nulls(subset=["prob"])
#         # this is an offline fix for a vstim logging issue where time increment messes up vstim logging
#         vstim = vstim[:-1]

#         early_flag = self.state_outcome
#         if self.state_outcome != -1 and vstim.is_empty():
#             self.logger.warning(f"Empty vstim data for non-early trial!!")
#             early_flag = -1

#         temp_dict = {}
#         for col in vstim.columns:
#             if col in ignore:
#                 continue
#             if len(vstim.select(col).unique()) == 1:
#                 # if a column has all the same values, take the first entry of the column as the value
#                 # sf, tf, contrast, stim_side, correct, opto, opto_pattern should run through here
#                 temp_dict[col] = vstim[0, col]
#             elif len(vstim.select(col).unique()) > 1 and col not in ["reward"]:
#                 # if different values exist in the column, take it as a list, this should not happen in detection task
#                 self.logger.error(
#                     f"{col} has multiple unique entries ({len(vstim.select(col).unique())}). This shouldn't be the case"
#                 )
#                 temp_dict[col] = vstim[col].to_list()
#             else:
#                 temp_dict[col] = None

#         vstim_dict = {
#             "contrast": None,
#             "spatial_freq": None,
#             "temporal_freq": None,
#             "stim_pos": None,
#             "opto_pattern": None,
#             "prob": None,
#             "rig_reaction_time": None,
#             "rig_reaction_tick": None,
#             "median_loop_time": None,
#         }

#         if early_flag != -1:
#             vstim_dict["median_loop_time"] = (
#                 np.mean(np.diff(vstim["presentTime"].to_numpy())) * 1000
#             )  # ms

#             contrast_temp = (
#                 100 * round(temp_dict["contrast_r"], 5)
#                 if temp_dict["correct"]
#                 else 100 * round(temp_dict["contrast_l"], 5)
#             )
#             if contrast_temp % 1 == 0:
#                 contrast_temp = round(contrast_temp, 1)
#             else:
#                 contrast_temp = round(contrast_temp, 2)
#             vstim_dict["contrast"] = contrast_temp

#             vstim_dict["spatial_freq"] = (
#                 round(temp_dict["sf_r"], 2)
#                 if temp_dict["correct"]
#                 else round(temp_dict["sf_l"], 2)
#             )
#             vstim_dict["temporal_freq"] = (
#                 round(temp_dict["tf_r"], 2)
#                 if temp_dict["correct"]
#                 else round(temp_dict["tf_l"], 2)
#             )
#             vstim_dict["stim_pos"] = (
#                 int(temp_dict["posx_r"])
#                 if temp_dict["correct"]
#                 else int(temp_dict["posx_l"])
#             )

#             vstim_dict["opto_pattern"] = temp_dict.get("opto_pattern", None)
#             vstim_dict["prob"] = temp_dict["prob"]

#             # training failsafe
#             if "opto_pattern" not in temp_dict.keys():
#                 vstim_dict["opto_pattern"] = -1
#                 self.logger.warning(
#                     f"No opto_pattern found in vstim log, setting to -1(nonopto)"
#                 )

#             if vstim_dict["contrast"] == 0:
#                 vstim_dict["stim_pos"] = 0  # no meaningful side when 0 contrast

#             #
#             if "rig_react_t" in vstim.columns:
#                 rig_react = vstim.filter(
#                     (pl.col("rig_react_t").is_not_null()) & (pl.col("rig_react_t") != -1)
#                 )
#                 if len(rig_react):
#                     if len(rig_react.unique("rig_react_t")) == 1:
#                         # should be only one unique value in rig react gotten from the vstim log
#                         try:
#                             vstim_dict["rig_reaction_time"] = (
#                                 rig_react[0, "rig_react_t"] * 1000 - self.t_stimstart_rig
#                             )  # ms
#                         except:
#                             vstim_dict["rig_reaction_time"] = (
#                                 rig_react[0, "rig_react_t"] * 1000 - self.t_stimstart
#                             )  # ms
#                         vstim_dict["rig_reaction_tick"] = np.abs(
#                             rig_react[0, "rig_react_diff"]
#                         )
#                     else:
#                         raise ValueError(
#                             f"!!!Whoa there cowboy this shouldn't happen with rig_react_t!!!!"
#                         )

#         self._attrs_from_dict(vstim_dict)
#         return vstim_dict

#     def get_wheel_traces(self, **kwargs) -> dict:
#         """Extracts the wheel trajectories and resets the positions according to time_anchor"""
#         thresh_in_ticks = kwargs.get("tick_thresh", self.meta.wheelThresholdStim)

#         _loop_t = (
#             self.median_loop_time if self.median_loop_time is not None else 16.5
#         )  # avg ms for a loop
#         speed_thresh = thresh_in_ticks / kwargs.get("time_thresh", _loop_t)
#         interp_freq = kwargs.get("freq", 5)

#         wheel_data = self.data["position"]
#         wheel_arr = wheel_data.select(["duinotime", "value"]).to_numpy()

#         # instantiate a wheel Trajectory object
#         traj = WheelTrace(wheel_arr[:, 0], wheel_arr[:, 1], interp_freq=interp_freq)

#         wheel_dict = traj.make_dict_to_log()

#         if len(wheel_arr) <= 2:
#             self.logger.warning(f"Less than 2 sample points for wheel data")
#             return wheel_dict

#         if self.t_stimstart_rig is None:
#             if self.state_outcome != -1:
#                 self.logger.warning(
#                     f"No stimulus start based on photodiode in a stimulus trial, using stateMachine time!"
#                 )
#                 time_anchor = self.t_stimstart
#                 #  the window of analysis for trajectories
#                 window_end = self.t_stimend
#             else:
#                 # for early trials use the blank period
#                 time_anchor = self.t_trialinit
#                 window_end = self.t_trialinit + self.t_blank_dur
#         else:
#             time_anchor = self.t_stimstart_rig
#             window_end = self.t_stimend_rig
#             if window_end is None:
#                 self.logger.warning(
#                     f"No stimend signal from screen data, using corrected state timing!"
#                 )
#                 window_end = self.data["state"].filter(
#                     pl.col("transition").str.contains("stimend")
#                 )[0, "corrected_elapsed"]

#         time_window = [time_anchor, window_end]

#         # initialize the trace
#         traj.init_trace(time_anchor=time_anchor)
#         # get the movements from interpolated positions
#         traj.get_movements(
#             pos_thresh=kwargs.get("pos_thresh", 0.02),
#             t_thresh=kwargs.get("t_thresh", 0.5),
#         )

#         if len(traj.onsets) == 0 and self.state_outcome == 1:
#             self.logger.error("No movement onset detected in a correct trial!")
#             return wheel_dict

#         # there are onsets after 50(stim appearance + 50ms)
#         if len(np.where(traj.onsets > 50)[0]) == 0:
#             if self.state_outcome == 1:
#                 self.logger.error(
#                     f"No detected wheel movement in correct trial after stim!"
#                 )

#         # these has to be run before calculating reaction times to constrain the region of traces we are interested in
#         interval_mask = traj.make_interval_mask(time_window=time_window)
#         traj.select_trace_interval(mask=interval_mask)

#         # get all the reaction times and outcomes here:
#         traj.get_speed_reactions(speed_threshold=speed_thresh)
#         traj.get_tick_reactions(tick_threshold=thresh_in_ticks)

#         # Logging discrapencies
#         # stateMachine vs delta tick
#         if traj.pos_outcome is not None:
#             if self.state_outcome != traj.pos_outcome:
#                 self.logger.critical(
#                     f"stateMachine outcome and delta tick outcome does not match!!! {self.state_outcome}=/={traj.pos_outcome}!"
#                 )
#         else:
#             self.logger.error(f"Can't calculate wheel reaction time in correct trial!!")

#         # stateMachine vs wheel speed
#         if traj.speed_outcome is not None:
#             if self.state_outcome != traj.speed_outcome:
#                 self.logger.critical(
#                     f"stateMachine outcome and wheel speed outcome does not match!!! {self.state_outcome}=/={traj.speed_outcome}!"
#                 )
#         else:
#             self.logger.error(f"Can't calculate speed reaction time in correct trial!!")

#         # delta tick vs wheel speed
#         if traj.pos_outcome != traj.speed_outcome:
#             self.logger.critical(
#                 f"delta tick outcome and wheel speed outcome does not match!!! {traj.pos_outcome}=/={traj.speed_outcome}!"
#             )

#         self.trace = traj
#         # fill the dict
#         wheel_dict = traj.make_dict_to_log()
#         self._attrs_from_dict(wheel_dict)
#         return wheel_dict

#     def get_state_changes(self) -> dict:
#         """
#         Looks at state changes in a given data slice and set class attributes according to them
#         every key starting with t_ is an absolute time starting from experiment start
#         """
#         empty_log_data = {
#             "t_trialstart": self.t_trialstart,  # this is an absolute value
#             "vstim_offset": self.vstim_offset,
#             "state_offset": self.state_offset,
#             "t_stimstart": None,
#             "t_stimend": None,
#             "state_outcome": None,
#         }

#         state_log_data = {**empty_log_data}
#         # in the beginning check if state data is complete
#         if "trialend" not in self.data["state"]["transition"].to_list():
#             self._attrs_from_dict(empty_log_data)
#             return empty_log_data

#         # iscatch?
#         if len(self.data["state"].filter(pl.col("transition") == "catch")):
#             state_log_data["isCatch"] = True
#         else:
#             state_log_data["isCatch"] = False

#         # trial init and blank duration
#         cue = self.data["state"].filter(pl.col("transition") == "cuestart")
#         if len(cue):
#             state_log_data["t_trialinit"] = cue[0, "corrected_elapsed"]
#             state_log_data["t_quiescence_dur"] = cue[0, "stateElapsed"]
#             try:
#                 temp_blank = cue[0, "blankDuration"]
#             except:
#                 temp_blank = cue[0, "trialType"]  # old logging for some sessions
#             state_log_data["t_blank_dur"] = temp_blank
#         else:
#             self.logger.warning("No cuestart after trialstart")

#         # early
#         early = self.data["state"].filter(pl.col("transition") == "early")
#         if len(early):
#             state_log_data["state_outcome"] = -1
#             state_log_data["response_latency"] = early[0, "stateElapsed"]

#         # stimulus start
#         else:
#             state_log_data["t_stimstart"] = self.data["state"].filter(
#                 pl.col("transition") == "stimstart"
#             )[0, "corrected_elapsed"]

#             # hit
#             hit = self.data["state"].filter(pl.col("transition") == "hit")
#             if len(hit):
#                 state_log_data["state_outcome"] = 1
#                 state_log_data["response_latency"] = hit[0, "stateElapsed"]

#             # miss
#             miss = self.data["state"].filter(pl.col("transition") == "miss")
#             if len(miss):
#                 temp_resp = miss[0, "stateElapsed"]
#                 if temp_resp <= 200:
#                     # this is actually early, higher threshold here compared to stimpy because of logging lag
#                     state_log_data["state_outcome"] = -1
#                     state_log_data["response_latency"] = temp_resp
#                 elif 150 < temp_resp < 1000:
#                     # This should not happen
#                     # DISCARD TRIAL
#                     self.logger.error(
#                         f"Trial categorized as MISS with {temp_resp}s response time!! DISCARDING....."
#                     )
#                     self._attrs_from_dict(empty_log_data)
#                     return empty_log_data
#                 else:
#                     # actual miss >= 1000
#                     state_log_data["state_outcome"] = 0
#                     state_log_data["response_latency"] = temp_resp

#         if state_log_data["state_outcome"] is None:
#             # this happens when training with 0 contrast, -1 means there was no answer
#             state_log_data["state_outcome"] = -1
#             state_log_data["response_latency"] = -1

#         # stimulus end
#         if state_log_data["t_stimstart"] is not None:

#             if "stimendincorrect" in self.data["state"]["transition"]:
#                 _end = "stimendincorrect"
#             elif "catch" in self.data["state"]["transition"]:
#                 _end = "catch"
#             elif "stimendcorrect" in self.data["state"]["transition"]:
#                 _end = "stimendcorrect"
#             else:
#                 raise ValueError(
#                     "Stimulus appeared but not disappeared, is this expected??"
#                 )

#             state_log_data["t_stimend"] = self.data["state"].filter(
#                 (pl.col("transition") == _end)
#             )[0, "corrected_elapsed"]

#         state_log_data["t_trialend"] = self.t_trialend
#         self._attrs_from_dict(state_log_data)
#         return state_log_data

#     def get_frames(self, get_from: str = None, **kwargs) -> dict:
#         """Gets the frame ids for stimstart and stimend for trials with visual stimulus(hit or miss)
#         for early trials gets the frame ids of trialinit and "response_latency" aka when the animal gave a response to blank screen at wait period
#         """
#         frame_ids = []
#         if not self.meta.imaging_mode is None:
#             if get_from in self.data.keys():
#                 """
#                 NOTE: even if there's no actual recording for onepcam through labcams(i.e. the camera is running in the labcams GUI without saving),
#                 if there is onepcam frame TTL signals coming into the Arduino it will save them.
#                 This will lead to having onepcam_frame_ids column to be created but there will be no actual tiff files.
#                 """
#                 rig_frames_data = self.data[
#                     get_from
#                 ]  # this should already be the frames of trial dur

#                 if self.state_outcome != -1:
#                     if self.t_stimstart_rig is not None:
#                         # get stim present slice
#                         rig_frames_data = rig_frames_data.filter(
#                             (pl.col("duinotime") >= self.t_stimstart_rig)
#                             & (pl.col("duinotime") <= self.t_stimend_rig)
#                         )

#                         if len(rig_frames_data):
#                             frame_ids = [
#                                 int(rig_frames_data[0, "value"]),
#                                 int(rig_frames_data[-1, "value"]),
#                             ]

#                         else:
#                             self.logger.critical(
#                                 f"{get_from} no camera pulses recorded during stim presentation!!! THIS IS BAD!"
#                             )
#                     else:
#                         self.logger.error(
#                             "No rig screen pulses to gather frames in between screen events!!"
#                         )
#                 else:
#                     # if there is no strimstart_rig(meaning no stimulus shown) then take the frames between trial_init and trial_init + response_time
#                     rig_frames_data = rig_frames_data.filter(
#                         (pl.col("duinotime") >= self.t_trialinit)
#                         & (
#                             pl.col("duinotime")
#                             <= (self.t_trialinit + self.response_latency)
#                         )
#                     )

#                     if len(rig_frames_data):
#                         frame_ids = [
#                             int(rig_frames_data[0, "value"]),
#                             int(rig_frames_data[-1, "value"]),
#                         ]

#         frames_dict = {f"{get_from}_frame_ids": frame_ids}
#         self._attrs_from_dict(frames_dict)
#         return frames_dict

#     def trial_data_from_logs(self, **wheel_kwargs) -> tuple[list, list]:
#         """
#         The main loop of parsing previously sliced rawdata to trial events
#         :return: A dictionary to be appended in the session dataframe
#         """
#         trial_log_data = {"trial_no": self.trial_no}

#         # state machine
#         state_dict = self.get_state_changes()
#         if state_dict["state_outcome"] is None:
#             return {**trial_log_data, **state_dict}
#         # screen
#         screen_dict = self.get_screen_events()
#         # vstim
#         vstim_dict = self.get_vstim_properties()
#         # wheel
#         wheel_dict = self.get_wheel_traces(**wheel_kwargs)
#         # lick
#         lick_dict = self.get_licks()
#         # reward
#         reward_dict = self.get_reward()
#         # opto
#         opto_dict = self.get_opto()
#         # camera frames
#         frames_dict = {}
#         for c in ["eyecam", "facecam", "onepcam"]:
#             tmp = self.get_frames(get_from=c)
#             frames_dict = {**frames_dict, **tmp}

#         trial_log_data = {
#             **trial_log_data,
#             **state_dict,
#             **screen_dict,
#             **vstim_dict,
#             **wheel_dict,
#             **lick_dict,
#             **reward_dict,
#             **opto_dict,
#             **frames_dict,
#         }

#         return trial_log_data
