import numpy as np
from ...utils import *
from ...core.trial import *


class PassiveTrial(Trial):
    def __init__(self, trial_no: int, meta, logger) -> None:
        super().__init__(trial_no, meta, logger)

    def correct_timeframes(self, rawdata: dict) -> None:
        """Corrects the timing of statemachine and vstim using the stimulus onset in screen log"""
        super().correct_timeframes(rawdata)

        state = self.data["state"]
        screen = rawdata.get("screen")

        # use trial no to get vstim data slice
        vstim = rawdata["vstim"].filter(pl.col("total_iStim") == self.trial_no)

        # below is a failsafe there should be exactly 2x trial count screen events
        # idx = 1 if screen[0,'value'] == 0 else 0 # sometimes screen events have a 0 value entry at the beginning
        if len(screen) / 2 != (self.meta.opts["nTrials"] * len(self.meta.params)):
            # faulty signaling, usually happens at the start of the experiment, # get screen events that are after blank duration
            screen = screen.filter(pl.col("duinotime") > self.meta.blankDuration * 1000)
            # reset the value column
            screen = screen.with_columns(pl.col("value") - (screen[0, "value"] - 1))

        # for passive trials the trialNo should match with "value" of screen pulse
        screen_slice = screen.filter(pl.col("value") == self.trial_no)
        rig_stim_onset = screen_slice[0, "duinotime"]
        state_stim_onset = state.filter(pl.col("transition") == "stimstart")[0, "elapsed"]

        # some stimpy version has inverted photostim values, so adaptively set it
        photo_stim = not vstim["photo"].drop_nulls()[0]

        try:
            vstim_onset = (
                vstim.filter(pl.col("photo") == photo_stim)[0, "presentTime"] * 1000
            )  # ms
        except:
            vstim_onset = (
                vstim.filter(pl.col("presentTime") >= rig_stim_onset)[0, "presentTime"]
                * 1000
            )  # ms
            self.logger.error(f"No photosensor logged in vstim!!")

        self.vstim_offset = vstim_onset - rig_stim_onset
        self.state_offset = state_stim_onset - rig_stim_onset

        # if offset is negative, that means the rig_onset time happened after the python timing
        self.data["screen"] = screen_slice
        self.data["state"] = state.with_columns(
            (pl.col("elapsed") - self.state_offset).alias("corrected_elapsed")
        )
        self.data["vstim"] = vstim.with_columns(
            (pl.col("presentTime") - self.vstim_offset).alias("corrected_presentTime")
        )

        # correct the trialstart and end times
        self.t_trialstart = self.data["state"].filter(
            pl.col("transition") == "trialstart"
        )[0, "corrected_elapsed"]
        try:
            self.t_trialend = self.data["state"].filter(
                pl.col("transition") == "trialend"
            )[0, "corrected_elapsed"]
        except:
            self.t_trialend = self.data["state"].filter(
                pl.col("transition") == "stimtrialend"
            )[0, "corrected_elapsed"]

    def set_data_slices(self, rawdata: dict) -> None:
        """Passive viewing experiment use the screen events to slice the data rather than states"""
        rig_cols = [
            "screen",
            "imaging",
            "position",
            "lick",
            "button",
            "reward",
            "lap",
            "facecam",
            "eyecam",
            "onepcam",
            "act0",
            "act1",
            "opto",
        ]

        if self.is_trial_complete(rawdata):
            self.correct_timeframes(rawdata)

            # rig and stimlog
            for k, v in rawdata.items():
                if k in ["statemachine", "screen", "vstim"]:
                    # skip because already looked into it
                    continue
                if not v.is_empty():
                    t_start = self.t_trialstart
                    t_end = self.t_trialend

                    # rig logs
                    if k in rig_cols:
                        temp_v = v.filter(
                            (pl.col("duinotime") >= self.t_trialstart)
                            & (pl.col("duinotime") <= self.t_trialend)
                        )

                    self.data[k] = temp_v

    def get_vstim_properties(self, ignore: list = None) -> dict:
        """Extracts the necessary properties from vstim data"""
        if ignore is None:
            ignore = ["code", "presentTime", "stim_idx", "duinotime", "photo"]

        vstim = self.data["vstim"]

        vstim = vstim.filter((pl.col("corrected_presentTime")) < self.t_stimend_rig)
        vstim = vstim[
            :-5
        ]  # 5 is arbitrary to make sure no extra rows from next trial due to timing imperfections

        vstim_dict = {}
        for col in vstim.columns:
            if col in ignore:
                continue

            _entries = vstim[col].drop_nulls().to_list()

            if col in ["iStim", "iTrial", "total_iStim"]:
                vstim_dict[col] = _entries[0]
            else:
                vstim_dict[col] = _entries

        self._attrs_from_dict(vstim_dict)
        return vstim_dict

    def get_state_changes(self) -> dict:
        """Looks at state changes in a given data slice and set class attributes according to them
        every key starting with t_ is an absolute time starting from experiment start
        """
        empty_log_data = {
            "t_trialstart": self.t_trialstart,  # this is an absolute value
            "vstim_offset": self.vstim_offset,
            "state_offset": self.state_offset,
            "t_stimstart": None,
            "t_stimend": None,
        }

        state_log_data = {**empty_log_data}
        # in the beginning check if state data is complete
        if (
            "trialend" not in self.data["state"]["transition"].to_list()
            and "stimtrialend" not in self.data["state"]["transition"].to_list()
        ):
            self._attrs_from_dict(empty_log_data)
            return empty_log_data

        # get time changes from statemachine
        state_log_data["t_stimstart"] = self.data["state"].filter(
            pl.col("transition") == "stimstart"
        )[0, "elapsed"]
        state_log_data["t_stimend"] = self.data["state"].filter(
            (pl.col("transition") == "stimend") | (pl.col("transition") == "stimtrialend")
        )[0, "elapsed"]

        state_log_data["t_trialend"] = self.data["state"].filter(
            (pl.col("transition") == "trialend")
            | (pl.col("transition") == "stimtrialend")
        )[0, "elapsed"]

        if "t_trialend" not in state_log_data.keys():
            print("iuasdfdfsdfasdf")
        self._attrs_from_dict(state_log_data)
        return state_log_data

    def trial_data_from_logs(self) -> dict:
        """
        :return: A dictionary to be appended in the session dataframe
        """
        trial_log_data = {"trial_no": self.trial_no}

        # state machine
        state_dict = self.get_state_changes()
        # screen
        screen_dict = self.get_screen_events()
        # vstim
        vstim_dict = self.get_vstim_properties()
        # camera frames
        frames_dict = {}
        for c in ["eyecam", "facecam", "onepcam", "imaging"]:  # imaging is 2P
            tmp = self.get_frames(get_from=c)
            frames_dict = {**frames_dict, **tmp}

        trial_log_data = {
            **trial_log_data,
            **state_dict,
            **screen_dict,
            **vstim_dict,
            **frames_dict,
        }

        return trial_log_data
