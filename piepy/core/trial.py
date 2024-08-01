import polars as pl
import numpy as np
from .exceptions import *


class Trial:
    __slots__ = [
        "trial_no",
        "data",
        "column_keys",
        "t_trialstart",
        "t_trialend",
        "db_interface",
        "total_trial_count",
        "reward_ms_per_ul",
        "meta",
        "logger",
    ]

    def __init__(self, trial_no: int, meta, logger) -> None:
        self.trial_no = trial_no
        self.meta = meta
        self.data = {}
        self.logger = logger
        self.reward_ms_per_ul = 0
        self.logger.set_msg_prefix(f"TRIAL-[{self.trial_no}]")

    def __repr__(self) -> str:
        rep = f"""Trial No :{self.trial_no}
        {self.data.get("state",None)}"""
        return rep

    def _attrs_from_dict(self, log_data: dict) -> None:
        """
        Creates class attributes from a dictionary
        :param log_data: any dictionary  to set class attributes
        """
        for k, v in log_data.items():
            setattr(self, k, v)

    def is_trial_complete(self, rawdata: dict) -> bool:
        """Sets state data, t_trialstart and t_trialend for data slices to be extracted
        Returns False if trialend and trialstart not present"""
        states = rawdata.get("statemachine")

        self.data["state"] = states.filter(pl.col("trialNo") == self.trial_no)
        _state_transitions = self.data["state"]["transition"].to_list()

        # Trial start
        if "trialstart" not in _state_transitions:
            raise StateMachineError(
                f"[Trial {self.trial_no}] NO TRIALSTART FOR STATEMACHINE "
            )
        else:
            self.t_trialstart = self.data["state"].filter(
                pl.col("transition") == "trialstart"
            )[0, "elapsed"]

        # trial end
        does_trial_end = False
        if "trialend" in _state_transitions:
            self.t_trialend = self.data["state"].filter(
                pl.col("transition") == "trialend"
            )[0, "elapsed"]
            does_trial_end = True
        elif "stimtrialend" in _state_transitions:
            self.t_trialend = self.data["state"].filter(
                pl.col("transition") == "stimtrialend"
            )[0, "elapsed"]
            does_trial_end = True

        return does_trial_end

    def set_data_slices(self, rawdata: dict) -> bool:
        """
        Extracts the relevant portion from each data, using the statemachine log
        WARNING: statemachine log is python timing, so it's likely that it will be not as accurate as the Arduino timing

        rawdata(dict): Rawdata dictionary including all of the logs
        Returns: DataFrame slice of corresponding trial no
        """
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

            # riglog
            for k, v in rawdata.items():
                if k in ["statemachine", "screen", "vstim"]:
                    # skip because already looked into it
                    continue
                if not v.is_empty():
                    if k in rig_cols:
                        temp_v = v.filter(
                            (pl.col("duinotime") >= self.t_trialstart)
                            & (pl.col("duinotime") <= self.t_trialend)
                        )

                    self.data[k] = temp_v
            return True
        else:
            return False

    # ======
    # Abstract methods to be overwritten
    # ======

    def correct_timeframes(self, rawdata: dict) -> None:
        """ """
        pass

    def get_state_cahnges(self) -> None:
        """ """
        pass

    # ======
    # Experiment type independentrial event extraction
    # ======

    def get_licks(self) -> dict:
        """Extracts the lick data from slice"""
        lick_data = self.data.get("lick", None)
        if lick_data is not None:
            if len(lick_data):
                # lick_arr = np.array(lick_data[['duinotime', 'value']])
                lick_arr = lick_data.select(["duinotime", "value"]).to_series().to_list()
            else:
                if self.state_outcome == 1:
                    self.logger.error(f"Empty lick data in correct trial")
                lick_arr = None
        else:
            self.logger.warning(f"No lick data in trial")
            lick_arr = None

        lick_dict = {"lick": lick_arr}
        self._attrs_from_dict(lick_dict)
        return lick_dict

    def get_reward(self) -> dict:
        """Extracts the reward clicks from slice"""

        reward_data = self.data.get("reward", None)
        if reward_data is not None:
            # no reward data, shouldn't happen a lot, usually in shitty sessions
            reward_arr = reward_data.select(["duinotime", "value"]).to_numpy()
            if len(reward_arr):
                try:
                    reward_amount_uL = np.unique(self.data["vstim"]["reward"])[0]
                except:
                    reward_amount_uL = self.meta.rewardSize
                    self.logger.warning(
                        f"No reward logged from vstim, using rewardSize from prot file"
                    )
                reward_arr = np.append(reward_arr, reward_arr[:, 1])
                reward_arr[1] = reward_amount_uL
                reward_arr = reward_arr.tolist()
                # reward is a 3 element array: [time,value_il, value_ms]
            else:
                reward_arr = None
        else:
            reward_arr = None

        reward_dict = {"reward": reward_arr}
        self._attrs_from_dict(reward_dict)
        return reward_dict

    def get_opto(self) -> dict:
        """Extracts the opto boolean from opto slice from riglog"""
        if self.meta.opto:
            if "opto" in self.data.keys() and len(self.data["opto"]):
                opto_arr = self.data["opto"].select(["duinotime"]).to_numpy()
                if len(opto_arr) > 1 and self.meta.opto_mode == 0:
                    self.logger.warning(
                        f"Something funky happened with opto stim, there are {len(opto_arr)} pulses"
                    )
                    opto_arr = opto_arr[0]
                elif len(opto_arr) > 1 and self.meta.opto_mode == 1:
                    opto_arr = opto_arr[:, 0]
                is_opto = True
                opto_arr = opto_arr.tolist()
            else:
                # is_opto = int(bool(vstim_dict.get('opto',0)) or bool(len(opto_pulse)))
                is_opto = False
                opto_arr = []
                if self.opto_pattern is not None and self.opto_pattern >= 0:
                    self.logger.warning(
                        "stimlog says opto, but no opto logged in riglog, using screen event as time!!"
                    )
                    is_opto = True
                    opto_arr = [[self.t_stimstart_rig]]
        else:
            is_opto = False
            opto_arr = [[]]

        opto_dict = {"opto": is_opto, "opto_pulse": opto_arr}
        self._attrs_from_dict(opto_dict)
        return opto_dict

    def get_screen_events(self) -> dict:
        """Gets the screen pulses from rig data"""
        screen_dict = {"t_stimstart_rig": None, "t_stimend_rig": None}

        if "screen" in self.data.keys():
            screen_data = self.data["screen"]
            screen_arr = screen_data.select(["duinotime", "value"]).to_numpy()

            if len(screen_arr) == 1:
                self.logger.error(
                    "Only one screen event! Stimulus appeared but not dissapeared?"
                )
                # assumes the single pulse is stim on
                screen_dict["t_stimstart_rig"] = screen_arr[0, 0]
            elif len(screen_arr) > 2:
                # TODO: 3 SCREEN EVENTS WITH OPTO BEFORE STIMULUS PRESENTATION?
                self.logger.error(
                    "More than 2 screen events per trial, this is not possible"
                )
            elif len(screen_arr) == 0:
                self.logger.critical("NO SCREEN EVENT FOR STIMULUS TRIAL!")
            else:
                # This is the correct stim ON/OFF scenario
                screen_dict["t_stimstart_rig"] = screen_arr[0, 0]
                screen_dict["t_stimend_rig"] = screen_arr[1, 0]

        self._attrs_from_dict(screen_dict)
        return screen_dict

    def get_frames(self, get_from: str = None, **kwargs) -> dict:
        """Extracts the frames from designated imaging mode, returns None if no"""

        frame_ids = []
        if not self.meta.imaging_mode is None:
            if get_from in self.data.keys():
                """
                NOTE: even if there's no actual recording for onepcam through labcams(i.e. the camera is running in the labcams GUI without saving),
                if there is onepcam frame TTL signals coming into the Arduino it will save them.
                This will lead to having onepcam_frame_ids column to be created but there will be no actual tiff files.
                """
                rig_frames_data = self.data[
                    get_from
                ]  # this should already be the frames of trial dur

                if self.t_stimstart_rig is not None:
                    # get stim present slice
                    rig_frames_data = rig_frames_data.filter(
                        (pl.col("duinotime") >= self.t_stimstart_rig)
                        & (pl.col("duinotime") <= self.t_stimend_rig)
                    )

                    if len(rig_frames_data):
                        frame_ids = [
                            int(rig_frames_data[0, "value"]),
                            int(rig_frames_data[-1, "value"]),
                        ]

                    else:
                        self.logger.critical(
                            f"{get_from} no camera pulses recorded during stim presentation!!! THIS IS BAD!"
                        )
        frames_dict = {f"{get_from}_frame_ids": frame_ids}

        self._attrs_from_dict(frames_dict)
        return frames_dict
