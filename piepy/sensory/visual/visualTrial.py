import polars as pl
import patito as pt
from ...core.exceptions import ScreenPulseError
from ...core.utils import nonan_unique
from ...core.trial import Trial, TrialHandler


class VisualTrial(Trial):
    t_vstimstart_rig: float | None = pt.Field(default=None, gt=0, dtype=pl.Float64)
    t_vstimend_rig: float | None = pt.Field(default=None, gt=0, dtype=pl.Float64)
    t_vstimstart: float | None = pt.Field(default=None, gt=0, dtype=pl.Float64)
    t_vstimend: float | None = pt.Field(default=None, gt=0, dtype=pl.Float64)
    vstim_time_diff: float | None = pt.Field(default=None, dtype=pl.Float64)
    state_time_diff: float | None = pt.Field(default=None, dtype=pl.Float64)


class VisualTrialHandler(TrialHandler):
    def __init__(self) -> None:
        super().__init__()
        self.was_screen_off = True  # flag for not having OFF pulse in screen data
        self.set_model(VisualTrial)

    def get_trial(
        self, trial_no: int, rawdata: dict, return_as: str = "dict"
    ) -> pt.DataFrame | dict | list:
        """Main function that is called from outside, sets the trial, validates data type and returns it"""
        self.init_trial()
        self.set_trial(trial_no, rawdata)
        self.set_screen_events()  # should return a 2x2 matrix, first column is timings for screen ON and OFF.
        self.sync_timeframes()  # syncs the state and vstim log times, using screen ONSET

        # NOTE: sometimes due to state machine logic, the end of trial will be end of stimulus
        # this causes a trial to have a single screen event (ON) to be parsed into a given trial
        # to remedy this, we check the screen data after syncing the timeframes of rig(arduoino) and statemachine(python)
        if not self.was_screen_off:
            self.recheck_screen_events(rawdata["screen"])

        self.set_vstim_properties()  # should be run after sync_timeframes

        return self._update_and_return(return_as)

    def set_screen_events(self) -> None:
        """Sets the visual stimulus start and end times from screen photodiode events"""
        screen_array = self._get_rig_event("screen")

        if screen_array is None:
            self._trial["t_vstimstart_rig"] = None
            self._trial["t_vstimend_rig"] = None

        if screen_array is not None and len(screen_array) == 2:
            # This is the correct stim ON/OFF scenario
            self._trial["t_vstimstart_rig"] = screen_array[0, 0]
            self._trial["t_vstimend_rig"] = screen_array[1, 0]
        elif screen_array is not None and len(screen_array) == 1:
            self._trial["t_vstimstart_rig"] = screen_array[0, 0]
            self.was_screen_off = False
        elif screen_array is not None:
            raise ScreenPulseError(
                f"[TRIAL-{self._trial['trial_no']}] Funky screen pulses with length {len(screen_array)}"
            )

    def recheck_screen_events(self, screen_data: pl.DataFrame) -> None:
        """Takes the screen dataframe to check the screen events once again
        NOTE: This function should be called after sync_timeframes, so it can check with the corrected timeframe
        """
        _found_it = False
        _screen_new = screen_data.filter(
            (pl.col("duinotime") >= self._trial["t_trialstart"])
            & (pl.col("duinotime") <= self._trial["t_trialend"])
        )
        if _screen_new is None:
            return None

        if _screen_new is not None and len(_screen_new) == 2:
            # ON and OFF values, check they have the same value for same stim
            if _screen_new.n_unique("value") == 1:
                # found new screen event, add it
                self._trial["t_vstimend_rig"] = _screen_new[1, "duinotime"]
                _found_it = True

        if not _found_it:
            raise ScreenPulseError(
                f"""[TRIAL-{self._trial['trial_no']}] Only 1 screen event for stimulus ON.\n
                                   Tried checking after syncing rig and state machine times, issue persists..."""
            )

    def sync_timeframes(self) -> None:
        """Syncs the timeframes according to visual stimulus appearance from screen events"""
        _state = self.data["state"]
        _vstim = self.data["vstim"]

        _rig_onset = self._trial["t_vstimstart_rig"]
        if _rig_onset is None:
            # no screen event to sync
            vstim_diff = 0.0
            state_diff = 0.0
        else:
            _state_onset = _state.filter(pl.col("transition") == "stimstart")[
                0, "elapsed"
            ]

            # some stimpy version has inverted photostim values, so adaptively set it
            # first entry is always the inverse of "stim_on"
            photo_stim = not _vstim["photo"].drop_nulls()[0]
            try:
                _vstim_onset = (
                    _vstim.filter(pl.col("photo") == photo_stim)[0, "presentTime"] * 1000
                )  # ms
            except Exception:
                _vstim_onset = (
                    _vstim.filter(pl.col("presentTime") >= _rig_onset)[0, "presentTime"]
                    * 1000
                )  # ms

            # if difference is negative, that means the rig_onset time happened after the python timing
            vstim_diff = float(round(_vstim_onset - _rig_onset, 3))
            state_diff = float(round(_state_onset - _rig_onset, 3))

        # update the data
        _state = _state.with_columns(
            (pl.col("elapsed") - state_diff).alias("corrected_elapsed")
        )
        self.data["state"] = _state

        _vstim = _vstim.with_columns(
            (pl.col("presentTime") * 1000 - vstim_diff).alias("corrected_presentTime")
        )
        self.data["vstim"] = _vstim

        # update the trial endpoints
        self._trial["t_trialstart"] = _state.filter(pl.col("transition") == "trialstart")[
            0, "corrected_elapsed"
        ]
        self._trial["t_trialend"] = _state.filter(
            pl.col("transition").str.contains("trialend")
        )[0, "corrected_elapsed"]

        # add the time difference values to _trial
        self._trial["vstim_time_diff"] = vstim_diff
        self._trial["state_time_diff"] = state_diff

    def set_vstim_properties(self) -> None:
        """Extracts the necessary properties from vstim data"""
        ignore = ["code", "presentTime", "stim_idx", "duinotime", "photo", "reward"]

        _vstim = self.data["vstim"]
        if self._trial["t_vstimstart_rig"] is None:
            _vstim = _vstim.filter(pl.col("photo") != self.data["vstim"][0, "photo"])
        else:
            _vstim = _vstim.filter(
                (
                    pl.col("corrected_presentTime").is_between(
                        self._trial["t_vstimstart_rig"], self._trial["t_vstimend_rig"]
                    )
                )
            )

        for col in _vstim.columns:
            if col in ignore:
                continue

            _entries = _vstim[col].drop_nulls().to_list()

            if col in ["iStim", "iTrial", "total_iStim"]:
                self._trial[col] = int(_entries[0]) if len(_entries) else None

            else:
                if len(_entries):
                    if len(nonan_unique(_entries)) == 1:
                        self._trial[col] = [[_entries[0]]]
                    else:
                        self._trial[col] = [_entries]
                else:
                    self._trial[col] = None
