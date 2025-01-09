import numpy as np
import polars as pl
import patito as pt
from typing import Any
from .exceptions import StateMachineError


class Trial(pt.Model):
    """This is the Trial dataframe model class that ends up as a row in session data frame after being validated.
    This is the most basic form of a Trial, which has a trial no, start and an end"""

    # provided in instantiation
    trial_no: int = pt.Field(gt=0, frozen=True, dtype=pl.UInt64)
    t_trialstart: float = pt.Field(gt=0, dtype=pl.Float64)
    t_trialend: float = pt.Field(gt=0, dtype=pl.Float64)


class TrialHandler:
    """A class that houses methods for parsing/checking/filling the Trial DataFrame class"""

    def __init__(self) -> None:
        self.data = {}
        self.set_model(Trial)

    def set_model(self, model: pt.Model) -> None:
        """Sets the model the handler will use to validate the trial"""
        self.trial_model = model

    def _update_model(self) -> None:
        """Updates the model with new keys in the dictionary, if there are any new ones"""

        def list_field_fixer(field_val) -> tuple[type, Any]:
            """Fixes the typing and defult value for lists"""
            if isinstance(field_val, list):
                return (list[float], [])
            else:
                return (type(field_val), None)

        _new_cols = {
            k: list_field_fixer(v)
            for k, v in self._trial.items()
            if k not in self.trial_model.columns
        }
        self.trial_model = self.trial_model.with_fields(**_new_cols)

    def _update_and_return(self, return_as: str = "dict") -> pt.DataFrame | dict | list:
        """First validates, then returns the self._trial in the form given in return_as"""
        # update the model with new vstim columns
        self._update_model()
        _t = pt.DataFrame(self._trial)

        # validate the data
        valid_data = _t
        #   .set_model(self.trial_model) # Specify the schema of the given data frame
        #   .derive() # Derive the columns that have derived_from in their Field
        #   .drop() # Drop the columns that are not a part of the model definition
        #   .cast() # Cast the columns that have dtype argument in their Field
        #   .fill_null(strategy="defaults") # Fill missing values with the default values specified in the schema
        #   .validate() # Assert that the data frame now complies with the schema
        #   )

        if return_as == "df":
            return valid_data
        elif return_as == "dict":
            return valid_data.to_dict(as_series=False)
        elif return_as == "list":
            return valid_data.to_dicts()
        else:
            raise ValueError(
                f"Unexpected value for return_as. Got {return_as}, expected values are: df, dict_of_lists, list_of_dicts"
            )

    def init_trial(self) -> None:
        """Creates an _trial object filled with None values"""
        self._trial = {k: None for k in self.trial_model.columns}

    def get_trial(
        self, trial_no: int, rawdata: dict, return_as: str = "dict"
    ) -> pt.DataFrame | dict | list | None:
        """Main function that is called from outside, sets the trial, validates data type and returns it"""
        self.init_trial()
        _is_trial_set = self.set_trial(trial_no, rawdata)

        if not _is_trial_set:
            return None
        else:
            return self._update_and_return(return_as)

    def set_trial(self, trial_no: int, rawdata: dict) -> bool:
        """Sets the trialstart and end times, and sets the data slice corresponding to the current trial"""
        self._trial["trial_no"] = trial_no

        _state = rawdata["statemachine"].filter(pl.col("trialNo") == trial_no)
        _state_transitions = _state["transition"].to_list()

        # check if trial is complete
        if self.is_trial_complete(_state_transitions):
            self._trial["t_trialstart"] = _state.filter(
                pl.col("transition") == "trialstart"
            )[0, "elapsed"]
            if "trialend" in _state_transitions:
                self._trial["t_trialend"] = _state.filter(
                    pl.col("transition") == "trialend"
                )[0, "elapsed"]
            else:
                self._trial["t_trialend"] = _state.filter(
                    pl.col("transition") == "stimtrialstart"
                )[0, "elapsed"]
        else:
            return False

        # fill a dictionary with slices from the dataframe that correspond to the trial
        for k, v in rawdata.items():
            if k == "statemachine":
                self.data["state"] = _state
                continue
            if not v.is_empty():
                if "presentTime" not in v.columns:
                    temp_v = v.filter(
                        pl.col("duinotime").is_between(
                            self._trial["t_trialstart"], self._trial["t_trialend"]
                        )
                    )
                else:
                    if k == "vstim":
                        # for vstim use total_istim to get trial related data,
                        # timing is not very robust
                        # temp_v = v.filter(pl.col("total_iStim") == trial_no)
                        temp_v = v.filter(
                            (pl.col("presentTime") * 1000).is_between(
                                self._trial["t_trialstart"], self._trial["t_trialend"]
                            )
                        )

                self.data[k] = temp_v
        return True

    # TODO: Semantically, this function can be somewhere else, but where?
    def set_opto(self) -> None:
        """Sets the opto boolean from opto slice from riglog"""
        opto_array = self._get_rig_event("opto")

        if opto_array is not None:
            _is_opto = True
            _opto_time = [opto_array[:, 0].tolist()]
        else:
            _is_opto = False
            _opto_time = [[]]
        self._trial["opto"] = _is_opto
        self._trial["opto_pulse"] = _opto_time

    def set_frame_endpoints(self, imaging_mode: str, epoch_enpoints: list) -> tuple:
        """Gets the start and end frame ids for the provided imaging mode, given that it exists in the logged data
        NOTE: even if there's no actual recording for onepcam through labcams(i.e. the camera is running in the labcams GUI without saving),
        if there is onepcam frame TTL signals coming into the Arduino, it will save them as pulses.
        This will lead to having frame_ids column to be created BUT there will be no actual camera frames recorded that correspond to the frame signals
        """
        frames_data = self._get_rig_event(imaging_mode)
        frames_data = frames_data.filter(
            (pl.col("duinotime") >= epoch_enpoints[0])
            & (pl.col("duinotime") <= epoch_enpoints[0])
        )
        if len(frames_data):
            frame_ids = (
                int(frames_data[0, "value"]),
                int(frames_data[-1, "value"]),
            )
        else:
            frame_ids = None

        self._trial[f"{imaging_mode}_frame"] = frame_ids

    @staticmethod
    def is_trial_complete(transitions: list) -> bool:
        """Check if the trial starts and ends correctly, using transitions from the state machine"""

        # Trial start
        if "trialstart" not in transitions:
            raise StateMachineError("NO TRIALSTART FOR STATEMACHINE!!")

        # Trial end
        _trial_ended = False
        if "trialend" in transitions or "stimtrialend" in transitions:
            _trial_ended = True

        return _trial_ended

    # =====================================
    # HARDWARE EVENTS RELATED TO THE TRIALS
    # =====================================
    def _get_rig_event(self, event_name: str) -> np.ndarray:
        """Gets the hardware TTL entries(duinotime,value) from given event_name"""
        if event_name not in self.data.keys():
            # raise LogTypeMissingError(f"No hardware event logged with the name: {event_name}")
            # display(f"No hardware event logged with the name: {event_name}")
            return None

        event_data = self.data[event_name]
        if not event_data.is_empty():
            event_arr = event_data.select(["duinotime", "value"]).to_numpy()
        else:
            event_arr = None
        return event_arr
