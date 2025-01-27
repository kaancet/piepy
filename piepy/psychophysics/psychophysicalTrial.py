import patito as pt
import polars as pl

from ..core.trial import Trial, TrialHandler


class PsychophysicalTrial(Trial):
    reward: list[float] | None = pt.Field(default=[], dtype=pl.List(pl.Float64))
    lick: list[float] | None = pt.Field(default=[], dtype=pl.List(pl.Float64))


class PsychophysicalTrialHandler(TrialHandler):
    def __init__(self):
        super().__init__()
        self._trial = {k: None for k in PsychophysicalTrial.columns}
        self.was_screen_off = True  # flag for not having OFF pulse in screen data
        self.set_model(PsychophysicalTrial)

    def get_trial(
        self, trial_no: int, rawdata: dict, return_as: str = "dict"
    ) -> pt.DataFrame | dict | list:
        """Main function that is called from outside, sets the trial, validates data type and returns it"""
        self.init_trial()
        self.set_trial(trial_no, rawdata)

        self.set_licks()
        self.set_reward()

        return self._update_and_return()

    def get_state_changes(self) -> None:
        """
        This method will be rewritten in child classes as it is experiment dependent
        """
        pass

    def set_licks(self) -> None:
        """Sets the lick timings as a list"""
        lick_array = self._get_rig_event("lick")
        if lick_array is not None:
            self._trial["lick"] = [lick_array[:, 0].tolist()]

    def set_reward(self) -> None:
        """Sets the reward timings as a list"""
        reward_array = self._get_rig_event("reward")    
        if reward_array is not None:
            self._trial["reward"] = [reward_array[0,:].tolist()]
