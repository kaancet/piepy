import numpy as np
from ..utils import *
from ..core.trial import *


class VisualTrial(Trial):
    def __init__(self, trial_no: int, meta, logger) -> None:
        super().__init__(trial_no, meta, logger)
        
    def get_vstim_properties(self,ignore:list=None) -> dict:
        """ Extracts the necessary properties from vstim data """
        if ignore is None:
            ignore = []
        
        vstim = self.data['vstim']
        # vstim = vstim.drop_nulls(subset=['prob'])
        # this is an offline fix for a vstim logging issue where time increment messes up vstim logging
        vstim = vstim[:-1]
        
        vstim_dict = {}
        for col in vstim.columns:
            if col in ignore:
                continue
            if len(vstim.select(col).unique()) == 1:
                # if a column has all the same values, take the first entry of the column as the value
                # sf, tf, contrast, stim_side, correct, opto, opto_pattern should run through here
                vstim_dict[col] = vstim[0,col]
            elif len(vstim.select(col).unique()) > 1:
                # if different values exist in the column, take it as a list
                self.logger.error(f"{col} has multiple unique entries ({len(vstim.select(col).unique())}). This shouldn't be the case")
                vstim_dict[col] = vstim[col].to_list()
            else:
                vstim_dict[col] = None
        
        self._attrs_from_dict(vstim_dict)
        return vstim_dict
        
    def get_state_changes(self) -> dict:
        """ Looks at state changes in a given data slice and set class attributes according to them
        every key starting with t_ is an absolute time starting from experiment start
        """
        empty_log_data = {'t_trialstart' : self.t_trialstart, # this is an absolute value
                          'vstim_offset' : self.vstim_offset,
                          'state_offset' : self.state_offset,
                          't_stimstart' : None,
                          't_stimend': None}
        
        state_log_data = {**empty_log_data}
        # in the beginning check if state data is complete
        if 'trialend' not in self.data['state']['transition'].to_list():
            self._attrs_from_dict(empty_log_data)
            return empty_log_data
        
        # get time changes from statemachine
        state_log_data['t_stimstart'] = self.data['state'].filter(pl.col('transition')=='stimstart')[0,'corrected_elapsed']
        state_log_data['t_stimend'] = self.data['state'].filter(pl.col('transition')=='stimend')[0,'corrected_elapsed']
        state_log_data['t_trialend'] = self.t_trialend
        
        self._attrs_from_dict(state_log_data)
        return state_log_data
        
    def trial_data_from_logs(self) -> dict:
        """ 
        :return: A dictionary to be appended in the session dataframe
        """
        trial_log_data = {'trial_no':self.trial_no}
        
        #state machine
        state_dict = self.get_state_changes()
        # screen
        screen_dict = self.get_screen_events()
        # vstim
        vstim_dict = self.get_vstim_properties()
        # camera frames
        frames_dict = {}
        for c in ['eyecam','facecam','onepcam']:
            tmp = self.get_frames(get_from=c)
            frames_dict = {**frames_dict, **tmp}
        
        trial_log_data = {**trial_log_data,
                          **state_dict,
                          **screen_dict,
                          **vstim_dict,
                          **frames_dict}
        
        return trial_log_data