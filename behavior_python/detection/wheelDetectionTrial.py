import numpy as np
from ..utils import *
from ..core.trial import *


class WheelDetectionTrial(Trial):
    def __init__(self,trial_no:int,log_column_keys:dict,meta) -> None:
        super().__init__(trial_no,log_column_keys,meta)
        
    def get_vstim_props(self,early_flag:bool) -> dict:
        """ Extracts the necessary properties from vstim data"""
        ignore = ['iTrial','photo','code','presentTime']
        vstim_dict = {}
        vstim = self.data['vstim']
        
        # this is an offline fix for a vstim logging issue where time increment messes up vstim logging
        vstim = vstim[:-1]
        
        if vstim.empty:
            early_flag = -1
        for col in vstim.columns:
            if col in ignore:
                continue

            temp_col = vstim[col]
            # if a column has all the same values, take the first entry of the column as the value
            # sf, tf, contrast, stim_side, correct, opto, opto_pattern should run through here
            uniq = np.unique(temp_col)
            uniq = uniq[~np.isnan(uniq)]
            
            # failsafe for animal not moving the wheel
            # TODO: maybe not do this to save memory and time, and keep non-moved trials' values as a single value?
            if len(uniq) == 1 and col != 'posx_r' and col != 'posx_l':
                vstim_dict[col] = temp_col.iloc[0]
            # if different values exist in the column, take it as a list
            # stim_pos runs through here 
            else:
                vstim_dict[col] = np.array(temp_col)
                
        if early_flag==-1:
            vstim_dict['contrast'] = np.nan
            vstim_dict['spatial_freq'] = np.nan
            vstim_dict['temporal_freq'] = np.nan
            vstim_dict['stim_side'] = np.nan
            vstim_dict['opto_pattern'] = np.nan
        else:
            vstim_dict['contrast'] = vstim_dict['contrast_r'] if vstim_dict['correct'] else vstim_dict['contrast_l']
            vstim_dict['spatial_freq'] = vstim_dict['sf_r'] if vstim_dict['correct'] else vstim_dict['sf_l']
            vstim_dict['temporal_freq'] = vstim_dict['tf_r'] if vstim_dict['correct'] else vstim_dict['tf_l']
            # vstim_dict['stim_side'] = vstim_dict['stim_pos'][0]
            vstim_dict['stim_side'] = vstim_dict['posx_r'][0] if vstim_dict['correct'] else vstim_dict['posx_l'][0]
            if vstim_dict['contrast'] == 0:
                vstim_dict['stim_side'] = 0 # no meaningful side when 0 contrast
        return vstim_dict
    
    def get_wheel_pos(self,time_anchor:float) -> np.ndarray:
        """ Extracts the wheel trajectories and resets the positions according to time_anchor"""
        wheel_deg_per_tick = (self.meta.wheelGain * WHEEL_CIRCUM) / WHEEL_TICKS_PER_REV
        
        wheel_data = self.data['position']
        wheel_arr = np.array(wheel_data[['duinotime', 'value']])
        
        # resetting the wheel position so the 0 point is aligned with trialstart and converting encoder ticks into degrees
        # also resetting the time frame into the trial itself rather than the whole session
        if len(wheel_arr):
            reset_idx = find_nearest(wheel_arr[:,0],time_anchor)[0]
            wheel_arr[:,1] = np.apply_along_axis(reset_wheel_pos,0,wheel_arr[:,1],reset_idx) * wheel_deg_per_tick
            wheel_arr[:,0] = np.apply_along_axis(lambda x: x-time_anchor,0,wheel_arr[:,0])
        return wheel_arr
    
    def trial_data_from_logs(self) -> list:
        """ Iterates over each state change in a DataFrame slice that belongs to one trial(and corrections for pyvstim)
            Returns a list of dictionaries that have data parsed from stimlog and riglog
        """
        trial_log_data = {'trial_no': int(self.trial_no)}
        trial_log_data['isCatch'] = 0
        # iterrows is faster for small DataFrames
        for _,row in self.data['state'].iterrows():
            curr_trans = row['transition']
            #trial start
            if curr_trans == 'trialstart':
                trial_log_data['trial_start'] = row[self.column_keys['elapsed']]
                
            # cue start
            elif curr_trans == 'cuestart':
                trial_log_data['cue_start'] = row[self.column_keys['elapsed']]
                if 'blankDuration' in self.column_keys.keys(): 
                     trial_log_data['blank_time'] = row[self.column_keys['blankDuration']]
                else:
                    #old logging for some sessions
                    trial_log_data['blank_time'] = row[self.column_keys['trialType']]  
                trial_log_data['openstart_absolute'] = row[self.column_keys['elapsed']]
            
            # stim start
            elif curr_trans == 'stimstart':
                # get the stim start from riglog
                if 'screen' in self.data.keys():
                    if not self.data['screen'].empty:
                        trial_log_data['stim_start_rig'] = self.data['screen']['duinotime'].iloc[0]
                trial_log_data['stim_start'] = row[self.column_keys['elapsed']]
                
            # correct 
            elif curr_trans == 'correct':
                trial_log_data['response_latency'] = row[self.column_keys['stateElapsed']]        
                trial_log_data['answer'] = 1


            # incorrect(noanswer)
            elif curr_trans == 'incorrect':
                if row[self.column_keys['stateElapsed']] <= 150:
                    # means too fast answer, classified as an early
                    trial_log_data['response_latency'] = row[self.column_keys['stateElapsed']] + trial_log_data['blank_time']
                    trial_log_data['answer'] = -1
                else:
                    trial_log_data['response_latency'] = row[self.column_keys['stateElapsed']]
                    trial_log_data['answer'] = 0
                    
            # early
            elif curr_trans == 'earlyanswer':
                trial_log_data['response_latency'] = row[self.column_keys['stateElapsed']]
                trial_log_data['answer'] = -1
                
            # catch
            elif curr_trans == 'catch':
                trial_log_data['isCatch'] = 1
                # technically there is no answer
                trial_log_data['answer'] = 0
                
            # stim dissappear
            elif curr_trans == 'stimendcorrect' or curr_trans == 'stimendincorrect':
                if 'screen' in self.data.keys():
                    if not self.data['screen'].empty:
                        trial_log_data['stim_end_rig'] = self.data['screen']['duinotime'].iloc[1]

                trial_log_data['stim_end'] = row[self.column_keys['elapsed']]

            # correction or trial end
            elif curr_trans == 'correction' or curr_trans == 'trialend':
                trial_log_data['trial_end'] = row[self.column_keys['elapsed']]
                
                vstim_log = self.get_vstim_props(trial_log_data['answer'])
                
                rig_logs = {}
                
                if 'stim_start_rig' in trial_log_data.keys():
                    # stim trials have their wheels reset on cue start
                    rig_logs['wheel'] = self.get_wheel_pos(trial_log_data['stim_start_rig'])
                else:
                    # early trials have their wheels reset on cue start
                    rig_logs['wheel'] = self.get_wheel_pos(trial_log_data['cue_start'])
                rig_logs['lick'] = self.get_licks()
                rig_logs['reward'] = self.get_reward()
                
                if self.meta.opto:
                    rig_logs['opto_pulse'] = self.get_opto()
                    
                    vstim_log['opto'] = int(bool(vstim_log.get('opto',0)) or bool(len(rig_logs['opto_pulse'])))
                else:
                    vstim_log['opto'] = 0
                    
                trial_log_data = {**trial_log_data, **vstim_log, **rig_logs}
                self.trial_data = trial_log_data
                
        return self.trial_data