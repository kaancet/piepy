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
        
        if vstim.is_empty():
            early_flag = -1
        for col in vstim.columns:
            if col in ignore:
                continue
            
            # if a column has all the same values, take the first entry of the column as the value
            # sf, tf, contrast, stim_side, correct, opto, opto_pattern should run through here
    
            # failsafe for animal not moving the wheel
            # TODO: maybe not do this to save memory and time, and keep non-moved trials' values as a single value?
            if len(vstim.select(col).unique()) >= 1:
                vstim_dict[col] = vstim[0,col]
            # if different values exist in the column, take it as a list
            # stim_pos runs through here 
            else:
                vstim_dict[col] = None
                
        if early_flag==-1:
            vstim_dict['contrast'] = np.nan
            vstim_dict['spatial_freq'] = np.nan
            vstim_dict['temporal_freq'] = np.nan
            vstim_dict['stim_side'] = np.nan
            vstim_dict['opto_pattern'] = np.nan
        else:
            vstim_dict['contrast'] = 100*vstim_dict['contrast_r'] if vstim_dict['correct'] else 100*vstim_dict['contrast_l']
            vstim_dict['spatial_freq'] = vstim_dict['sf_r'] if vstim_dict['correct'] else vstim_dict['sf_l']
            vstim_dict['temporal_freq'] = vstim_dict['tf_r'] if vstim_dict['correct'] else vstim_dict['tf_l']
            # vstim_dict['stim_side'] = vstim_dict['stim_pos'][0]
            vstim_dict['stim_side'] = vstim_dict['posx_r'] if vstim_dict['correct'] else vstim_dict['posx_l']

            # training failsafe
            if 'opto_pattern' not in vstim_dict.keys():
                vstim_dict['opto_pattern'] = -1
            
            if vstim_dict['contrast'] == 0:
                vstim_dict['stim_side'] = 0 # no meaningful side when 0 contrast
        return vstim_dict
    
    def get_wheel_pos(self,time_anchor:float) -> list:
        """ Extracts the wheel trajectories and resets the positions according to time_anchor"""
        wheel_deg_per_tick = (self.meta.wheelGain * WHEEL_CIRCUM) / WHEEL_TICKS_PER_REV
        
        wheel_data = self.data['position']
        # wheel_arr = np.array(wheel_data[['duinotime', 'value']])
        wheel_arr = wheel_data.select(['duinotime','value']).to_numpy()
        
        # resetting the wheel position so the 0 point is aligned with trialstart and converting encoder ticks into degrees
        # also resetting the time frame into the trial itself rather than the whole session
        if len(wheel_arr):
            reset_idx = find_nearest(wheel_arr[:,0],time_anchor)[0]
            wheel_arr[:,1] = np.apply_along_axis(reset_wheel_pos,0,wheel_arr[:,1],reset_idx) * wheel_deg_per_tick
            wheel_pos = wheel_arr[:,1].tolist()
            wheel_arr[:,0] = np.apply_along_axis(lambda x: x-time_anchor,0,wheel_arr[:,0])
            wheel_time = wheel_arr[:,0].tolist()
        else:
            wheel_time = None
            wheel_pos = None
        return wheel_time,wheel_pos
    
    def trial_data_from_logs(self) -> dict:
        """ Iterates over each state change in a DataFrame slice that belongs to one trial(and corrections for pyvstim)
            Returns a list of dictionaries that have data parsed from stimlog and riglog
        """
        #trial_no
        trial_log_data = {'trial_no':int(self.trial_no)}
        
        # trial_start
        # temp_trial_start = self.data['state'].filter(pl.col('transition') == 'trialstart').select(self.column_keys['elapsed']).alias('trial_start')
        # trial_pl_data = trial_pl_data.join(temp_trial_start,how='left')
        trial_log_data['trial_start'] = self.data['state'].filter(pl.col('transition') == 'trialstart')[0,self.column_keys['elapsed']]
        
        # cue_start
        # temp_cue = self.data['state'].filter(pl.col('transition') == 'cuestart').select([self.column_keys['elapsed'],self.column_keys['elapsed']]).alias(['cue_start','openstart_absolute'])
        # trial_pl_data = trial_pl_data.join(temp_cue,how='left')
        try:
            trial_log_data['cue_start'] = self.data['state'].filter(pl.col('transition') == 'cuestart')[0,self.column_keys['elapsed']]
            trial_log_data['open_start_absolute'] = trial_log_data['cue_start']
        except:
            return {}
        # blank_duration
        if 'blankDuration' in self.column_keys.keys(): 
            # temp_blank = self.data['state'].filter(pl.col('transition') == 'cuestart').select(self.column_keys['blankDuration']).alias('blank_time')
            temp_blank = self.data['state'].filter(pl.col('transition') == 'cuestart')[0,self.column_keys['blankDuration']]
        else:
            #old logging for some sessions
            # temp_blank = self.data['state'].filter(pl.col('transition') == 'cuestart').select(self.column_keys['trialType']).alias('blank_time')
            temp_blank = self.data['state'].filter(pl.col('transition') == 'cuestart')[0,self.column_keys['trialType']]
        # trial_pl_data = trial_pl_data.join(temp_blank,how='left')
        trial_log_data['blank_time'] = temp_blank
        
        # catch
        if len(self.data['state'].filter(pl.col('transition') == 'catch')):
            is_catch = 1
        else:
            is_catch = 0
        # trial_pl_data = trial_pl_data.join(pl.DataFrame({"isCatch":is_catch}),how='left')
        trial_log_data['isCatch'] = is_catch
        
        #correct
        correct = self.data['state'].filter(pl.col('transition') == 'correct')
        if len(correct):
            # trial_pl_data = trial_pl_data.join(pl.DataFrame({"answer" : 1,
            #                                                  "response_latency" : correct[0,self.column_keys['stateElapsed']]}),how='left')
            trial_log_data['answer'] = 1
            trial_log_data['response_latency'] = correct[0,self.column_keys['stateElapsed']]
            
        #incorrect(noanswer)
        incorrect = self.data['state'].filter(pl.col('transition') == 'incorrect')
        if len(incorrect):
            if incorrect[0,self.column_keys['stateElapsed']] < 1000:
                # trial_pl_data = trial_pl_data.join(pl.DataFrame({"answer" : -1,
                #                                                  "response_latency": incorrect[0,self.column_keys['stateElapsed']] + trial_pl_data[0,'blank_time']}),how='left')
                trial_log_data['answer'] = -1
                trial_log_data['response_latency'] = incorrect[0,self.column_keys['stateElapsed']] + trial_log_data['blank_time']
            else:
                # trial_pl_data = trial_pl_data.join(pl.DataFrame({"answer" : 0,
                #                                                  "response_latency": incorrect[0,self.column_keys['stateElapsed']]}),how='left')
                trial_log_data['answer'] = 0
                trial_log_data['response_latency'] = incorrect[0,self.column_keys['stateElapsed']]
                
        # early
        early = self.data['state'].filter(pl.col('transition') == 'earlyanswer')
        if len(early):
            # trial_pl_data = trial_pl_data.join(pl.DataFrame({"answer" : -1,
            #                                                  "response_latency" : early[0,self.column_keys['stateElapsed']]}),how='left')
            trial_log_data['answer'] = -1
            trial_log_data['response_latency'] = early[0,self.column_keys['stateElapsed']]
        
        if 'answer' not in trial_log_data.keys():
            # this happens when training with 0 contrast, -1 means there was no answer
            trial_log_data['answer'] = -1
            trial_log_data['response_latency'] = -1
        
        # stim_start
        if trial_log_data['answer']!=-1:
            if 'screen' in self.data.keys():
                if not self.data['screen'].is_empty():
                    # trial_pl_data = trial_pl_data.join(pl.DataFrame({"stim_start_rig":self.data['screen']['duinotime'][0]}),how='left')
                    trial_log_data['stim_start_rig'] = self.data['screen'][0,'duinotime']
                else:
                    trial_log_data['stim_start_rig'] = None
            else:
                trial_log_data['stim_start_rig'] = None
            # temp_stim_start = self.data['state'].filter(pl.col('transition') == 'stimstart').select(self.column_keys['elapsed']).alias('stim_start')
            # trial_pl_data = trial_pl_data.join(temp_stim_start,how='left')
            trial_log_data['stim_start'] = self.data['state'].filter(pl.col('transition') == 'stimstart')[0,self.column_keys['elapsed']]
        else:
            trial_log_data['stim_start'] = None
            trial_log_data['stim_start_rig'] = None
        
        # stim dissappear
        if trial_log_data['stim_start'] is not None:
            # temp_stim_end = self.data['state'].filter((pl.col('transition') == 'stimendcorrect') | (pl.col('transition') == 'stimendincorrect')).select(self.column_keys['elapsed']).alias('stim_end')
            # trial_pl_data = trial_pl_data.join(temp_stim_end,how='left')
            try:
                trial_log_data['stim_end'] = self.data['state'].filter((pl.col('transition') == 'stimendcorrect') | (pl.col('transition') == 'stimendincorrect'))[0,self.column_keys['elapsed']]
            except:
                # this means that the trial was cut short, should only happen in last trial
                return {}
            if 'screen' in self.data.keys():
                if len(self.data['screen']) == 2:
                    # this should be the way for stim appear and dissappear
                    stim_end_rig = self.data['screen'][1,'duinotime']
                    
                elif len(self.data['screen']) == 1:
                    #???
                    stim_end_rig = self.data['screen'][0,'duinotime']
                # trial_pl_data = trial_pl_data.join(pl.DataFrame({"stim_end_rig" : stim_end_rig}),how='left')
                elif len(self.data['screen']) == 0:
                    stim_end_rig = None
                trial_log_data['stim_end_rig'] = stim_end_rig
            else:
                trial_log_data['stim_end_rig'] = None
        else:
            trial_log_data['stim_end'] = None
            trial_log_data['stim_end_rig'] = None

        # correction or trial end
        # temp_trial_end = self.data['state'].filter((pl.col('transition') == 'trialend') | (pl.col('transition') == 'correction')).select(self.column_keys['elapsed']).alias('trial_end')
        # trial_pl_data = trial_pl_data.join(temp_trial_end,how='left')
        try:
            trial_log_data['trial_end'] = self.data['state'].filter((pl.col('transition') == 'trialend') | (pl.col('transition') == 'correction'))[0,self.column_keys['elapsed']]
        except:
            # this means that the trial was cut short, should only happen in last trial
            return {}
            
        #vstim
        
        vstim_log = self.get_vstim_props(trial_log_data['answer'])
        
        rig_log_data = {}
        #wheel
        # trial_pl_data = trial_pl_data.join(pl.DataFrame({"wheel" : self.get_wheel_pos(trial_pl_data[0,'cue_start'])}),how='left')
        rig_log_data['wheel_time'],rig_log_data['wheel_pos'] = self.get_wheel_pos(trial_log_data['cue_start'])
        
        #lick
        # trial_pl_data = trial_pl_data.join(pl.DataFrame({"lick" : self.get_licks()}),how='left')
        rig_log_data['lick'] = self.get_licks()
        
        #reward
        # trial_pl_data = trial_pl_data.join(pl.DataFrame({"reward" : self.get_reward()}),how='left')
        rig_log_data['reward'] = self.get_reward()
        
        #opto
        if self.meta.opto:
            opto_pulse = self.get_opto()
            # trial_pl_data = trial_pl_data.join(pl.DataFrame({"opto_pulse" : opto_pulse}),how='left')
            # rig_log_data['opto_pulse'] = opto_pulse
            is_opto = int(bool(vstim_log.get('opto',0)) or bool(len(opto_pulse)))
        else:
            is_opto = 0
        # trial_pl_data = trial_pl_data.join(pl.DataFrame({"opto" : is_opto}),how='left')
        trial_log_data['opto'] = is_opto
        
        trial_log_data = {**trial_log_data, **vstim_log, **rig_log_data}
        self.trial_data = trial_log_data

        return self.trial_data
