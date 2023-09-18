import numpy as np
from ..utils import *
from ..core.trial import *
from ..wheelUtils import *


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
    
    def get_wheel_pos(self,time_anchor:float=None,time_range=[-100,1000]) -> list:
        """ Extracts the wheel trajectories and resets the positions according to time_anchor"""
        
        wheel_data = self.data['position']
        # wheel_arr = np.array(wheel_data[['duinotime', 'value']])
        wheel_arr = wheel_data.select(['duinotime','value']).to_numpy()
        
        # resetting the wheel position so the 0 point is aligned with trialstart and converting encoder ticks into degrees
        # also resetting the time frame into the trial itself rather than the whole session
        thresh_in_ticks = 10
        wheel_time = None
        wheel_pos_deg = None
        wheel_reaction_time = None
        wheel_bias = None
        if len(wheel_arr)>=2:

            t_tick = wheel_arr[:,0]
            pos_tick = wheel_arr[:,1]
            
            # #threshold in degrees
            # thresh_cm = samples_to_cm(thresh_in_ticks) #~0.190
            # thresh_deg = cm_to_deg(thresh_cm) #~3.515
            
            if time_anchor is None:
                # if no time anchor provided than reset the time and position 
                # from the first recorded wheel sample in that trial
                time_anchor = t_tick[0]
                
            # make the reset position 0
            reset_pos = np.apply_along_axis(lambda x: x-pos_tick[0],0,pos_tick)
            wheel_pos = reset_pos.tolist()
            # zero the time
            reset_time = np.apply_along_axis(lambda x: x-time_anchor,0,t_tick)
            wheel_time = reset_time.tolist()
            
            # interpolate the wheels
            pos,t = interpolate_position(wheel_time,wheel_pos,freq=20)
            
            # don't look at too late 
            # mask = t < t[0] + time_range[1]
            # pos = pos[mask]
            # t = t[mask]
            
            onsets,offsets,onset_samps,offset_samps,peak_amps,peak_vel_times = movements(t,pos, freq=20, pos_thresh=0.03,t_thresh=0.5)

            if len(onsets):
                # there are onsets 
                try:
                    # there are onsets after 0(stim appearance)
                    _idx = np.where(onsets>0)[0]
                    onsets = onsets[_idx]
                    offsets = offsets[_idx]
                    onset_samps = onset_samps[_idx]
                    offset_samps = offset_samps[_idx]
                except:
                    _idx = None

                if _idx is not None:
                    for i,o in enumerate(onsets):
                        
                        tick_onset_idx,_ = find_nearest(wheel_time,o)
                        tick_offset_idx,_ = find_nearest(wheel_time,offsets[i])                        
                        tick_extent = wheel_pos[tick_offset_idx] - wheel_pos[tick_onset_idx]
                        
                        if np.abs(tick_extent) >= thresh_in_ticks:
                            after_onset_abs_pos = np.abs(wheel_pos[tick_onset_idx:tick_offset_idx+1]) #abs to make thresh comparison easy
                            after_onset_time = wheel_time[tick_onset_idx:tick_offset_idx+1]
                            try:
                                wheel_pass_idx,_ = find_nearest(after_onset_abs_pos,after_onset_abs_pos[0]+thresh_in_ticks)
                            except:
                                print('kk')
                            wheel_reaction_time = after_onset_time[wheel_pass_idx]
                            wheel_bias = np.sign(tick_extent)
                            break
            # convert pos to degs
            wheel_pos_cm = samples_to_cm(np.array(wheel_pos))
            wheel_pos_deg = cm_to_deg(wheel_pos_cm).tolist()

        return wheel_time,wheel_pos_deg,wheel_reaction_time,wheel_bias
    
    def trial_data_from_logs(self) -> dict:
        """ Iterates over each state change in a DataFrame slice that belongs to one trial(and corrections for pyvstim)
            Returns a list of dictionaries that have data parsed from stimlog and riglog
        """
        #trial_no
        trial_log_data = {'trial_no':int(self.trial_no)}
        
        # trial_start
        trial_log_data['trial_start'] = self.data['state'].filter(pl.col('transition') == 'trialstart')[0,self.column_keys['elapsed']]
        
        # cue_start
        try:
            trial_log_data['cue_start'] = self.data['state'].filter(pl.col('transition') == 'cuestart')[0,self.column_keys['elapsed']]
            trial_log_data['open_start_absolute'] = trial_log_data['cue_start']
        except:
            return {}
        # blank_duration
        if 'blankDuration' in self.column_keys.keys(): 
            temp_blank = self.data['state'].filter(pl.col('transition') == 'cuestart')[0,self.column_keys['blankDuration']]
        else:
            #old logging for some sessions
            temp_blank = self.data['state'].filter(pl.col('transition') == 'cuestart')[0,self.column_keys['trialType']]
        trial_log_data['blank_time'] = temp_blank
        
        # catch
        if len(self.data['state'].filter(pl.col('transition') == 'catch')):
            is_catch = 1
        else:
            is_catch = 0
        trial_log_data['isCatch'] = is_catch
        
        #correct
        correct = self.data['state'].filter(pl.col('transition') == 'correct')
        if len(correct):
            trial_log_data['answer'] = 1
            trial_log_data['response_latency'] = correct[0,self.column_keys['stateElapsed']]
            
        #incorrect(noanswer)
        incorrect = self.data['state'].filter(pl.col('transition') == 'incorrect')
        if len(incorrect):
            if incorrect[0,self.column_keys['stateElapsed']] < 1000:
                trial_log_data['answer'] = -1
                trial_log_data['response_latency'] = incorrect[0,self.column_keys['stateElapsed']] + trial_log_data['blank_time']
            else:
                trial_log_data['answer'] = 0
                trial_log_data['response_latency'] = incorrect[0,self.column_keys['stateElapsed']]
                
        # early
        early = self.data['state'].filter(pl.col('transition') == 'earlyanswer')
        if len(early):
            trial_log_data['answer'] = -1
            trial_log_data['response_latency'] = early[0,self.column_keys['stateElapsed']]
        
        if 'answer' not in trial_log_data.keys():
            # this happens when training with 0 contrast, -1 means there was no answer
            trial_log_data['answer'] = -1
            trial_log_data['response_latency'] = -1
        
        # stim_start
        if trial_log_data['answer']!=-1:
            trial_log_data['stim_start'] = self.data['state'].filter(pl.col('transition') == 'stimstart')[0,self.column_keys['elapsed']]
            if 'screen' in self.data.keys():
                if not self.data['screen'].is_empty():
                    trial_log_data['stim_start_rig'] = self.data['screen'][0,'duinotime']
                else:
                    # this should't happen
                    trial_log_data['stim_start_rig'] = None
            else:
                # this is an approximation from loooking into the data and time diff between state log and screen log
                trial_log_data['stim_start_rig'] = trial_log_data['stim_start'] + 45
        else:
            trial_log_data['stim_start'] = None
            trial_log_data['stim_start_rig'] = None
        
        
        # stim dissappear
        if trial_log_data['stim_start'] is not None:
            try:
                if trial_log_data['answer'] != 1:
                    trial_log_data['stim_end'] = self.data['state'].filter((pl.col('transition') == 'incorrect'))[0,self.column_keys['elapsed']]
                else:
                    trial_log_data['stim_end'] = self.data['state'].filter((pl.col('transition') == 'stimendcorrect'))[0,self.column_keys['elapsed']]
            except:
                # this means that the trial was cut short, should only happen in last trial
                return {}
            
            if 'screen' in self.data.keys():
                try:
                    # this should be the way for stim appear and dissappear
                    stim_end_rig = self.data['screen'][1,'duinotime']
                except:
                    stim_end_rig = None
   
                trial_log_data['stim_end_rig'] = stim_end_rig
            else:
                # this is an approximation from loooking into the data and time diff between state log and screen log
                trial_log_data['stim_end_rig'] = trial_log_data['stim_end'] + 45
        else:
            trial_log_data['stim_end'] = None
            trial_log_data['stim_end_rig'] = None

        # correction or trial end
        try:
            trial_log_data['trial_end'] = self.data['state'].filter((pl.col('transition') == 'trialend') | (pl.col('transition') == 'correction'))[0,self.column_keys['elapsed']]
        except:
            # this means that the trial was cut short, should only happen in last trial
            return {}
            
        #vstim
        vstim_log = self.get_vstim_props(trial_log_data['answer'])
        
        rig_log_data = {}
        #wheel
        if trial_log_data['answer'] == -1:
            # if early anchor the positions to cue start
            rig_log_data['wheel_time'],rig_log_data['wheel_pos'],rig_log_data['wheel_reaction_time'],rig_log_data['wheel_bias'] = self.get_wheel_pos(trial_log_data['cue_start'])
        else:
           rig_log_data['wheel_time'],rig_log_data['wheel_pos'],rig_log_data['wheel_reaction_time'],rig_log_data['wheel_bias'] = self.get_wheel_pos(trial_log_data['stim_start_rig'])

        #lick
        rig_log_data['lick'] = self.get_licks()
        
        #reward
        rig_log_data['reward'] = self.get_reward()
        
        #opto
        if self.meta.opto:
            opto_pulse = self.get_opto()
            is_opto = int(bool(vstim_log.get('opto',0)) or bool(len(opto_pulse)))
        else:
            is_opto = 0
        trial_log_data['opto'] = is_opto
        
        trial_log_data = {**trial_log_data, **vstim_log, **rig_log_data}
        self.trial_data = trial_log_data

        return self.trial_data
