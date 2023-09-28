import numpy as np
from ..utils import *
from ..core.trial import *
from ..wheelUtils import *


class WheelDetectionTrial(Trial):
    def __init__(self,trial_no:int,meta,logger:Logger) -> None:
        super().__init__(trial_no,meta,logger)
    
    def get_vstim_props(self) -> dict:
        """ 
        Extracts the necessary properties from vstim data
        """
        ignore = ['iTrial','photo','code','presentTime']
        
        vstim = self.data['vstim']
        vstim = vstim.drop_nulls()
        # this is an offline fix for a vstim logging issue where time increment messes up vstim logging
        vstim = vstim[:-1]
        
        early_flag = self.state_outcome
        if self.state_outcome!=-1 and vstim.is_empty():
            self.logger.warning(f'Empty vstim data for non-early trial!!')
            early_flag = -1
        
        temp_dict = {}
        for col in vstim.columns:
            if col in ignore:
                continue
            if len(vstim.select(col).unique()) == 1:
                # if a column has all the same values, take the first entry of the column as the value
                # sf, tf, contrast, stim_side, correct, opto, opto_pattern should run through here
                temp_dict[col] = vstim[0,col]
            elif len(vstim.select(col).unique()) > 1 and col not in ['reward']:
                # if different values exist in the column, take it as a list, this should not happen in detection task
                self.logger.error(f"{col} has multiple unique entries ({len(vstim.select(col).unique())}). This shouldn't be the case")
                temp_dict[col] = vstim[col].to_list()
            else:
                temp_dict[col] = None
                        
        vstim_dict = {'contrast' : None,
                      'spatial_freq': None,
                      'temporal_freq' : None,
                      'stim_pos' : None,
                      'opto_pattern' : None,
                      'prob': None}
        
        if early_flag!=-1:
            vstim_dict['contrast'] = 100*temp_dict['contrast_r'] if temp_dict['correct'] else 100*temp_dict['contrast_l']
            vstim_dict['spatial_freq'] = round(temp_dict['sf_r'],2) if temp_dict['correct'] else round(temp_dict['sf_l'],2)
            vstim_dict['temporal_freq'] = round(temp_dict['tf_r'],2) if temp_dict['correct'] else round(temp_dict['tf_l'],2)
            vstim_dict['stim_pos'] = temp_dict['posx_r'] if temp_dict['correct'] else temp_dict['posx_l']
            vstim_dict['opto_pattern'] = temp_dict['opto_pattern']
            vstim_dict['prob'] = temp_dict['prob']

            # training failsafe
            if 'opto_pattern' not in temp_dict.keys():
                vstim_dict['opto_pattern'] = -1
                self.logger.warning(f'No opto_pattern found in vstim log, setting to -1(nonopto)')

            if vstim_dict['contrast'] == 0:
                vstim_dict['stim_pos'] = 0 # no meaningful side when 0 contrast
                
        self._attrs_from_dict(vstim_dict)      
        return vstim_dict
    
    def get_wheel_pos(self,**kwargs) -> list:
        """ Extracts the wheel trajectories and resets the positions according to time_anchor"""
        thresh_in_ticks = 10
        wheel_data = self.data['position']
        wheel_arr = wheel_data.select(['duinotime','value']).to_numpy()

        wheel_dict = {'wheel_time' : None,
                      'wheel_pos' : None,
                      'wheel_reaction_time' : None,
                      'wheel_bias' : None,
                      'wheel_outcome' : self.state_outcome,
                      'onsets' : None,
                      'offsets': None}
       
        if len(wheel_arr)<=2:
            self.logger.warning(f'Less than 2 sample points for wheel data')
            return wheel_dict

        t_tick = wheel_arr[:,0]
        pos_tick = wheel_arr[:,1]
        
        if self.t_stimstart_rig is None:
            if self.state_outcome!=-1:
                self.logger.warning(f'No stimulus start based on photodiode in a stimulus trial, using stateMachine time!')
                time_anchor = self.t_stimstart
            else:
                #early trial, use first sample point in trial
                time_anchor = t_tick[0]
        else:
            time_anchor = self.t_stimstart_rig
        
        # zero the time and position
        wheel_time = t_tick - time_anchor
        t_idx, t_val= find_nearest(wheel_time,0)
        if t_val != 0:
            idx_region = [t_idx-1, t_idx] if t_val>0 else [t_idx,t_idx+1]
            t_region = [wheel_time[i] for i in idx_region]
            p_region = [pos_tick[i] for i in idx_region]
            pos_at0 = int(np.interp(0,t_region,p_region))
        else:
            pos_at0 = pos_tick[t_idx]
        wheel_pos = pos_tick - pos_at0
   
        # interpolate the sample points
        pos,t = interpolate_position(wheel_time,wheel_pos,freq=kwargs.get('freq',20))
        
        wheel_dict['wheel_time'] = wheel_time.tolist()
        # convert pos to degs
        wheel_dict['wheel_pos'] = cm_to_deg(samples_to_cm(wheel_pos)).tolist()
        
        # look at a certain time window
        time_window = kwargs.get('time_window',[-100,1500])
        mask = np.where((time_window[0]<t) & (t<time_window[1]))
        t = t[mask]
        pos = pos[mask]
        
        
        onsets,offsets,_,_,_,_ = movements(t,pos, 
                                           freq=kwargs.get('freq',20),
                                           pos_thresh=kwargs.get('pos_thresh',0.03),
                                           t_thresh=kwargs.get('t_thresh',0.5))
        
        wheel_dict['wheel_onsets'] = onsets
        wheel_dict['wheel_offsets'] = offsets

        if len(onsets) == 0 and self.state_outcome == 1:
            self.logger.error('No movement onset detected in a correct trial!')
            return wheel_dict
        
        try:
            # there are onsets after 0(stim appearance)
            _idx = np.where(onsets>0)[0]
            onsets = onsets[_idx]
            offsets = offsets[_idx]
            # onset_samps = onset_samps[_idx]
            # offset_samps = offset_samps[_idx]
        except:
            _idx = None
            if self.state_outcome == 1:
                self.logger.warning(f'No detected wheel movement in correct trial after stimulus presentation!')

        if _idx is not None:
            for i,o in enumerate(onsets):
                t_onset_idx,_ = find_nearest(t,o)
                t_offset_idx,_ = find_nearest(t,offsets[i])                        
                move_extent = pos[t_offset_idx] - pos[t_onset_idx]
                if np.abs(move_extent) >= thresh_in_ticks:
                    after_onset_abs_pos = np.abs(pos[t_onset_idx:t_offset_idx+1]) #abs to make thresh comparison easy
                    after_onset_time = t[t_onset_idx:t_offset_idx+1]
                    wheel_pass_idx,_ = find_nearest(after_onset_abs_pos,after_onset_abs_pos[0]+thresh_in_ticks)

                    wheel_dict['wheel_reaction_time'] = after_onset_time[wheel_pass_idx]
                    wheel_dict['wheel_bias'] = np.sign(move_extent)
                    break
            
            if self.state_outcome == 1 and wheel_dict['wheel_reaction_time'] is None:
                self.logger.error(f"Can't calculate wheel reaction time in correct trial!! Using stateMachine time")
                wheel_dict['wheel_reaction_time'] = self.response_latency
                
        if wheel_dict['wheel_reaction_time'] is not None and wheel_dict['wheel_reaction_time'] < 1000:
            if self.state_outcome == 0:
                self.logger.critical(f"The trial was classified as a MISS, but wheel reaction time is {wheel_dict['wheel_reaction_time']}!")
                wheel_dict['wheel_outcome'] = 1

        self._attrs_from_dict(wheel_dict)
        return wheel_dict

    def get_state_changes(self) -> dict:
        """ 
        Looks at state changes in a given data slice and set class attributes according to them
        """
        state_log_data = {'t_trialstart' : self.t_trialstart,
                          't_stimstart' : None,
                          't_stimstart_absolute' : None,
                          't_stimend': None,
                          'state_outcome' : None}
        
        # iscatch?
        if len(self.data['state'].filter(pl.col('transition') == 'catch')):
            state_log_data['isCatch'] = True
        else:
            state_log_data['isCatch'] = False
        
        # trial start and blank duration
        cue  = self.data['state'].filter(pl.col('transition') == 'cuestart')
        if len(cue):
            state_log_data['t_quiescence_dur'] = cue[0,'stateElapsed']
            try:
                temp_blank = cue[0,'blankDuration']
            except:
                temp_blank = cue[0,'trialType'] # old logging for some sessions
            state_log_data['t_blank_dur'] = temp_blank 
        else:
            self.logger.warning('No cuestart after trialstart')
        
        # early
        early = self.data['state'].filter(pl.col('transition') == 'early')
        if len(early):
            state_log_data['state_outcome'] = -1
            state_log_data['response_latency'] = early[0,'stateElapsed']
        
        # stimulus start
        if state_log_data['state_outcome']!=-1:
            state_log_data['t_stimstart'] = self.data['state'].filter(pl.col('transition') == 'stimstart')[0,'stateElapsed'] 
            state_log_data['t_stimstart_absolute'] = self.data['state'].filter(pl.col('transition') == 'stimstart')[0,'elapsed']
            
        #hit
        hit = self.data['state'].filter(pl.col('transition') == 'hit')
        if len(hit):
            state_log_data['state_outcome'] = 1
            state_log_data['response_latency'] = hit[0,'stateElapsed']
            
        #miss
        miss = self.data['state'].filter(pl.col('transition') == 'miss')
        if len(miss):
            temp_resp = miss[0,'stateElapsed']
            if temp_resp <= 150:
                # this is actually early
                state_log_data['state_outcome'] = -1
                state_log_data['response_latency'] = temp_resp + state_log_data['t_blank_dur']
            elif 200 < temp_resp <1000:
                # This should not happen
                self.logger.critical(f'Trial categorized as MISS with {temp_resp}s response time!!')
            else:
                state_log_data['state_outcome'] = 0
                state_log_data['response_latency'] = miss[0,'stateElapsed']
        
        if state_log_data['state_outcome'] is None:
            # this happens when training with 0 contrast, -1 means there was no answer
            state_log_data['state_outcome'] = -1
            state_log_data['response_latency'] = -1
        
        # stimulus end
        if state_log_data['t_stimstart'] is not None:
            try:
                if state_log_data['state_outcome'] != 1:
                    temp_stim_end = self.data['state'].filter((pl.col('transition') == 'stimendincorrect'))[0,'elapsed']
                else:
                    temp_stim_end = self.data['state'].filter((pl.col('transition') == 'stimendcorrect'))[0,'elapsed']
                    
                state_log_data['t_stimend'] = temp_stim_end - state_log_data['t_stimstart_absolute']
            except:
                # this means that the trial was cut short, should only happen in last trial
                self.logger.warning('Stimulus appeared but not disappeared, is this expected??')

        state_log_data['t_trialend'] = self.t_trialend
        self._attrs_from_dict(state_log_data)
        return state_log_data
    
    def get_screen_events(self) -> dict:
        """
        Gets the screen pulses from rig data
        """
        screen_data = self.data['screen']
        screen_arr = screen_data.select(['duinotime','value']).to_numpy()
        
        screen_dict = {'t_stimstart_rig' : None,
                       't_stimend_rig' : None}
        
        #TODO: 3 SCREEN EVENTS WITH OPTO BEFORE STIMULUS PRESENTATION
        if self.state_outcome != -1:
            if len(screen_arr) == 1:
                self.logger.error('Only one screen event! Stimulus appeared but not dissapeared?')
                screen_dict['t_stimstart_rig'] = screen_arr[0,0]
            elif len(screen_arr) > 2:
                self.logger.error('More than 2 screen events per trial, this is not possible')
            elif len(screen_arr) == 0:
                self.logger.critical('NO SCREEN EVENT FOR STIMULUS TRIAL!')
            else:
                screen_dict['t_stimstart_rig'] = screen_arr[0,0]
                screen_dict['t_stimend_rig'] = screen_arr[1,0]
        
        self._attrs_from_dict(screen_dict)
        return screen_dict
    
    def trial_data_from_logs(self) -> dict:
        """ 
        :return: A dictionary to be appended in the session dataframe
        """
        # state machine
        state_dict = self.get_state_changes()
        # screen
        screen_dict = self.get_screen_events()
        # vstim
        vstim_dict = self.get_vstim_props()
        # wheel
        wheel_dict = self.get_wheel_pos()
        # lick
        lick_dict = self.get_licks()
        # reward
        reward_dict = self.get_reward()
        # opto
        opto_dict = self.get_opto()
        
        trial_log_data = {'trial_no':self.trial_no,
                          **state_dict,
                          **screen_dict,
                          **vstim_dict, 
                          **wheel_dict, 
                          **lick_dict,
                          **reward_dict,
                          **opto_dict}
        
        return trial_log_data
