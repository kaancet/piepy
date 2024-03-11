import numpy as np
from ..utils import *
from ..core.trial import *
from ..wheelTrace import WheelTrace


class WheelDetectionTrial(Trial):
    def __init__(self,trial_no:int,meta,logger) -> None:
        super().__init__(trial_no,meta,logger)
    
    def get_vstim_properties(self) -> dict:
        """ 
        Extracts the necessary properties from vstim data
        """
        ignore = ['iTrial','photo','code','presentTime']
        
        vstim = self.data['vstim']
        vstim = vstim.drop_nulls(subset=['prob'])
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
            
            vstim_dict['opto_pattern'] = temp_dict.get('opto_pattern',None)
            vstim_dict['prob'] = temp_dict['prob']

            # training failsafe
            if 'opto_pattern' not in temp_dict.keys():
                vstim_dict['opto_pattern'] = -1
                self.logger.warning(f'No opto_pattern found in vstim log, setting to -1(nonopto)')

            if vstim_dict['contrast'] == 0:
                vstim_dict['stim_pos'] = 0 # no meaningful side when 0 contrast
                
        self._attrs_from_dict(vstim_dict)      
        return vstim_dict
    
    def get_wheel_traces(self,**kwargs) -> dict:
        """ Extracts the wheel trajectories and resets the positions according to time_anchor"""
        thresh_in_ticks = kwargs.get('tick_thresh',self.meta.wheelThresholdStim)
        speed_thresh = thresh_in_ticks/kwargs.get('time_thresh',17) #17 is the avg ms for a loop
        interp_freq = kwargs.get('freq',5)
                
        wheel_data = self.data['position']
        wheel_arr = wheel_data.select(['duinotime','value']).to_numpy()

        #instantiate a wheel Trajectory object
        traj = WheelTrace(wheel_arr[:,0],wheel_arr[:,1],interp_freq=interp_freq) 
        
        wheel_dict = traj.make_dict_to_log()
       
        if len(wheel_arr)<=2:
            self.logger.warning(f'Less than 2 sample points for wheel data')
            return wheel_dict
        
        if self.t_stimstart_rig is None:
            if self.state_outcome!=-1:
                self.logger.warning(f'No stimulus start based on photodiode in a stimulus trial, using stateMachine time!')
                time_anchor = self.t_stimstart
                #  the window of analysis for trajectories
                window_end = self.t_stimend
            else:
                # for early trials use the time after quiescence period
                time_anchor = self.t_trialinit
                window_end = self.t_trialinit + 3000
        else:
            time_anchor = self.t_stimstart_rig
            window_end = self.t_stimend_rig
        
        time_window = [self.t_trialinit, window_end]
        
        #initialize the trace
        traj.init_trace(time_anchor=time_anchor)
        # get the movements from interpolated positions
        traj.get_movements(pos_thresh=kwargs.get('pos_thresh',0.02),
                           t_thresh=kwargs.get('t_thresh',0.5))
        
        if len(traj.onsets) == 0 and self.state_outcome == 1:
            self.logger.error('No movement onset detected in a correct trial!')
            return wheel_dict
        
        # there are onsets after 50(stim appearance + 50ms)
        if len(np.where(traj.onsets>50)[0]) == 0:
            if self.state_outcome == 1:
                self.logger.error(f'No detected wheel movement in correct trial after stim!')
        
        # these has to be run before calculating reaction times to constrain the region of traces we are interested in
        interval_mask = traj.make_interval_mask(time_window=time_window)
        traj.select_trace_interval(mask=interval_mask)
        
        # get all the reaction times and outcomes here:
        traj.get_speed_reactions(speed_threshold=speed_thresh)
        traj.get_tick_reactions(tick_threshold=thresh_in_ticks)

        # Logging discrapencies 
        # stateMachine vs delta tick
        if traj.pos_outcome is not None:
            if self.state_outcome != traj.pos_outcome:
                self.logger.critical(f"stateMachine outcome and delta tick outcome does not match!!! {self.state_outcome}=/={traj.pos_outcome}!")
        else:
            self.logger.error(f"Can't calculate wheel reaction time in correct trial!!")
        
        # stateMachine vs wheel speed   
        if traj.speed_outcome is not None:
            if self.state_outcome != traj.speed_outcome:
                self.logger.critical(f"stateMachine outcome and wheel speed outcome does not match!!! {self.state_outcome}=/={traj.speed_outcome}!")
        else:
            self.logger.error(f"Can't calculate speed reaction time in correct trial!!")
        
        # delta tick vs wheel speed
        if traj.pos_outcome != traj.speed_outcome:
            self.logger.critical(f"delta tick outcome and wheel speed outcome does not match!!! {traj.pos_outcome}=/={traj.speed_outcome}!")
        
        self.trace = traj
        # fill the dict 
        wheel_dict = traj.make_dict_to_log()
        self._attrs_from_dict(wheel_dict)
        return wheel_dict
    
    def get_state_changes(self) -> dict:
        """ 
        Looks at state changes in a given data slice and set class attributes according to them
        every key starting with t_ is an absolute time starting from experiment start
        """
        empty_log_data = {'t_trialstart' : self.t_trialstart, # this is an absolute value
                          'vstim_offset' : self.vstim_offset,
                          'state_offset' : self.state_offset,
                          't_stimstart' : None,
                          't_stimend': None,
                          'state_outcome' : None}
        
        state_log_data = {**empty_log_data}
        # in the beginning check if state data is complete
        if 'trialend' not in self.data['state']['transition'].to_list():
            self._attrs_from_dict(empty_log_data)
            return empty_log_data
        
        # iscatch?
        if len(self.data['state'].filter(pl.col('transition') == 'catch')):
            state_log_data['isCatch'] = True
        else:
            state_log_data['isCatch'] = False
        
        # trial init and blank duration
        cue  = self.data['state'].filter(pl.col('transition') == 'cuestart')
        if len(cue):
            state_log_data['t_trialinit'] = cue[0,'corrected_elapsed']
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
        else:
            state_log_data['t_stimstart'] = self.data['state'].filter(pl.col('transition') == 'stimstart')[0,'corrected_elapsed'] 
            
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
                    state_log_data['response_latency'] = state_log_data['t_trialinit'] + temp_resp
                elif 150 < temp_resp <1000:
                    # This should not happen
                    # DISCARD TRIAL
                    self.logger.error(f'Trial categorized as MISS with {temp_resp}s response time!! DISCARDING.....')
                    self._attrs_from_dict(empty_log_data)
                    return empty_log_data
                else:
                    # actual miss >= 1000
                    state_log_data['state_outcome'] = 0
                    state_log_data['response_latency'] = temp_resp
        
        if state_log_data['state_outcome'] is None:
            # this happens when training with 0 contrast, -1 means there was no answer
            state_log_data['state_outcome'] = -1
            state_log_data['response_latency'] = -1
        
        # stimulus end
        if state_log_data['t_stimstart'] is not None:
            try:
                if state_log_data['state_outcome'] != 1:
                    state_log_data['t_stimend'] = self.data['state'].filter((pl.col('transition') == 'stimendincorrect'))[0,'corrected_elapsed']
                else:
                    state_log_data['t_stimend'] = self.data['state'].filter((pl.col('transition') == 'stimendcorrect'))[0,'corrected_elapsed']
            except:
                # this means that the trial was cut short, should only happen in last trial
                self.logger.warning('Stimulus appeared but not disappeared, is this expected??')

        state_log_data['t_trialend'] = self.t_trialend
        self._attrs_from_dict(state_log_data)
        return state_log_data
    
    def trial_data_from_logs(self,**wheel_kwargs) -> dict:
        """ 
        :return: A dictionary to be appended in the session dataframe
        """
        trial_log_data = {'trial_no':self.trial_no}
        
        # state machine
        state_dict = self.get_state_changes()
        if state_dict['state_outcome'] is None:
            return {**trial_log_data,**state_dict}
        # screen
        screen_dict = self.get_screen_events()
        # vstim
        vstim_dict = self.get_vstim_properties()
        # wheel
        wheel_dict = self.get_wheel_traces(**wheel_kwargs)
        # lick
        lick_dict = self.get_licks()
        # reward
        reward_dict = self.get_reward()
        # opto
        opto_dict = self.get_opto()
        # camera frames
        frames_dict = {}
        for c in ['eyecam','facecam','onepcam']:
            tmp = self.get_frames(get_from=c)
            frames_dict = {**frames_dict, **tmp}
        
        trial_log_data = {**trial_log_data,
                          **state_dict,
                          **screen_dict,
                          **vstim_dict, 
                          **wheel_dict, 
                          **lick_dict,
                          **reward_dict,
                          **opto_dict,
                          **frames_dict}
        
        return trial_log_data
