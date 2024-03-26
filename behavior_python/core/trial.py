import polars as pl
import numpy as np
from ..utils import parseConfig


class Trial:
    __slots__ = ['trial_no','data','column_keys',
                 't_trialstart','t_trialend','db_interface','total_trial_count',
                 'reward_ms_per_ul','meta','logger']
    def __init__(self, trial_no:int, meta, logger) -> None:
        
        self.trial_no = trial_no
        self.meta = meta
        self.logger = logger
        self.reward_ms_per_ul = 0
        self.logger.set_msg_prefix(f'trial-[{self.trial_no}]')
        
    def __repr__(self) -> str:
        rep = f"""Trial No :{self.trial_no}
        {self.data['state']}"""
        return rep
    
    def _attrs_from_dict(self,log_data:dict) -> None:
        """
        Creates class attributes from a dictionary
        :param log_data: any dictionary  to set class attributes
        """
        for k,v in log_data.items():
            setattr(self,k,v)
    
    def get_state_slice_and_correct_time_frame(self, rawdata:dict,use_state:bool=False) -> None:
        """ """
        states = rawdata['statemachine']
        vstim = rawdata['vstim']
        vstim = vstim.with_columns((pl.col('presentTime')*1000).alias('presentTime')) #ms
        screen = rawdata['screen']
    
        state_slice = states.filter(pl.col('trialNo')==self.trial_no)
              
        if use_state:
            _start = state_slice[0,'elapsed']
            _end = state_slice[-1,'elapsed']
        
            screen_slice = screen.filter((pl.col('duinotime') >= _start) &
                                        ((pl.col('duinotime') <= _end)))

        else:
            # there should be exactly 2x trial count screen events
            # idx = 1 if screen[0,'value'] == 0 else 0 # sometimes screen events have a 0 value entry at the beginning
            screen_slice = screen.filter(pl.col('value') == self.trial_no)
            _start = screen_slice[0,'duinotime'] - self.meta.blankDuration*1000 # ms
            _end = screen_slice[1,'duinotime']
            
        vstim_slice = vstim.filter((pl.col('presentTime')>=_start) & 
                                   (pl.col('presentTime')<=_end))  
        if len(screen_slice):
            #there is an actual screen event
            rig_onset = screen_slice[0,'duinotime']
            
            # vstim time
            vstim_onset = vstim_slice.filter(pl.col('photo')==True)[0,'presentTime']
            
            # state time
            state_onset = state_slice.filter(pl.col('transition')=='stimstart')[0,'elapsed']
            
            vstim_offset = vstim_onset - rig_onset
            state_offset = state_onset - rig_onset
        else:
            vstim_offset = 0
            state_offset = 0
        
        vstim_slice = vstim_slice.with_columns((pl.col('presentTime')-vstim_offset).alias('corrected_presentTime'))
        state_slice = state_slice.with_columns((pl.col('elapsed')-state_offset).alias('corrected_elapsed'))
        
        self.vstim_offset = vstim_offset
        self.state_offset = state_offset
        
        self.data = {'state' : state_slice}
        self.data['vstim'] = vstim_slice
        self.data['screen'] = screen_slice
        
        self.t_trialstart = state_slice['corrected_elapsed'][0]
        self.t_trialend = state_slice['corrected_elapsed'][-1]
    
    def get_data_slices(self,rawdata:dict,use_state:bool=False) -> None:
        """
        Extracts the relevant portion from each data 
        
        :param rawdata: Rawdata dictionary
        :return: DataFrame slice of corresponding trial no
        """
        rig_cols = ['screen','imaging','position','lick',
                    'button','reward','lap','facecam','eyecam',
                    'onepcam','act0','act1','opto']
        
        self.get_state_slice_and_correct_time_frame(rawdata,use_state)

        # rig and stimlog
        for k,v in rawdata.items():
            if k in ['statemachine','vstim','screen']:
                # skip because already looked into it
                continue
            if not v.is_empty():
                t_start = self.t_trialstart
                t_end = self.t_trialend
                
                if k in rig_cols:
                     temp_v = v.filter((pl.col('duinotime') >= self.t_trialstart) & 
                                       (pl.col('duinotime') <= self.t_trialend))
                    
                if k == 'vstim':
                    # since we don't need any time info from vstim, just get trial no
                    temp_v = v.filter(pl.col('iTrial')==self.trial_no)
                    
                    # to check timing is good in vstim log
                    time_col = 'presentTime'
                    t_start = self.t_trialstart / 1000
                    t_end = self.t_trialend / 1000
                    fake_v = v.filter((pl.col(time_col) >= t_start) & (pl.col(time_col) <= t_end))
                    if len(fake_v['iTrial'].unique())>1:
                        msg = str(fake_v['iTrial'].unique().to_list())
                        self.logger.warning(f"The timing of vstim is funky, has multiple trial no's {msg}")
            
                self.data[k] = temp_v
    
    def get_state_changes(self) -> dict:
        """ This depends on the experimetn type at hand so need to be overwritten"""
        pass
    
    def get_licks(self) -> dict:
        """ Extracts the lick data from slice"""
        lick_data = self.data.get('lick', None)
        if lick_data is not None:
            if len(lick_data):
            # lick_arr = np.array(lick_data[['duinotime', 'value']])
                lick_arr = lick_data.select(['duinotime','value']).to_series().to_list()
            else:
                if self.state_outcome == 1:
                    self.logger.error(f'Empty lick data in correct trial')
                lick_arr = None
        else:
            self.logger.warning(f'No lick data in trial')
            lick_arr = None
        
        lick_dict = {'lick':lick_arr}
        self._attrs_from_dict(lick_dict)
        return lick_dict
    
    def get_reward(self) -> dict:
        """ Extracts the reward clicks from slice"""
        
        reward_data = self.data.get('reward',None)
        if reward_data is not None:
            # no reward data, shouldn't happen a lot, usually in shitty sessions
            reward_arr = reward_data.select(['duinotime','value']).to_numpy()
            if len(reward_arr):
                try:
                    reward_amount_uL = np.unique(self.data['vstim']['reward'])[0]
                except:
                    reward_amount_uL = self.meta.rewardSize
                    self.logger.warning(f'No reward logged from vstim, using rewardSize from prot file')
                reward_arr = np.append(reward_arr,reward_arr[:,1])
                reward_arr[1] = reward_amount_uL
                reward_arr = reward_arr.tolist() 
                # reward is a 3 element array: [time,value_il, value_ms]
            else:
                reward_arr = None
        else:
            reward_arr = None
        
        reward_dict = {'reward':reward_arr}
        self._attrs_from_dict(reward_dict)
        return reward_dict
        
    def get_opto(self) -> dict:
        """ Extracts the opto boolean from opto slice from riglog"""
        if self.meta.opto:
            if 'opto' in self.data.keys() and len(self.data['opto']):
                opto_arr = self.data['opto'].select(['duinotime']).to_numpy()
                if len(opto_arr) > 1 and self.meta.opto_mode == 0:
                    self.logger.warning(f'Something funky happened with opto stim, there are {len(opto_arr)} pulses')
                    opto_arr = opto_arr[0]
                elif len(opto_arr) > 1 and self.meta.opto_mode == 1:
                    opto_arr = opto_arr[:,0]
                is_opto = True
                opto_arr = opto_arr.tolist()
            else:
                # is_opto = int(bool(vstim_dict.get('opto',0)) or bool(len(opto_pulse)))
                is_opto = False
                opto_arr = []
                if self.opto_pattern is not None and self.opto_pattern >= 0:
                    self.logger.warning('stimlog says opto, but no opto logged in riglog, using screen event as time!!')
                    is_opto = True
                    opto_arr = [[self.t_stimstart_rig]]
        else:
            is_opto = False
            opto_arr = [[]]
        
        opto_dict = {'opto' : is_opto,
                     'opto_pulse' : opto_arr}
        self._attrs_from_dict(opto_dict)
        return opto_dict
    
    def get_screen_events(self) -> dict:
        """ Gets the screen pulses from rig data """
        screen_dict = {'t_stimstart_rig' : None,
                       't_stimend_rig' : None}
        
        if 'screen' in self.data.keys():
            screen_data = self.data['screen']
            screen_arr = screen_data.select(['duinotime','value']).to_numpy()

            if len(screen_arr) == 1:
                self.logger.error('Only one screen event! Stimulus appeared but not dissapeared?')
                # assumes the single pulse is stim on
                screen_dict['t_stimstart_rig'] = screen_arr[0,0]
            elif len(screen_arr) > 2:
                # TODO: 3 SCREEN EVENTS WITH OPTO BEFORE STIMULUS PRESENTATION?
                self.logger.error('More than 2 screen events per trial, this is not possible')
            elif len(screen_arr) == 0:
                self.logger.critical('NO SCREEN EVENT FOR STIMULUS TRIAL!')
            else:
                # This is the correct stim ON/OFF scenario
                screen_dict['t_stimstart_rig'] = screen_arr[0,0]
                screen_dict['t_stimend_rig'] = screen_arr[1,0]
                
        self._attrs_from_dict(screen_dict)        
        return screen_dict
    
    def get_frames(self,get_from:str=None, **kwargs) -> dict:
        """ Extracts the frames from designated imaging mode, returns None if no"""

        frame_ids = []
        if not self.meta.imaging_mode is None:
            if get_from in self.data.keys():
                """
                NOTE: even if there's no actual recording for onepcam through labcams(i.e. the camera is running in the labcams GUI without saving), 
                if there is onepcam frame TTL signals coming into the Arduino it will save them.
                This will lead to having onepcam_frame_ids column to be created but there will be no actual tiff files.
                """
                rig_frames_data = self.data[get_from] # this should already be the frames of trial dur

                if self.t_stimstart_rig is not None:
                    # get stim present slice
                    rig_frames_data = rig_frames_data.filter((pl.col('duinotime') >= self.t_stimstart_rig) & 
                                                            (pl.col('duinotime') <= self.t_stimend_rig))
                    
                    if len(rig_frames_data):
                        frame_ids = [int(rig_frames_data[0,'value']), int(rig_frames_data[-1,'value'])]
                
                    else:
                        self.logger.critical(f'{get_from} no camera pulses recorded during stim presentation!!! THIS IS BAD!')
        frames_dict = {f'{get_from}_frame_ids':frame_ids}
        
        self._attrs_from_dict(frames_dict)
        return frames_dict