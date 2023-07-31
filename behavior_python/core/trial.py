from numpy import ndarray
import pandas as pd
from ..utils import *
from .dbinterface import DataBaseInterface


class Trial:
    __slots__ = ['trial_no','data','trial_data','column_keys',
                 'trial_start','trial_end','meta','db_interface','total_trial_count',
                 'reward_ms_per_ul']
    def __init__(self,trial_no:int,column_keys:dict,meta) -> None:
        
        self.trial_no = trial_no
        self.column_keys = column_keys
        self.trial_data = {}
        self.meta = meta
        self.reward_ms_per_ul = 0
        
        config = getConfig()
        self.db_interface = DataBaseInterface(config['databasePath'])
        
    def __repr__(self):
        rep = f"""Trial No :{self.trial_no}
        {self.data['state']}"""
        return rep
    
    def get_data_slices(self,rawdata) -> pd.DataFrame:
        """ Extracts the relevant portion from each data """
        rig_cols = ['screen','imaging','position','lick',
                    'button','reward','lap','cam1','cam2',
                    'cam3','act0','act1','opto']
        
        states = rawdata['stateMachine']
        state_slice = states.filter(pl.col(self.column_keys['trialNo'])==self.trial_no)
        self.data = {'state' : state_slice}
        
        self.trial_start = state_slice[self.column_keys['elapsed']][0]
        self.trial_end = state_slice[self.column_keys['elapsed']][-1]

        for k,v in rawdata.items():
            if k == 'stateMachine':
                continue
            if not v.is_empty():
                t_start = self.trial_start
                t_end = self.trial_end
                
                if k in rig_cols:
                    time_col = 'duinotime'
                if k == 'vstim':
                    time_col = 'presentTime'
                    t_start = self.trial_start / 1000
                    t_end = self.trial_end / 1000
                
                temp_v = v.filter((pl.col(time_col) >= t_start) & (pl.col(time_col) <= t_end)).drop_nulls()
                 
                self.data[k] = temp_v
                
    def get_licks(self) -> np.ndarray:
        """ Extracts the lick data from slice"""
        if 'lick' in self.data.keys():
            lick_data = self.data['lick']
            if len(lick_data):
            # lick_arr = np.array(lick_data[['duinotime', 'value']])
                lick_arr = lick_data.select(['duinotime','value']).to_series().to_list()
            else:
                lick_arr = None
        else:
            lick_arr = None
        
        return lick_arr
    
    def get_reward(self) -> np.ndarray:
        """ Extracts the reward clicks from slice"""
        
        reward_data = self.data['reward']
        # reward_arr = np.array(reward_data[['duinotime', 'value']])
        reward_arr = reward_data.select(['duinotime','value']).to_numpy()
        if len(reward_arr):
            try:
                reward_amount_uL = np.unique(self.data['vstim']['reward'])[0]
            except:
                reward_amount_uL = self.meta.rewardSize
            reward_arr = np.append(reward_arr,reward_arr[:,1])
            reward_arr[1] = reward_amount_uL
            reward_arr = reward_arr.tolist() 
            # reward is a 3 element array: [time,value_il, value_ms]
        else:
            reward_arr = None
        return reward_arr
        
    def get_opto(self) -> np.ndarray:
        """ Extracts the opto boolean from opto slice from riglog"""
        if 'opto' in self.data.keys():
            opto_data = self.data['opto']
            # opto_arr = np.array(opto_data[['duinotime','value']])
            opto_arr = opto_data.select(['duinotime','value']).to_numpy()
            if len(opto_arr) > 1:
                display(f'>> WARNING << Something funky happened with opto stim, there are {len(opto_arr)} pulses')
                opto_arr = np.array([opto_arr[0]]).tolist()
        else:
            opto_arr = None
        return opto_arr
    
    def save_to_db(self,in_dict:dict,table_name:str=None,**kwargs):
        """ Checks if an entry exists and saves/updates accordingly"""
        if table_name is None:
            table_name = 'trials'
        
        total_trial_no = int(self.trial_no) + kwargs.get('total_trial_no',0)
        const_dict = {'id':self.meta.animalid,
                      'sessionId':self.meta.session_id,
                      'date':self.meta.baredate,
                      'trial_no':int(self.trial_no),
                      'total_trial_no':total_trial_no}
        
        db_dict = {**const_dict ,**in_dict}
        
        if not self.db_interface.exists(db_dict,table_name):
            self.db_interface.add_entry(db_dict,table_name,verbose=False)
        else:
            display(f'Trial with id {self.meta.session_id} is already in database, updating the entry')
            self.db_interface.update_entry(const_dict,db_dict,table_name,verbose=False)
