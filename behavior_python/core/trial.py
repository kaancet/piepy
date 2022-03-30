import pandas as pd
from ..utils import *
from .dbinterface import DataBaseInterface


WHEEL_CIRCUM = 2 * np.pi* 31.2
WHEEL_TICKS_PER_REV = 1024

def reset_wheel_pos(traj,reset_idx=0):
    """ Function that resets the wheel pos"""
    return traj - traj[reset_idx]

class Trial:
    __slots__ = ['trial_no','data','trial_data','column_keys',
                 'trial_start','trial_end','meta','db_interface','total_trial_count']
    def __init__(self,trial_no:int,column_keys:dict,meta) -> None:
        
        self.trial_no = trial_no
        self.column_keys = column_keys
        self.trial_data = {}
        self.meta = meta
        
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
        state_slice = states[states[self.column_keys['trialNo']] == self.trial_no]
        self.data = {'state' : state_slice}
        
        self.trial_start = state_slice[self.column_keys['elapsed']].iloc[0]
        self.trial_end = state_slice[self.column_keys['elapsed']].iloc[-1]

        for k,v in rawdata.items():
            if k == 'stateMachine':
                continue
            if not v.empty:
                t_start = self.trial_start
                t_end = self.trial_end
                
                if k in rig_cols:
                    time_col = 'duinotime'
                if k == 'vstim':
                    time_col = 'presentTime'
                    t_start = self.trial_start / 1000
                    t_end = self.trial_end / 1000
                  
                temp_v = v[(v[time_col] >= t_start) & (v[time_col] <= t_end)].dropna()
                self.data[k] = temp_v
                
    def get_licks(self) -> np.ndarray:
        """ Extracts the lick data from slice"""
        if 'lick' in self.data.keys():
            lick_data = self.data['lick']
            lick_arr = np.array(lick_data[['duinotime', 'value']])
        else:
            lick_arr = np.array([])
        
        return lick_arr
    
    def get_reward(self) -> np.ndarray:
        """ Extracts the reward clicks from slice"""
        if 'reward' in self.data.keys():
            reward_data = self.data['reward']
            reward_arr = np.array(reward_data[['duinotime', 'value']])
            reward_arr[:,1] *= self.meta.rewardSize
        else:
            reward_arr = np.array([])
        return reward_arr    
        
    def get_opto(self) -> np.ndarray:
        """ Extracts the opto boolean from opto slice from riglog"""
        if 'opto' in self.data.keys():
            opto_data = self.data['opto']
            opto_arr = np.array(opto_data[['duinotime','value']])
        else:
            opto_arr = np.array([])
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
