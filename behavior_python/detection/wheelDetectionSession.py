import time
from os.path import join as pjoin
import scipy.stats as st
from behavior_python.core.session import Session, SessionData, SessionMeta
from .wheelDetectionTrial import *

class WheelDetectionData(SessionData):
    __slots__ = ['stim_data']
    def __init__(self,data:pd.DataFrame,isgrating:bool=False) -> None:
        super().__init__(data,cutoff_time=-1)
        self._convert = ['wheel','lick','reward']
        self.data = data
        self.make_loadable()
        self.data = get_running_stats(self.data)
        
        self.stim_data = self.seperate_stim_data(self.data,isgrating)
    
    def get_answered_trials(self,data_in:pd.DataFrame) -> pd.DataFrame:
        """ Correct answers and early answers """
        answered_df = data_in[data_in['answer']!=0]
        return answered_df
    
    def get_wait_trials(self,data_in:pd.DataFrame) -> pd.DataFrame:
        wait_df = data_in[data_in['answer']!=-1]
        return wait_df
    
    def seperate_stim_data(self,data_in:pd.DataFrame,isgrating:bool=False) -> None:
        """ Seperates the data into diffeerent types"""
        stim_data = {}
        nonearly_data = data_in[data_in['answer']!=-1]
        sfreq = nonan_unique(nonearly_data['spatial_freq'].to_numpy())
        tfreq = nonan_unique(nonearly_data['temporal_freq'].to_numpy())
        optogenetic = np.unique(nonearly_data['opto'])
        
        if 'opto_pattern' in nonearly_data.columns:
            pattern_ids = nonan_unique(nonearly_data['opto_pattern'])

            # remove -1(no opto no pattern, thi is extra failsafe)
            pattern_ids = pattern_ids[pattern_ids >= 0]
        else:
            pattern_ids = [-1]
    
        # analysing each stim type and opto and opto_pattern seperately
        for opto in optogenetic:
            for i,_ in enumerate(sfreq):
                skey = float(sfreq[i])
                tkey = float(tfreq[i])
                key = f'{skey}cpd_{tkey}Hz'
                if isgrating:
                    key += '_grating'
                
                if opto:
                    for opto_pattern in pattern_ids:
                        key_new = '{0}_opto_{1}'.format(key,int(opto_pattern))
                    
                        print(key_new)
                        # stimuli data
                        try:
                            # if opto_pattern exists, ususally it should exist
                            stimuli_data = self.get_subset({'spatial_freq':sfreq[i],
                                                            'temporal_freq':tfreq[i],
                                                            'opto':opto,
                                                            'opto_pattern':opto_pattern})
                            
                        except:
                            stimuli_data = self.get_subset({'spatial_freq':sfreq[i],
                                                            'temporal_freq':tfreq[i],
                                                            'opto':opto})
                            
                    
                else:
                    key_new = key
                    stimuli_data = self.get_subset({'spatial_freq':sfreq[i],
                                                    'temporal_freq':tfreq[i],
                                                    'opto':opto})
                    # early trials don't have any of vstim values => spatial_freq, temporal_freq and opto_pattern
                    # a very crude fix is to get all the early data, concat and order by trial_no
                    early_data = self.get_subset({'answer':-1})
                    stimuli_data = pd.concat([stimuli_data,early_data])
                    stimuli_data.sort_values('trial_no',inplace=True)
                    
                stim_data[key_new] = stimuli_data
        return stim_data
        

class WheelDetectionStats:
    __slots__ = ['all_trials','answered_trials','stim_trials','early_trials','miss_trials',
                 'all_correct_percent','answered_correct_percent',
                 'easy_answered_trials','easy_hit_rate',
                 'hit_rate','false_alarm','d_prime',
                 'median_response_time','nogo_percent']
    def __init__(self,dict_in:dict=None,data_in:WheelDetectionData=None) -> None:
        if data_in is not None:
            self.init_from_data(data_in)
        elif dict_in is not None:
            self.init_from_dict(dict_in)

    def __repr__(self):
        rep = ''''''
        for k in self.__slots__:
            rep += f'''{k} = {getattr(self,k,None)}\n'''
        return rep
    
    def init_from_data(self,data_in:WheelDetectionData):
        data = data_in.data
        answered_data = data_in.get_answered_trials(data)
        early_data = answered_data[answered_data['answer']==-1]
        stim_data = data_in.get_wait_trials(data)
        miss_data = stim_data[stim_data['answer']==0]
        #counts
        self.all_trials = len(data)
        self.answered_trials =len(answered_data)
        self.early_trials = len(early_data)
        self.stim_trials = len(stim_data)
        self.miss_trials = len(miss_data)
        # percents
        self.all_correct_percent = round(100 * len(data[data['answer']==1]) / self.all_trials,3)
        self.answered_correct_percent = round(100 * len(answered_data[answered_data['answer']==1]) / self.answered_trials,3)
        self.hit_rate = round(100 * len(stim_data[stim_data['answer']==1]) / self.stim_trials,3)
        self.false_alarm = round(100 * len(answered_data[answered_data['answer']==-1]) / self.answered_trials,3)
        
        ## performance on easy trials
        easy_trials = data[data['contrast'].isin([1.0,0.5])] # earlies can't be easy or hard
        easy_answered_data = easy_trials[easy_trials['answer']==1]
        self.easy_answered_trials = len(easy_answered_data)
        self.easy_hit_rate = round(100 * self.easy_answered_trials / len(easy_trials),3)
        
        
        self.median_response_time = round(np.median(stim_data[stim_data['answer']==1]['response_latency']),3)
        
        #d prime
        self.d_prime = st.norm.ppf(self.hit_rate/100) - st.norm.ppf(self.false_alarm/100)
        
        if self.all_trials >= 200:
            data200 = data[:200]
        else:
            data200 = data
            
        data200_answered = len(data200[data200['answer']!=0])
        data200_nogo = len(data200[data200['answer']==0])
        self.nogo_percent = round(100 * (data200_nogo / data200_answered),3)
        

    def init_from_dict(self,dict_in:dict):
        for k,v in dict_in.items():
            setattr(self,k,v)

    def get_dict(self)->dict:
        return {key : getattr(self, key, None)for key in self.__slots__}

    
class WheelDetectionSession(Session):
    def __init__(self, sessiondir, *args, **kwargs):
        super().__init__(sessiondir, *args, **kwargs)
        
        start = time.time()
        
        # add specific data paths
        self.data_paths.metaPath = pjoin(self.data_paths.savePath,'sessionMeta.json')
        self.data_paths.statPath = pjoin(self.data_paths.savePath,'sessionStats.json')
        
        if self.isSaved() and self.load_flag:
            display('Loading from {0}'.format(self.data_paths.savePath))
            self.load_session()
        else:
            self.set_meta()
            self.read_data()
            self.set_statelog_column_keys()

            session_data = self.get_session_data()
            # session_data = get_running_stats(session_data)
            
            g = 'grating' in self.data_paths.stimlog
            self.data = WheelDetectionData(session_data,isgrating=g)
            self.stats = WheelDetectionStats(data_in=self.data)
            
            if self.meta.water_consumed is not None:
                self.meta.water_per_reward = self.meta.water_consumed / len(self.data.data[self.data.data['answer']==1])
            else:
                display('CONSUMED REWARD NOT ENTERED IN GOOGLE SHEET')
                self.meta.water_per_reward = -1
            
            self.save_session()
            display('Saving data to {0}'.format(self.data_paths.savePath))
        
        end = time.time()
        display('Done! t={0:.2f} s'.format(end-start))
        
    def __repr__(self):
        r = f'Detection Session {self.sessiondir}'
        return r
    
    def set_meta(self):
        self.meta = SessionMeta(self.data_paths.prot)
        self.meta.logversion = self.logversion
        self.meta.set_rig(self.data_paths.prefs)
        
        self.meta.contrastVector = [float(i) for i in self.meta.contrastVector.strip('] [').strip(' ').split(',')]
        if hasattr(self.meta,'easyContrast'):
            self.meta.easyContrast = [float(i) for i in self.meta.easyContrast.strip('] [').strip(' ').split(',')]

        if hasattr(self.meta,'stimRegion'):
            self.meta.stimRegion = [float(i) for i in self.meta.stimRegion.strip('] [').strip(' ').split(',')]

        tmp = self.data_paths.prot

        lvl = ''
        if tmp.find('level') != -1:
            tmp = tmp[tmp.find('level')+len('level'):]
            for char in tmp:
                if char not in ['.','_']:
                    lvl += char
                else:
                    break      
        else:
            lvl = 'exp'
        self.meta.level = lvl
    
    @timeit('Saving...')
    def save_session(self) -> None:
        """ Saves the session data, meta and stats"""
        self.save_session_data(self.data.make_saveable())

        save_dict_json(self.data_paths.metaPath, self.meta.__dict__)
        display("Saved session metadata")

        save_dict_json(self.data_paths.statPath, self.stats.get_dict())
        display("Saved session stats")
    
    @timeit('Loaded all data')
    def load_session(self):
        """ Loads the saved session data """
        rawdata = self.load_session_data()

        meta = load_json_dict(self.data_paths.metaPath)
        self.meta = SessionMeta(init_dict=meta)
        display('Loaded session metadata')

        g = 'grating' in self.data_paths.stimlog
        self.data = WheelDetectionData(rawdata,isgrating=g)
        display('Loaded session data')

        stats = load_json_dict(self.data_paths.statPath)
        self.stats = WheelDetectionStats(dict_in=stats)
        display('Loaded Session stats')
        
    def translate_transition(self,oldState,newState):
        """ A function to be called that add the meaning of state transitions into the state DataFrame"""
        curr_key = '{0}->{1}'.format(int(oldState), int(newState))
        state_keys = {'0->1' : 'trialstart',
                      '1->2' : 'cuestart',
                      '2->3' : 'stimstart',
                      '2->5' : 'earlyanswer',
                      '3->4' : 'correct',
                      '3->5' : 'incorrect',
                      '3->6' : 'catch',
                      '6->0' : 'trialend',
                      '4->6' : 'stimendcorrect',
                      '5->6' : 'stimendincorrect'}
        
        return state_keys[curr_key] 
    
    def get_session_data(self) -> pd.DataFrame:
        session_data = pd.DataFrame()
        data_to_append = []
        self.states = self.rawdata['stateMachine']
        self.states['transition'] = self.states.apply(lambda x: self.translate_transition(x[self.column_keys['oldState']],x[self.column_keys['newState']]),axis=1)
        display('Setting global indexing keys for {0} logging'.format(self.logversion))
        
        if self.states.shape[0] == 0:
            display("""NO STATE MACHINE TO ANALYZE
                    LOGGING PROBLEMATIC.
                    SOLVE THIS ISSUE FAST""")
            return None
        
        trials = np.unique(self.states[self.column_keys['trialNo']])
        # this is a failsafe for some early stimpy data where trial count has not been incremented
        if len(trials) == 1 and len(self.states) > 6 and self.logversion=='stimpy':
            self.extract_trial_count()
            trials = np.unique(self.states[self.column_keys['trialNo']])
        pbar = tqdm(trials,desc='Extracting trial data:',leave=True,position=0)
        for t in pbar:
            temp_trial = WheelDetectionTrial(t,self.column_keys,meta=self.meta)
            temp_trial.get_data_slices(self.rawdata)
            trial_row = temp_trial.trial_data_from_logs()
            pbar.update()
            if len(trial_row):
                data_to_append.append(trial_row)
            else:
                if t == len(trials):
                    display(f'Last trial {t} is discarded')
                    
        session_data = pd.DataFrame(data_to_append)

        if session_data.empty:
            print('''WARNING THIS SESSION HAS NO DATA
            Possible causes:
            - Session has only one trial with no correct answer''')
            return None
        else:
            return session_data

@timeit('Getting rolling averages...')
def get_running_stats(data_in:pd.DataFrame,window_size:int=10) -> pd.DataFrame:
    """ Gets the running statistics of certain columns"""
    data_in.reset_index(drop=True, inplace=True)
    # response latency
    data_in.loc[:,'running_response_latency'] = data_in.loc[:,'response_latency'].rolling(window_size,min_periods=5).median()
    
    # answers
    answers = {'correct':1,
               'nogo':0,
               'early':-1}
    
    for k,v in answers.items():
        key = 'fraction_' + k
        data_arr = data_in['answer'].to_numpy()
        data_in[key] = get_fraction(data_arr, fraction_of=v)
        
    return data_in

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Wheel Detection Session Analysis')

    parser.add_argument('expname',metavar='expname',
                        type=str,help='Experiment filename (e.g. 200325_KC020_wheel_KC)')
    parser.add_argument('-l','--load',metavar='load_flag',default=True,
                        type=str,help='Flag for loading existing data')

    opts = parser.parse_args()
    expname = opts.expname
    load_flag = opts.load

    w = WheelDetectionSession(sessiondir=expname, load_flag=load_flag)

if __name__ == '__main__':
    main()