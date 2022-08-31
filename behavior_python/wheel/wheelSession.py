import time
from os.path import join as pjoin
from behavior_python.core.session import *
from behavior_python.model.transformer import GLMHMMTransfromer
from behavior_python.model.glmModel import Glm
from sklearn import preprocessing
from .wheelTrial import *

LOWSF = 0.05
HIGHSF = 0.4
LOWTF = 0.5
HIGHTF = 16


class WheelData(SessionData):
    __slots__ = ['stim_data']
    def __init__(self,data:pd.DataFrame,isgrating:bool=False) -> None:
        self._convert = ['stim_pos','wheel','lick','reward']
        self.data = data
        self.make_loadable()
        self.stim_data:dict = self.seperate_stim_data(isgrating)

    def __repr__(self):
        rep = f'''WheelData Object 
        stim_types = {list(self.stim_data.keys())}'''
        return rep
        
    def get_novel_trials(self,stim_type:str=None,get_running:bool=True) -> pd.DataFrame:
        if stim_type is not None:
            if stim_type in self.stim_data.keys():
                novel_df = self.stim_data[stim_type]
        else:
            novel_df = self.data[self.data['correction']==0]

        if get_running:
           novel_df = get_running_stats(novel_df)

        return novel_df
    
    def get_answered_trials(self) -> pd.DataFrame:
        answered_df = self.data[(self.data['correction']==0) & (self.data['answer']!=0)]
        return answered_df
    
    def seperate_stim_data(self,isgrating:bool=False) -> None:
        """ Seperates the data into diffeerent types"""
        stim_data = {}
        sfreq,s_idx = np.unique(self.data['spatial_freq'],return_index=True)
        sfreq = sfreq[s_idx.argsort()]
        tfreq,t_idx = np.unique(self.data['temporal_freq'],return_index=True)
        tfreq = tfreq[t_idx.argsort()]

        optogenetic,o_idx = np.unique(self.data['opto'],return_index=True)
        optogenetic = optogenetic[o_idx.argsort()]
        
        if 'opto_pattern' in self.data.columns:
            pattern_ids = np.unique(self.data['opto_pattern'])
            # remove nans
            pattern_ids = pattern_ids[~np.isnan(pattern_ids)]
            # remove -1(no opto no pattern, thi is extra failsafe)
            pattern_ids = pattern_ids[pattern_ids >= 0]
        else:
            pattern_ids = [-1]
    
        # analysing each stim type and opto and opto_pattern seperately
        for opto in optogenetic:
            for i,_ in enumerate(sfreq):
                skey = sfreq[i] if sfreq[i]%1 else int(sfreq[i])
                tkey = tfreq[i] if tfreq[i]%1 else int(tfreq[i])
                key = f'{skey}cpd_{tkey}Hz'
                if isgrating:
                    key += '_grating'

                for opto_pattern in pattern_ids:
                    if opto == 0:
                        opto_pattern = -1
                        key_new = key
                    else:
                        key_new = '{0}_opto_{1}'.format(key,int(opto_pattern))

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
                    stim_data[key_new] = stimuli_data
        return stim_data


class WheelStats:
    __slots__ = ['all_trials','novel_trials','answered_trials',
                 'all_correct_percent','novel_correct_percent','answered_correct_percent',
                 'easy_answered_trials','easy_answered_correct_percent',
                 'left_distribution','right_distribution','median_response_time','cutoff_time',
                 'bias','nogo_percent','model_info']
    def __init__(self,dict_in:dict=None,data_in:WheelData=None) -> None:

        if data_in is not None:
            self.init_from_data(data_in)
        elif dict_in is not None:
            self.init_from_dict(dict_in)

    def __repr__(self):
        rep = ''''''
        for k in self.__slots__:
            rep += f'''{k} = {getattr(self,k,None)}\n'''
        return rep

    def init_from_data(self,data_in:WheelData):
        data = data_in.data
        novel_data = data_in.get_novel_trials()
        answered_data = data_in.get_answered_trials()
        #counts
        self.all_trials = len(data)
        self.novel_trials = len(novel_data)
        self.answered_trials =len(answered_data)
        
        # percents
        self.all_correct_percent = round(100 * len(data[data['answer']==1]) / len(data),3)
        self.novel_correct_percent = round(100 * len(novel_data[novel_data['answer']==1]) / len(novel_data),3)
        self.answered_correct_percent = round(100 * len(answered_data[answered_data['answer']==1]) / len(answered_data),3)
        
        ## performance on easy trials
        easy_trials = data[data['contrast'].isin([1.0,0.5])]
        easy_answered_data = easy_trials[(easy_trials['correction']==0) & (easy_trials['answer']!=0)]
        self.easy_answered_trials = len(easy_answered_data)
        self.easy_answered_correct_percent = round(100 * len(easy_answered_data[easy_answered_data['answer']==1]) / len(easy_answered_data),3)

        #left
        left_data = novel_data[novel_data['stim_side'] < 0]
        self.left_distribution= [len(left_data[left_data['answer']==1]), len(left_data[left_data['answer']==-1]),len(left_data[left_data['answer']==0])]
        #right 
        right_data = novel_data[novel_data['stim_side'] > 0]
        self.right_distribution = [len(right_data[right_data['answer']==1]), len(right_data[right_data['answer']==-1]),len(right_data[right_data['answer']==0])]

        self.median_response_time = round(np.median(answered_data['response_latency']),3)

        # add session response_time cutoff to summary by taking the mean of first 10% of trials
        # the overall dataset is used for this to have a more balanced cutoff threshold
        first10p = int(len(data)/10)
        cutoff = np.mean(data.loc[:first10p,'response_latency'],axis=None) * 2
        self.cutoff_time = round(cutoff/1000,3)
        
        n_left_resp = self.left_distribution[0] + self.right_distribution[1]
        n_right_resp = self.right_distribution[0] + self.left_distribution[1]
        self.bias = np.round((n_right_resp - n_left_resp)/(n_right_resp + n_left_resp),3)
        
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
        return {key : getattr(self, key, None) for key in self.__slots__}


class WheelSession(Session):
    """ A class to analyze the wheel trainings and experiments

        :param sessiondir: directory of the session inside the presentation folder(e.g. 200619_KC33_wheel_KC)
        :type sessiondir:  str
        :type autplot: boolean
        """
    def __init__(self,sessiondir, *args,**kwargs):
        super().__init__(sessiondir,*args,**kwargs)
        
        start = time.time()

        # add specific data paths
        self.data_paths.metaPath = pjoin(self.data_paths.savePath,'sessionMeta.json')
        self.data_paths.statPath = pjoin(self.data_paths.savePath,'sessionStats.json')

        self.running_stats = ['response_latency']

        if self.isSaved() and self.load_flag:
            display('Loading from {0}'.format(self.data_paths.savePath))
            self.load_session()
        else:
            self.set_meta()
            self.read_data()
            self.set_statelog_column_keys()

            session_data = self.get_session_data()
            session_data = get_running_stats(session_data)
            
            g = 'grating' in self.data_paths.stimlog
            self.data = WheelData(session_data,isgrating=g)
            self.stats = WheelStats(data_in=self.data)
            self.meta.water_given = round(float(np.sum([a[1] for a in self.data.data['reward'] if len(a)])),3)
            if self.meta.water_consumed is not None:
                self.meta.water_per_reward = self.meta.water_consumed / len(self.data.data[self.data.data['answer']==1])
            else:
                display('CONSUMED REWARD NOT ENTERED IN GOOGLE SHEET')
                self.meta.water_per_reward = -1
            
            self.save_session()

        end = time.time()
        display('Done! t={0:.2f} s'.format(end-start))

    def __repr__(self):
        rep = f'''{self.meta.animalid} Wheel Session at {self.meta.nicedate}
        answered trials = {self.stats.answered_trials}
        correct percent = {self.stats.answered_correct_percent}'''
        return rep

    def set_meta(self) -> None:

        self.meta = SessionMeta(self.data_paths.prot)
        self.meta.logversion = self.logversion
        self.meta.set_rig(self.data_paths.prefs)

        self.meta.contrastVector = [float(i) for i in self.meta.contrastVector.strip('] [').strip(' ').split(',')]
        if hasattr(self.meta,'easyContrast'):
            self.meta.easyContrast = [float(i) for i in self.meta.easyContrast.strip('] [').strip(' ').split(',')]

        if hasattr(self.meta,'stimRegion'):
            self.meta.stimRegion = [float(i) for i in self.meta.stimRegion.strip('] [').strip(' ').split(',')]

        tmp = self.data_paths.prot
        # get level 
        # TODO: (Make this better)
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

        save_dict_json(self.data_paths.statPath, self.stats.get_dict())
        
        # self.save_session_db()
        display(f' Saving to {self.data_paths.savePath}')
    

    @timeit('Loaded all data')
    def load_session(self):
        """ Loads the saved session data """
        rawdata = self.load_session_data()

        meta = load_json_dict(self.data_paths.metaPath)
        self.meta = SessionMeta(init_dict=meta)
        display('Loaded session metadata')

        g = 'grating' in self.data_paths.stimlog
        self.data = WheelData(rawdata,isgrating=g)
        display('Loaded session data')

        stats = load_json_dict(self.data_paths.statPath)
        self.stats = WheelStats(dict_in=stats)
        display('Loaded Session stats')

    def translate_transition(self,oldState,newState):
        """ A function to be called that add the meaning of state transition into the state DataFrame"""
        curr_key = '{0}->{1}'.format(int(oldState), int(newState))
        state_keys = {'0->1' : 'trialstart',
                      '1->2' : 'openloopstart',
                      '2->3' : 'closedloopstart',
                      '3->4' : 'correct',
                      '3->5' : 'incorrect',
                      '6->0' : 'trialend',
                      '4->6' : 'stimendcorrect',
                      '5->6' : 'stimendincorrect'}

        if self.logversion == 'pyvstim':
            state_keys['3->7'] = 'nonanswer'
            state_keys['2->5'] = 'earlyanswer'
            state_keys['7->1'] = 'correction'
            state_keys['5->7'] = 'na'
            state_keys['7->6'] = 'na'

            if self.level == 'level0':
                state_keys['2->8'] = 'closedloopstart'
                state_keys['8->4'] = 'correct'
        
        return state_keys.get(curr_key,None)

    @timeit('Extracted Session data')
    def get_session_data(self) -> pd.DataFrame:
        """ The main loop where the parsed session data is created
            :return: Parsed sessiondata
            :rtype:  DataFrame
        """
        session_data = pd.DataFrame()
        data_to_append = []
        self.states = self.rawdata['stateMachine']
        self.states['transition'] = self.states.apply(lambda x: self.translate_transition(x[self.column_keys['oldState']],x[self.column_keys['newState']]),axis=1)
        display('Setting global indexing keys for {0} logging'.format(self.logversion))
        latest_total_rial_count = self.get_latest_trial_count()
        
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
            temp_trial = WheelTrial(t, self.column_keys, meta=self.meta)
            temp_trial.get_data_slices(self.rawdata)
            trial_row = temp_trial.trial_data_from_logs()
            # temp_trial.save_trial_db(total_trial_no=latest_total_rial_count)
            pbar.update()
            if len(trial_row):
                data_to_append.append(trial_row)
            else:
                if t == len(trials):
                    display(f'Last trial {t} is discarded')
     
        session_data = session_data.append(data_to_append,ignore_index=True)

        if session_data.empty:
            print('''WARNING THIS SESSION HAS NO DATA
            Possible causes:
            - Session has only one trial with no correct answer''')
            return None
        else:
            return session_data
        
    def save_session_db(self) -> None:
        """ Saves the session data to the database sessions table in the database"""
        mouse = self.db_interface.get_entries({'id':self.meta.animalid},'animals')
        if mouse.empty:
            display(f'There is no entry for animal {self.meta.animalid}')
            return 0
        self.current_session_no = self.overall_session_no()

        db_dict = {'sessionName':self.sessiondir,
                    'user':self.meta.user,
                    'sessionId':self.meta.session_id,
                    'framework':mouse['framework'].iloc[0],
                    'id':self.meta.animalid,
                    'date':self.meta.baredate,
                    'time':self.meta.time,
                    'age':mouse['age'].iloc[0],
                    'sessionNo': self.current_session_no,
                    'sessionLevel': self.meta.level,
                    'trialCount':self.stats.all_trials,
                    'wheelGain': self.meta.wheelGain,
                    'rewardSize': self.meta.rewardSize,
                    'weight':self.meta.weight,
                    'water_given' : self.meta.water_given,
                    'water_consumed' : self.meta.water_consumed,
                    'screenPosition':'bino',
                    'nExperiments':1,
                    'nCells':0}
        self.save_to_db(db_dict)
        
@timeit('Getting rolling averages...')
def get_running_stats(data_in:pd.DataFrame,window_size:int=10) -> pd.DataFrame:
    """ Gets the running statistics of certain columns"""
    data_in.reset_index(drop=True, inplace=True)
    # response latency
    data_in.loc[:,'running_response_latency'] = data_in.loc[:,'response_latency'].rolling(window_size,min_periods=5).median()
    
    # answers
    answers = {'correct':1,
               'nogo':0,
               'incorrect':-1}
    
    for k,v in answers.items():
        key = 'fraction_' + k
        data_arr = data_in['answer'].to_numpy()
        data_in[key] = get_fraction(data_arr, fraction_of=v)
        
    return data_in        


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Wheel Session Analysis')

    parser.add_argument('expname',metavar='expname',
                        type=str,help='Experiment filename (e.g. 200325_KC020_wheel_KC)')
    parser.add_argument('-l','--load',metavar='load_flag',default=True,
                        type=str,help='Flag for loading existing data')

    opts = parser.parse_args()
    expname = opts.expname
    load_flag = opts.load

    w = WheelSession(sessiondir=expname, load_flag=load_flag)

if __name__ == '__main__':
    main()

