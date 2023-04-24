import time
import tifffile as tf
import scipy.stats as st
from PIL import Image
from os.path import join as pjoin
from behavior_python.core.session import Session, SessionData, SessionMeta
from .wheelDetectionTrial import *

class WheelDetectionData(SessionData):
    __slots__ = ['stim_data','data_paths','key_pairs','pattern_imgs','patterns']
    def __init__(self,data:pl.DataFrame,data_paths,isgrating:bool=False, ) -> None:
        super().__init__(data)
        self._convert = ['wheel','lick','reward']
        self.data = data
        self.data_paths = data_paths
        # self.data = get_running_stats(self.data)
        self.pattern_imgs, self.patterns, pattern_names = self.get_session_images()
        
        # self.stim_data = self.seperate_stim_data(self.data,pattern_names,isgrating)
        self.enhance_data(self.data,pattern_names,isgrating)
        
    def enhance_data(self,data_in:pl.DataFrame,pattern_names:dict,isgrating:bool=False) -> None:
        
        # add a isgrating column
        self.data = pl.concat([data_in,pl.DataFrame({"is_grating":[int(isgrating)]*len(data_in)})],how='horizontal')
        
        # add the pattern name depending on pattern id
        self.data = self.data.with_columns(pl.struct(["opto_pattern", "answer"]).apply(lambda x: pattern_names[x['opto_pattern']] if x['answer']!=-1 else None).alias('opto_region'))        
    
    def seperate_stim_data(self,data_in:pl.DataFrame,pattern_names:dict,isgrating:bool=False) -> None:
        """ Seperates the data into diffeerent types"""
        stim_data = {}
        nonearly_data = data_in.filter(pl.col('answer')!=-1)
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
        self.key_pairs = {}
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
                        # make a key pair dict for better labels when plotting, i.e area names
                        self.key_pairs[key_new] = '_'.join(key_new.split('_')[:-1]) + '_' + pattern_names[opto_pattern]

                        # stimuli data
                        try:
                            # if opto_pattern exists, ususally it should exist
                            stimuli_data = self.get_subset(data_in,{'spatial_freq':sfreq[i],
                                                            'temporal_freq':tfreq[i],
                                                            'opto':opto,
                                                            'opto_pattern':opto_pattern})
                            
                        except:
                            stimuli_data = self.get_subset(data_in,{'spatial_freq':sfreq[i],
                                                            'temporal_freq':tfreq[i],
                                                            'opto':opto})
                        
                        stim_data[key_new] = stimuli_data
                else:
                    key_new = key
                    self.key_pairs[key_new] = key_new
                    stimuli_data = self.get_subset(data_in,{'spatial_freq':sfreq[i],
                                                    'temporal_freq':tfreq[i],
                                                    'opto':opto})
                    # early trials don't have any of vstim values => spatial_freq, temporal_freq and opto_pattern
                    # a very crude fix is to get all the early data, concat and order by trial_no
                    # early_data = self.get_subset(data_in,{'answer':-1})
                    # stimuli_data = pd.concat([stimuli_data,early_data])
                    # stimuli_data.sort_values('trial_no',inplace=True)
                    
                    stim_data[key_new] = stimuli_data
        return stim_data
    
    def get_session_images(self):
        """ Reads the related session images(window, pattern,etc)
        Returns a dict with images and also a dict that """
        sesh_imgs = {}
        pattern_names = {}
        sesh_patterns = {}
        if os.path.exists(self.data_paths.patternPath):
            for im in os.listdir(self.data_paths.patternPath):
                if im.endswith('.tif'):
                    pattern_id = int(im[:-4].split('_')[-1])
                    read_img = tf.imread(pjoin(self.data_paths.patternPath,im))
                    if pattern_id == -1:
                        sesh_imgs['window'] = read_img
                        pattern_names[pattern_id] = 'nonopto'
                    else:  
                        name = im[:-4].split('_')[-2]
                        pattern_names[pattern_id] = name
                        sesh_imgs[name] = read_img
                elif im.endswith('.bmp'):
                    pattern_id = int(im.split('_')[0])
                    name = im.split('_')[1]
                    read_bmp = np.array(Image.open(pjoin(self.data_paths.patternPath,im)))
                    sesh_patterns[name] = read_bmp[::-1,::-1]
        return sesh_imgs,sesh_patterns,pattern_names
        

class WheelDetectionStats:
    __slots__ = ['all_count','early_count','stim_count','correct_count','miss_count',
                 'all_correct_percent','hit_rate','easy_hit_rate','false_alarm','nogo_percent',
                 'median_response_time','d_prime']
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
        early_data = data.filter(pl.col('answer')==-1)
        stim_data = data.filter(pl.col('answer')!=-1)
        correct_data = data.filter(pl.col('answer')==1)
        miss_data = data.filter(pl.col('answer')==0)
        
        #counts
        self.all_count = len(data)
        self.early_count = len(early_data)
        self.stim_count = len(stim_data)
        self.correct_count = len(correct_data) 
        self.miss_count = len(miss_data)
        
        # percents
        self.all_correct_percent = round(100 * self.correct_count / self.all_count, 3)
        self.hit_rate = round(100 * self.correct_count / self.stim_count, 3)
        self.false_alarm = round(100 * self.early_count / (self.early_count + self.correct_count), 3)
        self.nogo_percent = round(100 * self.miss_count / self.stim_count, 3)
        
        ## performance on easy trials
        easy_data = data.filter(pl.col('contrast').is_in([1.0,0.5])) # earlies can't be easy or hard
        easy_correct_count = len(easy_data.filter(pl.col('answer')==1))
        self.easy_hit_rate = round(100 * easy_correct_count / len(easy_data),3)
        
        # median response time
        self.median_response_time = round(stim_data.filter(pl.col('answer')==1).median()[0,'response_latency'],3)
        
        #d prime(?)
        self.d_prime = st.norm.ppf(self.hit_rate/100) - st.norm.ppf(self.false_alarm/100)
        
        # if self.all_trials >= 200:
        #     data200 = data[:200]
        # else:
        #     data200 = data
            
        # data200_answered = len(data200[data200['answer']!=0])
        # data200_nogo = len(data200[data200['answer']==0])
        
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
            self.data = WheelDetectionData(session_data,self.data_paths,isgrating=g)
            self.stats = WheelDetectionStats(data_in=self.data)
            
            if self.meta.water_consumed is not None:
                self.meta.water_per_reward = self.meta.water_consumed / self.stats.correct_count
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
        self.save_session_data(self.data.data)

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
        self.data = WheelDetectionData(rawdata,self.data_paths,isgrating=g)
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
    
    def get_session_data(self) -> pl.DataFrame:
        data_to_append = []
        self.rawdata['stateMachine'] = self.rawdata['stateMachine'].with_column(pl.struct(['oldState','newState']).apply(lambda x: self.translate_transition(x['oldState'],x['newState'])).alias('transition'))
        self.states = self.rawdata['stateMachine']
        
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
                if t == 1:
                    data_to_append = {k:[v] for k,v in trial_row.items()}
                else:
                    for k,v in trial_row.items():
                        data_to_append[k].append(v)
            else:
                if t == len(trials):
                    display(f'Last trial {t} is discarded')
                    
        session_data = pl.DataFrame(data_to_append)

        if session_data.is_empty():
            print('''WARNING THIS SESSION HAS NO DATA
            Possible causes:
            - Session has only one trial with no correct answer''')
            return None
        else:
            return session_data


@timeit('Getting rolling averages...')
def get_running_stats(data_in:pd.DataFrame,window_size:int=10) -> pd.DataFrame:
    """ Gets the running statistics of certain columns"""
   
    # response latency
    data_in.loc[:,'running_response_latency'] = data_in.loc[:,'response_latency'].rolling(window_size,min_periods=5).median()
    
    data_in.with_columns(pl.col('response_latency').rolling_median(window_size).alias('running_response_latency'))

    
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