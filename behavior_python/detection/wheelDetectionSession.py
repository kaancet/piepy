import time
from PIL import Image
import tifffile as tf
import scipy.stats as st
from os.path import join as pjoin
from behavior_python.core.session import Session, SessionData, SessionMeta
from .wheelDetectionTrial import *


class WheelDetectionData(SessionData):
    def __init__(self,data_paths,data:pl.DataFrame=None) -> None:
        super().__init__()
        self.set_paths(data_paths)
        self.set_data(data)

    def set_data(self,data:pl.DataFrame,isgrating:bool=False) -> None:
        if data is not None:
            super().set_data(data)
            self.pattern_imgs, self.patterns, pattern_names = self.get_session_images()
            self.enhance_data(pattern_names,isgrating)
    
    def enhance_data(self,pattern_names:dict,isgrating:bool=False) -> None:
        # add a isgrating column
        if 'is_grating' not in self.data.columns:
            self.data = pl.concat([self.data,pl.DataFrame({"is_grating":[int(isgrating)]*len(self.data)})],how='horizontal')
            # add stim type
            self.data = self.data.with_columns((pl.col('spatial_freq').round(2)+'cpd_'+pl.col('temporal_freq')+'Hz').alias('stim_type'))
            
            if len(self.data['opto'].unique().to_numpy()) == 1:
                # add the pattern name depending on pattern id
                self.data = self.data.with_columns(pl.lit(None).alias('opto_region'))
                # add 'stimkey' from sftf
                self.data = self.data.with_columns((pl.col("stim_type")+'_-1').alias('stimkey'))
                # add stim_label for legends and stuff
                self.data = self.data.with_columns((pl.col("stim_type")).alias('stim_label'))
            else:
                try:
                # add the pattern name depending on pattern id
                    self.data = self.data.with_columns(pl.struct(["opto_pattern", "answer"]).apply(lambda x: pattern_names[x['opto_pattern']] if x['answer']!=-1 else None).alias('opto_region'))        
                except:
                    raise KeyError(f'Opto pattern not set correctly. You need to change the number at the end of the opto pattern image file to 0!')
                # add 'stimkey' from sftf
                self.data = self.data.with_columns((pl.col("stim_type")+'_'+pl.col("opto_pattern").cast(pl.Int8,strict=False)).alias('stimkey'))
                # add stim_label for legends and stuff
                self.data = self.data.with_columns((pl.col("stim_type")+'_'+pl.col("opto_region")).alias('stim_label'))
            
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
        early_data = data.filter((pl.col('answer')==-1) & (pl.col('isCatch')==0))
        stim_data = data.filter((pl.col('answer')!=-1) & (pl.col('isCatch')==0))
        catch_data = data.filter(pl.col('isCatch')==1)
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
        easy_data = data.filter(pl.col('contrast').is_in([100,50])) # earlies can't be easy or hard
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
        self.data = WheelDetectionData(self.data_paths)
        
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
            
            self.data.set_data(session_data,g)
            self.stats = WheelDetectionStats(data_in=self.data)
            
            if self.meta.water_consumed is not None:
                self.meta.water_per_reward = self.meta.water_consumed / self.stats.correct_count
            else:
                display('CONSUMED REWARD NOT ENTERED IN GOOGLE SHEET')
                self.meta.water_per_reward = -1
                
            # add some metadata to the dataframe
            self.data.data = self.data.data.with_columns([pl.lit(self.meta.animalid).alias('animalid'),
                                                          pl.lit(self.meta.baredate).alias('baredate')])
            
            self.data.data = self.data.data.with_columns(pl.col('baredate').str.strptime(pl.Date, fmt='%y%m%d').cast(pl.Date).alias('date'))
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
        self.data.save_data(self.save_mat)

        save_dict_json(self.data_paths.metaPath, self.meta.__dict__)
        display("Saved session metadata")

        save_dict_json(self.data_paths.statPath, self.stats.get_dict())
        display("Saved session stats")
    
    @timeit('Loaded all data')
    def load_session(self):
        """ Loads the saved session data """
        meta = load_json_dict(self.data_paths.metaPath)
        self.meta = SessionMeta(init_dict=meta)
        display('Loaded session metadata')
        
        self.data.load_data()
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
        session_data = session_data.rename({"stim_side":"stim_pos"})
        
        # add a stim_side column for ease of access
        session_data = session_data.with_columns(pl.when(pl.col("stim_pos") > 0).then("contra")
                                                    .when(pl.col("stim_pos") < 0).then("ipsi")
                                                    .when(pl.col("stim_pos") == 0).then("catch")
                                                    .otherwise(None).alias("stim_side"))

        if session_data.is_empty():
            print('''WARNING THIS SESSION HAS NO DATA
            Possible causes:
            - Session has only one trial with no correct answer''')
            return None
        else:
            return session_data


@timeit('Getting rolling averages...')
def get_running_stats(data_in:pd.DataFrame,window_size:int=20) -> pd.DataFrame:
    """ Gets the running statistics of certain columns"""
   
    data_in = data_in.with_columns(pl.col('response_latency').rolling_median(window_size).alias('running_response_latency'))
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