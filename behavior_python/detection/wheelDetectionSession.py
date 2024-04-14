import time
from PIL import Image
import tifffile as tf
import scipy.stats as st
import multiprocessing
from multiprocessing import Pool
from tabulate import tabulate
from os.path import join as pjoin
from tqdm.contrib.concurrent import process_map


from ..core.session import Session
from ..core.run import RunData, Run, RunMeta
from ..core.pathfinder import *
from .wheelDetectionTrial import *


class WheelDetectionRunMeta(RunMeta):
    def __init__(self, prot_file: str) -> None:
        super().__init__(prot_file)
        
        self.sf_values = nonan_unique(self.params['sf'].to_numpy()).tolist()
        self.tf_values = nonan_unique(self.params['tf'].to_numpy()).tolist()


class WheelDetectionRunData(RunData):
    def __init__(self,data:pl.DataFrame=None) -> None:
        super().__init__(data)
        
    def set_data(self,data:pl.DataFrame) -> None:
        """ Sets the data of the session and augments it"""
        if data is not None:
            super().set_data(data)
            self.add_qolumns()
            
    def set_outcome(self,outcome_type:str='state') -> None:
        """ Sets the outcome column to the selected column"""
        display(f'Setting outcome to {outcome_type}')
        col_name = f'{outcome_type}_outcome'
        if 'outcome' not in self.data.columns:
            self.data = self.data.with_columns(pl.col(col_name).alias('outcome'))
        else:
            if col_name in self.data.columns:
                tmp = self.data[col_name]
                self.data.replace('outcome', tmp)
            else:
                raise ValueError(f"{outcome_type} is not a valid outcome type!!! Try 'pos', 'speed' or 'state'.")
            
    def compare_outcomes(self) -> None:
        """ Compares the different outcome types and prints a small summary table"""
        out_cols = [c for c in self.data.columns if '_outcome' in c]
        q = self.data.groupby(out_cols).agg([pl.count().alias("count")]).sort(["state_outcome"])
        tmp = q.to_pandas()
        print(tabulate(tmp,headers=q.columns))
        
    def add_qolumns(self) -> None:
        """ Adds some quality of life (qol) columns """
        # add a stim_side column for ease of access
        self.data = self.data.with_columns(pl.when(pl.col("stim_pos") > 0).then(pl.lit("contra"))
                                           .when(pl.col("stim_pos") < 0).then(pl.lit("ipsi"))
                                           .when(pl.col("stim_pos") == 0).then(pl.lit("catch"))
                                           .otherwise(None).alias("stim_side"))
        
        # round sf and tf
        self.data = self.data.with_columns([(pl.col('spatial_freq').round(2).alias('spatial_freq')),
                                      (pl.col('temporal_freq').round(1).alias('temporal_freq'))])
        
        #adds string stimtype
        self.data = self.data.with_columns((pl.col('spatial_freq').round(2).cast(str) + 'cpd_' + 
                                            pl.col('temporal_freq').cast(str) + 'Hz').alias('stim_type'))
        
        # add signed contrast
        self.data = self.data.with_columns(pl.when(pl.col("stim_side")=="ipsi")
                                           .then((pl.col("contrast")*-1))
                                           .otherwise(pl.col("contrast")).alias("signed_contrast"))
        
        # add easy/hard contrast type groups
        self.data = self.data.with_columns(pl.when(pl.col('contrast') >= 25).then(pl.lit("easy"))
                                           .when((pl.col('contrast') < 25) & (pl.col('contrast') > 0)).then(pl.lit("hard"))
                                           .when(pl.col('contrast') == 0).then(pl.lit("catch"))
                                           .otherwise(None).alias("contrast_type"))
        
    def add_pattern_related_columns(self) -> None:
        """ Adds columns related to the silencing pattern if they exist """
        if len(self.data['opto'].unique()) == 1:
            # Regular training sessions
            # add the pattern name depending on pattern id
            self.data = self.data.with_columns(pl.lit(None).alias('opto_region'))
            # add 'stimkey' from sftf
            self.data = self.data.with_columns((pl.col("stim_type")+'_-1').alias('stimkey'))
            # add stim_label for legends and stuff
            self.data = self.data.with_columns((pl.col("stim_type")).alias('stim_label'))
        else:
            if isinstance(self.pattern_names,dict):
                try:
                # add the pattern name depending on pattern id
                    self.data = self.data.with_columns(pl.struct(["opto_pattern", "state_outcome"])
                                                       .apply(lambda x: self.pattern_names[x['opto_pattern']] if x['state_outcome']!=-1 else None).alias('opto_region'))        
                except:
                    raise KeyError(f'Opto pattern not set correctly. You need to change the number at the end of the opto pattern image file to an integer (0,-1,1,..)!')
            elif isinstance(self.pattern_names,str):
                display(f"{self.paths.opto_pattern} NO OPTO PATTERN DIRECTORY!!")
                self.data = self.data.with_columns(pl.struct(["opto_pattern", "state_outcome"])
                                                   .apply(lambda x: self.pattern_names if x['state_outcome']!=-1 else None).alias('opto_region'))
            else:
                raise ValueError(f"Weird pattern name: {self.pattern_names}")
            
            # add 'stimkey' from sftf
            self.data = self.data.with_columns((pl.col("stim_type")+'_'+pl.col("opto_pattern").cast(pl.Int8,strict=False).cast(str)).alias('stimkey'))
            # add stim_label for legends and stuff
            self.data = self.data.with_columns((pl.col("stim_type")+'_'+pl.col("opto_region")).alias('stim_label'))

    def get_opto_images(self,get_from:str) -> None:
        """ Reads the related run images(window, pattern, etc)
            Returns a dict with images and also a dict that """
        if get_from is not None and os.path.exists(get_from):
            self.sesh_imgs = {}
            self.pattern_names = {}
            self.sesh_patterns = {}
            for im in os.listdir(get_from):
                if im.endswith('.tif'):
                    pattern_id = int(im[:-4].split('_')[-1])
                    read_img = tf.imread(pjoin(get_from,im))
                    if pattern_id == -1:
                        self.sesh_imgs['window'] = read_img
                        self.pattern_names[pattern_id] = 'nonopto'
                    else:  
                        name = im[:-4].split('_')[-2]
                        self.pattern_names[pattern_id] = name
                        self.sesh_imgs[name] = read_img
                elif im.endswith('.bmp'):
                    pattern_id = int(im.split('_')[0])
                    name = im.split('_')[1]
                    read_bmp = np.array(Image.open(pjoin(get_from,im)))
                    self.sesh_patterns[name] = read_bmp[::-1,::-1]
        else:
            self.sesh_imgs = None
            self.sesh_patterns = None
            self.pattern_names = None# '*'+self.paths.pattern.split('_')[4]+'*'


class WheelDetectionStats:
    __slots__ = ['all_count','early_count','stim_count','correct_count','miss_count',
                 'all_correct_percent','hit_rate','easy_hit_rate','false_alarm','nogo_percent',
                 'median_response_time','d_prime']
    def __init__(self,dict_in:dict=None,data_in:WheelDetectionRunData=None) -> None:
        if data_in is not None:
            self.init_from_data(data_in)
        elif dict_in is not None:
            self.init_from_dict(dict_in)

    def __repr__(self):
        rep = ''''''
        for k in self.__slots__:
            rep += f'''{k} = {getattr(self,k,None)}\n'''
        return rep
    
    def init_from_data(self,data_in:WheelDetectionRunData):
        data = data_in.data
        early_data = data.filter((pl.col('outcome')==-1) & (pl.col('isCatch')==0))
        stim_data = data.filter((pl.col('outcome')!=-1) & (pl.col('isCatch')==0))
        catch_data = data.filter(pl.col('isCatch')==1)
        correct_data = data.filter(pl.col('outcome')==1)
        miss_data = data.filter(pl.col('outcome')==0)
        
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
        easy_correct_count = len(easy_data.filter(pl.col('outcome')==1))
        if len(easy_data):
            self.easy_hit_rate = round(100 * easy_correct_count / len(easy_data),3)
        else:
            self.easy_hit_rate = 0
        
        # median response time
        self.median_response_time = round(stim_data.filter(pl.col('outcome')==1)['response_latency'].median(),3)
        
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


class WheelDetectionRun(Run):
    def __init__(self, run_no: int, _path: PathFinder, **kwargs) -> None:
        super().__init__(run_no, _path)
        self.trial_list = []
        self.data = WheelDetectionRunData()
        
    def analyze_run(self) -> None:
        """ """
        self.read_run_data()
        run_data = self.get_all_trials()
        
        # set the data object
        self.data.set_data(run_data)
        
        # get related images(silencing patterns etc)
        self.data.get_opto_images(self.paths.opto_pattern)
        self.data.add_pattern_related_columns()
        
        self.data.set_outcome('state')
        self.stats = WheelDetectionStats(data_in=self.data)
        
    def init_run_meta(self) -> None:
        """ Initializes the metadata for the run """
        self.meta = WheelDetectionRunMeta(self.paths.prot)
        
    def get_all_trials(self) -> pl.DataFrame | None:
        """ """
        if not self.check_and_translate_state_data():
            return None
        
        trial_nos = np.unique(self.rawdata['statemachine']['trialNo'])
        trial_nos = [int(t) for t in trial_nos]     
        
        if cfg.multiprocess['enable']:
            mngr = multiprocessing.Manager()
            self.queue = mngr.Queue(-1)
            self._list_data = mngr.list()
            
            listener = multiprocessing.Process(
                target=self.logger.listener_process, args=(self.queue,)
            )
            listener.start()
            
            process_map(self._get_trial, trial_nos, max_workers=cfg.multiprocess['cores'],chunksize=100)
            
        else:
            self.logger.listener_configurer()
            self._list_data = []
            pbar = tqdm(trial_nos,desc='Extracting trial data:',leave=True,position=0)
            for t in pbar:
                self._get_trial(t)
                
        # convert list of dicts to dict of lists
        data_to_frame = {k:[v] for k,v in self._list_data[0].items()}
        for i in range(1,len(self._list_data)):
            for k,v in self._list_data[i].items():
                data_to_frame[k].append(v)
        
        r_data = pl.DataFrame(data_to_frame)
        # order by trial no
        r_data = r_data.sort('trial_no')

        # add contrast titration boolean
        uniq_stims = nonan_unique(r_data['contrast'].to_numpy())
        isTitrated = 0
        if len(uniq_stims) > len(self.meta.contrastVector):
            isTitrated = 1
        r_data = r_data.with_columns([pl.lit(isTitrated).cast(pl.Boolean).alias('isTitrated')])

        if r_data.is_empty():
            self.logger.error("THERE IS NO SESSION DATA !!!", cml=True)
            return None
        else:
            return r_data

    def _get_trial(self,trial_no:int) -> pl.DataFrame | None:
        """ Main loop that parses the rawdata into a polars dataframe where each row corresponds to a trial """
        # self.logger.attach_to_queue(self.log_queue)
        if cfg.multiprocess['enable']:
            self.logger.worker_configurer(self.queue)
        
        temp_trial = WheelDetectionTrial(trial_no = trial_no,
                                         meta = self.meta,
                                         logger = self.logger)
        # get the data slice using state changes
        temp_trial.get_data_slices(self.rawdata,use_state=True)
        _trial_data = temp_trial.trial_data_from_logs()
        
        if _trial_data['state_outcome'] is not None:
            self._list_data.append(_trial_data)
    
    def translate_transition(self,oldState,newState) -> dict:
        """ 
        A function to be called that add the meaning of state transitions into the state DataFrame
        """
        curr_key = '{0}->{1}'.format(int(oldState), int(newState))
        state_keys = {'0->1' : 'trialstart',
                      '1->2' : 'cuestart',
                      '2->3' : 'stimstart',
                      '2->5' : 'early',
                      '3->4' : 'hit',
                      '3->5' : 'miss',
                      '3->6' : 'catch',
                      '6->0' : 'trialend',
                      '4->6' : 'stimendcorrect',
                      '5->6' : 'stimendincorrect'}
        
        return state_keys[curr_key]

    def save_run(self) -> None:
        """ Saves the run data, meta and stats """
        super().save_run()

        for s_path in self.paths.save:
            save_dict_json(pjoin(s_path,'sessionStats.json'), self.stats.get_dict())
            self.logger.info(f"Saved session stats to {s_path}")
        
    def load_run(self) -> None:
        """ Loads the run data and stats if exists """
        super().load_run()
        
        for s_path in self.paths.save:
            stat_path = pjoin(s_path,'sessionStats.json')
            if os.path.exists(stat_path):
                stats = load_json_dict(stat_path)
                self.stats = WheelDetectionStats(dict_in=stats)
                self.logger.info(f'Loaded Session stats from {s_path}')
                break
        
    
class WheelDetectionSession(Session):
    def __init__(self, sessiondir,load_flag:bool,save_mat:bool=False,**kwargs):
        start = time.time()
        super().__init__(sessiondir, load_flag, save_mat)
        
        # sets session meta
        self.set_session_meta(skip_google=kwargs.get('skip_google',False))
        
        # initialize runs : read and parse or load the data 
        self.init_session_runs()
        
        end = time.time()
        display(f'Done! t={(end-start):.2f} s')
        
    def __repr__(self):
        r = f'Detection Session {self.sessiondir}'
        return r
    
    def init_session_runs(self) -> None:
        """ Initializes runs in a session """
        self.runs = []
        self.run_count = len(self.paths.all_paths['stimlog'])
        for r in range(self.run_count):
            run = WheelDetectionRun(r,self.paths)
            run.init_run_meta()
            # transferring some session metadata to run metadata
            run.meta.imaging_mode = self.meta.imaging_mode
            if run.is_run_saved() and self.load_flag:
                display(f'Loading from {run.paths.save}')
                run.load_run()
                # initially sets the outcome to be the outcome from the python state machine
                run.data.set_outcome('state')
            else:
                run.analyze_run()
        
                # add some metadata to run datas for ease of manipulation
                run.data.data = run.data.data.with_columns([pl.lit(self.meta.animalid).alias('animalid'),
                                                            pl.lit(self.meta.baredate).alias('baredate')])
                run.data.data = run.data.data.with_columns(pl.col('baredate').str.strptime(pl.Date, format='%y%m%d').cast(pl.Date).alias('date'))

                run.save_run()
            
            self.runs.append(run)
        

@timeit('Getting rolling averages...')
def get_running_stats(data_in:pd.DataFrame,window_size:int=20) -> pd.DataFrame:
    """ Gets the running statistics of certain columns"""
   
    data_in = data_in.with_columns(pl.col('response_latency').rolling_median(window_size).alias('running_response_latency'))
    # answers
    outcomes = {'correct':1,
                'nogo':0,
                'early':-1}
    
    for k,v in outcomes.items():
        key = 'fraction_' + k
        data_arr = data_in['state_outcome'].to_numpy()
        data_in[key] = get_fraction(data_arr, fraction_of=v)
        
    return data_in

def main():
    from argparse import ArgumentParser
    import cProfile, pstats
    from io import StringIO
    parser = ArgumentParser(description='Wheel Detection Session Analysis')

    parser.add_argument('expname',metavar='expname',
                        type=str,help='Experiment filename (e.g. 200325_KC020_wheel_KC)')
    parser.add_argument('-l','--load',metavar='load_flag',default=True,
                        type=str,help='Flag for loading existing data')

    opts = parser.parse_args()
    expname = opts.expname
    load_flag = opts.load
    
    profiler = cProfile.Profile()
    profiler.enable()
    print(load_flag)
    w = WheelDetectionSession(sessiondir=expname, load_flag=False)
    
    profiler.disable()
    s = StringIO
    stats = pstats.Stats(profiler,stream=s)
    stats.strip_dirs()
    stats.sort_stats('cumtime')
    stats.dump_stats(w.data_paths.analysisPath+os.sep+'profile.prof')
    # stats.print_stats()

if __name__ == '__main__':
    main()