from ..core.session import Session
from .visualTrial import *
from ..core.run import RunData, Run, RunMeta
from ..core.pathfinder import PathFinder


class VisualRunMeta(RunMeta):
    def __init__(self, prot_file: str) -> None:
        super().__init__(prot_file)
        

class VisualRunData(RunData):
    def __init__(self, data: pl.DataFrame = None) -> None:
        super().__init__(data)
        

class VisualRun(Run):
    def __init__(self, run_no: int, _path: PathFinder) -> None:
        super().__init__(run_no, _path)
        self.trial_list = []
        self.data = VisualRunData()
        
    def read_run_data(self) -> None:
        """ Add experiment specific modifiactins to read data in this method"""
        super().read_run_data()
        
        # sometimes iTrial starts from 0, shift all to start from 1
        self.rawdata['vstim'] = self.rawdata['vstim'].with_columns((pl.col('iTrial')+1).alias('iTrial'))
        
    def analyze_run(self) -> None:
        """ """
        self.read_run_data()
        run_data = self.get_run_trials_from_data()
        
        # set the data object
        self.data.set_data(run_data)
        
    def init_run_meta(self) -> None:
        """ Initializes the metadata for the run """
        self.meta = VisualRunMeta(self.paths.prot)
    
    def get_run_trials_from_data(self) -> pl.DataFrame:
        """ Main loop that parses the rawdata into a polars dataframe where each row corresponds to a trial """
        data_to_append = []
        
        if not self.check_and_translate_state_data():
            return None
        
        trials = np.unique(self.rawdata['statemachine']['trialNo'])
        if len(trials) == 1:
            self.extract_trial_count(num_state_changes=4)
            trials = self.rawdata['statemachine']['trialNo'].unique().to_list()
        pbar = tqdm(trials,desc='Extracting trial data:',leave=True,position=0)
        for t in pbar:
            # instantiate a trial      
            temp_trial = VisualTrial(trial_no = int(t),
                                     meta = self.meta,
                                     logger = self.logger)
            # get the data slice using state changes
            temp_trial.get_data_slices(self.rawdata,use_state=False)
            trial_row = temp_trial.trial_data_from_logs()
            
            self.trial_list.append(temp_trial)
            if t == 1:
                data_to_append = {k:[v] for k,v in trial_row.items()}
            else:
                for k,v in trial_row.items():
                    data_to_append[k].append(v)
            
            pbar.update()
            
        self.logger.set_msg_prefix('session')
        r_data = pl.DataFrame(data_to_append)
        
        if r_data.is_empty():
            self.logger.error("THERE IS NO SESSION DATA !!!", cml=True)
            return None
        else:
            return r_data
             
    def translate_transition(self,oldState,newState) -> dict:
        """ A function to be called that add the meaning of state transitions into the state DataFrame """
        curr_key = f'{int(oldState)}->{int(newState)}'
        state_keys = {'0->1' : 'trialstart',
                      '1->2' : 'stimstart',
                      '2->0' : 'stimtrialend',
                      '2->3' : 'stimend',
                      '3->0' : 'trialend'}
        
        return state_keys[curr_key]
    
    def extract_trial_count(self,num_state_changes:int):
        """ Extracts the trial no from state changes, this works for stimpy for now"""
        display('Trial increment faulty, extracting from state changes...',color='yellow')
        
        trialends = self.rawdata['statemachine'].with_columns(pl.when(pl.col('transition').str.contains('trialend')).then(1).otherwise(0).alias('end_flag'))
        
        trial_no = []
        t_cntr = 1
        for i in trialends['end_flag'].to_list():
            trial_no.append(t_cntr)
            if i:
                t_cntr += 1
                      
        new_trial_no = pl.Series('trialNo',trial_no)
        self.rawdata['statemachine'] = self.rawdata['statemachine'].with_columns(new_trial_no)

class VisualSession(Session):
    def __init__(self, sessiondir, load_flag=False, save_mat=False):
        start = time.time()
        super().__init__(sessiondir, load_flag, save_mat)
        
        # sets session meta
        self.set_session_meta(skip_google=True)
        
        # initialize runs : read and parse or load the data 
        self.init_session_runs()
        
        end = time.time()
        display(f'Done! t={(end-start):.2f} s')
        
    def init_session_runs(self) -> None:
        """ Initializes runs in a session """
        self.runs = []
        self.run_count = len(self.paths.all_paths['stimlog'])
        for r in range(self.run_count):
            run = VisualRun(r,self.paths)
            run.init_run_meta()
            # transferring some session metadata to run metadata
            run.meta.imaging_mode = self.meta.imaging_mode
            if run.is_run_saved() and self.load_flag:
                display(f'Loading from {run.paths.save}')
                run.load_run()
            else:
                run.analyze_run()
                # add some metadata to run datas for ease of manipulation
                run.data.data = run.data.data.with_columns([pl.lit(self.meta.animalid).alias('animalid'),
                                                            pl.lit(self.meta.baredate).alias('baredate')])
                run.data.data = run.data.data.with_columns(pl.col('baredate').str.strptime(pl.Date, format='%y%m%d').cast(pl.Date).alias('date'))

                run.save_run()
            
            self.runs.append(run)
        
    
        