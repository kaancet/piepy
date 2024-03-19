import scipy.io as sio
from os.path import join as pjoin
from collections import namedtuple
from os.path import exists as exists

from ..utils import *
from .exceptions import *
from .logger import Logger
from .pathfinder import PathFinder


class RunMeta:
    def __init__(self, prot_file:str) -> None:
        self.prot_file = prot_file
        self.init_from_prot()
        
    def init_from_prot(self) -> None:
        """ Initializes the run meta object from the protfile"""
        self.run_name = self.prot_file.split(os.sep)[-1].split('.')[0]
        ignore = ['picsFolder', 'picsNameFormat', 'shuffle', 'mask', 'nTrials',
                  'progressWindow', 'debiasingWindow', 'decimationRatio']
        self.opto = False
        self.opts,self.params = parseProtocolFile(self.prot_file)
        # put all of the options into meta attributes
        for k, v in self.opts.items():
            if k not in ignore:
                try:
                    v = float(v.strip(' '))
                except:
                    pass
                if k == 'controller':
                    if 'Opto' in v:
                        self.opto = True
                elif k == 'contrastVector':
                    v = [float(i) for i in v.strip('] [').strip(' ').split(',')]
                setattr(self, k, v)

        if self.opto:
            self.opto_mode = int(self.opts.get('optoMode',0)) #0 continuous, 1 pulsed

        lvl = ''
        if self.prot_file.find('level') != -1:
            tmp = self.prot_file[self.prot_file.find('level')+len('level'):]
            for char in tmp:
                if char not in ['.','_']:
                    lvl += char
                else:
                    break      
        else:
            lvl = 'exp'
        self.level = lvl
        
        os_stat = os.stat(self.prot_file)
        if sys.platform == 'darwin':
            create_epoch = os_stat.st_birthtime
        elif sys.platform == 'win32':
            create_epoch = os_stat.st_ctime
        self.run_time = dt.fromtimestamp(create_epoch).strftime('%H%M')


class RunData:
    def __init__(self, data:pl.DataFrame=None) -> None:
        self.set_data(data)
    
    def set_data(self,data:pl.DataFrame) -> None:
        self.data = data
        
    def save_data(self,save_path:str,save_mat:bool=False) -> None:
        """ Saves the run data as .parquet (and .mat file if desired)"""
        data_save_path = pjoin(save_path,'runData.parquet')
        self.data.write_parquet(data_save_path)
        if save_mat:
            self.save_as_mat(save_path)
            display(f'Saved .mat file at {save_path}',color='green')
            
    def load_data(self, load_path:str) -> pd.DataFrame:
        """Loads the data from J:/analysis/<exp_folder> as a pandas data frame"""
        # data = pd.read_csv(self.paths.data)
        data = pl.read_parquet(load_path)
        self.set_data(data)    
            
    def save_as_mat(self,save_path:str) -> None:
        """Helper method to convert the data into a .mat file"""
        datafile = pjoin(save_path,'sessionData.mat')
        
        save_dict = {name: col.values for name, col in self.data.stim_data.items()}
        sio.savemat(datafile, save_dict)
        display(f'Saved .mat file at {datafile}')


class Run:
    def __init__(self,
                 run_no:int,
                 _path:PathFinder) -> None:
        self.data = None
        self.run_no = run_no
        self.init_run_paths(_path)
                # initialize the logger(only log at one analysis location, currently arbitrary)
        self.logger = Logger(log_path=self.paths.save[0])
    
    def init_run_paths(self,path_finder:PathFinder) -> None:
        """ Sets the paths related to the run"""
        tmp_dict = {name:(path[self.run_no] if isinstance(path,list) 
                    else path) for name,path in path_finder.all_paths.items()}
        tmp_paths = namedtuple("Paths", list(tmp_dict.keys()))
        self.paths = tmp_paths(**tmp_dict)
        
        # create save paths
        for s_path in self.paths.save:
            if not exists(s_path):
                os.makedirs(s_path)
        
    def init_run_meta(self):
        """ Initializes the metadata for the run """
        self.meta = RunMeta(self.paths.prot)
    
    @staticmethod
    def read_combine_logs(stimlog_path:str|list, riglog_path:str|list) -> tuple[pl.DataFrame, list]:
        """ Reads the logs and combines them if multiple logs of same type exist in the run directory"""
        if isinstance(stimlog_path,list) and isinstance(riglog_path,list):
            assert len(stimlog_path) == len(riglog_path), f'The number stimlog files need to be equal to amount of riglog files {len(stimlog_path)}=/={len(riglog_path)}'
                
            stim_data_all = []
            rig_data_all = []
            stim_comments = []
            rig_comments = []
            for i,s_log in enumerate(stimlog_path):
                try:
                    temp_slog,temp_scomm = parseStimpyLog(s_log)
                except:
                    # probably not the right stimpy version, try github
                    temp_slog, temp_scomm = parseStimpyGithubLog(s_log)
                temp_rlog,temp_rcomm = parseStimpyLog(riglog_path[i])
                stim_data_all.append(temp_slog)
                rig_data_all.append(temp_rlog)
                stim_comments.extend(temp_scomm)
                rig_comments.extend(temp_rcomm)
                
            stim_data = stitchLogs(stim_data_all,isStimlog=True) # stimlog
            rig_data = stitchLogs(rig_data_all,isStimlog=False)  # riglog
        else:
            try:
                stim_data, stim_comments = parseStimpyLog(stimlog_path)
            except:
                stim_data, stim_comments = parseStimpyGithubLog(stimlog_path)
            rig_data, rig_comments = parseStimpyLog(riglog_path)
        
        rawdata = {**stim_data, **rig_data}
        comments = {'stimlog':stim_comments,
                    'riglog':rig_comments}
        return rawdata, comments
    
    def read_run_data(self) -> None:
        """ Reads the data from concatanated riglog and stimlog files, and if exists from camlog files"""
        # stimlog and camlog
        rawdata, self.comments = self.read_combine_logs(self.paths.stimlog,self.paths.riglog)
        self.rawdata = extrapolate_time(rawdata)
        
        # onep or two depending on imaging mode
        if self.meta.imaging_mode == '1P':
            try:
                if os.path.exists(self.paths.onepcamlog):
                    self.rawdata['onepcam_log'],self.comments['onepcam'],_ = parseCamLog(self.paths.onepcamlog)
            except:
                display("\n No 1P cam data for 1P experiment! IS THIS EXPECTED?! \n", color='yellow')
        elif self.meta.imaging_mode == '2P':
            pass
    
        # try eyecam and facecam either way
        if self.paths.eyecam is not None and os.path.exists(self.paths.eyecamlog):
            self.rawdata['eyecam_log'],self.comments['eyecam'],_ = parseCamLog(self.paths.eyecamlog)
        
        if self.paths.facecam is not None and os.path.exists(self.paths.facecamlog):
            self.rawdata['facecam_log'],self.comments['facecam'],_ = parseCamLog(self.paths.facecamlog)            
                
        display('Read rawdata')
        
    def check_and_translate_state_data(self) -> bool:
        """ Checks if state data exists and translated the state transitions according to defined translation dictionary
        This function needs the translate transition to be defined beforehand """
        if self.rawdata['statemachine'].is_empty():
            self.logger.critical("NO STATE MACHINE TO ANALYZE. LOGGING PROBLEMATIC. SOLVE THIS ISSUE FAST!!",cml=True)
            return False
        
        # do the translation
        try:
            self.rawdata['statemachine'] = self.rawdata['statemachine'].with_columns(pl.struct(['oldState','newState']).apply(lambda x: self.translate_transition(x['oldState'],x['newState'])).alias('transition'))
        except:
            raise WrongSessionTypeError(f'Unable to translate state changes to valid transitions. Make sure you are using the correct session type to analyze your data!')
        # rename cycle to 'trialNo for semantic reasons
        self.rawdata['statemachine'] = self.rawdata['statemachine'].rename({"cycle":"trialNo"})
        
        return True
        
    def is_run_saved(self) -> bool:
        """ Initializes the necessary save paths and checks if data already exists"""
        loadable = False
        for d_path in self.paths.data:
            if exists(d_path):
                loadable = True
                display(f'Found saved data: {d_path}',color='cyan')
                break
            else:
                display(f"{d_path} does not exist...",color='yellow')
        return loadable
    
    def save_run(self,save_mat:bool=False) -> None:
        """ Saves the run data, meta and stats"""
        if self.data is not None:
            for s_path in self.paths.save:
                self.data.save_data(s_path,save_mat)
                display(f"Saved session data to {s_path}", color='green')
            
    def load_run(self) -> None:
        """ Loads the saved """
        for d_path in self.paths.data:
            if exists(d_path):
                self.data.load_data(d_path)
                display(f"Loaded session data from {d_path}",color='green')
                break
            
    def extract_trial_count(self,num_state_changes:int):
        """ Extracts the trial no from state changes, this works for stimpy for now"""
        display('Trial increment faulty, extracting from state changes...')
        
        trial_cnt = int(len(self.rawdata['statemachine'])/num_state_changes)
        trial_no = np.repeat(np.arange(1,trial_cnt+1),num_state_changes)
        
        new_trial_no = pl.Series('trialNo',trial_no)
        self.rawdata['statemachine'] = self.rawdata['statemachine'].with_columns(new_trial_no)
            
    def compare_cam_logging(self) -> None:
        """ Compares the camlogs with corresponding riglog recordings """
        # !! IMPORTANT !!
        # rawdata keys that end only with 'cam' are from the rig
        # the ones that end with 'cam_log' are from labcams
        rig_cams = [k for k in self.rawdata.keys() if k.endswith('cam')]
        labcam_cams = [k for k in self.rawdata.keys() if k.endswith('cam_log')]
        
        assert len(rig_cams) == len(labcam_cams), f"Number of camlogs in rig({len(rig_cams)}) and labcams({len(labcam_cams)}) are not equal!! "
        
        for i,lab_cam_key in enumerate(labcam_cams):
            rig_cam_frames = len(self.rawdata[rig_cams[i]])
            labcams_frames = len(self.rawdata[lab_cam_key])
            
            if labcams_frames < rig_cam_frames:
                display(f'{rig_cam_frames - labcams_frames} logged frame(s) not recorded by {lab_cam_key}!!',color='yellow')
            elif labcams_frames > rig_cam_frames:
                # remove labcams camlog frames if they are more than riglog recordings
                display(f'{labcams_frames - rig_cam_frames} logged frame(s) not logged in {rig_cams[i]}!',color='yellow')
                
                self.rawdata[lab_cam_key] = self.rawdata[lab_cam_key].slice(0,rig_cam_frames)     # removing extra recorded frames
                if len(self.rawdata[lab_cam_key])==rig_cam_frames:
                    display(f'Camlogs are equal now!',color='cyan')
                    