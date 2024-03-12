from os.path import exists as exists

from ..utils import *
from .run import Run, RunMeta
from .pathfinder import PathFinder
from ..gsheet_functions import GSheet
from .dbinterface import DataBaseInterface


class SessionMeta:
    """ An object to hold Session meta data"""
    def __init__(self, sessiondir:str, skip_google:bool=False) -> None:
        """ Initializes the meta either from a prot file"""
        self.set_attrs_from_sessiondir(sessiondir)
        self.set_session_weight_and_water(skip_google=skip_google)

    def __repr__(self):
        kws = [f'{key}={value!r}' for key, value in self.__dict__.items() if key != 'init_dict']
        return '{}\n{}'.format(type(self).__name__, ',\n'.join(kws))
            
    def set_attrs_from_sessiondir(self,sessiondir:str) -> None:
        """ Parses the exp names from the dir path of experiment"""
        tmp = sessiondir.split('_')
        # date
        date_str = tmp[0]
        self.set_session_date(date_str)
        
        #animalid
        self.animalid = tmp[1]
        
        #imagingmode
        self.imaging_mode = tmp[-2]
        if self.imaging_mode == 'cam':
            # because "_no_cam also gets parsed here..."
            self.imaging_mode == None
        elif self.imaging_mode not in ['1P','2P']:
            raise ValueError(f'Parsed {self.imaging_mode} as imaging mode, this is not possible, check the session_name!!')
        #userid
        self.user = tmp[-1]

    def set_session_rig(self,pref_file:str) -> None:
        """ Sets the rig from pref file """
        prefs = parsePref(pref_file)
        if 'name' in prefs['rig']:
            self.rig = prefs['rig']['name']
        else:
            self.rig = prefs['tmpFolder'].split(os.sep)[-1]

    def set_session_date(self,date_str:str) -> None:
        """ Sets various types of date formating in addition to datetime """
        self.baredate = date_str
        self.date = dt.strptime(self.baredate, '%y%m%d').date()
        self.nicedate = dt.strftime(self.date, '%d %b %y') 
        
    def set_session_weight_and_water(self,skip_google:bool=False) -> None:
        """ Gets the session weight from google sheet"""
        self.weight = None
        self.water_consumed = None
        if not skip_google:
            logsheet = GSheet('Mouse Database_new')
            gsheet_df = logsheet.read_sheet(2)
            gsheet_df = gsheet_df[(gsheet_df['Mouse ID'] == self.animalid) & (gsheet_df['Date [YYMMDD]'] == int(self.baredate))]
            if not gsheet_df.empty:
                gsheet_df.reset_index(inplace=True)
                self.weight = gsheet_df['weight [g]'].iloc[0]
                try:
                    self.water_consumed = int(gsheet_df['rig water [µl]'].iloc[0])
                except:
                    self.water_consumed = None
      
    # DATABASE RELATED UNUSED FOR NOW   
    # def generate_session_id(self,sessiondir) -> None:
    #     """ Generates a unique session id for the session """
    #     try:    
    #         mouse_part = ''.join([n for n in self.animalid if n.isdigit()])
    #         self.session_id = self.baredate + mouse_part
    #     except:
    #         raise RuntimeError(f'Failed to create session id for {sessiondir}')


class Session:
    """ A base Session object, reads and aggregates the recorded data which can then be used in user specific 
        analysis pipelines
        :param sessiondir: directory of the session inside the presentation folder(e.g. 200619_KC033_wheel_KC)
        :param load_flag:  flag to either load previously parsed data or to parse it again
        :param save_mat:   flag to make the parser also output a .mat file to be used in MATLAB  scripts
        :type sessiondir:  str
        :type load_flag:   bool
        :type save_mat:    bool
    """
    def __init__(self, 
                 sessiondir, 
                 load_flag=False, 
                 save_mat=False):
        self.sessiondir = sessiondir
        # an empty dictionary that can be populated with session related data
        # every Session object has a meta, session_data and stat attribute
        self.load_flag = load_flag
        self.save_mat = save_mat
        self.runs = []
        
        # find relevant data paths
        self.paths = PathFinder(self.sessiondir)
             
    def set_session_meta(self,skip_google:bool=False):
        """ Sets the metadata from session name, pref and prot files,
             to be overwritten by other Session types(e.g. WheelDetectionSession) """
        self.meta = SessionMeta(self.sessiondir,skip_google)
        self.meta.set_session_rig(self.paths.all_paths['prefs'][0])
                
    def init_session_runs(self) -> None:
        """ Initializes runs in a session, 
            to be overwritten by other Session types(e.g. WheelDetectionSession)"""
        self.run_count = len(self.paths.stimlog)
        for r in range(self.run_count):
            self.runs.append(Run(r,self.paths))
                
    @timeit('Saving...')
    def save_session(self) -> None:
        """ Saves the session data, meta and stats"""
        for run in self.runs:
            run.save_run(self.save_mat)
            
    @timeit('Loading...')
    def load_session(self) -> None:
        """ Helper method to loop through the runs and load data and stats """
        for run in self.runs:
            run.load_run()

    def get_meta(self):
        if len(self.runs) == 1:
            meta = self.runs[0].meta
            for k,v in self.meta.__dict__.items():
                setattr(meta,k,v)
            return meta
        else:
            return self.meta
        
    @property
    def data(self):
        if len(self.runs) == 1:
            return self.runs[0].data
        else:
            raise ValueError(f"Session has {len(self.runs)} runs, can't get a single session data :(")
    
    @property
    def stats(self):
        if len(self.runs) == 1:
            return self.runs[0].stats
        else:
            raise ValueError(f"Session has {len(self.runs)} runs, can't get a single session stats :(")
    
    ####
    # DATABASE RELATED, NOT USED AT THE MOMENT
    ###
            
    def save_to_db(self,db_dict:dict) -> None:
        """ Checks if an entry for the session already exists and saves/updates accordingly"""
        if not self.db_interface.exists({'sessionId':self.meta.session_id},'sessions'):
            self.db_interface.add_entry(db_dict,'sessions')
            self.db_interface.update_entry({'id':self.meta.animalid},{'nSessions':self.current_session_no},'animals')
        else:
            self.db_interface.update_entry({'sessionId':self.meta.session_id},db_dict,'sessions')
            display(f'Session with id {self.meta.session_id} is already in database, updated the entry')
        
    def get_latest_trial_count(self):
        """ Gets the last trial count from """
        prev_trials = self.db_interface.get_entries({'id':self.meta.animalid},'trials')
        try:
            return int(prev_trials['total_trial_no'].iloc[-1])
        except:
            return 0    
    
    def overall_session_no(self) -> int:
        """ Gets the session number of the session"""
        mouse_entry = self.db_interface.get_entries({'id':self.meta.animalid},table_name='animals')
        if len(mouse_entry):
            last_session_no = mouse_entry['nSessions'].iloc[0]
        else:
            display(f'No entry for mouse {self.meta.animalid} in animals table!')
            last_session_no = 0
        
        current_session_no = last_session_no + 1
        return current_session_no
    
    def remove_session_db(self):
        """ """
        pass
