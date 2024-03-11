from .logger import *
import glob
import natsort
import scipy.io as sio


from ..gsheet_functions import GSheet
from .dbinterface import DataBaseInterface

class BehaviorData:
    """ Keeps the data as a huge csv of appended session data"""
    __slots__ = ['summary_data','cumul_data']
    def __init__(self) -> None:
        self.summary_data = None
        self.cumul_data = None
        
    def filter_dates(self,dateinterval:list):
        """Filters the behavior data to analyze only the data in the date range"""

        # dateinterval is a list of two date strings e.g. ['200127','200131']
        if isinstance(dateinterval,str):
            dateinterval = [dateinterval]
            # add current day as end date
            dateinterval.append(dt.today().strftime('%y%m%d'))
        else:
            assert len(dateinterval)<=2, f'You need to provide a single start(1) or start and end dates(2), got {len(dateinterval)} dates'

        startdate = dt.strptime(dateinterval[0], '%y%m%d')
        enddate = dt.strptime(dateinterval[1], '%y%m%d')

        display('Retreiving between {0} - {1}'.format(startdate,enddate))

        self.summary_data['dt_date'] = self.summary_data['date'].apply(lambda x: dt.strptime(str(x),'%y%m%d'))
        self.cumul_data['dt_date'] = self.cumul_data['date'].apply(lambda x: dt.strptime(str(x),'%y%m%d'))

        self.summary_data[(self.summary_data['dt_date'] >= startdate) & (self.summary_data['dt_date'] <= enddate)]
        self.cumul_data[(self.cumul_data['dt_date'] >= startdate) & (self.cumul_data['dt_date'] <= enddate)]
        
    def save(self,path:str,task_type:str)->None:
        """ Saves the data in the given location"""
        cumul_save_name = pjoin(path,f'{task_type}TrainingData.parquet').replace("\\","/")
        summary_save_name = pjoin(path,f'{task_type}TrainingDataSummary.csv').replace("\\","/")
        
        # cast sf and tf to str
        summary_save_data = self.summary_data.with_columns([("[" + pl.col("sf").cast(pl.List(pl.Utf8)).list.join(", ") + "]").alias('sf'),
                                                            ("[" + pl.col("tf").cast(pl.List(pl.Utf8)).list.join(", ") + "]").alias('tf')])
        
        
        summary_save_data.write_csv(summary_save_name)
        self.cumul_data.write_parquet(cumul_save_name)


class Behavior:
    """ Analyzes the training progression of animals through multiple sessions
        animalid:     id of the animal to be analyzed(e.g. KC033)
        load_type:    string to set how to load
                      'lazy' = loads only the last analyzed data, doesn't analyze and add new sessions since last analysis
                      'full' = loads all the data and adds new sessions to the loaded data
                      'sessions' = loads the session data and reanalyzes the behavior data from that
                      'noload' = doesn't load anything reanalyzes the sessions data from scratch
    """
    def __init__(self,
                 animalid:str,
                 load_type:str='full', # ['lazy','full','sessions','noload']
                 *args,**kwargs) -> None:
        self.animalid = animalid
        self.load_type = load_type

        self.init_data_paths()
        self.session_list = self.get_sessions()
        
        self.db_interface = DataBaseInterface(self.databasepath)
        self.read_googlesheet()
        
    def init_data_paths(self) -> None:
        """ Initializes data paths """
        config = getConfig()
        self.presentationfolder = config['presentationPath']
        self.trainingfolder = config['trainingPath']
        self.analysisfolder = config['analysisPath']
        self.databasepath = config['databasePath']
        
    def get_sessions(self) -> list:
        """ self.session_list   n x 3 matrix that has the sesion directory names in the 1st column
                                and session dates in datetime format in the 2nd column and
                                session type in the 3rd column"""
        
        experiment_sessions = glob.glob('{0}/*{1}*__no_cam_*/'.format(self.presentationfolder,self.animalid))
        onep_sessions = glob.glob('{0}/*{1}*__1P_*/'.format(self.presentationfolder,self.animalid))
        twop_sessions = glob.glob('{0}/*{1}*__2P_*/'.format(self.presentationfolder,self.animalid))
        training_sessions = glob.glob('{0}/*{1}*__no_cam_*/'.format(self.trainingfolder,self.animalid))
        
        s = experiment_sessions + training_sessions + onep_sessions + twop_sessions
        tmp = [x.split(os.sep)[-2] for x in s]
        tmp_ordered = natsort.natsorted(tmp, reverse=False)
        
        # failsafe for duplicates due to data existing both in training and presentation
        tmp_ordered = np.unique(np.array(tmp_ordered)).tolist()
        
        session_list = []
        for sesh in tmp_ordered:
            date = dt.strptime(sesh.split('_')[0], '%y%m%d')
            if 'opto' in sesh:
                session_list.append([sesh,'opto',date])
            elif '1P' in sesh:
                session_list.append([sesh,'1P',date])
            elif '2P' in sesh:
                session_list.append([sesh,'2P',date])
            else:
                session_list.append([sesh,'train',date])

        return session_list
    
    def read_googlesheet(self) -> None:
        """ Reads all the entries from the googlesheet with the current animal id"""
        logsheet = GSheet('Mouse Database_new')
        # below, 2 is the log2021 sheet ID
        self.gsheet_df = logsheet.read_sheet(2)
        self.gsheet_df = self.gsheet_df[self.gsheet_df['Mouse ID'] == self.animalid]
        
        #convert decimal "," to "." and date string to datetime and drop na
        self.gsheet_df['weight [g]'] = self.gsheet_df['weight [g]'].apply(lambda x: str(x).replace(',','.'))
        self.gsheet_df['weight [g]'] = pd.to_numeric(self.gsheet_df['weight [g]'],errors='coerce')
        self.gsheet_df['supp water [µl]'] = pd.to_numeric(self.gsheet_df['supp water [µl]']).fillna(0)
        
        self.gsheet_df['Date [YYMMDD]'] = self.gsheet_df['Date [YYMMDD]'].apply(lambda x: str(x))
        self.gsheet_df['Date_dt'] = pd.to_datetime(self.gsheet_df['Date [YYMMDD]'], format='%y%m%d')
        
    def get_googlesheet_data(self,date:str,cols:list=None) -> pd.DataFrame:
        sheet_stats = {}
        if cols is None:
            cols = ['weight [g]','supp water [µl]','user','time [hh:mm]']
        #current date data
        row = self.gsheet_df[self.gsheet_df['Date [YYMMDD]']==date]
        for c in cols:
            key = c.split('[')[0].strip(' ') # get rid of units in column names
            if len(row):
                sheet_stats[key] = row[c].values[0]
            else:
                sheet_stats[key] = None
            
        return sheet_stats
    
    def get_non_data(self):
        """ Gets the non-data file rows from the googlesheet"""
        non_data = pd.DataFrame()
        temp = self.gsheet_df[self.gsheet_df['paradigm'].isin(['handling','water restriction start','surgery'])]
    
        non_data = pl.DataFrame({'weight'})
        non_data['date'] = temp['Date [YYMMDD]']
        non_data['weight'] = temp['weight [g]']
        non_data['paradigm'] = temp['paradigm']
        non_data['supp water [µl]'] = temp['supp water [µl]']
        non_data['user'] = temp['user']
        non_data['time [hh:mm]'] = temp['time [hh:mm]']
        return non_data
    
    def save_data(self,task_type:str) -> None:
        """ Saves the behavior data """
        latest_session = self.session_list[-1][0]
        self.savepath = pjoin(self.analysisfolder,latest_session).replace("\\","/")
        
        self.behavior_data.save(self.savepath)
        display(f'Behavior data saved in {self.savepath}')
        
        # deleting the old data
        if not self.behavior_data.summary_data.is_empty() and not self.behavior_data.cumul_data.is_empty():
            if self.cumul_file_loc is not None:
                if self.cumul_file_loc != latest_session:
                    display(f'Deleting the old data in {self.cumul_file_loc}')
                    os.remove(pjoin(self.analysisfolder,self.cumul_file_loc,f'{task_type}TrainingData.parquet'))
                    os.remove(pjoin(self.analysisfolder,self.summary_file_loc,f'{task_type}TrainingDataSummary.csv'))

    def isSaved(self,task_type:str) -> bool:
        """ Finds the session folder that has the saved behavior data"""
        cumul_data_saved_loc = glob.glob(f'{self.analysisfolder}/*{self.animalid}*/{task_type}TrainingData.parquet')
        summary_data_saved_loc = glob.glob(f'{self.analysisfolder}/*{self.animalid}*/{task_type}TrainingDataSummary.csv')
        
        if len(cumul_data_saved_loc) > 1 and len(summary_data_saved_loc) > 1:
            display(f'There should be only single _trainingData.parquet (most recent one) found {summary_data_saved_loc}')
            cumul_data_saved_loc = cumul_data_saved_loc[:1]
            summary_data_saved_loc = summary_data_saved_loc[:1]
        
        if len(cumul_data_saved_loc) == 1 and len(summary_data_saved_loc) == 1:
            # check if the location is same for both data, should be the case
            self.cumul_file_loc = cumul_data_saved_loc[0].split(os.sep)[-2]
            self.summary_file_loc = summary_data_saved_loc[0].split(os.sep)[-2]
            if self.cumul_file_loc == self.summary_file_loc:
                return True
            else:
                raise FileExistsError(f'Location of the cumulative data {self.cumul_file_loc} is not the same with summary data {self.summary_file_loc}')
        elif len(cumul_data_saved_loc) == 0 and len(summary_data_saved_loc) == 0:
            self.cumul_file_loc = None
            self.summary_file_loc = None
            return False
        else:
            raise RuntimeError('!! This should not happen! Saving of behavior data is messed up !!')
    
    def get_unanalyzed_sessions(self,task_type:str) -> list:
        """ Returns the list of sessions that have not been added to the behavior analysis"""
        is_saved = self.isSaved(task_type)
        if self.load_type == 'full':
            if is_saved:
                reverse_session_list = self.session_list[::-1] #this will have sessions listed from new to old for ease of search
                display(f'Found behavior data at {self.cumul_file_loc}')
                tmp = [i for i, x in enumerate(reverse_session_list) if x[0] == self.cumul_file_loc]
                missing_sessions = reverse_session_list[:tmp[0]] 
                missing_sessions = missing_sessions[::-1] # reverse again to have sessions added from old to new(chronological order)
                display(f'Adding {len(missing_sessions)} missing sessions to last analysis data')
            else:
                display('No behavior data present, creating new')
                missing_sessions = self.session_list
                
        elif self.load_type == 'sessions' or self.load_type == 'noload':
            missing_sessions = self.session_list
        elif self.load_type == 'lazy':
            # returns an empty list, no new session will be analyzed
            missing_sessions = []
            
        
        return missing_sessions
