from .core import *
import glob
import natsort
        

class BehaviorData:
    """ Keeps the data as a huge csv of appended session data"""
    __slots__ = ['summary_data','cumul_data','_convert','dateinterval']
    def __init__(self,dateinterval:str=None) -> None:
        self.summary_data = None
        self.cumul_data = None
        self.dateinterval = dateinterval
        
    def filter_dates(self):
        """Filters the behavior data to analyze only the data in the date range"""
        if self.dateinterval is None:
            self.dateinterval = [self.summary_data['date'].iloc[0],self.summary_data['date'].iloc[-1]]
            
        # dateinterval is a list of two date strings e.g. ['200127','200131']
        if type(self.dateinterval) is not list:
            self.dateinterval = [self.dateinterval]
            # add current day as end date
            self.dateinterval.append(dt.today().strftime('%y%m%d'))

        if len(self.dateinterval) == 2:
            startdate = dt.strptime(self.dateinterval[0], '%y%m%d')
            enddate = dt.strptime(self.dateinterval[1], '%y%m%d')
        else: 
            raise Exception('Date interval wrong!')

        display('Retreiving between {0} - {1}'.format(startdate,enddate))

        self.summary_data['dt_date'] = self.summary_data['date'].apply(lambda x: dt.strptime(str(x),'%y%m%d'))
        self.cumul_data['dt_date'] = self.cumul_data['date'].apply(lambda x: dt.strptime(str(x),'%y%m%d'))

        self.summary_data[(self.summary_data['dt_date'] >= startdate) & (self.summary_data['dt_date'] <= enddate)]
        self.cumul_data[(self.cumul_data['dt_date'] >= startdate) & (self.cumul_data['dt_date'] <= enddate)]
        
    def save(self,path:str,task_type:str)->None:
        """ Saves the data in the given location"""
        cumul_save_name = pjoin(path,f'{task_type}TrainingData.behave').replace("\\","/")
        summary_save_name = pjoin(path,f'{task_type}TrainingDataSummary.csv').replace("\\","/")
        self.summary_data.to_csv(summary_save_name,index=False)
        self.cumul_data.to_pickle(cumul_save_name)
        
    def make_saveable(self) -> pd.DataFrame:
        """ The columns that have numpy.ndarrays need to be saved as lists in DataFrame columns!
        A copy of the session_data is created with converted columns to be saved"""        
        save_df = self.cumul_data.copy(deep=True)
        for col in self._convert:
            save_df[col] = save_df[col].apply(lambda x: x.tolist())
        return save_df

    def make_loadable(self) -> None:
        """ Converts the necessary columns into numpy array """
        for col in self._convert:
            if not isinstance(self.cumul_data[col].iloc[0], np.ndarray):
                self.cumul_data[col] = self.cumul_data[col].apply(eval).apply(lambda x: np.array(x))
    

class BehaviorStats:
    __slots__ = ['handling_days','habituation_days','training_days',
                 'experiment_days','trial_count','session_count']
    def __init__(self,data:BehaviorData) -> None:
        self.init_from_data(data)
        
    def init_from_data(self,data:BehaviorData) -> None:
        summary_data = data.summary_data
        cumul_data = data.cumul_data
        
        self.session_count = summary_data['session_no'].iloc[-1]
        self.handling_days = len(summary_data[summary_data['paradigm']=='handling'])
        self.habituation_days = len(summary_data[summary_data['paradigm']=='habituation'])
        self.training_days = len(summary_data[summary_data['paradigm'].str.contains('training')])
        self.experiment_days = len(summary_data[summary_data['paradigm'].str.contains('experiment')])
        
        self.tral_count = cumul_data['cumul_trial_no'].iloc


class Behavior:
    """ Analyzes the training progression of animals through multiple sessions
            animalid:     id of the animal to be analyzed(e.g. KC033)
            dateinterval: interval of dates to analyze the data, 
                          can be a list of two dates, or a string of starting date
                          (e.g. ['200110','200619'] OR '200110')
            load_behave:  flag to either load previously analyzed behavior data or to analyze it from scratch
            load_data:    flag to either load previously parsed data or to parse it again
            """
    def __init__(self,animalid,dateinterval=None,load_data:bool=True,load_behave:bool=True,*args,**kwargs) -> None:
        self.animalid = animalid
        self.dateinterval = dateinterval
        self.load_behave = load_behave
        self.load_data = load_data

        # no point in reading saved behavior data if reanalyzing session data
        if not self.load_data:
            self.load_behave = False

        self.init_data_paths()
        self.get_sessions()
        
        self.db_interface = DataBaseInterface(self.databasepath)
        self.read_googlesheet()
        
    def init_data_paths(self) -> None:
        """ Initializes data paths """
        config = getConfig()
        self.presentationfolder = config['presentationPath']
        self.trainingfolder = config['trainingPath']
        self.analysisfolder = config['analysisPath']
        self.databasepath = config['databasePath']
        
    def get_sessions(self) -> None:
        """ self.session_list   n x 2 matrix that has the sesion directory names in the 1st column
                                and session dates in datetime format in the 2nd column"""
        experiment_sessions = glob.glob('{0}/*{1}*__no_cam_*/'.format(self.presentationfolder,self.animalid))
        training_sessions = glob.glob('{0}/*{1}*__no_cam_*/'.format(self.trainingfolder,self.animalid))
        onep_sessions = glob.glob('{0}/*{1}*__1P_*/'.format(self.trainingfolder,self.animalid))
        twop_sessoins = glob.glob('{0}/*{1}*__2P_*/'.format(self.trainingfolder,self.animalid))
        
        s = experiment_sessions + training_sessions + onep_sessions
        session_list = [x.split(os.sep)[-2] for x in s]
        session_list = natsort.natsorted(session_list, reverse=False)
        
        # failsafe for duplicates due to data existing both in training and presentation
        session_list = np.unique(np.array(session_list)).tolist()

        date_list = [dt.strptime(x.split('_')[0], '%y%m%d') for x in session_list]

        self.session_list = [[session_list[i], date] for i, date in enumerate(date_list)]
    
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
        if len(row):
            for c in cols:
                key = c.split('[')[0].strip(' ') # get rid of units in column names
                sheet_stats[key] = row[c].values[0]
        else:
            display('No Training Log entry {0} {1}'.format(self.animalid, date))
            sheet_stats = {}

        return sheet_stats
    
    def get_non_data(self):
        """ Gets the non-data file rows from the googlesheet"""
        non_data = pd.DataFrame()
        temp = self.gsheet_df[self.gsheet_df['paradigm'].isin(['handling','water restriction start','surgery'])]
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
        if not self.behavior_data.summary_data.empty and not self.behavior_data.cumul_data.empty:
            if self.cumul_file_loc is not None:
                if self.cumul_file_loc != latest_session:
                    display(f'Deleting the old data in {self.cumul_file_loc}')
                    os.remove(pjoin(self.analysisfolder,self.cumul_file_loc,f'{task_type}TrainingData.behave'))
                    os.remove(pjoin(self.analysisfolder,self.summary_file_loc,f'{task_type}TrainingDataSummary.csv'))

    def isSaved(self,task_type:str) -> bool:
        """ Finds the session folder that has the saved behavior data"""
        cumul_data_saved_loc = glob.glob(f'{self.analysisfolder}/*{self.animalid}*/{task_type}TrainingData.behave')
        summary_data_saved_loc = glob.glob(f'{self.analysisfolder}/*{self.animalid}*/{task_type}TrainingDataSummary.csv')
        
        if len(cumul_data_saved_loc) > 1 and len(summary_data_saved_loc) > 1:
            display(f'There should be only single _trainingData.behave (most recent one) found {summary_data_saved_loc}')
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
        if self.load_behave:
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
        else:
            missing_sessions = self.session_list
        
        return missing_sessions
