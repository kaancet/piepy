from .wheelDetectionSession import *
from behavior_python.core.behavior import Behavior,BehaviorData


class WheelDetectionBehaviorData(BehaviorData):
    def __init__(self) -> None:
        super().__init__()
        self._convert = ['wheel','lick','reward']
        
    def __repr__(self):
        rep = f''' Wheel Detection Behavior Data'''
        return rep
    
    def save(self,path:str)->None:
        """ Saves the data in the given location"""
        super().save(path,'detect')


class WheelDetectionBehavior(Behavior):
    def __init__(self, 
                 animalid:str,
                 load_type:str=None,
                 *args,**kwargs):
        super().__init__(animalid,load_type,*args,**kwargs)
        
        self.behavior_data = WheelDetectionBehaviorData()
        # get only detection
        self.session_list = self.filter_sessions() # this is a failsafe to get only detect sessions
        
        self.get_behavior()
        
        # here is where task specific things could go
        self.save_behavior()
        
    def filter_sessions(self):
        return [i for i in self.session_list if 'detect' in i[0]]

    @timeit('Getting behavior data...')
    def get_behavior(self):
        """ Loads the behavior data(cumul and summary)"""
        missing_sessions = self.get_unanalyzed_sessions('detect')
        pbar = tqdm(missing_sessions)
        
        if len(missing_sessions) == len(self.session_list):
            # noload and sessions will enter here for sure
            # full will enter here if only there is no data to load
            cumul_data = pl.DataFrame()
            summary_data = pl.DataFrame()
            session_counter = 0
        else:
            # this loads the most recent found data
            # full and lazy will enter here
            cumul_data = pl.read_parquet(pjoin(self.analysisfolder,self.cumul_file_loc,'detectTrainingData.parquet'))
            summary_data = pl.read_csv(pjoin(self.analysisfolder,self.summary_file_loc,'detectTrainingDataSummary.csv'))
            
            # sf and tf needs to be converted back to lists
            summary_data = summary_data.with_columns([pl.col("sf")
                                                     .str.replace_all("[",'',literal=True)
                                                     .str.replace_all(']','',literal=True)
                                                     .str.split(',')
                                                     .apply(lambda x: [float(i) for i in x])
                                                     .alias('sf'),
                                                     pl.col("tf")
                                                     .str.replace_all("[",'',literal=True)
                                                     .str.replace_all(']','',literal=True)
                                                     .str.split(',')
                                                     .apply(lambda x: [float(i) for i in x])
                                                     .alias('tf')
                                                   ])
            
            session_counter = summary_data[-1,'session_no']
        
        summary_to_append = []
        for i,sesh in enumerate(pbar):
            # lazy will not enter here as missing sessions will be []
            pbar.set_description(f'Analyzing {sesh[0]} [{i+1}/{len(missing_sessions)}]')
            
            if self.load_type == 'noload':
                detect_session = WheelDetectionSession(sesh[0],load_flag=False)
            else:
                # 'full' or 'sessions' load type should enter here
                detect_session = WheelDetectionSession(sesh[0],load_flag=True)

            session_data = detect_session.data.data
            gsheet_dict = self.get_googlesheet_data(detect_session.meta.baredate,
                                                    cols=['paradigm','supp water [µl]','user','time [hh:mm]','rig water [µl]'])
            
            if len(session_data):
                # add behavior related fields as a dictiionary
                summary_temp = {}
                summary_temp['date'] = detect_session.meta.baredate
                summary_temp['blank_time'] = detect_session.meta.openStimDuration
                summary_temp['response_window'] = detect_session.meta.closedStimDuration
                try:
                    summary_temp['level'] = int(detect_session.meta.level)
                except:
                    summary_temp['level'] = -1
                summary_temp['session_no'] = session_counter + 1
                
                # put data from session stats
                for k in detect_session.stats.__slots__:
                    summary_temp[k] = getattr(detect_session.stats,k,None)

                # put values from session meta data
                summary_temp['weight'] = detect_session.meta.weight
                summary_temp['task'] = detect_session.meta.controller
                summary_temp['sf'] = detect_session.meta.sf_values
                summary_temp['tf'] = detect_session.meta.tf_values
                summary_temp['rig'] = detect_session.meta.rig
                summary_temp = {**summary_temp, **gsheet_dict}
                
                # cumulative data
                session_data = session_data.with_columns([(pl.lit(session_counter+1)).alias('session_no'),
                                                            (pl.lit(sesh[1])).alias('session_type')])
                
                if i == 0:
                    summary_to_append = {k:[v] for k,v in summary_temp.items()}
                    cumul_data = session_data
                else:
                    for k,v in summary_temp.items():
                        summary_to_append[k].append(v)
                    #sorting the columns
                    session_data = session_data.select(cumul_data.columns)
                    try:
                        cumul_data = pl.concat([cumul_data,session_data])
                    except:
                        session_data = session_data.with_columns(pl.col('opto').cast(pl.Int64).alias('opto'))
                        cumul_data = cumul_data.with_columns(pl.col('opto_region').cast(str).alias('opto_region'))
                        session_data = session_data.with_columns(pl.col('opto_region').cast(str).alias('opto_region'))
                        cumul_data = pl.concat([cumul_data,session_data])
                
                session_counter += 1 
            else:
                display(f' >>> WARNING << NO DATA FOR SESSION {sesh[0]}')
                continue
        
        if 'cumul_trial_no' in cumul_data.columns:
            cumul_data = cumul_data.drop('cumul_trial_no')
            
        cumul_trial_no = pl.Series('cumul_trial_no',np.arange(len(cumul_data)) + 1)
        cumul_data = cumul_data.hstack([cumul_trial_no])
        # cumul_data = get_running_stats(cumul_data,window_size=50)
        if len(summary_to_append):
            summary_data = pl.concat([summary_data,pl.DataFrame(summary_to_append)])
               
        # adding the non-data stages of training once in the beginning
        # this should happen when loading non-lazy and 
        # if len(missing_sessions) == len(self.session_list):
        #     non_data = self.get_non_data()
        #     summary_data = pl.concat([summary_data,non_data])
        # Failsafe date sorting for non-analyzed all trials and empty sessions(?)
            # summary_data = summary_data.sort('date')

        self.behavior_data.summary_data = summary_data
        self.behavior_data.cumul_data = cumul_data
            
    def save_behavior(self):
        """ Saves the behavior data """
        # save behavior data to the last session analysis folder
        self.save_data('detect')


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Wheel Behavior Data Parsing Tool')

    parser.add_argument('id',metavar='animalid',
                        type=str,help='Animal ID (e.g. KC020)')
    parser.add_argument('-d','--date',metavar='dateinterval',
                        type=str,help='Analysis start date (e.g. 191124)')
    parser.add_argument('-c','--criteria',metavar='criteria',default=[20,0],
                        type=str, help='Criteria dict for analysis thresholding, delimited list input')
    
    '''
    wheelbehave -d 200501 -c "20, 10" KC028
    '''

    opts = parser.parse_args()
    animalid = opts.id
    dateinterval = opts.date
    tmp = [int(x) for x in opts.criteria.split(',')]
    criteria = dict(answered_trials=tmp[0],
                    answered_correct=tmp[1])

    display('Updating Wheel Behavior for {0}'.format(animalid))
    display('Set criteria: {0}: {1}\n\t\t{2}: {3}'.format(list(criteria.keys())[0],
                                                  list(criteria.values())[0],
                                                  list(criteria.keys())[1],
                                                  list(criteria.values())[1]))
    w = WheelDetectionBehavior(animalid=animalid, dateinterval=dateinterval, criteria=criteria)

if __name__ == '__main__':
    main()
