from .wheelDetectionSession import *
from tabulate import tabulate

class WheelDetectionExperiment:
    __slots__ = ['exp_list','data','summary_data']
    def __init__(self,exp_list:list,load_sessions:bool=False) -> None:
        self.exp_list = exp_list
        self.data = self.parse_sessions(load_sessions=load_sessions)
        self.summary_data = self.make_summary_data()
    
    @staticmethod
    def parse_session_name(exp_dir) -> dict:
        """ Parses the exp names from the dir path of experiment"""
        exp_name = exp_dir.split('\\')[-1]
        return {
                    'opto_power' : int(exp_name.split('_')[3][-3:])/100,
                    'area' : exp_name.split('_')[4],
                    'exp_name' : exp_name
                } 
        
    def parse_sessions(self,load_sessions:bool=False) -> pl.DataFrame:
        """ Parses the sessions, names, meta info, some stats
        Returns a dict to 
        """
        pbar = tqdm(self.exp_list,desc='Reading sessions...',leave=True,position=0)
        for i,exp in enumerate(pbar):
            temp = self.parse_session_name(exp)

            w = WheelDetectionSession(temp['exp_name'],load_flag=load_sessions)
            data_len = len(w.data.data)
            lit = {
                'trial_count' : w.stats.all_count,
                'early_count' : w.stats.early_count,
                'correct_count' : w.stats.correct_count,
                'miss_count' : w.stats.miss_count,
                'hit_rate' : w.stats.hit_rate,
                'false_alarm' : w.stats.false_alarm,
                'opto_ratio' : w.meta.optoRatio,
                'opto_targets' : len(nonan_unique(w.data.data['opto_pattern'].to_numpy()))-1,
                'stimulus_count' : len(w.meta.sf_values),
                'isTitrated' : bool(w.data.data['isTitrated'][0]), # this should only give 0 or 1
                'rig' : w.meta.rig,
            }

            non_lit = {
                'contrast_vector' : [w.meta.contrastVector] * data_len,
                'sf_values' : [[float(i) for i in w.meta.sf_values]] * data_len,
                'tf_values' : [[float(i) for i in w.meta.tf_values]] * data_len
            }
            lit_dict = {**lit,**temp}
            
            # get the actual data and add the above meta, stat and temp(name stuff) to it as columns
            if i==0:
                # create the polars frame
                df = w.data.data.with_columns([pl.lit(v).alias(k) for k,v in lit_dict.items()])
                df = df.with_columns(pl.lit(i+1).cast(pl.Int32).alias('session_no'))
                list_df = pl.DataFrame(non_lit)
                df = pl.concat([df,list_df],how='horizontal')
                
            else:
                # concat to polars frame
                temp_df = w.data.data.with_columns([pl.lit(v).alias(k) for k,v in lit_dict.items()])
                temp_df = temp_df.with_columns(pl.lit(i+1).cast(pl.Int32).alias('session_no'))
                list_df = pl.DataFrame(non_lit)
                temp_df = pl.concat([temp_df,list_df],how='horizontal')
                
                
                #sorting the columns
                temp_df = temp_df.select(df.columns)
                try:
                    df = pl.concat([df,temp_df])
                except Exception as err:
                    print(f"SOMETHING WRONG WITH {temp['exp_name']}")
                    raise err
                    
            pbar.update()
        
        # make reorder column list
        reorder = ['total_trial_no'] + df.columns
        # in the end add a final all trial count column
        df = df.with_columns(pl.Series(name='total_trial_no',values=np.arange(1,len(df)+1).tolist()))
        # reorder
        df = df.select(reorder)
        
        return df
    
    def filter_sessions(self,filter_dict:dict=None) -> pl.DataFrame:
        """Filters the self.data according to filter_dict, if None returns self.data as is"""
        list_names = ['contrast_vector','sf_values','tf_values']
        
        filt_df = self.data.select(pl.col('*'))
        
        for k,v in filter_dict.items():
            
            if k not in filt_df.columns:
                raise KeyError(f'The filter key {k} is not present in the data columns, make sure you have the correct data column names')

            if k not in list_names:
                if isinstance(v,list):
                    temp_df = pl.DataFrame()
                    for v_elem in v:
                        t = filt_df.filter(pl.col(k) == v_elem)
                        
                        uniq_sessions = t['session_no'].unique().to_list()
                        if len(uniq_sessions)>1:
                            for sesh_id in uniq_sessions:
                                #sometimes some dates have multiple sessions
                                # loop over dates
                                t2 = t.filter(pl.col('session_no')==sesh_id)
                                temp_df = pl.concat([temp_df,t2])
                        else:
                            temp_df = pl.concat([temp_df,t])
                    filt_df = temp_df
                else:
                    filt_df = filt_df.filter(pl.col(k) == v)
            else:
                for l in v:
                    filt_df = filt_df.filter(pl.col(k).arr.contains(l))
            
            
        return filt_df

    def make_summary_data(self) -> pl.DataFrame:
        """ Creates a summary data and and prints a tabulated text description of it"""
        q = (
            self.data.lazy()
            .groupby(["animalid","area","stimulus_count","opto_targets","isTitrated"])
            .agg(
                [   (pl.col("stim_type").unique(maintain_order=True)),
                    (pl.col("date").unique().count().alias("experiment_count")),
                    (pl.col("date").unique(maintain_order=True)),
                    (pl.col("session_no").unique(maintain_order=True).alias("session_ids")),
                    (pl.col("trial_count").unique(maintain_order=True)),
                    (pl.count().alias("total_trials")),
                    (pl.col("hit_rate").unique(maintain_order=True)),
                    (pl.col("false_alarm").unique(maintain_order=True))
                ]
            ).drop_nulls()
            .sort(["animalid","area","stimulus_count","opto_targets","isTitrated"])
        )
        df = q.collect()
        return df
        
    def print_summary(self) -> None:
        """Prints the summary data"""
        tmp = self.summary_data.to_pandas()
        print(tabulate(tmp,headers=self.summary_data.columns))
    
    def filter_experiments(self,
                           area:str,
                           stim_count:int=1,
                           stim_type:str=None,
                           opto_targets:int=1,
                           isTitrated:bool=False,
                           verbose:bool=True) -> pl.DataFrame:
        """ Filters the summary data according to 3 arguments,
        Then uses the dates in those filtered sessions to filter self.data"""
        
        filt_summ = self.summary_data.filter((pl.col('area')==area) &
                                             (pl.col('stimulus_count')==stim_count) &
                                             (pl.col('opto_targets')==opto_targets) & 
                                             (pl.col('isTitrated')==isTitrated))
        
        if verbose:
            print(filt_summ)
            
        ids = filt_summ['session_ids'].explode().unique().to_list()
        filter_dict = {'area':area,
                       'session_no':ids}
        
        if stim_count==1 and stim_type is not None:
            filter_dict['stim_type'] = stim_type
            
        filt_df = self.filter_sessions(filter_dict)

        return filt_df
                
        
        