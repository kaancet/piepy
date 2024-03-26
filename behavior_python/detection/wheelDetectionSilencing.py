from .wheelDetectionSession import *
from scipy.optimize import curve_fit

class WheelDetectionExperiment:
    def __init__(self,exp_list:list,load_sessions:bool=False) -> None:
        self.exp_list = exp_list
        self.load_sessions = load_sessions
        
    def set_data(self,data:pl.DataFrame=None) -> None:
        if data is not None:
            self.data = data    
        else:
            self.data = self.parse_sessions(load_sessions=self.load_sessions)
        self.summary_data = self.make_summary_data()
        
    def transform_to_rig_time(self) -> None:
        """ Transforms the reaction time of trials that dont't have rig_reaction_time to that time frame """
        with_rig_time = self.data.drop_nulls('rig_reaction_time')
        resp_time = with_rig_time['response_latency']
        rig_time = with_rig_time['rig_reaction_time']
        
        def m1_func(x,a):
            m = 1
            return m*x + a
        
        popt,pcov = curve_fit(m1_func,resp_time,rig_time) #popt[0] is the time diff intercept
        
        all_resp_time = self.data['response_latency']
        new_rig_times = m1_func(all_resp_time,*popt)

        tmp = pl.Series('transformed_response_times',new_rig_times)
        self.data = self.data.with_columns(tmp)
    
    @staticmethod
    def parse_session_name(exp_dir) -> dict:
        """ Parses the exp names from the dir path of experiment"""
        exp_name = exp_dir.split('\\')[-1]
        area = exp_name.split('_')[4]
        isCNO = False
        if 'CNO' in area:
            area = area.strip('CNO')
            isCNO = True
        return {
                    'opto_power' : int(exp_name.split('_')[3][-3:])/100,
                    'area' : area,
                    'exp_name' : exp_name,
                    'isCNO' : isCNO
                } 
        
    def parse_sessions(self,load_sessions:bool=False) -> pl.DataFrame:
        """ Parses the sessions, names, meta info, some stats
        Returns a dict to 
        """
        pbar = tqdm(self.exp_list,desc='Reading sessions...',leave=True,position=0)
        for i,exp in enumerate(pbar):
            temp = self.parse_session_name(exp)

            w = WheelDetectionSession(temp['exp_name'],load_flag=load_sessions,skip_google=True)
            data_len = len(w.data.data)
            meta = w.get_meta()
            lit = {
                'trial_count' : w.stats.all_count,
                'early_count' : w.stats.early_count,
                'correct_count' : w.stats.correct_count,
                'miss_count' : w.stats.miss_count,
                'hit_rate' : w.stats.hit_rate,
                'false_alarm' : w.stats.false_alarm,
                'opto_ratio' : meta.optoRatio,
                'opto_targets' : len(nonan_unique(w.data.data['opto_pattern'].to_numpy()))-1,
                'stimulus_count' : len(meta.sf_values),
                'isTitrated' : bool(w.data.data['isTitrated'][0]), # this should only give 0 or 1
                'rig' : meta.rig,
            }

            non_lit = {
                'contrast_vector' : [meta.contrastVector] * data_len,
                'sf_values' : [[float(i) for i in meta.sf_values]] * data_len,
                'tf_values' : [[float(i) for i in meta.tf_values]] * data_len
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
                
                #if there are columns that are not in df add them
                for c in temp_df.columns:
                    if c not in df.columns:
                        df = df.with_columns(pl.lit(None).alias(c))
                
                temp_df = temp_df.select(df.columns)
                # fixing column datatypes
                if df.dtypes != temp_df.dtypes:
                    df = df.with_columns([pl.col(n).cast(t) for n,t in zip(df.columns,temp_df.dtypes) if t!=pl.Null])
                try:
                    df = pl.concat([df,temp_df])
                except pl.SchemaError:
                    raise pl.SchemaError(f"WEIRDNESS WITH COLUMNS AT {temp['exp_name']}")

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
            .groupby(["animalid","area","stimulus_count","opto_targets","isTitrated","isCNO"])
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
            .sort(["animalid","area","stimulus_count","opto_targets","isTitrated","isCNO"])
        )
        df = q.collect()
        return df
        
    def print_summary(self) -> None:
        """Prints the summary data"""
        tmp = self.summary_data.to_pandas()
        print(tabulate(tmp,headers=self.summary_data.columns))
    
    
    def filter_by_params2(self,
                         filter_kwargs:dict,
                         verbose:bool=True) -> pl.DataFrame:
        
        for k in filter_kwargs.keys():
            if k not in self.summary_data.columns:
                raise ValueError(f"Filtering parameter {k} not in data columns!!")
        
        filt_summ = self.summary_data.filter(**filter_kwargs)
        
        if verbose:
            print(filt_summ)
            
        ids = filt_summ['session_ids'].explode().unique().to_list()
        filter_dict = {'session_no':ids}
        
        if stim_count==1 and stim_type is not None:
            filter_dict['stim_type'] = stim_type
            
        filt_df = self.filter_sessions(filter_dict)

        return filt_df
    
    ##TODO: Below filtering logic can be better
    def filter_by_params(self,
                         df:pl.DataFrame=None,
                         stim_count:int=1,
                         stim_type:str=None,
                         opto_targets:int=1,
                         isTitrated:bool=False,
                         isCNO:bool=False,
                         verbose:bool=True) -> pl.DataFrame:
        
        if df is None:
            df = self.summary_data
        
        filt_summ = df.filter((pl.col('stimulus_count')==stim_count) &
                              (pl.col('opto_targets')==opto_targets) & 
                              (pl.col('isCNO')==isCNO) &
                              (pl.col('isTitrated')==isTitrated))
        
        if verbose:
            print(filt_summ)
            
        ids = filt_summ['session_ids'].explode().unique().to_list()
        filter_dict = {'session_no':ids}
        
        if stim_count==1 and stim_type is not None:
            filter_dict['stim_type'] = stim_type
            
        filt_df = self.filter_sessions(filter_dict)

        return filt_df
        
    def filter_by_animal(self,
                         animalid:str,
                         stim_count:int=1,
                         stim_type:str=None,
                         opto_targets:int=1,
                         isTitrated:bool=False,
                         isCNO:bool=False,
                         verbose:bool=True) -> pl.DataFrame:
        
        filt_summ = self.summary_data.filter((pl.col('animalid')==animalid))
        filt_df = self.filter_by_params(filt_summ,stim_count,stim_type,opto_targets,isTitrated,isCNO,verbose)
        return filt_df
        
    
    def filter_by_area(self,
                       area:str,
                       stim_count:int=1,
                       stim_type:str=None,
                       opto_targets:int=1,
                       isTitrated:bool=False,
                       isCNO:bool=False,
                       verbose:bool=True) -> pl.DataFrame:
        """ Filters the summary data according to 3 arguments,
        Then uses the dates in those filtered sessions to filter self.data"""
        
        filt_summ = self.summary_data.filter((pl.col('area')==area))
        filt_df = self.filter_by_params(filt_summ,stim_count,stim_type,opto_targets,isTitrated,isCNO,verbose)
        return filt_df
                
        
        