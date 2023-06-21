from .wheelDetectionSession import *
from tabulate import tabulate

class WheelDetectionExperiment:
    __slots__ = ['exp_list','data']
    def __init__(self,exp_list:list,load_sessions:bool=False) -> None:
        self.exp_list = exp_list
        self.data = self.parse_sessions(load_sessions=load_sessions)
    
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
                'opto_targets' : len(np.unique(w.data.data.drop_nulls()['opto_pattern']))-1,
                'stimulus_types' : len(w.meta.sf_values),
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
                df = pl.concat([df,temp_df])
                
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
        if filter_dict is None:
            return self.data
        
        filt_df = self.data.select(pl.col('*'))
        
        for k,v in filter_dict.items():
            try:
                if k not in list_names:
                    filt_df = filt_df.filter(pl.col(k) == v)
                else:
                    for l in v:
                        filt_df = filt_df.filter(pl.col(k).arr.contains(l))
            except:
                raise KeyError(f'The filter key {k} is not present in the data columns, make sure you have the correct data column names')
            
        return filt_df

    def print_data_summary(self) -> None:
        """ Prints a tabulated text description of the data argument """
        q = (
            self.data.lazy()
            .groupby(["animalid","area","stimulus_types","opto_targets"])
            .agg(
                [   (pl.col("date").unique().count().alias("experiment_count")),
                    (pl.col("date").unique(maintain_order=True)),
                    (pl.col("trial_count").unique(maintain_order=True)),
                    (pl.count().alias("total_trials")),
                    (pl.col("hit_rate").unique(maintain_order=True)),
                    (pl.col("false_alarm").unique(maintain_order=True))
                ]
            ).drop_nulls()
            .sort(["animalid","area","stimulus_types","opto_targets"])
        )
        df = q.collect()
        
        tmp = df.to_pandas()
        print(tabulate(tmp,headers=df.columns))
        