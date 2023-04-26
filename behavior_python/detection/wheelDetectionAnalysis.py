import polars as pl
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon,fisher_exact,barnard_exact,boschloo_exact
from ..utils import nonan_unique
from collections import defaultdict


class DetectionAnalysis:
    def __init__(self,data:pl.DataFrame) -> None:
        self.data = data
        self.agg_data = self.make_agg_data()
        
    def make_agg_data(self) -> pl.DataFrame:
        """ Gets the hit rates, counts and confidence intervals for each contrast for each side """
        q = (
            self.agg_data.lazy()
            .groupby(["stimkey","contrast","stim_side"])
            .agg(
                [
                    (pl.col("stim_label").first()),
                    (pl.col("stim_pos").first()),
                    pl.count().alias("count"),
                    (pl.col("answer")==1).sum().alias("correct_count"),
                    (pl.col("answer")==0).sum().alias("miss_count"),
                    (pl.col("response_latency").median().alias("median_response_time")),
                    (pl.col("opto").first().cast(pl.Int8))
                ]
            ).sort("contrast")
            )

        q = q.with_columns((pl.col("correct_count") / pl.col("count")).alias("hit_rate"))
        q = q.with_columns((1.96 * np.sqrt((pl.col("hit_rate")*(1.0 - pl.col("hit_rate"))) / pl.col("count"))).alias("confs"))

        # reorder stim_label to last column
        cols = q.columns
        del cols[3]
        cols.append('stim_label')
        q = q.select(cols)
        
        df = q.collect()
        return df

    def get_deltahit(self,contrast:float,side:str='contra'):
        """Return the delta between the hitrates of a given contrast """
        
        filt_df = self.agg_data.filter((pl.col("contrast")==contrast) & 
                                       (pl.col("stim_side")==side)).sort('opto_pattern')
        
        q = (
            filt_df.lazy()
            .groupby(["spatial_freq","temporal_freq"])
            .agg(
                [   
                    (pl.col("stimkey").first()),
                    (pl.col("stim_side").first()),
                    (pl.col("contrast").first()),
                    (pl.col("hit_rate").first()).alias('base_HR'),
                    (pl.col("hit_rate").first()-pl.col("hit_rate").last()).alias('delta_HR'),
                    (pl.col("median_response_time").last()-pl.col("median_response_time").first()).alias('delta_resp'),
                ]
            )
            )

        filt_agg = filt_agg.with_columns((pl.col('delta_HR')/pl.col('base_HR')).alias('normalized_delta_HR'))
        filt_agg = q.collect()

        return filt_agg
    
    def get_hitrate_pvalues_exact(self, stim_data_keys:list=None, stim_side:str='contra', method:str='barnard') -> dict:
        """ Calculates the p-values for each contrast in different stim datasets"""
        if stim_data_keys is None:
            stim_data_keys = self.data.keys()
            
        if len(stim_data_keys) != 2:
            raise ValueError(f'There should be 2 different sets of stimuli data to get the p value in hit rate differences')
        
        def def_value():
            contingency_table = np.zeros((2,2))
            contingency_table[:] = np.nan
            return contingency_table
        
        table_dict = defaultdict(def_value) # this guy gets contrast values as keys
        for i,k in enumerate(stim_data_keys):
            v = self.data[k]
            if stim_side == 'contra':
                side_data = v[v['stim_side'] > 0]
            elif stim_side == 'ipsi':
                side_data = v[v['stim_side'] < 0]
            elif stim_side == 'catch':
                side_data = v[v['stim_side'] == 0]
            else:
                raise ValueError(f'stim_side argument only takes [contra,ipsi,catch] values')
            
            contrast_list = nonan_unique(side_data['contrast'],sort=True)
            for c in contrast_list:
                data_correct = side_data[(side_data['contrast']==c) & (side_data['answer']==1)]
                data_incorrect = side_data[(side_data['contrast']==c) & (side_data['answer']==0)]

                table_dict[c][i,:] = [len(data_correct), len(data_incorrect)]
            
        p_values = {}
        for k,table in table_dict.items():
            if np.all(np.isnan(table)==False): # all elements are filled
                if method == 'barnard':
                    res = barnard_exact(table, alternative='two-sided')
                elif method == 'boschloo':
                    res = boschloo_exact(table,alternative='two-sided')
                elif method == 'fischer':
                    res = fisher_exact(table,alternative='two-sided')
                p_values[k] = res.pvalue
            else:
                p_values[k] = np.nan
        
        return p_values

    @staticmethod
    def get_pvalues_nonparametric(x1,x2,method:str='mannu') -> dict:
        """ Returns the significance value of two distributions """
        if method not in ['mannu','wilcoxon']:
            raise ValueError(f'{method} not a valid statistical test yet, try mannu or wilcoxon')
        if method == 'mannu':
            _,p = mannwhitneyu(x1,x2)
        elif method == 'wilcoxon':
            res = wilcoxon(x1,x2)
            p = res.p
        return p