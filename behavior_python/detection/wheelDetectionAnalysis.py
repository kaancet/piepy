import polars as pl
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon,fisher_exact,barnard_exact,boschloo_exact
from ..utils import display


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
    
    def get_hitrate_pvalues_exact(self,side:str='contra', method:str='barnard') -> pl.DataFrame:
        """ Calculates the p-values for each contrast in different stim datasets"""
        q = (
                self.agg_data.lazy()
                .groupby(["stim_type","contrast","stim_side","opto"])
                .agg(
                    [        
                        (pl.col("opto_pattern")),
                        (pl.col("correct_count").first()),
                        (pl.col("miss_count").first()),
                        (pl.col("stimkey").first()),
                        (pl.col("stim_label").first())
                    ]
                ).sort(["stim_type","contrast","stim_side"])
            )
        
        temp = q.filter((pl.col("stim_side")==side)).sort(["stim_type","contrast","opto"]).collect()
        
        stimkey_list = temp['stim_type'].unique().to_numpy()
        c_list = temp['contrast'].unique().to_numpy()
        o_list = temp['opto'].unique().to_numpy()
        # there should be 2 entries(opto-nonopto) per stim_type and contrast combination
        # otherwise cannot create the table to do p_value analyses
        # if len(temp) != (len(c_list)*len(stimkey_list)*2): 
        if len(o_list) != 2: # opto and nonopto
            display("CAN'T DO P-VALUE ANALYSIS, MISSING OPTO COMPONENTS!! RETURNING AN EMPTY DATAFRAME")
            df = pl.DataFrame()
        else:
            stims = []
            contrasts = []
            p_values = []
            for s in stimkey_list:
                for c in c_list:
                    filt = temp.filter((pl.col("contrast")==c) &
                                    (pl.col("stim_type")==s))
                    table = filt[:,['correct_count','miss_count']].to_numpy()
                    if np.all(np.isnan(table)==False): # all elements are filled
                        if method == 'barnard':
                            res = barnard_exact(table, alternative='two-sided')
                        elif method == 'boschloo':
                            res = boschloo_exact(table,alternative='two-sided')
                        elif method == 'fischer':
                            res = fisher_exact(table,alternative='two-sided')
                        p = res.pvalue
                    else:
                        p = np.nan
                    
                    stims.append(s)
                    contrasts.append(c)
                    p_values.append(p)
            
            df = pl.DataFrame({"stim_type":stims,
                            "contrast":contrasts,
                            "p_values":p_values})        
        return df

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