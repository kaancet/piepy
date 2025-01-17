import polars as pl
import numpy as np
from scipy.stats import (
    mannwhitneyu,
    wilcoxon,
    fisher_exact,
    barnard_exact,
    boschloo_exact,
)

from ....core.data_functions import make_subsets


class WheelDetectionGroupedAnalyzer:
    """ """
    def __init__(self,data: pl.DataFrame = None, **kwargs) -> None:
        if data is not None:
            self.set_data(data,**kwargs)

    def set_data(self, data: pl.DataFrame, **kwargs) -> None:
        self.data = data
        self.grouped_data = self.group_data(kwargs.get("group_by",None),
                                            kwargs.get("do_sort",True))
        self.set_hit_rates(kwargs.get("p_method","barnard"))
    
    def group_data(self, group_by:list[str]=None, do_sort:bool=True) -> pl.DataFrame:
        """ Groups the data by guven group_by column names """
        
        if group_by is None:
            group_by = ["stim_type", "contrast", "stim_side", "opto_pattern"]
        
        for c_name in group_by:
            if c_name not in self.data.columns:
                raise ValueError(f"{c_name} not in data columns!!")
            
        q = (self.data
             .group_by(group_by)
             .agg(
                 [     
                    (pl.col("stim_pos").first()),
                    pl.count().alias("count"),
                    (pl.col("outcome") == "hit").sum().alias("hit_count"),
                    (pl.col("outcome") == "miss").sum().alias("miss_count"),
                    (pl.col("response_time").alias("response_times")),
                    (pl.col("response_time")
                     .filter(pl.col("outcome")=="hit")
                     .alias("hit_response_times")),
                    (pl.col("response_time")
                     .filter(pl.col("outcome")=="hit")
                     .median()
                     .alias("hit_median_response_time")
                    ),
                    (pl.col("rig_response_time").alias("rig_response_times")),
                    (pl.col("rig_response_time")
                     .filter(pl.col("outcome")=="hit")
                     .alias("hit_rig_response_times")),
                    (pl.col("rig_response_time")
                     .filter(pl.col("outcome")=="hit")
                     .median()
                     .alias("hit_median_rig_response_time")
                    ),
                    (pl.col("reaction_time").alias("reaction_times")),
                    (pl.col("reaction_time")
                     .filter(pl.col("outcome")=="hit")
                     .alias("hit_reaction_times")),
                    (pl.col("reaction_time")
                     .filter(pl.col("outcome")=="hit")
                     .median()
                     .alias("hit_median_reaction_time")
                    ),
                    (pl.col("wheel_t")),
                    (pl.col("wheel_pos")),
                    (pl.col("signed_contrast").first()),
                    (pl.col("opto").first()),
                    (pl.col("stimkey").first()),
                    (pl.col("stim_label").first()),
                 ]
             )
        )
        
        if do_sort:
            q = q.sort(group_by)
        return q
    
    def set_hit_rates(self,p_method:str="barnard") -> None:
        """ Sets the hit rates for each condition based on binomial distribution of hit count,
        Needs data to be grouped first"""
        # hit rates
        self.grouped_data = self.grouped_data.with_columns((pl.col("hit_count") / pl.col("count")).alias("hit_rate"))
        self.grouped_data = self.grouped_data.with_columns(
            (
                1.96
                * np.sqrt(
                    (pl.col("hit_rate") * (1.0 - pl.col("hit_rate"))) / pl.col("count")
                )
            ).alias("confs")
        )
        
        # p-values
        p_vals = [None] 
        for filt_tup in make_subsets(self.grouped_data,["stim_type", "contrast", "stim_side"]):
            _df = filt_tup[-1]
            p = None
            if 0 not in _df["opto_pattern"]:
                # print("CAN'T DO P-VALUE ANALYSIS, MISSING OPTO COMPONENTS!! RETURNING AN EMPTY DATAFRAME")
                pass
                
            if len(_df):
                table = _df[:,["hit_count", "miss_count"]].to_numpy()
                if table.shape == (2, 2) and not np.any(np.isnan(table)):  
                    # all elements are filled
                    if p_method == "barnard":
                        res = barnard_exact(table, alternative="two-sided")
                    elif p_method == "boschloo":
                        res = boschloo_exact(table, alternative="two-sided")
                    elif p_method == "fischer":
                        res = fisher_exact(table, alternative="two-sided")
                    p = res.pvalue
                    p_vals.extend([None,p])
                else:
                    p_vals.extend([p]*len(_df))
            else:
                p_vals.extend([p]*len(_df))

        #p values are ordered because make_subsets sorts the dataframe and then runs through it
        self.grouped_data = self.grouped_data.with_columns(pl.Series("p_hit_rate",p_vals))
    
    @staticmethod
    def get_pvalues_nonparametric(x1, x2, method: str = "mannu") -> dict:
        """Returns the significance value of two distributions"""
        if method not in ["mannu", "wilcoxon"]:
            raise ValueError(
                f"{method} not a valid statistical test yet, try mannu or wilcoxon"
            )
        if method == "mannu":
            _, p = mannwhitneyu(x1, x2)
        elif method == "wilcoxon":
            res = wilcoxon(x1, x2)
            p = res.p
        return p
        
    