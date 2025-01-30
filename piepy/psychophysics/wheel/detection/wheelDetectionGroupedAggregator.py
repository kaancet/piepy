import itertools
import numpy as np
import polars as pl
from typing import Literal
from scipy.stats import (
    fisher_exact,
    barnard_exact,
    boschloo_exact,
)

from ....core.data_functions import make_subsets
from ..wheelGroupedAggregator import WheelGroupedAggregator


class WheelDetectionGroupedAggregator(WheelGroupedAggregator):
    def __init__(self):
        super().__init__()
        self.set_outcomes(["hit","miss"])
    
    def calculate_hit_rates(self,) -> None:
        """Sets the hit rates for each condition based on binomial distribution of hit count,
        Needs data to be grouped first
        
        Args:
            p_method: method to calculate the p-value
        """
        # hit rates
        self.grouped_data = self.grouped_data.with_columns(
            (pl.col("hit_count") / pl.col("count")).alias("hit_rate")
        )
        self.grouped_data = self.grouped_data.with_columns(
            (
                1.96
                * np.sqrt(
                    (pl.col("hit_rate") * (1.0 - pl.col("hit_rate"))) / pl.col("count")
                )
            ).alias("confs")
        )
        
    def calculate_opto_pvalues(self, p_method: Literal["barnard", "boschloo", "fischer"] = "barnard") -> None:
        """ 
        
        Args:
            p_method:
        """
        # use the same as group_by but remove opto_pattern
        if "opto_pattern" not in self.group_by:
            raise ValueError("")
        else:
            p_group = [g_n for g_n in self.group_by if g_n != "opto_pattern"]
        
        early_row_cnt = self.grouped_data["stim_type"].null_count()
        non_early = self.grouped_data.filter(pl.col("stim_type").is_not_null())
        p_max_width = non_early["opto_pattern"].n_unique()
        p_vals = np.ones((early_row_cnt,p_max_width)) * -1 # first is always early
        for filt_tup in make_subsets(non_early, p_group):
            _df = filt_tup[-1]
            if _df["opto_pattern"].n_unique() == 1:
                # print("CAN'T DO P-VALUE ANALYSIS, MISSING OPTO COMPONENTS!! RETURNING AN EMPTY DATAFRAME")                
                p_vals = np.vstack((p_vals,np.ones((len(_df),p_max_width)) * -1))
                continue
            
            if len(_df):
                curr_p = np.ones((len(_df),p_max_width)) * -1 #init all p-values with -1

                for i,j in list(itertools.combinations([x for x in range(len(_df))], 2)):
                    table = np.vstack((_df[i, ["hit_count", "miss_count"]].to_numpy(),
                                       _df[j, ["hit_count", "miss_count"]].to_numpy()))
                    
                    if table.shape == (2, 2) and not np.any(np.isnan(table)):
                        # all elements are filled
                        if p_method == "barnard":
                            res = barnard_exact(table, alternative="two-sided")
                        elif p_method == "boschloo":
                            res = boschloo_exact(table, alternative="two-sided")
                        elif p_method == "fischer":
                            res = fisher_exact(table, alternative="two-sided")
                        curr_p[i,j] = res.pvalue
                        curr_p[j,i] = res.pvalue
                        
                    else:
                        curr_p = np.ones_like(table) * -1
                        # p_vals.extend([p] * len(_df))
                    
                p_vals = np.vstack((p_vals,curr_p))

        assert len(p_vals) == len(self.grouped_data)

        # p values are ordered because make_subsets sorts the dataframe and then runs through it
        self.grouped_data = self.grouped_data.with_columns(
            pl.Series("p_hit_rate", p_vals)
        )
