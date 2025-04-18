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
from ..wheelGroupedAggregator import (
    WheelGroupedAggregator,
    bootstrap_confidence_interval,
)


class WheelDiscriminationGroupedAggregator(WheelGroupedAggregator):
    def __init__(self):
        super().__init__()
        self.set_outcomes(["correct", "incorrect"])

    def group_data(self, group_by, do_sort=True):
        super().group_data(group_by, do_sort)
        q = self.data.group_by(group_by).agg(
            [
                pl.col("right_choice")
                .filter(right_choice=1)
                .len()
                .alias("right_choice_count"),
                pl.col("right_choice")
                .filter(right_choice=0)
                .len()
                .alias("left_choice_count"),
            ]
            + [
                (pl.col("wheel_t")),
                (pl.col("wheel_pos")),
            ]
        )

        # calculate confidence intervals of each columns that has "time" in it
        time_cols = [
            c
            for c in q.columns
            if "time" in c and "median" not in c and "confs" not in c
        ]
        for t_c in time_cols:
            _temp_ci = []
            for v in q[t_c].to_list():
                v = [i for i in v if i is not None]  # drop the nulls
                med, ci_p, ci_n = bootstrap_confidence_interval(v, statistic=np.median)
                _temp_ci.append([ci_p, ci_n])

            q = q.with_columns(pl.Series(f"median_{t_c}_confs", _temp_ci))

        if do_sort:
            q = q.sort(group_by)

        self.grouped_data = self.grouped_data.join(
            q, on=group_by, how="full", join_nulls=True
        )

    def calculate_hit_rate():
        pass

    def calculate_proportion(self) -> None:
        """Sets the hit rates and confidence intervals for each condition based on binomial distribution of hit count,
        Needs data to be grouped first
        """

        wanted_side_count = (
            self.grouped_data["right_choice_count"].to_numpy().reshape(-1, 1)
        )
        other_side_count = (
            self.grouped_data["left_choice_count"].to_numpy().reshape(-1, 1)
        )

        conf_upper, hr, conf_lower = self.confidence95(
            wanted_side_count, other_side_count
        )
        confs = np.hstack((conf_lower, conf_upper))

        self.grouped_data = self.grouped_data.with_columns(
            [
                pl.Series("right_choice_prob", hr.flatten()),
                pl.Series("right_choice_confs", confs),
            ]
        )

    def calculate_opto_pvalues(
        self, p_method: Literal["barnard", "boschloo", "fischer"] = "barnard"
    ) -> None:
        """Calculates the statistical significance between opto and non-opto trials

        Args:
            p_method (Literal["barnard", "boschloo", "fischer"], optional): Method to calculate the p-value. Defaults to "barnard".

        Raises:
            ValueError: Missing opto_pattern in previously grouped data
        """
        # use the same as group_by but remove opto_pattern
        if "opto_pattern" not in self.group_by:
            raise ValueError(
                "Currently this method is specialized for opto comparison, and opto-pattern is not present in instance group_by values"
            )
        else:
            p_group = [g_n for g_n in self.group_by if g_n != "opto_pattern"]

        early_row_cnt = self.grouped_data["stim_type"].null_count()
        non_early = self.grouped_data.filter(pl.col("stim_type").is_not_null())
        p_max_width = non_early["opto_pattern"].n_unique()
        p_vals = np.ones((early_row_cnt, p_max_width)) * -1  # first is always early
        for filt_tup in make_subsets(non_early, p_group):
            _df = filt_tup[-1]
            if _df["opto_pattern"].n_unique() == 1:
                # print("CAN'T DO P-VALUE ANALYSIS, MISSING OPTO COMPONENTS!! RETURNING AN EMPTY DATAFRAME")
                p_vals = np.vstack((p_vals, np.ones((len(_df), p_max_width)) * -1))
                continue

            if len(_df):
                curr_p = (
                    np.ones((len(_df), p_max_width)) * -1
                )  # init all p-values with -1

                for i, j in list(
                    itertools.combinations([x for x in range(len(_df))], 2)
                ):
                    table = np.vstack(
                        (
                            _df[
                                i, ["right_choice_count", "left_choice_count"]
                            ].to_numpy(),
                            _df[
                                j, ["right_choice_count", "left_choice_count"]
                            ].to_numpy(),
                        )
                    )

                    if table.shape == (2, 2) and not np.any(np.isnan(table)):
                        # all elements are filled
                        if p_method == "barnard":
                            res = barnard_exact(table, alternative="two-sided")
                        elif p_method == "boschloo":
                            res = boschloo_exact(table, alternative="two-sided")
                        elif p_method == "fischer":
                            res = fisher_exact(table, alternative="two-sided")
                        curr_p[i, j] = res.pvalue
                        curr_p[j, i] = res.pvalue

                    else:
                        curr_p = np.ones_like(table) * -1
                        # p_vals.extend([p] * len(_df))

                p_vals = np.vstack((p_vals, curr_p))

        assert len(p_vals) == len(self.grouped_data)

        # p values are ordered because make_subsets sorts the dataframe and then runs through it
        self.grouped_data = self.grouped_data.with_columns(
            pl.Series("p_right_choice_prob", p_vals)
        )
