import polars as pl
import numpy as np
from scipy.stats import (
    mannwhitneyu,
    wilcoxon,
    fisher_exact,
    barnard_exact,
    boschloo_exact,
)
from ...core.io import display


class DetectionAnalysis:
    def __init__(self, data: pl.DataFrame = None) -> None:
        if data is not None:
            self.set_data(data)

    def set_data(self, data: pl.DataFrame) -> None:
        self.data = data
        self.agg_data = self.make_agg_data()

    def make_agg_data(self) -> pl.DataFrame:
        """Gets the hit rates, counts and confidence intervals for each contrast for each side"""

        if "transformed_response_times" not in self.data.columns:
            self.data = self.data.with_columns(
                pl.col("rig_reaction_time").alias("transformed_response_times")
            )

        q = (
            self.data.lazy()
            .group_by(["stim_type", "contrast", "stim_side", "opto_pattern"])
            .agg(
                [
                    (pl.col("stim_pos").first()),
                    pl.count().alias("count"),
                    (pl.col("outcome") == 1).sum().alias("correct_count"),
                    (pl.col("outcome") == 0).sum().alias("miss_count"),
                    (pl.col("response_latency").alias("response_times")),
                    (pl.col("pos_reaction_time").alias("pos_reaction_time")),
                    (pl.col("speed_reaction_time").alias("speed_reaction_time")),
                    (pl.col("rig_reaction_time").alias("rig_reaction_time")),
                    (
                        pl.col("transformed_response_times").alias(
                            "transformed_response_times"
                        )
                    ),
                    (
                        pl.when(pl.col("outcome") == 1)
                        .then(pl.col("transformed_response_times"))
                        .alias("_temp_transformed")
                    ),
                    (
                        pl.col("rig_reaction_time")
                        .median()
                        .alias("median_rig_reaction_time")
                    ),
                    (
                        pl.when(pl.col("outcome") == 1)
                        .then(pl.col("rig_reaction_time"))
                        .alias("_temp_rig_reaction")
                    ),
                    (
                        pl.col("transformed_response_times")
                        .median()
                        .alias("median_transformed_response_times")
                    ),
                    (pl.col("response_latency").median().alias("median_response_time")),
                    (
                        pl.when(pl.col("outcome") == 1)
                        .then(pl.col("response_latency"))
                        .alias("_temp_response_latency")
                    ),
                    (pl.col("wheel_time")),
                    (pl.col("signed_contrast").first()),
                    (pl.col("wheel_pos")),
                    (pl.col("opto").first()),
                    (pl.col("stimkey").first()),
                    (pl.col("stim_label").first()),
                ]
            )
            .sort(["stim_type", "contrast", "stim_side", "opto_pattern"])
        )

        q = q.with_columns((pl.col("correct_count") / pl.col("count")).alias("hit_rate"))
        q = q.with_columns(
            (
                1.96
                * np.sqrt(
                    (pl.col("hit_rate") * (1.0 - pl.col("hit_rate"))) / pl.col("count")
                )
            ).alias("confs")
        )
        q = q.with_columns(
            [
                pl.col("_temp_rig_reaction")
                .list.median()
                .alias("median_rig_reaction_time_correct"),
                pl.col("_temp_response_latency")
                .list.median()
                .alias("median_response_latency_correct"),
                pl.col("_temp_transformed")
                .list.median()
                .alias("median_transformed_response_times_correct"),
            ]
        )
        q = q.drop(["_temp_rig_reaction", "_temp_response_latency", "_temp_transformed"])

        # reorder stim_label to last column
        cols = q.collect_schema().names()
        move_cols = ["stimkey", "stim_label"]
        for to_del in move_cols:
            _del = cols.index(to_del)
            del cols[_del]
        cols.extend(move_cols)
        q = q.select(cols)

        df = q.collect()
        return df

    def get_deltahits(self) -> pl.DataFrame:
        """Return the delta between the hitrates of a given contrast"""

        b = self.agg_data.filter(
            (pl.col("contrast") == 0) & (pl.col("opto_pattern") == -1)
        ).sum()
        baseline_hr = b[0, "correct_count"] / b[0, "count"]
        baseline_hr_conf = 1.96 * np.sqrt(
            (baseline_hr * (1.0 - baseline_hr)) / b[0, "count"]
        )

        # TODO:ASSUMES ONE OPTO TARGET FOR NOW
        df = (
            self.agg_data.sort("opto_pattern")
            .group_by(["stim_type", "contrast", "stim_side"])
            .agg(
                [
                    (pl.col("opto_pattern")),
                    (pl.col("hit_rate").first()).alias("nonopto_HR"),
                    (pl.col("hit_rate").last()).alias("opto_HR"),
                    (pl.col("confs").first()).alias("nonopto_HR_err"),
                    (pl.col("confs").last()).alias("opto_HR_err"),
                    (
                        pl.col("median_transformed_response_times")
                        .first()
                        .alias("nonopto_RT")
                    ),
                    (pl.col("median_transformed_response_times").last().alias("opto_RT")),
                    (
                        pl.col("median_transformed_response_times_correct")
                        .first()
                        .alias("nonopto_RT_correct")
                    ),
                    (
                        pl.col("median_transformed_response_times_correct")
                        .last()
                        .alias("opto_RT_correct")
                    ),
                    (pl.col("hit_rate").first() - pl.col("hit_rate").last()).alias(
                        "delta_HR"
                    ),
                    (
                        (
                            pl.col("confs").first() ** 2 + pl.col("confs").last() ** 2
                        ).sqrt()
                    ).alias("delta_HR_err"),
                    (
                        pl.col("median_transformed_response_times").last()
                        - pl.col("median_transformed_response_times").first()
                    ).alias("delta_RT"),
                    (
                        pl.col("median_transformed_response_times_correct").last()
                        - pl.col("median_transformed_response_times_correct").first()
                    ).alias("delta_RT_correct"),
                ]
            )
            .sort(["stim_type", "contrast", "stim_side"])
        )

        # ΔHR as a percentage of nonopto HR (suppression index)
        df = df.with_columns(
            (pl.col("delta_HR") / (pl.col("opto_HR"))).alias("percent_delta_HR")
        )
        df = df.with_columns(
            (
                pl.col("percent_delta_HR")
                * (
                    (pl.col("delta_HR_err") / pl.col("delta_HR")) ** 2
                    + (pl.col("nonopto_HR_err") / pl.col("nonopto_HR")) ** 2
                ).sqrt()
            ).alias("percent_delta_HR_err")
        )

        # ΔHR as a percentage of nonopto HR - baseline (suppression index)
        df = df.with_columns(
            (pl.col("delta_HR") / (pl.col("opto_HR") - baseline_hr)).alias("SI")
        )
        df = df.with_columns(
            (
                pl.col("SI")
                * (
                    (pl.col("delta_HR_err") / pl.col("delta_HR")) ** 2
                    + (
                        ((pl.col("nonopto_HR_err") ** 2 + baseline_hr_conf**2).sqrt())
                        / (pl.col("nonopto_HR") - baseline_hr)
                    )
                ).sqrt()
            ).alias("SI_err")
        )

        # # wtf is this?
        # df = df.with_columns(
        #     [
        #         (
        #             (
        #                 pl.col("delta_HR") / pl.col("opto_HR")
        #                 - baseline_hr / pl.col("opto_HR")
        #             )
        #             / (
        #                 pl.col("delta_HR") / pl.col("opto_HR")
        #                 + baseline_hr / pl.col("opto_HR")
        #             )
        #         ).alias("BN_delta_HR"),
        #         pl.lit(baseline_hr).alias("baseline"),
        #     ]
        # )

        return df

    def get_delta_reaction_time(self) -> pl.DataFrame:
        """ """

    def get_baseline_normalized_suppression_index(self) -> pl.DataFrame:
        """Returns the baseline normalized suppression index, SI = (HRnonopto-HRopto)/(HRnonopto-HRopto)"""
        b = self.agg_data.filter(
            (pl.col("contrast") == 0) & (pl.col("opto_pattern") == -1)
        ).sum()
        baseline_hr = b[0, "correct_count"] / b[0, "count"]
        baseline_normalized = self.agg_data.filter(pl.col("contrast") != 0).with_columns(
            (pl.col("hit_rate") - baseline_hr).alias("norm_hit_rate")
        )

        q = (
            self.agg_data.lazy()
            .sort("opto_pattern")
            .group_by(["stim_type", "contrast", "stim_side"])
            .agg(
                [
                    (pl.col("hit_rate").first() - pl.col("hit_rate").last()).alias(
                        "delta_HR"
                    ),
                    (
                        (pl.col("norm_hit_rate").first() - baseline_hr)
                        / (pl.col("norm_hit_rate").last() - baseline_hr)
                    ).alias("norm_SI"),
                    pl.col("count").sum(),
                ]
            )
            .sort(["stim_type", "contrast", "stim_side"])
        )

        df = q.collect()
        return df

    def get_hitrate_pvalues_exact(
        self, side: str = "contra", method: str = "barnard"
    ) -> pl.DataFrame:
        """Calculates the p-values for each contrast in different stim datasets"""
        q = (
            self.agg_data.lazy()
            .group_by(["stim_type", "contrast", "stim_side", "opto_pattern"])
            .agg(
                [
                    (pl.col("correct_count").first()),
                    (pl.col("miss_count").first()),
                    (pl.col("stimkey").first()),
                    (pl.col("stim_label").first()),
                ]
            )
            .sort(["stim_type", "contrast", "stim_side"])
        )

        temp = (
            q.filter((pl.col("stim_side") == side))
            .sort(["stim_type", "contrast", "opto_pattern"])
            .collect()
        )

        stimlabel_list = temp["stim_type"].unique().to_numpy()
        c_list = temp["contrast"].unique().to_numpy()
        o_list = temp["opto_pattern"].unique().sort().to_numpy()
        # there should be 2 entries(opto-nonopto) per stim_type and contrast combination
        # otherwise cannot create the table to do p_value analyses
        # if len(temp) != (len(c_list)*len(stimlabel_list)*2):
        if 0 not in o_list:  # no opto_pattern
            display(
                f"CAN'T DO P-VALUE ANALYSIS on {side}, MISSING OPTO COMPONENTS!! RETURNING AN EMPTY DATAFRAME"
            )
            df = pl.DataFrame()
        else:
            stims = []
            contrasts = []
            p_values = []
            stimkeys = []
            sides = []
            for s in stimlabel_list:
                for c in c_list:
                    filt = temp.filter(
                        (pl.col("contrast") == c) & (pl.col("stim_type") == s)
                    )
                    if len(filt):
                        #
                        for i, _ in enumerate(range(len(filt) - 1), start=1):
                            sub_filt = pl.concat([filt[0], filt[i]])

                            table = sub_filt[
                                :, ["correct_count", "miss_count"]
                            ].to_numpy()
                            if table.shape == (2, 2) and np.all(
                                np.isnan(table) == False
                            ):  # all elements are filled
                                if method == "barnard":
                                    res = barnard_exact(table, alternative="two-sided")
                                elif method == "boschloo":
                                    res = boschloo_exact(table, alternative="two-sided")
                                elif method == "fischer":
                                    res = fisher_exact(table, alternative="two-sided")
                                p = res.pvalue
                                s_k = sub_filt[1, "stimkey"]
                            else:
                                p = np.nan
                                s_k = None

                            stims.append(s)
                            contrasts.append(c)
                            p_values.append(p)
                            stimkeys.append(s_k)
                            sides.append(side)

            df = pl.DataFrame(
                {
                    "stim_type": stims,
                    "stim_side": sides,
                    "contrast": contrasts,
                    "p_values": p_values,
                    "stimkey": stimkeys,
                }
            )
        return df

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
