from ..basePlotters import *
from ...psychophysics.detection.wheelDetectionSilencing import *
from ...psychophysics.detection.wheelDetectionAnalysis import *
import pingouin as pg


# constant animal colors
ANIMAL_COLORS = {
    "KC139": "#332288",
    "KC141": "#117733",
    "KC142": "#DDCC77",
    "KC143": "#AA4499",
    "KC144": "#882255",
    "KC145": "#88CCEE",
    "KC146": "#275D6D",
    "KC147": "#F57A6C",
    "KC148": "#ADFA9A",
    "KC149": "#A45414",
    "KC150": "#0000FF",
    "KC151": "#00FF11",
    "KC152": "#FFAA33",
}

# values from allen ccf_2022, using the shoelace algorithm to calculate the area from area boundary coordinates
AREA_SIZE = {
    "V1": 4.002,
    "HVA": 2.925,
    "dorsal": 1.428,
    "ventralPM": 1.496,
    "LM": 0.571,
    "AL": 0.389,
    "RL": 0.583,
    "PM": 0.719,
    "AM": 0.456,
}
# "LI":0.207
# "cortex" : 6.927,

# hierarchy scores from steinmetz?
AREA_SCORE = {}


def group_data(
    self, data: pl.DataFrame, group_by: list, sort_by_group: bool = True
) -> pl.DataFrame:
    """ """

    # check if group_by names are in the dataframe columns
    for c in group_by:
        if c not in data.columns:
            raise ValueError(
                f"{c} not a valid column name in given DataFrame, try one of: {data.columns}"
            )

    q = (
        data.lazy()
        .group_by(group_by)
        .agg(
            [
                (pl.col("outcome") != 1).sum().alias("trial_count"),
                (pl.col("outcome") == 1).sum().alias("correct_count"),
                (pl.col("outcome") == 0).sum().alias("miss_count"),
                (
                    pl.when(pl.col("outcome") == 1)
                    .then(pl.col("reaction_times"))
                    .alias("_temp_reaction_times")
                ),
                (pl.col("session_no").first()),
                (pl.col("stimkey").first()),
                (pl.col("stim_label").first()),
            ]
        )
        .drop_nulls()
    )

    if sort_by_group:
        q = q.sort(group_by)

    # reorder stim_label to last column
    cols = q.collect_schema().names()
    move_cols = ["stimkey", "stim_label"]
    for to_del in move_cols:
        _del = cols.index(to_del)
        del cols[_del]
    cols.extend(move_cols)
    q = q.select(cols)


class ComparisonLinePlotter:
    """This plotter expects data of a single experiment type:
    multiple animals, same area, same stim count(type)"""

    def __init__(self, data) -> None:
        self.data = data  # get area filtered data

        self.plot_data = self.make_plot_data()

    def make_plot_data(self) -> pl.DataFrame:
        """Returns a dataframe to later loop while plotting
        This automatically groups together two of the same experiment sessions"""

        q = (
            self.data.lazy()
            .group_by(["animalid", "stim_type", "stim_side", "contrast", "opto_pattern"])
            .agg(
                [
                    pl.count().alias("trial_count"),
                    (pl.col("outcome") == 1).sum().alias("correct_count"),
                    (pl.col("outcome") == 0).sum().alias("miss_count"),
                    (
                        pl.when(pl.col("outcome") == 1)
                        .then(pl.col("reaction_time"))
                        .alias("reaction_times_correct")
                    ),
                    (pl.col("reaction_time").alias("reaction_times")),
                    (pl.col("reaction_time").median().alias("median_reaction_time")),
                    (pl.col("opto").first()),
                    (pl.col("session_no").first()),
                    (pl.col("stimkey").first()),
                    (pl.col("stim_label").first()),
                ]
            )
            .drop_nulls()
            .sort(["animalid", "stim_type", "stim_side", "contrast", "opto_pattern"])
        )

        q = q.with_columns(
            pl.col("reaction_times_correct")
            .list.median()
            .alias("median_reaction_time_correct")
        )

        # hit rate and confidence intervals
        q = q.with_columns(
            (100 * pl.col("correct_count") / pl.col("trial_count")).alias("hit_rate")
        )
        q = q.with_columns(
            (
                1.96
                * np.sqrt(
                    (pl.col("hit_rate") * (100.0 - pl.col("hit_rate")))
                    / pl.col("trial_count")
                )
            ).alias("confs")
        )

        q = q.with_columns(
            pl.when(pl.col("stim_side") == "ipsi")
            .then((pl.col("contrast") * -1))
            .otherwise(pl.col("contrast"))
            .alias("signed_contrast")
        )

        # add contrast difficulty columns
        q = q.with_columns(
            pl.when((pl.col("contrast") > 0) & (pl.col("contrast") < 25))
            .then(pl.lit("hard"))
            .when(pl.col("contrast") > 25)
            .then(pl.lit("easy"))
            .otherwise(pl.lit("catch"))
            .alias("contrast_difficulty")
        )

        # reorder stim_label to last column
        cols = q.collect_schema().names()
        del cols[-7]
        del cols[-6]
        cols.extend(["stimkey", "stim_label"])
        q = q.select(cols)
        df = q.collect()
        return df

    def _plot_sessions_and_avg_hit_rate_(
        self, ax: plt.Axes, filtered_data: pl.DataFrame, **kwargs
    ) -> plt.Axes:
        uniq_sesh = filtered_data["session_no"].unique().to_numpy()
        for u_s in uniq_sesh:
            sesh_df = filtered_data.filter(pl.col("session_no") == u_s)

            if len(sesh_df["opto_pattern"].unique().to_list()) == 1:
                if sesh_df[0, "stim_side"] == "catch":
                    hr = (
                        100
                        * np.sum(sesh_df["correct_count"].to_numpy())
                        / np.sum(sesh_df["trial_count"].to_numpy())
                    )
                else:
                    hr = sesh_df["hit_rate"].to_list()
            else:
                hr = sesh_df["hit_rate"].to_list()

            ax.plot(
                sesh_df["opto_pattern"].unique().sort().to_list(),
                hr,
                marker="o",
                markersize=20,
                markeredgewidth=0,
                linewidth=kwargs.get("linewidth", 3),
                c=ANIMAL_COLORS[sesh_df[0, "animalid"]],
                alpha=0.5,
                label=sesh_df[0, "animalid"],
                zorder=2,
            )

        avg_df = (
            filtered_data.group_by(["opto_pattern"])
            .agg(
                [
                    pl.col("trial_count").sum(),
                    pl.col("correct_count").sum(),
                    pl.col("miss_count").sum(),
                    pl.count().alias("animal_count"),
                    pl.col("hit_rate").mean().alias("avg_hitrate"),
                    pl.col("hit_rate"),
                ]
            )
            .sort(["opto_pattern"])
        )
        avg_df = avg_df.with_columns(
            pl.col("hit_rate")
            .map_elements(lambda x: stats.sem(x), return_dtype=float)
            .alias("animal_confs")
        )

        ax.errorbar(
            avg_df["opto_pattern"].to_list(),
            avg_df["avg_hitrate"].to_list(),
            avg_df["animal_confs"].to_list(),
            marker="o",
            markersize=20,
            linewidth=kwargs.get("linewidth", 3) * 2,
            c="k",
            zorder=2,
        )

        return ax

    def plot_stim_type_hit_rates(
        self, stim_side: str = "contra", doP: bool = True, **kwargs
    ) -> plt.figure:
        """ """
        fontsize = kwargs.pop("fontsize", 30)
        linewidth = kwargs.pop("linewidth", 3)
        x_axis_dict = {-1: "Non\nOpto", 0: "Opto", 1: "Opto\nOff"}

        uniq_contrast = self.plot_data["contrast"].unique().sort().to_numpy()
        uniq_contrast = [i for i in uniq_contrast if i not in [100, 6.25]]
        uniq_pattern = self.plot_data["opto_pattern"].unique().sort().to_list()

        n_contrast = len(uniq_contrast)
        uniq_val = self.plot_data["stim_type"].unique().sort().to_numpy()
        n_val = len(uniq_val)

        self.fig, axes = plt.subplots(
            ncols=1 + (n_val * (n_contrast - 1)),
            nrows=1,
            figsize=kwargs.pop("figsize", (12, 10)),
        )

        self.p_values_hit_rate = {}

        for j, c in enumerate(uniq_contrast):
            contrast_df = self.plot_data.filter((pl.col("contrast") == c))
            self.p_values_hit_rate[c] = {}

            if c == 0:
                stim_df = contrast_df.clone()
                base_df = stim_df.filter(pl.col("opto") == 0)
                baseline_avg = np.mean(base_df["hit_rate"].to_numpy())
                baseline_sem = stats.sem(base_df["hit_rate"].to_numpy())
                stim_df = stim_df.filter(pl.col("stim_side") == "catch")
                ax = self._plot_sessions_and_avg_hit_rate_(axes[0], stim_df)

                ax.set_ylim([0, 105])
                ax.set_yticks([0, 25, 50, 75, 100])
                ax.set_xticks(uniq_pattern)
                ax.set_xticklabels([x_axis_dict[i] for i in uniq_pattern])
                ax.set_title("Gray Screen", fontsize=fontsize - 5)

                ax.set_xlabel(f"c={c}", fontsize=fontsize, labelpad=10)

                ax.tick_params(
                    axis="x",
                    labelsize=fontsize - 5,
                    length=10,
                    width=linewidth,
                    which="major",
                    color="k",
                )
                ax.grid(True, axis="y", alpha=0.4)
                ax.set_ylabel("Hit Rate", fontsize=fontsize)
                ax.tick_params(
                    axis="y",
                    labelsize=fontsize,
                    length=10,
                    width=linewidth,
                    which="major",
                    color="k",
                )

                ax.spines["bottom"].set_position(("outward", 20))
                ax.spines["bottom"].set_linewidth(2)
                ax.spines["left"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)
                continue

            for i, k in enumerate(uniq_val):

                stim_df = contrast_df.filter(pl.col("stim_type") == k)
                ax_i = i
                if j == 2:
                    ax_i = i + (len(uniq_val) - 1)
                ax = axes[j + ax_i]
                stim_df = stim_df.filter(pl.col("stim_side") == stim_side)
                ax = self._plot_sessions_and_avg_hit_rate_(ax, stim_df)

                ax.axhline(y=baseline_avg, linestyle=":", c="k", alpha=0.4, zorder=1)
                ax.axhspan(
                    baseline_avg + baseline_sem,
                    baseline_avg - baseline_sem,
                    color="gray",
                    alpha=0.05,
                    linewidth=0,
                    zorder=1,
                )

                if doP:
                    # do p-values with mann-whitney-u
                    non_opto = stim_df.filter(pl.col("opto_pattern") == -1)[
                        "hit_rate"
                    ].to_numpy()
                    opto = stim_df.filter(pl.col("opto_pattern") == 0)[
                        "hit_rate"
                    ].to_numpy()
                    if len(opto) != 0:
                        # _,p = mannwhitneyu(non_opto,opto)
                        # _, p = mannwhitneyu(non_opto, opto)
                        p_stats = pg.wilcoxon(non_opto, opto, alternative="two-sided")
                        p = p_stats["p-val"].iloc[0]
                        self.p_values_hit_rate[c][k] = p

                        stars = ""
                        if p < 0.001:
                            stars = "***"
                        elif 0.001 < p < 0.01:
                            stars = "**"
                        elif 0.01 < p < 0.05:
                            stars = "*"
                        ax.text(-0.6, 101, stars, color="k", fontsize=30)

                ax.set_ylim([0, 105])
                ax.set_yticks([0, 25, 50, 75, 100])
                _x_ticks = stim_df["opto_pattern"].unique().sort().to_list()
                ax.set_xticks(_x_ticks)
                ax.set_xticklabels([x_axis_dict[i] for i in _x_ticks])
                ax.set_title(k, fontsize=fontsize - 5, pad=20)

                ax.set_xlabel(f"c={c}", fontsize=fontsize, labelpad=10)

                ax.tick_params(
                    axis="x",
                    labelsize=fontsize - 5,
                    length=10,
                    width=linewidth,
                    which="major",
                    color="k",
                )
                ax.grid(True, axis="y", alpha=0.4)
                ax.tick_params(
                    axis="y", labelsize=0, length=0, width=0, which="major", color="k"
                )

                ax.spines["bottom"].set_position(("outward", 20))
                ax.spines["bottom"].set_linewidth(2)
                ax.spines["left"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)

        ax.legend(
            loc="center left", bbox_to_anchor=(1, 0.5), fontsize=fontsize, frameon=False
        )
        return ax

    def plot_opto_target_hit_rates(
        self, stim_side: str = "contra", doP: bool = True, **kwargs
    ):
        fontsize = kwargs.pop("fontsize", 30)
        linewidth = kwargs.pop("linewidth", 3)
        x_axis_dict = {-1: "Non\nOpto", 0: "Opto\nOn Target", 1: "Opto\nOff Target"}

        uniq_contrast = self.plot_data["contrast"].unique().sort().to_numpy()
        uniq_contrast = [i for i in uniq_contrast if i not in [100, 6.25]]
        uniq_pattern = self.plot_data["opto_pattern"].unique().sort().to_list()
        n_contrast = len(uniq_contrast)

        self.fig, axes = plt.subplots(
            ncols=n_contrast,
            nrows=1,
            constrained_layout=True,
            figsize=kwargs.pop("figsize", (12, 10)),
        )
        self.p_values_hit_rate = {}
        for j, c in enumerate(uniq_contrast):
            contrast_df = self.plot_data.filter((pl.col("contrast") == c))
            self.p_values_hit_rate[c] = {}
            if c != 0:
                contrast_df = contrast_df.filter((pl.col("stim_side") == stim_side))

            ax = axes[j]

            ax = self._plot_sessions_and_avg_hit_rate_(ax, contrast_df)

            if doP:
                # do p-values with mann-whitney-u
                non_opto = contrast_df.filter((pl.col("opto_pattern") == -1))[
                    "hit_rate"
                ].to_numpy()
                opto_ontarget = contrast_df.filter((pl.col("opto_pattern") == 0))[
                    "hit_rate"
                ].to_numpy()
                opto_offtarget = contrast_df.filter((pl.col("opto_pattern") == 1))[
                    "hit_rate"
                ].to_numpy()
                if len(opto_ontarget) != 0 and len(opto_offtarget) != 0:
                    _, self.p_values_hit_rate[c]["ontarget"] = mannwhitneyu(
                        non_opto, opto_ontarget
                    )
                    _, self.p_values_hit_rate[c]["offtarget"] = mannwhitneyu(
                        non_opto, opto_offtarget
                    )

                    for k, p in self.p_values_hit_rate[c].items():
                        stars = ""
                        linewidth = 0
                        if p < 0.001:
                            linewidth = 2
                            stars = "***"
                        elif 0.001 < p < 0.01:
                            linewidth = 2
                            stars = "**"
                        elif 0.01 < p < 0.05:
                            linewidth = 2
                            stars = "*"
                        if k == "ontarget":
                            star_xpos = -0.6
                            star_ypos = 106
                            line_end = 0
                            line_y = 103
                        else:
                            star_xpos = -0.1
                            star_ypos = 112
                            line_end = 1
                            line_y = 109
                        ax.hlines(line_y, -1, line_end, linewidth=linewidth, color="k")
                        ax.text(star_xpos, star_ypos, stars, color="k", fontsize=30)

            ax.set_ylim([0, 115])
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.set_xticks(uniq_pattern)
            ax.set_xticklabels([x_axis_dict[i] for i in uniq_pattern])
            # ax.set_title(k,fontsize=fontsize-5, pad=20)

            ax.set_xlabel(f"c={c}", fontsize=fontsize, labelpad=10)

            ax.tick_params(
                axis="x",
                labelsize=fontsize - 5,
                length=10,
                width=linewidth,
                which="major",
                color="k",
            )
            ax.grid(True, axis="y", alpha=0.4)
            ax.tick_params(
                axis="y", labelsize=0, length=0, width=0, which="major", color="k"
            )

            ax.spines["bottom"].set_position(("outward", 20))
            ax.spines["bottom"].set_linewidth(2)
            ax.spines["left"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)

    def _plot_sessions_and_avg_response_time_(
        self,
        ax: plt.Axes,
        filtered_data: pl.DataFrame,
        include_misses: bool = False,
        **kwargs,
    ) -> plt.Axes:

        if include_misses:
            _resp_col = "reaction_times"
        else:
            _resp_col = "reaction_times_correct"

        uniq_sesh = filtered_data["session_no"].unique().to_numpy()
        mean_of_medians = []
        for u_s in uniq_sesh:
            sesh_df = filtered_data.filter(pl.col("session_no") == u_s)

            if len(sesh_df["opto_pattern"].unique().to_list()) == 1:
                tmp = sesh_df[_resp_col].explode().to_numpy()
                median = np.nanmedian(tmp)
            else:
                tmp = sesh_df[_resp_col].to_numpy()
                median = [np.nanmedian(i) for i in tmp]

            mean_of_medians.append(median)

            ax.plot(
                sesh_df["opto_pattern"].unique().sort().to_list(),
                median,
                marker="o",
                markersize=20,
                markeredgewidth=0,
                linewidth=kwargs.get("linewidth", 3),
                c=ANIMAL_COLORS[sesh_df[0, "animalid"]],
                alpha=0.5,
                label=sesh_df[0, "animalid"],
                zorder=2,
            )

        mean_of_medians = np.array(mean_of_medians)
        mean = np.nanmean(mean_of_medians, axis=0)
        conf = stats.sem(mean_of_medians, axis=0, nan_policy="omit")
        _x = filtered_data["opto_pattern"].drop_nulls().unique().sort().to_list()

        ax.errorbar(
            _x,
            mean,
            conf,
            marker="o",
            markersize=20,
            linewidth=kwargs.get("linewidth", 3) * 2,
            c="k",
            zorder=2,
        )

        return ax, mean_of_medians

    def plot_stim_type_response_time(
        self,
        stim_side: str = "contra",
        doP: bool = True,
        include_misses: bool = False,
        **kwargs,
    ) -> plt.figure:

        fontsize = kwargs.pop("fontsize", 30)
        linewidth = kwargs.pop("linewidth", 3)
        x_axis_dict = {-1: "Non\nOpto", 0: "Opto", 1: "Opto\nOff"}

        uniq_contrast = self.plot_data["contrast"].unique().sort().to_numpy()
        uniq_contrast = [i for i in uniq_contrast if i not in [100, 6.25]]
        uniq_pattern = self.plot_data["opto_pattern"].unique().sort().to_list()

        n_contrast = len(uniq_contrast)
        uniq_val = self.plot_data["stim_type"].unique().sort().to_numpy()
        n_val = len(uniq_val)

        self.fig, axes = plt.subplots(
            ncols=1 + (n_val * (n_contrast - 1)),
            nrows=1,
            figsize=kwargs.pop("figsize", (12, 10)),
        )

        self.p_values_resp = {}

        if include_misses:
            _median_col = "median_reaction_time"
        else:
            _median_col = "median_reaction_time_correct"

        for j, c in enumerate(uniq_contrast):
            contrast_df = self.plot_data.filter((pl.col("contrast") == c))
            self.p_values_resp[c] = {}

            if c == 0:
                stim_df = contrast_df.clone()
                stim_df = stim_df.filter(pl.col("stim_side") == "catch")
                ax, _ = self._plot_sessions_and_avg_response_time_(
                    axes[0], stim_df, include_misses=include_misses
                )

                ax.set_yscale("symlog")
                minor_locs = [200, 400, 600, 800, 2000, 4000, 6000, 8000]

                ax.set_ylim([150, 1300])
                ax.set_xticks(uniq_pattern)
                ax.set_xticklabels([x_axis_dict[i] for i in uniq_pattern])
                ax.set_title("Gray Screen", fontsize=fontsize - 5)

                ax.tick_params(
                    axis="x",
                    labelsize=fontsize - 5,
                    length=10,
                    width=linewidth,
                    which="major",
                    color="k",
                )
                ax.grid(True, axis="y", alpha=0.4)
                ax.tick_params(
                    axis="y", labelsize=0, length=0, width=0, which="major", color="k"
                )

                ax.grid(True, axis="y", alpha=0.4, which="minor")
                ax.set_ylabel("Reaction Times (ms)", fontsize=fontsize)
                ax.yaxis.set_minor_locator(plt.FixedLocator(minor_locs))
                ax.yaxis.set_minor_formatter(plt.FormatStrFormatter("%d"))
                ax.yaxis.set_major_locator(ticker.FixedLocator([10, 100, 1000, 10000]))
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
                ax.tick_params(
                    axis="y",
                    labelsize=fontsize,
                    length=10,
                    width=linewidth,
                    which="major",
                    color="k",
                )
                ax.tick_params(
                    axis="y",
                    labelsize=fontsize - 3,
                    length=10,
                    width=linewidth,
                    which="minor",
                    color="k",
                )
                ax.spines["bottom"].set_position(("outward", 20))
                ax.spines["bottom"].set_linewidth(2)
                ax.spines["left"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)
                continue

            for i, k in enumerate(uniq_val):

                stim_df = contrast_df.filter(pl.col("stim_type") == k)
                ax_i = i
                if j == 2:
                    ax_i = i + (len(uniq_val) - 1)
                ax = axes[j + ax_i]
                stim_df = stim_df.filter(pl.col("stim_side") == stim_side)
                ax, cutoff_medians = self._plot_sessions_and_avg_response_time_(
                    ax, stim_df, include_misses
                )

                if doP:
                    # do p-values with mann-whitney-u
                    non_opto = stim_df.filter(pl.col("opto_pattern") == -1)[
                        _median_col
                    ].to_numpy()
                    opto = stim_df.filter(pl.col("opto_pattern") == 0)[
                        _median_col
                    ].to_numpy()
                    if len(opto) != 0:
                        # _, p = mannwhitneyu(non_opto, opto)
                        p_stats = pg.wilcoxon(non_opto, opto, alternative="two-sided")
                        p = p_stats["p-val"].iloc[0]
                        self.p_values_resp[c][k] = p

                        stars = ""
                        if p < 0.001:
                            stars = "***"
                        elif 0.001 < p < 0.01:
                            stars = "**"
                        elif 0.01 < p < 0.05:
                            stars = "*"
                        ax.text(-0.6, 1100, stars, color="k", fontsize=30)

                ax.set_yscale("symlog")
                minor_locs = [200, 400, 600, 800, 2000, 4000, 6000, 8000]

                _x_ticks = stim_df["opto_pattern"].unique().sort().to_list()
                ax.set_xticks(_x_ticks)
                ax.set_xticklabels([x_axis_dict[i] for i in _x_ticks])

                ax.set_ylim([150, 1300])
                _x_ticks = stim_df["opto_pattern"].unique().sort().to_list()
                ax.set_xticks(_x_ticks)
                ax.set_xticklabels([x_axis_dict[i] for i in _x_ticks])
                ax.set_title(k, fontsize=fontsize - 5)

                ax.set_xlabel(f"c={c}", fontsize=fontsize, labelpad=10)
                ax.grid(True, axis="y", alpha=0.4, which="minor")
                ax.tick_params(
                    axis="x",
                    labelsize=fontsize - 5,
                    length=10,
                    width=linewidth,
                    which="major",
                    color="k",
                )
                ax.yaxis.set_minor_locator(plt.FixedLocator(minor_locs))
                ax.yaxis.set_minor_formatter(plt.FormatStrFormatter("%d"))
                ax.tick_params(
                    axis="y", labelsize=0, length=0, width=0, which="major", color="k"
                )
                ax.tick_params(
                    axis="y", labelsize=0, length=0, width=0, which="minor", color="k"
                )

                ax.spines["bottom"].set_position(("outward", 20))
                ax.spines["bottom"].set_linewidth(2)
                ax.spines["left"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)

        ax.legend(
            loc="center left", bbox_to_anchor=(1, 0.5), fontsize=fontsize, frameon=False
        )
        return ax


class AllAreasScatterPlotter:
    def __init__(self, data, **kwargs) -> None:
        set_style(kwargs.pop("style", "presentation"))
        self.plot_data = data
        self.fig = None
        self.area_list = [
            "V1",
            "HVA",
            "dorsal",
            "ventralPM",
            "LM",
            "AL",
            "RL",
            "PM",
            "AM",
        ]
        self.add_area_size()

    def add_area_size(self) -> None:
        """Adds the size of the area as a column"""
        area_list = self.plot_data["area"].to_list()
        size_list = [AREA_SIZE.get(a) for a in area_list]
        self.plot_data = self.plot_data.with_columns(
            pl.Series(name="area_size", values=size_list)
        )

    @staticmethod
    def add_jitter(arr, jitter_lims: list = [-0.2, 0.2]) -> np.ndarray:
        """Adds jitter in x-dimension"""
        arr = np.array(arr)  # polars returns an immutable numpy array, this changes that

        jitter = np.random.choice(
            np.linspace(jitter_lims[0], jitter_lims[1], len(arr)), len(arr), replace=True
        )
        arr = arr + jitter
        return arr

    def plot_contrast_effect_difference(
        self, ax: plt.Axes = None, metric: str = "delta_HR", **kwargs
    ) -> plt.Axes:
        """ """
        if ax is None:
            self.fig = plt.figure(figsize=kwargs.pop("figsize", (10, 10)))
            ax = self.fig.add_subplot(1, 1, 1)

        use_animal_colors = kwargs.pop("color_animals", False)

        analyzer = DetectionAnalysis()

        # order by area size
        if kwargs.pop("order_by_size", False):
            _sorted_dict = {
                k: v
                for k, v in sorted(
                    AREA_SIZE.items(), key=lambda item: item[1], reverse=True
                )
            }
            _area_list = list(_sorted_dict.keys())
        else:
            _area_list = self.area_list

        for i, area in enumerate(_area_list):
            idx = i + 1
            area_df = self.plot_data.filter(pl.col("area") == area)

            uniq_sessions = area_df["session_no"].unique().to_list()
            area_contrast_hr_diff = []
            per_session_errs = []
            colors = []
            for sesh in uniq_sessions:
                sesh_df = area_df.filter(pl.col("session_no") == sesh)

                analyzer.set_data(sesh_df)

                d_hr = analyzer.get_deltahits()
                hit_rate_contra = d_hr.filter(pl.col("stim_side") == "contra")

                if len(hit_rate_contra):
                    hit_rate_contra = hit_rate_contra.sort("contrast")
                    _diff = (
                        hit_rate_contra[0, metric] - hit_rate_contra[1, metric]
                    )  # 12.5 - 50 contrast
                    _norm_diff = _diff / (
                        hit_rate_contra[0, metric] + hit_rate_contra[1, metric]
                    )
                    area_contrast_hr_diff.append(100 * _norm_diff)
                    per_session_errs.append(100 * hit_rate_contra[0, f"{metric}_err"])
                    if use_animal_colors:
                        colors.append(ANIMAL_COLORS[sesh_df[0, "animalid"]])
                    else:
                        colors.append("#424242")

            _x = self.add_jitter([idx] * len(area_contrast_hr_diff))

            ax.scatter(
                _x, area_contrast_hr_diff, s=250, color=colors, alpha=1, linewidths=0
            )

            # do violin
            if kwargs.get("violin_plot", False):
                parts = ax.violinplot(
                    area_contrast_hr_diff,
                    [idx],
                    showmedians=False,
                    showextrema=False,
                    widths=0.5,
                    side="low" if j == 0 else "high",
                )

                for pc in parts["bodies"]:
                    pc.set_facecolor("#424242")
                    pc.set_alpha(0.6)

            quartile1, medians, quartile3 = np.percentile(
                area_contrast_hr_diff, [25, 50, 75]
            )
            ax.scatter(idx, medians, marker="_", color="k", s=100, zorder=3)
            ax.vlines(
                idx,
                quartile1,
                quartile3,
                color="k",
                linestyle="-",
                linewidth=1.5,
            )

        ax.axhline(0, color="k", linestyle=":", linewidth=1)

        fontsize = 20
        ax.set_xlabel("Area", fontsize=fontsize)
        ax.set_ylabel(f"{metric} difference 12.5% - 50%", fontsize=fontsize)
        ax.set_yticks([0, 25, 50, 75, 100])

        ax.tick_params(labelsize=fontsize, length=6)

        ax.set_xticks(np.arange(1, len(_area_list) + 1))
        ax.set_xticklabels(_area_list)
        ax.tick_params(axis="x", rotation=45)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        return ax

    def plot_hit_rates(
        self,
        ax: plt.Axes = None,
        metric: str = "percent_delta_HR",
        contrast: float = -1,
        **kwargs,
    ) -> plt.Axes:
        """metric can be delta_HR, percent_delta_HR, delta_HR_baseline"""

        if ax is None:
            self.fig = plt.figure(figsize=kwargs.pop("figsize", (10, 10)))
            ax = self.fig.add_subplot(1, 1, 1)

        analyzer = DetectionAnalysis()

        # order by area size
        if kwargs.pop("order_by_size", False):
            _sorted_dict = {
                k: v
                for k, v in sorted(
                    AREA_SIZE.items(), key=lambda item: item[1], reverse=True
                )
            }
            _area_list = list(_sorted_dict.keys())
        else:
            _area_list = self.area_list

        for i, area in enumerate(_area_list):
            area_df = self.plot_data.filter(pl.col("area") == area)

            uniq_contrast = area_df["contrast"].drop_nulls().unique().sort().to_numpy()
            uniq_contrast = [c for c in uniq_contrast if c not in [100, 6.25, 0]]

            contrast_axis_shift = [-0.1, 0.1]
            for j, c in enumerate(uniq_contrast):

                # to plot single contrast or both
                if contrast == c:
                    contrast_ind = i
                elif contrast != -1 and contrast != c:
                    continue
                else:
                    contrast_ind = i + contrast_axis_shift[j]

                contrast_df = area_df.filter(
                    (pl.col("contrast") == c) | (pl.col("contrast") == 0)
                )

                uniq_sessions = area_df["session_no"].unique().to_list()
                area_hrs = []
                per_session_errs = []
                colors = []
                for sesh in uniq_sessions:
                    sesh_df = contrast_df.filter(pl.col("session_no") == sesh)

                    analyzer.set_data(sesh_df)

                    d_hr = analyzer.get_deltahits()
                    hit_rate_contra = d_hr.filter(pl.col("stim_side") == "contra")

                    if len(hit_rate_contra):
                        area_hrs.append(100 * hit_rate_contra[0, metric])
                        per_session_errs.append(100 * hit_rate_contra[0, f"{metric}_err"])
                        if kwargs.pop("color_animals", False):
                            colors.append(ANIMAL_COLORS[sesh_df[0, "animalid"]])
                        else:
                            _clr = "#bfbfbf" if c == 12.5 else "#424242"
                            colors.append(_clr)

                _x = self.add_jitter([contrast_ind] * len(area_hrs))

                # ax.errorbar(_x, area_hrs,per_session_errs,
                #             marker='o',
                #             color = _clr,
                #             linewidth=0,
                #             elinewidth=plt.rcParams['lines.linewidth'])

                ax.scatter(_x, area_hrs, s=250, color=colors, alpha=1, linewidths=0)

                # do violin
                if kwargs.get("violin_plot", False):
                    parts = ax.violinplot(
                        area_hrs,
                        [contrast_ind],
                        showmedians=False,
                        showextrema=False,
                        widths=0.5,
                        side="low" if j == 0 else "high",
                    )

                    for pc in parts["bodies"]:
                        if j == 0:
                            pc.set_facecolor("#bfbfbf")
                        else:
                            pc.set_facecolor("#424242")
                        pc.set_alpha(0.6)

                quartile1, medians, quartile3 = np.percentile(area_hrs, [25, 50, 75])
                ax.scatter(contrast_ind, medians, marker="_", color="k", s=100, zorder=3)
                ax.vlines(
                    contrast_ind,
                    quartile1,
                    quartile3,
                    color="k",
                    linestyle="-",
                    linewidth=1.5,
                )

        ax.axhline(0, color="k", linestyle=":", linewidth=1)

        fontsize = 20
        ax.set_xlabel("Area", fontsize=fontsize)
        ax.set_ylabel(metric, fontsize=fontsize)
        ax.set_yticks([0, 25, 50, 75, 100])

        ax.tick_params(labelsize=fontsize, length=6)

        ax.set_xticks(np.arange(0, len(_area_list)))
        ax.set_xticklabels(_area_list)
        ax.tick_params(axis="x", rotation=45)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        return ax

    def plot_resp_times(
        self, ax: plt.Axes = None, contrast: float = -1, **kwargs
    ) -> plt.Axes:
        """metric can be delta_HR, percent_delta_HR, delta_HR_baseline"""

        if ax is None:
            self.fig = plt.figure(figsize=kwargs.pop("figsize", (10, 10)))
            ax = self.fig.add_subplot(1, 1, 1)

        # order by area size
        if kwargs.pop("order_by_size", False):
            _sorted_dict = {
                k: v
                for k, v in sorted(
                    AREA_SIZE.items(), key=lambda item: item[1], reverse=True
                )
            }
            _area_list = list(_sorted_dict.keys())
        else:
            _area_list = self.area_list

        for i, area in enumerate(_area_list):
            area_df = self.plot_data.filter(pl.col("area") == area)

            uniq_contrast = area_df["contrast"].drop_nulls().unique().sort().to_numpy()
            uniq_contrast = [c for c in uniq_contrast if c not in [100, 6.25, 0]]

            contrast_axis_shift = [-0.1, 0.1]
            for j, c in enumerate(uniq_contrast):

                # to plot single contrast or both
                if contrast == c:
                    contrast_ind = i
                elif contrast != -1 and contrast != c:
                    continue
                else:
                    contrast_ind = i + contrast_axis_shift[j]

                contrast_df = area_df.filter(pl.col("contrast") == c)

                uniq_sessions = area_df["session_no"].unique().to_list()
                area_resps = []
                colors = []
                for sesh in uniq_sessions:
                    sesh_df = contrast_df.filter(
                        (pl.col("session_no") == sesh) & (pl.col("stim_side") == "contra")
                    )

                    q = (
                        sesh_df.group_by(
                            ["stim_type", "contrast", "stim_side", "opto_pattern"]
                        )
                        .agg(
                            [
                                (pl.col("stim_pos").first()),
                                pl.count().alias("count"),
                                (pl.col("outcome") == 1).sum().alias("correct_count"),
                                (pl.col("outcome") == 0).sum().alias("miss_count"),
                                (pl.col("reaction_time").alias("response_times")),
                                (
                                    pl.when(pl.col("outcome") == 1)
                                    .then(pl.col("reaction_time"))
                                    .alias("response_times_correct")
                                ),
                            ]
                        )
                        .sort(["stim_type", "contrast", "stim_side", "opto_pattern"])
                    )
                    nonopto_resp = [
                        r for r in q[0, "response_times_correct"].to_numpy() if r < 1000
                    ]
                    opto_resp = [
                        r for r in q[1, "response_times_correct"].to_numpy() if r < 1000
                    ]

                    if not len(opto_resp):
                        opto_resp = [1000]

                    median_sesh_resp_diff = np.nanmedian(opto_resp) - np.nanmedian(
                        nonopto_resp
                    )
                    area_resps.append(median_sesh_resp_diff)

                    if kwargs.pop("color_animals", False):
                        colors.append(ANIMAL_COLORS[sesh_df[0, "animalid"]])
                    else:
                        _clr = "#bfbfbf" if c == 12.5 else "#424242"
                        colors.append(_clr)

                _x = self.add_jitter([contrast_ind] * len(area_resps))

                ax.scatter(_x, area_resps, s=100, color=colors, alpha=1, linewidths=0)

                # do violin
                if kwargs.get("violin_plot", False):
                    parts = ax.violinplot(
                        area_resps,
                        [contrast_ind],
                        showmedians=False,
                        showextrema=False,
                        widths=0.5,
                        side="low" if j == 0 else "high",
                    )

                    for pc in parts["bodies"]:
                        if j == 0:
                            pc.set_facecolor("#bfbfbf")
                        else:
                            pc.set_facecolor("#424242")
                        pc.set_alpha(0.6)

                quartile1, medians, quartile3 = np.percentile(area_resps, [25, 50, 75])
                ax.scatter(contrast_ind, medians, marker="_", color="k", s=100, zorder=3)
                ax.vlines(
                    contrast_ind,
                    quartile1,
                    quartile3,
                    color="k",
                    linestyle="-",
                    linewidth=1.5,
                )

        ax.axhline(0, color="k", linestyle=":", linewidth=1)

        fontsize = 20
        ax.set_xlabel("Area", fontsize=fontsize)
        ax.set_ylabel("delta_resp_time", fontsize=fontsize)

        ax.tick_params(labelsize=fontsize, length=6)

        ax.set_xticks(np.arange(0, len(_area_list)))
        ax.set_xticklabels(_area_list)
        ax.tick_params(axis="x", rotation=45)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        ax.grid(alpha=0.4)

        return ax

    def plot_correlation_of_measures(
        self, ax: plt.Axes, hr_metric: str = "percent_delta_HR", **kwargs
    ) -> plt.Axes:
        """ """
        if ax is None:
            self.fig = plt.figure(figsize=kwargs.pop("figsize", (10, 10)))
            ax = self.fig.add_subplot(1, 1, 1)

    @staticmethod
    def make_dot_cloud(
        y: ArrayLike, center_pos: float = 0, nbins=None, width: float = 0.8
    ):
        """
        Returns x coordinates for the points in ``y``, so that plotting ``x`` and
        ``y`` results in a bee swarm plot.
        """
        y = np.asarray(y)
        if nbins is None:
            nbins = len(y) // 6
            print(nbins)

        # Get upper bounds of bins
        counts, bin_edges = np.histogram(y, bins=5)
        print(bin_edges)
        print(counts)

        # get the indices that correspond to points inside the bin edges
        ibs = []
        for ymin, ymax in zip(bin_edges[:-1], bin_edges[1:]):
            i = np.nonzero((y >= ymin) * (y < ymax))[0]
            ibs.append(i)

        x_coords = np.zeros(len(y))
        dx = width / (np.nanmax(counts) // 2)
        for i in ibs:
            _points = y[i]  # value of points that fall into the bin
            # if less then 2, leave untouched, will put it in the mid line
            if len(i) > 1:
                j = len(i) % 2
                i = i[np.argsort(_points)]
                # if even numbers of points, j will be 0, which will allocate the points equally to left and right
                # if odd, j will be 1, then, below lines will leave idx 0 at the midline and start from idx 1
                a = i[j::2]
                b = i[j + 1 :: 2]

                x_coords[a] = (0.5 + j / 3 + np.arange(len(a))) * dx
                x_coords[b] = (0.5 + j / 3 + np.arange(len(b))) * -dx

        return x_coords + center_pos

    def plot_raw_reaction_times(
        self,
        ax: plt.Axes = None,
        contrast: float = -1,
        reaction_of: str = "transformed",
        include_misses: bool = False,
        bin_width: int = 5,  # ms
        **kwargs,
    ) -> plt.Axes:
        """"""
        if ax is None:
            self.fig = plt.figure(figsize=kwargs.pop("figsize", (10, 10)))
            ax = self.fig.add_subplot(1, 1, 1)

        # order by area size
        if kwargs.pop("order_by_size", False):
            _sorted_dict = {
                k: v
                for k, v in sorted(
                    AREA_SIZE.items(), key=lambda item: item[1], reverse=True
                )
            }
            _area_list = list(_sorted_dict.keys())
        else:
            _area_list = self.area_list

        _offset_coeff = kwargs.pop("area_offset", 2)
        pattern_offset = [-_offset_coeff, _offset_coeff]
        _idx_coeff = (
            len(self.plot_data["opto_pattern"].drop_nulls().unique()) + _offset_coeff * 3
        )
        for i, area in enumerate(_area_list):
            idx = _idx_coeff * i
            area_df = self.plot_data.filter(
                (pl.col("area") == area)
                & (pl.col("contrast") == contrast)
                & (pl.col("stim_side") == "contra")
            )

            if include_misses == False:
                area_df = area_df.filter(pl.col("outcome") == 1)

            _both_rts = []
            for j, pattern in enumerate([-1, 0]):
                _rt = area_df.filter(pl.col("opto_pattern") == pattern)[
                    "reaction_time"
                ].to_list()
                _rt = [r for r in _rt if r > 150]

                _both_rts.append(_rt)
                _x = self.make_dot_cloud(
                    _rt, idx + pattern_offset[j], nbins=11, width=_offset_coeff / 2
                )

                ax.scatter(_x, _rt, s=160, c="k" if pattern == -1 else "#88CCEE")

                ax.hlines(
                    y=np.nanmedian(_rt),
                    xmin=idx + pattern_offset[j] - _offset_coeff,
                    xmax=idx + pattern_offset[j] + _offset_coeff,
                    linewidth=5,
                    color="w" if pattern == -1 else "b",
                )

            res = stats.kruskal(*_both_rts, nan_policy="omit")
            p = res.pvalue
            stars = ""
            if p < 0.0001:
                stars = "****"
            elif p < 0.001:
                stars = "***"
            elif 0.001 < p < 0.01:
                stars = "**"
            elif 0.01 < p < 0.05:
                stars = "*"
            else:
                continue
            ax.text(idx, 1000, stars, color="k")
            ax.text(
                idx,
                1100,
                round(p, 6),
                color="k",
            )

        ax.axhline(0, color="k", linestyle=":", linewidth=1)

        fontsize = 20
        ax.set_yscale("log")
        ax.set_xlabel("Area", fontsize=fontsize)
        ax.set_ylabel("Reaction Times", fontsize=fontsize)

        ax.tick_params(labelsize=fontsize, length=6)

        ax.set_xticks(np.arange(0, len(_area_list) * _idx_coeff, _idx_coeff))
        ax.set_xticklabels(_area_list)
        ax.tick_params(axis="x", rotation=45)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        return ax


class EffectMatrixPlotter:
    pass
