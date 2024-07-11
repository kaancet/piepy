from typing import Any
from ..basePlotters import *
from ...psychophysics.detection.wheelDetectionSilencing import *
from ...psychophysics.detection.wheelDetectionAnalysis import *


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
}


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
            .groupby(["animalid", "stim_type", "stim_side", "contrast", "opto_pattern"])
            .agg(
                [
                    pl.count().alias("trial_count"),
                    (pl.col("outcome") == 1).sum().alias("correct_count"),
                    (pl.col("outcome") == 0).sum().alias("miss_count"),
                    (
                        pl.when(pl.col("outcome") == 1)
                        .then(pl.col("transformed_response_times"))
                        .alias("response_times_correct")
                    ),
                    (pl.col("transformed_response_times").alias("response_times")),
                    (
                        pl.col("transformed_response_times")
                        .median()
                        .alias("median_response_times")
                    ),
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
            (pl.col("correct_count") / pl.col("trial_count")).alias("hit_rate")
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
        q = q.with_columns((100 * pl.col("hit_rate")).alias("hit_rate"))
        q = q.with_columns(
            pl.when(pl.col("stim_side") == "ipsi")
            .then((pl.col("contrast") * -1))
            .otherwise(pl.col("contrast"))
            .alias("signed_contrast")
        )
        q = q.with_columns(
            pl.when((pl.col("contrast") > 0) & (pl.col("contrast") < 25))
            .then(pl.lit("hard"))
            .when(pl.col("contrast") > 25)
            .then(pl.lit("easy"))
            .otherwise(pl.lit("catch"))
            .alias("contrast_difficulty")
        )

        # reorder stim_label to last column
        cols = q.columns
        del cols[-6]
        del cols[-5]
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
            filtered_data.groupby(["opto_pattern"])
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
            pl.col("hit_rate").apply(lambda x: stats.sem(x)).alias("animal_confs")
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
        x_axis_dict = {-1: "Non\nOpto", 0: "Opto"}

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
                        _, p = mannwhitneyu(non_opto, opto)
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
                ax.set_xticks(stim_df["opto_pattern"].unique().sort().to_list())
                ax.set_xticklabels(["Non\nOpto", "Opto"])
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
        self, ax: plt.Axes, filtered_data: pl.DataFrame, **kwargs
    ) -> plt.Axes:

        uniq_sesh = filtered_data["session_no"].unique().to_numpy()
        mean_of_medians = []
        for u_s in uniq_sesh:
            sesh_df = filtered_data.filter(pl.col("session_no") == u_s)

            if len(sesh_df["opto_pattern"].unique().to_list()) == 1:
                if sesh_df[0, "stim_side"] == "catch":
                    tmp = sesh_df["response_times_correct"].explode().to_numpy()
                    median = np.nanmedian([i for i in tmp if i < 1000])
                else:
                    tmp = sesh_df["response_times_correct"].to_numpy()
                    median = np.nanmedian([i for i in tmp if i < 1000])
            else:
                tmp = sesh_df["response_times_correct"].to_numpy()
                median = [np.nanmedian(i[i < 1000]) for i in tmp]

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

        avg_df = (
            filtered_data.groupby(["opto_pattern"])
            .agg(
                [
                    pl.count().alias("animal_count"),
                    pl.col("hit_rate").mean().alias("avg_hitrate"),
                    pl.col("hit_rate"),
                ]
            )
            .sort(["opto_pattern"])
        )
        avg_df = avg_df.with_columns(
            pl.col("hit_rate").apply(lambda x: stats.sem(x)).alias("animal_confs")
        )

        mean_of_medians = np.array(mean_of_medians)
        mean = np.nanmean(mean_of_medians, axis=0)
        conf = stats.sem(mean_of_medians, axis=0, nan_policy="omit")

        ax.errorbar(
            avg_df["opto_pattern"].to_list(),
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
        remove_miss: bool = True,
        **kwargs,
    ) -> plt.figure:

        fontsize = kwargs.pop("fontsize", 30)
        linewidth = kwargs.pop("linewidth", 3)
        x_axis_dict = {-1: "Non\nOpto", 0: "Opto"}

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

        for j, c in enumerate(uniq_contrast):
            contrast_df = self.plot_data.filter((pl.col("contrast") == c))
            self.p_values_resp[c] = {}

            if c == 0:
                stim_df = contrast_df.clone()
                stim_df = stim_df.filter(pl.col("stim_side") == "catch")
                ax, _ = self._plot_sessions_and_avg_response_time_(axes[0], stim_df)

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
                ax.set_ylabel("Response Times (ms)", fontsize=fontsize)
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
                    ax, stim_df
                )

                if doP:
                    # do p-values with mann-whitney-u
                    non_opto = stim_df.filter(pl.col("opto_pattern") == -1)[
                        "median_response_times"
                    ].to_numpy()
                    opto = stim_df.filter(pl.col("opto_pattern") == 0)[
                        "median_response_times"
                    ].to_numpy()
                    if len(opto) != 0:
                        # _,p = mannwhitneyu(non_opto,opto)
                        _, p = mannwhitneyu(non_opto, opto)
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

                ax.set_ylim([150, 1300])
                ax.set_xticks(uniq_pattern)
                ax.set_xticklabels([x_axis_dict[i] for i in uniq_pattern])
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

    @staticmethod
    def add_jitter(arr, jitter_lims: list = [-0.2, 0.2]) -> np.ndarray:
        """Adds jitter in x-dimension"""
        arr = np.array(arr)  # polars returns an immutable numpy array, this changes that

        jitter = np.random.choice(
            np.linspace(jitter_lims[0], jitter_lims[1], len(arr)), len(arr), replace=True
        )
        arr = arr + jitter
        return arr

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

        for i, area in enumerate(self.area_list):
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

        ax.set_xticks(np.arange(0, len(self.area_list)))
        ax.set_xticklabels(self.area_list)
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

        for i, area in enumerate(self.area_list):
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
                        sesh_df.groupby(
                            ["stim_type", "contrast", "stim_side", "opto_pattern"]
                        )
                        .agg(
                            [
                                (pl.col("stim_pos").first()),
                                pl.count().alias("count"),
                                (pl.col("outcome") == 1).sum().alias("correct_count"),
                                (pl.col("outcome") == 0).sum().alias("miss_count"),
                                (
                                    pl.col("transformed_response_times").alias(
                                        "response_times"
                                    )
                                ),
                                (
                                    pl.when(pl.col("outcome") == 1)
                                    .then(pl.col("transformed_response_times"))
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

        ax.set_xticks(np.arange(0, len(self.area_list)))
        ax.set_xticklabels(self.area_list)
        ax.tick_params(axis="x", rotation=45)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        ax.grid(alpha=0.4)

        return ax

    def plot_reaction_times(self, ax: plt.Axes = None, **kwargs) -> plt.Axes:
        """"""
        if ax is None:
            self.fig = plt.figure(figsize=kwargs.pop("figsize", (10, 10)))
            ax = self.fig.add_subplot(1, 1, 1)


class EffectMatrixPlotter:
    pass
