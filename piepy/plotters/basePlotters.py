import polars as pl
from scipy import stats
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

from ..core.io import display

from .plotting_utils import *
from ..psychophysics.wheelUtils import *
from ..psychophysics.wheelTrace import *
from ..psychophysics.detection.wheelDetectionAnalysis import DetectionAnalysis
from ..core.exceptions import *


class CumulativeReactionTimePlotter(BasePlotter):
    """Plots the cumulative distribution of trial reaction times"""

    def __init__(self, data: pl.DataFrame, **kwargs) -> None:
        super().__init__(data, **kwargs)

    @staticmethod
    def get_cumulative(time_data_arr: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
        """Gets the cumulative distribution"""
        sorted_times = np.sort(time_data_arr)
        counts, _ = np.histogram(sorted_times, bins=bin_edges)
        pdf = counts / np.sum(counts)
        cum_sum = np.cumsum(pdf)
        return cum_sum

    def plot(
        self,
        ax: plt.Axes = None,
        reaction_of: str = "state",
        bin_width: float = 10,
        **kwargs,
    ) -> None:
        """
        Parameters:
        ax (plt.axes): An axes object to place to plot,default is None, which creates the axes
        reaction_of (str): string that indicates which reaction type to use, default is 'state'
                            state: timing from stimpy statemachine
                            pos:   timing from wheel trajectory position that passes the threshold value
                            speed: timing from wheel movement speed that passes the threshold value
                            rig:   timing recorded from rig Arduino(not present in all sessions due to stimpy branch differences)
        bin_width (float): Width of bins in ms, default is 10

        Returns:
        plt.axes: Axes object
        """

        if ax is None:
            self.fig = plt.figure(figsize=kwargs.pop("figsize", (8, 8)))
            ax = self.fig.add_subplot(1, 1, 1)

        if reaction_of == "state":
            reaction_of = "response_latency"
        elif reaction_of == "transformed":
            reaction_of = "transformed_response_times"
        elif reaction_of in ["pos", "speed", "rig"]:
            reaction_of = reaction_of + "_reaction_time"

        sorted_data = self.plot_data.sort(reaction_of)
        sorted_data = sorted_data.filter(pl.col(reaction_of) != -1)

        reaction_times = sorted_data[reaction_of].to_numpy()

        bin_edges_early = np.arange(reaction_times[0], 0, bin_width)
        bin_edges = np.arange(0, 1000, bin_width)
        bin_edges_miss = np.arange(1000, reaction_times[-1], bin_width)

        bin_edges = np.hstack((bin_edges_early, bin_edges, bin_edges_miss))

        cum_sum = self.get_cumulative(reaction_times, bin_edges)

        hit_start = np.where(bin_edges >= 150)[0][0]
        hit_end = np.where(bin_edges >= 1000)[0][0]

        ax.plot(bin_edges[0:hit_start], cum_sum[0:hit_start], color="#9c9c9c")
        ax.plot(bin_edges[hit_start:hit_end], cum_sum[hit_start:hit_end], color="#039612")
        ax.plot(bin_edges[hit_end - 1 : -1], cum_sum[hit_end - 1 :], color="#CA0000")

        ax.set_ylabel("Fraction of Trials")
        ax.set_xlabel("Time from Stimulus Onset(ms)")

        ax.set_xticks([-1000, 0, 1000])

        return ax


class ReactionCumulativePlotter(BasePlotter):
    """"""

    def __init__(self, data: pl.DataFrame, **kwargs):
        super().__init__(data, **kwargs)
        self.stat_analysis = DetectionAnalysis(data=self.plot_data)

    @staticmethod
    def get_cumulative(time_data_arr: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
        """Gets the cumulative distribution"""
        sorted_times = np.sort(time_data_arr)
        counts, _ = np.histogram(sorted_times, bins=bin_edges)
        pdf = counts / np.sum(counts)
        cum_sum = np.cumsum(pdf)
        return cum_sum

    def plot(
        self,
        seperate_by: str = "stim_type",
        bin_width: int = 50,
        from_wheel: bool = False,
        first_move: bool = False,
        **kwargs,
    ) -> plt.Axes:
        """Plots the response time progression through the session

        Parameters:
        ax (plt.axes): An axes object to place to plot,default is None, which creates the axes
        seperate_by (str): string that indicate sthe column name to seperate the data by, e.g, contrast,stimkey
        reaction_of (str): What reaction time value to use for plotting, e.g 'rig', 'pos', 'speed', 'state
        running_window (int): Window width for running average

        Returns:
        plt.axes: Axes object
        """
        # make the bin edges array
        bin_edges = np.arange(0, 2000, bin_width)

        data = self.stat_analysis.agg_data.drop_nulls().sort(
            ["stimkey", "opto"], descending=True
        )

        # add cut
        data = data.with_columns(
            pl.col("response_times")
            .apply(lambda x: [i for i in x if i < 1000])
            .alias("cutoff_response_times")
        )

        # get uniques
        u_stimtype = data["stim_type"].unique().sort().to_numpy()
        n_stim = len(u_stimtype)

        u_opto = self.plot_data["opto_pattern"].unique().sort().to_list()
        n_opto = len(u_opto)

        u_contrast = data["contrast"].unique().sort().to_numpy()
        u_contrast = u_contrast[1:]  # remove 0 contrast, we dont care about it here
        n_contrast = len(u_contrast)

        if seperate_by == "stim_type":
            self.fig, axes = plt.subplots(
                ncols=n_contrast,  # remove 0 contrast
                nrows=n_opto,
                constrained_layout=True,
                figsize=kwargs.pop("figsize", (15, 15)),
            )

            for i, o in enumerate(u_opto):
                for j, c in enumerate(u_contrast):

                    try:
                        ax = axes[i][j]
                    except:
                        ax = axes[j]

                    for k in u_stimtype:
                        filt_df = data.filter(
                            (pl.col("opto_pattern") == o)
                            & (pl.col("stim_side") == "contra")
                            & (pl.col("contrast") == c)
                            & (pl.col("stim_type") == k)
                        )

                        if not filt_df.is_empty():

                            if from_wheel:
                                reaction_times = (
                                    filt_df["wheel_reaction_time"].explode().to_numpy()
                                )
                            else:
                                reaction_times = (
                                    filt_df["response_times"].explode().to_numpy()
                                )

                            cumulative_reaction = self.get_cumulative(
                                reaction_times, bin_edges
                            )

                            ax.plot(
                                bin_edges[:-1],
                                cumulative_reaction,
                                color=self.color.stim_keys[filt_df[0, "stimkey"]][
                                    "color"
                                ],
                                linewidth=4,
                            )

                            # line
                            ax.axvline(1000, color="r", linewidth=2)

                            ax.set_ylim([-0.01, 1.05])
                            ax.set_xlim([-1, 1100])
                            if o:
                                ax.set_title(f"Opto c={c}%", pad=3)
                            else:
                                ax.set_title(f"Non-Opto c={c}%", pad=3)
                            if i == n_opto - 1:
                                ax.set_xlabel("Time from Stim Onset (ms)")
                            if j == 0:
                                ax.set_ylabel("Probability")
                            ax.grid(alpha=0.5, axis="both")

                        # ax.legend(loc='center left',bbox_to_anchor=(0.98,0.5),fontsize=fontsize-5,frameon=False)
        elif seperate_by == "opto":
            self.fig, axes = plt.subplots(
                ncols=n_contrast,  # remove 0 contrast
                nrows=n_stim,
                constrained_layout=True,
                figsize=kwargs.pop("figsize", (15, 15)),
            )
            for i, k in enumerate(u_stimtype):
                for j, c in enumerate(u_contrast):
                    try:
                        ax = axes[i][j]
                    except:
                        ax = axes[j]
                    for o in u_opto:
                        filt_df = data.filter(
                            (pl.col("opto_pattern") == o)
                            & (pl.col("stim_side") == "contra")
                            & (pl.col("contrast") == c)
                            & (pl.col("stim_type") == k)
                        )

                        if not filt_df.is_empty():
                            if from_wheel:
                                reaction_times = (
                                    filt_df["wheel_reaction_time"].explode().to_numpy()
                                )
                            else:
                                reaction_times = (
                                    filt_df["response_times"].explode().to_numpy()
                                )

                            cumulative_reaction = self.get_cumulative(
                                reaction_times, bin_edges
                            )

                            ax.plot(
                                bin_edges[:-1],
                                cumulative_reaction,
                                color=self.color.stim_keys[filt_df[0, "stimkey"]][
                                    "color"
                                ],
                                linewidth=4,
                            )

                            # line
                            ax.axvline(1000, color="r", linewidth=2)

                            ax.set_ylim([-0.01, 1.05])
                            ax.set_xlim([-1, 1100])
                            ax.set_title(f"{filt_df[0,'stim_label']} c={c}%", pad=3)
                            if i == n_stim - 1:
                                ax.set_xlabel("Response Time (ms)")
                            if j == 0:
                                ax.set_ylabel("Probability")
                            ax.grid(alpha=0.5, axis="both")


class ReactionTimeDistributionPlotter(BasePlotter):
    """Plots the reaction times as a dot cloud on x-axis as contrast and y-axis as reaction times"""

    def __init__(self, data, stimkey: str = None, **kwargs) -> None:
        super().__init__(data, **kwargs)
        self.stat_analysis = DetectionAnalysis(data=self.plot_data)
        contrast_axis = self.make_linear_contrast_axis(self.plot_data)
        contrast_idx = pl.Series(
            "linear_contrast_idx",
            [
                float(contrast_axis[x]) if x is not None else None
                for x in self.stat_analysis.agg_data["signed_contrast"].to_list()
            ],
        )
        self.stat_analysis.agg_data = self.stat_analysis.agg_data.with_columns(
            contrast_idx
        )

    @staticmethod
    def make_dot_cloud2(
        response_times: ArrayLike, pos: float, cloud_width: float = 0.33
    ) -> tuple[list, list]:
        """Makes a dot cloud by adding random jitter to inidividual points

        Parameters:
        response_times (ArrayLike): Response times as an array
        pos (float) : Center position to make the dot cloud
        cloud_width (float) : Determines how wide the dot cloud is

        Returns:
        tuple(list,list): x and y coordinates of the dots
        """
        counts, bin_edges = np.histogram(response_times, bins=5)

        part_x = []
        part_y = []
        for j, point_count in enumerate(counts):
            points = [
                p for p in response_times if p >= bin_edges[j] and p <= bin_edges[j + 1]
            ]
            # generate x points around the actual x
            # range is determined by point count in the bin and scaling factor

            scatter_range = np.linspace(
                pos - cloud_width * (point_count / np.max(counts)),
                pos + cloud_width * (point_count / np.max(counts)),
                len(points),
            ).tolist()
            part_x += scatter_range
            part_y += points

        return part_x, part_y

    @staticmethod
    def make_dot_cloud(
        y: ArrayLike, center_pos: float = 0, nbins=None, width: float = 0.6
    ):
        """
        Returns x coordinates for the points in ``y``, so that plotting ``x`` and
        ``y`` results in a bee swarm plot.
        """
        y = np.asarray(y)
        if nbins is None:
            nbins = len(y) // 6

        # Get upper bounds of bins
        counts, bin_edges = np.histogram(y, bins=nbins)
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

    @staticmethod
    def __plot_scatter__(ax, contrast, time, median, pos, cloud_width, **kwargs):
        """Plots the trial response times as a scatter cloud plot"""
        ax.scatter(
            contrast,
            time,
            linewidths=0,
            s=(plt.rcParams["lines.markersize"] ** 2) / 2,
            **kwargs,
        )

        # median
        ax.plot(
            [pos - cloud_width / 2, pos + cloud_width / 2],
            [median, median],
            linewidth=3,
            c=kwargs.get("color", "b"),
            path_effects=[pe.Stroke(linewidth=6, foreground="k"), pe.Normal()],
        )

        return ax

    @staticmethod
    def __plot_line__(ax, x, y, err, **kwargs):
        """Plots the trial response times as a line plot with errorbars"""
        if "fontsize" in kwargs.keys():
            kwargs.pop("fontsize")

        ax.errorbar(
            x,
            y,
            err,
            linewidth=2,
            markeredgecolor=kwargs.get("markeredgecolor", "w"),
            markeredgewidth=kwargs.get("markeredgewidth", 2),
            elinewidth=kwargs.get("elinewidth", 3),
            capsize=kwargs.get("capsize", 0),
            **kwargs,
        )
        return ax

    @staticmethod
    def add_jitter_to_misses(
        resp_times: ArrayLike, jitter_lims: list = [0, 100]
    ) -> np.ndarray:
        """Adds jitter in y-dimension to missed trial dots

        Parameters:
        resp_times (ArrayLike) : Response times as an array
        jitter_lims (list) : The jitter range in ms

        Returns:
        np.ndarray: response times
        """
        resp_times = np.array(
            resp_times
        )  # polars returns an immutable numpy array, this changes that
        miss_locs = np.where(resp_times >= 1000)[0]
        if len(miss_locs):
            jitter = np.random.choice(
                np.arange(jitter_lims[0], jitter_lims[1]), len(miss_locs), replace=True
            )
            resp_times[miss_locs] = resp_times[miss_locs] + jitter
        return resp_times

    def plot_scatter(
        self,
        ax: plt.Axes = None,
        t_cutoff: float = 1_000,
        cloud_width: float = 0.33,
        reaction_of: str = "state",
        color: str = None,
        **kwargs,
    ) -> plt.Axes:
        """Plots the distribution as a scatter cloud plot

        Parameters:
        t_cutoff (float) : Cutoff response time value in ms, the values larger than this will be discarded
        cloud_width (float) : Determines how wide the dot cloud is
        reaction_of (str) : What reaction time value to use for plotting, e.g 'rig', 'pos', 'speed', 'state

        Returns:
        plt.axes: Axes object
        """
        if ax is None:
            self.fig = plt.figure(figsize=kwargs.pop("figsize", (15, 10)))
            ax = self.fig.add_subplot(1, 1, 1)

        if reaction_of == "state":
            reaction_of = "response_times"
        elif reaction_of in ["pos", "speed", "rig"]:
            reaction_of = reaction_of + "_reaction_time"
        elif reaction_of == "transformed_times":
            reaction_of = "transformed_response_times"

        data = self.stat_analysis.agg_data.drop_nulls("contrast").sort(
            ["stimkey", "opto"], descending=True
        )

        # do cutoff
        data = data.with_columns(
            pl.col(reaction_of)
            .list.eval(pl.when(pl.element().is_between(150, t_cutoff)).then(pl.element()))
            .alias("cutoff_response_times")
        )

        # make a key,value pair from signed_contrast and linear_contrast_idx
        cpos = self.make_linear_contrast_axis(data)
        for filt_tup in self.subsets(data, ["stimkey", "stim_side", "signed_contrast"]):
            filt_df = filt_tup[-1]
            if not filt_df.is_empty():
                if filt_tup[1] == "catch":
                    continue

                resp_times = filt_df[0, "cutoff_response_times"].to_numpy()
                resp_times = resp_times[~np.isnan(resp_times)]

                response_times = self.add_jitter_to_misses(resp_times)

                x_dots = self.make_dot_cloud(
                    response_times, cpos[filt_tup[2]], 100, cloud_width
                )
                y_dots = response_times
                median = np.median(response_times)

                ax = self.__plot_scatter__(
                    ax,
                    x_dots,
                    y_dots,
                    median,
                    cpos[filt_tup[2]],
                    cloud_width,
                    color=(
                        self.color.stim_keys[filt_tup[0]]["color"]
                        if color is None
                        else color
                    ),
                    label=(
                        filt_df[0, "stim_label"]
                        if filt_tup[1] == "contra" and filt_tup[2] == 12.5
                        else "_"
                    ),
                    **kwargs,
                )
        # baseline
        baseline = data.filter(
            (pl.col("stim_side") == "catch") & (pl.col("opto") == False)
        )
        if len(baseline):
            catch_resp_times = (
                baseline["cutoff_response_times"].explode().drop_nulls().to_numpy()
            )
            catch_resp_times = self.add_jitter_to_misses(catch_resp_times)
            x_dots = self.make_dot_cloud(catch_resp_times, cpos[0], 100, cloud_width / 2)
            y_dots = catch_resp_times
            median = np.median(catch_resp_times)
            ax = self.__plot_scatter__(
                ax,
                x_dots,
                y_dots,
                median,
                cpos[0],
                cloud_width / 2,
                color="#909090",
                label="Catch Trials",
                **kwargs,
            )

        p_data = data.sort(["stim_type", "contrast", "stim_side"])
        for i, filt_tup in enumerate(
            self.subsets(p_data, ["stim_type", "signed_contrast"])
        ):
            pfilt_df = filt_tup[-1]
            if len(pfilt_df) < 2:
                continue
            elif len(pfilt_df) >= 2:
                pfilt_df = pfilt_df.sort("opto_pattern")
                for k in range(1, len(pfilt_df)):
                    if len(pfilt_df[k, "cutoff_response_times"].to_numpy()):

                        p = self.stat_analysis.get_pvalues_nonparametric(
                            pfilt_df[0, "cutoff_response_times"].to_numpy(),
                            pfilt_df[k, "cutoff_response_times"].to_numpy(),
                        )
                        stars = ""
                        if p < 0.001:
                            stars = "***"
                        elif 0.001 < p < 0.01:
                            stars = "**"
                        elif 0.01 < p < 0.05:
                            stars = "*"

                        ax.text(
                            cpos[filt_tup[1]],
                            1100 + i * 200 + k * 100,
                            stars,
                            color=self.color.stim_keys[pfilt_df[k, "stimkey"]]["color"],
                        )
            else:
                raise ValueError(f"WEIRD DATA FRAME FOR P-VALUE ANALYSIS!")

        # mid line
        ax.set_ylim([90, 1500])
        ax.plot([0, 0], ax.get_ylim(), color="gray", linewidth=2, alpha=0.5)

        # miss line
        ax.axhline(1000, color="r", linewidth=1.5, linestyle=":")

        ax.set_xlabel("Stimulus Contrast (%)")
        ax.set_ylabel("Response Time (ms)")

        ax.set_yscale("symlog")
        minor_locs = [200, 400, 600, 800, 2000, 4000, 6000, 8000]
        ax.yaxis.set_minor_locator(plt.FixedLocator(minor_locs))
        ax.yaxis.set_minor_formatter(plt.FormatStrFormatter("%d"))
        ax.yaxis.set_major_locator(ticker.FixedLocator([100, 1000, 10000]))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
        ax.xaxis.set_major_locator(ticker.FixedLocator(list(cpos.values())))
        ax.xaxis.set_major_formatter(ticker.FixedFormatter([i for i in cpos.keys()]))
        ax.grid()
        # ax.legend(loc='center left',bbox_to_anchor=(0.98,0.5),fontsize=fontsize-5,frameon=False)
        return ax

    def plot_line(self, ax: plt.Axes = None, t_cutoff: float = 10_000, **kwargs):

        if ax is None:
            self.fig = plt.figure(figsize=kwargs.get("figsize", (15, 10)))
            ax = self.fig.add_subplot(1, 1, 1)
            if "figsize" in kwargs:
                kwargs.pop("figsize")

        data = self.stat_analysis.agg_data.drop_nulls().sort(
            ["stimkey", "opto"], reverse=True
        )

        # do cutoff
        data = data.with_columns(
            pl.col("response_times")
            .apply(lambda x: [i for i in x if i < t_cutoff])
            .alias("cutoff_response_times")
        )

        # add 95% confidence
        def get_conf(arr):
            x = np.sort(arr)
            j = round(len(x) * 0.5 - 1.96 * np.sqrt(len(x) ** 0.5))
            k = round(len(x) * 0.5 + 1.96 * np.sqrt(len(x) ** 0.5))
            return [x[j], x[k]]

        # data = data.with_columns(pl.col('response_times').apply(lambda x: [np.mean(x.to_numpy())-2*np.std(x.to_numpy()),np.mean(x.to_numpy())+2*np.std(x.to_numpy())]).alias('resp_confs'))
        data = data.with_columns(
            pl.col("response_times").apply(lambda x: get_conf(x)).alias("resp_confs")
        )

        # get uniques
        u_stimkey = data["stimkey"].unique().to_numpy()
        u_stimtype = data["stim_type"].unique().to_numpy()
        u_stim_side = data["stim_side"].unique().to_numpy()
        u_scontrast = data["signed_contrast"].unique().sort().to_numpy()

        for k in u_stimkey:
            for s in u_stim_side:
                filt_df = data.filter(
                    (pl.col("stimkey") == k) & (pl.col("stim_side") == s)
                )

                contrast = filt_df["signed_contrast"].to_numpy()
                confs = filt_df["resp_confs"].to_list()
                confs = np.array(confs).T
                if not filt_df.is_empty():
                    resp_times = filt_df["cutoff_response_times"].to_numpy()
                    # do cutoff, default is 10_000 to involve everything
                    medians = []
                    for i, c_r in enumerate(resp_times):
                        response_times = self.time_to_log(c_r)
                        response_times = self.add_jitter_to_misses(response_times)

                        median = np.median(response_times)
                        medians.append(median)
                        # jittered_offset = np.array([np.random.uniform(0,jitter)*c for c in contrast])
                        # jittered_offset[0] += np.random.uniform(0,jitter)/100

                    ax = self.__plot_line__(
                        ax,
                        contrast,
                        medians,
                        confs,
                        color=self.color.stim_keys[k]["color"],
                        label=filt_df[0, "stim_label"] if s == "contra" else "_",
                        **kwargs,
                    )

        # mid line
        fontsize = kwargs.get("fontsize", 25)
        ax.set_ylim([90, 1500])
        ax.plot([0, 0], ax.get_ylim(), color="gray", linewidth=2, alpha=0.5)

        fontsize = kwargs.get("fontsize", 20)
        ax.set_xlabel("Stimulus Contrast (%)", fontsize=fontsize)
        ax.set_ylabel("Response Time (ms)", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize, which="both", axis="both")

        ax.set_yscale("symlog")
        minor_locs = [200, 400, 600, 800, 2000, 4000, 6000, 8000]
        ax.yaxis.set_minor_locator(plt.FixedLocator(minor_locs))
        ax.yaxis.set_minor_formatter(plt.FormatStrFormatter("%d"))
        ax.yaxis.set_major_locator(ticker.FixedLocator([100, 1000, 10000]))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))

        ax.grid(alpha=0.5, axis="both")

        ax.spines["bottom"].set_position(("outward", 10))
        ax.spines["left"].set_position(("outward", 10))

        ax.legend(
            loc="center left",
            bbox_to_anchor=(0.98, 0.5),
            fontsize=fontsize - 5,
            frameon=False,
        )

        return ax


class ResponseTimeHistogramPlotter(BasePlotter):
    __slots__ = ["plot_data", "uniq_keys"]

    def __init__(self, data, **kwargs):
        super().__init__(data=data, **kwargs)

    @staticmethod
    def bin_times(time_arr, bin_width=50, bins: np.ndarray = None):
        """Counts the response times in bins(ms)"""
        if bins is None:
            bins = np.arange(
                np.min(time_arr) - bin_width, np.max(time_arr) + bin_width, bin_width
            )

        return np.histogram(time_arr, bins)

    @staticmethod
    def __plot__(ax, counts, bins, **kwargs):
        # adapt the bar width to the bin width
        bar_width = bins[1] - bins[0]

        if kwargs.get("color") is not None:
            cl = kwargs.get("color")
            kwargs.pop("color")
        else:
            cl = "forestgreen"

        color = ["orangered" if i <= 150 else cl for i in bins]

        ax.bar(bins[1:], counts, width=bar_width, color=color, linewidth=0, **kwargs)

        # zero line
        ax.axvline(x=0, color="k", linestyle=":", linewidth=3)

        return ax


class ResponseTypeBarPlotter(BasePlotter):
    __slots__ = ["stimkey", "plot_data", "uniq_keys"]

    def __init__(self, data, stimkey: str = None, **kwargs):
        super().__init__(data=data, **kwargs)

    @staticmethod
    def __plot__(ax, x_locs, bar_heights, **kwargs):
        ax.bar(x_locs, bar_heights, **kwargs)
        return ax


class LickScatterPlotter(BasePlotter):
    """Lick scatters for each trial, wrt reward or response time"""

    def __init__(self, data: dict, **kwargs):
        super().__init__(data, **kwargs)

    @staticmethod
    def __plot_scatter__(ax, t, lick_arr, **kwargs):

        t_arr = [t] * len(lick_arr)

        ax.scatter(
            lick_arr, t_arr, marker="|", c="deepskyblue", s=kwargs.get("s", 20), **kwargs
        )

        return ax

    @staticmethod
    def pool_licks(data, wrt: str = "reward"):
        pooled_lick = np.array([])
        error_ctr = []

        for row in data.iter_rows(named=True):
            if len(row["lick"]):
                if wrt == "reward":
                    try:
                        wrt_time = row["reward"][0]
                    except:
                        error_ctr.append(row.trial_no)
                        display(
                            f"\n!!!!!! NO REWARD IN CORRECT TRIAL, THIS IS A VERY SERIOUS ERROR! SOLVE THIS ASAP !!!!!!\n"
                        )
                elif wrt == "response":
                    wrt_time = row["response_latency_absolute"]

                pooled_lick = np.append(pooled_lick, np.array(row["lick"]) - wrt_time)
        print(f"Trials with reward issue: {error_ctr}")
        return pooled_lick

    @staticmethod
    def __plot_density__(ax, x_bins, y_dens, **kwargs):
        ax.plot(
            x_bins[1:], y_dens, c="aqua", alpha=0.8, linewidth=3, **kwargs
        )  # right edges
        return ax


class WheelTrajectoryPlotter(BasePlotter):
    def __init__(self, data: pl.DataFrame, **kwargs) -> None:
        super().__init__(data, **kwargs)
        self.trace = WheelTrace()

    @staticmethod
    def __plot_density__(ax, x_bins, y_dens, **kwargs):
        ax.plot(
            x_bins[1:], y_dens, c="k", alpha=0.8, linewidth=3, **kwargs
        )  # right edges
        return ax

    def pool_trial_ends(self) -> np.ndarray:
        """Gets the relative(from stim start) stimulus end times"""
        pooled_ends = []
        pool_data = self.plot_data[self.stimkey].copy(deep=True)
        # pool_data = pool_data[(pool_data['answer']==1) & (pool_data['isCatch']==0)]
        pool_data = pool_data[pool_data["isCatch"] == 0]
        for row in pool_data.itertuples():
            try:
                temp = row.stim_end_rig - row.stim_start_rig
            except AttributeError:
                temp = row.stim_end - row.stim_start
            pooled_ends.append(temp)
        return np.array(pooled_ends)

    def plot(
        self,
        tuple_f_axs: tuple = None,
        seperate_by: str = "contrast",
        anchor_by: str = "t_stimstart_rig",
        include_misses: bool = True,
        **kwargs,
    ):

        time_lims = kwargs.pop("time_lims", None)
        if time_lims is None:
            time_lims = [-200, 1500]

        traj_lims = kwargs.pop("traj_lims", None)
        if traj_lims is None:
            traj_lims = [None, None]

        plot_speed = kwargs.pop("plot_speed", False)

        if include_misses:
            data = self.plot_data.filter(pl.col("outcome") != -1)
        else:
            data = self.plot_data.filter(pl.col("outcome") == 1)

        uniq_opto = data["opto_pattern"].drop_nulls().unique().sort().to_list()
        n_opto = len(uniq_opto)

        uniq_stim = data["stim_type"].drop_nulls().unique().sort().to_list()
        n_stim = len(uniq_stim)

        if seperate_by == "contrast":
            color = self.color.contrast_keys
        elif seperate_by == "outcome":
            color = self.color.outcome_keys

        if tuple_f_axs is None:
            self.fig, axes = plt.subplots(
                nrows=n_stim * n_opto,
                constrained_layout=True,
                figsize=kwargs.pop("figsize", (20, 15)),
            )
        else:
            axes = tuple_f_axs

        if not isinstance(axes, np.ndarray):
            axes = [axes]
        # dummy variables to control axis switching
        _target_id = uniq_opto[0]  # opto patterns are on the columns
        _stim_name = uniq_stim[0]  # stim types are on the rows
        ax_idx = 0
        all_rig_react = []
        # loop through all subsets of data (N=multiplication of the number of unique elements in each column)
        for filt_tup in self.subsets(data, ["opto_pattern", "stim_type", seperate_by]):
            filt_df = filt_tup[-1]
            sep = filt_tup[2]

            if filt_tup[0] != _target_id or filt_tup[1] != _stim_name:
                ax_idx += 1
                # make the target match the current opto_pattern and stim type in concurrent loops
                _target_id = filt_tup[0]
                _stim_name = filt_tup[1]

            ax = axes[ax_idx]

            if "stimstart" in anchor_by:
                filt_df = filt_df.filter(pl.col("state_outcome") != -1)

            if len(filt_df):

                _temp_wheel_list = []
                _rig_react_rad_list = []
                _longest_trace_len = 0
                t_interp = None
                _counter = (
                    0  # counter is used because some trials have <2 wheel data points
                )
                for i, trial in enumerate(filt_df.iter_rows(named=True)):
                    wheel_time = np.array(trial["wheel_time"])
                    wheel_pos = np.array(trial["wheel_pos"])
                    time_anchor = trial[anchor_by]

                    if len(wheel_time) > 2:

                        self.trace.set_trace_data(wheel_time, wheel_pos)
                        if time_anchor is None:
                            print("ojbsdfjibsdf")
                        self.trace.init_trace(time_anchor)
                        speed = np.abs(self.trace.velo_interp) * 1000

                        # take the relative portion from 0
                        _temp = np.where(
                            (self.trace.tick_t_interp >= time_lims[0])
                            & (self.trace.tick_t_interp <= time_lims[1])
                        )
                        if len(_temp):
                            time_window = self.trace.tick_t_interp[_temp]
                            pos_window = self.trace.tick_pos_interp[_temp]
                            speed_window = speed[_temp]
                            try:
                                a = pos_window[0]
                            except:
                                print("lkjbasdadfdfk;n")

                            # TODO: Arbitrary filtering for weird wheel traces
                            if -5 < pos_window[0] < 5:
                                if plot_speed:
                                    _temp_wheel_list.append(speed_window.tolist())
                                    if trial["rig_reaction_tick"] is not None:
                                        speed_thresh = trial["rig_reaction_tick"] / (
                                            trial["median_loop_time"] * 5
                                        )  # 5 is wheelbuffer
                                        speed_thresh = self.trace.cm_to_rad(
                                            self.trace.ticks_to_cm(speed_thresh)
                                        )
                                        _rig_react_rad_list.append(
                                            speed_thresh * 1000
                                        )  # x1000 for rad/s
                                else:
                                    # convert to rad
                                    wheel_pos_rad = self.trace.cm_to_rad(
                                        self.trace.ticks_to_cm(np.array(pos_window))
                                    )
                                    _temp_wheel_list.append(wheel_pos_rad.tolist())
                                    if trial["rig_reaction_tick"] is not None:
                                        pos_thresh = self.trace.cm_to_rad(
                                            self.trace.ticks_to_cm(
                                                trial["rig_reaction_tick"]
                                            )
                                        )
                                        _rig_react_rad_list.append(pos_thresh)

                                if len(_temp_wheel_list[_counter]) >= _longest_trace_len:
                                    _longest_trace_len = len(_temp_wheel_list[_counter])
                                    t_interp = time_window
                                _counter += 1

                all_rig_react.extend(_rig_react_rad_list)
                # make amatrix to avreage over the rows
                all_traces_mat = np.array(
                    [
                        xi + [None] * (_longest_trace_len - len(xi))
                        for xi in _temp_wheel_list
                    ],
                    dtype=float,
                )

                avg = np.nanmean(all_traces_mat, axis=0)
                sem = stats.sem(all_traces_mat, axis=0, nan_policy="omit")

                ax.fill_between(
                    t_interp,
                    avg - sem,
                    avg + sem,
                    color=color[str(sep)]["color"],
                    alpha=0.2,
                    linewidth=0,
                )

                ax.plot(t_interp, avg, **color[str(sep)], **kwargs)

            ax.set_xlim(time_lims[0] - 10, time_lims[1] + 10)
            ax.set_ylim(traj_lims[0], traj_lims[1])

            # anchor line(stimstart init start)
            ax.axvline(0, color="k", linewidth=1, alpha=0.6)

            # stim end
            ax.axvline(1000, color="k", linewidth=1, alpha=0.6)

            ax.set_title(f"{filt_tup[0]} {filt_tup[1]}")
            if plot_speed:
                ax.set_ylabel("Wheel\nspeed (rad/s)")
            else:
                ax.set_ylabel("Wheel\nmovement (rad)")
            ax.set_xlabel("Time from stim onset (ms)")

            # make it pretty
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)

            ax.grid(axis="y")
            ax.legend(frameon=False)

        react_mean = np.mean(all_rig_react)
        for a in axes:
            a.axhline(react_mean, color="#147800", linewidth=0.5, alpha=0.8)
            if not plot_speed:
                ax.axhline(-react_mean, color="#147800", linewidth=0.5, alpha=0.8)

        return axes
