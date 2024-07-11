from piepy.detection.wheelDetectionSession import WheelDetectionSession
from ..behaviorBasePlotters import *
from .wheelDetectionSessionPlotter import *


class WheelDetectionBehaviorPlotter:
    """ """

    __slots__ = ["animalid", "plot_data", "plot_stats", "fig", "stimkey"]

    def __init__(
        self,
        animalid: str,
        session_list: list,
        stimkey: str = None,
        day_count: int = 9,
        **kwargs,
    ) -> None:
        self.animalid = animalid
        self.plot_data, self.plot_stats = self.get_sessions_data(session_list, day_count)
        self.stimkey = stimkey
        self.fig = None

    def get_sessions_data(self, session_list, day_count) -> list:
        """Returns the stim_data of last x sessions.
        The returned data is a dict of dicts."""
        data = {}
        stats = {}

        past_days = np.arange(1, day_count + 1)

        for day in past_days[::-1]:
            try:
                sesh = session_list[-day]
                date_str = sesh[1].strftime("%y%m%d")
                w = WheelDetectionSession(sesh[0], load_flag=True)
                data[date_str] = w.data.stim_data
                stats[date_str] = w.stats
            except:
                pass

        return data, stats

    def save(self) -> None:
        cfg = getConfig()
        analysis_path = cfg["analysisPath"]
        last_date = list(self.plot_data.keys())[-1]
        if self.fig is not None:
            saveloc = pjoin(
                analysis_path,
                "behavior_results",
                "python_figures",
                self.animalid,
                last_date,
            )
            if not os.path.exists(saveloc):
                os.makedirs(saveloc)
            savename = f"{last_date}_{self.__class__.__name__}_{self.animalid}.pdf"
            saveloc = pjoin(saveloc, savename)
            self.fig.savefig(saveloc, bbox_inches="tight")
            display(f"Saved {savename} plot")


class WheelDetectionProgressionPlotter(BehaviorProgressionPlotter):
    def __init__(
        self,
        animalid: str,
        cumul_data: pd.DataFrame = None,
        summary_data: pd.DataFrame = None,
        **kwargs,
    ) -> None:
        super().__init__(animalid, cumul_data, summary_data, **kwargs)

    def plot_cumul(
        self, x_axis: str, y_axis: str, color: str = "k", ax: plt.Axes = None, **kwargs
    ) -> plt.Axes:
        self.check_axes(x_axis, y_axis, "cumul")
        if ax is None:
            self.fig = plt.figure(figsize=kwargs.get("figsize", (8, 8)))
            ax = self.fig.add_subplot(1, 1, 1)
            if "figsize" in kwargs:
                kwargs.pop("figsize")

    def plot_summary(
        self, x_axis: str, y_axis: str, color: str = "k", ax: plt.Axes = None, **kwargs
    ) -> plt.Axes:

        self.check_axes(x_axis, y_axis, "summary")

        if ax is None:
            self.fig = plt.figure(figsize=kwargs.get("figsize", (8, 8)))
            ax = self.fig.add_subplot(1, 1, 1)
            if "figsize" in kwargs:
                kwargs.pop("figsize")

        # This gets actual training data
        plot_data = self.summary_data[
            self.summary_data["paradigm"] == "training_detection"
        ]

        x_axis_data = plot_data[x_axis].to_numpy()
        y_axis_data = plot_data[y_axis].to_numpy()

        ax = self.__plot__(ax, x_axis_data, y_axis_data, color=color, **kwargs)

        # prettify
        fontsize = kwargs.get("fontsize", 14)
        ax.set_xlabel(x_axis, fontsize=fontsize)
        ax.set_ylabel(y_axis, fontsize=fontsize)

        ax.tick_params(axis="x", rotation=45, length=20, width=2, which="major")
        ax.tick_params(axis="both", labelsize=fontsize)
        ax.grid(alpha=0.8, axis="both")

        return ax


class WheelDetectionScatterPlotter(BehaviorScatterPlotter):
    def __init__(
        self,
        animalid: str,
        cumul_data: pd.DataFrame,
        summary_data: pd.DataFrame,
        **kwargs,
    ):
        super().__init__(animalid, cumul_data, summary_data, **kwargs)

    def plot_cumul(
        self, x_axis: str, y_axis: str, color: str = "k", ax: plt.Axes = None, **kwargs
    ) -> plt.Axes:
        self.check_axes(x_axis, y_axis, "cumul")
        pass

    def plot_summary(
        self, x_axis: str, y_axis: str, color: str = "k", ax: plt.Axes = None, **kwargs
    ) -> plt.Axes:

        self.check_axes(x_axis, y_axis, "summary")

        if ax is None:
            self.fig = plt.figure(figsize=kwargs.get("figsize", (8, 8)))
            ax = self.fig.add_subplot(1, 1, 1)
            if "figsize" in kwargs:
                kwargs.pop("figsize")

        # This gets actual training data
        plot_data = self.summary_data[
            self.summary_data["paradigm"] == "training_detection"
        ]

        x_axis_data = plot_data[x_axis].to_numpy()
        y_axis_data = plot_data[y_axis].to_numpy()

        ax = self.__plot__(ax, x_axis_data, y_axis_data, color=color, **kwargs)

        # prettify
        fontsize = kwargs.get("fontsize", 14)
        ax.set_xlabel(x_axis, fontsize=fontsize)
        ax.set_ylabel(y_axis, fontsize=fontsize)

        ax.tick_params(axis="x", rotation=45, length=20, width=2, which="major")
        ax.tick_params(axis="both", labelsize=fontsize)
        ax.grid(alpha=0.8, axis="both")

        return ax


class WheelDetectionPCFAPlotter(WheelDetectionProgressionPlotter):
    def __init__(
        self,
        animalid: str,
        cumul_data: pd.DataFrame = None,
        summary_data: pd.DataFrame = None,
        **kwargs,
    ) -> None:
        super().__init__(animalid, cumul_data, summary_data, **kwargs)

    def plot(
        self, x_axis: str = "session_difference", ax: plt.Axes = None, **kwargs
    ) -> plt.Axes:
        if ax is None:
            self.fig = plt.figure(figsize=kwargs.get("figsize", (8, 8)))
            ax = self.fig.add_subplot(1, 1, 1)
            if "figsize" in kwargs:
                kwargs.pop("figsize")

        ax = self.plot_summary(
            x_axis=x_axis, y_axis="hit_rate", color="forestgreen", ax=ax, **kwargs
        )
        ax2 = ax.twinx()
        ax2 = self.plot_summary(
            x_axis=x_axis, y_axis="false_alarm", color="maroon", ax=ax2, **kwargs
        )

        # prettify
        fontsize = kwargs.get("fontsize", 14)
        ax.set_xlabel(x_axis, fontsize=fontsize)
        ax.set_ylabel("Hit Rate (%)", fontsize=fontsize)
        ax2.set_ylabel("FA Rate (%)", fontsize=fontsize)

        ax.tick_params(axis="x", rotation=45, length=20, width=2, which="major")
        ax.tick_params(axis="both", labelsize=fontsize)
        ax.grid(alpha=0.8, axis="both")

        return ax


class WheelDetectionPsychometricOverlayPlotter(WheelDetectionBehaviorPlotter):
    __slots__ = []

    def __init__(
        self,
        animalid: str,
        session_list: list,
        stimkey: str = None,
        day_count: int = 9,
        **kwargs,
    ) -> None:
        super().__init__(animalid, session_list, stimkey, day_count, **kwargs)

    def plot(self, ax: plt.Axes = None, do_avg: bool = False, **kwargs) -> plt.Axes:
        if ax is None:
            self.fig = plt.figure(figsize=kwargs.get("figsize", (10, 8)))
            ax = self.fig.add_subplot(1, 1, 1)

        c_range = Color.make_color_range("#333333", len(self.plot_data))
        alpha_range = np.linspace(0.1, 1, len(self.plot_data))
        labels = []
        for i, date in enumerate(self.plot_data.keys()):
            p = DetectionPsychometricPlotter(self.plot_data[date])
            ax = p.plot(ax=ax, seperate_sides=False, alpha=alpha_range[i], zorder=i)
            labels.append(date)

        if do_avg:
            pass

        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, labels)


class WheelDetectionPastDaysGridSummary(WheelDetectionBehaviorPlotter):
    __slots__ = ["plot_type"]

    def __init__(
        self,
        animalid: str,
        session_list: list,
        stimkey: str = None,
        day_count: int = 9,
        plot_type: str = "summary",
        **kwargs,
    ):
        super().__init__(animalid, session_list, stimkey, day_count)
        self.plot_type = plot_type

    def init_plotters(self, data: dict) -> dict:
        if self.plot_type == "summary":
            return {
                "responsescatter": DetectionResponseHistogramPlotter(data, self.stimkey),
                "psychometric": DetectionPsychometricPlotter(data),
            }
        else:
            # do stuff here for choosing plot type like wheel, performance,licks etc..
            pass

    def plot(self, nrows=3, ncols=3, **kwargs) -> None:
        self.fig = plt.figure(figsize=kwargs.get("figsize", (20, 15)))
        main_gs = self.fig.add_gridspec(
            nrows=nrows,
            ncols=ncols,
            left=kwargs.get("left", 0),
            right=kwargs.get("right", 1),
            wspace=kwargs.get("wspace", 0.5),
            hspace=kwargs.get("hspace", 0.5),
        )
        row_nos = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        for i, date in enumerate(self.plot_data.keys()):
            row_no = row_nos[i]
            col_no = i % ncols
            sub_gs = main_gs[row_no, col_no].subgridspec(nrows=1, ncols=2, wspace=0.5)

            data = self.plot_data[date]
            plotters = self.init_plotters(data)

            ax_psych = self.fig.add_subplot(sub_gs[0, 0])
            ax_psych = plotters["psychometric"].plot(ax=ax_psych)
            ax_psych.set_title(date, fontsize=20)

            ax_resp = self.fig.add_subplot(sub_gs[0, 1])
            ax_resp = plotters["responsescatter"].plot(ax=ax_resp)
            ax_resp.set_title(
                f"PC%={self.plot_stats[date].all_correct_percent} \t HR={self.plot_stats[date].hit_rate}"
            )

        self.fig.tight_layout()

    def save(self, saveloc) -> None:
        if self.fig is not None:
            saveloc = pjoin(saveloc, "gridSummaries", self.animalid)
            if not os.path.exists(saveloc):
                os.makedirs(saveloc)
            last_date = list(self.plot_data.keys())[-1]
            savename = f"{last_date}_past9days_{self.plot_type}.pdf"
            saveloc = pjoin(saveloc, savename)
            self.fig.savefig(saveloc, bbox_inches="tight")
            display(f"Saved {savename} plot")


class WheelDetectionContrastLevelsPlotter(ContrastLevelsPlotter):
    def __init__(self, animalid: str, cumul_data, summary_data, **kwargs) -> None:
        super().__init__(animalid, cumul_data, summary_data, **kwargs)
        self.animalid = animalid
        self.cumul_data = self.add_difference_columns(self.cumul_data)

    def seperate_contrasts(self, do_opto: bool = False):
        """Seperates the contrast performances for each session throughout the training and experiments"""

        data = self.cumul_data[
            self.cumul_data["session_difference"] >= 0
        ]  # start from first actual training
        contrast_names = [str(c) for c in nonan_unique(data["contrast"])[::-1]]

        contrast_column_map = {name: idx for idx, name in enumerate(contrast_names)}
        contrast_column = np.zeros((len(contrast_column_map), 1))

        if do_opto:
            data = data[data["opto"] == 1]
        else:
            try:
                data = data[data["opto"] == 0]
            except:
                pass

        session_nos = np.unique(data["session_no"])
        all_sessions = np.zeros((len(contrast_names), len(session_nos)))
        all_sessions[:] = np.nan
        for k, s_no in enumerate(session_nos):
            sesh_data = data[data["session_no"] == s_no]

            sesh_contrasts = nonan_unique(
                sesh_data["contrast"]
            )  # this also removes the early trials which have np.nan values for contrasts

            contrast_column[:] = np.nan
            for i, c in enumerate(sesh_contrasts):
                c_data = sesh_data[sesh_data["contrast"] == c]
                key = str(c)

                if len(c_data):
                    correct_percent = len(c_data[c_data["answer"] == 1]) / len(c_data)
                    contrast_column[contrast_column_map[key]] = 100 * correct_percent
                else:
                    pass
            # concat the column to overall sessions image
            all_sessions[:, k] = np.ravel(contrast_column)

        self.contrast_column_map = contrast_column_map
        self.session_contrast_image = all_sessions

    def plot(
        self, ax: plt.Axes = None, cmap: str = "Reds", do_opto: bool = False, **kwargs
    ) -> plt.Axes:
        if ax is None:
            self.fig = plt.figure(figsize=kwargs.get("figsize", (15, 10)))
            ax = self.fig.add_subplot(1, 1, 1)

        self.seperate_contrasts(do_opto=do_opto)

        ax, im = self.__plot__(
            ax=ax, matrix=self.session_contrast_image, cmap=cmap, **kwargs
        )

        fontsize = 15
        ax.set_xlabel("Session from 1st Level1", fontsize=fontsize)
        ax.set_ylabel("Contrast Level (%)", fontsize=fontsize)
        x_axis = np.arange(self.session_contrast_image.shape[1])
        if x_axis[-1] % 5 != 0:
            x_axis = np.hstack((x_axis[::5], x_axis[-1]))
        else:
            x_axis = x_axis[::5]
        ax.set_xticks(x_axis)
        ax.set_yticks([v for k, v in self.contrast_column_map.items()])
        ax.set_yticklabels(
            [f"{100*float(k)}" for k, v in self.contrast_column_map.items()]
        )
        ax.tick_params(labelsize=fontsize)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_position(("outward", 10))
        ax.spines["bottom"].set_position(("outward", 10))

        # add space for colour bar
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.3 inch.

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.8)
        cbar = self.fig.colorbar(
            im, cax=cax, label="Hit Rate (%)", ticks=[0, 25, 50, 75, 100]
        )
        cax.tick_params(width=0, size=0, pad=5, labelsize=15)
        cbar.ax.spines[:].set_visible(False)
        cbar.ax.yaxis.set_label_position("left")
        cbar.ax.yaxis.set_ticks_position("left")

        return ax, cax


class WheelDetectionBehaviorSummaryPlotter(BehaviorBasePlotter):
    def __init__(
        self, animalid: str, cumul_data, summary_data, dateinterval: list = None, **kwargs
    ) -> None:
        super().__init__(animalid, cumul_data, summary_data, **kwargs)
        self.filter_dates(dateinterval)
        self.init_plotters()

    def init_plotters(self):
        self.plotters = {
            "weight": WeightProgressionPLotter(
                self.animalid, self.cumul_data, self.summary_data
            ),
            "perf": WheelDetectionPCFAPlotter(
                self.animalid, self.cumul_data, self.summary_data
            ),
        }

    def plot(self, ax: plt.Axes = None, **kwargs) -> plt.Axes:

        self.fig = plt.figure(figsize=kwargs.get("figsize", (15, 10)))
        widths = [1, 1]
        heights = [1, 1]

        gs = self.fig.add_gridspec(
            ncols=len(widths),
            nrows=len(heights),
            width_ratios=widths,
            height_ratios=heights,
            left=kwargs.get("left", 0),
            right=kwargs.get("right", 1),
            wspace=kwargs.get("wspace", 0.3),
            hspace=kwargs.get("hspace", 0.4),
        )

        ax_weight = self.fig.add_subplot(gs[0, 0])
        ax_weight = self.plotters["weight"].plot(ax=ax_weight, marker="o", markersize=5)

        # plot %80-%90 line

        ax_perf = self.fig.add_subplot(gs[1, 0])
        ax_perf = self.plotters["perf"].plot(ax=ax_perf, marker="o", markersize=5)

        self.fig.tight_layout()
