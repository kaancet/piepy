import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from ..plotting_utils import make_subsets, set_style
from ..color import Color


class ProgressPlotter:
    data: pl.DataFrame = None
    figs: list = []
    color: Color = Color()

    def __init__(self, data: pl.DataFrame = None):
        self.set_data(data)

    @staticmethod
    def set_style(style_name: str = "print") -> None:
        set_style(style_name)

    @classmethod
    def reset(cls) -> None:
        cls.data = None
        cls.figs = []

    @classmethod
    def set_data(cls, data: pl.DataFrame = None) -> None:
        """Sets the data of the plotter"""
        cls.data = data

    @staticmethod
    def _set_x_axis(data: pl.DataFrame, is_time: bool = False) -> tuple[str, np.ndarray]:
        """Returns the label title and xaxis depending on is_time plotting flag"""
        if is_time:
            _x_label = "Time (mins)"
            _x = data["t_trialend"].to_numpy() / 60_000
        else:
            _x = data["trial_no"].to_numpy()
            _x_label = "Trial No"

        return _x_label, _x

    @classmethod
    def plot_performance(
        cls,
        ax: plt.Axes = None,
        plot_in_time: bool = False,
        seperate_by: list = ["stimkey"],
        running_window: int = None,
        mpl_kwargs: dict = {},
        **kwargs,
    ) -> plt.Axes:
        """Plots the accuracy of subset of trials through the run"""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=mpl_kwargs.pop("figsize", (15, 8)))
            cls.figs.append(fig)

        def _get_perf_(arr: np.ndarray) -> np.ndarray:
            """Returns the hit rates as an array"""
            arr = arr.astype(float)
            # convert -1(earlies) to np.nan
            arr[arr == -1] = np.nan
            # cumulative count of hits ignoring nans (1 hit, 0 miss)
            _hits = np.nancumsum(arr)
            # cumulative counts of all not nans (earlies in this case)
            _non_earlies = np.cumsum(~np.isnan(arr))
            return (_hits / _non_earlies) * 100

        for subs in make_subsets(cls.data, seperate_by):
            _dat = subs[-1]
            clr = {}
            if seperate_by == "stimkey":
                clr = cls.color.stim_keys[subs[0]]
            elif seperate_by == "contrast":
                clr = cls.color.contrast_keys[subs[0]]

            _x_label, x = cls._set_x_axis(_dat, plot_in_time)
            y = _get_perf_(_dat["state_outcome"].to_numpy())
            if running_window is not None:
                y = np.rolling(running_window).mean(y)

            ax.plot(x, y, label=f"{subs[:-1]}", **clr, **mpl_kwargs)

        ax.set_ylim([0, 110])
        ax.set_xlabel(_x_label)
        ax.set_ylabel("Accuracy(%)")
        return ax

    @classmethod
    def plot_reactiontime(
        cls,
        ax: plt.Axes = None,
        reaction_of: str = "state_response_time",
        include_misses: bool = False,
        plot_in_time: bool = False,
        seperate_by: list = ["stimkey"],
        running_window: int = None,
        mpl_kwargs: dict = {},
        **kwargs,
    ) -> plt.Axes:
        """ """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=mpl_kwargs.pop("figsize", (15, 8)))
            cls.figs.append(fig)

        if reaction_of not in cls.data.columns:
            raise ValueError(f"{reaction_of} not in data columns!")

        for subs in make_subsets(cls.data, seperate_by):
            _dat = subs[-1]
            clr = {}
            if seperate_by == "stimkey":
                clr = cls.color.stim_keys[subs[0]]
            elif seperate_by == "contrast":
                clr = cls.color.contrast_keys[subs[0]]

            _x_label, x = cls._set_x_axis(_dat, plot_in_time)
            y = _dat[reaction_of].to_numpy()

            if running_window is not None:
                y = np.rolling(running_window).mean(y)

            ax.plot(x, y, label=f"{subs[:-1]}", **clr, **mpl_kwargs)

        ax.set_xlabel(_x_label)
        # parse the axis label
        ax.set_ylabel(f'{reaction_of.replace("_", " ").capitalize()} (ms)')

        # ax.set_yscale("symlog")
        # minor_locs = [200, 400, 600, 800, 2000, 4000, 6000, 8000]
        # ax.yaxis.set_minor_locator(plt.FixedLocator(minor_locs))
        # ax.yaxis.set_minor_formatter(plt.FormatStrFormatter("%d"))
        # ax.yaxis.set_major_locator(ticker.FixedLocator([100, 1000, 10000]))
        # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))

        return ax

    @classmethod
    def plot_lick(
        cls,
        ax: plt.Axes = None,
        plot_in_time: bool = False,
        running_window: int = None,
        mpl_kwargs: dict = {},
        **kwargs,
    ) -> plt.Axes:
        """ """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=mpl_kwargs.pop("figsize", (15, 8)))
            cls.figs.append(fig)

        if plot_in_time:
            _x_label = "Time (mins)"
            x = cls.data["lick"].explode().drop_nulls().to_numpy() / 60_000
        else:
            _x_label = "Trial No"
            licks = cls.data["lick"].to_list()
            trial_nos = cls.data["trial_no"].to_list()
            listed_trial_nos = [
                [t] * len(l) for t, l in zip(trial_nos, licks) if l is not None
            ]  # noqa: E741
            df = cls.data.with_columns(pl.Series("lick_trial_nos", listed_trial_nos))
            x = df["lick_trial_nos"].explode().to_numpy()

        y = np.arange(1, len(x) + 1)
        ax.plot(x, y, color="#00BBFF")

        ax.set_xlabel(_x_label)
        # parse the axis label
        ax.set_ylabel("Lick count")
