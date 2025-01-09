import polars as pl
from .plotting_utils import *
from ..core.mouse import MouseData
from .basePlotters import BasePlotter


class BehaviorBasePlotter(BasePlotter):
    def __init__(self, data: MouseData = None, **kwargs) -> None:
        self.fig = None
        self.data = None
        self.color = Color()
        set_style(kwargs.pop("style", "presentation"))
        self.plot_type = kwargs.pop("plot_type", "cumulative")
        if data is not None:
            self.set_data(data)

    @property
    def plot_type(self) -> str:
        return self._plot_type

    @plot_type.setter
    def plot_type(self, val: str) -> None:
        if val not in ["cumulative", "summary"]:
            raise ValueError("plot type can only be cumulative or summary!")
        self._plot_type = val

    def set_data(
        self,
        data: MouseData = None,
        cumul_data: pl.DataFrame = None,
        summary_data: pl.DataFrame = None,
    ) -> None:
        """Sets the data for the plotter"""
        if data is None:
            if cumul_data is not None and summary_data is not None:
                super().set_data(cumul_data)  # this goes to base plotter
                self.summary_data = summary_data
            else:
                raise ValueError(
                    "Both cumulative and summary data needs to be given if no MouseData object is provided"
                )
        else:
            super().set_data(data.cumul_data)  # this goes to base plotter
            self.summary_data = data.summary_data
        self.animalid = self.plot_data[0, "animalid"]

    def filter_dates(
        self, dateinterval: list = None
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Filters both summary and cumul data according to given date interval"""
        if dateinterval is None:
            return None

        if len(dateinterval) > 2:
            raise ValueError(
                f"dateinterval argument needs to have 2 dates max, got {len(dateinterval)}"
            )
        elif len(dateinterval) == 1:
            dateinterval.append(dt.today().strftime("%y%m%d"))

        try:
            dateinterval_dt = [dt.strptime(d, "%y%m%d").date() for d in dateinterval]
        except:
            raise ValueError(
                f"dates need to be in YYMMDD format, got {dateinterval[0]} instead"
            )

        ret_summ = self.summary_data.filter(
            (pl.col("dt_date").is_between(dateinterval_dt[0], dateinterval_dt[1]))
        )
        ret_cumm = self.cumul_data.filter(
            (pl.col("date").is_between(dateinterval_dt[0], dateinterval_dt[1]))
        )

        return ret_summ, ret_cumm

    # def check_axes(self, x_axis, y_axis, data_type: str = "summary") -> None:
    #     if data_type == "summary":
    #         data_to_check = self.summary_data

    #         if x_axis not in ["session_difference", "day_difference", "dt_date"]:
    #             raise KeyError(
    #                 f"""{x_axis} is not a valid value for x_axis, try one of \n - session_difference, \n - day_difference,\n - dt_date"""
    #             )

    #     elif data_type == "cumul":
    #         data_to_check = self.cumul_data

    #     if y_axis not in data_to_check.columns:
    #         raise KeyError(
    #             f"""{y_axis} is not a valid value for y_axis, try one of {self.summary_data.columns}"""
    #         )

    # @staticmethod
    # def add_difference_columns(data:pl.DataFrame) -> None:
    #     """Adds difference columns to the plot data like the day difference, session difference"""
    #     try:
    #         # order by date(just in case)
    #         data.sort(pl.col("dt_date"))
    #         _train_data = data.filter((pl.col("task")=="SimpleDetectionTaskController") )

    #         start_day = data[data["paradigm"].str.contains("training", na=False)][
    #             "dt_date"
    #         ].iloc[0]
    #         sesh_idx = data.index[data["dt_date"] == start_day].to_list()[0]
    #         start_sesh = data["session_no"].iloc[int(sesh_idx)]
    #     except:
    #         start_day = data["dt_date"].iloc[0]
    #         sesh_idx = len(data) - 1
    #         start_sesh = data["session_no"].iloc[int(sesh_idx)]

    #     # day diff
    #     data["day_difference"] = dates_to_deltadays(data["dt_date"].to_numpy(), start_day)

    #     # session_diff
    #     data["session_difference"] = data.apply(
    #         lambda x: (
    #             x["session_no"] - start_sesh
    #             if not np.isnan(x["session_no"])
    #             else x.name - sesh_idx
    #         ),
    #         axis=1,
    #     )
    #     return data


# class BehaviorProgressionPlotter(BehaviorBasePlotter):
#     """This is a general progression plotter class which can be extended to plot specific progressions
#     such as weight, performance, responsetime, etc.
#     It has a plotting function that takes the x and y axis and color values"""

#     def __init__(
#         self,data: MouseData = None, **kwargs) -> None:
#         super().__init__(data, **kwargs)

#     def check_axes(self, x_axis, y_axis, data_type: str = "summary") -> None:
#         if data_type == "summary":
#             data_to_check = self.summary_data

#             if x_axis not in ["session_difference", "day_difference", "dt_date"]:
#                 raise KeyError(
#                     f"""{x_axis} is not a valid value for x_axis, try one of \n - session_difference, \n - day_difference,\n - dt_date"""
#                 )

#         elif data_type == "cumul":
#             data_to_check = self.cumul_data

#         if y_axis not in data_to_check.columns:
#             raise KeyError(
#                 f"""{y_axis} is not a valid value for y_axis, try one of {self.summary_data.columns}"""
#             )

#     @staticmethod
#     def __plot__(ax, x, y, color, **kwargs) -> plt.Axes:
#         ax.plot(x, y, color, linewidth=3, **kwargs)
#         return ax


# class BehaviorScatterPlotter(BehaviorBasePlotter):
#     """This is a general scatter plotter class which can be extended to plot specific
#     such as weight, performance, responsetime, etc.
#     It has a plotting function that takes the x and y axis and color values"""

#     def __init__(
#         self,
#         animalid: str,
#         cumul_data: pd.DataFrame,
#         summary_data: pd.DataFrame,
#         **kwargs,
#     ) -> None:
#         super().__init__(animalid, cumul_data, summary_data, **kwargs)

#     def check_axes(self, x_axis, y_axis, data_type: str = "summary") -> None:
#         if data_type == "summary":
#             data_to_check = self.summary_data
#         elif data_type == "cumul":
#             data_to_check = self.cumul_data

#         try:
#             tmp = data_to_check[x_axis]
#         except KeyError as k:
#             raise KeyError(
#                 f"The column name {x_axis} is not valid for x_axis, try one of:\n{self.summary_data.columns}"
#             )

#         try:
#             tmp = data_to_check[y_axis]
#         except KeyError as k:
#             raise KeyError(
#                 f"The column name {y_axis} is not valid for x_axis, try one of:\n{self.summary_data.columns}"
#             )

#     @staticmethod
#     def __plot__(ax, x, y, **kwargs):
#         ax.scatter(x, y, **kwargs)
#         return ax


# class ContrastLevelsPlotter(BehaviorBasePlotter):
#     __slots__ = ["animalid", "session_contrast_image", "contrast_column_map", "cbar"]

#     def __init__(self, animalid: str, cumul_data, summary_data, **kwargs) -> None:
#         super().__init__(cumul_data, summary_data, **kwargs)
#         self.animalid = animalid
#         self.cumul_data = self.add_difference_columns(self.cumul_data)

#     @staticmethod
#     def __plot__(ax: plt.Axes, matrix, cmap, **kwargs):
#         im = ax.imshow(matrix, vmin=0, vmax=100, cmap=cmap)
#         return ax, im


# class WeightProgressionPLotter(BehaviorProgressionPlotter):
#     def __init__(
#         self,
#         animalid: str,
#         cumul_data: pd.DataFrame,
#         summary_data: pd.DataFrame,
#         **kwargs,
#     ) -> None:
#         super().__init__(animalid, cumul_data, summary_data, **kwargs)

#     def plot(self, x_axis: str = "session_difference", ax: plt.Axes = None, **kwargs):
#         if ax is None:
#             self.fig = plt.figure(figsize=kwargs.get("figsize", (8, 8)))
#             ax = self.fig.add_subplot(1, 1, 1)
#             if "figsize" in kwargs:
#                 kwargs.pop("figsize")

#         water_res_starts = self.summary_data[
#             self.summary_data["paradigm"] == "water restriction start"
#         ]

#         for i in range(
#             10
#         ):  # arbitrary check count, should not be this many water restriction start/stops in reality
#             latest_water_restriction_weight = water_res_starts["weight"].iloc[-1 - i]
#             if not np.isnan(latest_water_restriction_weight):
#                 break

#         x_axis_data = self.summary_data[x_axis].to_numpy()
#         y_axis_data = self.summary_data["weight"].to_numpy()

#         ax = self.__plot__(ax, x=x_axis_data, y=y_axis_data, color="k", **kwargs)

#         ax.axhline(
#             y=latest_water_restriction_weight * 0.9,
#             color="orange",
#             linewidth=2,
#             linestyle=":",
#         )

#         ax.axhline(
#             y=latest_water_restriction_weight * 0.8,
#             color="red",
#             linewidth=2,
#             linestyle=":",
#         )

#         # prettify
#         fontsize = kwargs.get("fontsize", 14)
#         ax.set_xlabel(x_axis, fontsize=fontsize)
#         ax.set_ylabel("Weight (g)", fontsize=fontsize)

#         ax.tick_params(axis="x", rotation=45, length=20, width=2, which="major")
#         ax.tick_params(axis="both", labelsize=fontsize)
#         ax.grid(alpha=0.8, axis="both")

#         return ax
