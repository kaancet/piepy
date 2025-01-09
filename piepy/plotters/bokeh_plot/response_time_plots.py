from piepy.plotters.bokeh_plot.bokeh_base import *
from piepy.plotters.plotting_utils import Color

from bokeh.transform import jitter
from bokeh.models import LogScale, LinearScale
import polars as pl
import numpy as np


class ReactionTimeScatterGraph(Graph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cds_dots = None
        self.cds_curves = None
        self.f_log = None
        self.f_linear = None
        self.set_reaction_type()
        self.set_include_miss()
        self.color = Color()

    def set_reaction_type(self, resp_type: str = "response_latency") -> None:
        self.reaction_type = resp_type

    def set_include_miss(self, include_miss: bool = False) -> None:
        self.include_miss = include_miss

    def reset_cds(self) -> None:
        self.cds_dots = None
        self.cds_curves = None

    def set_cds(self, data: pl.DataFrame) -> None:
        temp = data.drop_nulls(subset=["contrast"])
        if not self.include_miss:
            temp = temp.filter(pl.col("outcome") == 1)
        temp = temp.select(
            [
                "trial_no",
                "signed_contrast",
                "response_latency",
                "pos_reaction_time",
                "speed_reaction_time",
                "stimkey",
            ]
        )
        temp = temp.with_columns((pl.col(self.reaction_type) * 1).alias("reaction_time"))

        self.c_key = np.unique(temp["signed_contrast"].to_numpy())
        self.c_val = np.arange(-int(len(self.c_key) / 2), int(len(self.c_key) / 2) + 1)
        if 0 not in self.c_key:
            self.c_val = np.delete(self.c_val, int(len(self.c_val) / 2))
        contrast_map = {self.c_key[i]: c for i, c in enumerate(self.c_val)}

        colors = [self.color.stim_keys[c]["color"] for c in temp["stimkey"].to_list()]
        df = temp.to_pandas(use_pyarrow_extension_array=True)
        df["color"] = colors
        df["contrast_axis"] = [contrast_map[c] for c in df["signed_contrast"]]

        q = (
            temp.lazy()
            .group_by(["stimkey", "signed_contrast"])
            .agg([(pl.col("reaction_time").median().alias("median_reaction_time"))])
            .sort(["stimkey", "signed_contrast"])
        )
        df_curves = q.drop_nulls().collect().to_pandas(use_pyarrow_extension_array=True)
        colors_curve = [
            self.color.stim_keys[c]["color"] for c in df_curves["stimkey"].to_list()
        ]
        df_curves["color"] = colors_curve
        df_curves["x0"] = df_curves["signed_contrast"] - 5
        df_curves["x1"] = df_curves["signed_contrast"] + 5

        df_curves["contrast_axis"] = [
            contrast_map[c] for c in df_curves["signed_contrast"]
        ]
        df_curves["x0_axis"] = df_curves["contrast_axis"] - 0.3
        df_curves["x1_axis"] = df_curves["contrast_axis"] + 0.3

        if self.cds_dots is None:
            self.cds_dots = ColumnDataSource(data=df)
            self.cds_curves = ColumnDataSource(data=df_curves)
        else:
            self.cds_dots.data = df
            self.cds_curves.data = df_curves

    def plot(self):
        f = figure(
            title="",
            width=450,
            height=400,
            y_axis_type="log",
            x_axis_label="Contrast(%)",
            y_axis_label="Response Time(ms)",
            x_range=(-3, 3),
        )

        f.vspan(x=0, line_dash="dashed", color="#000000", line_width=3)
        f.hspan(y=1000, line_dash="dashdot", color="#f00000", line_width=3)

        f.circle(
            x=jitter("contrast_axis", width=0.3, distribution="uniform"),
            y="reaction_time",
            size=10,
            color="color",
            source=self.cds_dots,
            alpha=0.6,
        )

        f.segment(
            x0="x0_axis",
            y0="median_reaction_time",
            x1="x1_axis",
            y1="median_reaction_time",
            line_color="color",
            line_width=5,
            source=self.cds_curves,
        )

        hover = HoverTool(
            tooltips=[
                ("Trial No", "@trial_no"),
                ("Contrast", "@signed_contrast"),
                ("Response Time", "@response_latency"),
                ("Wheel Reaction Time", "@pos_reaction_time"),
                ("Wheel Speed reaction Time", "@speed_reaction_time"),
            ]
        )
        f.add_tools(hover)

        self.fig = f
