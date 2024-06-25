from piepy.detection.wheelDetectionAnalysis import DetectionAnalysis
from piepy.plotters.bokeh_plot.bokeh_base import *
from piepy.plotters.plotter_utils import Color

from bokeh.models import Whisker, TeeHead
import polars as pl
import numpy as np


class PsychometricGraph(Graph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cds_dots = {}
        self.cds_curves = {}
        self.stat_analysis = DetectionAnalysis()
        self.color = Color()

    @staticmethod
    def _makelabel(name: np.ndarray, count: np.ndarray) -> str:
        ret = f"""\nN=["""
        for i, n in enumerate(name):
            ret += rf"""{float(n)}:$\bf{count[i]}$, """
        ret = ret[:-2]  # remove final space and comma
        ret += """]"""
        return ret

    @staticmethod
    def _jitter_contrast(contrast_in: np.ndarray) -> list:
        jitter_rng = 0.3
        jittered = contrast_in + np.random.uniform(-jitter_rng / 2, jitter_rng / 2)
        jittered[0] += np.random.uniform(0, jitter_rng) / 100
        return jittered

    def reset_cds(self) -> None:
        self.cds_dots = {}
        self.cds_curves = {}

    def set_cds(self, data: pl.DataFrame, **kwargs) -> None:
        self.stat_analysis.set_data(data)
        q = self.stat_analysis.agg_data.drop_nulls().sort(
            ["stimkey", "opto_pattern"], descending=True
        )

        self.c_key = np.unique(q["signed_contrast"].to_numpy())
        self.c_val = np.arange(-int(len(self.c_key) / 2), int(len(self.c_key) / 2) + 1)
        if 0 not in self.c_key:
            self.c_val = np.delete(self.c_val, int(len(self.c_val) / 2))
        contrast_map = {self.c_key[i]: c for i, c in enumerate(self.c_val)}

        temp_dots = {}
        temp_curves = {}
        for i, k in enumerate(self.possible_stims):
            filt_df = q.filter(pl.col("stimkey") == k)
            if not filt_df.is_empty():

                contrast = np.array(
                    [contrast_map[c] for c in filt_df["signed_contrast"].to_numpy()]
                )
                contrast = self._jitter_contrast(contrast)

                hit_rates = [100 * h for h in filt_df["hit_rate"].to_list()]
                confs = [100 * c for c in filt_df["confs"].to_list()]

                temp_dots[k] = {
                    "contrast": contrast,
                    "signed_contrast": filt_df["signed_contrast"].to_list(),
                    "hit_rate": hit_rates,
                    "confs_up": [x + e for x, e in zip(hit_rates, confs)],
                    "confs_down": [x - e for x, e in zip(hit_rates, confs)],
                    "correct_count": filt_df["correct_count"].to_list(),
                    "miss_count": filt_df["miss_count"].to_list(),
                    "label": filt_df["stim_label"].to_list(),
                    "color": [self.color.stim_keys[k]["color"]] * len(contrast),
                }

                if "-1" in k and filt_df[0, "stim_side"] == "catch":
                    # catch trial in nonopto trials
                    baseline = [filt_df[0, "hit_rate"]]
                    temp_curves["baseline"] = {"y": baseline}
            else:
                temp_dots[k] = {
                    "contrast": [],
                    "signed_contrast": [],
                    "hit_rate": [],
                    "confs_up": [],
                    "confs_down": [],
                    "correct_count": [],
                    "miss_count": [],
                    "label": [],
                    "color": [],
                }
                temp_curves["baseline"] = {"y": []}

        baseline_df = q.filter(
            (pl.col("opto_pattern") == -1) & (pl.col("stim_side") == "catch")
        )
        temp_curves["baseline"] = {
            "y": [100 * np.mean(baseline_df["hit_rate"].to_list())]
        }

        if len(self.cds_dots) == 0:
            for k, v in temp_dots.items():
                self.cds_dots[k] = ColumnDataSource(data=v)

            for k, v in temp_curves.items():
                self.cds_curves[k] = ColumnDataSource(data=v)
        else:
            for k, v in temp_dots.items():
                self.cds_dots[k].data = v
            for k, v in temp_curves.items():
                self.cds_curves[k].data = v

    def plot(self, **kwargs) -> None:
        f = figure(
            title="",
            width=450,
            height=400,
            x_axis_label="Contrast(%)",
            y_axis_label="Hit Rate(%)",
            x_range=(-3, 3),
            y_range=(-5, 105),
        )
        f.grid.grid_line_alpha = 0.5

        f.hspan(y=50, line_dash="dashed", color="#000000", line_width=3)
        f.vspan(x=0, line_dash="dashed", color="#000000", line_width=3)

        for kk, source_c in self.cds_curves.items():
            if kk == "baseline":
                f.hspan(
                    y="y",
                    source=source_c,
                    color="#b00000",
                    line_dash="dotted",
                    line_width=3,
                )

        for k, source in self.cds_dots.items():
            f.circle(
                x="contrast",
                y="hit_rate",
                color="color",
                size=15,
                line_alpha=0,
                source=source,
            )

            f.add_layout(
                Whisker(
                    source=source,
                    base="contrast",
                    upper="confs_up",
                    lower="confs_down",
                    line_color="color",
                    line_width=3,
                    upper_head=TeeHead(line_width=0),
                    lower_head=TeeHead(line_width=0),
                )
            )

        hover = HoverTool(
            tooltips=[
                ("Contrast", "@signed_contrast"),
                ("Correct Count", "@correct_count"),
                ("Miss Count", "@miss_count"),
            ]
        )
        f.add_tools(hover)

        # TODO: Make this programmatic
        f.xaxis.ticker = [-2, -1, 0, 1, 2]
        f.xaxis.major_label_overrides = {-2: "50", -1: "12.5", 0: "0", 1: "12.5", 2: "50"}

        self.fig = f
