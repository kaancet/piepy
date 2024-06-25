from piepy.wheelUtils import get_trajectory_stats
import copy
import os
import sys
import numpy as np
from collections import defaultdict
import bokeh.plotting as bok
from bokeh.models import ColumnDataSource, Whisker, TeeHead, HoverTool, FactorRange
from bokeh.palettes import Spectral11, Set1_9

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from piepy.wheel.wheelAnalysis import WheelAnalysis
from piepy.wheelUtils import find_nearest

iter_stim = iter(Spectral11)
stim_styles = defaultdict(lambda: next(iter_stim))

iter_contrast = iter(Set1_9)
contrast_styles = defaultdict(lambda: next(iter_contrast))


class DashBoard:
    def __init__(self):
        self.widgets = {}
        self.graphs = {}
        self.callbacks = {}

        self.current_animal = None
        self.current_session = None
        self.scope_list = None
        self.data = None
        self.current_scope = None

        self.show_repeat = False
        self.show_individual = False
        self.time_axis = False

    def add_graph(self, graph_name, graph2add):
        self.graphs[graph_name] = graph2add

    def add_widget(self, widget_name, widget2add):
        self.widgets[widget_name] = widget2add

    def init_data(self, data_in):
        """set the data, also sets the available scope list"""
        self.data = data_in
        self.shown_data = copy.deepcopy(self.data)
        self.scope_list = [
            k for k in self.data["novel_stim_data"].keys() if k != "overall"
        ]
        # init the scope with first one in list
        self.current_scope = self.scope_list[0]
        self.prep_meta()
        self.prep_stats()
        self.set_data()

    def set_data(self):
        """Sets the shown data"""
        if self.data is not None:
            self.shown_data = copy.deepcopy(self.data)
            if self.show_repeat:
                self.shown_data = {
                    self.current_scope: self.shown_data["data"][self.current_scope]
                }
            else:
                self.shown_data = {
                    self.current_scope: self.shown_data["novel_stim_data"][
                        self.current_scope
                    ]
                }
            print("Dash - Set data {0}".format(self.current_scope), flush=True)

            self.shown_trial = self.shown_data[self.current_scope].iloc[0].to_dict()

    def set_trial(self, trial_no):
        """Sets the shown trial"""
        temp = self.shown_data[self.current_scope]
        self.shown_trial = temp[temp["trial_no"] == trial_no].iloc[0].to_dict()

    def prep_meta(self):
        """Prepeares the meta data that can be shown in a div with relevant HTML tags"""
        self.meta_data = "<pre>"
        for key, val in self.data["meta"].items():
            self.meta_data += "<b>{0} :</b> {1}\n".format(key, val)

        self.meta_data += "</pre>"

    def prep_stats(self):
        """Prepeares the meta data that can be shown in a div with relevant HTML tags"""
        self.stats = {}
        for scope in self.data["summaries"].keys():
            temp = "<pre>"
            for key, val in self.data["summaries"][scope].items():
                temp += "<b>{0} :</b> {1}\n".format(key, val)
            temp += "</pre>"
            self.stats[scope] = temp

    def change_repeat(self, make_novel=False):
        """Changes whether to show repeat trials or not"""
        self.show_repeat = make_novel
        print("Dash - Change repeat {0}".format(self.show_repeat), flush=True)
        self.set_data()

    def change_scope(self, scope):
        """Changes the scope of the shown data"""
        if self.scope_list is not None:
            if scope in self.scope_list:
                self.current_scope = scope
            else:
                raise ValueError(
                    "Data scope {0} does not exist, try one of {1}".format(
                        scope, self.scope_list
                    )
                )
        print("Dash - Change scope", flush=True)
        self.set_data()

    def filter_data(self, filters=None):
        if filters is None:
            filters = {}
        else:
            if self.data is not None:
                if "trial_limit" in filters.keys():
                    self.shown_data = copy.deepcopy(self.data)
                    if isinstance(self.shown_data, dict):
                        self.shown_data = {
                            k: v[v["trial_no"] <= filters["trial_limit"]]
                            for k, v in self.shown_data.items()
                        }
                    else:
                        self.shown_data = self.shown_data[
                            self.shown_data["trial_no"] <= filters["trial_limit"]
                        ]
                elif "response_cutoff" in filters.keys():
                    self.shown_data = copy.deepcopy(self.data)
                    if isinstance(self.shown_data, dict):
                        self.shown_data = {
                            self.shown_data[k]: v[
                                v["response_latency"] <= filters["response_cutoff"]
                            ]
                            for k, v in self.shown_data.items()
                        }
                    else:
                        self.shown_data = self.shown_data[
                            self.shown_data["response_latency"]
                            <= filters["response_cutoff"]
                        ]


class Graph:
    def __init__(self, *args, **kwargs):
        self.cds = {}

    def save(self, plotname):
        pass

    def plot(self, *args, **kwargs):
        pass

    def init_cds(self):
        pass

    @staticmethod
    def line_color_from_answer(x):
        if x == -1:
            return "#ba1401"
        elif x == 1:
            return "#ffffff"
        elif x == 0:
            return "#a3a3a3"

    @staticmethod
    def dot_color_from_answer(x):
        if x == -1:
            return "#ba1401"
        elif x == 1:
            return "#048e26"
        elif x == 0:
            return "#a3a3a3"

    def pretty_axes(self, fig, fontsize):
        """Makes simple pretty axes"""
        fig.xaxis.axis_line_width = 2
        fig.yaxis.axis_line_width = 2

        if fontsize is None:
            fontsize = 22
        fig.axis.axis_label_text_font_size = self.pt_font(fontsize)
        fig.axis.axis_label_text_font_style = "normal"
        fig.axis.major_label_text_font_size = self.pt_font(fontsize)
        return fig

    def pt_font(self, int_font):
        return "{0}pt".format(int_font)


class Psychometric(Graph):
    def __init__(self, *args, **kwargs):
        self.fig = None
        self.cds_dots = {}
        self.cds_curves = {}
        self.fitter = WheelAnalysis()
        self.side = "right"

    def plot(self, *args, **kwargs):
        fontsize = kwargs.get("fontsize", 14)

        fig = bok.figure(
            toolbar_location=kwargs.get("toolbarlocation", "right"),
            tools="pan,box_zoom,wheel_zoom,reset",
            plot_width=500,
            plot_height=500,
        )

        # midlines
        fig.line(
            [0, 0],
            [0, 1],
            line_color="gray",
            line_dash="dotted",
            line_width=2,
            line_alpha=0.7,
            level="underlay",
        )
        fig.line(
            [-100, 100],
            [0.5, 0.5],
            line_color="gray",
            line_dash="dotted",
            line_width=2,
            line_alpha=0.7,
            level="underlay",
        )

        for scope in self.cds_dots.keys():

            fig.circle(
                x="contrast_x",
                y="prob",
                size=kwargs.get("markersize", 15),
                fill_color=stim_styles[scope],
                line_color=kwargs.get("markeredgecolor", "#ffffff"),
                line_width=kwargs.get("markeredgewidth", 2),
                source=self.cds_dots[scope],
                legend_label="{0}".format(scope),
                name="points",
            )

            fig.add_layout(
                Whisker(
                    source=self.cds_dots[scope],
                    base="contrast_x",
                    upper="err_up",
                    lower="err_down",
                    line_color=stim_styles[scope],
                    line_width=kwargs.get("elinewidth", 3),
                    upper_head=TeeHead(line_width=0),
                    lower_head=TeeHead(line_width=0),
                )
            )
            fig.line(
                x="contrast_x",
                y="prob",
                line_color=stim_styles[scope],
                level="underlay",
                line_width=kwargs.get("linewidth", 9),
                line_cap="round",
                source=self.cds_curves[scope],
            )

        fig = self.pretty_axes(fig, fontsize)
        fig.xaxis.axis_label = "Contrast"
        fig.yaxis.axis_label = "Prob. Choosing Right(%)"

        fig.yaxis.bounds = (0, 100)
        fig.legend.location = "top_left"
        fig.legend.label_text_font_size = self.pt_font(fontsize - 5)

        if not kwargs.get("showlegend", False):
            fig.legend.visible = False
        # add hover tool
        fig.add_tools(
            HoverTool(
                names=["points"],
                tooltips=[("Trial Count", "@trial_count"), ("%95", "@err")],
                mode="mouse",
            )
        )

        self.fig = fig

    def set_cds(self, data, **kwargs):
        """Creates a column data source(cds)"""
        if not isinstance(data, dict):
            raise TypeError(
                "Data should be in an dict. If it" "s a DataFrame put it in a dict."
            )

        self.fitter.set_data(data, keep_overall=False)
        fitted_data = self.fitter.curve_fit(model="erf_psycho2", **kwargs)

        for scope, data in fitted_data.items():

            temp_dots = {
                "contrast_x": 100 * np.array(data[self.side]["contrast"]),
                "prob": data[self.side]["percentage"],
                "err": data[self.side]["confidence"],
                "err_up": [
                    x + e
                    for x, e in zip(
                        data[self.side]["percentage"], data[self.side]["confidence"]
                    )
                ],
                "err_down": [
                    x - e
                    for x, e in zip(
                        data[self.side]["percentage"], data[self.side]["confidence"]
                    )
                ],
                # 'prob_left' : data['left']['percentage'],
                # 'prob_nogo' : data['nogo']['percentage'],
                # 'err_right' : data['right']['confidence'],
                # 'err_left'  : data['left']['confidence'],
                # 'err_nogo' : data['nogo']['confidence'],
                # 'error_up_right' : [x+e for x,e in zip(data['right']['percentage'],data['right']['confidence'])],
                # 'error_down_right' : [x-e for x,e in zip(data['right']['percentage'],data['right']['confidence'])],
                # 'error_up_left' : [x+e for x,e in zip(data['left']['percentage'],data['left']['confidence'])],
                # 'error_down_left' : [x-e for x,e in zip(data['left']['percentage'],data['left']['confidence'])],
                # 'error_up_nogo' : [x+e for x,e in zip(data['nogo']['percentage'],data['nogo']['confidence'])],
                # 'error_down_nogo' : [x-e for x,e in zip(data['nogo']['percentage'],data['nogo']['confidence'])],
                "trial_count": self.fitter.analysis["summary"][scope]["contrast_count"],
            }

            temp_curves = {
                "contrast_x": 100 * data[self.side]["fitted_x"],
                "prob": data[self.side]["fitted_y"],
            }

            if scope not in self.cds_dots.keys():
                print("PERFORMANCE GRAPH - Setting Data", flush=True)
                self.cds_dots[scope] = ColumnDataSource(data=temp_dots)
                self.cds_curves[scope] = ColumnDataSource(data=temp_curves)
            else:
                print("PERFORMANCE GRAPH - Updating data", flush=True)
                self.cds_dots[scope].data = temp_dots
                self.cds_curves[scope].data = temp_curves


class Performance(Graph):
    def __init__(self, *args, **kwargs):
        self.fig = None
        self.cds = {}

    def plot(self, *args, **kwargs):
        fontsize = kwargs.get("fontsize", 14)
        # set up axes
        fig = bok.figure(
            toolbar_location=kwargs.get("toolbarlocation", "above"),
            tools="pan,box_zoom,wheel_zoom,reset",
            plot_width=800,
            plot_height=400,
        )

        for scope in self.cds.keys():
            fig.line(
                x="x_axis",
                y="performance_percent",
                line_color=stim_styles[scope],
                line_width=kwargs.get("linewidth", 5),
                source=self.cds[scope],
                legend_label="Accuracy(%)",
            )

            fig.multi_line(
                xs="_x_axis",
                ys="_x_axis_line",
                line_color="_trial_color",
                line_alpha=0.3,
                source=self.cds[scope],
                name="trials",
            )

        fig = self.pretty_axes(fig, fontsize)
        fig.xaxis.axis_label = "Trial Number"
        fig.yaxis.axis_label = "Accuracy(%)"

        fig.yaxis.bounds = (0, 100)
        fig.legend.location = "top_left"
        fig.legend.label_text_font_size = self.pt_font(fontsize - 5)

        # add hover tool
        fig.add_tools(
            HoverTool(
                names=["trials"],
                tooltips=[
                    ("Trial No/Time", "@x_axis"),
                    ("Side", "@_trial_side"),
                    ("Contrast", "@_trial_contrast"),
                    ("Answer", "@answer"),
                ],
                mode="mouse",
            )
        )
        self.fig = fig

    def set_cds(self, data, **kwargs):
        """ """
        if not isinstance(data, dict):
            raise TypeError(
                "Data should be in an dict. If it" "s a DataFrame put it in a dict."
            )

        for i, scope in enumerate(data.keys()):
            d = data[scope]

            if kwargs.get("drawtrials", True):
                d["trial_color"] = d["answer"].apply(self.line_color_from_answer)

                temp_trials = {
                    "_trial_color": d["trial_color"],
                    "_trial_side": d["stim_side"],
                    "_trial_contrast": d["contrast"],
                }
            else:
                temp_trials = {
                    "_trial_no": [[]],
                    "_trial_line": [[]],
                    "_trial_color": [],
                    "_trial_side": [],
                    "_trial_contrast": [],
                }

            if kwargs.get("time_axis", True):
                x_axis = {
                    "x_axis": d["openstart_absolute"] / 60000,
                    "_x_axis": [[t / 60000, t / 60000] for t in d["openstart_absolute"]],
                    "_x_axis_line": [[0, 100]] * len(d["openstart_absolute"]),
                }
            else:
                x_axis = {
                    "x_axis": d["trial_no"],
                    "_x_axis": [[t, t] for t in d["trial_no"]],
                    "_x_axis_line": [[0, 100]] * len(d["trial_no"]),
                }

            temp = {
                "answer": d["answer"],
                "performance_percent": d["fraction_correct"] * 100,
                **temp_trials,
                **x_axis,
            }

            if scope not in self.cds.keys():
                print("PERFORMANCE GRAPH - Setting Data", flush=True)
                self.cds[scope] = ColumnDataSource(data=temp)
            else:
                print("PERFORMANCE GRAPH - Updating data", flush=True)
                self.cds[scope].data = temp


class ResponseTime(Graph):
    def __init__(self, *args, **kwargs):
        self.fig = None
        self.cds = {}

    def plot(self, *args, **kwargs):
        fontsize = kwargs.get("fontsize", 14)
        # set up axes
        fig = bok.figure(
            y_axis_type="log",
            toolbar_location=kwargs.get("toolbarlocation", "above"),
            tools="pan,box_zoom,wheel_zoom,reset",
            plot_width=800,
            plot_height=400,
        )

        for scope in self.cds.keys():
            fig.line(
                x="x_axis",
                y="running_response_latency",
                line_color=stim_styles[scope],
                line_width=kwargs.get("linewidth", 5),
                source=self.cds[scope],
            )

            if kwargs.get("plottrials", True):
                fig.circle(
                    x="x_axis",
                    y="response_latency",
                    size=7,
                    fill_color=stim_styles[scope],
                    line_color="trial_color",
                    line_width=1,
                    line_alpha=0.5,
                    muted_alpha=0.2,
                    legend_label="Mean Response Time",
                    source=self.cds[scope],
                    name="points",
                )

        fig = self.pretty_axes(fig, fontsize)
        fig.xaxis.axis_label = "Trial No"
        fig.yaxis.axis_label = "Response Time (s)"

        fig.y_range.start = 0
        fig.y_range.end = 10

        fig.legend.location = "top_left"
        fig.legend.label_text_font_size = self.pt_font(fontsize - 5)
        fig.legend.click_policy = "mute"

        # add hover tool
        fig.add_tools(
            HoverTool(
                names=["points"],
                tooltips=[
                    ("Trial No/Time", "@x_axis"),
                    ("Response Time", "@response_latency"),
                ],
                mode="mouse",
            )
        )
        self.fig = fig

    def set_cds(self, data, **kwargs):
        """Creates a column data source(cds)"""
        if not isinstance(data, dict):
            raise TypeError(
                "Data should be in an dict. If it" "s a DataFrame put it in a dict."
            )

        for i, key in enumerate(data.keys()):
            d = data[key]

            d["trial_color"] = d["answer"].apply(self.dot_color_from_answer)

            if kwargs.get("time_axis", True):
                x_axis = {"x_axis": d["openstart_absolute"] / 60000}
            else:
                x_axis = {"x_axis": d["trial_no"]}

            temp = {
                "running_response_latency": d["running_response_latency"] / 1000,
                "response_latency": d["response_latency"] / 1000,
                "trial_color": d["trial_color"],
                **x_axis,
            }

            if key not in self.cds.keys():
                print("RESPONSETIME GRAPH - Setting Data", flush=True)
                self.cds[key] = ColumnDataSource(data=temp)
            else:
                print("RESPONSETIME GRAPH - Updating data", flush=True)
                self.cds[key].data = temp


class AnswerDistribution(Graph):
    def __init__(self, *args, **kwargs):
        self.cds = {}
        self.answers = {}
        self.colors = []
        self.x_axis_types = None

    def plot(self, *arg, **kwargs):
        fontsize = kwargs.get("fontsize", 14)

        # set up axes
        fig = bok.figure(
            x_range=FactorRange(*self.x_axis_types),
            toolbar_location=kwargs.get("toolbarlocation", "above"),
            tools="pan,box_zoom,reset",
            plot_width=400,
            plot_height=400,
        )

        for scope in self.cds.keys():
            fig.x_range = FactorRange(*self.cds[scope].data["x"])
            fig.vbar_stack(
                self.answers[scope],
                x="x",
                width=0.9,
                color=self.colors,
                source=self.cds[scope],
                legend_label=self.answers[scope],
            )

        fig = self.pretty_axes(fig, fontsize)
        fig.xaxis.major_label_orientation = 1

        # add hover tool
        fig.add_tools(HoverTool(tooltips=[("Count", "$y")], mode="mouse"))

        self.fig = fig

    def set_cds(self, data, **kwargs):
        """Creates a column data source(cds)"""
        if not isinstance(data, dict):
            raise TypeError(
                "Data should be in an dict. If it" "s a DataFrame put it in a dict."
            )

        for i, key in enumerate(data.keys()):

            d = data[key]
            self.answers[key] = [str(a) for a in np.unique(d["answer"])]

            # create types
            temp = {"x": []}
            for side in np.unique(d["stim_side"]):
                s_key = "Left" if side < 0 else "Right"
                d_side = d[d["stim_side"] == side]
                for opto in np.unique(d_side["opto"]):
                    o_key = "opto" if opto == 1 else "non-opto"
                    d_opto = d_side[d_side["opto"] == opto]
                    temp["x"].append((s_key, o_key))
                    for a in np.unique(d_opto["answer"]):
                        d_answer = d_opto[d_opto["answer"] == a]
                        if str(a) not in temp:
                            temp[str(a)] = [np.sum(len(d_answer))]
                            if a == 1:
                                if "#048c00" not in self.colors:
                                    self.colors.append("#048c00")
                            elif a == -1:
                                if "#aa1600" not in self.colors:
                                    self.colors.append("#aa1600")
                            else:
                                if "#282828" not in self.colors:
                                    self.colors.append("#282828")
                        else:
                            temp[str(a)].append(np.sum(len(d_answer)))

            self.x_axis_types = temp["x"]

            if key not in self.cds.keys():
                print("ANSWER DISTRIBUTION GRAPH - Setting Data", flush=True)
                self.cds[key] = ColumnDataSource(data=temp)
            else:
                print("ANSWER DISTRIBUTION GRAPH - Updating data", flush=True)
                self.cds[key].data = temp


class ContrastDistribution(Graph):
    def __init__(self, *args, **kwargs):
        self.cds = {}
        self.contrasts = {}
        self.colors = []
        self.x_axis_types = None

    def plot(self, *args, **kwargs):
        fontsize = kwargs.get("fontsize", 14)

        # set up axes
        fig = bok.figure(
            x_range=FactorRange(*self.x_axis_types),
            toolbar_location=kwargs.get("toolbarlocation", "above"),
            tools="pan,box_zoom,reset",
            plot_width=400,
            plot_height=400,
        )

        for scope in self.cds.keys():
            fig.vbar_stack(
                self.contrasts[scope],
                x="x",
                width=0.9,
                color=self.colors,
                source=self.cds[scope],
                legend_label=self.contrasts[scope],
            )

        fig = self.pretty_axes(fig, fontsize)
        fig.xaxis.major_label_orientation = 1

        fig.legend.location = "top_left"
        fig.legend.label_text_font_size = self.pt_font(fontsize - 5)

        # add hover tool
        fig.add_tools(HoverTool(tooltips=[("Count", "$y")], mode="mouse"))

        self.fig = fig

    def set_cds(self, data, **kwargs):
        """Creates a column data source(cds)"""
        if not isinstance(data, dict):
            raise TypeError(
                "Data should be in an dict. If it" "s a DataFrame put it in a dict."
            )

        for i, key in enumerate(data.keys()):

            d = data[key]
            self.contrasts[key] = [str(c) for c in np.unique(d["contrast"])]

            if len(self.colors) == 0:
                self.colors = [contrast_styles[c] for c in np.unique(d["contrast"])]

            # create types
            temp = {"x": []}
            for side in np.unique(d["stim_side"]):
                s_key = "Left" if side < 0 else "Right"
                d_side = d[d["stim_side"] == side]
                for opto in np.unique(d_side["opto"]):
                    o_key = "opto" if opto == 1 else "non-opto"
                    d_opto = d_side[d_side["opto"] == opto]
                    temp["x"].append((s_key, o_key))
                    for c in np.unique(d_opto["contrast"]):
                        d_contrast = d_opto[d_opto["contrast"] == c]
                        if str(c) not in temp:
                            temp[str(c)] = [np.sum(len(d_contrast))]
                        else:
                            temp[str(c)].append(np.sum(len(d_contrast)))

            # TODO: This might cause issues
            self.x_axis_types = temp["x"]

            if key not in self.cds.keys():
                print("CONTRAST DISTRIBUTION GRAPH - Setting Data", flush=True)
                self.cds[key] = ColumnDataSource(data=temp)
            else:
                print("CONTRAST DISTRIBUTION GRAPH - Updating data", flush=True)
                self.cds[key].data = temp


class TrialPicture(Graph):
    def __init__(self, *args, **kwargs):
        self.fig = None
        self.trace_color = "#000000"
        self.cds_wheel = None
        self.cds_areas = None
        self.cds_lines = None
        self.cds_licks = None

    def plot(self, **kwargs):
        fontsize = kwargs.get("fontsize", 14)
        # set up axes
        fig = bok.figure(
            toolbar_location=kwargs.get("toolbarlocation", "above"),
            tools="pan,box_zoom,wheel_zoom,reset",
            plot_width=800,
            plot_height=500,
        )

        # static lines and zones
        # stim on screen
        fig.harea(
            x1="stimstart",
            x2="stimend",
            y="dummy_vert_line",
            fill_color="#bcbcbc",
            fill_alpha=0.5,
            legend_label="Stim on screen",
            source=self.cds_areas,
        )

        # decision
        fig.harea(
            x1="closedloopstart",
            x2="closedloopend",
            y="dummy_vert_line",
            fill_color="#727272",
            fill_alpha=0.5,
            legend_label="Decision Window",
            source=self.cds_areas,
        )

        # stim_side
        fig.segment(
            x0="horiz_line_start",
            y0="correct_stimside",
            x1="horiz_line_end",
            y1="correct_stimside",
            line_color="#048e26",
            line_alpha=0.8,
            line_dash="dashed",
            line_width=2,
            legend_label="Correct",
            source=self.cds_lines,
        )

        fig.segment(
            x0="horiz_line_start",
            y0="incorrect_stimside",
            x1="horiz_line_end",
            y1="incorrect_stimside",
            line_color="#c91823",
            line_alpha=0.8,
            line_dash="dashed",
            line_width=2,
            legend_label="Incorrect",
            source=self.cds_lines,
        )

        # reward
        fig.segment(
            x0="reward",
            y0="vert_line_start",
            x1="reward",
            y1="vert_line_end",
            line_color="#f79413",
            line_alpha=0.8,
            line_width=2,
            legend_label="Reward",
            source=self.cds_lines,
        )

        # openloopstart
        fig.segment(
            x0="openloopstart",
            y0="vert_line_start",
            x1="openloopstart",
            y1="vert_line_end",
            line_color="#000000",
            line_alpha=0.8,
            line_width=2,
            legend_label="Stim start",
            source=self.cds_lines,
        )

        # is_opto line
        fig.segment(
            x0="horiz_line_start",
            y0=30,
            x1="horiz_line_end",
            y1=30,
            line_color="#0a4bbb",
            line_width=5,
            line_alpha="is_opto",
            source=self.cds_lines,
        )

        # dynamic stuff (wheel and lick)
        fig.line(
            x="wheel_t",
            y="wheel_pos",
            line_color=self.trace_color,
            source=self.cds_wheel,
            line_alpha=1,
            line_width=4,
            legend_label="Wheel",
            name="wheel",
        )

        fig.segment(
            x0="lick_t",
            y0=-10,
            x1="lick_t",
            y1=10,
            line_color="#13f7f7",
            line_width=5,
            legend_label="Licks",
            source=self.cds_licks,
        )

        fig = self.pretty_axes(fig, fontsize)
        fig.xaxis.axis_label = "Time"
        fig.yaxis.axis_label = "Wheel Position(deg)"

        fig.legend.location = "bottom_right"
        fig.legend.label_text_font_size = self.pt_font(fontsize - 5)

        # add hover tool
        fig.add_tools(
            HoverTool(
                names=["wheel"],
                tooltips=[("Time", "@wheel_t"), ("Position", "@wheel_pos")],
                mode="mouse",
            )
        )
        self.fig = fig

    def set_cds(self, data, **kwargs):
        """Creates a column data source(cds)"""

        self.trace_color = contrast_styles[data["contrast"]]

        # has the wheel trace data
        temp_wheel = {"wheel_t": data["wheel"][:, 0], "wheel_pos": data["wheel"][:, 1]}

        # has the lick data
        temp_licks = {"lick_t": data["lick"][:, 0], "lick_cnt": data["lick"][:, 1]}

        # has the len=2 data for drawing the zones
        temp_areas = {
            "stimstart": [data["stimdur"][0]] * 2,
            "stimend": [data["stimdur"][1]] * 2,
            "closedloopstart": [data["closedloopdur"][0]] * 2,
            "closedloopend": [data["closedloopdur"][1]] * 2,
            "dummy_vert_line": [
                -np.abs(data["stim_side"]) * 2,
                np.abs(data["stim_side"]) * 2,
            ],
        }

        # has the len=1 data for drawing lines
        if len(data["reward"]):
            reward = [data["reward"][0][0]]
            reward_alpha = [0.8]
        else:
            reward = [0]
            reward_alpha = [0]
        temp_lines = {
            "correct_stimside": [-1 * data["stim_side"]],
            "incorrect_stimside": [data["stim_side"]],
            "openloopstart": [data["openloopstart"]],
            "reward": reward,
            "reward_alpha": reward_alpha,  # this is a fix for reward being an empty list in incorrect trials
            "is_correction": [data["correction"]],
            "is_opto": [data["opto"]],
            "opto_pattern": [data.get("opto_pattern", -1)],
            "horiz_line_start": [data["stimdur"][0]],
            "horiz_line_end": [data["stimdur"][1]],
            "vert_line_start": [-np.abs(data["stim_side"]) * 2],
            "vert_line_end": [np.abs(data["stim_side"]) * 2],
        }

        if self.cds_wheel is None:
            print("TRIAL PICTURE GRAPH - Setting Data", flush=True)
            self.cds_wheel = ColumnDataSource(data=temp_wheel)
            self.cds_areas = ColumnDataSource(data=temp_areas)
            self.cds_lines = ColumnDataSource(data=temp_lines)
            self.cds_licks = ColumnDataSource(data=temp_licks)
        else:
            print("RESPONSETIME GRAPH - Updating data", flush=True)
            self.cds_wheel.data = temp_wheel
            self.cds_areas.data = temp_areas
            self.cds_lines.data = temp_lines
            self.cds_licks.data = temp_licks


# TODO: Add these plots
class ResponseBox(Graph):
    def __init__(self, *args, **kwargs):
        self.fig = None
        self.cds = {}

    def plot(self, *args, **kwargs):
        pass

    def set_cds(self, data):
        pass


class WheelTrace(Graph):
    def __init__(self, *args, **kwargs):
        self.fig = None
        self.cds_lines = {}
        self.cds_avg = {}
        self.cds_indiv = {}

    def plot(self, *args, **kwargs):
        fontsize = kwargs.get("fontsize", 14)

        fig = bok.figure(
            toolbar_location=kwargs.get("toolbarlocation", "above"),
            tools="pan,box_zoom,wheel_zoom,reset",
            plot_width=500,
            plot_height=800,
        )

        for scope in self.cds_avg.keys():

            fig.line(
                x="avg_pos",
                y="avg_time",
                line_color=stim_styles[scope],
                line_width=kwargs.get("linewidth", 5),
                source=self.cds_avg[scope],
                legend_label="hh",
                name="avgs",
            )

            # static lines
            # stim_side
            fig.segment(
                x0="horiz_line_start",
                y0="correct_stimside",
                x1="horiz_line_end",
                y1="correct_stimside",
                line_color="#048e26",
                line_alpha=0.8,
                line_dash="dashed",
                line_width=2,
                legend_label="Correct",
                source=self.cds_lines[scope],
            )

            fig.segment(
                x0="horiz_line_start",
                y0="incorrect_stimside",
                x1="horiz_line_end",
                y1="incorrect_stimside",
                line_color="#c91823",
                line_alpha=0.8,
                line_dash="dashed",
                line_width=2,
                legend_label="Incorrect",
                source=self.cds_lines[scope],
            )

            # openloopstart
            fig.segment(
                x0="openloopstart",
                y0="vert_line_start",
                x1="openloopstart",
                y1="vert_line_end",
                line_color="#000000",
                line_alpha=0.8,
                line_width=2,
                legend_label="Stim start",
                source=self.cds_lines[scope],
            )

            # individual traces
            fig.multi_line(
                xs="indiv_pos",
                ys="indiv_time",
                line_color="#127a45",
                line_alpha=0.3,
                source=self.cds_indiv[scope],
                name="trials",
            )

        fig = self.pretty_axes(fig, fontsize)
        fig.xaxis.axis_label = "Wheel Position(deg)"
        fig.yaxis.axis_label = "Time(s)"

        fig.legend.location = "top_left"
        fig.legend.label_text_font_size = self.pt_font(fontsize - 5)

        # add hover tool
        fig.add_tools(
            HoverTool(
                names=["trials"],
                tooltips=[
                    ("Trial No/Time", "@x_axis"),
                    ("Side", "@_trial_side"),
                    ("Contrast", "@_trial_contrast"),
                    ("Answer", "@answer"),
                ],
                mode="mouse",
            )
        )
        self.fig = fig

    def set_cds(self, data, **kwargs):
        """Creates a column data source(cds)"""
        if not isinstance(data, dict):
            raise TypeError(
                "Data should be in an dict. If it" "s a DataFrame put it in a dict."
            )

        # iterate over all stim types
        for i, scope in enumerate(data.keys()):
            d = data[scope]
            sides = np.unique(d["stim_side"])

            temp_lines = {
                "correct_stimside": [0],
                "incorrect_stimside": [2 * d["stim_side"]],
                "openloopstart": [d["openloopstart"]],
                "horiz_line_start": [d["stimdur"][0]],
                "horiz_line_end": [d["stimdur"][1]],
                "vert_line_start": [-np.abs(d["stim_side"]) * 2],
                "vert_line_end": [np.abs(d["stim_side"]) * 2],
            }

            # iterate over contrast and answer seperation
            for seperator in ["contrast", "answer"]:
                sep_list = np.unique(d[seperator])

                # iterate over stim sides
                for i, s in enumerate(sides, 1):
                    side_slice = d[d["stim_side"] == s]

                    # iterate over different seperator values (e.g. [1,0.5,0.25] for contrast)
                    for sep in sep_list:
                        sep_slice = side_slice[side_slice[seperator] == sep]

                        # shift wheel according to side
                        sep_slice["wheel"] = sep_slice["wheel"].apply(lambda x: x + s)

                        wheel_stat_dict = get_trajectory_stats(sep_slice)
                        avg = wheel_stat_dict["average"]
                        s_limit = kwargs.get("wheel_limit", None)

                        if avg is not None:

                            if s_limit is not None:
                                avg = avg[
                                    find_nearest(avg[:, 0], -200)[0] : find_nearest(
                                        avg[:, 0], s_limit
                                    )[0]
                                ]

                            temp_avg = {"avg_pos": avg[:, 1], "avg_time": avg[:, 0]}

                        temp_indiv = {"indiv_pos": [], "indiv_time": []}
                        if kwargs.get("show_individual", False):
                            for trial in sep_slice.itertuples():
                                indiv = trial.wheel
                                if len(indiv):
                                    if s_limit is not None:
                                        indiv = indiv[
                                            find_nearest(indiv[:, 0], -200)[
                                                0
                                            ] : find_nearest(indiv[:, 0], s_limit)[0],
                                            :,
                                        ]

                                    temp_indiv["indiv_pos"].append(indiv[:, 1])
                                    temp_indiv["indiv_time"].append(indiv[:, 0])

            if scope not in self.cds_avg.keys():
                print("WHEEL TRAJ GRAPH - Setting Data", flush=True)
                self.cds_avg[scope] = ColumnDataSource(data=temp_avg)
                self.cds_lines[scope] = ColumnDataSource(data=temp_lines)
                self.cds_indiv[scope] = ColumnDataSource(data=temp_indiv)
            else:
                print("WHEEL TRAJ GRAPH - Updating data", flush=True)
                self.cds_avg[scope].data = temp_avg
                self.cds_lines[scope].data = temp_lines
                self.cds_indiv[scope].data = temp_indiv
