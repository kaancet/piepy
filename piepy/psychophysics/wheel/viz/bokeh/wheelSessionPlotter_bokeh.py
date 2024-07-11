import scipy.stats
from collections import defaultdict
from .wheelBasePlotter_bokeh import *


class WheelSessionPlotter(WheelBasePlotter):
    """A class to create wheel session related plots
    :param session      : A dictionary that contains all of the dta
    :param session['meta'] : Session meta data
    :param savepath     : session directory path(usually in the analysis folder)
    :type session       : dict
    :type session['meta']  : dict
    :type savepath      : str
    """

    def __init__(self, session, savepath, *args, **kwargs):
        self.session = session
        self.fig = None
        self.savepath = savepath

    def prep_text(self):
        """Prepares summary text from session dictionaries"""
        text_meta = """META \n"""
        for k in [
            "animalid",
            "date",
            "rig",
            "wheelgain",
            "water_on_rig",
            "rewardsize",
            "level",
        ]:
            if k in self.session["meta"].keys():
                text_meta += "{0}  :  {1}\n".format(k, self.session["meta"][k])

        text_summ = """SUMMARY \n"""
        for key, value in self.session["summaries"]["overall"].items():
            if key != "latency":
                text_summ += "{0}  :  {1}\n".format(key, value)

        return text_meta + """\n""" + text_summ

    def save(self, plotkey):
        """Saves the figure
        :param plotkey : Name to be used when saving the figure
        :type plotkey  : str
        """
        figsave_loc = pjoin(self.savepath, "sessionFigures")
        if not os.path.exists(figsave_loc):
            os.mkdir(figsave_loc)

        savename = "{0}_{1}_{2}.html".format(
            self.session["meta"]["baredate"], self.session["meta"]["animalid"], plotkey
        )
        savepath = pjoin(figsave_loc, savename)
        bok.output_file(savepath)
        bok.save(self.fig)
        display("{0} plot saved in {1}".format(plotkey, self.savepath))

    def set_scope(self, scope="all"):
        # set the scope of data
        if scope == "novel":
            data = self.session["novel_stim_data"]["overall"]
        elif scope == "all":
            data = self.session["data"]
        elif scope == "psychometric":
            data = self.session["novel_stim_data"]
        else:
            data = self.session["novel_stim_data"].get(scope, None)
            if data is None:
                raise ValueError(
                    "Data scope {0} does not exist, try one of {1}".format(
                        list(scope, self.session["novel_stim_data"].keys())
                    )
                )
        return data

    def filter_data(self, data, filters):
        if filters is None:
            filters = {}
            filtered_data = data
        else:
            if "trial_limit" in filters.keys():
                if isinstance(data, dict):
                    filtered_data = {
                        k: v[v["trial_no"] <= filters["trial_limit"]]
                        for k, v in data.items()
                    }
                else:
                    filtered_data = data[data["trial_no"] <= filters["trial_limit"]]
            elif "response_cutoff" in filters.keys():
                if isinstance(data, dict):
                    filtered_data = {
                        data[k]: v[v["response_latency"] <= filters["response_cutoff"]]
                        for k, v in data.items()
                    }
                else:
                    filtered_data = data[
                        data["response_latency"] <= filters["response_cutoff"]
                    ]
        return filtered_data

    def plot(self, plt_func, func_params=None, *args, **kwargs):
        """Main plot function that calls specific plotting functions
        :param plt_func  : function nam to be plotted
        :param close_fig : Flag to control closing figures
        :type plt_func   : str
        :type close_fig  : boolean
        """
        if func_params is None:
            func_params = []

        scope = kwargs.get("scope", "all")
        filters = kwargs.get("filters", {})

        callables = []
        for name in dir(self):
            if not is_special(name):
                value = getattr(self, name)
                if callable(value):
                    callables.append(name)

        if plt_func not in callables:
            display(
                "{0} not a plotter function for WheelSessionPlotter try {1}".format(
                    plt_func, ", ".join(callables)
                )
            )
            raise ValueError()

        getattr(self, plt_func)(*func_params, *args, **kwargs)

        # save

        if kwargs.get("savefig", True):
            self.save(
                "{0}_{1}_{2}".format(
                    plt_func, scope, "_".join([str(v) for v in filters.values()])
                )
            )

        return self.fig

    def psychometric(self, mode="src", fig=None, *args, **kwargs):
        """Plots the psychometric curve for each stimuli type
        :param fig  : a premade fig object to plot on
        :type fig   : bokeh.model.Figure
        :return fig : plotted axes
        :rtype      : bokeh.model.Figure
        """
        fontsize = kwargs.get("fontsize", 22)
        side = kwargs.get("side", "right")
        filters = kwargs.get("filters", None)

        scope_data = self.set_scope("psychometric")
        data = self.filter_data(scope_data, filters)

        # set up axes
        if fig is None:
            fig = bok.figure(
                toolbar_location=kwargs.get("toolbarlocation", "right"),
                tools="pan,box_zoom,reset",
            )

        # plotting
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

        for key, d in data.items():
            if key == "overall":
                continue

            curve = WheelCurve(name=key, rawdata=d)
            source = ColumnDataSource(
                data={
                    "contrast_x": 100 * curve.contrast_x,
                    "prob_right": curve.proportion_right,
                    "prob_left": curve.proportion_left,
                    "err_right": curve.binomial_right,
                    "err_left": curve.binomial_left,
                    "error_up_right": [
                        x + e
                        for x, e in zip(curve.proportion_right, curve.binomial_right)
                    ],
                    "error_down_right": [
                        x - e
                        for x, e in zip(curve.proportion_right, curve.binomial_right)
                    ],
                    "error_up_left": [
                        x + e for x, e in zip(curve.proportion_left, curve.binomial_left)
                    ],
                    "error_down_left": [
                        x - e for x, e in zip(curve.proportion_left, curve.binomial_left)
                    ],
                    "trial_count": curve.counts,
                    "correct_pcts": 100 * curve.correct_pcts,
                }
            )

            fig.circle(
                x="contrast_x",
                y="prob_{0}".format(side),
                size=kwargs.get("markersize", 15),
                fill_color=area_colors[key],
                line_color=kwargs.get("markeredgecolor", "#ffffff"),
                line_width=kwargs.get("markeredgewidth", 2),
                source=source,
                legend_label="{0}".format(key),
                name="points",
            )

            fig.add_layout(
                Whisker(
                    source=source,
                    base="contrast_x",
                    upper="error_up_{0}".format(side),
                    lower="error_down_{0}".format(side),
                    line_color=area_colors[key],
                    line_width=kwargs.get("elinewidth", 3),
                    upper_head=TeeHead(line_width=0),
                    lower_head=TeeHead(line_width=0),
                )
            )

            if side == "right":
                fit_y = curve.proportion_right_fitted[:, 0]
            else:
                fit_y = curve.proportion_left_fitted[:, 0]

            fig.line(
                x=100 * curve.contrast_x_fitted[:, 0],
                y=fit_y,
                line_color=area_colors[key],
                level="underlay",
                line_width=kwargs.get("linewidth", 9),
                line_cap="round",
            )

        # add hover tool
        fig.add_tools(
            HoverTool(
                names=["points"],
                tooltips=[
                    ("Trial Count", "@trial_count"),
                    ("Correct(%)", "@correct_pcts"),
                    ("%95", "@err_{0}".format(side)),
                ],
                mode="mouse",
            )
        )

        # make it pretty
        fig = self.pretty_axes(fig)
        fig.xaxis.axis_label = "Contrast Value"
        fig.yaxis.axis_label = "Prob. Chosing {0}".format(side.capitalize())

        fig.xaxis.bounds = (-100, 100)
        fig.yaxis.bounds = (0, 1)
        fig.legend.location = "top_left" if side == "right" else "top_right"
        fig.legend.label_text_font_size = self.pt_font(fontsize - 5)
        fig.xgrid.grid_line_color = None
        fig.ygrid.grid_line_color = None

        # some logic for embedding in summary figures
        if not kwargs.get("notitle", False):
            fig.title.text = "{0} Psychometric Curve {1}".format(
                self.session["meta"]["animalid"], self.session["meta"]["date"]
            )
            fig.title.text_font_size = self.pt_font(fontsize + 2)
            fig.title.text_font_style = "bold"

        if len(data.keys()) <= 2:
            fig.legend.visible = False

        self.fig = fig
        # save
        if kwargs.get("savefig", True):
            self.save("psychometric_{0}".format(side))

        return self.fig

    def performance(self, fig=None, *args, **kwargs):
        """Plots the psychometric curve for each stimuli type
        :param fig  : a premade fig object to plot on
        :type fig   : bokeh.model.Figure
        :return fig : plotted axes
        :rtype      : bokeh.model.Figure
        """
        fontsize = kwargs.get("fontsize", 22)
        scope = kwargs.get("scope", "all")

        # set up axes
        if fig is None:
            fig = bok.figure(
                toolbar_location=kwargs.get("toolbarlocation", "above"),
                tools="pan,box_zoom,reset",
            )

        # plotting
        source = ColumnDataSource(
            data={
                "time_in_secs": data["openstart_absolute"] / 60000,
                "performance_percent": data["fraction_correct"] * 100,
            }
        )

        fig.line(
            x="time_in_secs",
            y="performance_percent",
            line_color="#148428",
            line_width=kwargs.get("linewidth", 5),
            source=source,
            legend_label="Accuracy(%)",
        )

        # make it pretty
        fig = self.pretty_axes(fig)
        fig.xaxis.axis_label = "Time (min)"
        fig.yaxis.axis_label = "Accuracy(%)"

        fig.yaxis.bounds = (0, 100)
        fig.legend.location = "top_left"
        fig.legend.label_text_font_size = self.pt_font(fontsize - 5)

        # some logic for embedding in summary figures
        if not kwargs.get("notitle", False):
            fig.title.text = "{0} Performance {1}".format(
                self.session["meta"]["animalid"], self.session["meta"]["date"]
            )
            fig.title.text_font_size = self.pt_font(fontsize + 2)
            fig.title.text_font_style = "bold"

        if not kwargs.get("showlegend", False):
            fig.legend.visible = False

        self.fig = fig
        # save
        if kwargs.get("savefig", True):
            self.save("performance_{0}".format(scope))

        return self.fig

    def responseLatency(self, sep_sides=False, fig=None, *args, **kwargs):
        """Plots the psychometric curve for each stimuli type
        :param sep_sides : boolean whether to seperate left/right responses when plotting
        :param fig       : a premade fig object to plot on
        :type sep_sides  : boolean
        :type fig        : bokeh.model.Figure
        :return fig      : plotted axes
        :rtype           : bokeh.model.Figure
        """
        fontsize = kwargs.get("fontsize", 22)
        scope = kwargs.get("scope", "all")

        # set up axes
        if fig is None:
            fig = bok.figure(
                toolbar_location=kwargs.get("toolbarlocation", "above"),
                tools="pan,box_zoom,reset",
                y_axis_type="log",
            )

        # set the scope of data
        if scope == "novel":
            temp_data = self.session["novel_stim_data"]["overall"]
        elif scope == "all":
            temp_data = self.session["data"]
        else:
            temp_data = self.session["novel_stim_data"].get("", None)
            if temp_data is None:
                raise ValueError(
                    "Data scope {0} does not exist, try one of {1}".format(
                        list(scope, self.session["novel_stim_data"].keys())
                    )
                )

        data = {}
        colors = {}
        if sep_sides:
            data["Left"] = temp_data[temp_data["stim_side"] < 0]
            data["Right"] = temp_data[temp_data["stim_side"] > 0]
            colors["Left"] = "#BA2740"
            colors["Right"] = "#34B8BA"
        else:
            data["Both"] = temp_data
            colors["Both"] = "#4576AE"

        # plotting
        for key in data.keys():

            source_line = ColumnDataSource(
                data={
                    "running_response_latency": data[key]["running_response_latency"]
                    / 1000,
                    "trial_no": data[key]["trial_no"],
                }
            )

            fig.line(
                x="trial_no",
                y="running_response_latency",
                line_color=colors[key],
                line_width=kwargs.get("linewidth", 5),
                legend_label="{0} Response Lat.".format(key),
                source=source_line,
            )

            if kwargs.get("plottrials", True):
                source_scatter = ColumnDataSource(
                    data={
                        "trial_no": data[key]["trial_no"],
                        "response_latency": data[key]["response_latency"] / 1000,
                    }
                )
                fig.circle(
                    x="trial_no",
                    y="response_latency",
                    size=kwargs.get("markersize", 5),
                    fill_color=colors[key],
                    line_width=0,
                    fill_alpha=0.3,
                    source=source_scatter,
                    name="trials",
                )

                fig.add_tools(
                    HoverTool(
                        names=["trials"],
                        tooltips=[
                            ("Trial No", "@trial_no"),
                            ("Response Time", "@response_latency"),
                        ],
                        mode="mouse",
                    )
                )

        # make it pretty
        fig = self.pretty_axes(fig)
        fig.xaxis.axis_label = "Trial No"
        fig.yaxis.axis_label = "Response Times(s)"

        fig.legend.location = "top_left"
        fig.legend.label_text_font_size = self.pt_font(fontsize - 5)

        # some logic for embedding in summary figures
        if not kwargs.get("notitle", False):
            fig.title.text = "{0} Response Time {1}".format(
                self.session["meta"]["animalid"], self.session["meta"]["date"]
            )
            fig.title.text_font_size = self.pt_font(fontsize + 2)
            fig.title.text_font_style = "bold"

        if not kwargs.get("showlegend", False):
            fig.legend.visible = False

        self.fig = fig
        # save
        if kwargs.get("savefig", True):
            self.save("responselatency_{0}".format(scope))

        return self.fig

    # This is not so good and/or useful
    def probability(self, fig=None, *args, **kwargs):
        """Plots the psychometric curve for each stimuli type
        :param fig  : a premade fig object to plot on
        :type fig   : bokeh.model.Figure
        :return fig : plotted axes
        :rtype      : bokeh.model.Figure
        """
        fontsize = kwargs.get("fontsize", 22)
        scope = kwargs.get("scope", "all")

        # set up the axes
        if fig is None:
            fig = bok.figure(
                toolbar_location=kwargs.get("toolbarlocation", "above"),
                tools="pan,box_zoom,reset",
            )

        # set the scope of data
        if scope == "novel":
            data = self.session["novel_stim_data"]["overall"]
        elif scope == "all":
            data = self.session["data"]
        else:
            data = self.session["novel_stim_data"].get("", None)
            if data is None:
                raise ValueError(
                    "Data scope {0} does not exist, try one of {1}".format(
                        list(scope, self.session["novel_stim_data"].keys())
                    )
                )

        # plotting
        probs_left = data[data["stim_side"] < 0]["running_prob"]
        probs_right = data[data["stim_side"] > 0]["running_prob"]

        time_left = data[data["stim_side"] < 0]["openstart_absolute"] / 60000
        time_right = data[data["stim_side"] > 0]["openstart_absolute"] / 60000

        fig.line(
            time_left,
            probs_left,
            line_color="#BA2740",
            line_width=kwargs.get("linewidth", 5),
            legend_label="Left",
        )

        fig.line(
            time_right,
            probs_right,
            line_color="#34B8BA",
            line_width=kwargs.get("linewidth", 5),
            legend_label="Right",
        )

        # make it pretty
        fig = self.pretty_axes(fig)
        fig.xaxis.axis_label = "Time(mins)"
        fig.yaxis.axis_label = "Probability"

        fig.yaxis.bounds = (0, 1)
        fig.legend.location = "top_left"
        fig.legend.label_text_font_size = self.pt_font(fontsize - 5)

        # some logic for embedding in summary figures
        if not kwargs.get("notitle", False):
            fig.title.text = "{0} Stim Side Probabilities {1}".format(
                self.session["meta"]["animalid"], self.session["meta"]["date"]
            )
            fig.title.text_font_size = self.pt_font(fontsize + 2)
            fig.title.text_font_style = "bold"

        if not kwargs.get("showlegend", False):
            fig.legend.visible = False

        self.fig = fig
        # save
        if kwargs.get("savefig", True):
            self.save("stim_probabilities_{0}".format(scope))
        return self.fig

    def parameterCompare(self, p1, p2, fig=None, *args, **kwargs):
        """Compares two parameters in a scatter plot with a linear fit
        :param p1  : name of x-axis parameter
        :param p2  : name of y-axis parameter
        :param ax  : a premade axes object to plot on
        :type p1   : str
        :type p2   : str
        :type ax   : matplotlib.axes
        :return ax : plotted axes
        :rtype     : matplotlib.axes
        """
        fontsize = kwargs.get("fontsize", 22)
        scope = kwargs.get("scope", "all")

        # set up the axes
        # set up axes
        if fig is None:
            fig = bok.figure(
                toolbar_location=kwargs.get("toolbarlocation", "above"),
                tools="pan,box_zoom,reset",
            )

        # set the scope of data
        if scope == "novel":
            data = self.session["novel_stim_data"]["overall"]
        elif scope == "all":
            data = self.session["data"]
        else:
            data = self.session["novel_stim_data"].get("", None)
            if data is None:
                raise ValueError(
                    "Data scope {0} does not exist, try one of {1}".format(
                        list(scope, self.session["novel_stim_data"].keys())
                    )
                )

        if p1 not in data.columns:
            raise ValueError("No parameter named {0} in data".format(p1))
        if p2 not in data.columns:
            raise ValueError("No parameter named {0} in data".format(p2))

        # fit
        m, b, r, p, stderr = scipy.stats.linregress(data[p1], data[p2])
        fit_line = b + m * data[p1]

        source_fit = ColumnDataSource(data={"fit_x": data[p1], "fit_y": fit_line})

        source_scatter = ColumnDataSource(data={"x": data[p1], "y": data[p2]})

        # plotting

        fig.scatter(
            x="x",
            y="y",
            size=kwargs.get("markersize", 8),
            fill_color="#ABBDB1",
            fill_alpha=0.4,
            hover_fill_alpha=1,
            line_width=0,
            source=source_scatter,
            name="points",
        )

        fig.line(
            x="fit_x",
            y="fit_y",
            line_width=kwargs.get("linewidth", 2),
            line_color="#010101",
            legend_label="r={0:.2f}".format(r),
            source=source_fit,
        )

        fig.add_tools(
            HoverTool(names=["points"], tooltips=[(p1, "@x"), (p2, "@y")], mode="mouse")
        )

        # make it pretty
        fig = self.pretty_axes(fig)
        fig.xaxis.axis_label = p1
        fig.yaxis.axis_label = p2

        fig.legend.location = "top_left"
        fig.legend.label_text_font_size = self.pt_font(fontsize - 5)

        # some logic for embedding in summary figures
        if not kwargs.get("notitle", False):
            fig.title.text = "{0} {1}".format(
                self.session["meta"]["animalid"], self.session["meta"]["date"]
            )
            fig.title.text_font_size = self.pt_font(fontsize + 2)
            fig.title.text_font_style = "bold"

        if not kwargs.get("showlegend", False):
            fig.legend.visible = False

        self.fig = fig
        # save
        if kwargs.get("savefig", True):
            self.save("{0}_vs_{1}_{2}".format(p1, p2, scope))

        return self.fig

    def parameterFractions(self, param_list, fig=None, *args, **kwargs):
        """Compares two parameters in a scatter plot with a linear fit
        :param param_list : list of fraction_parameters to plot
        :param colors     : list of colors for lines
        :param ax         : a premade axes object to plot on
        :type param_list  : list
        :type colors      : list
        :type ax          : matplotlib.axes
        :return ax        : plotted axes
        :rtype            : matplotlib.axes
        """
        fontsize = kwargs.get("fontsize", 22)
        scope = kwargs.get("scope", "all")

        # set up the axes
        if fig is None:
            fig = bok.figure(
                toolbar_location=kwargs.get("toolbarlocation", "above"),
                tools="pan,box_zoom,reset",
            )

        # set the scope of data
        if scope == "novel":
            data = self.session["novel_stim_data"]["overall"]
        elif scope == "all":
            data = self.session["data"]
        else:
            data = self.session["novel_stim_data"].get("", None)
            if data is None:
                raise ValueError(
                    "Data scope {0} does not exist, try one of {1}".format(
                        list(scope, self.session["novel_stim_data"].keys())
                    )
                )

        # plotting
        trials_dict = defaultdict(list)
        if kwargs.get("drawtrials", True):
            for row in data.itertuples():
                if row.answer != 1:
                    trials_dict["trial_no"].append([row.trial_no, row.trial_no])
                    trials_dict["trial_line"].append([0, 1])

                    if row.answer == -1:
                        color = "#DB291D"
                    elif row.answer == 0:
                        color = "#66635E"

                    trials_dict["color"].append(color)

            lines_source = ColumnDataSource(data=trials_dict)

            fig.multi_line(
                xs="trial_no",
                ys="trial_line",
                line_width=1.3,
                line_color="color",
                line_alpha=0.3,
                source=lines_source,
            )

        param_dict = defaultdict(list)
        for i, param in enumerate(param_list):
            if "fraction" not in param:
                display(
                    "PARAMETER {0} IS NOT A FRACTION, Y-AXIS IS INCOMPATIBLE".format(
                        param
                    )
                )

            param_dict["trial_no"].append(data["trial_no"].tolist())
            param_dict["params"].append(data[param].tolist())
            param_dict["color"].append(fraction_palette[i])
            param_dict["labels"].append(param)

        source = ColumnDataSource(data=param_dict)

        fig.multi_line(
            xs="trial_no",
            ys="params",
            line_color="color",
            line_width=kwargs.get("linewidth", 4),
            line_alpha=0.7,
            hover_line_alpha=1,
            legend_label="labels",
            source=source,
            name="fractions",
        )

        # hover_temp = [(param,'@{0}'.format(param)) for param in param_list]
        # hover_temp.append(('trial_no','@trial_no'))
        # fig.add_tools(HoverTool(names=['fractions'],
        #                         tooltips=hover_temp,
        #                         mode='mouse'))

        # make it pretty
        fig = self.pretty_axes(fig)
        fig.xaxis.axis_label = "Trial No"
        fig.yaxis.axis_label = "Fraction"
        fig.yaxis.bounds = (0, 1)

        # some logic for embedding in summary figures
        if not kwargs.get("notitle", False):
            fig.title.text = "{0} {1}".format(
                self.session["meta"]["animalid"], self.session["meta"]["date"]
            )
            fig.title.text_font_size = self.pt_font(fontsize + 2)
            fig.title.text_font_style = "bold"

        if not kwargs.get("showlegend", False):
            fig.legend.visible = False

        self.fig = fig
        # save
        if kwargs.get("savefig", True):
            name = "_".join(param_list)
            self.save("{1}_{0}".format(name, scope))

        return self.fig

    def wheelTrajectory_vert(self, seperate_by="contrast", ax=None, *args, **kwargs):
        """Plots the wheel trajectory in vertical mode and seperating
        :param seperate_by : Propert to seperate the wheel traces by (corresponds to a column name in the session data)
        :param ax          : a premade axes object to plot on
        :type seperate_by  : str
        :type ax           : matplotlib.axes
        :return ax         : plotted axes
        :rtype             : matplotlib.axes
        """

        fontsize = kwargs.get("fontsize", 20)
        scope = kwargs.get("scope", "all")
        s_limit = kwargs.get("s_limit", 1500)

        # set up axes
        if ax is None:
            self.fig = plt.figure(figsize=kwargs.get("figsize", (8, 16)))
            ax = self.fig.add_subplot(1, 1, 1)

        # set the scope of data
        if scope == "novel":
            data = self.session["novel_stim_data"]["overall"]
        elif scope == "all":
            data = self.session["data"]
        else:
            data = self.session["novel_stim_data"].get("", None)
            if data is None:
                raise ValueError(
                    "Data scope {0} does not exist, try one of {1}".format(
                        list(scope, self.session["novel_stim_data"].keys())
                    )
                )

        if seperate_by is not None:
            sep_list = np.unique(data[seperate_by])
        else:
            sep_list = [1]

        sides = np.unique(data["stim_side"])
        for i, s in enumerate(sides, 1):
            side_slice = data[data["stim_side"] == s]

            for sep in sep_list:
                if seperate_by is None:
                    temp_slice = side_slice
                else:
                    temp_slice = side_slice[side_slice[seperate_by] == sep]

                # shift wheel according to side
                temp_slice.loc[:, "wheel"] = temp_slice.loc[:, "wheel"] + s

                wheel_stat_dict = get_trajectory_stats(temp_slice)
                avg = wheel_stat_dict["average"]

                # plotting
                for trial in temp_slice.itertuples():
                    # individual trajectories
                    indiv = trial.wheel
                    if len(indiv):
                        if s_limit is not None:

                            indiv = indiv[
                                find_nearest(indiv[:, 0], -200)[0] : find_nearest(
                                    indiv[:, 0], s_limit
                                )[0],
                                :,
                            ]

                        indiv_line = ax.plot(
                            indiv[:, 1],
                            indiv[:, 0],
                            linewidth=3,
                            color=colors[seperate_by][sep][1],
                            alpha=0.3,
                            zorder=1,
                        )
                    if trial.opto == 1:
                        indiv_line[0].set_path_effects(
                            [
                                path_effects.Stroke(
                                    linewidth=2.5, foreground="b", alpha=0.3
                                ),
                                path_effects.Normal(),
                            ]
                        )

                if s_limit is not None:
                    # plot for values between -200 ms and s_limit
                    if avg is not None:
                        avg = avg[
                            find_nearest(avg[:, 0], -200)[0] : find_nearest(
                                avg[:, 0], s_limit
                            )[0]
                        ]

                        # avg_line
                        avg_line = ax.plot(
                            avg[:, 1],
                            avg[:, 0],
                            linewidth=kwargs.get("linewidth", 5),
                            color=colors[seperate_by][sep][0],
                            label="{0} {1}".format(seperate_by, sep) if i == 1 else "_",
                            alpha=1,
                            zorder=2,
                        )

        # closed loop start line
        ax.plot(ax.get_xlim(), [0, 0], "k", linewidth=2, alpha=0.8)

        # trigger zones
        ax.plot([0, 0], ax.get_ylim(), "green", linestyle="--", linewidth=2, alpha=0.8)

        ax.plot([-50, -50], ax.get_ylim(), "red", linestyle="--", linewidth=2, alpha=0.8)
        ax.plot([50, 50], ax.get_ylim(), "red", linestyle="--", linewidth=2, alpha=0.8)

        # make it pretty
        ax = self.pretty_axes(ax)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_ylim([-500, s_limit + 100])
        ax.set_xlim([-125, 125])
        ax.set_xlabel("Wheel Position (deg)", fontsize=fontsize)
        ax.set_ylabel("Time(ms)", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)

        # some logic for embedding in summary figures
        if not kwargs.get("notitle", False):
            ax.set_title(
                "{0}  Wheel Trajectory {1}".format(
                    self.session["meta"]["animalid"], self.session["meta"]["date"]
                ),
                fontsize=fontsize + 2,
                fontweight="bold",
            )
        if kwargs.get("showlegend", True):
            ax.legend(fontsize=fontsize - 5)
        # save
        if kwargs.get("savefig", True):
            self.save("wheeltrajectory_{0}".format(scope))

        return ax

    def metric(self, metric_name, ax=None, *args, **kwargs):
        """Plot a single metric with respec to trials
        :param metric_name : name of the metric to be plotted
        :param ax          : a premade axes object to plot on
        :type metric_name  : str
        :type ax           : matplotlib.axes
        :return ax         : plotted axes
        :rtype             : matplotlib.axes
        """
        fontsize = kwargs.get("fontsize", 22)
        scope = kwargs.get("scope", "all")

        # set up the axes
        if ax is None:
            self.fig = plt.figure(figsize=kwargs.get("figsize", (10, 10)))
            ax = self.fig.add_subplot(1, 1, 1)
            show_legend = True

        # set the scope of data
        if scope == "novel":
            data = self.session["novel_stim_data"]["overall"]
        elif scope == "all":
            data = self.session["data"]
        else:
            data = self.session["novel_stim_data"].get("", None)
            if data is None:
                raise ValueError(
                    "Data scope {0} does not exist, try one of {1}".format(
                        list(scope, self.session["novel_stim_data"].keys())
                    )
                )

        # plotting
        ax.plot(
            data["trial_no"],
            data["running_" + metric_name],
            linewidth=kwargs.get("linewidth", 4),
            color=kwargs.get("color", "k"),
        )

        ax.scatter(
            data["trial_no"],
            data[metric_name],
            c=kwargs.get("color", "gray"),
            s=25,
            alpha=0.4,
        )

        if metric_name == "reaction_t":
            ax.set_yscale("log")
        # zero line
        ax.plot(ax.get_xlim(), [0, 0], linewidth=1, color="k")

        # make it pretty
        if metric_name == "path_surplus":
            y_label = "Path Surplus [norm.]"
        elif metric_name == "reaction_t":
            y_label = "Reaction Time (ms)"
        elif metric_name == "avg_lick_t_diff":
            y_label = "Relative Lick Time (ms)"
        elif metric_name == "avg_speed":
            y_label = "Avg. Stim Speed (deg/ms)"

        ax = self.pretty_axes(ax)
        ax.set_xlabel("Trial No.", fontsize=fontsize)
        ax.set_ylabel(y_label, fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.grid(alpha=0.5, axis="both")

        # save
        if kwargs.get("savefig", True):
            self.save("{0}_{1}".format(metric_name, scope))

        return ax

    def metricSummary(self, *args, **kwargs):
        """Plots a bunch of graphs that summarize the session
        :return fig      : final figure
        :rtype           : matplotlib.figure
        """
        fontsize = kwargs.get("fontsize", 22)
        self.fig = plt.figure(figsize=kwargs.get("figsize", (15, 15)))

        ax1 = self.fig.add_subplot(221)
        ax1 = self.metric(
            "path_surplus", ax=ax1, notitle=True, savefig=False, *args, **kwargs
        )

        ax2 = self.fig.add_subplot(222)
        ax2 = self.metric(
            "reaction_t", ax=ax2, notitle=True, savefig=False, *args, **kwargs
        )

        ax3 = self.fig.add_subplot(223)
        ax3 = self.metric(
            "avg_lick_t_diff", ax=ax3, notitle=True, savefig=False, *args, **kwargs
        )

        ax4 = self.fig.add_subplot(224)
        ax4 = self.metric(
            "avg_speed", ax=ax4, notitle=True, savefig=False, *args, **kwargs
        )

        self.fig.suptitle(
            "{0} Session Metrics {1}".format(
                self.session["meta"]["animalid"], self.session["meta"]["date"]
            ),
            fontsize=fontsize + 3,
            fontweight="bold",
        )
        self.save("metrics_{0}".format(kwargs.get("scope", "all")))
        return self.fig

    def metricMatrix(self, *args, **kwargs):
        pass

    def lickTotal(self, ax=None, *args, **kwargs):
        """Plots the total licks through the trial
        :param ax        : a premade axes object to plot on
        :type ax         : matplotlib.axes
        :return ax       : plotted axes
        :rtype           : matplotlib.axes
        """
        fontsize = kwargs.get("fontsize", 22)
        scope = kwargs.get("scope", "all")

        # set up the axis
        if ax is None:
            self.fig = plt.figure(figsize=kwargs.get("figsize", (16, 8)))
            ax = self.fig.add_subplot(1, 1, 1)

        # set the scope of data
        if scope == "novel":
            data = self.session["novel_stim_data"]["overall"]
        elif scope == "all":
            data = self.session["data"]
        else:
            data = self.session["novel_stim_data"].get("", None)
            if data is None:
                raise ValueError(
                    "Data scope {0} does not exist, try one of {1}".format(
                        list(scope, self.session["novel_stim_data"].keys())
                    )
                )

        # pool lick data
        all_lick = np.array([]).reshape(-1, 2)
        for row in data.itertuples():
            if len(row.lick):
                temp_lick = row.lick.copy()
                temp_lick[:, 0] = +row.openstart_absolute
                all_lick = np.append(all_lick, temp_lick, axis=0)

        # plotting
        if len(all_lick):
            if kwargs.get("trialaxis", True):
                all_lick[:, 0] = (all_lick[:, 0] / np.max(all_lick[:, 0])) * data[
                    "trial_no"
                ].iloc[-1]
                trial_axis = np.interp(data["trial_no"], all_lick[:, 0], all_lick[:, 1])
                ax.plot(
                    data["trial_no"], trial_axis, linewidth=4, color="c", label="Licks"
                )
            else:
                ax.plot(
                    all_lick[:, 0] / 60000,
                    all_lick[:, 1],
                    linewidth=4,
                    color="c",
                    label="Licks",
                )
        else:
            display("No Lick data found for session :(")

        # make it pretty
        ax = self.pretty_axes(ax)
        ax.grid(b=True, alpha=0.5)
        ax.tick_params(labelsize=fontsize)
        ax.set_xlabel("Trial No", fontsize=fontsize)
        ax.set_ylabel("Total Licks", fontsize=fontsize)

        # some logic for embedding in summary figures
        if not kwargs.get("notitle", False):
            ax.set_title(
                "{0} Lick Progression {1}".format(
                    self.session["meta"]["animalid"], self.session["meta"]["date"]
                ),
                fontsize=fontsize,
                fontweight="bold",
            )
        if kwargs.get("showlegend", True):
            ax.legend(fontsize=fontsize - 5)

        # save
        if kwargs.get("savefig", True):
            self.save("licktotal_{0}".format(scope))

        return ax

    def lickHistogram(
        self, bin_dur=500, time_range=[-5000, 5000], ax=None, *args, **kwargs
    ):
        """Plots the reward triggered histogram of licks in correct trials and total licks through the trial
        :param bin_dur   : histogram bin duration in ms or matplotlib hist_methods
        :param ax        : a premade axes object to plot on
        :type bin_dur    : int, float
        :type ax         : matplotlib.axes
        :return ax       : plotted axes
        :rtype           : matplotlib.axes
        """
        fontsize = kwargs.get("fontsize", 22)
        scope = kwargs.get("scope", "all")

        # set up the axis
        if ax is None:
            self.fig = plt.figure(figsize=kwargs.get("figsize", (16, 8)))
            ax = self.fig.add_subplot(1, 1, 1)

        # set the scope of data
        if scope == "novel":
            data = self.session["novel_stim_data"]["overall"]
        elif scope == "all":
            data = self.session["data"]
        else:
            data = self.session["novel_stim_data"].get("", None)
            if data is None:
                raise ValueError(
                    "Data scope {0} does not exist, try one of {1}".format(
                        list(scope, self.session["novel_stim_data"].keys())
                    )
                )

        # pool the lick data
        hist_lick = {"correct": np.array([]), "incorrect": np.array([])}
        hist_lick_incorrect = np.array([])
        for row in data.itertuples():
            if len(row.lick):
                if row.answer == 1:
                    reward_time = (
                        row.reward[0][0]
                        if len(row.reward)
                        else row.closedloopdur[1] + 500
                    )
                    hist_lick["correct"] = np.append(
                        hist_lick["correct"], row.lick[:, 0] - reward_time
                    )
                else:
                    reward_time = row.closedloopdur[1] + 500
                    hist_lick["incorrect"] = np.append(
                        hist_lick["incorrect"], row.lick[:, 0] - reward_time
                    )
        total_len = np.sum([len(x) for x in hist_lick.values()])

        # plotting
        for key in hist_lick.keys():
            if len(hist_lick[key]):

                hist_range = np.array(
                    [
                        x
                        for x in hist_lick[key]
                        if x >= time_range[0] and x <= time_range[1]
                    ]
                )

                bins_l = -1 * np.arange(0, np.abs(np.min(hist_range)), bin_dur)
                bins_r = np.arange(1, np.max(hist_range), bin_dur)
                bins_l = bins_l[::-1]
                bins = np.append(bins_l, bins_r)

                weights = np.ones_like(hist_range) / total_len

                ax.hist(
                    hist_range,
                    bins=bins,
                    weights=weights,
                    color="cyan" if key == "correct" else "darkcyan",
                    alpha=0.5 if key == "incorrect" else 1,
                    rwidth=0.9,
                    label=key,
                )
            else:
                display("No Lick data found for session :(")

        # reward line
        ax.plot([0, 0], ax.get_ylim(), color="r", linewidth=3, label="Reward")

        # avg_first_lick = float(np.nanmean(data.loc[~np.isnan(data['first_lick_t']),'first_lick_t']))
        # axes[0].plot([avg_first_lick,avg_first_lick],axes[0].get_ylim(),color='gray',linewidth=1.5, linestyle=':',label='Avg. First Lick')
        ax.set_xlim(time_range[0] - 100, time_range[1] + 100)

        ax.tick_params(labelsize=fontsize)

        # make it pretty
        ax = self.pretty_axes(ax)
        ax.set_ylabel("Norm. Lick Counts", fontsize=fontsize)
        ax.set_xlabel("Time (ms)", fontsize=fontsize)
        ax.spines["bottom"].set_bounds(ax.set_xlim()[0], ax.set_xlim()[1])
        ax.spines["bottom"].set_position(("outward", 10))
        ax.spines["left"].set_position(("outward", 10))

        # some logic for embedding in summary figures
        if not kwargs.get("notitle", False):
            ax.set_title(
                "{0} Lick Histogram {1}".format(
                    self.session["meta"]["animalid"], self.session["meta"]["date"]
                ),
                fontsize=fontsize,
                fontweight="bold",
            )
        if kwargs.get("showlegend", True):
            ax.legend(loc="upper left", fontsize=fontsize - 5)

        # save
        if kwargs.get("savefig", "True"):
            self.save("lickhist_{0}".format(scope))

        return ax

    def sessionSummary(self, *args, **kwargs):
        """Plots a dashboard of graphs that summarize the session
        :return fig      : final figure
        :rtype           : matplotlib.figure
        """

        fontsize = kwargs.get("fontsize", 22)
        self.fig = plt.figure(figsize=kwargs.get("figsize", (20, 20)))
        widths = [1, 1.5, 2]
        heights = [1, 1, 1]
        gs = self.fig.add_gridspec(
            ncols=3, nrows=3, width_ratios=widths, height_ratios=heights
        )
        # info text
        ax_text = self.fig.add_subplot(gs[0, :1])

        text = self.prep_text()
        ax_text.text(0.01, 0.93, text, va="top", fontsize=fontsize - 3)
        ax_text = self.empty_axes(ax_text)

        # performance progression
        ax_perf = self.fig.add_subplot(gs[2, :2])
        ax_perf = self.parameterFractions(
            param_list=["fraction_correct", "fraction_nogo"],
            colors=["darkgreen", "gray"],
            ax=ax_perf,
            savefig=False,
            notitle=True,
            showlegend=False,
            *args,
            **kwargs,
        )
        h_perf, l_perf = ax_perf.get_legend_handles_labels()

        # total licks
        ax_cumu = ax_perf.twinx()
        ax_cumu = self.lickTotal(
            ax=ax_cumu, savefig=False, notitle=True, showlegend=False, *args, **kwargs
        )
        h_cumu, l_cumu = ax_cumu.get_legend_handles_labels()

        # make two axes fit nicely
        h_perf.extend(h_cumu)
        l_perf.extend(l_cumu)
        ax_cumu.spines["bottom"].set_visible(False)
        ax_cumu.spines["left"].set_visible(False)
        ax_cumu.spines["right"].set_visible(True)
        ax_cumu.spines["right"].set_linewidth(2)
        ax_cumu.grid(axis="y", b=False)
        ax_cumu.legend(h_perf, l_perf, loc="lower left", fontsize=fontsize - 5)

        # fractions
        ax_frac = self.fig.add_subplot(gs[1, :2])
        ax_frac = self.parameterFractions(
            param_list=["fraction_repeat", "fraction_stim_right"],
            colors=["olive", "teal"],
            ax=ax_frac,
            drawtrials=False,
            savefig=False,
            notitle=True,
            showlegend=False,
            *args,
            **kwargs,
        )
        h_frac, l_frac = ax_frac.get_legend_handles_labels()

        # response latency
        ax_time = ax_frac.twinx()
        ax_time = self.responseLatency(
            sep_sides=False,
            ax=ax_time,
            savefig=False,
            notitle=True,
            showlegend=False,
            *args,
            **kwargs,
        )
        h_time, l_time = ax_time.get_legend_handles_labels()

        # make two axes fit nicely
        h_frac.extend(h_time)
        l_frac.extend(l_time)
        ax_time.spines["bottom"].set_visible(False)
        ax_time.spines["left"].set_visible(False)
        ax_time.spines["right"].set_visible(True)
        ax_time.spines["right"].set_linewidth(2)
        ax_time.grid(axis="y", b=False)
        ax_time.legend(h_frac, l_frac, loc="lower left", fontsize=fontsize - 5)

        # psychometric
        ax_psycho = self.fig.add_subplot(gs[0, 2])
        ax_psycho = self.psychometric(
            ax=ax_psycho, notitle=True, savefig=False, showlegend=False, *args, **kwargs
        )

        # lick histogram
        ax_hist = self.fig.add_subplot(gs[0, 1])
        ax_hist = self.lickHistogram(
            ax=ax_hist, notitle=True, savefig=False, *args, **kwargs
        )

        # wheel
        ax_wheel = self.fig.add_subplot(gs[1:, 2])
        ax_wheel = self.wheelTrajectory_vert(
            ax=ax_wheel, notitle=True, savefig=False, *args, **kwargs
        )

        self.fig.tight_layout()
        # save
        self.save("summary_{0}".format(kwargs.get("scope", "all")))

        return self.fig
