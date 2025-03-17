from piepy.wheel.viz.bokeh.bokeh_plotting import WheelTrace
from bokeh.io import curdoc
from bokeh.layouts import column, row, gridplot, layout
from bokeh.models import (
    ColumnDataSource,
    Slider,
    RangeSlider,
    Select,
    CheckboxGroup,
    Panel,
    Tabs,
    Div,
    FileInput,
    Spinner,
)

from bokeh_plotting import (
    DashBoard,
    Graph,
    Performance,
    ResponseTime,
    Psychometric,
    TrialPicture,
    ContrastDistribution,
    AnswerDistribution,
    WheelTrace,
)

import os
import glob
import sys

# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)

from piepy.wheel.wheelSession import *
from piepy.wheel.wheelBehavior import *

config = getConfig()
presentation_folder = config["presentationPath"]
analysis_folder = config["analysisPath"]

for cmd in sys.argv:
    print(cmd)
    if "--animals" in cmd:
        animals = cmd.split("=")[1]
        animal_list = [a for a in animals.split(",")]

# comment this if not debugging
animal_list = ["KC115"]

# a dictionary that maps experiment names with session attributes
# this changes whenever a new animal is selected
mapping_dict = {}
# this is a dictionary to hold a list of sessions for a given animal
main_session_dict = {a: [] for a in animal_list}

for animal in animal_list:
    print(animal, flush=True)
    if "win32" in sys.platform:
        splitter = "\\"
    elif "darwin" in sys.platform:
        splitter = "/"
    # print([s.split(splitter) for s in glob.glob('{0}/*_{1}_*_{2}/'.format(analysis_folder,animal,'KC'))],flush=True)
    main_session_dict[animal] = [
        s.split(splitter)[-2]
        for s in glob.glob("{0}/*_{1}_*_{2}/".format(analysis_folder, animal, "KC"))
    ]

# Instantiate and initialize the DashBoard
dash = DashBoard()
# initialize with first animal in the list
dash.current_animal = animal_list[0]
dash.current_session = main_session_dict[dash.current_animal][-1]
temp_w = WheelSession(sessiondir=dash.current_session, load_flag=True)
mapping_dict[dash.current_session] = temp_w
dash.init_data(temp_w.session)
dash.show_individual = False
dash.show_repeat = False

#############
# AESTHETIC #
#############
main_title = Div(text="<h1> Wheel Behavior Session Dashboard </h1>")

###################
# CONTROL WIDGETS #
###################
# animalselector
dash.add_widget(
    "animal_selector",
    Select(title="Select Animal", options=animal_list, value=animal_list[0]),
)
# session selector
dash.add_widget(
    "session_selector",
    Select(
        title="Select Session",
        options=main_session_dict[dash.current_animal],
        value=dash.current_session,
    ),
)
# trial count slider
dash.add_widget(
    "trial_count_slider",
    RangeSlider(
        start=1,
        end=dash.shown_data[dash.current_scope]["trial_no"].iloc[-1],
        value=(
            dash.shown_data[dash.current_scope]["trial_no"].iloc[0],
            dash.shown_data[dash.current_scope]["trial_no"].iloc[-1],
        ),
        step=1,
        title="Trial Count Filter",
    ),
)
# response time filter
dash.add_widget(
    "response_time_slider",
    Slider(start=200, end=5000, value=200, step=100, title="Response Latency Filter(ms)"),
)
# trial no slider
dash.add_widget(
    "trial_no_spinner",
    Spinner(
        low=1,
        high=dash.shown_data[dash.current_scope]["trial_no"].iloc[-1],
        width=100,
        value=dash.shown_trial["trial_no"],
        step=1,
        title="Trial No",
    ),
)
# scope selector
dash.add_widget(
    "scope_selector",
    Select(
        title="Scope of Data",
        options=dash.scope_list,
        value=dash.current_scope,
        width=150,
    ),
)
# novelity
dash.add_widget(
    "is_novel",
    CheckboxGroup(labels=["Include repeat trials"], active=[int(dash.show_repeat)]),
)

# trial vs time axis checkbox
dash.add_widget(
    "time_axis",
    CheckboxGroup(labels=["Toggle Time x-Axis"], active=[int(dash.time_axis)]),
)
# wheel individual
dash.add_widget(
    "show_individual",
    CheckboxGroup(
        labels=["Show Individual Wheel Traces"], active=[int(dash.show_individual)]
    ),
)

# meta and stats
dash.add_widget("meta_div", Div(text=dash.meta_data, style={"font-size": "15px"}))

dash.add_widget(
    "stats_div", Div(text=dash.stats[dash.current_scope], style={"font-size": "15px"})
)


#############################
# CALLBACKS AND OTHER FUNCS #
#############################


def animal_selector_callback(attr, old, new):
    dash.current_animal = dash.widgets["animal_selector"].value
    print("Animal {0} selected".format(dash.current_animal))
    # refill the session selector
    temp_options = main_session_dict[dash.current_animal]
    temp_options.insert(0, "")
    dash.widgets["session_selector"].options = temp_options


dash.widgets["animal_selector"].on_change("value", animal_selector_callback)


def session_selector_callback(attr, old, new):
    global mapping_dict
    val = dash.widgets["session_selector"].value
    if val != "":
        dash.current_session = val
        print("Session {0} selected".format(dash.current_session))

        # re-initialize the dashboard data
        if dash.current_session in list(mapping_dict.keys()):
            temp_w = mapping_dict[dash.current_session]
        else:
            # load data
            print("Data not in memory yet, loading....")
            temp_w = WheelSession(sessiondir=dash.current_session, load_flag=True)
            mapping_dict[dash.current_session] = temp_w
        dash.init_data(temp_w.session)
        # update the meta and stats
        dash.widgets["meta_div"].text = dash.meta_data
        dash.widgets["stats_div"].text = dash.stats[dash.current_scope]

        # update the scopes according to selected session
        dash.widgets["scope_selector"].options = dash.scope_list
        # change the trial range slider range
        dash.widgets["trial_count_slider"].end = dash.shown_data[dash.current_scope][
            "trial_no"
        ].iloc[-1]
        # initialize the column data sources for all the plots
        for k, g in dash.graphs.items():
            if k != "trialPicture":
                g.set_cds(data=dash.shown_data, time_axis=dash.time_axis)
            else:
                g.set_cds(data=dash.shown_trial)


dash.widgets["session_selector"].on_change("value", session_selector_callback)


def trial_count_slider_callback(attr, old, new):
    trial_count = dash.widgets["trial_count_slider"].value
    dash.filter_data(filters={"trial_limit": trial_count})
    for k, g in dash.graphs.items():
        if k != "trialPicture":
            g.set_cds(data=dash.shown_data, time_axis=dash.time_axis)
        else:
            g.set_cds(data=dash.shown_trial)


dash.widgets["trial_count_slider"].on_change("value", trial_count_slider_callback)


def trial_no_spinner_callback(attr, old, new):
    selected_trial_no = dash.widgets["trial_no_spinner"].value
    dash.set_trial(selected_trial_no)
    dash.graphs["trialPicture"].set_cds(dash.shown_trial)
    dash.graphs["trialPicture"].plot()


dash.widgets["trial_no_spinner"].on_change("value", trial_no_spinner_callback)


def scope_callback(attr, old, new):
    scope = dash.widgets["scope_selector"].value
    is_novel = not dash.widgets["is_novel"].active
    dash.change_repeat(is_novel)
    dash.change_scope(scope)
    dash.set_data()
    # update the stats
    dash.widgets["stats_div"].text = dash.stats[dash.current_scope]
    for k, g in dash.graphs.items():
        if k != "trialPicture":
            g.set_cds(data=dash.shown_data, time_axis=dash.time_axis)
        else:
            g.set_cds(data=dash.shown_trial)


dash.widgets["scope_selector"].on_change("value", scope_callback)


def novel_callback(attr, old, new):
    is_novel = not dash.widgets["is_novel"].active
    scope = dash.widgets["scope_selector"].value
    dash.change_repeat(is_novel)
    dash.change_scope(scope)
    dash.set_data()
    for k, g in dash.graphs.items():
        g.set_cds(data=dash.shown_data, time_axis=dash.time_axis)


dash.widgets["is_novel"].on_change("active", novel_callback)


def time_axis_callback(attr, old, new):
    time_axis = dash.widgets["time_axis"].active
    dash.time_axis = time_axis
    # manually change axis label
    if time_axis:
        label = "Time(s)"
    else:
        label = "Trial No"

    dash.graphs["performance"].fig.xaxis.axis_label = label
    dash.graphs["performance"].set_cds(data=dash.shown_data, time_axis=dash.time_axis)
    dash.graphs["responseTime"].fig.xaxis.axis_label = label
    dash.graphs["responseTime"].set_cds(data=dash.shown_data, time_axis=dash.time_axis)


dash.widgets["time_axis"].on_change("active", time_axis_callback)

#########
# PLOTS #
#########

# performance plot
dash.add_graph("performance", Performance())
dash.add_graph("responseTime", ResponseTime())
dash.add_graph("psychometric", Psychometric())
dash.add_graph("contrastDistribution", ContrastDistribution())
dash.add_graph("answerDistribution", AnswerDistribution())
dash.add_graph("trialPicture", TrialPicture())
dash.add_graph("wheelTrace", WheelTrace())


for k, g in dash.graphs.items():
    if k != "trialPicture":
        print(k, flush=True)
        g.set_cds(data=dash.shown_data, time_axis=dash.time_axis)
    else:
        g.set_cds(data=dash.shown_trial)
    g.plot()

# Arrange plots and widgets in layouts
tab_meta = Panel(child=dash.widgets["meta_div"], title="Meta Data")
tab_stats = Panel(child=dash.widgets["stats_div"], title="Stats.")

tab_perf = Panel(child=dash.graphs["performance"].fig, title="Performance")
tab_time = Panel(child=dash.graphs["responseTime"].fig, title="Response Time")

tab_contrast = Panel(
    child=dash.graphs["contrastDistribution"].fig, title="Contrast Distribution"
)
tab_answer = Panel(
    child=dash.graphs["answerDistribution"].fig, title="Answer Distribution"
)

animal_sesh_ctrl_box = row(
    dash.widgets["animal_selector"], dash.widgets["session_selector"]
)
sesh_ctrl_info_box = column(
    dash.widgets["trial_count_slider"],
    dash.widgets["scope_selector"],
    dash.widgets["response_time_slider"],
    dash.widgets["is_novel"],
    Tabs(tabs=[tab_meta, tab_stats]),
)

perf_resp_tab_box = column(
    row(dash.widgets["time_axis"]), Tabs(tabs=[tab_perf, tab_time])
)

distributions_tab_box = column(Tabs(tabs=[tab_contrast, tab_answer]))

trial_pic_box = column(
    row(dash.widgets["trial_no_spinner"]), dash.graphs["trialPicture"].fig
)

wheel_trace_box = column(
    row(dash.widgets["show_individual"]), dash.graphs["wheelTrace"].fig
)

layout = column(
    main_title,
    animal_sesh_ctrl_box,
    row(
        sesh_ctrl_info_box,
        column(
            perf_resp_tab_box, row(distributions_tab_box, dash.graphs["psychometric"].fig)
        ),
        trial_pic_box,
        wheel_trace_box,
    ),
)

curdoc().add_root(layout)
curdoc().title = "Wheel Session Dashboard"
