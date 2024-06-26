from bokeh.models import (
    Div,
    Select,
    CustomJS,
    Spinner,
    Switch,
    RangeSlider,
    TabPanel,
    Tabs,
)
from bokeh.layouts import row, column, layout
from piepy.plotters.bokeh_plot.dashboard import *
from piepy.plotters.bokeh_plot.trial_plots import TrialGraph
from piepy.plotters.bokeh_plot.psychometric_plot import PsychometricGraph
from piepy.plotters.bokeh_plot.response_time_plots import ReactionTimeScatterGraph
from piepy.plotters.bokeh_plot.trial_type_plot import TrialTypeBarGraph
from piepy.plotters.bokeh_plot.pattern_image_plot import PatternImageGraph


summary_div_style = {
    "font-size": "30px",
    "font-family": " Helvetica, Arial, sans-serif",
    "font-weight": "bold",
    "text-align": "center",
    "padding": "15px",
    "margin": "10px",
    "color": "#FFFFFF",
    "border-radius": "15px",
    "background": "#e6402e",
}

summary_div_key = {
    0: "stim_count",
    1: "hit_rate",
    2: "false_alarm",
    3: "median_response_time",
}

# hardcoded animal list
animal_list = [
    "KC139",
    "KC141",
    "KC142",
    "KC143",
    "KC144",
    "KC145",
    "KC146",
    "KC147",
    "KC148",
    "KC149",
]

# Instantiate and initialize the DashBoard
dash = DashBoard()
dash.set_mode("presentation")
dash.set_animal("KC147")  # arbitrary selection
dash.set_session(
    "240228_KC147_detect_opto120_dorsal__1P_KC"
)  # this will set the last session in session list

###################
# CONTROL WIDGETS #
###################
# experiment/trial switch
dash.add_widget(
    "mode_selector",
    Select(
        title="Training/Experiment",
        options=["training", "presentation"],
        value=dash.current_mode,
    ),
)

# animalselector
dash.add_widget(
    "animal_selector",
    Select(title="Select Animal", options=animal_list, value=dash.current_animal),
)

# session selector
dash.add_widget(
    "session_selector",
    Select(title="Select Session", options=dash.session_list, value=dash.current_session),
)

# stim_type selector
dash.add_widget(
    "stimtype_selector",
    Select(title="Select Stimulus Type", options=["test11", "test12"], value="test11"),
)

# trial select
dash.add_widget(
    "trial_selector",
    Spinner(low=1, high=len(dash.data), step=1, value=1, title="Trial No.", width=200),
)

# reactiontrype selector
dash.add_widget(
    "reaction_type_selector",
    Select(
        title="Reaction Time Mode",
        options=["response_latency", "pos_reaction_time", "speed_reaction_time"],
        value="response_latency",
    ),
)

# include miss switch
dash.add_widget("include_miss_switch", Switch(active=False))

# palette select
dash.add_widget(
    "palette_select",
    Select(
        title="",
        options=["Greys", "Inferno", "Magma", "Plasma", "Viridis", "Cividis", "Turbo"],
        value="Greys",
    ),
)

# color map range
dash.add_widget(
    "cmap_range_select",
    RangeSlider(
        title="Adjust colormap range", start=0, end=65536, step=10, value=(0, 65536)
    ),
)

###################
# INFO WIDGETS #
###################
cols = dash.set_data_table()
# data viewer
dash.add_widget(
    "data_table", DataTable(source=dash.dash_cds, columns=cols, width=600, height=500)
)
# a list of
dash.add_widget(
    "summary_divs",
    [
        Div(text=dash.get_val_of(summary_div_key[i]), styles=summary_div_style)
        for i in range(4)
    ],
)

# stats text
dash.add_widget("stats_div", Div(text=dash.stats_text, styles={"font-size": "15px"}))

# include miss?
dash.add_widget("switch_text", Div(text="Include Miss?", styles={"font-size": "15px"}))


#########
# PLOTS #
#########
dash.add_graph("trial_plot", TrialGraph())
dash.add_graph("psychometric", PsychometricGraph())
dash.add_graph("reaction_time_scatter", ReactionTimeScatterGraph())
dash.add_graph("trial_type", TrialTypeBarGraph())
dash.add_graph("pattern_img", PatternImageGraph())
dash.make_graphs(isInit=True)


#############################
# CALLBACKS AND OTHER FUNCS #
#############################
def mode_selector_callback(attr, new, old):
    print(dash.widgets["mode_selector"].value)
    dash.set_mode(dash.widgets["mode_selector"].value)
    dash.widgets["session_selector"].options = dash.session_list
    dash.set_session(dash.session_list[-1])
    dash.widgets["session_selector"].value = dash.current_session


dash.widgets["mode_selector"].on_change("value", mode_selector_callback)


def animal_selector_callback(attr, old, new):
    dash.set_animal(dash.widgets["animal_selector"].value)
    dash.widgets["session_selector"].options = dash.session_list
    dash.set_session(dash.session_list[-1])
    dash.widgets["session_selector"].value = dash.current_session


dash.widgets["animal_selector"].on_change("value", animal_selector_callback)


def session_selector_callback(attr, old, new):
    dash.set_session(dash.widgets["session_selector"].value)
    dash.make_graphs()
    dash.widgets["stats_div"].text = dash.stats_text
    # set the summary div values
    for i, d in enumerate(dash.widgets["summary_divs"]):
        d.text = dash.get_val_of(summary_div_key[i])


dash.widgets["session_selector"].on_change("value", session_selector_callback)


def trial_selector_callback(attr, old, new):
    dash.current_trial_no = dash.widgets["trial_selector"].value
    # dash.dash_cds.selected.indices = [dash.current_trial_no]
    dash.set_trial(dash.current_trial_no)
    dash.graphs["trial_plot"].set_cds(dash.shown_trial)
    # dash.graphs['trial_plot'].plot()


dash.widgets["trial_selector"].on_change("value", trial_selector_callback)


def reaction_type_selector_callback(attr, old, new):
    dash.graphs["reaction_time_scatter"].set_reaction_type(
        dash.widgets["reaction_type_selector"].value
    )
    dash.graphs["reaction_time_scatter"].set_cds(dash.data)


dash.widgets["reaction_type_selector"].on_change("value", reaction_type_selector_callback)


def include_miss_switch_callback(attr, old, new):
    dash.graphs["reaction_time_scatter"].include_miss = not dash.graphs[
        "reaction_time_scatter"
    ].include_miss
    dash.graphs["reaction_time_scatter"].set_cds(dash.data)


dash.widgets["include_miss_switch"].on_change("active", include_miss_switch_callback)


def palette_select_callback(attr, old, new):
    dash.graphs["pattern_img"].set_palette(dash.widgets["palette_select"].value)


dash.widgets["palette_select"].on_change("value", palette_select_callback)


def colormap_range_select_callback(attr, old, new):
    low, high = dash.widgets["cmap_range_select"].value
    dash.graphs["pattern_img"].set_colormap_range(low, high)


dash.widgets["cmap_range_select"].on_change("value", colormap_range_select_callback)

source_code_datatable = """
const row = cb_obj.indices[0];
trial_spinner.value = source.data["trial_no"][row];
"""
datatable_callback = CustomJS(
    args=dict(source=dash.dash_cds, trial_spinner=dash.widgets["trial_selector"]),
    code=source_code_datatable,
)
dash.dash_cds.selected.js_on_change("indices", datatable_callback)

# source_code_plot = """
# const row = cb_obj.selected[0];
# console.log("helleo");
# trial_spinner.value = source.data["trial_no"][row];
# """
# plot_click_callback = CustomJS(args=dict(source=dash.graphs['reaction_time_scatter'].cds_dots,trial_spinner=dash.widgets['trial_selector']), code=source_code_datatable)
# dash.graphs['reaction_time_scatter'].cds_dots.selected.js_on_change('indices',plot_click_callback)


# put two stat relatet widgets to tabpanels
summary_tabs = column(
    row(dash.widgets["summary_divs"][0], dash.widgets["summary_divs"][1]),
    row(dash.widgets["summary_divs"][2], dash.widgets["summary_divs"][3]),
)

tab_summary = TabPanel(child=summary_tabs, title="Summary")
tab_stats = TabPanel(child=dash.widgets["stats_div"], title="Stats")

# final arrangement of the layout
controls = column(
    dash.widgets["mode_selector"],
    dash.widgets["animal_selector"],
    dash.widgets["session_selector"],
    Tabs(tabs=[tab_summary, tab_stats]),
)

psycho_plot = column(dash.graphs["psychometric"].fig)

resp_scatter_plot = column(
    dash.graphs["reaction_time_scatter"].fig,
    row(
        dash.widgets["reaction_type_selector"],
        column(dash.widgets["switch_text"], dash.widgets["include_miss_switch"]),
    ),
)
trial_type_plot = column(dash.graphs["trial_type"].fig)

trial_plot = row(
    column(dash.widgets["trial_selector"], dash.widgets["data_table"]),
    dash.graphs["trial_plot"].fig,
)

pattern_plot = column(
    row(dash.widgets["palette_select"], dash.widgets["cmap_range_select"]),
    dash.graphs["pattern_img"].fig,
)

layout = column(
    row(controls, psycho_plot, resp_scatter_plot, trial_type_plot),
    row(trial_plot, pattern_plot),
)


curdoc().add_root(layout)
set_theme("light")
curdoc().title = "Session Dashboard"
