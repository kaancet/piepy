from bokeh.models import Div, Select, CustomJS, Spinner, Switch, RangeSlider
from bokeh.layouts import row, column, layout
from behavior_python.plotters.bokeh_plot.dashboard import *
from behavior_python.plotters.bokeh_plot.trial_plots import TrialGraph
from behavior_python.plotters.bokeh_plot.psychometric_plot import PsychometricGraph
from behavior_python.plotters.bokeh_plot.response_time_plots import ReactionTimeScatterGraph
from behavior_python.plotters.bokeh_plot.trial_type_plot import TrialTypeBarGraph
from behavior_python.plotters.bokeh_plot.pattern_image_plot import PatternImageGraph

from behavior_python.detection.wheelDetectionSession import WheelDetectionSession

# hardcoded animal list
animal_list = ['KC139','KC140','KC141','KC142','KC143','KC144','KC145','KC146']

# Instantiate and initialize the DashBoard
dash = DashBoard()
dash.set_animal('KC143') # arbitrary selection
dash.current_session = dash.session_list[-1] # most recent session

w = WheelDetectionSession(sessiondir=dash.current_session,load_flag=True)
dash.set_session(w)

###################
# CONTROL WIDGETS #
###################
# animalselector
dash.add_widget('animal_selector',
                Select(title='Select Animal', options=animal_list,value=dash.current_animal))

# session selector
dash.add_widget('session_selector',
                Select(title='Select Session', options=dash.session_list,value=dash.current_session))

# stim_type selector
dash.add_widget('stimtype_selector',
                Select(title='Select Stimulus Type', options=['test11','test12'],value='test11'))

# trial select
dash.add_widget('trial_selector',
                Spinner(low=1,high=len(dash.data),step=1,value=1,title='Trial No.',width=200))

# reactiontrype selector
dash.add_widget('reaction_type_selector',
                Select(title="Reaction Time Mode", options=['response_latency','wheel_reaction_time','wheel_speed_reaction_time'],value='response_latency'))

# include miss switch
dash.add_widget('include_miss_switch',
                Switch(active=False))

# palette select
dash.add_widget('palette_select',
                Select(title="",options=['Greys','Inferno','Magma','Plasma','Viridis','Cividis','Turbo'],value='Greys'))

# color map range
dash.add_widget('cmap_range_select',
                RangeSlider(title="Adjust colormap range", start=0, end=65536, step=10, value=(0, 65536)))

###################
# INFO WIDGETS #
###################
cols = dash.set_data_table()
# data viewer
dash.add_widget('data_table',
                DataTable(source=dash.dash_cds,columns=cols,width=600,height=500))

#stats text
dash.add_widget('stats_div',
                Div(text=dash.stats,styles={'font-size':'15px'}))

#include miss?
dash.add_widget('switch_text',
                Div(text="Include Miss?",styles={'font-size':'15px'}))

#########
# PLOTS #
#########
dash.add_graph('trial_plot',TrialGraph())
dash.add_graph('psychometric',PsychometricGraph())
dash.add_graph('reaction_time_scatter',ReactionTimeScatterGraph())
dash.add_graph('trial_type',TrialTypeBarGraph())
dash.add_graph('pattern_img',PatternImageGraph())
dash.make_graphs(isInit=True)

#############################
# CALLBACKS AND OTHER FUNCS #
#############################

def animal_selector_callback(attr,old,new):
    dash.current_animal = dash.widgets['animal_selector'].value
    dash.set_animal_sessions()
    # refill the session selector
    dash.widgets['session_selector'].options = dash.session_list
    dash.current_session = dash.session_list[-1]
    dash.widgets['session_selector'].value = dash.current_animal
dash.widgets['animal_selector'].on_change('value',animal_selector_callback)

def session_selector_callback(attr,old,new):
    dash.current_session = dash.widgets['session_selector'].value
    try:
        w = WheelDetectionSession(sessiondir=dash.current_session,load_flag=True)
    except:
        w = WheelDetectionSession(sessiondir=dash.current_session,load_flag=False)
    dash.set_session(w)
    dash.make_graphs()
    dash.widgets['stats_div'].text = dash.stats
dash.widgets['session_selector'].on_change('value',session_selector_callback)

def trial_selector_callback(attr,old,new):
    dash.current_trial_no = dash.widgets['trial_selector'].value
    # dash.dash_cds.selected.indices = [dash.current_trial_no]
    dash.set_trial(dash.current_trial_no)
    dash.graphs['trial_plot'].set_cds(dash.shown_trial)
    # dash.graphs['trial_plot'].plot()
dash.widgets['trial_selector'].on_change('value',trial_selector_callback)

def reaction_type_selector_callback(attr,old,new):
     dash.graphs['reaction_time_scatter'].set_reaction_type(dash.widgets['reaction_type_selector'].value)
     dash.graphs['reaction_time_scatter'].set_cds(dash.data)
dash.widgets['reaction_type_selector'].on_change('value',reaction_type_selector_callback)

def include_miss_switch_callback(attr,old,new):
    dash.graphs['reaction_time_scatter'].include_miss = not dash.graphs['reaction_time_scatter'].include_miss
    dash.graphs['reaction_time_scatter'].set_cds(dash.data)
    
dash.widgets['include_miss_switch'].on_change('active',include_miss_switch_callback)

def palette_select_callback(attr,old,new):
    dash.graphs['pattern_img'].set_palette(dash.widgets["palette_select"].value)
dash.widgets['palette_select'].on_change('value',palette_select_callback)

def colormap_range_select_callback(attr,old,new):
    low,high = dash.widgets['cmap_range_select'].value
    dash.graphs['pattern_img'].set_colormap_range(low,high)
dash.widgets['cmap_range_select'].on_change('value',colormap_range_select_callback)

source_code_datatable = """
const row = cb_obj.indices[0];
trial_spinner.value = source.data["trial_no"][row];
"""
datatable_callback = CustomJS(args=dict(source=dash.dash_cds,trial_spinner=dash.widgets['trial_selector']), code=source_code_datatable)
dash.dash_cds.selected.js_on_change('indices', datatable_callback)

# source_code_plot = """
# const row = cb_obj.selected[0];
# console.log("helleo");
# trial_spinner.value = source.data["trial_no"][row];
# """
# plot_click_callback = CustomJS(args=dict(source=dash.graphs['reaction_time_scatter'].cds_dots,trial_spinner=dash.widgets['trial_selector']), code=source_code_datatable)
# dash.graphs['reaction_time_scatter'].cds_dots.selected.js_on_change('indices',plot_click_callback)


# final arrangement of the layout
controls = column(dash.widgets['animal_selector'],
                  dash.widgets['session_selector'],
                  dash.widgets['stats_div'])

psycho_plot = column(dash.graphs['psychometric'].fig)

resp_scatter_plot = column(dash.graphs['reaction_time_scatter'].fig,
                           row(dash.widgets['reaction_type_selector'],
                               column(dash.widgets['switch_text'],
                                      dash.widgets['include_miss_switch'])))
trial_type_plot = column(dash.graphs['trial_type'].fig)

trial_plot = row(column(dash.widgets['trial_selector'],
                        dash.widgets['data_table']),
                dash.graphs['trial_plot'].fig)

pattern_plot = column(row(dash.widgets['palette_select'],dash.widgets['cmap_range_select']),
                      dash.graphs['pattern_img'].fig)

layout = column(row(controls,psycho_plot,resp_scatter_plot,trial_type_plot),
                row(trial_plot,pattern_plot))


curdoc().add_root(layout)
set_theme('light')
curdoc().title = 'Session Dashboard'