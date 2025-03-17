import panel as pn
import numpy as np
import time

from piepy.psychophysics.detection.wheelDetectionSession import WheelDetectionSession
from piepy.plotters.plotting_utils import Color

from bokeh.plotting import figure, show

clr = Color()

pn.extension()


# STATES OF THE APP / GLOBAL VARIABLES
is_loading = pn.rx(False)
rx_loaded = is_loading.rx.where("Loading Session...", "Load Session")

current_session = None


time_in_trial = 0


# CONTROL PANEL

# selectors

animal_selector = pn.widgets.Select(
    description="Select Animal",
    name="Animal ID",
    options=["KC147", "KC148", "KC149"],
    width=250,
)
area_selector = pn.widgets.Select(
    description="Select Target Area",
    name="Session",
    options=["a", "b", "c"],
    width=250,
)

load_button = pn.widgets.Button(name=rx_loaded, button_type="primary", width=250)

load_indicator = pn.indicators.Progress(
    name="Progress", active=True, width=250, bar_color="success", visible=False
)


# temp function
def load_session(do_load):
    # UI stuff
    is_loading.rx.value = True
    load_indicator.active = True
    load_indicator.visible = True
    load_indicator.value = -1
    load_button.disabled = True

    load_indicator.visible = False
    load_button.disabled = False
    is_loading.rx.value = False


load_button.rx.watch(load_session)

## performance indicators
# 1. Trial count
trial_count_indicator = pn.indicators.Number(
    name="Trial Count",
    value=500,
    colors=[(200, "#ad0c00"), (650, "#e39f00"), (700, "#178a00")],
    font_size="30pt",
    title_size="18pt",
)

# 2. Hit rate (non opto)
hit_rate_indicator = pn.indicators.Number(
    name="Hit Rate",
    value=78,
    format="{value}%",
    colors=clr.gen_color_normalized(
        cmap="RdYlGn", data_arr=[25, 50, 75, 90], vmin=0, vmax=100
    ),
    font_size="30pt",
    title_size="18pt",
)

# 3. False alarm Rate
false_alarm_indicator = pn.indicators.Number(
    name="False Alarm Rate",
    value=25,
    format="{value}%",
    colors=clr.gen_color_normalized(
        cmap="RdYlGn", data_arr=[25, 50, 75, 90], vmin=0, vmax=100, reverse=True
    ),
    font_size="30pt",
    title_size="18pt",
)

# 4. Median reaction time
reaction_time_indicator = pn.indicators.Number(
    name="Median Reaction Time",
    value=450,
    format="{value}ms",
    colors=clr.gen_color_normalized(
        cmap="RdYlGn", data_arr=[300, 450, 750, 900], vmin=200, vmax=1000, reverse=True
    ),
    font_size="30pt",
    title_size="18pt",
)

perf_indicators = pn.Column(
    trial_count_indicator,
    hit_rate_indicator,
    false_alarm_indicator,
    reaction_time_indicator,
    width=180,
    height=500,
)


# plots and views

# tabs viewer for opto_pattern, facecam, eyecam, onepcam
N = 500
x = np.linspace(0, 10, N)
y = np.linspace(0, 10, N)
xx, yy = np.meshgrid(x, y)
d = np.sin(xx) * np.cos(yy)

p1 = figure(
    width=500, height=500, tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")]
)
p1.x_range.range_padding = p1.y_range.range_padding = 0

# must give a vector of image data for image parameter
p1.image(image=[d], x=0, y=0, dw=10, dh=10, palette="Spectral11", level="image")
p1.grid.grid_line_width = 0.5

p2 = figure(width=500, height=500, name="Line")
p2.line([0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 2, 1, 0])

tabs = pn.Tabs(("Image", p1), ("Line", p2))


# Instantiate the template with widgets displayed in the sidebar
template = pn.template.MaterialTemplate(
    title="Detection Experiment Dashboard",
    sidebar=[animal_selector, session_selector, load_button, load_indicator],
    sidebar_width=300,
    logo=r"C:\Users\kaan\code\piepy\res\piepy_logo.png",
    header_background="#678856",
    site_url="99",
)

template.main.append(
    pn.Row(pn.Card(perf_indicators, title="Performance"), pn.Card(tabs, title="Image"))
)


template.servable()
# pn.FlexBox(pn.Column(title,
#                      animal_selector,
#                      session_selector,
#                      load_button,
#                      load_indicator,
#                      )).servable()
