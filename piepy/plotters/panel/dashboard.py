import glob
import os
import numpy as np
from os.path import join as pjoin
from bokeh.models import DataTable, TableColumn, StringFormatter

from piepy.detection.wheelDetectionSession import WheelDetectionSession
from piepy.core.utils import getConfig
from piepy.plotters.bokeh_plot.bokeh_base import *


class DashBoard:
    def __init__(self):
        self.current_mode = None
        self.current_animal = None
        self.current_session = None
        self.current_trial_no = None

        self.data = None
        self.dash_cds = None
        self.pattern_imgs = None
        self.patern_bmps = None
        self.stats = None
        self.stats_text = ""
        self.shown_trial = None

        self.widgets = {}
        self.graphs = {}
        self.callbacks = {}
        self.session_list = []

        self.config = getConfig()

    def add_graph(self, graph_name: str, graph2add) -> None:
        self.graphs[graph_name] = graph2add

    def add_widget(self, widget_name: str, widget2add) -> None:
        self.widgets[widget_name] = widget2add

    def set_mode(self, mode: str) -> None:
        self.current_mode = mode
        self.set_animal_sessions()

    def set_animal(self, animalid: str) -> None:
        self.current_animal = animalid
        self.set_animal_sessions()

    def set_animal_sessions(self) -> None:
        if self.current_mode is None or self.current_animal is None:
            self.current_mode = "presentation"
            self.current_animal = "KC143"
        path_key = f"{self.current_mode}Path"
        path2look = pjoin(
            self.config[path_key],
            f"*_{self.current_animal}_detect_*",
        )
        self.session_list = [
            s.split(os.sep)[-1] for s in glob.glob(path2look)
        ]  # strip the J:\\presentation(or training)

    def set_session(self, session: str = None) -> None:
        if session is None:
            session = self.session_list[-1]
        self.current_session = session
        try:
            session_obj = WheelDetectionSession(
                sessiondir=self.current_session, load_flag=True
            )
        except:
            session_obj = WheelDetectionSession(
                sessiondir=self.current_session, load_flag=False
            )

        self.data = session_obj.data.data
        self.stats = session_obj.stats
        if self.current_mode == "presentation":
            self.pattern_img = session_obj.data.pattern_imgs[
                [k for k in session_obj.data.pattern_imgs.keys()][1]
            ]  # get the second key
            self.pattern_bmps = session_obj.data.patterns
        else:
            self.pattern_img = np.zeros((768, 1024))
            self.pattern_bmps = np.zeros((768, 1024))
        self.stats_text = self.prep_stats(session_obj.stats)
        self.set_data_table()
        self.set_trial()

    def set_trial(self, trial_no: int = 1) -> None:
        self.shown_trial = self.data.filter(pl.col("trial_no") == trial_no)

    def set_data_table(self) -> list:
        """Selects parts of the data to be shown in the data table"""
        columns_to_show = [
            "trial_no",
            "contrast",
            "outcome",
            "stim_type",
            "stim_side",
            "opto_pattern",
            "opto_region",
        ]
        data = self.data.select(columns_to_show)

        # make the columns
        columns = [TableColumn(field=c, title=c) for c in columns_to_show]
        # just to make first column bold
        columns[0] = TableColumn(
            field="trial_no",
            title="trial_no",
            formatter=StringFormatter(font_style="bold"),
        )
        if self.dash_cds is None:
            self.dash_cds = ColumnDataSource(data=data.to_dict(as_series=False))
        else:
            self.dash_cds.data = data.to_dict(as_series=False)

        return columns

    def make_graphs(self, isInit: bool = False):
        for k, g in self.graphs.items():
            if k == "trial_plot":
                g.set_cds(data=self.shown_trial)
            elif k == "pattern_img":
                g.set_cds(data=self.pattern_img)
            else:
                g.set_cds(data=self.data)
            if isInit:
                g.plot()

    def get_val_of(self, summary_of: str) -> str:
        try:
            val = str(getattr(self.stats, summary_of))
        except:
            val = "n/a"

        if summary_of == "hit_rate":
            val = "HR=" + val + "%"
        elif summary_of == "false_alarm":
            val = "FA=" + val + "%"
        elif summary_of == "median_response_time":
            val = "RT=" + val + "ms"
        elif summary_of == "stim_count":
            val = "n=" + val
        return val

    @staticmethod
    def prep_stats(stat_obj) -> str:
        """Prepares the stats to be printable"""
        ret = "<pre>"
        for k in stat_obj.__slots__:
            ret += f"<b>{k} :</b> {getattr(stat_obj,k)}\n"
        ret += "</pre>"
        return ret
