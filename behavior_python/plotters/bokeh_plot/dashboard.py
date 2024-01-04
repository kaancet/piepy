from bokeh.models import DataTable, TableColumn, StringFormatter
from behavior_python.plotters.bokeh_plot.bokeh_base import *

import glob
import os
from os.path import join as pjoin
from behavior_python.utils import getConfig


class DashBoard:
    def __init__(self):
        self.current_animal = None
        self.current_session = None
        self.current_trial_no = None
        
        self.data = None
        self.dash_cds = None
        self.pattern_imgs = None
        self.patern_bmps = None
        self.stats = None
        self.shown_trial = None
        
        self.widgets = {}
        self.graphs = {}
        self.callbacks = {}
        self.session_list = []
        
        self.config = getConfig()
        
    def add_graph(self,graph_name,graph2add) -> None:
        self.graphs[graph_name] = graph2add

    def add_widget(self,widget_name,widget2add) -> None:
        self.widgets[widget_name] = widget2add
        
    def set_animal(self,animalid:str) -> None:
        self.current_animal = animalid
        self.set_animal_sessions()
        
    def set_animal_sessions(self) -> None:
        path2look = pjoin(self.config['presentationPath'],f"*_{self.current_animal}_detect_opto*",)
        self.session_list = [s.split(os.sep)[-1] for s in glob.glob(path2look)] # strip the J:\\presentation
        
    def set_session(self, session_obj) -> None:
        self.data = session_obj.data.data
        self.pattern_img = session_obj.data.pattern_imgs[[k for k in session_obj.data.pattern_imgs.keys()][1]] # get the second key
        self.patern_bmps = session_obj.data.patterns
        self.stats = self.prep_stats(session_obj.stats)
        self.set_trial()
        
    def set_trial(self,trial_no:int=1) -> None:
        self.shown_trial = self.data.filter(pl.col('trial_no')==trial_no)
    
    def set_data_table(self) -> list:
        """ Selects parts of the data to be shown in the data table """
        columns_to_show = ['trial_no','contrast','outcome','stim_type','stim_side','opto_pattern','opto_region']
        data = self.data.select(columns_to_show)
        
        # make the columns
        columns = [TableColumn(field=c, title=c) for c in columns_to_show]
        # just to make first column bold
        columns[0] = TableColumn(field='trial_no',title='trial_no',formatter=StringFormatter(font_style="bold"))
        self.dash_cds = ColumnDataSource(data=data.to_dict(as_series=False))

        return columns

    def make_graphs(self,isInit:bool=False):
        for k,g in self.graphs.items():
            if k == 'trial_plot':
                g.set_cds(data=self.shown_trial)
            elif k == 'pattern_img':
                g.set_cds(data=self.pattern_img)
            else:
                g.set_cds(data=self.data)
            if isInit:
                g.plot()
        
    @staticmethod
    def prep_stats(stat_obj) -> str:
        """ Prepares the stats to be printable"""
        ret = "<pre>"
        for k in stat_obj.__slots__:
            ret += f"<b>{k} :</b> {getattr(stat_obj,k)}\n"
        ret += "</pre>"
        return ret