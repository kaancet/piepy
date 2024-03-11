from bokeh.plotting import figure, show, save, output_file
from bokeh.models import LinearAxis, Plot, ColumnDataSource, HoverTool
from bokeh.core.properties import value
from bokeh.io import curdoc, export_svgs

from bokeh.themes import Theme
import svglib.svglib as svglib
from reportlab.graphics import renderPDF

import os
from behavior_python.utils import display
import polars as pl


class Graph():
    """ Abstract class for individual bokeh plots on the dashboard """
    def __init__(self,*args,**kwargs):
        self.fig = None
        self.possible_stims = [
                                "0.04cpd_8.0Hz_-1",
                                "0.04cpd_8.0Hz_0",
                                "0.04cpd_8.0Hz_1",
                                "0.08cpd_2.0Hz_-1",
                                "0.08cpd_2.0Hz_0",
                                "0.08cpd_2.0Hz_1",
                                "0.16cpd_0.5Hz_-1",
                                "0.16cpd_0.5Hz_0",
                                "0.16cpd_0.5Hz_1",
                                "0.1cpd_4.0Hz_-1",
                                "0.1cpd_4.0Hz_0",
                                "0.1cpd_4.0Hz_1",
                                "0.1cpd_4.0Hz_grating_-1",
                                "0.1cpd_4.0Hz_grating_0",
                                "0.1cpd_4.0Hz_grating_1"
                              ]
        
    @staticmethod    
    def morph_data(data:pl.DataFrame) -> dict:
        pass

    def save(self,plotname):
        pass

    def plot(self,*args,**kwargs):
        pass

    def set_cds(self):
        pass
    
    def update_cds(self):
        pass
    

def select_stim_data(data_in:pl.DataFrame, stimkey:str=None, drop_early:bool=True):
    """ Returns the selected stimulus type from session data
            data_in : 
            stimkey : Dictionary key that corresponds to the stimulus type (e.g. lowSF_highTF)
    """
    # drop early trials
    if drop_early:
        data = data_in.filter(pl.col('outcome')!=-1)
    else:
        data = data_in.select(pl.col('*'))
        
    #should be no need for drop_nulls, but for extra failsafe
    uniq_keys = data.select(pl.col('stimkey')).drop_nulls().unique().to_series().to_numpy()

    if stimkey is not None and stimkey not in uniq_keys and stimkey != 'all':
        raise KeyError(f'{stimkey} not in stimulus data, try one of these: {uniq_keys}')

    if stimkey is not None:
        # this is the condition that is filtering the dataframe by stimkey
        key = stimkey
        data = data.filter(pl.col('stimkey') == stimkey)
    else:
        if len(uniq_keys) == 1:
            # if there is only one key just take the data
            key = uniq_keys[0]
        elif len(uniq_keys) > 1:
            # if there is more than one stimkey , take all the data
            key = 'all'
        else:
            # this should not happen
            raise ValueError('There is no stimkey in the data, this should not be the case. Check your data!')
        
    return data, key, uniq_keys
    
def set_theme(theme:str='light') -> None:
    themes_dir = 'C:\\Users\\kaan\\code\\visual-perception\\behavior_python\\plotters\\bokeh_plot\\themes'
    themes_list = os.listdir(themes_dir)
    if not f'{theme}_theme.yml' in themes_list:
        display(f'{theme} not in available custom themes. Stting the theme to light for now. Use one of: {themes_list}')
        theme = 'light'
        
    curdoc().theme = Theme(filename=f"{themes_dir}\\{theme}_theme.yml")
    
# def save_plot(f, save_name:str) -> None:
#     """ Bokeh can't directly save to pdf, so we first save as svg, read it and convert it to pdf """
#     f.output_backend = "svg"
#     save_path
#     # save as svg
#     export_svgs(f,filename=)
    
#     # see comment 2
#     svglib.register_font('helvetica', '/home/fonts/Helvetica.ttf')
#     # step 2: read in svg
#     svg = svglib.svg2rlg(test_name+".svg")

#     # step 3: save as pdf
#     renderPDF.drawToFile(svg, test_name+".pdf")
