import os
import numpy as np
import pandas as pd
from cycler import cycler
import matplotlib.pyplot as plt
from datetime import timedelta
from datetime import datetime as dt
from collections import defaultdict
from matplotlib import colors as mcolors
from behavior_python.utils import display
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator,MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ..utils import *
import colorsys


mplstyledict = {}
# styledict for day-to-day analysis
mplstyledict['analysis'] = {'pdf.fonttype' : 42,
                            'ps.fonttype' : 42,
                            'axes.titlesize' : 16,
                            'axes.labelsize' : 14,
                            'axes.facecolor': 'none',
                            'axes.linewidth' : 2,
                            'axes.spines.right': False,
                            'axes.spines.top': False,
                            'axes.titleweight': 'bold',
                            'ytick.major.size': 9,
                            'xtick.major.size': 9,
                            'ytick.minor.size': 7,
                            'xtick.minor.size': 7,
                            'xtick.labelsize' : 12,
                            'ytick.labelsize' : 12,
                            'xtick.major.width': 2,
                            'ytick.major.width': 2,
                            'xtick.minor.width':1,
                            'ytick.minor.width':1,
                            'figure.edgecolor': 'none',
                            'figure.facecolor': 'none',
                            'figure.frameon': False,
                            'font.family': ['sans-serif'],
                            'font.size'  : 12,
                            'font.sans-serif': ['Helvetica',
                                                'Arial',
                                                'DejaVu Sans',
                                                'Bitstream Vera Sans',
                                                'Computer Modern Sans Serif',
                                                'Lucida Grande',
                                                'Verdana',
                                                'Geneva',
                                                'Lucid',
                                                'Avant Garde',
                                                'sans-serif'],             
                            'lines.linewidth' : 1.5,
                            'lines.markersize' : 4,
                            'image.interpolation': 'none',
                            'image.resample': False,}
# styledict for putting in presentations
mplstyledict['presentation'] = {'pdf.fonttype' : 42,
                                'ps.fonttype' : 42,
                                'axes.titlesize' : 16,
                                'axes.labelsize' : 14,
                                'axes.facecolor': 'none',
                                'axes.linewidth' : 15,
                                'axes.spines.right': False,
                                'axes.spines.top': False,
                                'axes.titleweight': 'bold',
                                'ytick.major.size': 9,
                                'xtick.major.size': 9,
                                'ytick.minor.size': 7,
                                'xtick.minor.size': 7,
                                'xtick.labelsize' : 12,
                                'ytick.labelsize' : 12,
                                'xtick.major.width': 10,
                                'ytick.major.width': 10,
                                'xtick.minor.width':1,
                                'ytick.minor.width':1,
                                'figure.edgecolor': 'none',
                                'figure.facecolor': 'none',
                                'figure.frameon': False,
                                'font.family': ['sans-serif'],
                                'font.size'  : 12,
                                'font.sans-serif': ['Helvetica',
                                                    'Arial',
                                                    'DejaVu Sans',
                                                    'Bitstream Vera Sans',
                                                    'Computer Modern Sans Serif',
                                                    'Lucida Grande',
                                                    'Verdana',
                                                    'Geneva',
                                                    'Lucid',
                                                    'Avant Garde',
                                                    'sans-serif'],             
                                'lines.linewidth' : 1.5,
                                'lines.markersize' : 4,
                                'image.interpolation': 'none',
                                'image.resample': False,}
# styledict for putting in word etc.
mplstyledict['print']= {}

def set_style(styledict='analysis'):
    if styledict in ['analysis','presentation','print']:
        plt.style.use(mplstyledict[styledict])
    else:
        try:
            plt.style.use(styledict)
        except KeyError:
            plt.style.use('default')
            display(f'Matplotlib {styledict} style is nonexistant, using default style')
    display(f'Changed plotting style to {styledict}')
   
contrast_cycler = (cycler(color=['orangered',
                                 'darkgoldenrod',
                                 'turquoise',
                                 'limegreen',
                                 'darkorchid']))

iter_contrasts = iter(contrast_cycler)
contrast_styles = defaultdict(lambda : next(iter_contrasts))
contrast_styles = {1:{'color':'teal'},
                   0.5:{'color':'darkgoldenrod'},
                   0.25:{'color':'turquoise'},
                   0.125:{'color':'limegreen'},
                   0.0625:{'color':'indigo'},
                   0.03125:{'color':'hotpink'},
                   0:{'color':'black'}}


def create_color_palette(flex_size=10):
    """ Creates a color palette where there is a fixed and flexible part
    The fixed part is for common stimulus types and flexible is for new stim types
    flex_size argument determines how many the flexible part has """
    
    colors = {**mcolors.BASE_COLORS, **mcolors.CSS4_COLORS}
    chosen = np.random.choice(list(colors.keys()),replace=False,size=10)

    stim_cycler = (cycler(color=chosen))
    iter_styles = iter(stim_cycler)
    return defaultdict(lambda : next(iter_styles))

# fixed part
stim_styles = {'grating':{'color':'tab:gray'},
               '0.05SF_2TF':{'color':'tab:purple'},
               'lowSF_highTF':{'color':'tab:orange'},
               'highSF_lowTF':{'color':'tab:purple'},
               'lowSF_highTF_opto_0':{'color':'tab:green'},
               'lowSF_highTF_opto_1':{'color':'tab:red'},
               'highSF_lowTF_opto_0':{'color':'tab:cyan'},
               'highSF_lowTF_opto_1':{'color':'tab:pink'},
               'lowSF_highTF_opto_-1':{'color':'tab:green'}}

# flexible part
stim_styles_flex = create_color_palette()

def get_color(stim_key):
    """ Returns the color corresponding to the stimulus key
        If not present chooses from color cyler and puts it in the fixed colors"""
    if stim_key not in stim_styles.keys():
        new_color = stim_styles_flex[stim_key]
        stim_styles[stim_key] = {'color':new_color}
        
    return stim_styles[stim_key]


def plot_mondays(ax:plt.Axes,date_arr:list,dt_flag=True):
    """ Marks the mondays on a provided axes
    dt_flag determines whether mondays are calculated from """
    
    mondays = [day.weekday() for day in date_arr]
    ylims = ax.get_ylim()
    for row in mondays:
        ax.plot([row,row],ylims,
                color='gray',
                linewidth='4',
                alpha=0.5,
                zorder=1)


    ax.set_xticks(mond_list)
    ax.set_xticklabels(mond_list)
    return ax

def dates_to_deltadays(date_arr:list,start_date=dt.date):
    """ Converts the date to days from first start """
    date_diff = [(day-start_date).days for day in date_arr]
    return date_diff


class Color:
    def __init__(self):
        pass
    
    @staticmethod
    def hex2rgb(hex_code):
        hex_code = hex_code.lstrip('#')
        return tuple(int(hex_code[i:i+2],16) for i in (0,2,4))
    
    @staticmethod
    def rgb2hex(rgb_tuple):
        def clamp(x):
            return max(0, min(x, 255))
        rgb_tuple = tuple(int(clamp(i)*255) for i in rgb_tuple)
        r,g,b = rgb_tuple
        return f'#{r:02x}{g:02x}{b:02x}'
    
    @staticmethod
    def normalize_rgb(rgb_tuple):
        return tuple(i/255. for i in rgb_tuple)
    
    @staticmethod
    def rgb2hsv(rgb_tuple):
        rgb_tuple = Color.normalize_rgb(rgb_tuple)
        r,g,b = rgb_tuple
        return colorsys.rgb_to_hsv(r,g,b)
    
    @staticmethod
    def hsv2rgb(hsv_tuple):
        h,s,v = hsv_tuple
        return colorsys.hsv_to_rgb(h,s,v)
    
    @staticmethod
    def make_color_range(start_color:str,rng:int,s_limit:list=[20,100],v_limit:list=[20,100]) -> list:
        """ Returns a list of hex colors ranging from start color to specific limit values"""
        rgb = Color.hex2rgb(start_color)
        hsv = Color.rgb2hsv(rgb)
        h,s,v = hsv
        
        # limit the saturation and value
        # s= 20%-100% v=100%
        
        s_steps = np.linspace(s,1,rng)
        v_steps = np.linspace(v,1,rng)
        v_steps = v_steps[::-1] # reverse values to go from light to darker color
        
        color_range = []
        for i,s in enumerate(s_steps):
            
            new_hsv = (h,s_steps[i],v_steps[i])
            new_rgb = Color.hsv2rgb(new_hsv)
            color_range.append(Color.rgb2hex(new_rgb))
        
        return color_range