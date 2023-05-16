import os
import json
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
    __slots__ = ['colorkey_path','stim_keys','contrast_keys']
    def __init__(self):
         # this is hardcoded for now, fix this
        cfg = getConfig()
        self.colorkey_path = cfg['colorsPath']
        self.read_colors()
        
    
    def read_colors(self) -> None:
        """ Reads the colorkey.json and returns a dict of color keys for different sftf and contrast values"""
        with open(self.colorkey_path,'r') as f:
            keys = json.load(f)
            self.stim_keys = keys['spatiotemporal']
            self.contrast_keys = keys['contrast']
            
    def check_stim_colors(self,keys):
        """ Checks if the stim key has a corresponding color value, if not adds a randomly selected color the key"""
        new_colors = {}
        for k in keys:
            if k not in self.stim_keys:
                print(f'Stim key {k} not present in colors, generating random color...')
                colors = {**mcolors.BASE_COLORS, **mcolors.CSS4_COLORS}
                new_colors[k] = {'color':np.random.choice(list(colors.keys()),replace=False,size=1)[0]}
        if len(new_colors):
            self.stim_keys = {**self.stim_keys, **new_colors}
        else:
            print('Stimulus colors checkout!!')
            
    def check_contrast_colors(self,contrasts):
        """ Checks if the contrast key has a corresponding color value, if not adds a randomly selected color the key"""
        new_colors = {}
        for c in contrasts:
            str_key = str(c)
            if str_key not in self.contrast_keys:
                print(f'Contrast key {str_key} not present in colors, generating random color...')
                colors = {**mcolors.BASE_COLORS, **mcolors.CSS4_COLORS}
                new_colors[str_key] = {'color':np.random.choice(list(colors.keys()),replace=False,size=1)[0]}
        if len(new_colors):
            self.contrast_keys = {**self.contrast_keys, **new_colors}
        else:
            print('Contrast colors checkout!!')
                
    @staticmethod
    def name2hsv(color_name:str)->tuple:
        rgb = mcolors.to_rgb(color_name)
        return Color.rgb2hsv(rgb,normalize=False) # no need to normalize here, already 0-1 range from mcolors method
    
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
    def rgb2hsv(rgb_tuple,normalize:bool=True):
        if normalize:
            rgb_tuple = Color.normalize_rgb(rgb_tuple)
        r,g,b = rgb_tuple
        return colorsys.rgb_to_hsv(r,g,b)
    
    @staticmethod
    def hsv2rgb(hsv_tuple):
        h,s,v = hsv_tuple
        return colorsys.hsv_to_rgb(h,s,v)
    
    @staticmethod
    def lighten(hsv_tuple,l_coeff:float=0.33):
        """Lightens the hsv_tuple by l_coeff percent, aka from S subtracts l_coeff percent of the S value"""
        if not l_coeff <=1 and l_coeff>=0:
            raise ValueError(f'The l_coeff value needs to be 0<=l_coeff<= 1, got {l_coeff} instead')
        h,s,v = hsv_tuple
        s_new = s - (s*l_coeff)
        return Color.rgb2hex(Color.hsv2rgb((h,s_new,v)))
    
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
    
    # def add_colorkey(self) -> None:
    #     """ Add a colorkey to the json file """
    #     pass
    
    # @staticmethod
    # def make_new_key(colorkey_dict) -> str:
    #     """ Returns a hex color code, putting it equidistant from other keys"""
    #     # Get the colors and make them hsv
    #     h_values = []
        
    #     for k,v in colorkey_dict.items():
    #         for key,color in v.items:
    #             rgb = Color.hex2rgb(mcolors.cnames[color['color']])
    #             h,_,_ = Color.rgb2hsv(rgb)
    #             h_values.append(h)
