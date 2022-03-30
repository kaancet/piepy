import os
import re
import numpy as np
import pandas as pd
from os.path import join as pjoin
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib import colors as mcolors
import matplotlib.gridspec as gridspec
from cycler import cycler
from collections import defaultdict
from behavior_python.utils import *
from ..wheelUtils import *
from ..wheelAnalysis import WheelAnalysis

DUNDER = re.compile(r'^__[^\d\W]\w*__\Z', re.UNICODE)
is_special = DUNDER.match

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

stim_styles = {'grating':{'color':'tab:gray'},
               'lowSF_highTF':{'color':'tab:orange'},
               'highSF_lowTF':{'color':'tab:purple'},
               'lowSF_highTF_opto_0':{'color':'tab:green'},
               'lowSF_highTF_opto_1':{'color':'tab:red'},
               'highSF_lowTF_opto_0':{'color':'tab:cyan'},
               'highSF_lowTF_opto_1':{'color':'tab:pink'},
               'lowSF_highTF_opto_-1':{'color':'tab:green'}}


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


class WheelBasePlotter:
    def __init__(self):
        pass

    def save(self,plotkey):
        pass

    def empty_axes(self,ax):
        """ Erases all axes and related ticks and labels"""
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(bottom="off", left="off")
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        return ax

