import os
import re
import numpy as np
import pandas as pd
from os.path import join as pjoin

import bokeh.plotting as bok
from bokeh.models import ColumnDataSource, Whisker, TeeHead, HoverTool,SingleIntervalTicker, LogAxis, LinearAxis
import bokeh.palettes as bokcolor

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import matplotlib.gridspec as gridspec
from behavior_python.utils import *
from ....wheelUtils import *
from ...wheelAnalysis22 import WheelCurve

DUNDER = re.compile(r'^__[^\d\W]\w*__\Z', re.UNICODE)
is_special = DUNDER.match

area_palette = list(bokcolor.d3['Category20c'][20])
wheel_palette = list(bokcolor.brewer['Paired'][10])
answer_palette = list(bokcolor.brewer['RdYlGn'][11])
fraction_palette = list(bokcolor.brewer['Accent'][8])

colors = {}
colors['contrast'] = {0:wheel_palette[6:8],
                     0.125:wheel_palette[2:4],
                    0.25:wheel_palette[8:10],
                   0.5:wheel_palette[4:6],
                   1: wheel_palette[10:12]}

colors['answer'] = {-1 : answer_palette[9:],
                     0 : answer_palette[5:7],
                     1 : answer_palette[1:3]}

# color codes for different stimuli (areas)
area_colors = {'lowSF_highTF':area_palette[4],
               'lowSF_highTF_opto':area_palette[7],
               'highSF_lowTF':area_palette[8],
               'highSF_lowTF_opto':area_palette[11],
               'grating':area_palette[16],
               'grating_opto':area_palette[19]}


class WheelBasePlotter:
    def __init__(self):
        pass

    def save(self,plotkey):
        pass

    def pretty_axes(self,fig,fontsize=None):
        """ Makes simple pretty axes"""
        fig.xaxis.axis_line_width=2
        fig.yaxis.axis_line_width=2

        if fontsize is None:
          fontsize = 22
        fig.axis.axis_label_text_font_size = self.pt_font(fontsize)
        fig.axis.axis_label_text_font_style = 'normal'
        fig.axis.major_label_text_font_size = self.pt_font(fontsize)
        return fig

    def empty_axes(self,ax):
        """ Erases all axes and related ticks and labels"""
        fig.axis.visible = False
        return ax

    def pt_font(self,int_font):
      return '{0}pt'.format(int_font)


# def animalSummaries(animalids,*args,**kwargs):
#     """ """
#     if ~isinstance(animalids,list):
#         raise ValueError('You need to provide a list of animal names')
#     fig = plt.figure(figsize=kwargs.get('figsize',(20,20)))
#     nrows = 3
#     ncols = len(animalids)

#     behave_dict = {a_id:WheelBehavior(a_id,'200101',load_behave=True,load_data=True,criteria=criteria,autostart=True) for a_id in animalids}

#     for col,a_id in enumerate(animalids,1):
#         for rows in range(ncols):
#             ax_idx = row * ncols + col
#             ax = fig.add_subplot(nrows, ncols, ax_idx)

#             if row == 0:
#                 ax.set_title('{0}'.format(a_id),fontsize=fontsize,fontweight='bold')
#                 if col!=1:
#                     naked = True
#                     ax.spines['left'].set_visible(False)
#                     ax.set_yticklabels([])
#                 else:
#                     naked=False
#                 behave_dict[a_id].plot('performance', ax=ax, barebones=naked, savefig=False, notitle=True, *args,**kwargs)

#             if row == 1:
#                 if col!=1:
#                     naked = True
#                     ax.spines['left'].set_visible(False)
#                     ax.set_yticklabels([])
#                 else:
#                     naked=False
#                 behave_dict[a_id].plot('responsetimes', ax=ax, barebones=naked, savefig=False, notitle=True, *args,**kwargs)
#             if row == 2:
#                 if col!=1:
#                     naked = True
#                     ax.spines['left'].set_visible(False)
#                     ax.set_yticklabels([])
#                 else:
#                     naked=False
#                 behave_dict[a_id].plot('trialdistributions', ax=ax, barebones=naked, savefig=False, notitle=True, *args,**kwargs)
#             if row == 3:
#                 if col!=1:
#                     naked = True
#                     ax.spines['left'].set_visible(False)
#                 else:
#                     naked=False
#                 behave_dict[a_id].plot('performance', ax=ax, barebones=naked, savefig=False, notitle=True, *args,**kwargs)

#         # self.fig.suptitle('{0} Trianing Summary'.format(self.animalid),fontsize=fontsize+3,fontweight='bold')
#     plt.tight_layout()
#     return fig
