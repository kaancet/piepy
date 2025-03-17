import numpy as np
import polars as pl
from typing import Literal
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import wilcoxon,linregress,pearsonr # noqa: F401
from scipy.optimize import curve_fit # noqa: F401

from ....core.data_functions import make_subsets
from ...plotting_utils import (
    set_style, 
    override_plots)
from ...colors.color import Color
from ....psychophysics.wheel.detection.wheelDetectionGroupedAggregator import WheelDetectionGroupedAggregator


ANIMAL_COLORS = {
        "KC139" : "#332288",
        "KC141" : "#117733",
        "KC142" : "#DDCC77",
        "KC143" : "#AA4499",
        "KC144" : "#882255",
        "KC145" : "#88CCEE",
        "KC146" : "#275D6D",
        "KC147" : "#F57A6C",
        "KC148" : "#ADFA9A",
        "KC149" : "#A45414",
        "KC150" : "#0000FF",
        "KC151" : "#00FF11",
        "KC152" : "#FFAA33"
    }


X_AXIS_LABEL= {-1 : "Non\nOpto", 
                0 : "Opto", 
                1 : "Opto\nOff"
                }

def m_line(x, m, b):
    return m * x +b

def plot_contrast_effect_scatter_plot(
    data:pl.DataFrame,
    effect_on:Literal["hit_rate","reaction","response"],
    ax:plt.Axes|None=None,
    include_misses:bool=False,
    polarity:Literal[-1,1]=1,
    mpl_kwargs:dict|None=None,
    **kwargs
) -> tuple[plt.Figure, plt.Axes]:
    """ Plots a correlation scatter plot of the given "effect_on" readouts of two contrast values,
    seperates the stimulus types

    Args:
        data (pl.DataFrame): Experimental data 
        effect_on (Literal["hit_rate","reaction","response"]): The behavioral readout to plot. 
        ax (plt.Axes | None, optional): Precreated axis to plot to. Defaults to None.
        include_misses (bool, optional): Whether or not to include miss trials. Defaults to False. Doesn't make sense if effect_on is "hit_rate".
        polarity (Literal[-1,1], optional): Polarity of subtraction. 1 is nonopto - opto, -1 is opto - nonopto. Defaults to 1.
        mpl_kwargs (dict | None, optional): kwargs for styling matplotlib plots. Defaults to None.

    Returns:
        tuple[plt.Figure, plt.Axes]: Figure and axes of plot
    """
    if effect_on != "hit_rate":
        if not include_misses:
            effect_on = f"hit_{effect_on}"
        effect_on = f"median_{effect_on}_times"
    polarity = float(polarity)
    clr = Color(task="detection")
    override_plots()
    set_style(kwargs.get("style", "presentation"))
    if mpl_kwargs is None:
        mpl_kwargs = {}
    
    if ax is None:
        fig = plt.figure(figsize=mpl_kwargs.pop("figsize", (8, 8)))
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = ax.get_figure()
        
    aggregator = WheelDetectionGroupedAggregator()
    aggregator.set_data(data=data)
    aggregator.group_data(group_by = ["animalid", "session_id", "stim_type", "stim_side", "contrast","opto_pattern"])
    aggregator.calculate_hit_rates()
    aggregator.calculate_opto_pvalues()
    
    plot_data = aggregator.grouped_data.drop_nulls("contrast").filter(pl.col("stim_side")!="ipsi")
    diffs_values = defaultdict(list)
    
    for filt_tup in make_subsets(plot_data,["session_id"],start_enumerate=1):
        filt_df = filt_tup[-1]
        if not filt_df.is_empty():
            q = (filt_df
                .group_by(["stim_type","opto_pattern"])
                .agg(
                    pl.col("contrast"),
                    pl.col(effect_on)
                )
                .sort(["stim_type","opto_pattern"]))    
            
            for stim_tup in make_subsets(q,["stim_type"]):
                stim_df = stim_tup[-1]
                skey = stim_tup[0]    
                c_nonopto = stim_df[0,"contrast"].to_numpy()
                c_opto = stim_df[1,"contrast"].to_numpy()
                paired_idx = [i for i,c in enumerate(c_nonopto) if c in c_opto]
                
                e1 = stim_df[0,effect_on].to_numpy()
                e2 = stim_df[1,effect_on].to_numpy()
                
                diffs = e1[paired_idx] - e2
                diffs = polarity * diffs
                
                diffs_values[skey].append(diffs.tolist())
        else:
            print('soknfdksndf')
            
    for i,k in enumerate(diffs_values.keys()):
        v = diffs_values[k]
        v = np.array(v)
        if effect_on == "hit_rate":
            v *= 100
            
        ax.scatter(v[:,0],
                   v[:,1],
                   c=clr.stim_keys[f"{k}_-1"]["color"])

        x = v[:,0]
        y = v[:,1]
        x_nan = np.isnan(x)
        y_nan = np.isnan(y)
        
        y = y[~np.logical_or(x_nan,y_nan)]
        x = x[~np.logical_or(x_nan,y_nan)]
        
        res = pearsonr(x,y)        
        # x = x[:,np.newaxis]
        # m, _, _, _ = np.linalg.lstsq(x, y)
        # y_pred = m * x
        
        # Compute R²
        # SS_res = np.sum((y - y_pred) ** 2)  # Residual sum of squares
        # SS_tot = np.sum((y - np.mean(y)) ** 2)  # Total sum of squares
        # r2 = 1 - (SS_res / SS_tot)
        
        # ax.plot(np.arange(0,600,100), m_line(np.arange(0,600,100), *popt))
        ax.text(0,100*i,f"p_r={res.statistic:.4f}",c=clr.stim_keys[f"{k}_-1"]["color"])
    
    ax.plot([0,500],[0,500],linestyle="--",color="k",alpha=0.5)
    if "time" in effect_on:
        _y_ = np.arange(-400,800,200)
        y_ticks = [_t*-1*polarity for _t in _y_]
    else:
        _y_ = [0,25,50,75,100]
        y_ticks = [_t*polarity for _t in _y_]
        
    ax.set_ylim([min(y_ticks)-10, max(y_ticks)+10])
    ax.set_xlim([min(y_ticks)-10, max(y_ticks)+10])
    ax.set_yticks(y_ticks)
    ax.set_xticks(y_ticks)
    ax.set_ylabel(r"$\Delta$ Hit Rate 50 (%)")
    ax.set_xlabel(r"$\Delta$ Hit Rate 12.5 (%)")
    
    return fig,ax