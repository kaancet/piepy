import itertools
import numpy as np
import polars as pl
from typing import Literal
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker # noqa: F401
from collections import defaultdict
from scipy import stats

from ....core.data_functions import make_subsets
from ....core.statistics import bootstrap_confidence_interval
from ...plotting_utils import (
    set_style, 
    make_linear_axis, 
    make_dot_cloud,
    override_plots, 
    pval_plotter)
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

# values from allen ccf_2022, using the shoelace algorithm to calculate the area from area boundary coordinates
AREA_SIZE = {
    "V1": 4.002,
    "HVA": 2.925,
    "dorsal": 1.428,
    "ventralPM": 1.496,
    "LM": 0.571,
    "AL": 0.389,
    "RL": 0.583,
    "PM": 0.719,
    "AM": 0.456,
}
# "LI":0.207
# "cortex" : 6.927,

# hierarchy scores from steinmetz?
AREA_SCORE = {}


X_AXIS_LABEL= {-1 : "Non\nOpto", 
                0 : "Opto", 
                1 : "Opto\nOff"
                }

#TODO: axis positioning is a bit hardcoded
def plot_delta_effect_contrast(
    data:pl.DataFrame,
    effect_on:Literal["hit_rate","reaction","response"],
    plot_with:Literal["sem","conf","iqr"]="sem",
    ax:plt.Axes|None=None,
    include_misses:bool=False,
    polarity:Literal[-1,1]=1,
    mpl_kwargs:dict|None=None,
    **kwargs
          ) -> tuple[plt.Figure, plt.Axes]:
    """ Plots a connected scatter plot of the delta values between opto and non-opto conditions for different contrasts.
    Each connected dot-pair corresponds to a session and the delta in different contrast conditions
    
    Args:
        data (pl.DataFrame): Experimental data 
        effect_on (Literal["hit_rate","reaction","response"]): The behavioral readout to plot.
        plot_with (Literal["sem","conf","iqr"], optional): The distibution measurement. Defaults to "sem"
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
    lin_axis_dict = make_linear_axis(plot_data,"signed_contrast")
    _lin_axis = [float(lin_axis_dict[c]) if c is not None else None for c in plot_data["contrast"].to_list()]
    plot_data = plot_data.with_columns(pl.Series("linear_axis",_lin_axis))
    
    diff_values = defaultdict(list)
    for stim_tup in make_subsets(plot_data,["stim_type"],start_enumerate=-1):
        pos_ofs = stim_tup[0]
        stim_type = stim_tup[1]
        stim_df = stim_tup[-1]
        diff_values = defaultdict(list)
        x_pos_values = []
        for filt_tup in make_subsets(stim_df,["session_id"]):
            filt_df = filt_tup[-1]
            if not filt_df.is_empty():

                q = (filt_df
                    .group_by(["opto_pattern","contrast"])
                    .agg(
                        pl.col(effect_on).get(0)
                    )
                    .sort(["opto_pattern","contrast"]))
                
                vals_nonopto = q.filter(pl.col("opto_pattern")==-1)[effect_on].to_numpy()
                contrast_nonopto = q.filter(pl.col("opto_pattern")==-1)["contrast"].to_list()
            
                opto_patterns = q["opto_pattern"].sort().unique().to_numpy()
                opto_patterns = [o for o in opto_patterns if o!=-1]
                for o in opto_patterns:
                    vals_opto = q.filter(pl.col("opto_pattern")==o)[effect_on].to_numpy()
                    contrast_opto = q.filter(pl.col("opto_pattern")==0)["contrast"].to_list()

                    paired_idx = [i for i,c in enumerate(contrast_nonopto) if c in contrast_opto]
                    vals_nonopto = vals_nonopto[paired_idx]
                    
                    # delta hit rate
                    vals_diff = vals_nonopto[:len(vals_opto)] - vals_opto   
                    vals_diff = polarity * vals_diff
                    
                    diff_values[o].append(vals_diff.tolist())
                    x_pos_values.append(paired_idx)
                    
        for k,v in diff_values.items():
            data_mat = np.array(v) 
            data_mat_x = np.zeros_like(data_mat)
            for j in range(data_mat.shape[1]):
                cloud_pos = x_pos_values[0][j] + pos_ofs*kwargs.get("cloud_offset",0.25)
                x_points = make_dot_cloud(data_mat[:,j],
                                            cloud_pos,
                                            bin_width=kwargs.get("bin_width",0.05),
                                            width=kwargs.get("cloud_width",0.2))
                
                # add a very little random jitter 
                jit = np.random.randn(len(x_points))/25
                x_points += jit
                data_mat_x[:,j] = x_points
                y_vals = data_mat[:,j]
                if effect_on == "hit_rate":
                    y_vals = 100*y_vals
                ax._scatter(x_points,
                            y_vals,
                            c=clr.stim_keys[f"{stim_df[0,'stim_type']}_{int(o)}"]["color"],
                            zorder=1,
                            mpl_kwargs=mpl_kwargs)
                
                res = stats.wilcoxon(data_mat[:,j])
                p = res.pvalue
                print(stim_type,res,flush=True)
                ax = pval_plotter(ax,p,[cloud_pos,cloud_pos],
                                loc=polarity*100,
                                color=clr.stim_keys[f"{stim_df[0,'stim_type']}_{int(o)}"]["color"])
            
            if plot_with == "conf":         
                means,ci_plus,ci_neg = np.apply_along_axis(bootstrap_confidence_interval,axis=0,arr=data_mat,statistic=np.nanmean)
            elif plot_with == "sem":
                means = np.nanmean(data_mat,axis=0)
                ci_plus = stats.sem(data_mat,axis=0,nan_policy="omit")
                ci_neg = stats.sem(data_mat,axis=0,nan_policy="omit")
            elif plot_with =="iqr":
                means = np.nanmean(data_mat,axis=0)
                ci_plus = stats.iqr(data_mat,axis=0,nan_policy="omit")
                ci_neg = stats.iqr(data_mat,axis=0,nan_policy="omit")
            
            if effect_on == "hit_rate":
                means *= 100
                ci_plus *= 100
                ci_neg *= 100
            
            # shaded 95% CI
            ax.errorbar(
                np.unique(_lin_axis),
                means,
                yerr=(ci_plus,
                      ci_neg),
                color=clr.stim_keys[f"{stim_df[0,'stim_type']}_{int(o)}"]["color"],
                alpha=0.3,
                zorder=2,
                linewidth=0,
                elinewidth=10,
                markersize=0
            )
            
            # means
            ax.scatter(
                np.unique(_lin_axis),
                means,
                s = (plt.rcParams["lines.markersize"] ** 2),
                c = clr.stim_keys[f"{stim_df[0,'stim_type']}_{int(o)}"]["color"],
                marker="_",
                linewidths = 3,
                edgecolors = "w",
                zorder=3
            )
            
            # plotting connecting lines  
            for i in range(data_mat_x.shape[0]):
                y_vals = data_mat[i,:]
                if effect_on == "hit_rate":
                    y_vals = 100*y_vals
                ax.plot(data_mat_x[i,:],
                        y_vals,
                        linewidth=2,
                        alpha=0.2,
                        zorder=1,
                        c=clr.stim_keys[f"{stim_df[0,'stim_type']}_{int(o)}"]["color"])

    _x_axis = np.array([0, 12.5, 50])
    _x_axis_pos = np.mean(np.array(x_pos_values),axis=0,dtype=int)
    ax.set_xticks(_x_axis_pos)
    ax.set_xticklabels(_x_axis[_x_axis_pos])

    if "time" in effect_on:
        # ax.set_yscale("symlog",linthresh=500)
        # ax.set_yscale('asinh',linear_width=200, base=10)
        # major_ticks = [0]
        # ax.set_yticks(major_ticks)
        
        # minor_locator = plt.LogLocator(base=10.0, subs=np.arange(200, 1000,200), numticks=5)
        # ax.yaxis.set_minor_locator(minor_locator)
        
        # negative_minor_ticks = [-t for t in minor_locator.tick_values(10, 1000) if t > 0]  # Mirror positive minor ticks
        # all_minor_ticks = sorted(negative_minor_ticks + minor_locator.tick_values(10, 1000).tolist())  # Combine
        # ax.set_yticks(all_minor_ticks, minor=True)  # Apply minor ticks symmetrically

        # ax.yaxis.set_minor_formatter(plt.FormatStrFormatter("%d"))
        # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
        # ax.set_ylim([-400,600])
        _y_ = [-600, -400, -200, 0, 200, 400, 600]
        y_ticks = [_t*polarity for _t in _y_]
        ax.set_yticks(y_ticks)
        ax.set_ylim([min(y_ticks)-10, max(y_ticks)+10])
        
    else:
        _y_ = [0,25,50,75,100]
        y_ticks = [_t*polarity for _t in _y_]
        ax.set_yticks(y_ticks)
        ax.set_ylim([min(y_ticks)-10, max(y_ticks)+10])
    ax.set_xlabel("Stimulus contrast (%)")
    ylab = " ".join(effect_on.split("_")).capitalize()
    ax.set_ylabel(fr"$\Delta${ylab} (%)")
    return fig,ax


#TODO: axis positioning is a bit hardcoded
def plot_delta_effect_stimtype(
    data:pl.DataFrame,
    effect_on:Literal["hit_rate","reaction","response"],
    plot_with:Literal["sem","conf","iqr"]="sem",
    ax:plt.Axes|None=None,
    include_misses:bool=False,
    polarity:Literal[-1,1]=1,
    mpl_kwargs:dict|None=None,
    **kwargs) -> tuple[plt.Figure, plt.Axes]:
    """ Plots a connected scatter plot of the delta values between opto and non-opto conditions for different stimulus types.
    Each connected dot-pair corresponds to a session and the delta in different stimulus presentation conditions

    Args:
        data (pl.DataFrame): Experimental data 
        effect_on (Literal["hit_rate","reaction","response"]): The behavioral readout to plot. 
        plot_with (Literal["sem","conf","iqr"], optional): The distibution measurement. Defaults to "sem"
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
        bin_width = kwargs.get("bin_width",50)
    else:
        bin_width = kwargs.get("bin_width",0.05)

    clr = Color(task="detection")
    override_plots()
    set_style(kwargs.get("style", "presentation"))
    if mpl_kwargs is None:
        mpl_kwargs = {}
    polarity = float(polarity)
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
    plot_data = plot_data.filter(pl.col("contrast")==0.5)
    diff_values = defaultdict(list)
    x_pos_values = []
    stim_types = []
    for filt_tup in make_subsets(plot_data,["session_id"]):
        filt_df = filt_tup[-1]
        if not filt_df.is_empty():
            q = (filt_df
                .group_by(["opto_pattern","stim_type"])
                .agg(
                    pl.col(effect_on).get(0)
                )
                .sort(["opto_pattern","stim_type"]))
            
            vals_nonopto = q.filter(pl.col("opto_pattern")==-1)[effect_on].to_numpy()
            stype_nonopto = q.filter(pl.col("opto_pattern")==-1)["stim_type"].to_list()
        
            opto_patterns = q["opto_pattern"].sort().unique().to_numpy()
            opto_patterns = [o for o in opto_patterns if o!=-1]
            for o in opto_patterns:
                vals_opto = q.filter(pl.col("opto_pattern")==o)[effect_on].to_numpy()
                stype_opto = q.filter(pl.col("opto_pattern")==0)["stim_type"].to_list()

                paired_idx = [i for i,c in enumerate(stype_nonopto) if c in stype_opto]
                vals_nonopto = vals_nonopto[paired_idx]
                
                vals_diff = vals_nonopto[:len(vals_opto)] - vals_opto
                vals_diff = polarity * vals_diff
                
                diff_values[o].append(vals_diff.tolist())
                x_pos_values.append(paired_idx)
                stim_types.append(stype_opto)
    
    for k,v in diff_values.items():
        data_mat = np.array(v) 
        data_mat_x = np.zeros_like(data_mat)
        for j in range(data_mat.shape[1]):
            _st = [ss[j] for ss in stim_types]
            cloud_pos = x_pos_values[0][j]
            colors = [clr.stim_keys[f"{s}_{int(o)}"]["color"] for s in _st]
            
            x_points = make_dot_cloud(data_mat[:,j],
                                      cloud_pos,
                                      bin_width=bin_width,
                                      width=kwargs.get("cloud_width",0.05))
            data_mat_x[:,j] = x_points
            y_vals = data_mat[:,j]
            if effect_on == "hit_rate":
                y_vals = 100*y_vals
            ax.scatter(x_points,
                        y_vals,
                        zorder=2,
                        c=colors)
            
            for p1,p2 in list(itertools.combinations([x for x in range(data_mat.shape[1])], 2)):
                d1 = data_mat[:,p1]
                d2 = data_mat[:,p2]
                res = stats.wilcoxon(d1,d2)
                p = res.pvalue
                print(p)
                ax = pval_plotter(ax,p,[p1,p2],
                                loc=polarity*100)

        # plot the means and shaded regions
        if plot_with == "conf":         
            means,ci_plus,ci_neg = np.apply_along_axis(bootstrap_confidence_interval,axis=0,arr=diff_values,statistic=np.nanmean)
        elif plot_with == "sem":
            means = np.nanmean(diff_values,axis=0)
            ci_plus = stats.sem(diff_values,axis=0,nan_policy="omit")
            ci_neg = stats.sem(diff_values,axis=0,nan_policy="omit")
        elif plot_with =="iqr":
            means = np.nanmean(diff_values,axis=0)
            ci_plus = stats.iqr(diff_values,axis=0,nan_policy="omit")
            ci_neg = stats.iqr(diff_values,axis=0,nan_policy="omit")
        
        if effect_on == "hit_rate":
            means *= 100
            ci_plus *= 100
            ci_neg *= 100
        
        # shaded 95% CI
        ax.errorbar(
            np.unique(x_pos_values),
            means,
            yerr=(ci_plus,
                    ci_neg),
            # color=clr.stim_keys[f"{stim_df[0,'stim_type']}_{int(o)}"]["color"],
            alpha=0.3,
            zorder=2,
            linewidth=0,
            elinewidth=10,
            markersize=0
        )
            
        # # means
        ax.scatter(
            np.unique(x_pos_values),
            means,
            s = (plt.rcParams["lines.markersize"] ** 2),
            # c = clr.stim_keys[f"{stim_df[0,'stim_type']}_{int(o)}"]["color"],
            marker="_",
            linewidths = 3,
            edgecolors = "w",
            zorder=3
        )
        
        for i in range(data_mat_x.shape[0]):
            y_vals = data_mat[i,:]
            if effect_on == "hit_rate":
                y_vals = 100*y_vals
            ax.plot(data_mat_x[i,:],
                    y_vals,
                    linewidth=2,
                    alpha=0.2,
                    c="#bfbfbf",
                    zorder=0)
            
    _x_axis = np.array([0, 12.5, 50])
    _x_axis_pos = np.mean(np.array(x_pos_values),axis=0,dtype=int)
    ax.set_xticks(_x_axis_pos)
    ax.set_xticklabels(_x_axis[_x_axis_pos])

    if "time" in effect_on:
        _y_ = np.arange(-200,800,200)
        y_ticks = [_t*-1*polarity for _t in _y_]
    else:
        _y_ = [0,25,50,75,100]
        y_ticks = [_t*polarity for _t in _y_]
    
    ax.set_yticks(y_ticks)
    ax.set_ylim([min(y_ticks)-10, max(y_ticks)+10])
    ax.set_xlabel("Stimulus contrast (%)")
    ylab = " ".join(effect_on.split("_")).capitalize()
    ax.set_ylabel(fr"$\Delta${ylab} (%)")
    
    return fig,ax


def plot_delta_effect_areas(
    data:pl.DataFrame,
    effect_on:Literal["hit_rate","reaction","response"],
    contrast:float,
    areas:list[str]=["V1","HVA","dorsal","ventralPM","LM","AL","RL","PM","AM"],
    plot_with:Literal["sem","conf","iqr"]="sem",
    ax:plt.Axes|None=None,
    include_misses:bool=False,
    polarity:Literal[-1,1]=1,
    mpl_kwargs:dict|None=None,
    **kwargs) -> tuple[plt.Figure, plt.Axes]:
    """ Plots a connected scatter plot of the delta values between opto and non-opto conditions for different areas.
    Each connected dot-sequence corresponds to an animal (one session)

    Args:
        data (pl.DataFrame): Experimental data 
        effect_on (Literal["hit_rate","reaction","response"]): The behavioral readout to plot.
        contrast (float): contrast value to plot
        areas (list[str]): list of areas to be plotted(in that order)
        plot_with (Literal["sem","conf","iqr"], optional): The distibution measurement. Defaults to "sem"
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
        bin_width = kwargs.get("bin_width",50)
    else:
        bin_width = kwargs.get("bin_width",0.05)
        
    override_plots()
    set_style(kwargs.get("style", "presentation"))
    if mpl_kwargs is None:
        mpl_kwargs = {}
    polarity = float(polarity)
    if ax is None:
        fig = plt.figure(figsize=mpl_kwargs.pop("figsize", (8, 8)))
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = ax.get_figure()
        
    aggregator = WheelDetectionGroupedAggregator()
    aggregator.set_data(data=data)
    aggregator.group_data(group_by = ["animalid", "area", "stim_type", "stim_side", "contrast", "opto_pattern"])
    aggregator.calculate_hit_rates()
    aggregator.calculate_opto_pvalues()
    
    plot_data = aggregator.grouped_data.drop_nulls("contrast").filter(pl.col("stim_side")!="ipsi")
    plot_data = plot_data.filter(pl.col("contrast")==contrast)
    plot_data = plot_data.filter(pl.col("area").is_in(areas))
    
    diff_values = np.zeros((0,len(areas)))
    diff_values[:] = np.nan
    animal_ids = []
    for filt_tup in make_subsets(plot_data,["animalid"]):
        filt_df = filt_tup[-1]
        a_id = filt_tup[0]
        if not filt_df.is_empty():
            # order the areas in given order
            with pl.StringCache():
                pl.Series(areas).cast(pl.Categorical)
                filt_df = filt_df.with_columns(pl.col("area").cast(pl.Categorical))
                filt_df = filt_df.sort("area")
            
            q = (filt_df
                .group_by(["opto_pattern","area"])
                .agg(
                    pl.col(effect_on).get(0)
                )
                .sort(["opto_pattern","area"]))
            
            vals_nonopto = np.zeros((len(areas)))
            vals_opto = np.zeros_like(vals_nonopto)
            vals_nonopto[:] = np.nan
            vals_opto[:] = np.nan
            # loop for areas because there might be missing sessions for some animals 
            for j,a in enumerate(areas):
                _area = q.filter(pl.col("area")==a)
                if not _area.is_empty():
                    vals_nonopto[j] = _area.filter((pl.col("opto_pattern")==-1))[0,effect_on]
                    vals_opto[j] = _area.filter(pl.col("opto_pattern")==0)[0,effect_on]

            diff_vals = polarity * (vals_nonopto - vals_opto)
            
            diff_values = np.vstack((diff_values,diff_vals))
            animal_ids.append(a_id)
    x_axis = np.arange(len(areas))
    
    # plot the scatters
    clouded_x_values = np.zeros_like(diff_values)
    clouded_x_values[:] = np.nan
    for c in range(diff_values.shape[1]):
        area_vals = diff_values[:,c]
        x_points = make_dot_cloud(area_vals,
                                  x_axis[c],
                                  bin_width=bin_width,
                                  width=kwargs.get("cloud_width",0.05))
        clouded_x_values[:,c] = x_points
        
        if effect_on == "hit_rate":
            area_vals = 100*area_vals
        
        ax.scatter(x_points,
                   area_vals,
                   zorder=1,
                   c="#C7C7C7")
    
    # plot the connected lines
    for i,r in enumerate(diff_values):
        x_vals = clouded_x_values[i,:]
        
        if effect_on == "hit_rate":
            r = 100*r
        
        ax._plot(x_vals,
                r,
                zorder=1,
                markersize=0,
                c=kwargs.get("color", ANIMAL_COLORS[animal_ids[i]]),
                mpl_kwargs=mpl_kwargs)
    
    # plot the means and shaded regions
    if plot_with == "conf":         
        means,ci_plus,ci_neg = np.apply_along_axis(bootstrap_confidence_interval,axis=0,arr=diff_values,statistic=np.nanmean)
    elif plot_with == "sem":
        means = np.nanmean(diff_values,axis=0)
        ci_plus = stats.sem(diff_values,axis=0,nan_policy="omit")
        ci_neg = stats.sem(diff_values,axis=0,nan_policy="omit")
    elif plot_with =="iqr":
        means = np.nanmean(diff_values,axis=0)
        ci_plus = stats.iqr(diff_values,axis=0,nan_policy="omit")
        ci_neg = stats.iqr(diff_values,axis=0,nan_policy="omit")
    
    if effect_on == "hit_rate":
        means *= 100
        ci_plus *= 100
        ci_neg *= 100
        
    ax.scatter(
        x_axis,
        means,
        s = (plt.rcParams["lines.markersize"] ** 2),
        c = "k",
        marker="_",
        linewidths = 3,
        edgecolors = "w",
        zorder=3
    )
    
    # shaded 95% CI
    ax.errorbar(
        x_axis,
        means,
        yerr=(ci_plus,
                ci_neg),
        color="k",
        alpha=0.3,
        zorder=2,
        linewidth=0,
        elinewidth=10,
        markersize=0
    )
    
    ax.set_xticks(x_axis)
    ax.set_xticklabels(areas)

    if "time" in effect_on:
        # ax.set_yscale("symlog",linthresh=500)
        # ax.set_yscale('asinh',linear_width=200, base=10)
        # major_ticks = [0]
        # ax.set_yticks(major_ticks)
        
        # minor_locator = plt.LogLocator(base=10.0, subs=np.arange(200, 1000,200), numticks=5)
        # ax.yaxis.set_minor_locator(minor_locator)
        
        # negative_minor_ticks = [-t for t in minor_locator.tick_values(10, 1000) if t > 0]  # Mirror positive minor ticks
        # all_minor_ticks = sorted(negative_minor_ticks + minor_locator.tick_values(10, 1000).tolist())  # Combine
        # ax.set_yticks(all_minor_ticks, minor=True)  # Apply minor ticks symmetrically

        # ax.yaxis.set_minor_formatter(plt.FormatStrFormatter("%d"))
        # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
        # ax.set_ylim([-400,600])
        _y_ = [-600, -400, -200, 0, 200, 400, 600]
        y_ticks = [_t*polarity for _t in _y_]
        ax.set_yticks(y_ticks)
        ax.set_ylim([min(y_ticks)-10, max(y_ticks)+10])
        
    else:
        _y_ = [0,25,50,75,100]
        y_ticks = [_t*polarity for _t in _y_]
        ax.set_yticks(y_ticks)
        ax.set_ylim([min(y_ticks)-10, max(y_ticks)+10])
    ax.set_xlabel("Stimulus contrast (%)")
    ylab = " ".join(effect_on.split("_")).capitalize()
    ax.set_ylabel(fr"$\Delta${ylab} (%)")
    return fig, ax


# def plot_delta_effect_CNO(
#     data:pl.DataFrame,
#     effect_on:Literal["hit_rate","reaction","response"],
#     contrast:float,
#     stim_type:str,
#     plot_with:Literal["sem","conf","iqr"]="sem",
#     ax:plt.Axes|None=None,
#     include_misses:bool=False,
#     polarity:Literal[-1,1]=1,
#     mpl_kwargs:dict|None=None,
#     **kwargs) -> tuple[plt.Figure, plt.Axes]:
#     """ Plots a connected scatter plot of the delta values between non_CNO and CNO conditions.
#     Each connected dot-pair corresponds to two sessions of a given animal, without and with CNO, respectively
    
#     Args:
#         data (pl.DataFrame): Experimental data 
#         effect_on (Literal["hit_rate","reaction","response"]): The behavioral readout to plot.
#         contrast (float): contrast value to plot
#         stim_type (str): stimulus type to plot, e.g. 0.04cpd_8.0Hz
#         plot_with (Literal["sem","conf","iqr"], optional): The distibution measurement. Defaults to "sem"
#         ax (plt.Axes | None, optional): Precreated axis to plot to. Defaults to None.
#         include_misses (bool, optional): Whether or not to include miss trials. Defaults to False. Doesn't make sense if effect_on is "hit_rate".
#         polarity (Literal[-1,1], optional): Polarity of subtraction. 1 is nonopto - opto, -1 is opto - nonopto. Defaults to 1.
#         mpl_kwargs (dict | None, optional): kwargs for styling matplotlib plots. Defaults to None.

#     Returns:
#         tuple[plt.Figure, plt.Axes]: Figure and axes of plot
#     """
#     if effect_on != "hit_rate":
#         if not include_misses:
#             effect_on = f"hit_{effect_on}"
#         effect_on = f"median_{effect_on}_times"
#         bin_width = kwargs.get("bin_width",50)
#     else:
#         bin_width = kwargs.get("bin_width",0.05)
        
#     override_plots()
#     set_style(kwargs.get("style", "presentation"))
#     if mpl_kwargs is None:
#         mpl_kwargs = {}
#     polarity = float(polarity)
#     if ax is None:
#         fig = plt.figure(figsize=mpl_kwargs.pop("figsize", (8, 8)))
#         ax = fig.add_subplot(1, 1, 1)
#     else:
#         fig = ax.get_figure()
        
#     aggregator = WheelDetectionGroupedAggregator()
#     aggregator.set_data(data=data)
#     aggregator.group_data(group_by = ["animalid", "area", "stim_type", "stim_side", "contrast", "opto_pattern"])
#     aggregator.calculate_hit_rates()
#     aggregator.calculate_opto_pvalues()
    
#     plot_data = aggregator.grouped_data.drop_nulls("contrast").filter(pl.col("stim_side")!="ipsi")
#     plot_data = plot_data.filter((pl.col("contrast")==contrast) & 
#                                  (pl.col("stim_type")==stim_type))
    
#     diff_values = np.zeros((0,len(areas)))
#     diff_values[:] = np.nan
#     animal_ids = []
#     for filt_tup in make_subsets(plot_data,["animalid"]):
    
    