import numpy as np
import polars as pl
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def look_at_timing_differences(data:pl.DataFrame,
                               col1_name:str,
                               col2_name:str,
                               ax:plt.Axes=None,
                               only_hit:bool=False) -> plt.Axes:
    if ax is None:
        f = plt.figure(figsize=(8,8))
        ax = f.add_subplot(111)

    plot_data = data.with_columns(pl.when(pl.col('outcome')==1).then(pl.lit('#018f10'))
                                  .when(pl.col('outcome')==0).then(pl.lit('#b4b8b4'))
                                  .otherwise(pl.lit('#f22824')).alias('dot_color'))
    
    if only_hit:
        plot_data = plot_data.filter(pl.col('outcome')==1)
    
    
    plot_data = plot_data.drop_nulls(col1_name)
    time_on_y = plot_data[col2_name].to_numpy()
    time_on_x = plot_data[col1_name].to_numpy()
    trial_nos = plot_data['trial_no'].to_numpy()
    
    def m1_func(x,a):
        m = 1
        return m*x + a

    clrs = plot_data['dot_color'].to_list()
    s=ax.scatter(time_on_x,time_on_y,s=50,c=clrs,label='Data')


    nan_mask = ~np.isnan(time_on_x) & ~np.isnan(time_on_y)
    x = time_on_x[nan_mask]
    y = time_on_y[nan_mask]
    trial_nos = trial_nos[nan_mask]
    
    # fit curve with m=1
    popt,pcov = curve_fit(m1_func,x,y)
    # do linear regression
    res = stats.linregress(x,y)
    ax.plot(x, res.intercept + res.slope*time_on_x, 'k',label=f'R2={res.rvalue**2:.3f}\ny0={res.intercept:.2f}ms\nm={res.slope:.2f}')
    
    ax.plot(x,m1_func(x,*popt),color='r',linestyle=':',label='m=1 best fit')
    
    annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)
    
    def update_annot(ind):
        pos = s.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = f"Trial No: {trial_nos[ind['ind'][0]]}"
        annot.set_text(text)
        annot.get_bbox_patch().set_facecolor('white')
        annot.get_bbox_patch().set_alpha(0.4)
        
    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = s.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                f.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    f.canvas.draw_idle()
    
    fsize = 18
    ax.set_title(f"Timing Differnces",fontsize=14)
    ax.set_xlabel(f'{col1_name} (ms)',fontsize=fsize)
    ax.set_ylabel(f'{col2_name} (ms)',fontsize=fsize)
    
    # ax.set_xlim([-50,5010])
    # ax.set_ylim([-50,5010])
    
    ax.tick_params(labelsize=fsize,length=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.7)
    ax.legend(frameon=False,fontsize=fsize-2)
    
    return ax

def plot_all_timings(data:pl.DataFrame, only_hit:bool=False) -> None:
    
    timing_cols = ['pos_reaction_time','speed_reaction_time','rig_reaction_time']
    # if rig_reacttion_time in columns make 3 axes, else 2
    rig_react = data['rig_reaction_time'].drop_nulls()
    if len(rig_react) == 0:
        timing_cols = timing_cols[:2]

    f, axs = plt.subplots(nrows=1, ncols=len(timing_cols),sharey=True,figsize=(20,8))


    for i,c in enumerate(timing_cols):
        ax = axs[i]
        
        ax = look_at_timing_differences(data=data,
                                   col1_name=c,
                                   col2_name='response_latency',
                                   ax=ax,
                                   only_hit=only_hit)
        
        if i!=0:
            ax.set_ylabel("")
        ax.set_xlim([-100,1100])
        ax.set_ylim([-100,1100])
        
     
    plt.tight_layout()
    