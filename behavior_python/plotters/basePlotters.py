from .plotter_utils import *
from os.path import join as pjoin
from ..wheelUtils import *
from scipy import stats
import copy
import matplotlib.patheffects as pe
from behavior_python.detection.wheelDetectionAnalysis import DetectionAnalysis


class BasePlotter:
    __slots__ = ['data','fig','color']
    def __init__(self,data:pl.DataFrame,**kwargs):
        self.data = data
        self.fig = None
        set_style('analysis')
        self.color = Color()
        
    @staticmethod
    def select_stim_data(data_in:pl.DataFrame, stimkey:str=None, drop_early:bool=True) -> dict:
        """ Returns the selected stimulus type from session data
            data_in : 
            stimkey : Dictionary key that corresponds to the stimulus type (e.g. lowSF_highTF)
        """
        # drop early trials
        if drop_early:
            data = data_in.filter(pl.col('answer')!=-1)
        else:
            data = data_in.select(pl.col('*'))
        
        #should be no need for drop_nulls, but for extra failsafe
        uniq_keys = data.select(pl.col('stimkey')).drop_nulls().unique().to_series().to_numpy()
        
        if stimkey is not None and stimkey not in uniq_keys and stimkey != 'all':
            raise KeyError(f'{stimkey} not in stimulus data, try one of these: {uniq_keys}')
        
        if stimkey is None:
            if len(uniq_keys) == 1:
                # if there is only one key just take the data
                key = uniq_keys[0]
            elif len(uniq_keys) > 1:
                # if there is more than one stimkey , take all the data
                key = 'all'
            else:
                # this should not happen
                raise ValueError('There is no stimkey in the data, this should not be the case. Check your data!')
        else:
            # this is the condition that is filtering the dataframe by stimkey
            key = stimkey
            data = data.filter(pl.col('stimkey') == stimkey)
        return data, key, uniq_keys
    
    @staticmethod
    def _make_contrast_axis(contrast_arr:np.ndarray,center0:bool=True)->dict:
        """ Makes an array of sequential numbers to be used in the xaxis_ticks
            This is used in linear spacing mode""" 
        if center0:
            if 0 not in contrast_arr:
                contrast_arr = np.append(contrast_arr,0)
                contrast_arr = np.sort(contrast_arr)
            l = len(contrast_arr)
            return {c:i for i,c in enumerate(contrast_arr,start=int(-(l-1)/2))}

    def save(self,saveloc,date,animalid):
        if self.fig is not None:
            saveloc = pjoin(saveloc,'figures')
            if not os.path.exists(saveloc):
                os.mkdir(saveloc)
            savename = f'{date}_{self.__class__.__name__}_{animalid}.pdf'
            saveloc = pjoin(saveloc,savename)
            self.fig.suptitle(f'{date}_{animalid}',fontsize=23)
            self.fig.savefig(saveloc)
            display(f'Saved {savename} plot')


class PerformancePlotter(BasePlotter):
    __slots__ = ['stimkey','plot_data','uniq_keys']
    def __init__(self,data,stimkey:str=None,**kwargs):
        super().__init__(data, **kwargs)
        self.plot_data, self.stimkey, self.uniq_keys = self.select_stim_data(self.data,stimkey)
        
        #check color definitions
        self.color.check_stim_colors(self.uniq_keys)
        self.color.check_contrast_colors(nonan_unique(self.plot_data['contrast'].to_numpy()))

    @staticmethod
    def __plot__(ax,x,y,**kwargs):
        """ Private function that plots a performance graph with the given 
        x,y and err values are used to plot the points and 
        x_fit and y_fit values are used to plot the fitted curve
        """
        ax.plot(x,y * 100,linewidth=kwargs.get('linewidth',5),**kwargs)

        return ax

    def plot(self, ax:plt.axes=None, plot_in_time:bool=False, seperate_by:str=None, **kwargs) -> plt.axes:
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(8,8)))
            ax = self.fig.add_subplot(1,1,1)
            if 'figsize' in kwargs:
                kwargs.pop('figsize')
                
        if seperate_by is not None:
            if seperate_by not in self.plot_data.columns:
                raise ValueError(f'Cannot seperate the data by {seperate_by}')
            seperate_vals = self.plot_data.select(pl.col(seperate_by)).unique().to_series().to_numpy()
        else:
            seperate_vals = [-1] # dummy value        
        
        for sep in seperate_vals:
            if seperate_by is not None:
                data2plot = self.plot_data.filter(pl.col(seperate_by)==sep)
            else:
                data2plot = self.plot_data.select(pl.col('*'))
            
            y_axis = get_fraction(data2plot['answer'].to_numpy(),fraction_of=1,window_size=20,min_period=10)
            
            if plot_in_time:
                x_axis_ = data2plot['openstart_absolute'].to_numpy() / 60000
                x_label_ = 'Time (mins)'
            else:
                x_axis_ = data2plot['trial_no'].to_numpy()
                x_label_ = 'Trial No'

            ax = self.__plot__(ax,x_axis_,y_axis,
                                label = f'{sep}',
                                **kwargs)
                
        # prettify
        fontsize = kwargs.get('fontsize',20)
        ax.set_ylim([0, 100])
        ax.set_xlabel(x_label_, fontsize=fontsize)
        ax.set_ylabel('Accuracy(%)', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.grid(alpha=0.5,axis='both')
        if seperate_by is not None:
            if len(seperate_by) > 1:
                ax.legend(loc='center left',bbox_to_anchor=(1,0.5),fontsize=fontsize,frameon=False)
        
        return ax
    
    
class ResponseTimePlotter(BasePlotter):
    __slots__ = ['stimkey','plot_data','uniq_keys']
    def __init__(self,data,stimkey:str=None,**kwargs):
        super().__init__(data, **kwargs)
        self.plot_data,self.stimkey,self.uniq_keys = self.select_stim_data(self.data,stimkey)
        
        #check color definitions
        self.color.check_stim_colors(self.uniq_keys)
        self.color.check_contrast_colors(nonan_unique(self.plot_data['contrast'].to_numpy()))
        
    @staticmethod
    def __plot__(ax,x,y,**kwargs):
        ax.plot(x, y,linewidth=kwargs.get('linewidth',5),**kwargs)
        return ax
        
    def plot(self, ax:plt.axes=None,plot_in_time:bool=False,running_window:int=20,**kwargs) -> plt.axes:
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(8,8)))
            ax = self.fig.add_subplot(1,1,1)
            if 'figsize' in kwargs:
                kwargs.pop('figsize')
        
        data2plot = self.plot_data.with_columns(pl.col('response_latency').rolling_median(running_window).alias('running_response_latency'))
        
        if plot_in_time:
            x_axis_ = data2plot['openstart_absolute'].to_numpy() / 60000
            x_label_ = 'Time (mins)'
        else:
            x_axis_ = data2plot['trial_no'].to_numpy()
            x_label_ = 'Trial No'
        
        y_axis_ = data2plot['running_response_latency'].to_numpy()
        
        ax = self.__plot__(ax,x_axis_,y_axis_,
                           label=self.stimkey,
                           **self.color.stim_keys[self.stimkey])

        # prettify
        fontsize = kwargs.get('fontsize',20)
       
        ax.set_xlabel(x_label_, fontsize=fontsize)
        ax.set_ylabel('Response Time (ms)', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        
        ax.set_yscale('symlog')
        minor_locs = [200,400,600,800,2000,4000,6000,8000]
        ax.yaxis.set_minor_locator(plt.FixedLocator(minor_locs))
        ax.yaxis.set_minor_formatter(plt.FormatStrFormatter('%d'))
        ax.yaxis.set_major_locator(ticker.FixedLocator([100,1000,10000]))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

        # ax.set_yticklabels([format(y,'.0f') for y in ax.get_yticks()])
        ax.grid(alpha=0.5,axis='both')
        
        return ax


class ReactionCumulativePlotter(BasePlotter):
    def __init__(self, data: pl.DataFrame, stimkey:str=None, **kwargs):
        super().__init__(data, **kwargs)
        self.plot_data, self.stimkey, self.uniq_keys = self.select_stim_data(self.data,stimkey)
        
        #check color definitions
        self.color.check_stim_colors(self.uniq_keys)
        self.color.check_contrast_colors(nonan_unique(self.plot_data['contrast'].to_numpy()))
        
        self.stat_analysis = DetectionAnalysis(data=self.plot_data)

    
    @staticmethod
    def time_to_log(time_data_arr:np.ndarray) -> np.ndarray:
        response_times = np.log10(time_data_arr/1000)
        response_times = 1000 * np.power(10,response_times)
        response_times = response_times.astype(int)
        return response_times
    
    @staticmethod
    def get_cumulative(time_data_arr:np.ndarray,bin_edges:np.ndarray) -> np.ndarray:
        """ Gets the cumulative distribution"""
        sorted_times = np.sort(time_data_arr)
        counts,_ = np.histogram(sorted_times,bins=bin_edges)
        pdf =  counts/np.sum(counts)
        cum_sum = np.cumsum(pdf)
        return cum_sum
    
    def plot(self,compare_by:str='stim_type',
                  bin_width:int=50,
                  from_wheel:bool=False,
                  first_move:bool=False,**kwargs) -> plt.Axes:
        
        fontsize = kwargs.pop('fontsize',25)
        # make the bin edges array
        bin_edges = np.arange(0,2000,bin_width)
                
        data = self.stat_analysis.agg_data.drop_nulls().sort(['stimkey','opto'],descending=True)

        # add cut
        data = data.with_columns(pl.col('response_times').apply(lambda x: [i for i in x if i<1000]).alias('cutoff_response_times'))
        
        # get uniques
        
        u_stimtype = data['stim_type'].unique().sort().to_numpy()
        n_stim = len(u_stimtype)
        
        u_opto = self.plot_data['opto_pattern'].unique().sort().to_list()
        n_opto = len(u_opto)
        
        u_contrast = data['contrast'].unique().sort().to_numpy()
        u_contrast = u_contrast[1:] # remove 0 contrast, we dont care about it here
        n_contrast = len(u_contrast)
        
        if compare_by == 'stim_type':
            self.fig, axes = plt.subplots(ncols=n_contrast,# remove 0 contrast
                                      nrows=n_opto,
                                      constrained_layout=True,
                                      figsize=kwargs.pop('figsize',(15,15)))
            
            for i,o in enumerate(u_opto):
                for j,c in enumerate(u_contrast):
                    
                    try:
                        ax = axes[i][j]
                    except:
                        ax = axes[j]
                        
                    for k in u_stimtype:
                        filt_df = data.filter((pl.col('opto_pattern')==o) &
                                            (pl.col('stim_side')=="contra") &
                                            (pl.col('contrast')==c) &
                                            (pl.col('stim_type')==k)
                                            )
                        
                        if not filt_df.is_empty():
                            
                            if from_wheel:
                                reaction_times = filt_df['wheel_reaction_time'].explode().to_numpy()
                            else:
                                reaction_times = filt_df['response_times'].explode().to_numpy()
                                
                            cumulative_reaction = self.get_cumulative(reaction_times,bin_edges)
                            

                            ax.plot(bin_edges[:-1],
                                    cumulative_reaction,
                                    color=self.color.stim_keys[filt_df[0,'stimkey']]['color'],
                                    linewidth=4)
                        
                            #line
                            ax.axvline(1000,color='r',linewidth=2)
                            
                            ax.set_ylim([-0.01,1.05])
                            ax.set_xlim([-1,1100])
                            if o:
                                ax.set_title(f"Opto c={c}%", fontsize=fontsize,pad=3)
                            else:
                                ax.set_title(f"Non-Opto c={c}%", fontsize=fontsize,pad=3)
                            if i == n_opto-1:
                                ax.set_xlabel('Time from Stim Onset (ms)', fontsize=fontsize)
                            if j==0:
                                ax.set_ylabel('Probability', fontsize=fontsize)
                            ax.tick_params(labelsize=fontsize,which='both',axis='both')
                            ax.grid(alpha=0.5,axis='both')
                        
                        # ax.legend(loc='center left',bbox_to_anchor=(0.98,0.5),fontsize=fontsize-5,frameon=False)
        elif compare_by == 'opto':
            self.fig, axes = plt.subplots(ncols=n_contrast,# remove 0 contrast
                                          nrows=n_stim,
                                          constrained_layout=True,
                                          figsize=kwargs.pop('figsize',(15,15)))
            for i,k in enumerate(u_stimtype):
                for j,c in enumerate(u_contrast):
                    try:
                        ax = axes[i][j]
                    except:
                        ax = axes[j]
                    for o in u_opto:
                        filt_df = data.filter((pl.col('opto_pattern')==o) &
                                              (pl.col('stim_side')=="contra") &
                                              (pl.col('contrast')==c) &
                                              (pl.col('stim_type')==k)
                                            )
                    
                        if not filt_df.is_empty():
                            if from_wheel:
                                reaction_times = filt_df['wheel_reaction_time'].explode().to_numpy()
                            else:
                                reaction_times = filt_df['response_times'].explode().to_numpy()
                                
                            cumulative_reaction = self.get_cumulative(reaction_times,bin_edges)
                
                            ax.plot(bin_edges[:-1],
                                    cumulative_reaction,
                                    color=self.color.stim_keys[filt_df[0,'stimkey']]['color'],
                                    linewidth=4)
                        
                            #line
                            ax.axvline(1000,color='r',linewidth=2)
                            
                            ax.set_ylim([-0.01,1.05])
                            ax.set_xlim([-1,1100])
                            ax.set_title(f"{filt_df[0,'stim_label']} c={c}%", fontsize=fontsize,pad=3)
                            if i == n_stim-1:
                                ax.set_xlabel('Response Time (ms)', fontsize=fontsize)
                            if j==0:
                                ax.set_ylabel('Probability', fontsize=fontsize)
                            ax.tick_params(labelsize=fontsize,which='both',axis='both')
                            ax.grid(alpha=0.5,axis='both')  

class ResponseTimeDistributionPlotter(BasePlotter):
    def __init__(self, data, stimkey:str=None, **kwargs):
        super().__init__(data, **kwargs)
        self.plot_data, self.stimkey, self.uniq_keys = self.select_stim_data(self.data,stimkey)
        
        #check color definitions
        self.color.check_stim_colors(self.uniq_keys)
        self.color.check_contrast_colors(nonan_unique(self.plot_data['contrast'].to_numpy()))
        
        self.stat_analysis = DetectionAnalysis(data=self.plot_data)
    
    @staticmethod      
    def time_to_log(time_data_arr:np.ndarray) -> np.ndarray:
        response_times = np.log10(time_data_arr/1000)
        response_times = 1000 * np.power(10,response_times)
        response_times = response_times.astype(int)
        return response_times
    
    @staticmethod
    def make_dot_cloud(response_times,pos,cloud_width):
        counts,bin_edges = np.histogram(response_times,bins=5)

        part_x = []
        part_y = []
        for j,point_count in enumerate(counts):
            points = [p for p in response_times if p>=bin_edges[j] and p <=bin_edges[j+1]]
            # generate x points around the actual x
            # range is determined by point count in the bin and scaling factor

            scatter_range = np.linspace(pos - cloud_width*(point_count/np.max(counts)),
                                        pos + cloud_width*(point_count/np.max(counts)),
                                        len(points)).tolist()
            part_x += scatter_range
            part_y += points
        
        return part_x,part_y
            
    @staticmethod
    def __plot_scatter__(ax,contrast,time,median,pos,cloud_width,**kwargs):
        """ Plots the trial response times as a scatter cloud plot """
        if 'fontsize' in kwargs.keys():
            kwargs.pop('fontsize')
        ax.scatter(contrast,time,alpha=0.6,**kwargs)
        
        #median
        ax.plot([pos-cloud_width/2,pos+cloud_width/2],[median,median],linewidth=3,
                c=kwargs.get('color','b'),path_effects=[pe.Stroke(linewidth=6, foreground='k'), pe.Normal()])

        return ax
    
    @staticmethod
    def __plot_line__(ax,x,y,err,**kwargs):
        """ Plots the trial response times as a line plot with errorbars"""
        if 'fontsize' in kwargs.keys():
            kwargs.pop('fontsize')
        
        ax.errorbar(x, y, err,
                    linewidth=2,
                    markeredgecolor=kwargs.get('markeredgecolor','w'),
                    markeredgewidth=kwargs.get('markeredgewidth',2),
                    elinewidth=kwargs.get('elinewidth',3),
                    capsize=kwargs.get('capsize',0),
                    **kwargs)
        return ax
    
    @staticmethod
    def add_jitter_to_misses(resp_times,jitter_lims=[0,100]):
        """ Adds jitter in y-dimension to missed trial dot"""
        miss_locs = np.where(resp_times>=1000)[0]
        if len(miss_locs):
            jitter = np.random.choice(np.arange(jitter_lims[0],jitter_lims[1]),len(miss_locs),replace=True)
            resp_times[miss_locs] = resp_times[miss_locs] + jitter
        return resp_times
        
    def plot_scatter(self,ax:plt.Axes=None,
             t_cutoff:float=10_000,
             cloud_width:float=0.33,
             xaxis_type:str='linear_spaced',
             wheel_time:bool = True,
             **kwargs):
        
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(15,10)))
            ax = self.fig.add_subplot(1,1,1)
            if 'figsize' in kwargs:
                kwargs.pop('figsize')
            
        data = self.stat_analysis.agg_data.drop_nulls().sort(['stimkey','opto'],descending=True)
        
        # do cutoff
        data = data.with_columns([pl.col('response_times').apply(lambda x: [i for i in x if i<t_cutoff]).alias('cutoff_response_times'),
                                  pl.col('wheel_reaction_time').apply(lambda x: [i for i in x if i<t_cutoff and i is not None]).alias('cutoff_wheel_reaction_times')])
        
        # get uniques
        u_stimkey = data['stimkey'].unique().to_numpy()
        u_stimtype = data['stim_type'].unique().to_numpy()
        u_stim_side = data['stim_side'].unique().to_numpy()
        u_scontrast = data['signed_contrast'].unique().sort().to_numpy()
        
        cpos = self._make_contrast_axis(u_scontrast)
        
        for k in u_stimkey:
            for s in u_stim_side:
                for c in u_scontrast:
                    filt_df = data.filter((pl.col('stimkey')==k) &
                                          (pl.col('stim_side')==s) &
                                          (pl.col('signed_contrast')==c))
                    
                    if not filt_df.is_empty():
                        if wheel_time:
                            resp_times = filt_df[0,'cutoff_wheel_reaction_times'].to_numpy()
                        else:
                            resp_times = filt_df[0,'cutoff_response_times'].to_numpy()
                        # do cutoff, default is 10_000 to involve everything
                        
                        response_times = self.time_to_log(resp_times)
                        response_times = self.add_jitter_to_misses(response_times)
                        
                        x_dots,y_dots = self.make_dot_cloud(response_times,cpos[c],cloud_width)
                        median = np.median(response_times)
                        ax = self.__plot_scatter__(ax,x_dots,y_dots,median,cpos[c],cloud_width,
                                                   color=self.color.stim_keys[k]['color'],
                                                   label=filt_df[0,'stim_label'] if s=='contra' and c==12.5 else '_',
                                                   **kwargs)
                    
        p_data = data.sort(["stim_type","contrast","stim_side"])
        
        for j,s_t in enumerate(u_stimtype):
            for c in u_scontrast:
                pfilt_df = p_data.filter((pl.col('stim_type')==s_t) &
                                         (pl.col('signed_contrast')==c))
                
                if len(pfilt_df)<2:
                    continue
                elif len(pfilt_df)==2:
                    if wheel_time:
                        p = self.stat_analysis.get_pvalues_nonparametric(pfilt_df[0,'cutoff_wheel_reaction_times'].to_numpy(),
                                                                         pfilt_df[1,'cutoff_wheel_reaction_times'].to_numpy())            
                    else:
                        p = self.stat_analysis.get_pvalues_nonparametric(pfilt_df[0,'cutoff_response_times'].to_numpy(),
                                                                        pfilt_df[1,'cutoff_response_times'].to_numpy())            
                    stars = ''
                    if p < 0.001:
                        stars = '***'
                    elif 0.001 < p < 0.01:
                        stars = '**'
                    elif 0.01 < p < 0.05:
                        stars = '*'
                    ax.text(cpos[c], 1100+j*200, stars,color=self.color.stim_keys[pfilt_df[0,'stimkey']]['color'], fontsize=30)
                    
                else:
                    raise ValueError(f'WEIRD DATA FRAME FOR P-VALUE ANALYSIS!')
                
        # mid line
        fontsize = kwargs.get('fontsize',25)
        ax.set_ylim([90,1500])
        ax.plot([0,0],ax.get_ylim(),color='gray',linewidth=2,alpha=0.5)
        
        fontsize = kwargs.get('fontsize',20)
        ax.set_xlabel('Stimulus Contrast (%)', fontsize=fontsize)
        ax.set_ylabel('Response Time (ms)', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize,which='both',axis='both')
        
        ax.set_yscale('symlog')
        minor_locs = [200,400,600,800,2000,4000,6000,8000]
        ax.yaxis.set_minor_locator(plt.FixedLocator(minor_locs))
        ax.yaxis.set_minor_formatter(plt.FormatStrFormatter('%d'))
        ax.yaxis.set_major_locator(ticker.FixedLocator([100,1000,10000]))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        
        
        ax.xaxis.set_major_locator(ticker.FixedLocator(list(cpos.values())))
        ax.xaxis.set_major_formatter(ticker.FixedFormatter([int(i) for i in cpos.keys()]))
            
        ax.grid(alpha=0.5,axis='both')
        
        ax.spines['bottom'].set_position(('outward', 10))
        ax.spines['left'].set_position(('outward', 10))
        
        ax.legend(loc='center left',bbox_to_anchor=(0.98,0.5),fontsize=fontsize-5,frameon=False)
        
        return ax    
    
    def plot_line(self,ax:plt.Axes=None,
                  t_cutoff:float=10_000,
                  jitter:int=2,
                  xaxis_type:str='linear_spaced',
                  **kwargs):
        
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(15,10)))
            ax = self.fig.add_subplot(1,1,1)
            if 'figsize' in kwargs:
                kwargs.pop('figsize')
                
        data = self.stat_analysis.agg_data.drop_nulls().sort(['stimkey','opto'],reverse=True)
        
        # do cutoff
        data = data.with_columns(pl.col('response_times').apply(lambda x: [i for i in x if i<t_cutoff]).alias('cutoff_response_times'))
        
        # add 95% confidence 
        def get_conf(arr):
            x = np.sort(arr)  
            j = round(len(x)*0.5 - 1.96*np.sqrt(len(x)**0.5))
            k = round(len(x)*0.5 + 1.96*np.sqrt(len(x)**0.5))
            return [x[j],x[k]]

        # data = data.with_columns(pl.col('response_times').apply(lambda x: [np.mean(x.to_numpy())-2*np.std(x.to_numpy()),np.mean(x.to_numpy())+2*np.std(x.to_numpy())]).alias('resp_confs'))
        data = data.with_columns(pl.col('response_times').apply(lambda x: get_conf(x)).alias('resp_confs'))
        
        
        # get uniques
        u_stimkey = data['stimkey'].unique().to_numpy()
        u_stimtype = data['stim_type'].unique().to_numpy()
        u_stim_side = data['stim_side'].unique().to_numpy()
        u_scontrast = data['signed_contrast'].unique().sort().to_numpy()
        
        for k in u_stimkey:
            for s in u_stim_side:
                filt_df = data.filter((pl.col('stimkey')==k) &
                                      (pl.col('stim_side')==s))
                
                contrast = filt_df['signed_contrast'].to_numpy()    
                confs = filt_df['resp_confs'].to_list()
                confs = np.array(confs).T
                if not filt_df.is_empty():
                    resp_times = filt_df['cutoff_response_times'].to_numpy()
                    # do cutoff, default is 10_000 to involve everything
                    medians = []
                    for i,c_r in enumerate(resp_times):
                        response_times = self.time_to_log(c_r)
                        response_times = self.add_jitter_to_misses(response_times)
                    
                        median = np.median(response_times)
                        medians.append(median)
                        # jittered_offset = np.array([np.random.uniform(0,jitter)*c for c in contrast])
                        # jittered_offset[0] += np.random.uniform(0,jitter)/100
                        
                    ax = self.__plot_line__(ax, contrast, medians, confs, 
                                            color=self.color.stim_keys[k]['color'],
                                            label=filt_df[0,'stim_label'] if s=='contra' else '_',
                                           **kwargs)
                    
        # mid line
        fontsize = kwargs.get('fontsize',25)
        ax.set_ylim([90,1500])
        ax.plot([0,0],ax.get_ylim(),color='gray',linewidth=2,alpha=0.5)
        
        fontsize = kwargs.get('fontsize',20)
        ax.set_xlabel('Stimulus Contrast (%)', fontsize=fontsize)
        ax.set_ylabel('Response Time (ms)', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize,which='both',axis='both')
        
        ax.set_yscale('symlog')
        minor_locs = [200,400,600,800,2000,4000,6000,8000]
        ax.yaxis.set_minor_locator(plt.FixedLocator(minor_locs))
        ax.yaxis.set_minor_formatter(plt.FormatStrFormatter('%d'))
        ax.yaxis.set_major_locator(ticker.FixedLocator([100,1000,10000]))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        
            
        ax.grid(alpha=0.5,axis='both')
        
        ax.spines['bottom'].set_position(('outward', 10))
        ax.spines['left'].set_position(('outward', 10))
        
        ax.legend(loc='center left',bbox_to_anchor=(0.98,0.5),fontsize=fontsize-5,frameon=False)
        
        return ax
         

class ResponseTimeHistogramPlotter(BasePlotter):
    __slots__ = ['stimkey','plot_data','uniq_keys']
    def __init__(self, data, stimkey:str=None, **kwargs):
        super().__init__(data=data, **kwargs)
        self.plot_data, self.stimkey, self.uniq_keys = self.select_stim_data(self.data,stimkey, drop_early=False)
        
        #check color definitions
        self.color.check_stim_colors(self.uniq_keys)
        self.color.check_contrast_colors(nonan_unique(self.plot_data['contrast'].to_numpy()))
     
    @staticmethod
    def bin_times(time_arr,bin_width=50,bins:np.ndarray=None):
        """ Counts the response times in bins(ms)"""
        if bins is None:
            bins = np.arange(np.min(time_arr)-bin_width,np.max(time_arr)+bin_width,bin_width)
        
        return np.histogram(time_arr,bins)
        
    @staticmethod
    def __plot__(ax,counts,bins,**kwargs):
        # adapt the bar width to the bin width
        bar_width = bins[1] - bins[0]
        
        if kwargs.get('color') is not None:
            cl = kwargs.get('color')
            kwargs.pop('color')
        else:
            cl = 'forestgreen'
        
        color = ['orangered' if i<=150 else cl for i in bins]    
        
        ax.bar(bins[1:],counts,
               width=bar_width,
               color=color,
               edgecolor='k',
               **kwargs)
        
        # zero line
        ax.axvline(x=0,color='k',linestyle=':',linewidth=3)
        
        return ax
    
    
class ResponseTypeBarPlotter(BasePlotter):
    __slots__ = ['stimkey','plot_data','uniq_keys']
    def __init__(self, data, stimkey:str=None, **kwargs):
        super().__init__(data=data, **kwargs)
        self.plot_data, self.stimkey, self.uniq_keys = self.select_stim_data(self.data,stimkey, drop_early=False)
        
        #check color definitions
        self.color.check_stim_colors(self.uniq_keys)
        self.color.check_contrast_colors(nonan_unique(self.plot_data['contrast'].to_numpy()))
        
    @staticmethod    
    def __plot__(ax,x_locs,bar_heights,**kwargs):
        ax.bar(x_locs,bar_heights,**kwargs)
        return ax
 

class LickPlotter(BasePlotter):
    __slots__ =['stimkey','plot_data']
    def __init__(self, data: dict, stimkey:str=None, **kwargs):
        super().__init__(data, **kwargs)
        self.plot_data, self.stimkey = self.select_stim_data(self.data, stimkey)
        self.color.check_stim_colors(self.plot_data.keys())
        
        c_list = nonan_unique(self.plot_data[list(self.plot_data.keys())[0]]['contrast'])
        self.color.check_contrast_colors(c_list)
        
    def pool_licks(self):
        
        if len(self.plot_data) > 1:
            # multiple stim types, get the individual dataframes and sort according to trial_no
            lick_dataframe = pd.DataFrame()
            for k,v in self.plot_data.items():
                lick_dataframe = pd.concat([lick_dataframe,v])
            lick_dataframe.sort_values('trial_no',inplace=True)
        elif len(self.plot_data) == 1:
            # single stim type, use that
            lick_dataframe = self.plot_data[self.stimkey]
        
        all_lick = np.array([]).reshape(-1,2)
        for row in lick_dataframe.itertuples():
            if len(row.lick):
                temp_lick = row.lick.copy()
                temp_lick[:,0] =+ row.openstart_absolute
                all_lick = np.append(all_lick,temp_lick,axis=0)       
        return all_lick
    
    @staticmethod
    def __plot__(ax,x,y,**kwargs):
        ax.plot(x,y,**kwargs)
        return ax
             
    def plot(self,ax:plt.axes=None, plot_in_time:bool=False, **kwargs) -> plt.axes:
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(8,8)))
            ax = self.fig.add_subplot(1,1,1)
            
        all_licks = self.pool_licks()
        if len(all_licks):
            if plot_in_time:
                x_axis_ = all_licks[:,0] / 60000
                y_axis_ = all_licks[:,1]
                x_label_ = 'Time (mins)'
            else:
                x_axis_ = self.plot_data[self.stimkey]['trial_no']
                all_licks[:,0] = (all_licks[:,0]/np.max(all_licks[:,0])) * self.plot_data[self.stimkey]['trial_no'].iloc[-1]
                y_axis_ = np.interp(self.plot_data[self.stimkey]['trial_no'],all_licks[:,0],all_licks[:,1])
                x_label_ = 'Trial No'
            
            ax = self.__plot__(ax, x_axis_,y_axis_,**kwargs)
        
        else:
            display('No Lick data found for session :(')
            ax = self.__plot__(ax,[],[])
            
        fontsize = kwargs.get('fontsize',20)
        ax.set_xlabel(x_label_, fontsize=fontsize)
        ax.set_ylabel('Lick Counts', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.tick_params(axis='x')
        ax.grid(alpha=0.5,axis='y')
        
        return ax
    
    
class LickScatterPlotter(BasePlotter):
    """ Lick scatters for each trial, wrt reward or response time"""
    def __init__(self, data: dict, stimkey:str=None, **kwargs):
        super().__init__(data, **kwargs)
        self.plot_data, self.stimkey, self.uniq_keys = self.select_stim_data(self.data,stimkey)
        
        #check color definitions
        self.color.check_stim_colors(self.uniq_keys)
        self.color.check_contrast_colors(nonan_unique(self.plot_data['contrast'].to_numpy()))
    
    @staticmethod
    def __plot_scatter__(ax,t,lick_arr,**kwargs):
        
        t_arr = [t] * len(lick_arr)
        
        ax.scatter(lick_arr,t_arr,marker='|',c='deepskyblue',
                   s=kwargs.get('s',20),**kwargs)
        
        return ax
    
    @staticmethod
    def pool_licks(data,wrt:str='reward'):
        pooled_lick = np.array([])
        error_ctr = []

        for row in data.iter_rows(named=True):
            if len(row['lick']):
                if wrt=='reward':
                    try:
                        wrt_time = row['reward'][0]
                    except:
                        error_ctr.append(row.trial_no)
                        display(f'\n!!!!!! NO REWARD IN CORRECT TRIAL, THIS IS A VERY SERIOUS ERROR! SOLVE THIS ASAP !!!!!!\n')
                elif wrt=='response':
                    wrt_time = row['response_latency_absolute']
                
                pooled_lick = np.append(pooled_lick,np.array(row['lick']) - wrt_time)
        print(f'Trials with reward issue: {error_ctr}')         
        return pooled_lick
    
    @staticmethod
    def __plot_density__(ax,x_bins,y_dens,**kwargs): 
        ax.plot(x_bins[1:],y_dens,c='aqua',alpha=0.8,linewidth=3,**kwargs) #right edges
        return ax
    
    
class WheelTrajectoryPlotter(BasePlotter):
    def __init__(self, data: pl.DataFrame, stimkey:str=None,**kwargs) -> None:
        super().__init__(data, **kwargs)
        self.plot_data, self.stimkey, self.uniq_keys = self.select_stim_data(self.data,stimkey,
                                                                             drop_early=kwargs.pop('drop_early',True))
        
        #check color definitions
        self.color.check_stim_colors(self.uniq_keys)
        self.color.check_contrast_colors(nonan_unique(self.plot_data['contrast'].to_numpy()))
        
    @staticmethod
    def __plot__(ax:plt.Axes,wheel_pos,wheel_t,**kwargs):
        ax.plot(wheel_pos, wheel_t,
                linewidth=kwargs.get('linewidth',5),
                alpha=1,
                zorder=2,
                **kwargs)
        
        return ax
    
    @staticmethod
    def __plot_density__(ax,x_bins,y_dens,**kwargs): 
        ax.plot(x_bins[1:],y_dens,c='k',alpha=0.8,linewidth=3,**kwargs) #right edges
        return ax
    
    def pool_trial_ends(self) -> np.ndarray:
        """ Gets the relative(from stim start) stimulus end times"""
        pooled_ends = []
        pool_data = self.plot_data[self.stimkey].copy(deep=True)
        # pool_data = pool_data[(pool_data['answer']==1) & (pool_data['isCatch']==0)]
        pool_data = pool_data[pool_data['isCatch']==0]
        for row in pool_data.itertuples():
            try:
                temp = row.stim_end_rig - row.stim_start_rig
            except AttributeError:
                temp = row.stim_end - row.stim_start
            pooled_ends.append(temp)
        return np.array(pooled_ends)
             
    def plot(self,
             seperate_by:str='contrast',
             time_lims:list=None,
             traj_lims:list=None,
             trace_type:str='sem',
             **kwargs):
        
        fontsize = kwargs.pop('fontsize',20)
        if time_lims is None:
            time_lims = [-200,1500]
        if traj_lims is None:
            traj_lims = [-75,75]
        
        
        uniq_opto = self.plot_data['opto_pattern'].unique().sort().to_list()
        n_opto = len(uniq_opto)
        uniq_stims = self.plot_data['stim_type'].unique().sort().to_list()
        n_stim = len(uniq_stims)
        
        uniq_sides  = self.plot_data['stim_pos'].unique().sort().to_list()
        
        # this could be contrast, answer 
        uniq_sep = self.plot_data[seperate_by].unique().sort().to_list()
        if seperate_by == 'contrast':
            color = self.color.contrast_keys
        elif seperate_by == 'answer':
            color = self.color.answer_keys
        
        self.fig, axes = plt.subplots(ncols=n_stim,
                                      nrows=n_opto,
                                      constrained_layout=True,
                                      figsize=kwargs.pop('figsize',(15,15)))

        for i,opto in enumerate(uniq_opto):
            for j,stim in enumerate(uniq_stims):
                for side in uniq_sides:
                    for sep in uniq_sep:
                        try:
                            ax = axes[i][j] 
                        except:
                            ax=axes[i]
                        filt_data = self.plot_data.filter((pl.col('opto_pattern')==opto) &
                                                          (pl.col('stim_type')==stim) &
                                                          (pl.col('stim_pos')==side) & 
                                                          (pl.col(seperate_by)==sep))
                        if len(filt_data):
                            wheel_time = filt_data['wheel_time'].to_numpy()
                            wheel_pos = filt_data['wheel_pos'].to_numpy()
                        
                            
                            wheel_interp_t,wheel_stats = get_trajectory_avg(wheel_time,wheel_pos)
                            avg = wheel_stats['avg']
                            sem = wheel_stats['sem']
                            
                            wheel_y = avg[1,:]+side
                            sem_plus = wheel_y + sem[1,:]
                            sem_minus = wheel_y - sem[1,:]
                            
                            if avg is not None:
                                # avg = avg[find_nearest(avg[:,0],plot_range_time[0])[0]:find_nearest(avg[:,0],plot_range_time[1])[0]]
                                # sem = sem[find_nearest(sem[:,0],plot_range_time[0])[0]:find_nearest(sem[:,0],plot_range_time[1])[0]]
                                if trace_type=='sem':
                                    ax.fill_between(wheel_interp_t,sem_plus,sem_minus,
                                                alpha=0.2,
                                                color=color[str(sep)]['color'],
                                                linewidth=0)
                                elif trace_type=='indiv':
                                    for w in wheel_stats['indiv']:
                                        ax.plot(wheel_interp_t,w+side,
                                                color = color[str(sep)]['color'],
                                                linewidth=0.8,
                                                alpha=0.5)
                                
                                ax = self.__plot__(ax,wheel_interp_t,wheel_y,
                                                    color=color[str(sep)]['color'],
                                                    label=sep if side>=0 else '_', #only put label for 0 and right side(where opto is mostly present)
                                                    **kwargs) 
                    
                        
                        ax.set_xlim(time_lims)
                        ax.set_ylim(traj_lims)
                        
                        # closed loop start line
                        ax.axvline(0,color='k',linewidth=2,alpha=0.6)
                        
                        # stim end
                        ax.axvline(1000,color='k',linestyle=':',linewidth=2,alpha=0.6)
                        
                        #5*((3*2*np.pi*31.2)/1024) where 5 is the tick difference to detect answer
                        ax.axhline(side+2.871,color='r',linestyle="--",linewidth=2,alpha=0.6)
                        ax.axhline(side-2.871,color='r',linestyle="--",linewidth=2,alpha=0.6)

                        ax.set_title(f'{stim}_{opto}')
                        ax.set_ylabel('Wheel Position (deg)', fontsize=fontsize)
                        ax.set_xlabel('Time(ms)', fontsize=fontsize)
                        
                        # make it pretty
                        ax.spines['bottom'].set_visible(False)
                        ax.spines['left'].set_visible(False)
                
                        ax.tick_params(labelsize=fontsize)
                        ax.grid(axis='y')
                        ax.legend(frameon=False,fontsize=14)
