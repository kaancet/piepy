from matplotlib.pyplot import axes
from ..basePlotters import *
from scipy.stats import fisher_exact, barnard_exact


class DetectionPsychometricPlotter(BasePlotter):
    def __init__(self, data:pl.DataFrame, **kwargs) -> None:
        super().__init__(data,**kwargs)
        self.stat_analysis = DetectionAnalysis(data=data)
        
    @staticmethod
    def __plot__(ax,x,y,err,**kwargs):
        """ Private function that plots a psychometric curve with the given 
        x,y and err values are used to plot the points and 
        x_fit and y_fit values are used to plot the fitted curve
        """
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
    def _makelabel(name:np.ndarray,count:np.ndarray) -> str:
        ret = f'''\nN=['''
        for i,n in enumerate(name):
            ret += fr'''{float(n)}:$\bf{count[i]}$, '''
        ret = ret[:-2] # remove final space and comma
        ret += ''']'''
        return ret
                   
    def plot(self,ax:plt.Axes=None,
             jitter:int=2,
             xaxis_type:str='linear_spaced',
             doP:bool = True,
             color=None,
             **kwargs):
        """ Plots the hit rates with 95% confidence intervals"""
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(8,8)))
            ax = self.fig.add_subplot(1,1,1)
            if 'figsize' in kwargs:
                kwargs.pop('figsize')

        if doP:
        # get the p-values
            self.p_vals = pl.DataFrame()
            for side in ['ipsi','contra','catch']:
                p_side = self.stat_analysis.get_hitrate_pvalues_exact(side=side)
                if not p_side.is_empty():
                    self.p_vals = pl.concat([self.p_vals,p_side])

        nonearly_data = self.stat_analysis.agg_data.drop_nulls('contrast')
        q = nonearly_data.sort(['stimkey','opto_pattern'],descending=True)
        
        # get uniques
        u_stimkey = q['stimkey'].unique().to_numpy()
        u_stim_side = q['stim_side'].unique().to_numpy()
        u_stim_side = [i for i in u_stim_side if i!='catch']
        
        for k in u_stimkey:
            for s in u_stim_side:
                filt_df = q.filter((pl.col('stimkey')==k) &
                                   (pl.col('stim_side')==s))
                
                if not filt_df.is_empty():
                    _contrast = filt_df['signed_contrast'].to_numpy()
                    if xaxis_type == 'linear_spaced':
                        contrast = np.arange(1,len(_contrast)+1) if len(_contrast)>1 else [0]  
                    contrast = -1*contrast if s=='ipsi' else contrast
                    
                    confs = 100*filt_df['confs'].to_numpy()
                    count = filt_df['count'].to_numpy()
                    hr = 100*filt_df['hit_rate'].to_numpy()
                    stim_label = filt_df['stim_label'].unique().to_numpy()                    
                    
                    jittered_offset = np.array([np.random.uniform(0,jitter)*c for c in contrast])
                    jittered_offset[0] += np.random.uniform(0,jitter)/100
                    ax = self.__plot__(ax,
                                    contrast+jittered_offset,
                                    hr,
                                    confs,
                                    label=f"{stim_label[0]}{self._makelabel(_contrast,count)}",
                                    marker = 'o',
                                    markersize=18,
                                    color = self.color.stim_keys[k]['color'] if color is None else color,
                                    linestyle = self.color.stim_keys[k]['linestyle'],
                                    **kwargs)
        
        #baseline
        baseline = q.filter((pl.col('stim_side')=='catch') & (pl.col('opto')==False))
        if len(baseline):
            cnt = baseline['count'].to_numpy()
            base_hr = np.sum(baseline['correct_count'].to_numpy()) / np.sum(cnt)
            base_conf = 1.96 * np.sqrt((base_hr*(1.0 - base_hr)) / np.sum(cnt))
            self.__plot__(ax,
                          0, 100*base_hr, 100*base_conf,
                          label=f"Catch Trials{self._makelabel([0],cnt)}",
                          marker = 'o',
                          markersize=18,
                          color = '#909090',
                          **kwargs)
            ax.axhline(100*base_hr, color='k', linestyle=':', linewidth=2,alpha=0.7)
        
        if xaxis_type == 'log':
            ax.set_xscale('symlog')
            x_ticks = nonearly_data['signed_contrast'].unique().sort().to_numpy()
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
            ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0,subs=np.linspace(0.1,1,9,endpoint=False)))
            x_t = {t:t for t in x_ticks}
        
        elif xaxis_type=='linear':
            x_ticks = nonearly_data['signed_contrast'].unique().sort().to_numpy()
            ax.set_xticks(x_ticks)
            ax.set_xlim([x_ticks[0]-10,x_ticks[-1]+10])
            x_t = {t:t for t in x_ticks}
        
        elif xaxis_type=='linear_spaced':
            temp = nonearly_data['signed_contrast'].unique().sort().to_numpy()
            x_ticks = np.arange(-(len(temp)-1)/2,(len(temp)-1)/2+1)
            x_t = {temp[i]:t for i,t in enumerate(x_ticks)}
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(temp)
            ax.set_xlim([x_ticks[0]-0.5,x_ticks[-1]+0.5])
            
        # put the significance starts
        if doP:
            for i in range(len(self.p_vals)):
                p = self.p_vals[i,'p_values']
                c = self.p_vals[i,'contrast']
                s_k = self.p_vals[i,'stimkey']
                stars = ''
                if p < 0.001:
                    stars = '***'
                elif 0.001 < p < 0.01:
                    stars = '**'
                elif 0.01 < p < 0.05:
                    stars = '*'
                else:
                    continue
            
                ax.text(x_t[c], 102+2*i, stars,color=self.color.stim_keys[s_k]['color'], fontsize=30)
    
        # prettify
        fontsize = kwargs.get('fontsize',25)
        ax.set_ylim([0,110])
        ax.set_yticklabels(int(i) for i in ax.get_yticks())
        ax.set_xlabel('Stimulus Contrast (%)', fontsize=fontsize)
        ax.set_ylabel('Hit Rate (%)',fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        
        ax.spines['left'].set_bounds(0, 100) 
        # ax.spines['bottom'].set_bounds(0, 1)
        ax.spines['bottom'].set_position(('outward', 10))
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        ax.grid(alpha=0.4)
        ax.legend(loc='center left',bbox_to_anchor=(1,0.5),fontsize=fontsize,frameon=False)
        
        return ax


class DetectionPerformancePlotter(PerformancePlotter):
    __slots__ = []
    def __init__(self,data:pl.DataFrame,stimkey:str=None,**kwargs):
        super().__init__(data, stimkey, **kwargs)


class DetectionResponseTimeScatterCloudPlotter(ResponseTimeDistributionPlotter):
    __slots__ = []
    def __init__(self, data:pl.DataFrame, stimkey:str=None, **kwargs):
        super().__init__(data, stimkey, **kwargs)
   

class DetectionReactionCumulativePlotter(ReactionCumulativePlotter):
    __slots__ = []
    def __init__(self, data:pl.DataFrame, stimkey:str=None, **kwargs):
        super().__init__(data, stimkey, **kwargs)
    
 
class DetectionResponseHistogramPlotter(ResponseTimeHistogramPlotter):
    """ Plots an histogram of response times, showing earlies and hits"""
    __slots__ = []
    def __init__(self, data:pl.DataFrame, stimkey: str=None, **kwargs):
        super().__init__(data, stimkey, **kwargs)
        
    @staticmethod
    def shuffle_times(x_in,n_shuffle:int=1000) -> np.ndarray:
        """ Shuffles x_in n_shuffle times """
        gen = np.random.default_rng()
        x_in = x_in.reshape(-1,1)
        x_temp = x_in.copy()
        shuffled_matrix = np.zeros((n_shuffle,x_in.shape[0]))
        
        for i in range(n_shuffle):
            gen.shuffle(x_temp)
            shuffled_matrix[i,:] = x_temp.reshape(1,-1) 
            
        return shuffled_matrix
    
    def make_agg_data(self):
        """ Aggregates the data"""
        q = (
            self.plot_data.lazy()
            .groupby(["stim_type","contrast","opto"])
            .agg(
                [ 
                    (pl.col("response_latency").alias("response_times")),
                    (pl.col("response_latency").median().alias("median_response_time")),
                    (pl.col("opto_pattern").first()),
                    (pl.col("stimkey").first()),
                    (pl.col("stim_label").first()),
                ]
            ).sort(["stim_type","contrast","opto"])
            )
        df = q.collect()
        return df
        
    def plot(self,ax:plt.Axes=None,bin_width=50,seperate_stims:bool=False,**kwargs):
        n_shuffle = kwargs.get('n_shuffle',1000)
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(15,10)))
            ax = self.fig.add_subplot(1,1,1)
        
        self.plot_data = self.plot_data.with_columns(pl.when(pl.col('outcome')!=-1).then(pl.col('response_latency')+pl.col('blank_time'))
                                                     .otherwise(-(pl.col('blank_time')-pl.col('response_latency'))).alias("blanked_response_latency"))
        
        # first plot the earlies
        early_data = self.plot_data.filter(pl.col('outcome')==-1)
        resp_times = early_data['blanked_response_latency'].to_numpy()           
        counts,bins = self.bin_times(resp_times,bin_width)
        ax = self.__plot__(ax,counts,bins,color='r',label='Early')
        
        
        if seperate_stims:
            self.plot_data = self.make_agg_data()

        uniq_type = self.plot_data['stim_type'].unique().to_numpy()
        uniq_opto = self.plot_data['opto'].unique().to_numpy()
        
        for i,t in enumerate(uniq_type):
            for o in uniq_opto:
                filt_df = self.plot_data.filter((pl.col('stim_type')==t) & 
                                                (pl.col('opto')==o) & 
                                                (pl.col('outcome')==1))
                if len(filt_df):
                     
                    # resp_times_blanked = filt_df['blanked_response_latency'].to_numpy()
                    # blank_times = filt_df['blank_time'].to_numpy()
                    resp_times = filt_df['response_latency'].to_numpy()
                        
                    counts,bins = self.bin_times(resp_times,bin_width)
                    
                    k = filt_df[0,'stimkey']
                    label = filt_df[0,'stim_label']
                    ax = self.__plot__(ax,counts,
                                       bins,
                                       color=self.color.stim_keys[k]['color'],
                                       alpha=0.7,
                                       label=label)
                    
                    #plotting the median
                    ax.axvline(np.median(resp_times),
                               color=self.color.stim_keys[k]['color'],
                               linewidth=3,
                               label=f'{label} Median')
                    # plotting the shuffled histograms
                    # shuffled = self.shuffle_times(resp_times_blanked)
                    # shuffled_hists = np.zeros((n_shuffle,len(counts)))

                    # for i,row in enumerate(shuffled):

                    #     row -= blank_times
                    #     counts,_ = self.bin_times(row,bin_width,bins=bins)
                    #     shuffled_hists[i,:] = counts.reshape(1,-1)
                
                    # #mean & std
                    # shuf_mean = np.mean(shuffled_hists,axis=0)
                    # shuf_std = np.std(shuffled_hists,axis=0)
                    
                    # ax.fill_between(bins[1:],shuf_mean-shuf_std,shuf_mean+shuf_std,color='dimgrey',alpha=0.4,zorder=2)
                    # ax.plot(bins[1:],shuf_mean,color='dimgrey',alpha=0.6,linewidth=2,zorder=3)
        
        fontsize = kwargs.get('fontsize',25)
        ax.legend(loc='center left',bbox_to_anchor=(0.05,0.8),fontsize=fontsize-5,frameon=False)
        ax.set_xlabel('Time from Stimulus onset (ms)', fontsize=fontsize)
        ax.set_ylabel('Counts', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.grid(alpha=0.5,axis='y')
        
        return ax
        
    
class DetectionResponseTypeBarPlotter(ResponseTypeBarPlotter):
    __slots__ = []
    def __init__(self, data, stimkey:str=None, **kwargs):
        super().__init__(data, stimkey, **kwargs)
    
    def make_agg_data(self,remove_early:bool=False):
        """ Aggregates the data"""
        q = self.plot_data.lazy()
        if remove_early:
            q = q.filter(pl.col('outcome')!=-1)
        
        q = (
            q.groupby(["stimkey","outcome"])
            .agg(
                [  
                    pl.count().alias("count"),
                    (pl.col("stim_label").first()),
                ]
            ).sort(["stimkey","outcome"])
            )
        df = q.collect()
        return df
    
    def plot(self,ax:plt.Axes=None,bar_width=0.4,**kwargs):
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(15,10)))
            ax = self.fig.add_subplot(1,1,1)
        
        # do early first
        early_data = self.plot_data.filter(pl.col('outcome')==-1)
        ax = self.__plot__(ax,1,len(early_data),
                           width=bar_width,
                           color='r',linewidth=2,edgecolor='k')
        
        self.plot_data = self.make_agg_data(remove_early=True)
        
        uniq_k = self.plot_data['stimkey'].unique().to_numpy()
        
        label_dict = {1:'Early'}
        for i,k in enumerate(uniq_k,start=2):
            filt_df = self.plot_data.filter(pl.col('stimkey')==k)
             
            #0.5 because only 2 (correct and miss)
            bar_locs = [i-bar_width/2,i+bar_width/2]
            bars = filt_df['count'].to_list() #ordered by miss,correct
            
            label = filt_df[0,'stim_label']
            ax = self.__plot__(ax,bar_locs,bars,width=0.4,
                               color=['#333333','#32a852'],
                               linewidth=2,
                               edgecolor='k')

            label_dict[i] = label
  
        fontsize = kwargs.get('fontsize',25)
        ax.legend(loc='center left',bbox_to_anchor=(1,0.5),fontsize=fontsize-5,frameon=False)
        ax.set_ylabel('Counts', fontsize=fontsize)
        ax.set_xticks(list(label_dict.keys()))
        ax.set_xticklabels(list(label_dict.values()))
        ax.tick_params(labelsize=fontsize)
        ax.grid(alpha=0.5,axis='y')
        
        return ax
    
    
class DetectionResponseScatterPlotter(BasePlotter):
    __slots__ = ['stimkey','plot_data']
    def __init__(self, data: dict, stimkey:str=None, **kwargs):
        super().__init__(data, **kwargs)
        self.plot_data, self.stimkey = self.select_stim_data(self.data,stimkey)

    @staticmethod
    def __plot_scatter__(ax,t,times_arr,**kwargs):
        """ Plots the trial no and response times[blank time,answer time] by putting them on a scatter line """
        c = ['k'] 
        s = [10]
        if len(times_arr)==1:
            c = ['gainsboro']
        else:  
            if times_arr[0] < times_arr[1]:
                c.append('forestgreen') # correct
                
            else:
                c.append('orangered')
            s.append(20)
        t_arr = [t] * len(times_arr)
        ax.scatter(times_arr,t_arr,s=kwargs.get('s',s),c=c,alpha=0.7)
        return ax
    
    @staticmethod
    def __plot_density__(ax,x_bins,y_dens,**kwargs): 
        ax.plot(x_bins[1:],y_dens,c='dimgrey',alpha=0.8,linewidth=3,**kwargs) #right edges
        return ax
    
    def set_wrt_response_plot_data(self,wrt='sorted') -> np.ndarray:
        """ sets the plot data wrt to given argument and excludes nogo trials"""
        d = self.plot_data[self.stimkey]
        if wrt=='sorted':
            # add blank_time to correct answers 
            d['wrt_response_latency'] = d[['outcome','blank_time','response_latency']].apply(lambda x: x['response_latency']+x['blank_time'] if x['outcome']==1 else x['response_latency'],axis=1)
            
        elif wrt=='onset':
            d['wrt_response_latency'] = d[['outcome','blank_time','response_latency']].apply(lambda x: x['response_latency']-x['blank_time'],axis=1)
        else:
            raise ValueError(f'{wrt} is not a valid wrt value for response times')
        self.plot_data[self.stimkey] = d[d['outcome']!=0]
            
       
    def plot(self,ax:plt.Axes=None,bin_width:int=20,blanks:str='sorted',plt_range:list=None,**kwargs):
        if plt_range is None:
            # plt_range = [-100,self.plot_data['response_latency'].max()]
            plt_range = [-100,4900]
        
        bins = int((plt_range[1]-plt_range[0]) / bin_width)
        
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(8,8)))
            ax = self.fig.add_subplot(1,1,1)
            
        self.set_wrt_response_plot_data(wrt=blanks)
        times = self.plot_data[self.stimkey]['wrt_response_latency'].to_numpy()
        times_arr = []
        if blanks == 'sorted':
            sorted_data = self.plot_data[self.stimkey].sort_values('blank_time',ascending=False)
            for i,row in enumerate(sorted_data.itertuples()):
                times_arr = [row.blank_time, row.wrt_response_latency]
                ax = self.__plot_scatter__(ax,i,times_arr,**kwargs)
            x_label = 'Response Time (ms)'
        elif blanks == 'onset':
            for i,t in enumerate(times):
                times_arr = [0,t]
                ax = self.__plot_scatter__(ax,i,times_arr,**kwargs)
            x_label = 'Time from Stim Onset (ms)'
                
        ax_density = ax.inset_axes([0,0,1,0.1],frameon=False,sharex=ax)  
        
        hist,bins = np.histogram(times,bins=bins,range=plt_range)
        density = (hist / len(times)) / bin_width
        ax_density = self.__plot_density__(ax_density,bins,density,**kwargs) 
        
        fontsize = kwargs.get('fontsize',14)
        
        ax.set_ylim([-30,None])
        ax.set_yticks([i for i in range(len(self.plot_data[self.stimkey])) if i>=0 and i%50==0])
        ax.set_xlabel(x_label, fontsize=fontsize)
        ax.set_ylabel('Trial No.', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.grid(alpha=0.3,axis='x')
        
        ax.set_xlim(plt_range)
        ax_density.grid(alpha=0.3,axis='x')
        ax_density.tick_params(labelsize=fontsize)
        ax_density.set_yticks([])
        ax_density.set_yticklabels([])    
                
        return ax
        
        
class DetectionLickScatterPlotter(LickScatterPlotter):
    __slots__ = []
    def __init__(self, data, stimkey: str=None, **kwargs):
        super().__init__(data, stimkey, **kwargs)

    
    def plot(self,ax:plt.Axes=None,bin_width:int=20,wrt:str='reward',plt_range:list=None,**kwargs):
        if plt_range is None:
            plt_range = [-1000,1000]
        fontsize = kwargs.pop('fontsize',25)
               
        lick_data = self.plot_data.with_columns([(pl.when(pl.col('outcome')!=-1)
                                                  .then(pl.col('response_latency')+pl.col('blank_time'))
                                                  .otherwise(pl.col('response_latency'))).alias('blanked_response_latency'),
                                                 (pl.col('blank_time')+pl.col('response_latency')+pl.col('open_start_absolute')).alias('response_latency_absolute')])
            
        bins = int((plt_range[1]-plt_range[0]) / bin_width)
        
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.pop('figsize',(8,15)))
            ax = self.fig.add_subplot(1,1,1)
        
        if wrt == 'reward':
            lick_data = lick_data.drop_nulls(subset=['reward','lick'])
            x_label = 'Time from Reward (ms)'
            wrt_color = 'r'
            for i,row in enumerate(lick_data.iter_rows(named=True)):
                
                wrt_time = row['reward'][0]
                response_time = row['response_latency_absolute'] - wrt_time
                
                
                ax.scatter(response_time,row['trial_no'],
                           c='k',marker='|',s=20,zorder=2,
                           label='Reward*' if i==0 else '_')
                ax.axhspan(row['trial_no']-0.5,row['trial_no']+0.5,
                           color=self.color.stim_keys[row['stimkey']]['color'],
                           alpha=0.3)
                
                licks = np.array(row['lick']) - wrt_time
                ax = self.__plot_scatter__(ax,row['trial_no'],licks,**kwargs)   
                
        elif wrt == 'response':
            lick_data = lick_data.drop_nulls(subset=['lick'])
            x_label = 'Time from Response (ms)'
            wrt_color = 'k'
            for i,row in enumerate(lick_data.iter_rows(named=True)):
                
                wrt_time = row['response_latency_absolute']
                
                
                if row['reward'] is not None:
                    reward = row['reward'][0] - wrt_time
                else:
                    reward = 100 #potential reward time
                ax.scatter(100,row['trial_no'],c='r',
                           marker='|',s=20,zorder=2,
                           label='Reward*' if i==0 else '_')
                ax.axhspan(row['trial_no']-0.5,row['trial_no']+0.5,
                           color=self.color.stim_keys[row['stimkey']]['color'],
                           alpha=0.3)
                
                licks = np.array(row['lick']) - wrt_time
                ax = self.__plot_scatter__(ax,row['trial_no'],licks,**kwargs)   

        ax.axvline(0,c=wrt_color,linewidth=2,zorder=1)
    
        ax_density = ax.inset_axes([0,1,1,0.1],frameon=False,sharex=ax)
        
        pooled_licks = self.pool_licks(lick_data,wrt)
        
        hist,bins = np.histogram(pooled_licks,bins=bins,range=plt_range)
        density = (hist / len(pooled_licks)) / bin_width
        ax_density = self.__plot_density__(ax_density,bins,density,zorder=2,**kwargs) 
        
        ax_density.axvline(0,c=wrt_color,linewidth=2,zorder=1)
        
        
        ax.set_xlim(plt_range)
        ax.set_ylim([-30,None])
        # ax.set_yticks([i for i in range(len(lick_data)) if i>=0 and i%50==0])
        ax.set_xlabel(x_label, fontsize=fontsize)
        ax.set_ylabel('Trial No.', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.grid(alpha=0.3,axis='x')
        # ax.legend(loc='center left',bbox_to_anchor=(1,0.5),fontsize=fontsize,frameon=False)
        
        ax_density.set_xlim(plt_range)
        ax_density.grid(alpha=0.3,axis='x')
        ax_density.tick_params(labelsize=fontsize)
        # ax_density.set_yticks([])
        # ax_density.set_yticklabels([])
        ax_density.set_xticks([])
        ax_density.set_xticklabels([])

        return ax
    
    
class DetectionWheelTrajectoryPlotter(WheelTrajectoryPlotter):
    __slots__ = []
    def __init__(self, data: pl.DataFrame, stimkey:str=None,**kwargs) -> None:
        super().__init__(data, stimkey, **kwargs)

    def plot(self,ax:plt.Axes=None,
             time_lims:list=None,
             traj_lims:list=None,
             trace_type:str='sem',
             bin_width:int=None,**kwargs):
        
        ax = super().plot(time_lims=time_lims,
                          traj_lims=traj_lims,
                          trace_type=trace_type,
                          **kwargs)
        
        if bin_width is not None:
            bins = np.arange(0,time_lims[-1],bin_width,dtype='int')
            
            ax_density = ax.inset_axes([0,0,1,0.1],frameon=False,sharex=ax)
            
            pooled_licks = self.pool_trial_ends()
            
            hist,bins = np.histogram(pooled_licks,bins=bins,range=time_lims)
            ax_density = self.__plot_density__(ax_density,bins,hist,zorder=2,**kwargs)
            ax_density.set_yticks([])
            ax_density.set_yticklabels([])
        
        return ax

   
class DetectionSummaryPlotter:
    __slots__ = ['data','fig','plotters','stimkey']
    def __init__(self, data, stimkey:str=None,**kwargs):
        self.data = data # gets the stim data dict
        self.stimkey = stimkey
        self.fig = None
        self.init_plotters()
        
    def init_plotters(self):
        # TODO: Make this changable
        self.plotters = {'performance':DetectionPerformancePlotter(self.data, stimkey='all'),
                         'responsepertype':DetectionResponseTimeScatterCloudPlotter(self.data,self.stimkey),
                         'resptype':DetectionResponseTypeBarPlotter(self.data,stimkey='all'),
                         'licktotal':LickPlotter(self.data, stimkey='all'),
                         'resphist':DetectionResponseHistogramPlotter(self.data,stimkey='all'),
                         'respscatter':DetectionResponseScatterPlotter(self.data,stimkey='all'),
                         'lickdist':DetectionLickScatterPlotter(self.data,self.stimkey)}
    
    def plot(self,**kwargs):
        self.fig = plt.figure(figsize = kwargs.get('figsize',(20,10)))
        widths = [2,2,1]
        heights = [1,1]
        gs = self.fig.add_gridspec(ncols=3, nrows=2, 
                                   width_ratios=widths,height_ratios=heights,
                                   left=kwargs.get('left',0),right=kwargs.get('right',1),
                                   wspace=kwargs.get('wspace',0.3),hspace=kwargs.get('hspace',0.4))

        gs_in1 = gs[:,0].subgridspec(nrows=2,ncols=1,hspace=0.3)

        ax_perf = self.fig.add_subplot(gs_in1[1,0])
        ax_perf = self.plotters['performance'].plot(ax=ax_perf,seperate_by='contrast')
        
        ax_lick = ax_perf.twinx()
        ax_lick = self.plotters['licktotal'].plot(ax=ax_lick)
        ax_lick.grid(False)
        
        ax_resp = self.fig.add_subplot(gs_in1[0,0])
        ax_resp = self.plotters['resphist'].plot(ax=ax_resp)
        
        ax_resp2 = self.fig.add_subplot(gs[0,1])
        ax_resp2 = self.plotters['responsepertype'].plot(ax=ax_resp2)
        
        ax_type = self.fig.add_subplot(gs[0,2])
        ax_type = self.plotters['resptype'].plot(ax=ax_type)
        
        ax_scatter = self.fig.add_subplot(gs[1,1])
        ax_scatter = self.plotters['respscatter'].plot(ax=ax_scatter)
        
        ax_ldist = self.fig.add_subplot(gs[1,2])
        ax_ldist = self.plotters['lickdist'].plot(ax=ax_ldist)
        
        self.fig.tight_layout()
    
    
    def save(self,saveloc,date,animalid):
        if self.fig is not None:
            saveloc = pjoin(saveloc,'figures')
            if not os.path.exists(saveloc):
                os.mkdir(saveloc)
            savename = f'{date}_sessionSummary_{animalid}.pdf'
            saveloc = pjoin(saveloc,savename)
            self.fig.savefig(saveloc,bbox_inches='tight')
            display(f'Saved {savename} plot')