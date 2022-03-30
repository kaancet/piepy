from .plotter_utils import *
from os.path import join as pjoin
from ..wheel.wheelUtils import get_trajectory_avg

class BehaviorBasePlotter:
    __slots__ = ['cumul_data','summary_data','fig']
    def __init__(self,cumul_data:pd.DataFrame=None,summary_data:pd.DataFrame=None) -> None:
        self.cumul_data = cumul_data
        self.summary_data = summary_data
        self.fig = None
        set_style('analysis')


class BasePlotter:
    __slots__ = ['data','fig']
    def __init__(self,data:dict,**kwargs):
        self.data = self.set_data(data)
        self.fig = None
        # session data is a dict that has the stimulus types
        # this dictionary resides insides of another dict 
        # that is usually the novel_stim_data, or data dict inside session 
        set_style('analysis')

    def set_data(self,data:dict):
        if isinstance(data,dict):
            return data
        else:
            TypeError(f'The input session data should be a pandas DataFrame with keys corresponding to different stimulus types. Got {type(data)} instead')

    @staticmethod
    def select_stim_data(data: dict, stimkey: str=None) -> pd.DataFrame:
        """ Returns the selected stimulus type from session data
        data : Main session dictionary
        stimkey : Dictionary key that corresponds to the stimulus type (e.g. lowSF_highTF)
        """
        # error handling
        if stimkey is None and len(data.keys()) == 1:
            # if there is only one key just take that data in the dictionary
            key = list(data.keys())[0]
            return data[key],key
        elif stimkey is None and len(data.keys()) > 1:
            # if there are multiple keys and no stimkey selection, concat all the data and sort by trial_no
            data_append = []
            d = pd.DataFrame()
            for k,v in data.items():
                data_append.append(v)
            d = d.append(data_append,ignore_index=True)
            d = d.sort_values('trial_no')
            return d,stimkey
        elif stimkey not in data.keys():
            raise KeyError(f'{stimkey} not in stimulus data, try one of these: {list(data.keys())}')
        else:
            return data[stimkey],stimkey
    
    @staticmethod
    def threshold_responsetime(stim_data: pd.DataFrame, time_cutoff, cutoff_mode: str = 'low') -> pd.DataFrame:
        """ Returns the stimulus data filtered by the given response time
        stim_data: Stimulus data
        time_cutoff: Response time cutoff value(s)
        cutoff_mode: determines the filtering method, can  be low, mid, 
        """
        if time_cutoff is None:
            return stim_data
        
        mode_list = ['low','high','mid']
        if cutoff_mode not in mode_list:
            raise ValueError(f'The cutoff_mode argument needs to be one of {mode_list}, got {cutoff_mode} instead')
        if cutoff_mode == 'mid':
            try:
                t_low,t_high = time_cutoff
            except TypeError:
                display(f'The cutoff argument needs to be a list, got {type(time_cutoff)} instead')
            df_out = stim_data[(stim_data['response_latency'] <= t_high) & (stim_data['response_latency'] >= t_low)]
        elif cutoff_mode == 'low':
            df_out = stim_data[stim_data['response_latency'] <= time_cutoff]
        elif cutoff_mode == 'high':
            df_out = stim_data[stim_data['response_latency'] >= time_cutoff]
        
        if len(df_out) == 0:
            display('>>WARNING<< The filtered data has 0 elements! Check filtering values and/or the data')

        return df_out

    @staticmethod
    def threshold_trialinterval(stim_data: pd.DataFrame, trial_interval: list, interval_mode:str='count') -> pd.DataFrame:
        """ Returns the stimulus data in the given trial number interval
        stim_data: Stimulus data
        trial_interval: A list that has the ranges of the interval of trials
        interval_mode: Determines whether to use trial numbers or trial times for filtering
        """
        mode_list = ['count','time']
        if interval_mode not in mode_list:
            raise ValueError(f'The interval_mode argument needs to be one of {mode_list}, got {interval_mode} instead')

        if interval_mode == 'count':
            df_out = stim_data[(stim_data['trial_no'] >= trial_interval[0]) & (stim_data['trial_no'] <= trial_interval[1])]
        elif interval_mode == 'time':
            df_out = stim_data[(stim_data['openstart_absolute'] >= trial_interval[0]) & (stim_data['openstart_absolute'] <= trial_interval[1])]

        if len(df_out) == 0:
            display('>>WARNING<< The filtered data has 0 elements! Check filtering values and/or the data')

        return df_out

    def save(self,saveloc,date,animalid):
        if self.fig is not None:
            saveloc = pjoin(saveloc,'figures')
            if not os.path.exists(saveloc):
                os.mkdir(saveloc)
            savename = f'{date}_{self.__class__.__name__}_{animalid}.pdf'
            saveloc = pjoin(saveloc,savename)
            self.fig.savefig(saveloc)
            display(f'Saved {savename} plot')


class PerformancePlotter(BasePlotter):
    __slots__ = ['stimkey','plot_data']
    def __init__(self,data,stimkey:str=None,**kwargs):
        super().__init__(data, **kwargs)
        self.plot_data,self.stimkey = self.select_stim_data(self.data,stimkey)

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
        
        if seperate_by is not None:
            if seperate_by not in self.plot_data.columns:
                raise ValueError(f'Cannot seperate the data by {sepearte_by}')
            seperate_vals = nonan_unique(self.plot_data[seperate_by])
        else:
            seperate_vals = [-1] # dummy value

        for sep in seperate_vals:
            if seperate_by is not  None:
                data = self.plot_data[self.plot_data[seperate_by]==sep]
                y_axis = get_fraction(data['answer'].to_numpy(),fraction_of=1,window_size=20,min_period=10)
                color = contrast_styles[sep]['color']
            else:
                data = self.plot_data
                y_axis = data['fraction_correct']
                color = stim_styles[self.stimkey]['color']
            
            if plot_in_time:
                x_axis_ = data['openstart_absolute'] / 60000
                x_label_ = 'Time (mins)'
            else:
                x_axis_ = data['trial_no']
                x_label_ = 'Trial No'

            ax = self.__plot__(ax,x_axis_,y_axis,
                            color=color if self.stimkey is not None else 'forestgreen',
                            label=f'{self.stimkey}_{sep}' if self.stimkey is not None else 'all',
                            **kwargs)
            
        # prettify
        fontsize = kwargs.get('fontsize',14)
        ax.set_ylim([0, 100])
        ax.set_xlabel(x_label_, fontsize=fontsize)
        ax.set_ylabel('Accuracy(%)', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.grid(alpha=0.5,axis='both')
        if len(seperate_by)>1:
            ax.legend(loc='upper right',frameon=False,fontsize=fontsize)
        
        return ax
    
    
class ResponseTimePlotter(BasePlotter):
    __slots__ = ['stimkey','plot_data']
    def __init__(self,data,stimkey:str=None,**kwargs):
        super().__init__(data, **kwargs)
        self.plot_data,self.stimkey = self.select_stim_data(self.data,stimkey)

    @staticmethod
    def __plot__(ax,x,y,**kwargs):
        ax.plot(x, y,linewidth=kwargs.get('linewidth',5),**kwargs)
        return ax
        
    def plot(self, ax:plt.axes=None,plot_in_time:bool=False,**kwargs) -> plt.axes:
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(8,8)))
            ax = self.fig.add_subplot(1,1,1)
            
        if plot_in_time:
            x_axis_ = self.plot_data['openstart_absolute'] / 60000
            x_label_ = 'Time (mins)'
        else:
            x_axis_ = self.plot_data['trial_no']
            x_label_ = 'Trial No'
        
        ax = self.__plot__(ax,x_axis_,self.plot_data['running_response_latency']/1000,
                           color=stim_styles[self.stimkey]['color'] if self.stimkey is not None else 'royalblue',
                           label=self.stimkey if self.stimkey is not None else 'all',
                           **kwargs)

        # prettify
        fontsize = kwargs.get('fontsize',14)
        ax.set_yscale('log')
        ax.set_xlabel(x_label_, fontsize=fontsize)
        ax.set_ylabel('Response Time(sec)', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        
        ax.tick_params(axis='y', which='minor')
        minor_locs = np.append(np.linspace(0,0.9,9),np.linspace(1,9,9))
        ax.yaxis.set_minor_locator(plt.FixedLocator(minor_locs))
        ax.yaxis.set_minor_formatter(plt.FormatStrFormatter('%.1f'))

        ax.set_yticklabels([format(y,'.0f') for y in ax.get_yticks()])
        ax.grid(alpha=0.5,axis='both')
        
        return ax


class ResponseTimeScatterCloudPlotter(BasePlotter):
    __slots__ = ['stimkey','plot_data']
    def __init__(self, data, stimkey:str=None, **kwargs):
        super().__init__(data=data, **kwargs)
        self.plot_data, self.stimkey = self.select_stim_data(self.data,stimkey)
        self.plot_data = self.threshold_responsetime(self.plot_data,kwargs.get('cutoff'))   
    
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
    def __plot__(ax,contrast,time,median,mean,pos,cloud_width,**kwargs):
        
        ax.scatter(contrast,time,alpha=0.6,**kwargs)
        
        ax.plot([pos-cloud_width/2,pos+cloud_width/2],[median,median],linewidth=3,c='b')
        ax.plot([pos-cloud_width/2,pos+cloud_width/2],[mean,mean],linewidth=3,c='k')
                   
        #elements is returned to be able to modify properties of plot elements outside(e.g. color)

        return ax
    
    def plot(self,ax:plt.Axes=None,cloud_width=0.33,**kwargs):
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(15,10)))
            ax = self.fig.add_subplot(1,1,1)
        signed_contrasts = [] 
        plot_pos = []
        for i,c in enumerate(np.unique(self.plot_data['contrast']),start=1):
            contrast_data = self.plot_data[self.plot_data['contrast']==c]
            for side in np.unique(contrast_data['stim_side']):
                # location of dot cloud centers by contrast
                if c == 0:
                    side_data = contrast_data
                    cpos = 0
                else:
                    side_data = contrast_data[contrast_data['stim_side']==side]
                    cpos = np.sign(side) * i
                plot_pos.append(cpos)
                signed_contrasts.append(np.sign(side)*c)   
                response_times = self.time_to_log(side_data['response_latency'].to_numpy())
                x_dots,y_dots = self.make_dot_cloud(response_times,cpos,cloud_width)
                median = np.median(response_times)
                mean = np.mean(response_times)
                
                ax = self.__plot__(ax,x_dots,y_dots,median,mean,cpos,cloud_width,
                                   s=30,color=stim_styles[self.stimkey]['color'] if self.stimkey is not None else 'royalblue',
                                   label=self.stimkey if self.stimkey is not None else 'all',
                                   **kwargs)

        # mid line
        ax.set_ylim([90,30000])
        ax.plot([0,0],ax.get_ylim(),color='gray',linewidth=2,alpha=0.5)
        
        fontsize = kwargs.get('fontsize',14)
        ax.set_xlabel('Stimulus Contrast', fontsize=fontsize)
        ax.set_ylabel('Response Time (ms)', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.tick_params(axis='x')
        
        ax.set_yscale('symlog')
        minor_locs = [200,400,600,800,2000,4000,6000,8000]
        ax.yaxis.set_minor_locator(plt.FixedLocator(minor_locs))
        ax.yaxis.set_minor_formatter(plt.FormatStrFormatter('%d'))
        ax.yaxis.set_major_locator(ticker.FixedLocator([100,1000,10000]))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        
        if 0 not in plot_pos:
            plot_pos.append(0)
            signed_contrasts.append(0)
            
        ax.xaxis.set_major_locator(ticker.FixedLocator(np.sort(plot_pos)))
        ax.xaxis.set_major_formatter(ticker.FixedFormatter([str(int(100*c)) for c in np.sort(signed_contrasts)]))
            
        ax.grid(alpha=0.5,axis='y')
        
        return ax   
        

class ResponseTimeHistogramPlotter(BasePlotter):
    __slots__ = ['stimkey','plot_data']
    def __init__(self, data, stimkey:str=None, **kwargs):
        super().__init__(data=data, **kwargs)
        self.plot_data, self.stimkey = self.select_stim_data(self.data, stimkey)
        
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
        color = ['orangered' if i<=150 else 'forestgreen' for i in bins]    
        
        ax.bar(bins[1:],counts,width=bar_width,color=color,**kwargs)
        ax.axvline(x=0,color='k',linestyle=':',linewidth=3,**kwargs)
        
        return ax
    
    
class ResponseTypeBarPlotter(BasePlotter):
    __slots__ = ['stimkey','plot_data']
    def __init__(self, data, stimkey:str=None, **kwargs):
        super().__init__(data=data, **kwargs)
        self.plot_data, self.stimkey = self.select_stim_data(self.data,stimkey)
        
    @staticmethod
    def position_bars(bar_loc,bar_count,bar_width,padding):
        if bar_count == 1:
            return bar_loc
        else:
            locs = []
            mult = np.sign(np.linspace(-1,1,bar_count))
            for m in mult:
                locs.append(bar_loc + (m * bar_width * (bar_count-1)/2) + (m * padding * (bar_count-2)))
            return locs
        
    @staticmethod    
    def __plot__(ax,x_locs,bar_heights,**kwargs):
        ax.bar(x_locs,bar_heights,**kwargs)
        return ax


class LickPlotter(BasePlotter):
    __slots__ =['stimkey','plot_data']
    def __init__(self, data: dict, stimkey:str=None, **kwargs):
        super().__init__(data, **kwargs)
        self.plot_data, self.stimkey = self.select_stim_data(self.data, stimkey)
        
    def pool_licks(self):
        all_lick = np.array([]).reshape(-1,2)
        for row in self.plot_data.itertuples():
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
                x_axis_ = self.plot_data['trial_no']
                all_licks[:,0] = (all_licks[:,0]/np.max(all_licks[:,0])) * self.plot_data['trial_no'].iloc[-1]
                y_axis_ = np.interp(self.plot_data['trial_no'],all_licks[:,0],all_licks[:,1])
                x_label_ = 'Trial No'
            
            ax = self.__plot__(ax, x_axis_,y_axis_,**kwargs)
        
        else:
            display('No Lick data found for session :(')
            ax = self.__plot__(ax,[],[])
            
        fontsize = kwargs.get('fontsize',14)
        ax.set_xlabel(x_label_, fontsize=fontsize)
        ax.set_ylabel('Lick Counts', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.tick_params(axis='x')
        ax.grid(alpha=0.5,axis='y')
        
        return ax
    
    
class LickScatterPlotter(BasePlotter):
    """ Lick scatters for each trial, wrt reward or response time"""
    __slots__ =['stimkey','plot_data']
    def __init__(self, data: dict, stimkey:str=None, **kwargs):
        super().__init__(data, **kwargs)
        self.plot_data, self.stimkey = self.select_stim_data(self.data, stimkey)
    
    @staticmethod
    def __plot_scatter__(ax,t,lick_arr,**kwargs):
        
        t_arr = [t] * len(lick_arr)
        
        ax.scatter(lick_arr,t_arr,marker='|',c='aqua',s=kwargs.get('s',20),**kwargs)
        
        return ax
    
    def pool_licks(self,wrt:str='reward'):
        pooled_lick = np.array([])
        error_ctr = []
        for row in self.plot_data[self.plot_data['answer']==1].itertuples():
            if len(row.lick):
                if wrt=='reward':
                    try:
                        wrt_time = row.reward[0][0]
                    except:
                        error_ctr.append(row.trial_no)
                        display(f'\n!!!!!! NO REWARD IN CORRECT TRIAL, THIS IS A VERY SERIOUS ERROR! SOLVE THIS ASAP !!!!!!\n')
                elif wrt=='response':
                    wrt_time = row.response_latency_absolute
                pooled_lick = np.append(pooled_lick,row.lick[:,0] - wrt_time)
        print(f'Trials with reward issue: {error_ctr}')         
        return pooled_lick
    
    @staticmethod
    def __plot_density__(ax,x_bins,y_dens,**kwargs): 
        ax.plot(x_bins[1:],y_dens,c='aqua',alpha=0.8,linewidth=3,**kwargs) #right edges
        return ax
    
    
class WheelTrajectoryPlotter(BasePlotter):
    __slots__ = ['stimkey','plot_data']
    def __init__(self, data: dict, stimkey:str=None,**kwargs):
        super().__init__(data, **kwargs)
        self.plot_data, self.stimkey = self.select_stim_data(self.data,stimkey)
         
    @staticmethod
    def __plot__(ax:plt.Axes,wheel_t,wheel_pos,**kwargs):
         ax.plot(wheel_t, wheel_pos,
                 linewidth=kwargs.get('linewidth',5),
                 alpha=1,
                 zorder=2,
                 **kwargs)
         return ax
    
    def plot(self,ax:plt.Axes=None,seperate_by:str='contrast',plot_range:list=None,**kwargs):
        if seperate_by not in self.plot_data.columns:
            raise ValueError(f'{seperate_by} is not a valid field for this data. try: {self.plot_data.columns}')
        
        if plot_range is None:
            plot_range = [-200,1500]
        
        if ax is None:
            self.fig = plt.figure(figsize=(8,14))
            ax = self.fig.add_subplot(1,1,1)
        
        seperator_list = np.unique(self.plot_data[seperate_by])
        print(seperator_list)
        
        sides = np.unique(self.plot_data['stim_side'])
        
        for i,side in enumerate(sides,start=1):
            side_slice = self.plot_data[self.plot_data['stim_side'] == side]
            
            for sep in seperator_list:
                seperator_slice = side_slice[side_slice[seperate_by] == sep]
                
                # shift wheel according to side
                # wheel_arr = seperator_slice['wheel'].apply(lambda x: x+side)
                # seperator_slice.loc[:,'wheel'] = seperator_slice.loc[:,'wheel'] + s

                avg = get_trajectory_avg(seperator_slice['wheel'].to_numpy())
                
                if avg is not None:
                    avg = avg[find_nearest(avg[:,0],plot_range[0])[0]:find_nearest(avg[:,0],plot_range[1])[0]]
                    c = contrast_styles[sep]['color']
                    ax = self.__plot__(ax,avg[:,1]+side,avg[:,0],
                                       color=c,
                                       label=sep if i==1 else '_',
                                       **kwargs)
        
        ax.set_ylim(plot_range)
        ax.set_xlim([-75, 75])
        # closed loop start line
        ax.plot(ax.get_xlim(),[0,0],'k',linewidth=2, alpha=0.8)

        # trigger zones
        ax.plot([0,0], ax.get_ylim(), 'green', linestyle='--', linewidth=2,alpha=0.8)

        ax.plot([-50,-50], ax.get_ylim(), 'maroon', linestyle='--', linewidth=2,alpha=0.8)
        ax.plot([50,50], ax.get_ylim(), 'maroon', linestyle='--', linewidth=2,alpha=0.8)
        
        # make it pretty
        fontsize = kwargs.get('fontsize',20)
        ax.yaxis.set_label_position('right')
        ax.yaxis.tick_right()
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        ax.set_xlabel('Wheel Position (deg)', fontsize=fontsize)
        ax.set_ylabel('Time(ms)', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.grid(axis='y')
        ax.legend(frameon=False,fontsize=14)

        return ax