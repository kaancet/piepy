from .plotter_utils import *
from os.path import join as pjoin
from ..wheel.wheelUtils import get_trajectory_avg
from scipy import stats
import copy
import matplotlib.patheffects as pe
from behavior_python.detection.wheelDetectionAnalysis import DetectionAnalysis


class BasePlotter:
    __slots__ = ['data','fig','color']
    def __init__(self,data:dict,**kwargs):
        self.data = data
        self.fig = None
        set_style('analysis')
        self.color = Color()
    
    @staticmethod
    def select_stim_data(data_in:pl.DataFrame, stimkey:str=None) -> dict:
        """ Returns the selected stimulus type from session data
            data_in : 
            stimkey : Dictionary key that corresponds to the stimulus type (e.g. lowSF_highTF)
        """
        # drop early trials
        data = data_in.filter(pl.col('answer')!=-1)
        
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
                x_axis_ = data2plot['trial_no']
                x_label_ = 'Trial No'

            ax = self.__plot__(ax,x_axis_.to_numpy(),y_axis,
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
    __slots__ = ['stimkey','plot_data']
    def __init__(self,data,stimkey:str=None,**kwargs):
        super().__init__(data, **kwargs)
        self.plot_data,self.stimkey = self.select_stim_data(self.data,stimkey)
        self.color.check_stim_colors(self.plot_data.keys())
        
        c_list = nonan_unique(self.plot_data[list(self.plot_data.keys())[0]]['contrast'])
        self.color.check_contrast_colors(c_list)
        
    @staticmethod
    def __plot__(ax,x,y,**kwargs):
        ax.plot(x, y,linewidth=kwargs.get('linewidth',5),**kwargs)
        return ax
        
    def plot(self, ax:plt.axes=None,plot_in_time:bool=False,**kwargs) -> plt.axes:
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(8,8)))
            ax = self.fig.add_subplot(1,1,1)
            if 'figsize' in kwargs:
                kwargs.pop('figsize')
            
        if plot_in_time:
            if self.stimkey is not None:
                x_axis_ = self.plot_data[self.stimkey]['openstart_absolute'] / 60000
            x_label_ = 'Time (mins)'
        else:
            if self.stimkey is not None:
                x_axis_ = self.plot_data[self.stimkey]['trial_no']
            x_label_ = 'Trial No'
        
        ax = self.__plot__(ax,x_axis_,self.plot_data[self.stimkey]['running_response_latency'],
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


class ResponseTimeScatterCloudPlotter(BasePlotter):
    def __init__(self, data, stimkey:str=None, **kwargs):
        super().__init__(data=data, **kwargs)
        self.plot_data, self.stimkey = self.select_stim_data(self.data,stimkey)
        self.color.check_stim_colors(self.plot_data.keys())
        
        c_list = nonan_unique(self.plot_data[list(self.plot_data.keys())[0]]['contrast'])
        self.color.check_contrast_colors(c_list)
        self.plot_data = self.threshold_responsetime(self.plot_data,kwargs.get('cutoff')) 
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
    def __plot__(ax,contrast,time,median,mean,pos,cloud_width,**kwargs):
        
        if 'fontsize' in kwargs.keys():
            kwargs.pop('fontsize')
        ax.scatter(contrast,time,alpha=0.6,**kwargs)
        
        #median
        ax.plot([pos-cloud_width/2,pos+cloud_width/2],[median,median],linewidth=3,
                c=kwargs.get('color','b'),path_effects=[pe.Stroke(linewidth=6, foreground='k'), pe.Normal()])
        
        #mean
        # ax.plot([pos-cloud_width/2,pos+cloud_width/2],[mean,mean],linewidth=3,
        #         c=kwargs.get('color','k'),path_effects=[pe.Stroke(linewidth=6, foreground='k'), pe.Normal()])
                   
        #elements is returned to be able to modify properties of plot elements outside(e.g. color)

        return ax
    
    @staticmethod
    def add_jitter_to_misses(resp_times,jitter_lims=[0,100]):
        """ Adds jitter in y-dimension to missed trial dot"""
        
        miss_locs = np.where(resp_times>=1000)[0]
        jitter = np.random.choice(np.arange(jitter_lims[0],jitter_lims[1]),len(miss_locs),replace=True)
        
        resp_times[miss_locs] = resp_times[miss_locs] + jitter
        return resp_times
        
    
    def plot(self,ax:plt.Axes=None,cloud_width=0.33,plot_misses:bool=False,**kwargs):
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(15,10)))
            ax = self.fig.add_subplot(1,1,1)
            if 'figsize' in kwargs:
                kwargs.pop('figsize')
        
        signed_contrasts = [] 
        plot_pos = []
        resp_times_dict = defaultdict(dict)
        
        for si,skey in enumerate(self.plot_data.keys()):
            stim_data = self.plot_data[skey]
            has_zero = 0
            for i,c in enumerate(np.unique(stim_data['contrast']),start=1):
                contrast_data = stim_data[stim_data['contrast']==c]
                for j,side in enumerate(np.unique(contrast_data['stim_side'])):
                    # location of dot cloud centers by contrast
                    if c == 0:
                        side_data = contrast_data
                        cpos = 0
                        has_zero = 1
                    else:
                        side_data = contrast_data[contrast_data['stim_side']==side]
                        cpos = np.sign(side) * i if has_zero else np.sign(side)*i+1
                    
                    plot_pos.append(cpos)
                    signed_contrasts.append(np.sign(side)*c)
                    if plot_misses:
                        data2plot = side_data[side_data['answer']!=-1]
                    else:
                        data2plot = side_data[side_data['answer']==1]
                        
                    resp_times_dict[cpos][skey] = side_data['response_latency'].to_numpy()

                    response_times = self.time_to_log(data2plot['response_latency'].to_numpy())
                    response_times = self.add_jitter_to_misses(response_times)
                    
                    x_dots,y_dots = self.make_dot_cloud(response_times,cpos,cloud_width)
                    median = np.median(response_times)
                    mean = np.mean(response_times)
                    print(skey,side,c,median)
                    ax = self.__plot__(ax,x_dots,y_dots,median,mean,cpos,cloud_width,
                                    color=self.color.stim_keys[skey]['color'] if skey!='all' else 'k',
                                    label=skey if skey is not None and j==0 and c==0.125 else '_',
                                    **kwargs)
        
        
        if len(self.plot_data.keys())>1:
            non_opto_keys = [k for k in self.plot_data.keys() if 'opto' not in k]
            opto_keys = [k for k in self.plot_data.keys() if 'opto' in k]
            
            opto_nonopto_pairs = {non_opto_keys[i]:opto_keys[i] for i in range(len(non_opto_keys)) if non_opto_keys[i] in opto_keys[i]}
        
            # add p values
            for cpos,c_d in resp_times_dict.items():
                if len(c_d) < 2:
                    continue #skip contrasts with single stim data type
                
                for non_k,opto_k in opto_nonopto_pairs.items():
                    non_opto_rt = c_d[non_k]
                    if opto_k in c_d.keys():
                        opto_rt = c_d[opto_k]
                        p = self.stat_analysis.get_pvalues_nonparametric(non_opto_rt,opto_rt)
                    else:
                        continue
                    stars = ''
                    if p < 0.001:
                        stars = '***'
                    elif 0.001 < p < 0.01:
                        stars = '**'
                    elif 0.01 < p < 0.05:
                        stars = '*'
                    
                    ax.text(cpos, 2000+i*200, stars,color=self.color.stim_keys[opto_k]['color'], fontsize=30)
        
        # mid line
        ax.set_ylim([90,3000])
        ax.plot([0,0],ax.get_ylim(),color='gray',linewidth=2,alpha=0.5)
        
        fontsize = kwargs.get('fontsize',20)
        ax.set_xlabel('Stimulus Contrast', fontsize=fontsize)
        ax.set_ylabel('Response Time (ms)', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        
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
        ax.legend(loc='upper left',fontsize=fontsize,frameon=False)
        
        return ax    
        

class ResponseTimeHistogramPlotter(BasePlotter):
    __slots__ = ['stimkey','plot_data']
    def __init__(self, data, stimkey:str=None, **kwargs):
        super().__init__(data=data, **kwargs)
        self.plot_data, self.stimkey = self.select_stim_data(self.data, stimkey)
        self.color.check_stim_colors(self.plot_data.keys())
        
        c_list = nonan_unique(self.plot_data[list(self.plot_data.keys())[0]]['contrast'])
        self.color.check_contrast_colors(c_list)
        
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
        self.color.check_stim_colors(self.plot_data.keys())
        
        c_list = nonan_unique(self.plot_data[list(self.plot_data.keys())[0]]['contrast'])
        self.color.check_contrast_colors(c_list)
        
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
    __slots__ =['stimkey','plot_data']
    def __init__(self, data: dict, stimkey:str=None, **kwargs):
        super().__init__(data, **kwargs)
        self.plot_data, self.stimkey = self.select_stim_data(self.data, stimkey)
        self.color.check_stim_colors(self.plot_data.keys())
        
        c_list = nonan_unique(self.plot_data[list(self.plot_data.keys())[0]]['contrast'])
        self.color.check_contrast_colors(c_list)
    
    @staticmethod
    def __plot_scatter__(ax,t,lick_arr,**kwargs):
        
        t_arr = [t] * len(lick_arr)
        
        ax.scatter(lick_arr,t_arr,marker='|',c='aqua',s=kwargs.get('s',20),**kwargs)
        
        return ax
    
    def pool_licks(self,wrt:str='reward'):
        pooled_lick = np.array([])
        error_ctr = []
        for row in self.plot_data[self.stimkey][self.plot_data[self.stimkey]['answer']==1].itertuples():
            if len(row.lick):
                if wrt=='reward':
                    try:
                        wrt_time = row.reward[0]
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
    __slots__ = ['stimkey','plot_data','side_sep_dict']
    def __init__(self, data: dict, stimkey:str=None,**kwargs):
        super().__init__(data, **kwargs)
        self.plot_data, self.stimkey = self.select_stim_data(self.data,stimkey)
        self.color.check_stim_colors(self.plot_data.keys())
        
        c_list = nonan_unique(self.plot_data[list(self.plot_data.keys())[0]]['contrast'])
        self.color.check_contrast_colors(c_list)
        
         
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
     
    def plot(self,ax:plt.Axes=None,plot_range_time:list=None,plot_range_trj:list=None,orientation:str='vertical',**kwargs):
        
        if plot_range_time is None:
            plot_range_time = [-200,1500]
        if plot_range_trj is None:
            plot_range_trj = [-75,75]
        
        if ax is None:
            if orientation=='vertical':
                self.fig = plt.figure(figsize=(8,14))
            else:
                self.fig = plt.figure(figsize=(14,8))
            ax = self.fig.add_subplot(1,1,1)
            if 'figsize' in kwargs:
                kwargs.pop('figsize')
        
        for side,side_stats in self.side_sep_dict.items():
            
            for i,sep in enumerate(side_stats.keys()):
                sep_stats = side_stats[sep]
                avg = sep_stats['avg']
                sem = sep_stats['sem']
                
                if avg is not None:
                    avg = avg[find_nearest(avg[:,0],plot_range_time[0])[0]:find_nearest(avg[:,0],plot_range_time[1])[0]]
                    sem = sem[find_nearest(sem[:,0],plot_range_time[0])[0]:find_nearest(sem[:,0],plot_range_time[1])[0]]
                    
                    c = self.color.contrast_keys[str(sep)]['color']
                    if orientation=='vertical':
                        wheel_x = avg[:,1]+side
                        wheel_y = avg[:,0]
                        sem_plus = wheel_x + sem[:,1]
                        sem_minus = wheel_x - sem[:,1]
                        
                        ax.fill_betweenx(sem_plus,sem_minus,sem[:,0],
                                    alpha=0.2,
                                    color=c,
                                    linewidth=0)
                    else:
                        wheel_x = avg[:,0]
                        wheel_y = avg[:,1]+side
                        sem_plus = wheel_y + sem[:,1]
                        sem_minus = wheel_y - sem[:,1]
                        
                        ax.fill_between(sem[:,0],sem_plus,sem_minus,
                                    alpha=0.2,
                                    color=c,
                                    linewidth=0)
                        
                    ax = self.__plot__(ax,wheel_x,wheel_y,
                                    color=c,
                                    label=sep if side>=0 else '_', #only put label for 0 and right side(where opto is mostly present)
                                    **kwargs)
                    

                    
        
        fontsize = kwargs.get('fontsize',20)
        if orientation=='vertical':
            ax.set_ylim(plot_range_time)
            ax.set_xlim(plot_range_trj)
            # closed loop start line
            ax.plot(ax.get_xlim(),[0,0],'k',linewidth=2, alpha=0.8)

            ax.set_xlabel('Wheel Position (deg)', fontsize=fontsize)
            ax.set_ylabel('Time(ms)', fontsize=fontsize)
            ax.yaxis.set_label_position('right')
            ax.yaxis.tick_right()
        else:
            ax.set_xlim(plot_range_time)
            ax.set_ylim(plot_range_trj)
            
            # closed loop start line
            ax.plot([0,0],ax.get_ylim(),'k',linewidth=2, alpha=0.8)

            ax.set_ylabel('Wheel Position (deg)', fontsize=fontsize)
            ax.set_xlabel('Time(ms)', fontsize=fontsize)
        
        # make it pretty
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
   
        ax.tick_params(labelsize=fontsize)
        ax.grid(axis='y')
        ax.legend(frameon=False,fontsize=14)

        return ax