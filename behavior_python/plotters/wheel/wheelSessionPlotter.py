from ..basePlotters import *
from ...wheel.wheelAnalysis import WheelCurve
from scipy.stats import gaussian_kde


class WheelPsychometricPlotter(BasePlotter):
    __slots__ = ['fitters']
    def __init__(self, data: dict,**kwargs):
        super().__init__(data, **kwargs)
        self.fitters = {}
        # for psychometric we need different stimulus types in different dict keys
        if kwargs.get('load_params',False):
            pass
        else:
            self.fit_data()
        
    def fit_data(self):
        """ Fits data for each stimkey in the data dict"""
        for k,v in self.data.items():
            temp_curve = WheelCurve(data=v)
            temp_curve.fit_curve()
            self.fitters[k] = temp_curve
            
    @staticmethod
    def __plot__(ax,x,y,fitted_curve,err,**kwargs):
        """ Private function that plots a psychometric curve with the given 
        x,y and err values are used to plot the points and 
        x_fit and y_fit values are used to plot the fitted curve
        """
        ax.plot([0, 0], [0, 1], 'gray', linestyle=':', linewidth=2,alpha=0.7)
        ax.plot([-100, 100], [0.5, 0.5], 'gray', linestyle=':', linewidth=2,alpha=0.7)

        ax.errorbar((100 * x), y, err,
                    marker=kwargs.get('marker','o'),
                    linewidth=0,
                    markersize=kwargs.get('markersize',15),
                    markeredgecolor=kwargs.get('markeredgecolor','w'),
                    markeredgewidth=kwargs.get('markeredgewidth',2),
                    elinewidth=kwargs.get('elinewidth',3),
                    capsize=kwargs.get('capsize',0),
                    **kwargs)

        ax.plot(100 * fitted_curve[:,0], fitted_curve[:,1],
                linewidth=kwargs.get('linewidth',9),**kwargs)
        return ax
        
    def plot(self,ax:plt.Axes=None,color=None,**kwargs):
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(8,8)))
            ax = self.fig.add_subplot(1,1,1)
            
        for k,v in self.fitters.items():
            if color is None:
                color = stim_styles[k]['color']
            ax = self.__plot__(ax,v.signed_contrast,v.percentage,
                               v.fitted_curve,v.confidence,
                               color=color,
                               label=k,
                               **kwargs)

        # prettify
        fontsize = kwargs.get('fontsize',14)
        ax.set_xlabel('Contrast Value', fontsize=fontsize)
        ax.set_ylabel('Proability Choosing Right',fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.spines['left'].set_bounds(0, 1)
        ax.spines['bottom'].set_bounds(-100, 100)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        return ax


class WheelPerformancePlotter(PerformancePlotter):
    def __init__(self,data,stimkey:str,**kwargs):
        super().__init__(data, stimkey, **kwargs)
        self.modify_data()

    def modify_data(self,*args,**kwargs):
        # change the self.plot_data here with methods if need to plot something else
        pass

    # def plot(self, ax:plt.axes=None,*args,**kwargs):
    # override the plot function calling __plot__ 
    # <your code here>
    # self.__plot__(x,y,ax)
    

class WheelResponseTimePlotter(ResponseTimePlotter):
    def __init__(self,data,stimkey:str,**kwargs):
        super().__init__(data, stimkey, **kwargs)
        self.modify_data()

    def modify_data(self,*args,**kwargs):
        # change the self.plot_data here with methods if need to plot something else
        pass

    # def plot(self, ax:plt.axes=None,*args,**kwargs):
    # override the plot function calling __plot__ 
    # <your code here>
    # self.__plot__(x,y,ax)


class WheelResponseTimeScatterCloudPlotter(ResponseTimeScatterCloudPlotter):
    def __init__(self,data,stimkey:str=None,**kwargs):
        super().__init__(data, stimkey, **kwargs)
        self.modify_data()

    def modify_data(self,*args,**kwargs):
        # change the self.plot_data here if need to plot something else
        pass
        

class WheelResponseHistogramPlotter(ResponseTimeHistogramPlotter):
    def __init__(self, data, stimkey:str=None, **kwargs):
        super().__init__(data, stimkey, **kwargs)
        self.modify_data()
        
    def modify_data(self,*args,**kwargs):
        # change the self.plot_data here if need to plot something else
        pass
    
    def plot(self,bin_width=100,ax:plt.Axes=None,**kwargs):
        
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(15,10)))
            ax = self.fig.add_subplot(1,1,1)
        
        resp_times = self.plot_data['response_latency'].to_numpy()
        
        counts,bins = self.bin_times(resp_times,bin_width)
        ax = self.__plot__(ax,counts,bins)
            
        fontsize = kwargs.get('fontsize',22)
        ax.set_xlabel('Time from Stimulus onset (ms)', fontsize=fontsize)
        ax.set_ylabel('Counts', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.tick_params(axis='x')
        ax.grid(alpha=0.5,axis='y')
        
        return ax
    

class WheelResponseTypeBarPlotter(ResponseTypeBarPlotter):
    __slots__ = []
    def __init__(self, data, stimkey: str=None, **kwargs):
        super().__init__(data, stimkey, **kwargs)
        self.modify_data()
        
    def modify_data(self):
        pass
    
    def plot(self,ax:plt.Axes=None,padding=0.8,**kwargs) -> plt.Axes:
        if ax is None:
            self.fig = plt.figure(figsize=kwargs.get('figsize',(15,10)))
            ax = self.fig.add_subplot(111)
            
        for answer in np.unique(self.plot_data['answer']):
            answer_data = self.plot_data[self.plot_data['answer']==answer]
            counts = [len(answer_data[answer_data['stim_side']<0]),
                      len(answer_data[answer_data['stim_side']>0])]
            
            locs = self.position_bars(answer,len(counts),0.25,padding=padding)
            
            ax = self.__plot__(ax,locs,counts,width=0.25,
                                 color=['#630726','#32a852'],
                                 linewidth=2,
                                 edgecolor='k',
                                 **kwargs)
            
        fontsize = kwargs.get('fontsize',14)
        ax.set_ylabel('Counts', fontsize=fontsize)
        ax.set_xticks([-1,0,1])
        ax.set_xticklabels(['Incorrect','NoGo','Correct'])
        ax.tick_params(labelsize=fontsize)
        ax.grid(alpha=0.5,axis='y')
        
        return ax


class WheelLickScatterPlotter(LickScatterPlotter):
    __slots__ = []
    def __init__(self, data, stimkey: str=None, **kwargs):
        super().__init__(data, stimkey, **kwargs)
        self.modify_data()
        
    def modify_data(self,*args,**kwargs):
        # change the self.plot_data here if need to plot something else
        self.plot_data['response_latency_absolute'] = self.plot_data['response_latency'] + self.plot_data['openstart_absolute']
    
    
    def plot(self,ax:plt.Axes=None,bin_width:int=20,wrt:str='reward',plt_range:list=None,**kwargs):
        if plt_range is None:
            plt_range = [-1000,1000]
            
        bins = int((plt_range[1]-plt_range[0]) / bin_width)
        
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(8,8)))
            ax = self.fig.add_subplot(1,1,1)
            
        for row in self.plot_data[self.plot_data['answer']==1].itertuples():
            if len(row.reward):
                if len(row.lick):
                    if wrt == 'reward':
                        wrt_time = row.reward[0][0]
                        response_time = row.response_latency_absolute - row.reward[0][0]
                        x_label = 'Time from Reward (ms)'
                        wrt_color = 'r'
                        ax.scatter(response_time,row.trial_no,c='k',marker='|',s=20,zorder=2)
                    elif wrt == 'response':
                        wrt_time = row.response_latency_absolute
                        x_label = 'Time from Response (ms)'
                        wrt_color = 'k'
                        reward = row.reward[0][0] - row.response_latency_absolute
                        ax.scatter(reward,row.trial_no,c='r',marker='|',s=20,zorder=2)
                    
                    licks = row.lick[:,0] - wrt_time
                    ax = self.__plot_scatter__(ax,row.trial_no,licks,**kwargs)   

        ax.axvline(0,c=wrt_color,linewidth=2,zorder=1)
    
        ax_density = ax.inset_axes([0,0,1,0.1],frameon=False,sharex=ax)
        
        pooled_licks = self.pool_licks(wrt)
        hist,bins = np.histogram(pooled_licks,bins=bins,range=plt_range)
        density = (hist / len(pooled_licks)) / bin_width
        ax_density = self.__plot_density__(ax_density,bins,density,zorder=2,**kwargs) 
        
        ax_density.axvline(0,c=wrt_color,linewidth=2,zorder=1)
        
        fontsize = kwargs.get('fontsize',22)
        ax.set_xlim(plt_range)
        ax.set_ylim([-30,None])
        ax.set_yticks([i for i in range(len(self.plot_data)) if i>=0 and i%50==0])
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
    

class WheelWheelTrajectoryPlotter(WheelTrajectoryPlotter):
    __slots__ = []
    def __init__(self, data: dict, stimkey: str = None, **kwargs):
        super().__init__(data, stimkey, **kwargs)
        
    def modify_data(self):
        pass
            

class WheelSummaryPlotter:
    __slots__ = ['data','fig','plotters','stimkey','cutoff_time']
    def __init__(self, data:dict, stimkey:str=None,cutoff_time:int=None,**kwargs) -> None:
        self.data = data # gets the stim data dict
        self.stimkey = stimkey
        self.cutoff_time = cutoff_time
        self.fig = None
        
        self.init_plotters()
    
    def apply_cutoff(self) -> dict:
        out_data = {}
        if self.cutoff_time is not None:
            print(f'Applying {self.cutoff_time} second cutoff to data')
            for k,data in self.data.items():
                temp = data[data['response_latency'] <= self.cutoff_time]
                out_data[k] = temp.copy()
            return out_data
        else:
            return self.data
    
    def init_plotters(self):
        cutoff_data = self.apply_cutoff()
        # TODO: Make this changable
        self.plotters = {'performance':WheelPerformancePlotter(self.data, self.stimkey),
                         'responsepertype':WheelResponseTimeScatterCloudPlotter(cutoff_data,self.stimkey),
                         'curve':WheelPsychometricPlotter(cutoff_data),
                         'responsetime':WheelResponseTimePlotter(self.data,self.stimkey),
                         'perfpertype':WheelResponseTypeBarPlotter(cutoff_data,self.stimkey),
                         'licktotal':LickPlotter(self.data, self.stimkey),
                         'lickdist':WheelLickScatterPlotter(cutoff_data,self.stimkey)}
    
    def plot(self,**kwargs):
        self.fig = plt.figure(figsize = kwargs.get('figsize',(30,15)))
        widths = [2,1,1]
        heights = [1,1]
        gs = self.fig.add_gridspec(ncols=3, nrows=2,
                                   width_ratios=widths,height_ratios=heights,
                                   left=kwargs.get('left',0.04),right=kwargs.get('right',0.96),
                                   wspace=kwargs.get('wspace',0.35),hspace=kwargs.get('hspace',0.4))

        gs_in1 = gs[:,0].subgridspec(nrows=2,ncols=1,hspace=0.3)

        ax_time = self.fig.add_subplot(gs_in1[0,0])
        self.plotters['responsetime'].plot(ax=ax_time)
        if self.cutoff_time is not None:
            ax_time.axhline(self.cutoff_time/1000,linewidth=2,color='r')
        
        ax_perf = self.fig.add_subplot(gs_in1[1,0])
        ax_perf = self.plotters['performance'].plot(ax=ax_perf,seperate_by='contrast')
        
        ax_lick = ax_perf.twinx()
        ax_lick = self.plotters['licktotal'].plot(ax=ax_lick)
        ax_lick.grid(False)
        
        gs_in2 = gs[:,1].subgridspec(nrows=2,ncols=1,hspace=0.2)
        ax_resp = self.fig.add_subplot(gs_in2[0,0])
        self.plotters['responsepertype'].plot(ax=ax_resp)
        
        ax_ = self.fig.add_subplot(gs_in2[1,0])
        ax_ = self.plotters['perfpertype'].plot(ax=ax_)
        
        ax_psycho = self.fig.add_subplot(gs[0,2])
        ax_psycho = self.plotters['curve'].plot(ax=ax_psycho)
        
        ax_ldist = self.fig.add_subplot(gs[1,2])
        ax_ldist = self.plotters['lickdist'].plot(ax=ax_ldist)
        
        plt.tight_layout()
    
    
    def save(self,saveloc,date,animalid):
        if self.fig is not None:
            saveloc = pjoin(saveloc,'figures')
            if not os.path.exists(saveloc):
                os.mkdir(saveloc)
            savename = f'{date}_sessionSummary_{animalid}.pdf'
            saveloc = pjoin(saveloc,savename)
            self.fig.savefig(saveloc,bbox_inches='tight')
            display(f'Saved {savename} plot')
            
            
            
            
class WheelSummaryPresentationPlotter(BasePlotter):
    pass