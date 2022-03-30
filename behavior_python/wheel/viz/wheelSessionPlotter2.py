from ...basePlotters import *
from behavior_python.wheel.wheelAnalysis import WheelAnalysis
from ..wheelAnalysis import WheelCurve


class WheelPsychometricPlotter(BasePlotter):
    __slots__ = ['plot_data','fitters']
    def __init__(self, data: dict,**kwargs):
        super().__init__(data, **kwargs)
        self.fitters = {}
        # for psychometric we need different stimulus types in different dict keys
        self.plot_data = self.data
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
        
    def plot(self,ax:plt.Axes=None,**kwargs):
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(8,8)))
            ax = self.fig.add_subplot(1,1,1)
            
        for k,v in self.fitters.items():
            
            ax = self.__plot__(ax,v.signed_contrast,v.percentage,
                               v.fitted_curve,v.confidence,
                               color=stim_styles[k]['color'],
                               label=k,
                               **kwargs)

        # prettify
        fontsize = kwargs.get('fontsize',22)
        ax.set_xlabel('Contrast Value', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.spines['left'].set_bounds(0, 1)
        ax.spines['bottom'].set_bounds(-100, 100)
        
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
    def __init__(self,data,stimkey:str,**kwargs):
        super().__init__(data, stimkey, **kwargs)
        self.modify_data()

    def modify_data(self,*args,**kwargs):
        # change the self.plot_data here if need to plot something else
        pass
        

class WheelResponseHistogramPlotter(ResponseTimeHistogramPlotter):
    def __init__(self, data, stimkey: str, **kwargs):
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

        
class WheelSummaryPlotter:
    __slots__ = ['data','fig','plotters','stimkey']
    def __init__(self, data:dict, stimkey:str=None,**kwargs):
        self.data = data # gets the stim data dict
        self.stimkey = stimkey
        self.fig = None
        self.init_plotters()
        
    def init_plotters(self):
        # TODO: Make this changable
        self.plotters = {'performance':WheelPerformancePlotter(self.data, self.stimkey),
                         'responsepertype':WheelResponseTimeScatterCloudPlotter(self.data,self.stimkey),
                         'curve':WheelPsychometricPlotter(self.data),
                         'responsetime':WheelResponseTimePlotter(self.data,self.stimkey),
                         'perfpertype':WheelResponseHistogramPlotter(self.data,self.stimkey),
                         'licktotal':LickPlotter(self.data, self.stimkey)}
    
    def plot(self,**kwargs):
        self.fig = plt.figure(figsize = kwargs.get('figsize',(20,10)))
        widths = [2,1,1]
        heights = [1,1]
        gs = self.fig.add_gridspec(ncols=3, nrows=2, 
                                   width_ratios=widths,height_ratios=heights,
                                   left=kwargs.get('left',0),right=kwargs.get('right',1),
                                   wspace=kwargs.get('wspace',0.3),hspace=kwargs.get('hspace',0.4))

        gs_in1 = gs[:,0].subgridspec(nrows=2,ncols=1,hspace=0.3)

        ax_time = self.fig.add_subplot(gs_in1[0,0])
        self.plotters['responsetime'].plot(ax=ax_time)
        
        ax_perf = self.fig.add_subplot(gs_in1[1,0])
        ax_perf = self.plotters['performance'].plot(ax=ax_perf)
        
        ax_lick = ax_perf.twinx()
        ax_lick = self.plotters['licktotal'].plot(ax=ax_lick)
        
        
        gs_in2 = gs[:,1].subgridspec(nrows=2,ncols=1,hspace=0.2)
        ax_resp = self.fig.add_subplot(gs_in2[0,0])
        self.plotters['responsepertype'].plot(ax=ax_resp)
        
        ax_ = self.fig.add_subplot(gs_in2[1,0])
        ax_ = self.plotters['perfpertype'].plot(ax=ax_)
        
        ax_psycho = self.fig.add_subplot(gs[0,2])
        ax_psycho = self.plotters['curve'].plot(ax=ax_psycho)
        
        # this is a placeholder
        ax_psycho2 = self.fig.add_subplot(gs[1,2])
        ax_psycho2 = self.plotters['curve'].plot(ax=ax_psycho)
    
    
    def save(self,saveloc,date,animalid):
        if self.fig is not None:
            saveloc = pjoin(saveloc,'figures')
            if not os.exists(saveloc):
                os.mkdir(saveloc)
            savename = f'{date}_sessionSummary_{animalid}'
            saveloc = pjoin(saveloc,savename)
            self.fig.savefig(saveloc)
            display(f'Saved {savename} plot')