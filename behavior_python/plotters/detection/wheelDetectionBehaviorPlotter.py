from behavior_python.detection.wheelDetectionSession import WheelDetectionSession
from ..basePlotters import *
from .wheelDetectionSessionPlotter import *


class WheelDetectionBehaviorSummaryPlotter:
    __slots__ = ['animalid','plot_data','plot_stats','fig','stimkey']
    def __init__(self, animalid: str, session_list: list, stimkey: str = None, day_count: int = 9, **kwargs) -> None:
        self.animalid = animalid
        self.plot_data, self.plot_stats = self.get_sessions_data(session_list,day_count)
        self.stimkey = stimkey
        self.fig = None
        
    def get_sessions_data(self,session_list,day_count) -> list:
        """ Returns the stim_data of last x sessions. 
        The returned data is a dict of dicts."""
        data = {}
        stats = {}
        
        past_days = np.arange(1,day_count+1)
        
        for day in past_days[::-1]:
            try:
                sesh = session_list[-day]
                date_str = sesh[1].strftime('%y%m%d')
                w = WheelDetectionSession(sesh[0],load_flag=True)
                data[date_str] = w.data.stim_data
                stats[date_str] = w.stats
            except:
                pass
            
        return data,stats 


class WheelDetectionPastDaysGridSummary(WheelDetectionBehaviorSummaryPlotter):
    __slots__ = ['plot_type']
    def __init__(self, animalid:str,session_list:list,stimkey:str=None, day_count:int=9,plot_type:str='summary',**kwargs):
        super().__init__(animalid,session_list,stimkey,day_count)
        self.plot_type = plot_type
            
    def init_plotters(self,data:dict) -> dict:
        if self.plot_type == 'summary':
            return {'psychometric' : DetectionResponseHistogramPlotter(data,self.stimkey),
                    'responsescatter' : DetectionResponseScatterPlotter(data,self.stimkey)}
        else:
            # do stuff here for choosing plot type like wheel, performance,licks etc..
            pass 
            
    def plot(self,nrows=3,ncols=3,**kwargs) -> None:
        self.fig = plt.figure(figsize = kwargs.get('figsize',(20,15)))
        main_gs = self.fig.add_gridspec(nrows=nrows,ncols=ncols,
                                        left=kwargs.get('left',0),
                                        right=kwargs.get('right',1),
                                        wspace=kwargs.get('wspace',0.5),hspace=kwargs.get('hspace',0.5))
        row_nos = [0,0,0,1,1,1,2,2,2]
        for i,date in enumerate(self.plot_data.keys()):
            row_no = row_nos[i]
            col_no = i % ncols
            sub_gs = main_gs[row_no,col_no].subgridspec(nrows=1,ncols=2,wspace=0.5)
            
            data = self.plot_data[date]
            plotters = self.init_plotters(data)
            
            ax_psych = self.fig.add_subplot(sub_gs[0,0])
            ax_psych = plotters['psychometric'].plot(ax=ax_psych)
            ax_psych.set_title(date,fontsize=20)
            
            ax_resp = self.fig.add_subplot(sub_gs[0,1])
            ax_resp = plotters['responsescatter'].plot(ax=ax_resp)
            ax_resp.set_title(f'PC%={self.plot_stats[date].all_correct_percent} \t HR={self.plot_stats[date].hit_rate}')
         
        self.fig.tight_layout()
            
    def save(self,saveloc) -> None:
        if self.fig is not None:
            saveloc = pjoin(saveloc,'gridSummaries',self.animalid)
            if not os.path.exists(saveloc):
                os.makedirs(saveloc)
            last_date = list(self.plot_data.keys())[-1]
            savename = f'{last_date}_past9days_{self.plot_type}.pdf'
            saveloc = pjoin(saveloc,savename)
            self.fig.savefig(saveloc,bbox_inches='tight')
            display(f'Saved {savename} plot')
            

class DetectionContrastProgressionPlotter(ContrastProgressionPlotter):
    def __init__(self, animalid:str, cumul_data, summary_data, **kwargs) -> None:
        super().__init__(cumul_data, summary_data, **kwargs)
        self.animalid = animalid
        self.cumul_data = self.add_difference_columns(self.cumul_data)
    
    def plot(self,ax:plt.Axes=None,do_opto:bool=False,**kwargs) -> plt.Axes:
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(15,10)))
            ax = self.fig.add_subplot(1,1,1)
            
        contrast_names = ['1.0','0.5','0.25','0.125','0.0625','0.03125','0']
        self.seperate_contrasts(contrast_names=contrast_names,do_opto=do_opto)
        
        ax,im = self.__plot__(ax,self.session_contrast_image,**kwargs)
        
        fontsize = 15
        ax.set_xlabel('Session from 1st Level1',fontsize=fontsize)
        ax.set_ylabel('Contrast Level (%)',fontsize=fontsize)
        x_axis = np.arange(self.session_contrast_image.shape[1])
        if x_axis[-1]%5 != 0:
            x_axis = np.hstack((x_axis[::5],x_axis[-1]))
        else:
            x_axis = x_axis[::5]
        ax.set_xticks(x_axis)
        ax.set_yticks([v for k,v in self.contrast_column_map.items()])
        ax.set_yticklabels([f'{100*float(k)}' for k,v in self.contrast_column_map.items()])
        ax.tick_params(labelsize=fontsize)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['bottom'].set_position(('outward', 10))
        
        # add space for colour bar
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.3 inch.  
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.8)
        cbar = self.fig.colorbar(im, cax=cax,label='Hit Rate (%)',ticks=[0,25,50,75,100])
        cax.tick_params(width=0,size=0,pad=5,labelsize=15)
        cbar.ax.spines[:].set_visible(False)
        cbar.ax.yaxis.set_label_position('left')
        cbar.ax.yaxis.set_ticks_position('left')
        
        return ax, cax