from .plotter_utils import *
from os.path import join as pjoin
from ..wheel.wheelUtils import get_trajectory_avg
from ..utils import getConfig

class BehaviorBasePlotter:
    __slots__ = ['animalid','cumul_data','summary_data','fig']
    def __init__(self,animalid:str,cumul_data:pd.DataFrame=None,summary_data:pd.DataFrame=None) -> None:
        self.animalid = animalid
        self.cumul_data = cumul_data
        self.summary_data = summary_data
        self.fig = None
        set_style('analysis')
        self.summary_data = self.add_difference_columns(self.summary_data)
        
    @staticmethod
    def add_difference_columns(data) -> None:
        """ Adds difference columns to the plot data like the day difference, session difference """
        try:
            start_day = data[data['paradigm'].str.contains('training')]['dt_date'].iloc[0]
            sesh_idx = data.index[data['dt_date']==start_day].to_list()[0]
            start_sesh = data['session_no'].iloc[int(sesh_idx)]
        except:
            start_day = data['dt_date'].iloc[0]
            sesh_idx = len(data) - 1
            start_sesh = data['session_no'].iloc[int(sesh_idx)]
        
        # day diff
        data['day_difference'] = dates_to_deltadays(data['dt_date'].to_numpy(),start_day)
        
        #session_diff
        data['session_difference'] = data.apply(lambda x: x['session_no'] - start_sesh if not np.isnan(x['session_no']) else x.name - sesh_idx,axis=1)
        return data
    
    def save(self) -> None:
        cfg =  getConfig()
        analysis_path = cfg['analysisPath']
        last_date = self.cumul_data['date'].iloc[-1]
        if self.fig is not None:
            saveloc = pjoin(analysis_path,'behavior_results','python_figures',self.animalid,last_date,self.__class__.__name__)
            if not os.path.exists(saveloc):
                os.makedirs(saveloc)
            savename = f'{last_date}_{self.__class__.__name__}_{self.animalid}.pdf'
            saveloc = pjoin(saveloc,savename)
            self.fig.savefig(saveloc,bbox_inches='tight')
            display(f'Saved {savename} plot') 
            
        
class BehaviorProgressionPlotter(BehaviorBasePlotter):
    """ This is a general progression plotter class which can be extended to plot specific progressions
        such as weight, performance, responsetime, etc.
        It has a plotting function that takes the x and y axis and color values"""
    def __init__(self, animalid, cumul_data, summary_data, **kwargs) -> None:
        super().__init__(animalid,cumul_data, summary_data, **kwargs)
        
    @staticmethod
    def __plot__(ax,x,y,color,**kwargs) -> plt.Axes:
        ax.plot(x,y,color,
                linewidth=3,
                **kwargs)
        return ax
        
        
class BehaviorScatterPlotter(BehaviorBasePlotter):
    """ This is a general scatter plotter class which can be extended to plot specific 
        such as weight, performance, responsetime, etc.
        It has a plotting function that takes the x and y axis and color values"""
    def __init__(self, animalid:str, cumul_data, summary_data, **kwargs) -> None:
        super().__init__(animalid, cumul_data, summary_data, **kwargs)
        
    @staticmethod
    def __plot__(ax,x,y,**kwargs):
        ax.scatter(x,y,**kwargs)
        
        return ax
        

class ContrastLevelsPlotter(BehaviorBasePlotter):
    __slots__ = ['animalid','session_contrast_image','contrast_column_map','cbar']
    def __init__(self, animalid:str, cumul_data, summary_data, **kwargs) -> None:
        super().__init__(cumul_data, summary_data, **kwargs)
        self.animalid = animalid
        self.cumul_data = self.add_difference_columns(self.cumul_data)
    
    @staticmethod
    def __plot__(ax:plt.Axes,matrix,cmap,**kwargs):
        im = ax.imshow(matrix,vmin=0,vmax=100,
                       cmap=cmap)
        return ax,im
    
        
