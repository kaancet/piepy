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
        
    
class BehaviorPerformancePlotter(BehaviorBasePlotter):
    __slots__ = []
    def __init__(self, cumul_data, summary_data, **kwargs) -> None:
        super().__init__(cumul_data, summary_data, **kwargs)
        
    @staticmethod
    def __plot__(ax,x,y,**kwargs) -> plt.Axes:
        ax.plot(x,y,**kwargs)
        return ax
    
    def save(self,saveloc:str) -> None:
        if self.fig is not None:
            saveloc = pjoin(saveloc,'performance',self.animalid)
            if not os.path.exists(saveloc):
                os.makedirs(saveloc)
            last_date = self.cumul_data['date'].iloc[-1]
            savename = f'{last_date}_performance.pdf'
            saveloc = pjoin(saveloc,savename)
            self.fig.savefig(saveloc,bbox_inches='tight')
            display(f'Saved {savename} plot') 
        
        
class ContrastProgressionPlotter(BehaviorBasePlotter):
    __slots__ = ['animalid','session_contrast_image','contrast_column_map','cbar']
    def __init__(self, animalid:str, cumul_data, summary_data, **kwargs) -> None:
        super().__init__(cumul_data, summary_data, **kwargs)
        self.animalid = animalid
        self.cumul_data = self.add_difference_columns(self.cumul_data)
        
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
    
    def seperate_contrasts(self,contrast_names:list=None,do_opto:bool=False):
        """Seperates the contrast performances for each session throughout the training and experiments """
        # create a predefined contrast vector which includes every contrast and opto
        contrast_column_map = {name:idx for idx,name in enumerate(contrast_names)}
        contrast_column = np.zeros((len(contrast_column_map),1))
        
        data = self.cumul_data[self.cumul_data['session_difference']>=0] # start from first actual training
        
        
        if do_opto:
            data = data[data['opto']==1]
        else:
            try:
                data = data[data['opto']==0]
            except:
                pass
        
        session_nos = np.unique(data['session_no'])
        all_sessions = np.zeros((len(contrast_names),len(session_nos)))
        all_sessions[:] = np.nan
        for k,s_no in enumerate(session_nos):
            sesh_data = data[data['session_no']==s_no]
            
            sesh_contrasts = nonan_unique(sesh_data['contrast']) # this also removes the early trials which have np.nan values for contrasts
            
            contrast_column[:] = np.nan
            for i,c in enumerate(sesh_contrasts):
                c_data = sesh_data[sesh_data['contrast']==c] 
                key = str(c)
                
                if len(c_data):
                    correct_percent = len(c_data[c_data['answer']==1])/len(c_data)
                    contrast_column[contrast_column_map[key]] = 100*correct_percent
                else:
                    pass
            # concat the column to overall sessions image      
            all_sessions[:,k] = np.ravel(contrast_column)
            
        self.contrast_column_map = contrast_column_map        
        self.session_contrast_image = all_sessions
    
    @staticmethod
    def __plot__(ax:plt.Axes,matrix,cmap,**kwargs):
        im = ax.imshow(matrix,vmin=0,vmax=100,
                       cmap=cmap,
                       **kwargs)
        return ax,im
    
    def save(self,saveloc:str) -> None:
        if self.fig is not None:
            saveloc = pjoin(saveloc,'contrastProgression',self.animalid)
            if not os.path.exists(saveloc):
                os.makedirs(saveloc)
            last_date = self.cumul_data['date'].iloc[-1]
            savename = f'{last_date}_contrastProgress.pdf'
            saveloc = pjoin(saveloc,savename)
            self.fig.savefig(saveloc,bbox_inches='tight')
            display(f'Saved {savename} plot') 
    
        
class BehaviorScatterPlotter(BehaviorBasePlotter):
    __slots__ = ['animalid','plot_data','x_name','y_name']
    def __init__(self, animalid:str, cumul_data, summary_data, **kwargs) -> None:
        super().__init__(cumul_data, summary_data, **kwargs)
        self.animalid = animalid
        
    @staticmethod
    def __plot__(ax,x,y,**kwargs):
        ax.scatter(x,y,**kwargs)
        
        return ax
    
    def save(self,saveloc:str) -> None:
       if self.fig is not None:
            saveloc = pjoin(saveloc,'behaviorScatter',self.animalid)
            if not os.path.exists(saveloc):
                os.makedirs(saveloc)
            last_date = self.summary_data['date'].iloc[-1]
            savename = f'{last_date}_{self.x_name}_vs_{self.y_name}.pdf'
            saveloc = pjoin(saveloc,savename)
            self.fig.savefig(saveloc,bbox_inches='tight')
            display(f'Saved {savename} plot')  