from .plotter_utils import *
from os.path import join as pjoin
from ..wheelUtils import get_trajectory_avg
from ..utils import getConfig

class BehaviorBasePlotter:
    __slots__ = ['animalid','cumul_data','summary_data','fig','color']
    def __init__(self,animalid:str,cumul_data:pd.DataFrame=None,summary_data:pd.DataFrame=None) -> None:
        self.animalid = animalid
        self.cumul_data = cumul_data
        self.summary_data = summary_data
        self.fig = None
        set_style('analysis')
        self.color = Color()
        self.summary_data = self.add_difference_columns(self.summary_data)
        
    def list_valid_axes(self,data_type:str='summary'):
        temp = f'''The valid axes for {data_type} type data are:\n'''
        if data_type == 'summary':
            for c in self.summary_data.columns:
                temp += f'- {c}\n'
        elif data_type == 'cumul':
            for c in self.cumul_data.columns:
                temp += f'- {c}\n'
        print(temp)
        
    def filter_dates(self,dateinterval:list=None) -> None:
        """ Filters both summary and cumul data according to given date interval"""
        if dateinterval is None:
            return None
        
        if not isinstance(dateinterval,list):
            raise ValueError('dateinterval argument needs to be a list')
        else:
            if len(dateinterval) != 2:
                raise ValueError(f'dateinterval argument needs to have 2 dates, got {len(dateinterval)}')
            
        try:
            dateinterval_dt = [dt.strptime(d,"%y%m%d").date() for d in dateinterval]
        except:
            raise ValueError(f'dates need to be in YYMMDD format, got {dateinterval[0]} instead')
        
        self.summary_data = self.summary_data[(self.summary_data['dt_date']>=dateinterval_dt[0]) & (self.summary_data['dt_date']<=dateinterval_dt[1])]
        self.cumul_data = self.cumul_data[(self.cumul_data['dt_date']>=dateinterval_dt[0]) & (self.cumul_data['dt_date']<=dateinterval_dt[1])]
        display(f'Filtered data between the dates {dateinterval[0]} - {dateinterval[1]} !')        
        
    @staticmethod
    def add_difference_columns(data) -> None:
        """ Adds difference columns to the plot data like the day difference, session difference """
        try:
            start_day = data[data['paradigm'].str.contains('training',na=False)]['dt_date'].iloc[0]
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
            saveloc = pjoin(analysis_path,'behavior_results','python_figures',self.animalid,last_date)
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
    def __init__(self, animalid:str, cumul_data:pd.DataFrame, summary_data:pd.DataFrame, **kwargs) -> None:
        super().__init__(animalid,cumul_data, summary_data, **kwargs)
        
    def check_axes(self,x_axis,y_axis,data_type:str='summary') -> None:
        if data_type == 'summary':
            data_to_check = self.summary_data
            
            if x_axis not in ['session_difference','day_difference','dt_date']:
                raise KeyError(f'''{x_axis} is not a valid value for x_axis, try one of \n - session_difference, \n - day_difference,\n - dt_date''')
            
        elif data_type == 'cumul':
            data_to_check = self.cumul_data
        
        if y_axis not in data_to_check.columns:
            raise KeyError(f'''{y_axis} is not a valid value for y_axis, try one of {self.summary_data.columns}''')
           
        
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
    def __init__(self, animalid:str, cumul_data:pd.DataFrame, summary_data:pd.DataFrame, **kwargs) -> None:
        super().__init__(animalid,cumul_data, summary_data, **kwargs)
        
    def check_axes(self,x_axis,y_axis,data_type:str='summary') -> None:
        if data_type == 'summary':
            data_to_check = self.summary_data
        elif data_type == 'cumul':
            data_to_check = self.cumul_data
        
        try:
            tmp = data_to_check[x_axis]
        except KeyError as k:
            raise KeyError(f'The column name {x_axis} is not valid for x_axis, try one of:\n{self.summary_data.columns}')
        
        try:
            tmp = data_to_check[y_axis]
        except KeyError as k:
            raise KeyError(f'The column name {y_axis} is not valid for x_axis, try one of:\n{self.summary_data.columns}')

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
    
        
class WeightProgressionPLotter(BehaviorProgressionPlotter):
    def __init__(self, animalid:str, cumul_data:pd.DataFrame, summary_data:pd.DataFrame, **kwargs) -> None:
        super().__init__(animalid,cumul_data, summary_data, **kwargs)
        
    def plot(self,x_axis:str='session_difference',ax:plt.Axes=None,**kwargs):
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(8,8)))
            ax = self.fig.add_subplot(1,1,1)
            if 'figsize' in kwargs:
                kwargs.pop('figsize')
        
        water_res_starts = self.summary_data[self.summary_data['paradigm']=='water restriction start']
        
        for i in range(10): #arbitrary check count, should not be this many water restriction start/stops in reality
            latest_water_restriction_weight = water_res_starts['weight'].iloc[-1-i]
            if not np.isnan(latest_water_restriction_weight):
                break
            
        x_axis_data = self.summary_data[x_axis].to_numpy()
        y_axis_data = self.summary_data['weight'].to_numpy()
        
        ax = self.__plot__(ax,x=x_axis_data,y=y_axis_data,color='k',**kwargs)
        
        ax.axhline(y=latest_water_restriction_weight*0.9,
                     color='orange',
                     linewidth=2,
                     linestyle=':')
        
        ax.axhline(y=latest_water_restriction_weight*0.8,
                     color='red',
                     linewidth=2,
                     linestyle=':')
        
        #prettify
        fontsize=kwargs.get('fontsize',14)
        ax.set_xlabel(x_axis, fontsize=fontsize)
        ax.set_ylabel('Weight (g)', fontsize=fontsize)

        ax.tick_params(axis='x', rotation=45,length=20, width=2, which='major')
        ax.tick_params(axis='both', labelsize=fontsize)
        ax.grid(alpha=0.8,axis='both')

        return ax
        
        
        