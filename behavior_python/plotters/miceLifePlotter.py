from .basePlotters import *
from ..wheel.wheelBehavior import WheelBehavior,WheelBehaviorData
from ..detection.wheelDetectionBehavior import WheelDetectionBehavior


#TODO: USE A SIMPLE LOADER TO READ DATA TO PREVENT THE SESSION DIFFERENCEBUG

class LifeBasePlotter:
    __slots__ = ['fig','animalid','plot_data','dateinterval','x_column_name','appended_list','stage_colors']
      
    def __init__(self,animalid:str,dateinterval=None) -> None:
        self.animalid = animalid
        self.dateinterval = dateinterval
        self.plot_data = None
        self.fig = None
        self.appended_list = [] # to prevent appending same kind of data twice, holds the appended task types

        self.stage_colors = {'surgery': 'gray',
                             'experiment_img':'indigo',
                             'experiment_opto':'#2424f2',
                             'water restriction start': 'cyan',
                             'handling': 'orange',
                             'habituation': 'firebrick',
                             'training_wheel':'#a5c79d',
                             'training_detection':'#84b09e'
                             }
    
    
    def set_x_axis(self, diff_type:str=None) -> None:
        """ Sets the column name to be used for x_axis for the plot """
        if diff_type is None:
            #put dates on x axis
            self.x_column_name = 'dt_date'
        elif diff_type == 'days':
            # put day diff on x axis
            self.x_column_name = 'day_difference'
        elif diff_type == 'sessions':
            # put session difference on x axis
            self.x_column_name = 'session_difference'
        else:
            raise ValueError(f'{diff_type} is not a valid value, try days or sessions')
        
    def get_stages(self,diff_type:str='sessions',get_from:str='paradigm') -> dict:
        """ Returns the different kind of stages the animal has went through in a dictionary form,
        where each key holds a list(or list of lists) of start and end dates of each stage
        """
        if get_from not in self.plot_data.columns:
            raise ValueError(f'{get_from} is not a valid column in plot data!') 
        
        stages_dict = {}
        
        self.set_x_axis(diff_type)
        comp_val = 1
        if self.x_column_name == 'dt_date':
            comp_val = timedelta(comp_val)
        
        stages = self.plot_data[get_from].unique()
        for s in stages:
            dates = []
            paradigm_slice = self.plot_data[self.plot_data[get_from]==s]
            paradigm_slice.reset_index(inplace=True)

            date_range = [paradigm_slice[self.x_column_name].iloc[0]]
            prev_date = date_range[0]
            for i,row in paradigm_slice.iterrows():
                curr_date = row[self.x_column_name]
                
                if curr_date - prev_date > comp_val:
                    date_range.append(paradigm_slice[self.x_column_name].iloc[i-1])
                    dates.append(date_range)
                    date_range = [curr_date]
                    
                prev_date = curr_date
            
            if len(date_range) == 1:
                date_range.append(paradigm_slice[self.x_column_name].iloc[-1])
                dates.append(date_range)
     
            stages_dict[s] = dates
        return stages_dict
    
    def add_difference_columns(self) -> None:
        """ Adds difference columns to the plot data like the day difference, session difference """
        try:
            start_day = self.plot_data[self.plot_data['paradigm'].str.contains('training')]['dt_date'].iloc[0]
            sesh_idx = self.plot_data.index[self.plot_data['dt_date']==start_day].to_list()[0]
            start_sesh = self.plot_data['session_no'].iloc[int(sesh_idx)]
        except:
            start_day = self.plot_data['dt_date'].iloc[0]
            sesh_idx = len(self.plot_data) - 1
            start_sesh = self.plot_data['session_no'].iloc[int(sesh_idx)]
        
        # day diff
        self.plot_data['day_difference'] = dates_to_deltadays(self.plot_data['dt_date'].to_numpy(),start_day)
        
        #session_diff
        self.plot_data['session_difference'] = self.plot_data.apply(lambda x: x['session_no'] - start_sesh if not np.isnan(x['session_no']) else x.name - sesh_idx,axis=1)
        
    def get_data(self,task_name:str,data_type:str='summary') -> None:
        """ Gets the relevant data for plotting depending on task type"""
        task_list = ['wheel','detection']
        if task_name not in task_list:
            raise ValueError(f' Task name {task_name} not defined, try {task_list}')
        
        if task_name in self.appended_list:
            display(f'{task_name} data for {self.animalid} is already in the plot data')
            return None
        else:
            self.appended_list.append(task_name)
        
        if task_name == 'wheel':
            behave = WheelBehavior(self.animalid,
                                   self.dateinterval,
                                   load_data=True,
                                   load_behave=True,
                                   just_load=True)
        elif task_name == 'detection':
            behave = WheelDetectionBehavior(self.animalid,
                                            self.dateinterval,
                                            load_data=True,
                                            load_behave=True,
                                            just_load=True)
            
        if data_type == 'summary':
            temp_data = behave.behavior_data.summary_data
        elif data_type == 'cumul':
            temp_data = behave.behavior_data.cumul_data
        
        if self.plot_data is None:
            self.plot_data = temp_data
        else:
            self.plot_data = self.plot_data.append(temp_data)
            self.plot_data = self.plot_data.drop_duplicates(subset=['date'],ignore_index=True)
            self.plot_data['session_no'] = np.arange(1,len(self.plot_data)+1)
            
            self.plot_data.sort_values('date',ascending=True)
        
        self.add_difference_columns()
        
    def get_checkpoint_days(self,task_name:str):
        task_list = ['wheel','detection']
        if task_name not in task_list:
            raise ValueError(f' Task name {task_name} not defined, try {task_list}')
        
        if task_name == 'wheel':
            # the criteria for animal to have learned wheel movement is when the ratio of missed trials in the first 200 trials 
            nogo_threshold = 5
            bias_threshold = 0.2
            
            # self.plot_data['normalized_bias'] = self.plot_data['bias'].apply(lambda x:np.abs(x-100)/100)
            # self.plot_data['rolling_bias'] = self.plot_data['normalized_bias'].rolling(window=5).mean()
            # temp_data = self.plot_data[self.plot_data['paradigm'].str.contains('training')]
            # bias_learn_session_diff = temp_data[temp_data['rolling_bias'] <= bias_threshold]['session_difference'].iloc[0]
            
        elif task_name == 'detection':
            # the criteria for animal to have learned wheel movement is when the ratio of missed trials in the first 200 trials 
            nogo_threshold = 25
        
        self.plot_data['rolling_nogo'] = self.plot_data['nogo_percent'].rolling(window=5).mean()
        temp_data = self.plot_data[self.plot_data['paradigm'].str.contains('training')]
        wheel_learn_session_diff = temp_data[temp_data['rolling_nogo'] <= nogo_threshold]['session_difference'].iloc[0]

        return wheel_learn_session_diff
        

class TrainingPlotter(LifeBasePlotter):
    """ Uses summary_data from a behavior object to make plots"""
    __slots__ = []
    def __init__(self,animalid:str,dateinterval=None) -> None:
        super().__init__(animalid,dateinterval)
     
    def add_normalized_weights(self) -> None:
        """ Adds the normalized weight column to the plot data""" 
        # normalized weight
        restriction_start_weight = self.plot_data[self.plot_data['paradigm']=='water restriction start']['weight'].values
        if len(restriction_start_weight) == 1:
            start_weight = restriction_start_weight[0]
        elif len(restriction_start_weight) == 0:
            start_weight = self.plot_data['weight'].iloc[0]
        else:
            # if multiple water restriction starts, get the latest one
            start_weight = restriction_start_weight[-1]
        self.plot_data['normalized_weight'] = self.plot_data['weight'].apply(lambda x: x/start_weight)
        
    def get_data(self,task_name:str) -> None:
        super().get_data(task_name)
        self.add_normalized_weights()
        
    def plot(self,plot_range:int=None,diff_type:str='sessions',**kwargs) -> plt.Axes:
        self.fig,axs = plt.subplots(nrows=2,ncols=1,figsize=(12,8),sharex=True,
                                    gridspec_kw={'wspace':kwargs.get('wspace',0.1),
                                                 'hspace':kwargs.get('hspace',0.1)})
        ax1 = axs[0]
        ax2 = axs[1]

        stages = self.get_stages(diff_type)

        # plotting weight, performance and stages
        for stage,dates in stages.items():
            for i,rnge in enumerate(dates):
                #TODO: This does not work when plotting dates
                
                ax1.axvspan(rnge[0]-0.5,rnge[1]+0.5,
                            label=stage if i==0 else '_',
                            color=self.stage_colors[stage],
                            linewidth=0,
                            alpha=0.2)

        #weight
        ax1.plot(self.plot_data[self.x_column_name],self.plot_data['normalized_weight'],
                 color='k',
                 linewidth=4,
                 marker='o')
        
        # plot thresholds
        ax1.axhline(0.9,linestyle=':',linewidth=1.5,color='orange')
        ax1.axhline(0.8,linestyle=':',linewidth=1.5,color='red')
        
        ax12 = ax1.twinx()

        ax12.plot(self.plot_data[self.x_column_name],self.plot_data['correct_pct'],
                color='#15a100',
                marker='o',
                markersize=5,
                linewidth=4,
                label='Correct(%)',
                zorder=2)
        
        ax1.grid(alpha=0.3)
        ax1.set_ylim([0.60,None])
        
        if plot_range is not None:
            ax1.set_xlim([ax1.get_xlim()[0],plot_range])
            ax2.set_xlim([ax2.get_xlim()[0],plot_range])
        else:
            ax1.set_xlim([-20,50])
            ax2.set_xlim([-20,50])
        
        ax1.legend(loc='upper right',
                   facecolor='#ffffff',
                   frameon=False,
                   ncol=5,
                   framealpha=1,
                   bbox_to_anchor=(0.85,1.1))
        
        ax1.spines['right'].set_visible(False)
        ax1.tick_params(axis='both',length=5, width=2, which='major')
        ax1.set_ylim([0.75,None])
        
        ax1.set_ylabel('Normalized Weight', fontsize=15)
        ax1.tick_params(axis='both', labelsize=15)
        ax1.grid(alpha=0.3)
        
        ax12.tick_params(axis='y',length=5, width=2, which='major',color='#15a100')
        ax12.set_ylabel('Correct %', fontsize=15,color='#15a100')
        ax12.set_ylim([0,105])
        ax12.tick_params(axis='y', labelsize=15, colors='#15a100')
        ax12.spines['left'].set_visible(False)
        ax12.spines['right'].set_color('#15a100')
        ax12.yaxis.label.set_color('#15a100')
        
        # plotting trial count and response time
        ax2.plot(self.plot_data[self.x_column_name],self.plot_data['median_response_time'].to_numpy()/1000,
                 color='#0076d6',
                 marker='o',
                 markersize=5,
                 linewidth=4,
                 label='Avg. Response Time (s)',
                 zorder=2)
        
        ax22 = ax2.twinx()
        
        ax22.plot(self.plot_data[self.x_column_name],self.plot_data['trial_count'],
                  color='#b30053',
                  marker='o',
                  markersize=5,
                  linewidth=4,
                  label='Trial Count',
                  zorder=2)

        ax2.set_yscale('log')
        
        # make log axis look better
        minor_locs = [0.2,0.4,0.6,0.8,2,4,6,8]
        ax2.yaxis.set_minor_locator(plt.FixedLocator(minor_locs))
        ax2.yaxis.set_minor_formatter(plt.FormatStrFormatter('%.1f'))
        ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
        ax2.set_yticklabels([format(y,'.0f') for y in ax2.get_yticks()])
        
        ax2.set_xlabel(f'{diff_type.capitalize()} from first Level 1', fontsize=15)
        ax2.tick_params(axis='both',length=5, width=2, which='major',color='#0076d6')
        ax2.tick_params(axis='x', labelsize=15,length=5, width=2, which='major',color='k')
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True)) #label only integers
        ax2.set_ylabel('Median Response Time (s)', fontsize=15,color='#0076d6')
        ax2.tick_params(axis='y',which='major',length=8, labelsize=15, colors='#0076d6')
        ax2.tick_params(axis='y',which='minor', length=4, width=2, color='#0076d6',labelcolor='#0076d6')
        ax2.set_ylim([0.1,12])

        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_color('#0076d6')
        ax2.yaxis.label.set_color('#0076d6')
        ax2.grid(alpha=0.5,which='major')
        ax2.grid(alpha=0.3,which='minor')
        
        ax22.tick_params(axis='y',length=5, width=2, which='major',color='#b30053')
        ax22.set_ylabel('Trial Count', fontsize=15,color='#b30053')
        ax22.set_ylim([0,1000])
        ax22.tick_params(axis='y', labelsize=15, colors='#b30053')
        ax22.spines['bottom'].set_visible(False)
        ax22.spines['left'].set_visible(False)
        ax22.spines['right'].set_color('#b30053')
        ax22.yaxis.label.set_color('#b30053')
        
        return axs
    
    def save(self,saveloc) -> None:
        if self.fig is not None:
            saveloc = pjoin(saveloc,'lifePlots',self.animalid)
            if not os.path.exists(saveloc):
                os.makedirs(saveloc)
            last_date = self.plot_data['date'].iloc[-1]
            savename = f'{last_date}_trainingProgress.pdf'
            saveloc = pjoin(saveloc,savename)
            self.fig.savefig(saveloc,bbox_inches='tight')
            display(f'Saved {savename} plot')


class LifeLinePlotter(LifeBasePlotter):
    """ Plots a line that just shows the different stages in thr life of the animal with different colors"""
    __slots__ = []
    def __init__(self, animalid:str,dateinterval=None) -> None:
        super().__init__(animalid,dateinterval)
        
    def add_level_colors(self,task_name:str) -> None:
        """ Adds level colors to training_stages """
        if task_name == 'wheel':
            max_levels = 4
        elif task_name == 'detection':
            max_levels = 13 #12+1
        
        training_slice = self.plot_data[self.plot_data['leveled_paradigm'].str.contains('training')]
        
        #TODO: This can cause issues in the future, now it assumes that 
        # there is only one training paradigm with different levels per task
        start_color = self.stage_colors[training_slice['paradigm'].iloc[0]]
        uniq_levels = np.unique(training_slice['leveled_paradigm'])
        full_color_range = Color.make_color_range(start_color, rng=max_levels)
        current_color_range = full_color_range[:len(uniq_levels)]
        
        level_colors = {level : current_color_range[i] for i,level in enumerate(uniq_levels)}
        
        self.stage_colors = {**level_colors, **self.stage_colors}
        
    def add_levels(self,task_name:str) -> None:
        """ Adds a column that combines the levels for training stages, for ease of plotting """  
        def concat_level(paradigm,level):
            if paradigm is None:
                raise TypeError(f'The paradigm should not be zero on data! ')
            
            if np.isnan(level):
                return paradigm 
            else:
                if 'training' in paradigm:
                    return f'{paradigm}_{str(int(level))}'
                else:
                    return paradigm
                
        self.plot_data['leveled_paradigm'] = self.plot_data[['paradigm','level']].apply(lambda x: concat_level(x.paradigm,x.level),axis=1)
        self.add_level_colors(task_name)
    
    def get_data(self,task_name:str) -> None:
        super().get_data(task_name)
        self.add_levels(task_name)
        
    def plot(self,ax:plt.Axes=None,y_point:float=0.5,plot_range:int=None,diff_type:str='sessions',put_xaxis:bool=True,**kwargs) -> plt.Axes:
        
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(8,2)))
            ax = self.fig.add_subplot(1,1,1)

        stages = self.get_stages(diff_type,get_from='leveled_paradigm')

        prev_stage = 0
        for stage,dates in stages.items():
            for i,rnge in enumerate(dates):
                #TODO: This does not work when plotting dates
                bar_len = rnge[1] - rnge[0] + 1
                try:
                    c = self.stage_colors[stage],
                except:
                    c = '#202121' 
                ax.barh(y_point, bar_len, 
                        align='center', 
                        height=.5, 
                        left=rnge[0] - 0.5,
                        color=c,
                        label=stage if i==0 else '_',
                        alpha=0.9,
                        zorder=i if not 'water' in stage else 10,
                        )
             
        ax.axvline(0,color='k',linewidth=2)
        ax.axvline(50,color='r',linewidth=2)
        
        if plot_range is not None:
            plot_range = 52
        ax.set_xlim([-20,plot_range])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.tick_params(axis='x',length=5, width=2, labelsize=15)
        
        if put_xaxis:
            ax.set_xlabel(f'{diff_type.capitalize()} from first Level 1', fontsize=15) 
        else:
            ax.set_xticks([])
            ax.set_xticklabels([])
        
        for s in ax.spines:
            ax.spines[s].set_visible(False)
            
        # ax.legend(loc='upper right',
        #            facecolor='#ffffff',
        #            frameon=False,
        #            ncol=5,
        #            framealpha=1,
        #            bbox_to_anchor=(0.85,1.2))
            
        return ax
    
    def save(self,saveloc) -> None:
        if self.fig is not None:
            saveloc = pjoin(saveloc,'lifePlots',self.animalid)
            if not os.path.exists(saveloc):
                os.makedirs(saveloc)
            last_date = self.plot_data['date'].iloc[-1]
            savename = f'{last_date}_lifeLine.pdf'
            saveloc = pjoin(saveloc,savename)
            self.fig.savefig(saveloc,bbox_inches='tight')
            display(f'Saved {savename} plot')