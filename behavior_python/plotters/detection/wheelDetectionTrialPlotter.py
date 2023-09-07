from ..basePlotters import *
from matplotlib.collections import LineCollection


class DetectionTrialPlotter:
    def __init__(self,data:pl.DataFrame) -> None:
        self.data = data
    
    @staticmethod
    def get_trial_variables(trial_data:pl.DataFrame) -> dict:
        """ Extracts the static trial variables and puts them in a dict"""
        #side, answer, sftf, contrast
        return {'stim_side' : trial_data[0,"stim_side"],
                'contrast'  : trial_data[0,"contrast"],
                'answer'    : trial_data[0,"answer"],
                'sf'        : round(trial_data[0,"spatial_freq"],2),
                'tf'        : trial_data[0,"temporal_freq"]
                }
    
    def plot(self,ax:plt.Axes=None,trial_no:int=3,t_lim:list=None,**kwargs):
        fontsize = kwargs.pop('fontsize',25)
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(15,8)))
            ax = self.fig.add_subplot(1,1,1)
            if 'figsize' in kwargs:
                kwargs.pop('figsize')
                
        if t_lim is None:
            t_lim = [-500,1200]
        
        self.trial_data = self.data.filter(pl.col("trial_no")==trial_no)
        if self.trial_data[0,'answer'] == -1:
            time_anchor = 'open_start_absolute'
        else:
            time_anchor = 'stim_start_rig'
        
        self.trial_data = self.trial_data.with_columns([
            (pl.col('trial_start') - pl.col(time_anchor)).suffix('_reset'),
            (pl.col('open_start_absolute') - pl.col(time_anchor)).suffix('_reset'),
            (pl.when(pl.col('stim_start_rig').is_not_null()).then(pl.col('stim_start_rig') - pl.col(time_anchor)).otherwise(pl.col('blank_time'))).alias('stim_start_rig_reset'),
            (pl.col('stim_end_rig') - pl.col(time_anchor)).suffix('_reset')
            ])

        # plot the regions
        ax.axvspan(self.trial_data[0,"trial_start_reset"],
                   self.trial_data[0,"open_start_absolute_reset"],
                   color='gray',alpha=0.3,label='Trial Prep.')
        
        ax.axvspan(self.trial_data[0,"open_start_absolute_reset"],
                   self.trial_data[0,"stim_start_rig_reset"],
                   color='orange',alpha=0.3,label=f'Wait for Stim ({round(self.trial_data[0,"blank_time"],1)}ms)')
        
        ax.axvspan(self.trial_data[0,"stim_start_rig_reset"],
                   self.trial_data[0,"stim_end_rig_reset"],
                   color='green',alpha=0.3,label='Response Window')
        
        if self.trial_data[0,'opto_pattern']!=-1 and self.trial_data[0,'answer']!=-1:
            ax.barh(ax.get_ylim()[1],self.trial_data[0,'response_latency'],left=0,height=10,color='aqua')   
        
        
        # plot the wheels
        wheel_time = self.trial_data['wheel_time'].explode().to_numpy()
        wheel_pos = self.trial_data['wheel_pos'].explode().to_numpy()
        
        # interpolate the wheels
        pos,t = interpolate_position(wheel_time,wheel_pos,freq=100)
        
        ax.plot(t,pos,
                color='gray',linewidth=3,label='Wheel Trace interp')
        
        ax.plot(wheel_time,wheel_pos,'k+',
                linewidth=3,label='Wheel Trace')
        if self.trial_data[0,'answer'] != 0:
            ax.axvline(self.trial_data[0,'wheel_reaction_time'],
                    color='purple',linestyle='-.',linewidth=2,label=f"Wheel Response Time({self.trial_data[0,'wheel_reaction_time']}ms)")

            ax.axvline(self.trial_data[0,'response_latency'],
                    color='r',linestyle=':',linewidth=2,label=f"State Response Time({self.trial_data[0,'response_latency']}ms)")
        
        # plot the reward
        reward = self.trial_data[0,'reward']
        if reward is not None:
            reward = reward[0] - self.trial_data[0,time_anchor]
            ax.axvline(reward,
                       color='darkgreen',linestyle='--',label='Reward')
        
        # plot the lick
        lick_arr = self.trial_data['lick'].explode().to_numpy()
        if len(lick_arr):
            lick_arr = [l - self.trial_data[0,time_anchor] for l in lick_arr]
            ax.scatter(lick_arr,[0]*len(lick_arr),marker='|',c='darkblue',s=50)
        
        # prettify
        trial_vars = self.get_trial_variables(self.trial_data)
        title = f'Trial No : {trial_no}  '
        for k,v in trial_vars.items():
            title += f'{k}={v}, '
            if k == 'contrast':
                title += '\n'
        ax.set_title(title,fontsize=fontsize)
        if self.trial_data[0,'answer'] != -1:
            ax.set_xlabel('Time from Stim onset (ms)',fontsize=fontsize)
        else:
            ax.set_xlabel('Time from cue onset (ms)',fontsize=fontsize)
        
        ax.set_ylabel('Wheel Position (deg)',fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        
        ax.set_xlim(t_lim)
        
        ax.spines['bottom'].set_position(('outward', 10))
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.legend(loc='center left',bbox_to_anchor=(1,0.5),fontsize=fontsize,frameon=False)
        
        return ax
        