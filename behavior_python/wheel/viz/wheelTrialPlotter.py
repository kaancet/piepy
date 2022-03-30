from .wheelBasePlotter import *

class WheelTrialPlotter(WheelBasePlotter):
    def __init__(self,trial_data, metadata,*args, **kwargs):
        self.trial_data = trial_data
        self.metadata = metadata

    def save(self,plotkey):
        pass

    def trialPicture(self, ax=None, *args, **kwargs):
        """ Plots most of the things that hapened during a given trial
            :param trial_no  : the no of trial to be plotted
            :param ax        : a premade axes object to plot on
            :type trial_no   : int
            :type ax         : matplortlib.axes
            :return ax       : plotted axes
            :rtype           : matplotlib.axes
        """
        fontsize = kwargs.get('fontsize',22)
        barebones = kwargs.get('barebones',False)
        notitle = kwargs.get('notitle',False)
        savefig = kwargs.get('savefig',True)

        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(20,10)))
            ax = self.fig.add_subplot(1,1,1)

        cols_to_reset = ['closedloopdur','openloopstart','stimdur','wheel','lick','trialdur','reward','reaction_t','avg_lick_t_diff']
        # this should iterate once for stimpy data 
        for trial_dict in self.trial_data:
            # reset the times relative to trial start
            trial_dict = reset_times(trial_dict, cols_to_reset=cols_to_reset, anchor='trialdur')
            stim_side = trial_dict['stim_side']
            side_txt = 'left' if np.sign(stim_side)==-1 else 'right'

            if trial_dict['correction'] == 0:
                alpha = 1
            else:
                alpha = 0.5

            ax.plot(trial_dict['wheel'][:,0],trial_dict['wheel'][:,1],
                    marker='o',
                    linewidth=kwargs.get('linewidth',4),
                    label='Wheel',
                    alpha=alpha)

            # lick
            for i,l in enumerate(trial_dict['lick']):
                if len(l):
                    if i==0:
                        ax.plot([l[0],l[0]],[-5,5],
                                color='cyan',
                                linewidth=kwargs.get('stripwidth',6),
                                label='Licks',
                                alpha=alpha)
                    else:
                        ax.plot([l[0],l[0]],[-5,5],
                                color='cyan',
                                linewidth=kwargs.get('stripwidth',6),
                                alpha=alpha)
            # reward
            for r in trial_dict['reward']:
                if len(r):
                    ax.plot([r[0],r[0]],[-5,5],
                            color='red',
                            linewidth=kwargs.get('stripwidth',6),
                            label='Reward',
                            alpha=alpha)

            # state backgrounds
            # stimdur
            ax.axvspan(trial_dict['stimdur'][0],trial_dict['stimdur'][1],color='gray',alpha=0.1,label='Stim Present')
            
            #closedloopdur
            ax.axvspan(trial_dict['closedloopdur'][0],trial_dict['closedloopdur'][1],color='gray',alpha=0.2, label='Decision Window')
            
            # draw pass points
            # correct
            ax.plot(trial_dict['stimdur'],[-stim_side] *2,
                    linewidth=2, linestyle='--',color='forestgreen',label='Correct')
            # incorrect
            ax.plot(trial_dict['stimdur'],[stim_side] * 2,
                    linewidth=2, linestyle='--',color='maroon',label='Incorrect')

            #draw moments
            ax.plot([trial_dict['reaction_t'],trial_dict['reaction_t']],[-100,100],
                    linewidth=2, linestyle=":",color='red',label='Reaction moment')

            ax.plot([trial_dict['avg_lick_t_diff'],trial_dict['avg_lick_t_diff']],[-100,100],
                    linewidth=2, linestyle=':',color='orange',label='Avg. Lick Dist.')


        ax.set_xlim([trial_dict['openloopstart']-500,trial_dict['stimdur'][1]+1000])
        ax.set_ylim([-100,100])
        ax.set_xlabel('Time (ms)', fontsize=fontsize)
        
        ax.tick_params(labelsize=fontsize)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        if not barebones:
            ax.set_ylabel('Wheel Position (deg)', fontsize=fontsize)
            if not notitle:
                ax.set_title('{0} {1}\nTrial No: {2} (stim on {3})'.format(self.metadata['animalid'],
                                                                           self.metadata['date'],
                                                                           trial_dict['trial_no'],
                                                                           side_txt),
                             fontsize=fontsize+1)
            ax.legend(loc='lower right',fontsize=fontsize-8)

        if savefig:
            self.save('trialpicture')
        return ax