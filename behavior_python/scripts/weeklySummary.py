

from ..wheelBehavior import *




def weeklySummary():

    # if end of the week do weekly summary (pooled psychometric and behavior analysis)
    if day_of_week in ['Friday','Saturday','Sunday']:
        display('It\'s {0}! Doing end of week analysis'.format(day_of_week))
        week_start = dt.strftime(day_as_dt-delta(day_as_dt.weekday()),"%y%m%d")
        week_end = day_str

        week_analysed_behave = {}
        for i,animalid in enumerate(analysed_wheel.keys()):
            week_analysed_behave[animalid] = WheelBehavior(animalid,dateinterval=[week_start,week_end],load_behave=True,load_data=True)

        plot_cnt = len(to_plot)
        fig = plt.figure(figsize=(20,30))
        for i,animalid in enumerate(to_plot):
            # g = week_analysed_behave[animalid].plot('behaviorSummary')
            if plot_cnt <=3:
                ax_in1 = fig.add_subplot(1*3,plot_cnt,i+1)
                ax_in2 = fig.add_subplot(1*3,plot_cnt,i+4,sharex=ax_in1)
                ax_in3 = fig.add_subplot(1*3,plot_cnt,i+7,sharex=ax_in2)
            elif plot_cnt > 3 and plot_cnt <= 6:
                ax_in1 = fig_add_subplot(2*3,3,i+1)
                ax_in2 = fig_add_subplot(2*3,3,i+4,sharex=ax_in1)
                ax_in3 = fig_add_subplot(2*3,3,i+7,sharex=ax_in2)
            else:
                ax_in1 = fig.add_subplot(3*3,3,i+1)
                ax_in2 = fig.add_subplot(3*3,3,i+4,sharex=ax_in1)
                ax_in3 = fig.add_subplot(3*3,3,i+7,sharex=ax_in2)


            week_analysed_behave[animalid].plot('performance',ax=ax_in1)
            week_analysed_behave[animalid].plot('weight',ax=ax_in2)
            week_analysed_behave[animalid].plot('trialDistributions',ax=ax_in3)

            ax_in1.set_title(animalid,fontsize=18)
        fig.tight_layout()

        display('Saving weeklySummary plot')
        fig.savefig('J:\\data\\analysis\\behavior_results\\python_figures\\weeklyPlots\\{0}_summary.pdf'.format(day_str),dpi=100,bbox_inches='tight')