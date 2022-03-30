from .wheelBasePlotter import *
from ..wheelAnalysis import WheelCurve

class WheelBehaviorPlotter(WheelBasePlotter):
    """ Class for plotting behavior plots
        : param behavior_data : Behavior data to be plotted
        : param savepath      : save path(usually the last session path in the analysis folder in bkrunch J)
        : type behavior_data  : DataFrame
        : type savepath       : str
    """
    def __init__(self, behavior_data, savepath, *args, **kwargs):
        self.fig = None
        self.behavior_data = behavior_data

        # get the animalid form the last session
        last_row = self.behavior_data['session'].iloc[-1]
        self.animalid = last_row.session_meta['animalid']
        self.baredate = last_row.session_meta['baredate']
        self.start_weight = self.behavior_data['weight'].iloc[0]

        self.savepath = savepath
        self.savefig = kwargs.get('savefig',True)

        self.append_plot_data()
        self.pool_data()

    def prep_text(self):
        """ prepare summary text for summary plot"""
        current_weight = self.behavior_data['weight'].iloc[-1]
        current_weight_pct = round(100 * current_weight/self.start_weight,2)
        text = """OVERVIEW\n"""
        to_add = {'animalid' : self.animalid,
                  'start_weight' : self.start_weight,
                  'threshold_weight' : self.behavior_data['weight'].iloc[0]*0.8,
                  'current_weight' : '{0}({1}%)'.format(current_weight,current_weight_pct),
                  'start_date' : dt.strftime(self.behavior_data['session_date'].iloc[0], '%d %b %y')}
        
        for k,v in to_add.items():

            text += '\n{0} : {1}\n'.format(k,v)

        return text

    def append_plot_data(self):
        """ Appends some color values and parses the date if more than one session per day"""
        self.behavior_data = self.behavior_data
        # TODO: LOOK HERE
        self.behavior_data = self.behavior_data[self.behavior_data['level']!='level0']
        # set edge colors
        self.behavior_data['edge'] = self.behavior_data['level'].apply(lambda x: 'blue' if 'opto' in x else 'white')
        self.behavior_data.loc[self.behavior_data['level']=='level1','edge'] = 'silver'
        self.behavior_data.loc[self.behavior_data['level']=='level2','edge'] = 'orangered'
        self.behavior_data.loc[self.behavior_data['level']=='level3','edge'] = 'magenta'

        # set string times to handle duplicate dates and set day of week
        session_dates = self.behavior_data['session_date']
        str_session_dates = []
        day_of_week = []
        sesh_i = 1
        for date in session_dates:
            date_str = date.strftime('%d.%m.%y')
            day_of_week.append(date.weekday())
            if date_str in str_session_dates:
                sesh_i += 1
                str_session_dates.append(date.strftime('%d.%m.%y-{0}'.format(sesh_i)))
            else:
                str_session_dates.append(date_str)
                sesh_i = 1

        self.behavior_data['str_dates'] = str_session_dates
        self.behavior_data['day_of_week'] = day_of_week

    def plot_mondays(self,bd,ax):
        mondays = bd[bd['day_of_week']==0]['str_dates']
        ylims = ax.get_ylim()
        for row in mondays:
            ax.plot([row,row],ylims,
                    color='gray',
                    linewidth='4',
                    alpha=0.5,
                    zorder=1)

        ax.set_xticks(mondays)
        ax.set_xticklabels(mondays)
        return ax

    def pool_data(self):
        """ Pools the trial data from the sessions for continuos plots""" 
        data = self.behavior_data
        running_stats = ['response_latency','water_on_rig']

        # first concat all the trials into one giant dataframe
        self.big_df = pd.DataFrame()
        prev_day_last_trial = 0
        for row in data.to_dict(orient="records"):
            sesh_data_df = row['session'].session['data']
            sesh_data_df['session_no'] = row['session_no']
            sesh_data_df['session_date'] = row['session_date']
            sesh_data_df['weight'] = row['weight']
            sesh_data_df['level'] = row['level']
            sesh_data_df['water_on_rig'] = row['session'].session_meta['water_on_rig']

            sesh_data_df['total_trial_no'] = sesh_data_df['trial_no'] + prev_day_last_trial
            prev_day_last_trial = sesh_data_df['total_trial_no'].iloc[-1]

            self.big_df = pd.concat([self.big_df,sesh_data_df],ignore_index=True,sort=False)

        self.big_df = get_running_stats(self.big_df,stats=running_stats,window_size=100,copy_data=False)

    def save(self,plotkey):
        """ Saves the figure"""
        if self.savefig:
            figsave_loc = pjoin(self.savepath,'behaviorFigures')
            if not os.path.exists(figsave_loc):
                os.mkdir(figsave_loc)

            savename = '{0}_{1}_{2}.pdf'.format(self.baredate,
                                                self.animalid,
                                                plotkey)
            savepath = pjoin(figsave_loc, savename)
            self.fig.savefig(savepath,bbox_inches='tight')
            display('{0} plot saved in {1}'.format(plotkey,self.savepath))

    def plot(self,plt_func,closefig=False,*args,**kwargs):
        """ Main plot function that calls specific plotting functions
            :param plt_func  : function nam to be plotted
            :param close_fig : Flag to control closing figures
            :type plt_func   : str
            :type close_fig  : boolean
        """
        callables = []
        for name in dir(self):
            if not is_special(name):
                value = getattr(self, name)
                if callable(value):
                    callables.append(name)

        if plt_func not in callables:
            display('{0} not a plotter function for WheelBehaviorPlotter try {1}'.format(plt_func,', '.join(callables)))
            raise ValueError()

        getattr(self,plt_func)(*args,**kwargs)
        if closefig:
            plt.close(plt.gcf())

    def weight(self, ax=None, *args, **kwargs):
        """ Plots the weight of the animal 
            :param ax        : a premade axes object to plot on 
            :type ax         : matplortlib.axes
            :return ax       : plotted axes
            :rtype           : matplotlib.axes
        """
        fontsize = kwargs.get('fontsize',22)
        level = kwargs.get('level','all')
        tick_space = kwargs.get('tick_space',5)

        # set up the axes
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(16,8)))
            ax = self.fig.add_subplot(1,1,1)

        # set level of data(levels)
        levels = np.unique(self.behavior_data['level'])
        if level == 'all':
            bd = self.behavior_data
        else:
            if level not in levels:
                raise ValueError('Level {0} does not exist, try one of {1}'.format(levels))
            else:
                bd = self.behavior_data[self.behavior_data['level'] == level]
        
        # weight progression
        ax.plot(bd['str_dates'],bd['weight'], color='orange',
            marker='o',markersize=12,linewidth=kwargs.get('linewidth',8), label='Weight')

        # water consumption
        water_on_rig = [sesh.session_meta['water_on_rig'] for sesh in bd['session']]

        # #threshold weight line
        threshold = self.start_weight * 0.8

        # plotting
        ax.plot(ax.get_xlim(), [threshold,threshold],
                linestyle = ':',
                color='red',
                linewidth='2',
                label='Threshold Weight')

        ax = self.plot_mondays(bd,ax)
        # make it pretty
        ax = self.pretty_axes(ax)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='x', rotation=45,length=20, width=2, which='major')
        ax.set_xlabel('Session Dates', fontsize=fontsize)
        ax.set_ylabel('Weight(g)', fontsize=fontsize)
        ax.tick_params(axis='both', labelsize=fontsize)
        
        # some logic to embed the plot into other figures
        if not kwargs.get('notitle',False):
            ax.set_title('{0} Health'.format(self.animalid), fontsize=fontsize+3)
        if kwargs.get('showlegend',True):
            self.fig.legend(fontsize=fontsize-5)

        # save
        if kwargs.get('savefig',True):
            self.save('weight')

        return ax

    def water(self,ax=None, *args, **kwargs):
        """ Plots the vitals of the animal 
            :param ax        : a premade axes object to plot on 
            :type ax         : matplortlib.axes
            :return ax       : plotted axes
            :rtype           : matplotlib.axes
        """
        fontsize = kwargs.get('fontsize',22)
        level = kwargs.get('level','all')
        tick_space = kwargs.get('tick_space',5)

        # set up the axes
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(10,10)))
            ax = self.fig.add_subplot(1,1,1)

        # set level of data(levels)
        levels = np.unique(self.behavior_data['level'])
        if level == 'all':
            bd = self.behavior_data
        else:
            if level not in levels:
                raise ValueError('Level {0} does not exist, try one of {1}'.format(levels))
            else:
                bd = self.behavior_data[self.behavior_data['level'] == level]

        # water consumption
        water_on_rig = [sesh.session_meta['water_on_rig'] for sesh in bd['session']]

        # plotting
        ax.bar(bd['str_dates'],water_on_rig,width=0.75,
            color='deepskyblue',label='Water on rig',zorder=1,alpha=0.5)

        ax.bar(bd['str_dates'],bd['extra_water'],width=0.75,bottom=water_on_rig,
            color='steelblue',label='Extra Water',alpha=0.5)

        # make it pretty
        ax = self.pretty_axes(ax)
        ax.tick_params(axis='x', rotation=45,length=20, width=2, which='major')
        ax.set_xlabel('Session Dates', fontsize=fontsize)
        ax.set_ylabel('Consumed Water(uL)',fontsize=fontsize)
        xticks = [*range(len(bd['str_dates']))]
        ax.set_xticks(xticks[::tick_space])
        ax.set_xticklabels(bd['str_dates'][::tick_space])
        ax.tick_params(axis='both', labelsize=fontsize)
        ax.grid(alpha=0.5,axis='y')

        # some logic to embed the plot into other figures
        if not kwargs.get('notitle',True):
            ax.set_title('{0} Water'.format(self.animalid), fontsize=fontsize+3)
        if kwargs.get('showlegend',True):
            self.fig.legend(fontsize=fontsize-5)

        # Save
        if savefig:
            self.save('water')

        return ax

    def stackedSummary(self, days=None, *args, **kwargs):
        """"""
        fontsize = kwargs.get('fontsize',22)
        self.fig = plt.figure(figsize = kwargs.get('figsize',(20,20)))

        bd = self.behavior_data[self.behavior_data['level']!='level0']
        if days is None:
            #make equally spaced 4 days(1/4, 1/2, 3/4, last session)
            if len(bd) <= 3:
                days = [*range(1,len(bd))]
            else:
                days = [int(x) for x in np.linspace(1,len(bd),4)]
        elif isinstance(days,list):
            pass
        else:
            raise ValueError('Days can only be a scalar integer or list, got {0}'.format(type(days)))
        nrows = 4
        ncols = len(days)
        start_session = bd['session_no'].iloc[0]

        for row in range(nrows):
            for col,day in enumerate(days,1):
                ax_idx = row * ncols + col
                ax = self.fig.add_subplot(nrows, ncols, ax_idx)

                day_row = bd[bd['session_no']==start_session+day-1]
                day_session = day_row['session'].iloc[0]
                day_date = day_row['session_date'].iloc[0].strftime('%d.%m.%y')

                # first row, trial duration
                if row == 0:
                    ax.set_title('Day {0}: {1} Trials\n{2}'.format(day,day_session.session['summaries']['overall']['novel_trials'],day_date),fontsize=fontsize)
                    if col!=1:
                        naked = True
                        ax.spines['left'].set_visible(False)
                        ax.set_yticklabels([])
                    else:
                        naked=False
                    day_session.plot('responsetime', ax=ax, barebones=naked, savefig=False, notitle=True, *args,**kwargs)
                if row == 1:
                    if col!=1:
                        naked = True
                        ax.spines['left'].set_visible(False)
                        ax.set_yticklabels([])
                    else:
                        naked=False
                    day_session.plot('performance', ax=ax, barebones=naked, savefig=False, notitle=True, *args,**kwargs)
                if row == 2:
                    if col!=1:
                        naked = True
                        ax.spines['left'].set_visible(False)
                        ax.set_yticklabels([])
                    else:
                        naked=False
                    day_session.plot('psychometric', ax=ax, barebones=naked, savefig=False, notitle=True, *args,**kwargs)
                    if col!=1:
                        ax.spines['left'].set_visible(False)
                if row == 3:
                    if col!=1:
                        naked = True
                        ax.spines['left'].set_visible(False)
                    else:
                        naked=False
                    day_session.plot('lick', ax=ax, barebones=naked, savefig=False, notitle=True, *args,**kwargs)
                    if col!=1:
                        ax.spines['left'].set_visible(False)
        # self.fig.suptitle('{0} Trianing Summary'.format(self.animalid),fontsize=fontsize+3,fontweight='bold')
        plt.tight_layout()
        self.save('stackedSummary_'+'_'.join([str(x) for x in days]))
        return self.fig

    def performance(self, plt_mode='discrete', ax=None, *args, **kwargs):
        """ Plots the performance of the animal 
            :param plt_mode  : Variable to determine session by session or pooled trial plotting
            :param ax        : a premade axes object to plot on 
            :type plt_mode   : str
            :type ax         : matplortlib.axes
            :return ax       : plotted axes
            :rtype           : matplotlib.axes
        """
        fontsize = kwargs.get('fontsize',22)
        level = kwargs.get('level','all')
        tick_space = kwargs.get('tick_space',5)

        # set up the axes
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(16,8)))
            ax = self.fig.add_subplot(1,1,1)

        # set level of data(levels)
        levels = np.unique(self.behavior_data['level'])
        if level == 'all':
            bd = self.behavior_data
        else:
            if level not in levels:
                raise ValueError('Level {0} does not exist, try one of {1}'.format(levels))
            else:
                bd = self.behavior_data[self.behavior_data['level'] == level]

        # plotting
        if plt_mode == 'discrete':

            ax.plot(bd['str_dates'],bd['session_correct_pct'],
                    color='darkgreen',
                    marker='o',
                    markersize=kwargs.get('markersize',10),
                    linewidth=kwargs.get('linewidth',8),
                    label='Correct(%)',
                    zorder=2)

            ax = self.plot_mondays(bd,ax)
            ax.set_xlabel('Session Dates', fontsize=fontsize)

        elif plt_mode == 'cont':
            tick_space = 500
            ax.plot(self.big_df['total_trial_no'],100*self.big_df['fraction_correct'],
                    color='darkgreen',
                    linewidth=kwargs.get('linewidth',5),
                    label='Correct(%)',
                    zorder=2)

            for s in np.unique(self.big_df['session_no']):
                temp = self.big_df[self.big_df['session_no']==s]['total_trial_no']
                ax.axvspan(temp.iloc[0],temp.iloc[-1],
                    color='gray' if s%2 else 'whitesmoke',
                    zorder=1,
                    alpha=0.2)

            ax.set_xlabel('Trial No.', fontsize=fontsize)

            xticks = [*range(len(self.big_df['total_trial_no']))]
            ax.set_xticks(xticks[::tick_space])
            ax.set_xticklabels(self.big_df['total_trial_no'][::tick_space])
        
        # make it pretty
        ax = self.pretty_axes(ax)
        ax.spines['bottom'].set_bounds(ax.get_xlim()[0],ax.get_xlim()[1])
        ax.set_ylabel('Correct (%)', fontsize=fontsize)
        ax.set_ylim([0,100])
        ax.tick_params(axis='x', rotation=45,length=20, width=2, which='major')
        ax.tick_params(axis='both', labelsize=fontsize)
        ax.grid(alpha=0.5,axis='both')

        # some logic to embed the plot into other figures
        if not kwargs.get('notitle',False):
            ax.set_title('{0} Performance'.format(self.animalid), fontsize=fontsize+3)
        if kwargs.get('showlegend',True):
            ax.legend(fontsize=fontsize-5)

        # save
        if kwargs.get('savefig',True):
            self.save('perfromance')

        return ax
    
    def responseTimes(self,plt_mode='discrete', ax=None, *args, **kwargs):
        """ Plots the response time of the animal 
            :param plt_mode  : Variable to determine session by session or pooled trial plotting
            :param ax        : a premade axes object to plot on 
            :type plt_mode   : str
            :type ax         : matplortlib.axes
            :return ax       : plotted axes
            :rtype           : matplotlib.axes
        """
        fontsize = kwargs.get('fontsize',22)
        level = kwargs.get('level','all')
        tick_space = kwargs.get('tick_space',5)

        # set up the axes
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(16,8)))
            ax = self.fig.add_subplot(1,1,1)
            show_legend = True

        # set level of data(levels)
        levels = np.unique(self.behavior_data['level'])
        if level == 'all':
            bd = self.behavior_data
        else:
            if level not in levels:
                raise ValueError('Level {0} does not exist, try one of {1}'.format(levels))
            else:
                bd = self.behavior_data[self.behavior_data['level'] == level]

        latency_avg = [sesh.session['summaries']['overall']['latency']/1000 for sesh in bd['session']]

        # plotting
        if plt_mode == 'discrete':

            ax.plot(bd['str_dates'],latency_avg,
                    color='royalblue',
                    marker='o',
                    markersize=kwargs.get('markersize',10),
                    linewidth=kwargs.get('linewidth',8),
                    label='Correct Percent')

            ax.set_xlabel('Session Dates', fontsize=fontsize)
            xticks = [*range(len(bd['str_dates']))]
            ax.set_xticks(xticks[::5])
            ax.set_xticklabels(bd['str_dates'][::5])

        elif plt_mode == 'cont':
            tick_space = 500
            ax.plot(self.big_df['total_trial_no'],self.big_df['running_response_latency']/1000,
                    color='royalblue',
                    linewidth=kwargs.get('linewidth',5),
                    label='Response Latency',
                    zorder=2)

            ax.set_xlabel('Trial No.', fontsize=fontsize)

            xticks = [*range(len(self.big_df['total_trial_no']))]
            ax.set_xticks(xticks[::tick_space])
            ax.set_xticklabels(self.big_df['total_trial_no'][::tick_space])

        # make it pretty
        ax = self.pretty_axes(ax)
        ax.spines['bottom'].set_bounds(ax.get_xlim()[0],ax.get_xlim()[1])
        ax.set_yscale('log')
        ax.set_ylabel('Rsponse Times(s)', fontsize=fontsize)
        ax.tick_params(axis='both', labelsize=fontsize)
        ax.grid(alpha=0.5,axis='both')
        ax.tick_params(axis='x', rotation=45,length=20, width=2, which='major')

        # some logic to embed the plot into other figures
        if not kwargs.get('notitle',False):
            ax.set_title('{0} Performance'.format(self.animalid), fontsize=fontsize+3)
        if kwargs.get('showlegend',True):
            ax.legend(fontsize=fontsize-5)        
        # save
        if kwargs.get('savefig',True):
            self.save('perfromance')
        return ax 

    def EPIndex(self, ax=None, *args, **kwargs):
        """ Plots the Error Prediction(EP) index of the animal
            :param ax        : a premade axes object to plot on 
            :type ax         : matplortlib.axes
            :return ax       : plotted axes
            :rtype           : matplotlib.axes
        """
        fontsize = kwargs.get('fontsize',22)
        level = kwargs.get('level','all')
        tick_space = kwargs.get('tick_space',5)
        
        # set up the axes
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(10,10)))
            ax = self.fig.add_subplot(1,1,1)

        # set level of data(levels)
        levels = np.unique(self.behavior_data['level'])
        if level == 'all':
            bd = self.behavior_data
        else:
            if level not in levels:
                raise ValueError('Level {0} does not exist, try one of {1}'.format(levels))
            else:
                bd = self.behavior_data[self.behavior_data['level'] == level]
        ep = []
        for row in bd.itertuples():
            ep.append(row.session.session['summaries']['overall']['EP'])

        # plotting
        ax.plot(bd['str_dates'],ep,
                color='indigo',
                marker='o',
                markersize=kwargs.get('markersize',10),
                linewidth=kwargs.get('linewidth',8),
                label='EP Index')

        # make it pretty
        ax = self.pretty_axes(ax)
        ax.spines['left'].set_bounds(-1,1)
        ax.spines['bottom'].set_bounds(ax.get_xlim()[0],ax.get_xlim()[1])
        ax.set_xlabel('Session Dates', fontsize=fontsize)
        ax.set_ylabel('EP Index', fontsize=fontsize)
        ax.set_ylim([-1,1])

        xticks = [*range(len(bd['str_dates']))]
        ax.set_xticks(xticks[::tick_space])
        ax.set_xticklabels(bd['str_dates'][::tick_space])
        ax.tick_params(axis='both', labelsize=fontsize)
        ax.tick_params(axis='x', rotation=45,length=20, width=2, which='major')
        ax.grid(alpha=0.5,axis='both')

        # some logic to embed the plot into other figures
        if not kwargs.get('notitle',False):
            ax.set_title('{0} EP Index'.format(self.animalid), fontsize=fontsize+3)
        if kwargs.get('showlegend',True):
            ax.legend(fontsize=fontsize-5)
        
        # save
        if savefig:
            self.save('ep_index')

        return ax

    def trialDistributions(self, ax=None, *args, **kwargs):
        """ Plots the distribution of responses through sessions
            :param ax        : a premade axes object to plot on
            :type ax         : matplortlib.axes
            :return ax       : plotted axes
            :rtype           : matplotlib.axes
        """
        fontsize = kwargs.get('fontsize',22)
        level = kwargs.get('level','all')

        barwidth = kwargs.get('barwidth',1)
        edgewidth = kwargs.get('edgewidth',2)
        tick_space = kwargs.get('tick_space',5)

        # set up the axes
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(16,8)))
            ax = self.fig.add_subplot(1,1,1)

        # set level of data(levels)
        levels = np.unique(self.behavior_data['level'])
        if level == 'all':
            bd = self.behavior_data
        else:
            if level not in levels:
                raise ValueError('Level {0} does not exist, try one of {1}'.format(levels))
            else:
                bd = self.behavior_data[self.behavior_data['level'] == level]

        left = np.asarray([sesh.session['summaries']['overall']['left_profile'] for sesh in bd['session']])
        right = np.asarray([sesh.session['summaries']['overall']['right_profile'] for sesh in bd['session']])

        # PLOTTING
        #LEFT
        #correct
        ax.bar(bd['str_dates'], left[:,0], 
                width=barwidth, color='darkgreen',
                edgecolor=bd['edge'], linewidth=edgewidth,
                label='Correct')
        #incorrect
        ax.bar(bd['str_dates'], left[:,1], 
                width=barwidth, bottom=left[:,0],
                color='maroon', edgecolor=bd['edge'],
                linewidth=edgewidth, label='Incorrect')
        #non answer
        ans_hl = np.add(left[:,0], left[:,1]).tolist()
        ax.bar(bd['str_dates'], left[:,2], 
                width=barwidth, bottom=ans_hl,
                color='k', edgecolor=bd['edge'],
                linewidth=edgewidth, label='No Go') 

        #RIGHT
        #correct
        ax.bar(bd['str_dates'], right[:,0], 
                width=barwidth, color='darkgreen',
                edgecolor=bd['edge'], linewidth=edgewidth)
        #incorrect
        ax.bar(bd['str_dates'], right[:,1], 
                width=barwidth, bottom=right[:,0],
                color='maroon', edgecolor=bd['edge'],
                linewidth=edgewidth),
        #non answer
        ans_hr = np.add(right[:,0], right[:,1]).tolist()
        ax.bar(bd['str_dates'], right[:,2], 
                width=barwidth, bottom=ans_hr,
                color='k', edgecolor=bd['edge'],
                linewidth=edgewidth) 

        #midline
        ax.plot(ax.get_xlim(),[0,0],'white',linewidth=2)
        
        # make it pretty
        ax = self.pretty_axes(ax)
        ax.set_xlabel('Session Dates', fontsize=fontsize)
        ax.set_ylabel('Response Distribution(L<=>R)', fontsize=fontsize)
        ax.set_ylim([ax.get_ylim()[0]-15, ax.get_ylim()[1]+15])
        ax = self.plot_mondays(bd,ax)
        ax.tick_params(axis='both', labelsize=fontsize)
        ax.tick_params(axis='x', rotation=45,length=20, width=2, which='major')
        ax.grid(b=True,axis='y',linewidth=0.5)
        ax.set_axisbelow(b=True)

        # some logic to embed the plot into other figures
        if not kwargs.get('notitle',False):
            ax.set_title('{0} Performance'.format(self.animalid), fontsize=fontsize)
        if kwargs.get('showlegend',True): 
            ax.legend(loc='upper left',fontsize=fontsize-5)

        # save
        if kwargs.get('savefig',True):
            self.save('trialdistributions')

        return ax

    def bias(self,ax=None,*args,**kwargs):
        """Plots the bias of the animal 
            :param ax        : a premade axes object to plot on
            :type ax         : matplortlib.axes
            :return ax       : plotted axes
            :rtype           : matplotlib.axes
        """
        fontsize = kwargs.get('fontsize',22)
        level = kwargs.get('level','all')
        tick_space = kwargs.get('tick_space',5)

        # set up axes
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(10,10)))
            ax = self.fig.add_subplot(1,1,1)

        # set level of data(levels)
        levels = np.unique(self.behavior_data['level'])
        if level == 'all':
            bd = self.behavior_data
        else:
            if level not in levels:
                raise ValueError('Level {0} does not exist, try one of {1}'.format(levels))
            else:
                bd = self.behavior_data[self.behavior_data['level'] == level]

        bias_dict = {}
        for row in bd.to_dict(orient='records'):
            for k in row['session'].session['curves'].keys():
                if k not in bias_dict.keys():
                    bias_dict[k] = [row['session'].session['curves'][k].pars[0]]
                else:
                    bias_dict[k].append(row['session'].session['curves'][k].pars[0])

        # plotting
        for key in bias_dict.keys():

            ax.plot(bd['str_dates'],bias_dict[key],
                    color='olive',
                    linewidth=4,
                    marker='o',
                    label=key)

        ax = self.plot_mondays(bd,ax)
        # make it pretty
        ax = self.pretty_axes(ax)
        ax.spines['left'].set_position(('outward',10))
        ax.spines['bottom'].set_position(('outward',10))
        ax.set_xlabel('Session Dates', fontsize=fontsize)
        ax.set_ylabel('Bias', fontsize=fontsize)
        ax.tick_params(axis='both', labelsize=fontsize)
        ax.tick_params(axis='x', rotation=45,length=20, width=2, which='major')
        ax.grid(b=True,alpha=0.7)

        # some logic to embed the plot into other figures
        if not kwargs.get('notitle',False):
            ax.set_title('{0} Bias Progression'.format(self.animalid),
                         fontsize=fontsize+2,
                         fontweight='bold')
        if kwargs.get('showlegend',True):
            ax.legend(loc='upper left', fontsize=fontsize-5)

        #save
        if kwargs.get('savefig',True):
            self.save('bias_progress')

        return ax

    def psychometric(self, pool_depth=5, ax=None, *args, **kwargs):
        """ Plots the pychometric curve of given number of trials 
            :param pool_depth : 
            :param ax         : a premade axes object to plot on 
            :type pool_depth  : int
            :type ax          : matplortlib.axes
            :return ax        : plotted axes
            :rtype            : matplotlib.axes
        """
        fontsize = kwargs.get('fontsize',22)
        level = kwargs.get('level','all')

        # set up the axes
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(10,10)))
            ax = self.fig.add_subplot(1,1,1)

        # set level of data(levels)
        levels = np.unique(self.behavior_data['level'])
        if level == 'all':
            bd = self.behavior_data
        else:
            if level not in levels:
                raise ValueError('Level {0} does not exist, try one of {1}'.format(levels))
            else:
                bd = self.behavior_data[self.behavior_data['level'] == level]

        # pool sessions
        tmp_dict = {}
        pool_cnt = 0
        pooled_curces = {}
        for row in bd[::-1].to_dict(orient='records'):
            pool_cnt += 1
            for k in row['session'].session['curves'].keys():
                if k not in tmp_dict.keys():
                    tmp_dict[k] = row['session'].session['curves'][k].rawdata
                else:
                    tmp_dict[k] = pd.concat([tmp_dict[k],row['session'].session['curves'][k].rawdata],ignore_index=True,sort=False)
            if pool_cnt >= pool_depth:
                    break

        # plotting
        for key in tmp_dict.keys():
            name = '{0}_{1}'.format(level,key)
            curve = WheelCurve(key,tmp_dict[key])

            ax.errorbar(100 * curve.contrast_x, curve.proportion_right, curve.binomial_right,
                        marker=kwargs.get('marker','o'),
                        c=area_colors[key],
                        linewidth=0,
                        markersize=kwargs.get('markersize',15),
                        markeredgecolor=kwargs.get('markeredgecolor','w'),
                        markeredgewidth=kwargs.get('markeredgewidth',2),
                        elinewidth=kwargs.get('elinewidth',3),
                        capsize=kwargs.get('capsize',0),
                        ecolor=area_colors[key],
                        label='{0}({1} sessions)'.format(key,pool_depth))

            ax.plot(100 * curve.contrast_x_fitted, curve.proportion_right_fitted,
                    c=area_colors[key],
                    linewidth=kwargs.get('linewidth',9))
        #midlines 
        ax.plot([0, 0], [0, 1], 'gray', linestyle=':', linewidth=2,alpha=0.7)
        ax.plot([-100, 100], [0.5, 0.5], 'gray', linestyle=':', linewidth=2,alpha=0.7)

        # make it pretty
        ax = self.pretty_axes(ax)
        ax.spines['left'].set_bounds(0, 1)
        ax.spines['bottom'].set_bounds(-100, 100)
        ax.set_xlabel('Contrast Value', fontsize=fontsize)
        ax.set_ylabel('Prob. Choosing R(%)', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        
        # some logic to embed the plot into other figures
        if not kwargs.get('notitle',False):
            ax.set_title('{0} Pooled Psychometric Curve'.format(self.animalid),
                         fontsize=fontsize+2,
                         fontweight='bold')
        if kwargs.get('showlegend',True):
            ax.legend(loc='upper left', fontsize=fontsize-5)
        
        # save
        if kwargs.get('savefig',True):
            self.save('pooled_psychometric')

        return ax

    def behaviorSummary(self,*args,**kwargs):
        """ Dashboardlike summary of behavior """
        fontsize = kwargs.get('fontsize',22)
        
        self.fig = plt.figure(figsize = kwargs.get('figsize',(20,20)))
        widths = [1.3, 2]
        heights = [0.5,0.5,1,0.75]
        gs = self.fig.add_gridspec(ncols=2, nrows=4, width_ratios=widths,
                                  height_ratios=heights)
        # info text
        ax_text = self.fig.add_subplot(gs[0,0])
        text = self.prep_text()
        ax_text.text(0.01,0.93, text, va='top', fontsize=fontsize)
        ax_text = self.empty_axes(ax_text)

        # cont perf and reponse time
        ax_cont = self.fig.add_subplot(gs[3,:])
        ax_cont = self.performance(ax=ax_cont,plt_mode='cont',
                                   notitle=True,savefig=False,showlegend=False,*args,**kwargs)
        h_cont,l_cont = ax_cont.get_legend_handles_labels()
        ax_cont.spines['bottom'].set_position(('outward', 10))
        ax_cont.spines['left'].set_position(('outward',10))
        ax_cont.tick_params(axis='x', rotation=45,length=20, width=2, which='major')

        ax_time = ax_cont.twinx()
        ax_time = self.responseTimes(ax=ax_time,plt_mode='cont',
                                     notitle=True,savefig=False,showlegend=False,*args,**kwargs)
        h_time,l_time = ax_time.get_legend_handles_labels()
        ax_time.spines['left'].set_visible(False)
        ax_time.spines['bottom'].set_visible(False)
        ax_time.spines['right'].set_visible(True)
        ax_time.spines['right'].set_linewidth(2)
        ax_time.grid(axis='y',b=False)

        h_cont.extend(h_time)
        l_cont.extend(l_time)
        ax_time.legend(h_cont,l_cont,fontsize=fontsize-3)

        # bias progress
        ax_bias = self.fig.add_subplot(gs[1,0])
        ax_bias = self.bias(ax=ax_bias,
                            notitle=True,savefig=False,showlegend=False,*args,**kwargs)

        # psychometric
        ax_psycho = self.fig.add_subplot(gs[2,0])
        ax_psycho = self.psychometric(ax=ax_psycho,
                                      notitle=True,savefig=False,showlegend=False,*args,**kwargs)

        # discrete correct and weight
        ax_correct = self.fig.add_subplot(gs[:2,1])
        ax_correct = self.performance(ax=ax_correct,
                                      notitle=True,savefig=False,showlegend=False,*args,**kwargs)
        h_correct,l_correct = ax_correct.get_legend_handles_labels()
        ax_correct.spines['left'].set_visible(False)
        ax_correct.spines['bottom'].set_position(('outward', 10))
        ax_correct.tick_params(axis='x', rotation=45,length=20, width=2, which='major')

        ax_weight = ax_correct.twinx()
        ax_weight = self.weight(ax=ax_weight,
                                notitle=True,savefig=False,showlegend=False,*args,**kwargs)
        h_weight,l_weight = ax_weight.get_legend_handles_labels()
        ax_weight.spines['left'].set_visible(False)
        ax_weight.spines['bottom'].set_visible(False)

        h_correct.extend(h_weight)
        l_correct.extend(l_weight)
        ax_weight.legend(h_correct,l_correct,loc='lower right',fontsize=fontsize-3)

        # trial_dist
        ax_dist = self.fig.add_subplot(gs[2,1])
        ax_dist = self.trialDistributions(ax=ax_dist,
                                          notitle=True,savefig=False,*args,**kwargs)
        ax_dist.yaxis.set_label_position("right")
        ax_dist.yaxis.tick_right()
        ax_dist.spines['left'].set_visible(False)
        ax_dist.spines['bottom'].set_visible(False)
        ax_dist.spines['right'].set_visible(True)
        ax_dist.spines['right'].set_linewidth(2)


        self.fig.tight_layout(pad=0.1)

        # save
        self.save('summary')
        return self.fig
