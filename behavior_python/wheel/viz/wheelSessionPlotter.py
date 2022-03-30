import scipy.stats 
from .wheelBasePlotter import *

class WheelSessionPlotter(WheelBasePlotter):
    """ A class to create wheel session related plots
        :param session      : A dictionary that contains all of the dta 
        :param savepath     : session directory path(usually in the analysis folder)
        :type session       : dict 
        :type session_meta  : dict
        :type savepath      : str
        """
    def __init__(self, session, savepath, *args, **kwargs):
        self.session = session
        self.fig = None
        self.savepath = savepath
        self.exp_name = self.savepath.split('/')[-1]

    def prep_text(self):
        """ Prepares summary text from session dictionaries"""
        text_meta = """META \n"""
        for k in ['animalid','date','rig','wheelgain','water_on_rig','rewardsize','level']:
            if k in self.session['meta'].__dict__.keys():
                text_meta =+ f'{k}  :  {getattr(self.session["meta"],k,None)}\n'

        summary_skips = ['EP', 'latency', ]
        text_summ = """SUMMARY \n"""
        for key in self.session['stats'].__slots__:
            if key not in summary_skips:
                text_summ =+ f'{key}  :  {getattr(self.session["stats"],key,None)}\n'
        return text_meta + """\n""" + text_summ

    def save(self,plotkey):
        """ Saves the figure
            :param plotkey : Name to be used when saving the figure
            :type plotkey  : str
        """
        figsave_loc = pjoin(self.savepath,'sessionFigures')
        if not os.path.exists(figsave_loc):
            os.mkdir(figsave_loc)

        savename = '{0}_{1}_{2}.pdf'.format(self.session['meta'].baredate,
                                            self.session['meta'].animalid,
                                            plotkey)
        savepath = pjoin(figsave_loc, savename)
        self.fig.savefig(savepath,bbox_inches='tight')
        display('{0} plot saved in {1}'.format(plotkey,self.savepath))
 
    def filter_data(self,data,filters):
        if filters is None:
            filters = {}
            filtered_data = data
        else:
            if 'trial_interval' in filters.keys():
                if isinstance(data,dict):
                    filtered_data = {k: v[(v['trial_no'] >= filters['trial_interval'][0]) & (v['trial_no'] <= filters['trial_interval'][1])] for k,v in data.items()}
                else:
                    filtered_data = data[(data['trial_no'] >= filters['trial_interval'][0]) & (data['trial_no'] <= filters['trial_interval'][1])]
            elif 'trial_interval_mins' in filters.keys():
                if isinstance(data,dict):
                    filtered_data = {k: v[(v['openstart_absolute'] >= filters['trial_interval_mins'][0]*60000) & (v['openstart_absolute'] <= filters['trial_interval_mins'][1]*60000)] for k,v in data.items()}
                else:
                    filtered_data = data[(data['openstart_absolute'] >= filters['trial_interval_mins'][0]*60000) & (data['openstart_absolute'] <= filters['trial_interval_mins'][1]*60000)]
            elif 'response_cutoff' in filters.keys():
                if isinstance(data,dict):
                    filtered_data = {data[k]: v[v['response_latency'] <= filters['response_cutoff']] for k,v in data.items()}
                else:
                    filtered_data = data[data['response_latency'] <= filters['response_cutoff']]
        return filtered_data

    def plot(self,plt_func,ax=None,func_params=None,*args,**kwargs):
        """ Main plot function that calls specific plotting functions
            :param plt_func  : function nam to be plotted
            :param close_fig : Flag to control closing figures
            :type plt_func   : str
            :type close_fig  : boolean
        """
        if func_params is None:
            func_params = []

        scope = kwargs.get('scope','all')
        filters = kwargs.get('filters',None)

        callables = []
        for name in dir(self):
            if not is_special(name):
                value = getattr(self, name)
                if callable(value):
                    callables.append(name)

        if plt_func not in callables:
            display('{0} not a plotter function for WheelSessionPlotter try {1}'.format(plt_func,', '.join(callables)))
            raise ValueError()

        print(filters)
        if filters is not None:
            for k in self.session.keys():
                if 'data' not in k:
                    continue
                print(k)
                self.session[k] = self.filter_data(self.session[k],filters=filters)
        else:
            filters = {}

        if 'Summary' in plt_func:
            ax = getattr(self,plt_func)(*func_params,*args,**kwargs)
        else:
            ax = getattr(self,plt_func)(ax=ax,*func_params,*args,**kwargs)

        # save
        if kwargs.get('savefig',True):
            self.save('{0}_{1}_{2}'.format(plt_func,scope,'_'.join([str(v) for v in filters.values()])))
        
        return self.fig

    def psychometric(self, ax=None, *args, **kwargs):
        """ Plots the psychometric curve for each stimuli type 
            :param ax        : a premade axes object to plot on 
            :type ax         : matplortlib.axes
            :return ax       : plotted axes
            :rtype           : matplotlib.axes
        """    
        fontsize = kwargs.get('fontsize',22)   
        side = kwargs.get('side','right')

        # # TODO: VERY CRUDE FILTERING
        # data_to_fit = {}
        # for k in self.session['novel_stim_data'].keys():
        #     try:
        #         print('Setting dta for curve fit')
        #         data_to_fit[k] = self.session['novel_stim_data'][k][self.session['novel_stim_data'][k]['running_response_latency'] <= self.session['summaries']['overall']['response_cutoff']]
        #     except:
        #         display('Not enough points for some contrasts in data to filter. Skipping filtering')
        #         data_to_fit = self.session['novel_stim_data']
        #         break
        self.analyzer.set_data(self.session['data'].get_novel_trials())
        fitted_data = self.analyzer.curve_fit(model='erf_psycho2',**kwargs)
        self.analyzer.run_analysis(analysis_type='mantel_haenzsel',*args,**kwargs)

        for i,p_key in enumerate(self.analyzer.pairs.keys()):
            pair = self.analyzer.pairs[p_key]
            p_value = self.analyzer.analysis['mantel_haenzsel_Q'][p_key]['results']['p_value'][0][0]
            # ax.text(0.5,-0.55-i/25,"{0} - {1}: p={2:.7f}".format(pair[0],pair[1],p_value),fontsize=18,ha='center')

        # set up axes
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(8,8)))
            ax = self.fig.add_subplot(1,1,1)
        
        # plotting
        # midlines 
        ax.plot([0, 0], [0, 1], 'gray', linestyle=':', linewidth=2,alpha=0.7)
        ax.plot([-100, 100], [0.5, 0.5], 'gray', linestyle=':', linewidth=2,alpha=0.7)

        scope_size = len(fitted_data.keys())
        if scope_size > 1:
            jit = np.random.choice(np.linspace(-scope_size/2,scope_size/2,scope_size),size=scope_size,replace=False)
        else:
            # no jutter if theres only one curve
            jit = np.zeros((1,scope_size))
        
        for i,scope in enumerate(fitted_data.keys()):
            fit_side_data = fitted_data[scope][side]

            # temporary jitter generation for data points
            if 'opto' in scope:
                for i,p_key in enumerate(self.analyzer.pairs.keys()):
                    pair = self.analyzer.pairs[p_key]
                    print(pair)
                    if pair[1]==scope:
                        p_value = self.analyzer.analysis['mantel_haenzsel_Q'][p_key]['results']['p_value'][0][0]
                        label = scope + '(p={0:.5f})'.format(p_value)
            else:
                label = scope
            
            ax.errorbar((100 * np.array(fit_side_data['contrast']))+jit[i], fit_side_data['percentage'], fit_side_data['confidence'],
                            marker=kwargs.get('marker','o'),
                            linewidth=0,
                            markersize=kwargs.get('markersize',15),
                            markeredgecolor=kwargs.get('markeredgecolor','w'),
                            markeredgewidth=kwargs.get('markeredgewidth',2),
                            elinewidth=kwargs.get('elinewidth',3),
                            capsize=kwargs.get('capsize',0),
                            label=label,
                            **stim_styles[scope])

            ax.plot(100 * fit_side_data['fitted_x'], fit_side_data['fitted_y'],
                    linewidth=kwargs.get('linewidth',9),
                    **stim_styles[scope])

        # make it pretty
        ax = self.pretty_axes(ax)
        ax.set_xlabel('Contrast Value', fontsize=fontsize)
        ax.set_ylabel('Prob. Choosing {0}'.format(side.capitalize()), fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.spines['left'].set_bounds(0, 1)
        ax.spines['bottom'].set_bounds(-100, 100)

        # some logic for embedding in summary figures
        if not kwargs.get('notitle',False):
            ax.set_title('{0} Psychometric Curve {1}'.format(self.session['meta']['animalid'],self.session['meta']['date']),
                            fontsize=fontsize+2,
                            fontweight='bold')
        
        ax.legend(loc='lower left',bbox_to_anchor=(0,1.02,1,0.102),mode='expand',borderaxespad=0,ncol=1,fontsize=fontsize-5)
        
        return ax

    def performance(self, ax=None, *args, **kwargs):
        """ Plots the performance change through the session 
            :param ax        : a premade axes object to plot on 
            :type ax         : matplotlib.axes
            :return ax       : plotted axes
            :rtype           : matplotlib.axes
        """       
        fontsize = kwargs.get('fontsize',22)
        side = kwargs.get('side','right')
        filters = kwargs.get('filters',None)  

        # set up axes
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(10,10)))
            ax = self.fig.add_subplot(1,1,1)

        # plotting
        for scope in self.session['data'].stim_data.keys():
            if scope == 'overall':
                continue
            data = self.session['data'].get_novel_trials(scope)
            time_in_secs = data['openstart_absolute'] / 60000
            performance_in_percent = data['fraction_correct'] * 100

            if kwargs.get('plot_in_time',True):
                x_axis_ = time_in_secs
                x_label_ = 'Time (min)'
            else:
                x_axis_ = data['trial_no']
                x_label_ = 'Trial No'

            perf_line = ax.plot(x_axis_,performance_in_percent,
                    linewidth=kwargs.get('linewidth',5),
                    label=scope,
                    **stim_styles[scope])

            if 'opto' in scope:
                perf_line[0].set_path_effects([path_effects.Stroke(linewidth=8, foreground="b", alpha=1),
                                                        path_effects.Normal()])

        # make it pretty
        ax.set_ylim([0, 100])
        ax.set_xlabel(x_label_, fontsize=fontsize)
        ax.set_ylabel('Accuracy(%)', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.grid(alpha=0.5,axis='both')

        # some logic for embedding in summary figures        
        if not kwargs.get('notitle',False):
            ax.set_title('{0} Session Performance {1}'.format(self.session['meta'].animalid,self.session['meta'].nicedate,), 
                         fontsize=fontsize+2,
                         fontweight='bold')
        if kwargs.get('showlegend',True):
            ax.legend(loc='upper right',fontsize=fontsize-5)

        return ax

    def responseLatency(self, sep_sides=False, ax=None, *args, **kwargs):
        """ Plots the response time change through the session, can plot for two sides seperately too
            :param sep_sides : boolean whether to seperate left/right responses when plotting 
            :param ax        : a premade axes object to plot on 
            :type sep_sides  : boolean
            :type ax         : matplotlib.axes
            :return ax       : plotted axes
            :rtype           : matplotlib.axes
        """
        fontsize = kwargs.get('fontsize',22)
        filters = kwargs.get('filters',None)  

        # set up axes
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(8,8)))
            ax = self.fig.add_subplot(1,1,1)
        
        for scope in self.session['data'].keys():
            if scope=='overall':
                continue
            data = self.session['data'][scope]
            data = self.filter_data(data,filters)
            data_dict = {}

            if sep_sides:
                data_dict['L'] = data[data['stim_side'] < 0]
                data_dict['R'] = data[data['stim_side'] > 0]
            else:
                data_dict['L+R'] = data

            #plotting
            for i,key in enumerate(data_dict.keys()):
                latency_in_secs = data_dict[key]['running_response_latency']/1000
                time_in_secs = data_dict[key]['openstart_absolute']/60000

                if kwargs.get('plot_in_time',True):
                    x_axis_ = time_in_secs
                    x_label_ = 'Time (min)'
                else:
                    x_axis_ = data_dict[key]['trial_no']
                    x_label_ = 'Trial No'
               
                resp_time = ax.plot(x_axis_,latency_in_secs,
                                    linewidth=kwargs.get('linewidth',5),
                                    label='{0}_{1} Median Response Time'.format(scope,key),
                                    **stim_styles[scope],
                                    linestyle='-' if i==0 else ':')

                if 'opto' in scope:
                    resp_time[0].set_path_effects([path_effects.Stroke(linewidth=7, foreground="b", alpha=1),
                                                                path_effects.Normal()])
            
        # plot the filter line
        cutoff = self.session['summaries']['overall']['response_cutoff']
        ax.plot(ax.get_xlim(),[cutoff,cutoff], linewidth=4,color='r',label='Response Cutoff({0:.2f} s)'.format(cutoff))
        # make it pretty
        ax = self.pretty_axes(ax)
        ax.set_yscale('log')
        ax.set_ylim([0.1,self.session['meta']['answertime']])
        ax.set_xlabel(x_label_, fontsize=fontsize)
        ax.set_ylabel('Response Times(s)', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.set_yticklabels([format(y,'.0f') for y in ax.get_yticks()])
        ax.grid(alpha=0.5,axis='both')
        
        # some logic for embedding in summary figures
        if not kwargs.get('notitle',False):
            ax.set_title('{0} Response Times {1}'.format(self.session['meta']['animalid'],self.session['meta']['date']), 
                         fontsize=fontsize,
                         fontweight='bold')
        if kwargs.get('showlegend',True):
            ax.legend(loc='lower right',fontsize=fontsize-5)

        return ax

    def responsePerStim(self,ax=None,*args,**kwargs):
        """ Rain cloud plot (half-violin and scatter) of response times per stimulus type""" 
        fontsize = kwargs.get('fontsize',22)
        filters = kwargs.get('filters',None)  
        scatter_offset = kwargs.get('scatter_offset',0.03)
         
        violin_width = kwargs.get('violin_width',1.5)
        opto_offset = kwargs.get('opto_offset',0.58) 
        xtic = []
        xtic_label = []

        # set up axes
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(15,10)))
            ax = self.fig.add_subplot(1,1,1)

        scope_distance_mult = 1.2 * (len(self.session['data'].keys()) - 1)
        for offset_mult, scope in enumerate(self.session['data'].keys(),start=1):
            if scope == 'overall':
                continue
            scope_data = self.session['data'][scope]
            voffset = offset_mult * 1/len(np.unique(scope_data['contrast']))

            for vpos, c in enumerate(np.unique(scope_data['contrast']),start=1):
                contrast_data = scope_data[scope_data['contrast']==c]
                for side in np.unique(contrast_data['stim_side']):
                    if c == 0:
                        pos = 0
                    else:
                        pos = scope_distance_mult * vpos + voffset
                    
                    side_data = contrast_data[contrast_data['stim_side']==side]
                    data = np.log10(side_data['response_latency']/1000)
                    data_x = np.zeros((len(data)))
                    data_x_jittered = data_x + np.random.normal(0,0.005,size=len(data)) 
                    if 'opto' in scope:
                        violin_pos = np.sign(side) * (pos-scatter_offset+opto_offset)
                    else:
                        violin_pos = np.sign(side) * (pos-scatter_offset)
                    v1 = ax.violinplot(data, points=100, positions=[violin_pos],widths=violin_width,
                                       showmeans=False, showextrema=False, showmedians=True)
                    for b in v1['bodies']:
                        # get the center
                        m = np.mean(b.get_paths()[0].vertices[:, 0])
                        # modify the paths to not go further right than the center
                        b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
                        b.set_color(contrast_styles[c]['color'])
                        if 'opto' in scope:
                            b.set_edgecolor('b')
                            b.set_linewidths(3)

                    if 'opto' in scope:
                        scatter_pos = opto_offset + pos + scatter_offset
                        ax.scatter(np.sign(side) * (data_x_jittered + scatter_pos),data,s=20,color=contrast_styles[c]['color'],alpha=0.6,label=c,
                        edgecolors='b',linewidth=2)
                    else:
                        scatter_pos = pos + scatter_offset
                        xtic.append(np.sign(side) * pos)
                        xtic_label.append(scope + ' ' + str(c))
                        ax.scatter(np.sign(side) * (data_x_jittered + scatter_pos),data,color=contrast_styles[c]['color'],alpha=0.6,label=c)

        # mid line
        ax.plot([0,0],ax.get_ylim(),color='gray',linewidth=2,alpha=0.8)
        # make it pretty
        ax = self.pretty_axes(ax)
        ax.set_xlabel('Stimulus Type', fontsize=fontsize)
        ax.set_ylabel('Response Time(ms)', fontsize=fontsize)
        ax.set_xticks(xtic)
        ax.set_xticklabels(xtic_label)
        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels(['{0:d}'.format(int(1000*10**a)) for a in ax.get_yticks()])
        ax.tick_params(labelsize=fontsize)
        ax.tick_params(axis='x',rotation=90)
        ax.grid(alpha=0.5,axis='y')
        
        # some logic for embedding in summary figures
        if not kwargs.get('notitle',False):
            ax.set_title('{0} Response Times {1}'.format(self.session['meta']['animalid'],self.session['meta']['date']), 
                         fontsize=fontsize,
                         fontweight='bold')
        if kwargs.get('showlegend',True):
            ax.legend(loc='upper left',fontsize=fontsize-5)

        return ax

    def performancePerStim(self,ax=None,*args,**kwargs):
        fontsize = kwargs.get('fontsize',22)
        filters = kwargs.get('filters',None) 
        opto_offset = kwargs.get('opto_offset',0.58)
        xtic = []
        xtic_label = []

        # set up axes
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(8,8)))
            ax = self.fig.add_subplot(1,1,1)
        scope_distance_mult = len(self.session['data'].keys()) - 1
        for offset_mult,scope in enumerate(self.session['data'].keys(),start=1):
            if scope == 'overall':
                continue
            scope_data = self.session['data'][scope]
            baroffset = offset_mult * 1/len(np.unique(scope_data['contrast']))
            for barpos,c in enumerate(np.unique(scope_data['contrast'])):
                contrast_data = scope_data[scope_data['contrast']==c]
                for side in np.unique(contrast_data['stim_side']):
                    if c == 0:
                        pos = 0
                    else:
                        pos = scope_distance_mult * barpos + baroffset
                    side_data = contrast_data[contrast_data['stim_side']==side]
                    correct = len(side_data[side_data['answer']==1])
                    nogo = len(side_data[side_data['answer']==0])
                    incorrect = len(side_data[side_data['answer']==-1])


                if 'opto' in scope:
                    pos = np.sign(side) * (pos+opto_offset)
                    ax.bar(pos,correct,width=0.7,color='darkgreen',edgecolor='b',linewidth=2)
                    ax.bar(pos,incorrect,bottom=correct,width=0.7,color='maroon',edgecolor='b',linewidth=2)
                    ax.bar(pos, nogo, bottom=correct+incorrect, width=0.7,color='k',edgecolor='b',linewidth=2)
                else:
                    pos = np.sign(side) * pos
                    xtic.append(pos)
                    xtic_label.append(scope + ' ' + str(c))
                    ax.bar(pos,correct,width=0.7,color='darkgreen')
                    ax.bar(pos,incorrect,bottom=correct,width=0.7,color='maroon')
                    ax.bar(pos, nogo, bottom=correct+incorrect, width=0.7,color='k')

        # mid line
        ax.plot([0,0],ax.get_ylim(),color='gray',linewidth=2,alpha=0.8)

        # make it pretty
        ax = self.pretty_axes(ax)
        ax.set_xlabel('Stimulus Type', fontsize=fontsize)
        ax.set_ylabel('Answer Count', fontsize=fontsize)
        ax.set_xticks(xtic)
        ax.set_xticklabels(xtic_label)
        ax.tick_params(labelsize=fontsize)
        ax.tick_params(axis='x',rotation=45)
        ax.grid(alpha=0.5,axis='y')
        
        # some logic for embedding in summary figures
        if not kwargs.get('notitle',False):
            ax.set_title('{0} Answer Distribution {1}'.format(self.session['meta']['animalid'],self.session['meta']['date']), 
                         fontsize=fontsize,
                         fontweight='bold')
        if kwargs.get('showlegend',True):
            ax.legend(loc='upper left',fontsize=fontsize-5)

        return ax
    # This is not so good and/or useful
    def probability(self,ax=None,*args,**kwargs):
        """ Plots the stim probability change through the session, can plot for two sides seperately too
            :param ax        : a premade axes object to plot on 
            :type ax         : matplotlib.axes
            :return ax       : plotted axes
            :rtype           : matplotlib.axes
        """
        fontsize = kwargs.get('fontsize',22)
        scope = kwargs.get('scope','novel')
        filters = kwargs.get('filters',None)  

        scope_data = self.set_scope(scope)
        data = self.filter_data(scope_data,filters)

        # set up the axes
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(16,8)))
            ax = self.fig.add_subplot(1,1,1)

        # plotting
        probs_left = data[data['stim_side'] < 0]['running_prob']
        probs_right = data[data['stim_side'] > 0]['running_prob']

        time_left = data[data['stim_side'] < 0]['openstart_absolute']/60000
        time_right = data[data['stim_side'] > 0]['openstart_absolute']/60000

        ax.plot(time_left,probs_left*100,
                color='teal',
                linewidth=kwargs.get('linewidth',5),
                label='Prob. Left')

        ax.plot(time_right,probs_right*100,
                color='firebrick',
                linewidth=kwargs.get('linewidth',5),
                label='Prob. Right')
        
        # make it pretty
        ax = self.pretty_axes(ax)
        ax.set_xlabel('Time (min)', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.grid(alpha=0.5,axis='both')

        # some logic for embedding in summary figures
        if not kwargs.get('barebones',False):
            ax.set_ylabel('Probability', fontsize=fontsize)
            if not kwargs.get('notitle',False):
                ax.set_title('{0} Stimulus Side probabilities {1}'.format(self.session['meta']['animalid'],self.session['meta']['date']), 
                             fontsize=fontsize,
                             fontweight='bold')
            if kwargs.get('showlegend',True):
                ax.legend(fontsize=fontsize-5)

        return ax

    def parameterCompare(self,p1,p2,ax=None,*args,**kwargs):
        """ Compares two parameters in a scatter plot with a linear fit
            :param p1  : name of x-axis parameter
            :param p2  : name of y-axis parameter
            :param ax  : a premade axes object to plot on 
            :type p1   : str
            :type p2   : str
            :type ax   : matplotlib.axes
            :return ax : plotted axes
            :rtype     : matplotlib.axes
        """
        fontsize = kwargs.get('fontsize',22)
        scope = kwargs.get('scope','novel')
        filters = kwargs.get('filters',None)  

        scope_data = self.set_scope(scope)
        data = self.filter_data(scope_data,filters)

        # set up the axes
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(8,8)))
            ax = self.fig.add_subplot(1,1,1)

        if p1 not in data.columns:
            raise ValueError('No parameter named {0} in data'.format(p1))
        if p2 not in data.columns:
            raise ValueError('No parameter named {0} in data'.format(p2))

        # fit
        m, b, r, p ,stderr = scipy.stats.linregress(data[p1],data[p2])
        fit_line = b + m * data[p1]

        # plotting
        ax.scatter(data[p1],data[p2],
                   c='crimson',
                   s=kwargs.get('markersize',25))

        ax.plot(data[p1],fit_line,
                linewidth = kwargs.get('linewidth',2),
                color = 'k',
                label = 'r={0:.2f}'.format(r))

        # make it pretty
        ax = self.pretty_axes(ax)
        ax.set_xlabel(p1, fontsize=fontsize)
        ax.set_ylabel('Probability', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.grid(alpha=0.5,axis='both')
        ax.legend(fontsize=fontsize-5)

        # some logic for embedding in summary figures
        if not kwargs.get('notitle',False):
            ax.set_title('{0}  {1}'.format(self.session['meta']['animalid'],self.session['meta']['date']), 
                         fontsize=fontsize,
                         fontweight='bold')
        if kwargs.get('showlegend',True):
            ax.legend(fontsize=fontsize-5)

        return ax

    def parameterFractions(self,param_list,ax=None,*args,**kwargs):
        """ Compares two parameters in a scatter plot with a linear fit
            :param param_list : list of fraction_parameters to plot
            :param colors     : list of colors for lines
            :param ax         : a premade axes object to plot on 
            :type param_list  : list
            :type colors      : list
            :type ax          : matplotlib.axes
            :return ax        : plotted axes
            :rtype            : matplotlib.axes
        """
        fontsize = kwargs.get('fontsize',22)
        filters = kwargs.get('filters',None)
        colors = kwargs.get('colors',None) #meh
        
        # set up the axes
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(16,8)))
            ax = self.fig.add_subplot(1,1,1)

        for scope in self.session['data'].keys():
            if scope=='overall':
                continue
            data = self.session['data'][scope]
            data = self.filter_data(data,filters)

            # plotting
            if kwargs.get('drawtrials',False):
                for row in data.itertuples():
                    if row.answer != 1:
                        if row.answer == -1:
                            color = 'red'
                        elif row.answer == 0:
                            color = 'dimgrey'

                        ax.plot([row.trial_no,row.trial_no],[0,1],
                                linewidth=1.3, color=color,alpha=0.5)

            for i,param in enumerate(param_list):
                if 'fraction' not in param:
                    display('PARAMETER {0} IS NOT A FRACTION, Y-AXIS IS INCOMPATIBLE'.format(param))

                if kwargs.get('plot_in_time',True):
                    x_axis_ = data['openstart_absolute'] / 60000
                    x_label_ = 'Time (min)'
                else:
                    x_axis_ = data['trial_no']
                    x_label_ = 'Trial No'

                frac = ax.plot(x_axis_,data[param],
                        linewidth=kwargs.get('linewidth',5),
                        label='{0}_{1}'.format(scope,param),
                        # **stim_styles[scope],
                        color=stim_styles[scope]['color'] if colors is None else colors[i],
                        linestyle='-' if i==0 else ':')
                
                if 'opto' in scope:
                    frac[0].set_path_effects([path_effects.Stroke(linewidth=7, foreground="b", alpha=1),
                                                                path_effects.Normal()])

        # make it pretty
        ax = self.pretty_axes(ax)
        ax.spines['left'].set_position(('outward', 10)) 
        ax.spines['bottom'].set_position(('outward', 10))
        ax.set_ylim([0, 1])
        ax.set_xlabel(x_label_, fontsize=fontsize)
        ax.set_ylabel('Fraction', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.grid(alpha=0.8,axis='both')

        # some logic for embedding in summary figures
        if not kwargs.get('notitle',False):
            ax.set_title('{0} {1}'.format(self.session['meta']['animalid'],self.session['meta']['date']), 
                         fontsize=fontsize+2,
                         fontweight='bold')
        if kwargs.get('showlegend',True):
            ax.legend(fontsize=fontsize-5)
        return ax

    def wheelTrajectory_vert(self,seperate_by='contrast',ax=None,*args,**kwargs):
        """ Plots the wheel trajectory in vertical mode and seperating 
            :param seperate_by : Propert to seperate the wheel traces by (corresponds to a column name in the session data)
            :param ax          : a premade axes object to plot on 
            :type seperate_by  : str
            :type ax           : matplotlib.axes
            :return ax         : plotted axes
            :rtype             : matplotlib.axes
        """

        fontsize = kwargs.get('fontsize',20)
        s_limit = kwargs.get('s_limit',2000)
        trial_interval = kwargs.get('trial_interval',None)
        filters = kwargs.get('filters',None)
        show_individual = kwargs.get('show_individual',False)

        # set up axes
        if ax is None:
            self.fig = plt.figure(figsize=(10,10))
            ax = self.fig.add_subplot(1,1,1)

        for scope in self.session['data'].keys():
            if scope == 'overall':
                continue
            data = self.session['data'][scope]
            
            if seperate_by is not None:
                sep_list = np.unique(data[seperate_by])
            else:
                sep_list = [1]
    
            sides = np.unique(data['stim_side'])
            for i,s in enumerate(sides,1):
                side_slice =  data[data['stim_side'] == s]

                for sep in sep_list:
                    if seperate_by is None:
                        temp_slice = side_slice
                    else:
                        temp_slice = side_slice[side_slice[seperate_by] == sep]

                    # shift wheel according to side
                    temp_slice.loc[:,'wheel'] = temp_slice['wheel'].apply(lambda x: x+s)
                    # temp_slice.loc[:,'wheel'] = temp_slice.loc[:,'wheel'] + s

                    wheel_stat_dict = get_trajectory_stats(temp_slice)
                    avg = wheel_stat_dict['average']

                    # plotting
                    if show_individual:               
                        for k,trial in enumerate(temp_slice.itertuples()):
                            if trial_interval is not None:
                                if trial.trial_no >= trial_interval[0] and trial.trial_no <= trial_interval[1]:
                                    
                                    # individual trajectories
                                    indiv = trial.wheel
                                    if len(indiv):
                                        if s_limit is not None:

                                            indiv = indiv[find_nearest(indiv[:,0],-200)[0]:find_nearest(indiv[:,0],s_limit)[0],:]
                                        
                                        indiv_line = ax.plot(indiv[:,1], indiv[:,0],
                                                            linewidth=3,
                                                            **contrast_styles[sep],
                                                            label = '{0} {1}'.format(seperate_by,sep) if k==1 else "_",
                                                            alpha=0.5,
                                                            zorder=1)
                                        if trial.opto == 1:
                                            indiv_line[0].set_path_effects([path_effects.Stroke(linewidth=2.5, foreground="b", alpha=0.3),
                                                                        path_effects.Normal()])

                    if s_limit is not None:
                        # plot for values between -200 ms and s_limit
                        if avg is not None and trial_interval is None:
                            avg = avg[find_nearest(avg[:,0],-200)[0]:find_nearest(avg[:,0],s_limit)[0]]

                            #avg_line
                            avg_line = ax.plot(avg[:,1], avg[:,0],
                                            linewidth=kwargs.get('linewidth',5),
                                            **contrast_styles[sep], 
                                            label='{0} {1}'.format(seperate_by,sep) if i==1 else "_",
                                            alpha=1,
                                            zorder=2)

                            if 'opto' in scope:
                                avg_line[0].set_path_effects([path_effects.Stroke(linewidth=8, foreground="b", alpha=1),
                                                                path_effects.Normal()])


        ax.set_ylim([-500, s_limit+100])
        ax.set_xlim([-75, 75])
        # closed loop start line
        ax.plot(ax.get_xlim(),[0,0],'k',linewidth=2, alpha=0.8)

        # trigger zones
        ax.plot([0,0], ax.get_ylim(), 'green', linestyle='--', linewidth=2,alpha=0.8)

        ax.plot([-50,-50], ax.get_ylim(), 'maroon', linestyle='--', linewidth=2,alpha=0.8)
        ax.plot([50,50], ax.get_ylim(), 'maroon', linestyle='--', linewidth=2,alpha=0.8)
        
        # make it pretty
        ax = self.pretty_axes(ax)
        ax.yaxis.set_label_position('right')
        ax.yaxis.tick_right()
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        ax.set_xlabel('Wheel Position (deg)', fontsize=fontsize)
        ax.set_ylabel('Time(ms)', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.grid(axis='y')

        # some logic for embedding in summary figures
        if not kwargs.get('notitle',False):
            ax.set_title('{0}  Wheel Trajectory {1}'.format(self.session['meta']['animalid'],self.session['meta']['date']), 
                         fontsize=fontsize+2,
                         fontweight='bold')
        if kwargs.get('showlegend',True):
            ax.legend(fontsize=fontsize-5)

        return ax

    def metric(self,metric_name,ax=None,*args,**kwargs):
        """ Plot a single metric with respec to trials
            :param metric_name : name of the metric to be plotted
            :param ax          : a premade axes object to plot on 
            :type metric_name  : str
            :type ax           : matplotlib.axes
            :return ax         : plotted axes
            :rtype             : matplotlib.axes
        """
        fontsize = kwargs.get('fontsize',22)
        scope = kwargs.get('scope','novel')
        filters = kwargs.get('filters',None)  

        scope_data = self.set_scope(scope)
        data = self.filter_data(scope_data,filters)
        
        # set up the axes
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(10,10)))
            ax = self.fig.add_subplot(1,1,1)
            show_legend = True

        # plotting
        ax.plot(data['trial_no'],data['running_'+metric_name],
                linewidth=kwargs.get('linewidth',4),
                color=kwargs.get('color','k'))

        ax.scatter(data['trial_no'],data[metric_name],
                   c=kwargs.get('color','gray'),
                   s=25,
                   alpha=0.4)

        if metric_name=='reaction_t':
            ax.set_yscale('log')
        #zero line
        ax.plot(ax.get_xlim(),[0,0],linewidth=1,color='k')
        
        # make it pretty
        if metric_name == 'path_surplus':
            y_label = 'Path Surplus [norm.]'
        elif metric_name == 'reaction_t':
            y_label = 'Reaction Time (ms)'
        elif metric_name == 'avg_lick_t_diff':
            y_label = 'Relative Lick Time (ms)'
        elif metric_name == 'avg_speed':
            y_label = 'Avg. Stim Speed (deg/ms)'

        ax = self.pretty_axes(ax)
        ax.set_xlabel('Trial No.', fontsize=fontsize)
        ax.set_ylabel(y_label, fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.grid(alpha=0.5,axis='both')

        return ax

    def metricSummary(self, *args, **kwargs):
        """ Plots a bunch of graphs that summarize the session
            :return fig      : final figure
            :rtype           : matplotlib.figure
        """
        fontsize = kwargs.get('fontsize',22)
        scope = kwargs.get('scope','novel')
        filters = kwargs.get('filters',None)  

        scope_data = self.set_scope(scope)
        data = self.filter_data(scope_data,filters)

        self.fig = plt.figure(figsize = kwargs.get('figsize',(15,15)))

        ax1 = self.fig.add_subplot(221)
        ax1 = self.metric('path_surplus', data, ax=ax1,notitle=True,savefig=False,*args,**kwargs)

        ax2 = self.fig.add_subplot(222)
        ax2 = self.metric('reaction_t', data, ax=ax2, notitle=True,savefig=False,*args,**kwargs)

        ax3 = self.fig.add_subplot(223)
        ax3 = self.metric('avg_lick_t_diff', data, ax=ax3, notitle=True,savefig=False,*args,**kwargs)

        ax4 = self.fig.add_subplot(224)
        ax4 = self.metric('avg_speed', data, ax=ax4, notitle=True,savefig=False,*args,**kwargs)

        self.fig.suptitle('{0} Session Metrics {1}'.format(self.session['meta']['animalid'],self.session['meta']['date']),
                     fontsize=fontsize+3,fontweight='bold')
        self.save('metrics_{0}'.format(kwargs.get('scope','all')))
        return self.fig

    def metricMatrix(self, *args, **kwargs):
        pass

    def lickTotal(self,ax=None,*args,**kwargs):
        """ Plots the total licks through the trial
            :param ax        : a premade axes object to plot on 
            :type ax         : matplotlib.axes
            :return ax       : plotted axes
            :rtype           : matplotlib.axes
        """
        fontsize = kwargs.get('fontsize',22)
        scope = kwargs.get('scope','novel')
        filters = kwargs.get('filters',None)  

        scope_data = self.set_scope(scope)
        data = self.filter_data(scope_data,filters)

        # set up the axis
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(16,8)))
            ax = self.fig.add_subplot(1,1,1)

        # pool lick data
        all_lick = np.array([]).reshape(-1,2)
        for row in data.itertuples():
            if len(row.lick):
                temp_lick = row.lick.copy()
                temp_lick[:,0] =+ row.openstart_absolute
                all_lick = np.append(all_lick,temp_lick,axis=0)

        # plotting
        if len(all_lick):
            if kwargs.get('plot_in_time',True):
                ax.plot(all_lick[:,0]/60000,all_lick[:,1],linewidth=4,color='c',label='Licks')
            else:
                all_lick[:,0] = (all_lick[:,0]/np.max(all_lick[:,0])) * data['trial_no'].iloc[-1]
                trial_axis = np.interp(data['trial_no'],all_lick[:,0],all_lick[:,1])
                ax.plot(data['trial_no'],trial_axis,linewidth=4,color='c',label='Licks')
        else:
            display('No Lick data found for session :(')

        # make it pretty
        ax = self.pretty_axes(ax)
        ax.grid(b=True,alpha=0.5)
        ax.tick_params(labelsize=fontsize)
        ax.set_xlabel('Trial No', fontsize=fontsize)
        ax.set_ylim([0,20000])
        ax.set_yticklabels(['{0}k'.format(int(a/1000)) for a in ax.get_yticks()])
        ax.set_ylabel('Total Licks', fontsize=fontsize)
        

        # some logic for embedding in summary figures
        if not kwargs.get('notitle',False):
            ax.set_title('{0} Lick Progression {1}'.format(self.session['meta']['animalid'],self.session['meta']['date']), 
                         fontsize=fontsize,
                         fontweight='bold')
        if kwargs.get('showlegend',True):
            ax.legend(fontsize=fontsize-5)

        return ax

    def lickHistogram(self, bin_dur=500, time_range=[-5000,5000], ax=None, *args, **kwargs):
        """ Plots the reward triggered histogram of licks in correct trials and total licks through the trial
            :param bin_dur   : histogram bin duration in ms or matplotlib hist_methods
            :param ax        : a premade axes object to plot on 
            :type bin_dur    : int, float
            :type ax         : matplotlib.axes
            :return ax       : plotted axes
            :rtype           : matplotlib.axes
        """
        fontsize = kwargs.get('fontsize',22)
        scope = kwargs.get('scope','novel')
        filters = kwargs.get('filters',None)  

        scope_data = self.set_scope(scope)
        data = self.filter_data(scope_data,filters)
        
        # set up the axis
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(16,8)))
            ax = self.fig.add_subplot(1,1,1)

        # pool the lick data 
        hist_lick = {'correct' : np.array([]),
                     'incorrect' : np.array([])}
        hist_lick_incorrect = np.array([])
        for row in data.itertuples():
            if len(row.lick):
                if row.answer == 1:
                    reward_time = row.reward[0][0] if len(row.reward) else row.closedloopdur[1] + 500
                    hist_lick['correct'] = np.append(hist_lick['correct'],row.lick[:,0] - reward_time)
                else:
                    reward_time = row.closedloopdur[1] + 500
                    hist_lick['incorrect'] = np.append(hist_lick['incorrect'],row.lick[:,0] - reward_time)
        total_len = np.sum([len(x) for x in hist_lick.values()])

        # plotting
        for key in hist_lick.keys():
            if len(hist_lick[key]):

                hist_range = np.array([x for x in hist_lick[key] if x>=time_range[0] and x<=time_range[1]])

                bins_l = -1*np.arange(0,np.abs(np.min(hist_range)),bin_dur)
                bins_r = np.arange(1,np.max(hist_range),bin_dur)
                bins_l = bins_l[::-1]
                bins = np.append(bins_l, bins_r)

                weights = np.ones_like(hist_range) / total_len
                
                ax.hist(hist_range,bins=bins,weights=weights,
                    color='cyan' if key=='correct' else 'darkcyan',
                    alpha=0.5 if key=='incorrect' else 1,
                    rwidth=0.9,label=key)
            else:
                display('No Lick data found for session :(')

        # reward line
        ax.plot([0,0],ax.get_ylim(),color='r',linewidth=3,label='Reward')

        # avg_first_lick = float(np.nanmean(data.loc[~np.isnan(data['first_lick_t']),'first_lick_t']))
        # axes[0].plot([avg_first_lick,avg_first_lick],axes[0].get_ylim(),color='gray',linewidth=1.5, linestyle=':',label='Avg. First Lick')
        ax.set_xlim(time_range[0]-100,time_range[1]+100)

        ax.tick_params(labelsize=fontsize)
            
        # make it pretty
        ax = self.pretty_axes(ax)
        ax.set_ylabel('Norm. Lick Counts', fontsize=fontsize)
        ax.set_xlabel('Time (ms)', fontsize=fontsize)
        ax.spines['bottom'].set_bounds(ax.set_xlim()[0],ax.set_xlim()[1])
        ax.spines['bottom'].set_position(('outward', 10))
        ax.spines['left'].set_position(('outward', 10)) 

        # some logic for embedding in summary figures
        if not kwargs.get('notitle',False):
            ax.set_title('{0} Lick Histogram {1}'.format(self.session['meta']['animalid'],self.session['meta']['date']), 
                         fontsize=fontsize,
                         fontweight='bold')
        if kwargs.get('showlegend',True):
            ax.legend(bbox_to_anchor=(0,1.02,1,0.102),mode='expand',borderaxespad=0,ncol=1,fontsize=fontsize-5)

        return ax
        
    def sessionSummary(self, *args, **kwargs):
        """ Plots a dashboard of graphs that summarize the session
            :return fig      : final figure
            :rtype           : matplotlib.figure
        """
        fontsize = kwargs.get('fontsize',22)

        self.fig = plt.figure(figsize = kwargs.get('figsize',(20,10)))
        widths = [2,2,1,1,1]
        heights = [1,1]
        gs = self.fig.add_gridspec(ncols=5, nrows=2, width_ratios=widths,
                                  height_ratios=heights,left=kwargs.get('left',0),right=kwargs.get('right',1),wspace=kwargs.get('wspace',0.75),hspace=kwargs.get('hspace',0.7))
        # info text
        ax_text = self.fig.add_subplot(gs[0,4])

        text = self.prep_text()
        ax_text.text(0.01,0.93, text, va='top', fontsize=fontsize-5)
        ax_text = self.empty_axes(ax_text)
        
        gs_in1 = gs[:,0].subgridspec(nrows=2,ncols=1,hspace=0.2)

        # fractions
        ax_frac = self.fig.add_subplot(gs[0,0])
        ax_frac = self.parameterFractions(param_list=['fraction_repeat','fraction_stim_right'],
                                          ax=ax_frac,colors=['olive','teal'], 
                                          drawtrials=False, notitle=True, showlegend=False,*args,**kwargs)
        h_frac,l_frac = ax_frac.get_legend_handles_labels()

        # response latency
        ax_time = ax_frac.twinx()
        ax_time = self.responseLatency(sep_sides=False,
                                       ax=ax_time, notitle=True, showlegend=False,*args,**kwargs)
        h_time,l_time = ax_time.get_legend_handles_labels()
        
        # make two axes fit nicely
        h_frac.extend(h_time)
        l_frac.extend(l_time)
        ax_time.spines['bottom'].set_visible(False)
        ax_time.spines['left'].set_visible(False)
        ax_time.spines['right'].set_visible(True)
        ax_time.spines['right'].set_linewidth(2)
        ax_time.grid(axis='y',b=False)
        ax_time.legend(h_frac,l_frac,loc='lower left',bbox_to_anchor=(0,1.02,1,0.102),mode='expand',borderaxespad=0,ncol=1,fontsize=fontsize-10)

        # performance progression
        ax_perf = self.fig.add_subplot(gs_in1[1,0],sharex=ax_time)
        ax_perf = self.parameterFractions(param_list=['fraction_correct','fraction_nogo'],
                                          ax=ax_perf, notitle=True, showlegend=False,*args,**kwargs)
        h_perf,l_perf = ax_perf.get_legend_handles_labels()

        # total licks
        ax_cumu = ax_perf.twinx()
        ax_cumu = self.lickTotal(ax=ax_cumu, notitle=True, showlegend=False,*args,**kwargs)
        h_cumu,l_cumu = ax_cumu.get_legend_handles_labels()

        # make two axes fit nicely
        h_perf.extend(h_cumu)
        l_perf.extend(l_cumu)
        ax_cumu.spines['bottom'].set_visible(False)
        ax_cumu.spines['left'].set_visible(False)
        ax_cumu.spines['right'].set_visible(True)
        ax_cumu.spines['right'].set_linewidth(2)
        ax_cumu.grid(axis='y',b=False)
        ax_cumu.legend(h_perf,l_perf,loc='lower left',bbox_to_anchor=(0,1.02,1,0.102),mode='expand',borderaxespad=0,ncol=1,fontsize=fontsize-10)


        gs_in2 = gs[:,1].subgridspec(nrows=2,ncols=1,hspace=0.2)

        # reponse violin
        ax_violin = self.fig.add_subplot(gs_in2[0,0])
        ax_violin = self.responsePerStim(ax=ax_violin, notitle=True, showlegend=False,savefig=False,*args,**kwargs)
        ax_violin.spines['bottom'].set_visible(False)
        ax_violin.grid(b=True,axis='both')
        ax_violin.tick_params(bottom=False,labelbottom=False)
        ax_violin.set(xlabel=None)

        # performance bar
        ax_bars = self.fig.add_subplot(gs_in2[1,0],sharex=ax_violin)
        ax_bars = self.performancePerStim(ax=ax_bars,notitle=True,savefig=False,*args,**kwargs)

        # psychometric
        ax_psycho = self.fig.add_subplot(gs[0,2])
        ax_psycho = self.psychometric(ax=ax_psycho, 
                                      notitle=True, showlegend=False,markersize=10,linewidth=7,elinewidth=1.5,*args,**kwargs)

        # lick histogram
        ax_hist = self.fig.add_subplot(gs[1,2])
        ax_hist = self.lickHistogram(ax=ax_hist, notitle=True,*args,**kwargs)

        
        # wheel
        ax_wheel = self.fig.add_subplot(gs[:,3])
        ax_wheel = self.wheelTrajectory_vert(ax=ax_wheel, notitle=True,*args,**kwargs)

        # self.fig.tight_layout(h_pad=0.01,w_pad=0.01)