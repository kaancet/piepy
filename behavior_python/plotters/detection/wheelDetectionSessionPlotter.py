from ..basePlotters import *
from scipy.stats import fisher_exact, barnard_exact


class DetectionPsychometricPlotter(BasePlotter):
    def __init__(self, data:dict, **kwargs) -> None:
        super().__init__(data,**kwargs)
        self.stat_analysis = DetectionAnalysis(data=data)
        self.hit_rate_dict = self.stat_analysis.get_hitrates()
        
    @staticmethod
    def __plot__(ax,x,y,err,**kwargs):
        """ Private function that plots a psychometric curve with the given 
        x,y and err values are used to plot the points and 
        x_fit and y_fit values are used to plot the fitted curve
        """
        ax.errorbar(x, y, err,
                    linewidth=2,
                    markeredgecolor=kwargs.get('markeredgecolor','w'),
                    markeredgewidth=kwargs.get('markeredgewidth',2),
                    elinewidth=kwargs.get('elinewidth',3),
                    capsize=kwargs.get('capsize',0),
                    **kwargs)

        return ax
    
    @staticmethod
    def _dict2label(d:dict) -> str:
        ret = f'''\nN=['''
        for k,v in d.items():
            ret += fr'''{float(k)*100}:$\bf{v}$, '''
        ret += ''']'''
        return ret
                   
    def plot(self,ax:plt.Axes=None,color=None,seperate_sides:bool=False,jitter:int=2,**kwargs):
        """ Plots the hit rates with 95% confidence intervals"""
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(8,8)))
            ax = self.fig.add_subplot(1,1,1)
            if 'figsize' in kwargs:
                kwargs.pop('figsize')
        
        non_opto_key = [k for k in self.data.keys() if 'opto' not in k][0]
        opto_keys = [k for k in self.data.keys() if 'opto' in k]
        
        self.p_values = {}
        for o_k in opto_keys:
            p_values_contra = self.stat_analysis.get_hitrate_pvalues_exact(stim_data_keys=[non_opto_key,o_k])
            p_values_catch = self.stat_analysis.get_hitrate_pvalues_exact(stim_data_keys=[non_opto_key,o_k],stim_side='catch')
            self.p_values[o_k] = {**p_values_contra,**p_values_catch}
        
        for k,v in self.hit_rate_dict.items():
            if not seperate_sides:
                data = v['nonsided']
                jittered_offset = np.array([np.random.uniform(0,jitter)*c for c in data['contrasts']])
                jittered_offset[0] += np.random.uniform(0,jitter)/100
                ax = self.__plot__(ax,
                                   (100*data['contrasts'])+jittered_offset,
                                   data['hit_rate'],
                                   data['confs'],
                                   label=f"{k}{self._dict2label(data['counts'])}",
                                   marker = 'o',
                                   markersize=18,
                                   color = self.color.stim_keys[k]['color'] if color is None else color,
                                   linestyle = self.color.stim_keys[k]['linestyle'],
                                   **kwargs)
        
                if 'opto' not in k and 0 in data['contrasts']:
                    # draw the baseline only on non-opto
                    idx_0 = np.where(data['contrasts']==0)[0][0]
                    ax.plot([0, 100], [data['hit_rate'][idx_0], data['hit_rate'][idx_0]], 'k', linestyle=':', linewidth=2,alpha=0.7)
                
            
            else:
                sides = [k for k in v.keys() if isinstance(k,float)]
                for j,side in enumerate(sides):
                    side_data = v[side]
 
                    label = f"{k}{self._dict2label(side_data['counts'])}"
                    if side < 0:
                        init_color = self.color.name2hsv(self.color.stim_keys[k]['color'])
                        # make color lighter here
                        color = self.color.lighten(init_color,l_coeff=0.5)
                        marker = '<'
                        markersize = 24
                    elif side > 0:
                        color = self.color.stim_keys[k]['color']
                        marker = '>'
                        markersize = 24
                    else:
                        color = self.color.stim_keys[k]['color']
                        marker = 'o'
                        markersize = 18
                        
                        if 'opto' not in k:
                            # draw the baseline only on non-opto
                            ax.plot([0, 100], [side_data['hit_rate'], side_data['hit_rate']], 'k', linestyle=':', linewidth=2,alpha=0.7)
                        
                    jittered_offset = np.array([np.random.uniform(0,jitter)*c for c in side_data['contrasts']])
                    jittered_offset[0] += np.random.uniform(0,jitter)/100
                    ax = self.__plot__(ax,
                                    100*side_data['contrasts']+jittered_offset,
                                    side_data['hit_rate'],
                                    side_data['confs'],
                                    label=label,
                                    marker = marker,
                                    markersize=markersize,
                                    color=color,
                                    linestyle=self.color.stim_keys[k].get('linestyle','-')) 
                    
                    # if 'opto' not in k and 0 in side_data['contrasts']:
                    # # draw the baseline only on non-opto
                    #     idx_0 = np.where(side_data['contrasts']==0)[0][0]
                    #     ax.plot([0, 100], [side_data['hit_rate'][idx_0], side_data['hit_rate'][idx_0]], 'k', linestyle=':', linewidth=2,alpha=0.7)
        
        # prettify
        fontsize = kwargs.get('fontsize',15)
               
        # put the significance starts
        for i,k in enumerate(self.p_values.keys()):
            for c,p in self.p_values[k].items():
                stars = ''
                if p < 0.001:
                    stars = '***'
                elif 0.001 < p < 0.01:
                    stars = '**'
                elif 0.01 < p < 0.05:
                    stars = '*'
                    
                ax.text(100*c, 1.04+0.02*i, stars,color=self.color.stim_keys[k]['color'], fontsize=30)
        
        ax.set_xscale('symlog')
        # ax.xaxis.set_major_locator(ticker.FixedLocator([int(100*c) for c in contrast_list]))
        # ax.xaxis.set_major_locator(ticker.LogLocator(base=10,numticks=15))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0,subs=np.linspace(0.1,1,9,endpoint=False)))
        
        ax.set_ylim([0,1.05]) 
        
        ax.set_xlabel('Contrast Value (%)', fontsize=fontsize)
        ax.set_ylabel('Hit Rate (%)',fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        
        ax.spines['left'].set_bounds(0, 1) 
        # ax.spines['bottom'].set_bounds(0, 1)
        ax.spines['bottom'].set_position(('outward', 10))
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        ax.grid(alpha=0.4)
        ax.legend(loc='center left',bbox_to_anchor=(1,0.5),fontsize=fontsize,frameon=False)
        
        return ax


class DetectionPsychometricBarPlotter(BasePlotter):
    def __init__(self, data:dict, **kwargs) -> None:
        super().__init__(data,**kwargs)
        self.stat_analysis = DetectionAnalysis(data=data)
        self.hit_rate_dict = self.stat_analysis.get_hitrates()
    
    @staticmethod
    def _dict2label(d:dict) -> str:
        ret = f'''\nN=['''
        for k,v in d.items():
            sk = ''
            if k<0:
                sk = 'L'
            elif k>0:
                sk = 'R'
            ret += fr'''{sk}{float(k)*100}:$\bf{v}$, '''
        ret += ''']'''
        return ret
    
    def plot(self,ax=None,**kwargs):
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(8,8)))
            ax = self.fig.add_subplot(1,1,1)
            if 'figsize' in kwargs:
                kwargs.pop('figsize')
                
        if 'key_pairs' in kwargs:
            key_pairs = kwargs.pop('key_pairs')
        
        contrast_spacer = kwargs.get('contrast_spacer',0.2)
        total_bar_width = kwargs.get('total_bar_width',0.8)
        bar_width = total_bar_width / len(self.hit_rate_dict.keys())
        keys_barpos = np.linspace(-(total_bar_width-bar_width)/2,(total_bar_width-bar_width)/2,len(self.hit_rate_dict.keys()))
        center_barpos_dict = {}
        for ki,k in enumerate(self.hit_rate_dict.keys()):
            v = self.hit_rate_dict[k]
            sides = [k for k in v.keys() if isinstance(k,float)]
            counts4labels = {}
            for j,side in enumerate(sides):
                side_data = v[side]
                side_counts = {np.sign(side)*k:v for k,v in side_data['counts'].items()}
                counts4labels = {**counts4labels,**side_counts}
                if side == 0:
                    contrast_barpos = [0]
                else:
                    contrast_barpos = [np.sign(side)*(i+1)*(total_bar_width+contrast_spacer) for i in range(len(side_data['contrasts']))]
                    if side<0:
                        contrast_barpos = contrast_barpos[::-1]
                
                temp_dict = {np.sign(side)*side_data['contrasts'][i]:contrast_barpos[i] for i in range(len(contrast_barpos))}
                
                center_barpos_dict = {**center_barpos_dict,**temp_dict}
                barpos = [p+keys_barpos[ki] for p in contrast_barpos]
                
                ax.bar(barpos,
                       height=side_data['hit_rate'],
                       yerr=side_data['confs'],
                       width=bar_width,
                       color = self.color.stim_keys[k]['color'],
                       error_kw = {'elinewidth':2},
                       label = f"{key_pairs[k]}{self._dict2label(counts4labels)}" if j==len(sides)-1 else '_')

        has_opto = [k for k in self.hit_rate_dict.keys() if 'opto' in k]
        if len(has_opto):
            non_opto_keys = [k for k in self.data.keys() if 'opto' not in k]
            opto_keys = [k for k in self.data.keys() if 'opto' in k]
            
            opto_nonopto_pairs = {non_opto_keys[i]:opto_keys[i] for i in range(len(non_opto_keys)) if non_opto_keys[i] in opto_keys[i]}
            
            self.p_values = {}
            for non_k,opto_k in opto_nonopto_pairs.items():
                p_values_contra = self.stat_analysis.get_hitrate_pvalues_exact(stim_data_keys=[non_k,opto_k])
                p_values_catch = self.stat_analysis.get_hitrate_pvalues_exact(stim_data_keys=[non_k,opto_k],stim_side='catch')
                self.p_values[opto_k] = {**p_values_contra,**p_values_catch}
       
            # put the significance starts
            for i,k in enumerate(self.p_values.keys()):
                for c,p in self.p_values[k].items():
                    stars = ''
                    shift = 0
                    if p < 0.001:
                        stars = '***'
                        shift = 0.2
                    elif 0.001 < p < 0.01:
                        stars = '**'
                        shift = 0.1
                    elif 0.01 < p < 0.05:
                        stars = '*'
                        shift = 0.08
                        
                    ax.text(center_barpos_dict[c]-shift, 1.04+0.02*i, stars,color=self.color.stim_keys[k]['color'], fontsize=30)
        fontsize = 25
        ax.set_ylim([0,1.05]) 
        x_labels = 100 * np.array(list(center_barpos_dict.keys()))
        x_labels = [f'{str(np.abs(c))}' for c in x_labels]
        ax.set_xticks(list(center_barpos_dict.values()),labels=x_labels)
        ax.tick_params(axis='both', labelsize=fontsize,length=10, width=3, which='major',color='k')
        ax.tick_params(axis='both', labelsize=fontsize,length=8, width=2, which='minor',color='k')
        
        ax.spines['left'].set_bounds(0, 1) 
        # ax.spines['bottom'].set_bounds(0, 1)
        ax.spines['bottom'].set_position(('outward', 10))
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        ax.grid(alpha=0.4)
        ax.legend(loc='center left',bbox_to_anchor=(1,0.5),fontsize=fontsize-5,frameon=False)
        
        ax.set_xlabel('Stim Contrast',fontsize=fontsize)
        ax.set_ylabel('Hit Rate (%)',fontsize=fontsize)
        return ax,center_barpos_dict


class DetectionPerformancePlotter(PerformancePlotter):
    __slots__ = []
    def __init__(self,data,stimkey:str=None,**kwargs):
        super().__init__(data, stimkey, **kwargs)
        self.modify_data()

    def modify_data(self,*args,**kwargs):
        # change the self.plot_data here with methods if need to plot something else
        pass

    # def plot(self, ax:plt.axes=None,*args,**kwargs):
    # override the plot function calling __plot__ 
    # <your code here>
    # self.__plot__(x,y,ax)


class DetectionResponseTimeScatterCloudPlotter(ResponseTimeScatterCloudPlotter):
    __slots__ = []
    def __init__(self, data, stimkey:str=None, **kwargs):
        super().__init__(data, stimkey, **kwargs)
        self.modify_data()
        
    def modify_data(self,*args,**kwargs):
        pass
    
    def plot(self,ax:plt.Axes=None,cloud_width=0.33,plot_misses:bool=False,**kwargs):
        d = copy.deepcopy(self.plot_data)
        for k,v in d.items():
            v = v[v['answer']==1]
        self.plot_data = d
        ax = super().plot(ax,cloud_width,plot_misses,**kwargs)
        return ax
 
class DetectionResponseHistogramPlotter(ResponseTimeHistogramPlotter):
    """ Plots an histogram of response times, showing earlies and hits"""
    __slots__ = []
    def __init__(self, data, stimkey: str=None, **kwargs):
        super().__init__(data, stimkey, **kwargs)
        self.modify_data()
        
    def modify_data(self,*args,**kwargs):
        if isinstance(self.plot_data,dict):
            for k,v in self.plot_data.items():
                v['blanked_response_latency'] = v[['answer','response_latency','blank_time']].apply(lambda x: x['response_latency']+x['blank_time'] if x['answer']!=-1 else x['response_latency'],axis=1)
        else:
            self.plot_data['blanked_response_latency'] = self.plot_data[['answer','response_latency','blank_time']].apply(lambda x: x['response_latency']+x['blank_time'] if x['answer']!=-1 else x['response_latency'],axis=1)
        
    
    @staticmethod
    def shuffle_times(x_in,n_shuffle:int=1000) -> np.ndarray:
        """ Shuffles x_in n_shuffle times """
        gen = np.random.default_rng()
        x_in = x_in.reshape(-1,1)
        x_temp = x_in.copy()
        shuffled_matrix = np.zeros((n_shuffle,x_in.shape[0]))
        
        for i in range(n_shuffle):
            gen.shuffle(x_temp)
            shuffled_matrix[i,:] = x_temp.reshape(1,-1) 
            
        return shuffled_matrix
    
    def plot(self,bin_width=50,ax:plt.Axes=None,**kwargs):
        n_shuffle = kwargs.get('n_shuffle',1000)
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(15,10)))
            ax = self.fig.add_subplot(1,1,1)
        
        answered_data = self.plot_data[self.stimkey][self.plot_data[self.stimkey]['answer']!=0]
        resp_times_blanked = answered_data['blanked_response_latency'].to_numpy()[:]
        blank_times = answered_data['blank_time'].to_numpy()[:]
        resp_times = resp_times_blanked - blank_times
        
        counts,bins = self.bin_times(resp_times,bin_width)
        ax = self.__plot__(ax,counts,bins)
        #plotting the median
        # ax.axvline(np.median(resp_times),color='b',linewidth=3)
        # plotting the shuffled histograms
        shuffled = self.shuffle_times(resp_times_blanked)
        shuffled_hists = np.zeros((n_shuffle,len(counts)))

        for i,row in enumerate(shuffled):

            row -= blank_times
            counts,_ = self.bin_times(row,bin_width,bins=bins)
            shuffled_hists[i,:] = counts.reshape(1,-1)
        
        #mean & std
        shuf_mean = np.mean(shuffled_hists,axis=0)
        shuf_std = np.std(shuffled_hists,axis=0)
        
        ax.fill_between(bins[1:],shuf_mean-shuf_std,shuf_mean+shuf_std,color='dimgrey',alpha=0.4,zorder=2)
        ax.plot(bins[1:],shuf_mean,color='dimgrey',alpha=0.6,linewidth=2,zorder=3)
            
        fontsize = kwargs.get('fontsize',14)
        ax.set_xlabel('Time from Stimulus onset (ms)', fontsize=fontsize)
        ax.set_ylabel('Counts', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.tick_params(axis='x')
        ax.grid(alpha=0.5,axis='y')
        
        return ax
        
    
class DetectionResponseTypeBarPlotter(ResponseTypeBarPlotter):
    __slots__ = []
    def __init__(self, data, stimkey: str, **kwargs):
        super().__init__(data, stimkey, **kwargs)
        self.modify_data()
        
    def modify_data(self,*args,**kwargs):
        # change the self.plot_data here if need to plot something else
        pass
    
    def plot(self,ax:plt.Axes=None,padding=0.8,**kwargs):
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(15,10)))
            ax = self.fig.add_subplot(1,1,1)
        
        
        for answer in [0,1]:
            colors = ['#630726','#32a852','#333333']
            answer_data = self.plot_data[self.stimkey][self.plot_data[self.stimkey]['answer']==answer]
            counts = [len(answer_data[answer_data['stim_side']<0]),
                      len(answer_data[answer_data['stim_side']>0]),
                      len(answer_data[answer_data['stim_side']==0])]
            
            if counts[-1] == 0:
                counts = counts[:-1]
                colors = colors[:-1]
            
            locs = self.position_bars(answer,len(counts),0.25,padding=padding)
            
            ax = self.__plot__(ax,locs,counts,width=0.25,
                                 color=colors,
                                 linewidth=2,
                                 edgecolor='k',
                                 **kwargs)
        
        # early answers alone
        ax = self.__plot__(ax,[-1],[len(self.plot_data[self.stimkey][self.plot_data[self.stimkey]['answer']==-1])],
                           width=0.5,
                           color='orangered',
                           linewidth=2,
                           edgecolor='k',
                           **kwargs)
                
        fontsize = kwargs.get('fontsize',14)
        ax.set_ylabel('Counts', fontsize=fontsize)
        ax.set_xticks([-1,0,1])
        ax.set_xticklabels(['Early','Missed','Correct'])
        ax.tick_params(labelsize=fontsize)
        ax.grid(alpha=0.5,axis='y')
        
        return ax
    
    
class DetectionResponseScatterPlotter(BasePlotter):
    __slots__ = ['stimkey','plot_data']
    def __init__(self, data: dict, stimkey:str=None, **kwargs):
        super().__init__(data, **kwargs)
        self.plot_data, self.stimkey = self.select_stim_data(self.data,stimkey)

    @staticmethod
    def __plot_scatter__(ax,t,times_arr,**kwargs):
        """ Plots the trial no and response times[blank time,answer time] by putting them on a scatter line """
        c = ['k'] 
        s = [10]
        if len(times_arr)==1:
            c = ['gainsboro']
        else:  
            if times_arr[0] < times_arr[1]:
                c.append('forestgreen') # correct
                
            else:
                c.append('orangered')
            s.append(20)
        t_arr = [t] * len(times_arr)
        ax.scatter(times_arr,t_arr,s=kwargs.get('s',s),c=c,alpha=0.7)
        return ax
    
    @staticmethod
    def __plot_density__(ax,x_bins,y_dens,**kwargs): 
        ax.plot(x_bins[1:],y_dens,c='dimgrey',alpha=0.8,linewidth=3,**kwargs) #right edges
        return ax
    
    def set_wrt_response_plot_data(self,wrt='sorted') -> np.ndarray:
        """ sets the plot data wrt to given argument and excludes nogo trials"""
        d = self.plot_data[self.stimkey]
        if wrt=='sorted':
            # add blank_time to correct answers 
            d['wrt_response_latency'] = d[['answer','blank_time','response_latency']].apply(lambda x: x['response_latency']+x['blank_time'] if x['answer']==1 else x['response_latency'],axis=1)
            
        elif wrt=='onset':
            d['wrt_response_latency'] = d[['answer','blank_time','response_latency']].apply(lambda x: x['response_latency']-x['blank_time'],axis=1)
        else:
            raise ValueError(f'{wrt} is not a valid wrt value for response times')
        self.plot_data[self.stimkey] = d[d['answer']!=0]
            
       
    def plot(self,ax:plt.Axes=None,bin_width:int=20,blanks:str='sorted',plt_range:list=None,**kwargs):
        if plt_range is None:
            # plt_range = [-100,self.plot_data['response_latency'].max()]
            plt_range = [-100,4900]
        
        bins = int((plt_range[1]-plt_range[0]) / bin_width)
        
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(8,8)))
            ax = self.fig.add_subplot(1,1,1)
            
        self.set_wrt_response_plot_data(wrt=blanks)
        times = self.plot_data[self.stimkey]['wrt_response_latency'].to_numpy()
        times_arr = []
        if blanks == 'sorted':
            sorted_data = self.plot_data[self.stimkey].sort_values('blank_time',ascending=False)
            for i,row in enumerate(sorted_data.itertuples()):
                times_arr = [row.blank_time, row.wrt_response_latency]
                ax = self.__plot_scatter__(ax,i,times_arr,**kwargs)
            x_label = 'Response Time (ms)'
        elif blanks == 'onset':
            for i,t in enumerate(times):
                times_arr = [0,t]
                ax = self.__plot_scatter__(ax,i,times_arr,**kwargs)
            x_label = 'Time from Stim Onset (ms)'
                
        ax_density = ax.inset_axes([0,0,1,0.1],frameon=False,sharex=ax)  
        
        hist,bins = np.histogram(times,bins=bins,range=plt_range)
        density = (hist / len(times)) / bin_width
        ax_density = self.__plot_density__(ax_density,bins,density,**kwargs) 
        
        fontsize = kwargs.get('fontsize',14)
        
        ax.set_ylim([-30,None])
        ax.set_yticks([i for i in range(len(self.plot_data[self.stimkey])) if i>=0 and i%50==0])
        ax.set_xlabel(x_label, fontsize=fontsize)
        ax.set_ylabel('Trial No.', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.grid(alpha=0.3,axis='x')
        
        ax.set_xlim(plt_range)
        ax_density.grid(alpha=0.3,axis='x')
        ax_density.tick_params(labelsize=fontsize)
        ax_density.set_yticks([])
        ax_density.set_yticklabels([])    
                
        return ax
        
        
class DetectionLickScatterPlotter(LickScatterPlotter):
    __slots__ = []
    def __init__(self, data, stimkey: str=None, **kwargs):
        super().__init__(data, stimkey, **kwargs)
        self.modify_data()
        
    def modify_data(self,*args,**kwargs):
        # change the self.plot_data here if need to plot something else
        if isinstance(self.plot_data,dict):
            for k,v in self.plot_data.items():
                v['blanked_response_latency'] = v[['answer','response_latency','blank_time']].apply(lambda x: x['response_latency']+x['blank_time'] if x['answer']!=-1 else x['response_latency'],axis=1)
                v['response_latency_absolute'] = v['response_latency'] + v['openstart_absolute'] 
        else:
            self.plot_data['blanked_response_latency'] = self.plot_data[['answer','response_latency','blank_time']].apply(lambda x: x['response_latency']+x['blank_time'] if x['answer']!=-1 else x['response_latency'],axis=1)
            self.plot_data['response_latency_absolute'] = self.plot_data['response_latency'] + self.plot_data['openstart_absolute']
    
    def plot(self,ax:plt.Axes=None,bin_width:int=20,wrt:str='reward',plt_range:list=None,**kwargs):
        if plt_range is None:
            plt_range = [-1000,1000]
            
        bins = int((plt_range[1]-plt_range[0]) / bin_width)
        
        if ax is None:
            self.fig = plt.figure(figsize = kwargs.get('figsize',(8,8)))
            ax = self.fig.add_subplot(1,1,1)
            
        for row in self.plot_data[self.stimkey].itertuples():
            if len(row.reward):
                if len(row.lick):
                    if wrt == 'reward':
                        wrt_time = row.reward[0]
                        response_time = row.response_latency_absolute - row.reward[0]
                        x_label = 'Time from Reward (ms)'
                        wrt_color = 'r'
                        ax.scatter(response_time,row.trial_no,c='k',marker='|',s=20,zorder=2)
                    elif wrt == 'response':
                        wrt_time = row.response_latency_absolute
                        x_label = 'Time from Response (ms)'
                        wrt_color = 'k'
                        reward = row.reward[0] - row.response_latency_absolute
                        ax.scatter(reward,row.trial_no,c='r',marker='|',s=20,zorder=2)
                    
                    licks = row.lick[:,0] - wrt_time
                    ax = self.__plot_scatter__(ax,row.trial_no,licks,**kwargs)   

        ax.axvline(0,c=wrt_color,linewidth=2,zorder=1)
    
        ax_density = ax.inset_axes([0,0,1,0.1],frameon=False,sharex=ax)
        
        pooled_licks = self.pool_licks(wrt)
        
        hist,bins = np.histogram(pooled_licks,bins=bins,range=plt_range)
        density = (hist / len(pooled_licks)) / bin_width
        ax_density = self.__plot_density__(ax_density,bins,density,zorder=2,**kwargs) 
        
        ax_density.axvline(0,c=wrt_color,linewidth=2,zorder=1)
        
        fontsize = kwargs.get('fontsize',14)
        ax.set_xlim(plt_range)
        ax.set_ylim([-30,None])
        ax.set_yticks([i for i in range(len(self.plot_data)) if i>=0 and i%50==0])
        ax.set_xlabel(x_label, fontsize=fontsize)
        ax.set_ylabel('Trial No.', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.grid(alpha=0.3,axis='x')
        
        ax.set_xlim(plt_range)
        ax_density.grid(alpha=0.3,axis='x')
        ax_density.tick_params(labelsize=fontsize)
        ax_density.set_yticks([])
        ax_density.set_yticklabels([])

        return ax
    
    
class DetectionWheelTrajectoryPlotter(WheelTrajectoryPlotter):
    __slots__ = []
    def __init__(self, data: dict, stimkey: str = None, seperate_by:str='contrast',**kwargs):
        super().__init__(data, stimkey, **kwargs)
        
        
        self.side_sep_dict = self.seperate_wheel_data(seperate_by)
        
    def seperate_wheel_data(self,seperate_by):
        """ Seperates the wheel data depending on the seperate_by argument 
        Returns a dict with sides as keys, that has dictionaries with seperator values as keys"""
        
        side_sep_dict = {}
        if seperate_by not in self.plot_data[self.stimkey].columns:
            raise ValueError(f'{seperate_by} is not a valid field for this data. try: {self.plot_data.columns}')
        
        seperator_list = nonan_unique(self.plot_data[self.stimkey][seperate_by],sort=True)
        
        sides = nonan_unique(self.plot_data[self.stimkey]['stim_side'])
        
        for i,side in enumerate(sides,start=1):
            side_slice = self.plot_data[self.stimkey][self.plot_data[self.stimkey]['stim_side'] == side]
            
            side_sep_dict[side] = {}
            for sep in seperator_list:
                seperator_slice = side_slice[side_slice[seperate_by] == sep]
            
                if not seperator_slice.empty:
                    # shift wheel according to side
                    # wheel_arr = seperator_slice['wheel'].apply(lambda x: x+side)
                    # seperator_slice.loc[:,'wheel'] = seperator_slice.loc[:,'wheel'] + s

                    wheel_stats = get_trajectory_avg(seperator_slice['wheel'].to_numpy())
                    side_sep_dict[side][sep] = {'avg': wheel_stats['avg'],
                                                'sem':wheel_stats['sem']}
                else:
                    print(f'NO data in {side} and {sep}')
                                
        return side_sep_dict
    
    def plot(self,ax:plt.Axes=None,plot_range_time:list=None,plot_range_trj:list=None,orientation:str='vertical',bin_width:int=None,**kwargs):
        ax = super().plot(ax,plot_range_time,plot_range_trj,orientation,**kwargs)
        
        if bin_width is not None:
            bins = np.arange(0,plot_range_time[-1],bin_width,dtype='int')
            
            ax_density = ax.inset_axes([0,0,1,0.1],frameon=False,sharex=ax)
            
            pooled_licks = self.pool_trial_ends()
            
            hist,bins = np.histogram(pooled_licks,bins=bins,range=plot_range_time)
            ax_density = self.__plot_density__(ax_density,bins,hist,zorder=2,**kwargs)
            ax_density.set_yticks([])
            ax_density.set_yticklabels([])
        
        return ax

   
class DetectionSummaryPlotter:
    __slots__ = ['data','fig','plotters','stimkey']
    def __init__(self, data, stimkey:str=None,**kwargs):
        self.data = data # gets the stim data dict
        self.stimkey = stimkey
        self.fig = None
        self.init_plotters()
        
    def init_plotters(self):
        # TODO: Make this changable
        self.plotters = {'performance':DetectionPerformancePlotter(self.data, stimkey='all'),
                         'responsepertype':DetectionResponseTimeScatterCloudPlotter(self.data,self.stimkey),
                         'resptype':DetectionResponseTypeBarPlotter(self.data,stimkey='all'),
                         'licktotal':LickPlotter(self.data, stimkey='all'),
                         'resphist':DetectionResponseHistogramPlotter(self.data,stimkey='all'),
                         'respscatter':DetectionResponseScatterPlotter(self.data,stimkey='all'),
                         'lickdist':DetectionLickScatterPlotter(self.data,self.stimkey)}
    
    def plot(self,**kwargs):
        self.fig = plt.figure(figsize = kwargs.get('figsize',(20,10)))
        widths = [2,2,1]
        heights = [1,1]
        gs = self.fig.add_gridspec(ncols=3, nrows=2, 
                                   width_ratios=widths,height_ratios=heights,
                                   left=kwargs.get('left',0),right=kwargs.get('right',1),
                                   wspace=kwargs.get('wspace',0.3),hspace=kwargs.get('hspace',0.4))

        gs_in1 = gs[:,0].subgridspec(nrows=2,ncols=1,hspace=0.3)

        ax_perf = self.fig.add_subplot(gs_in1[1,0])
        ax_perf = self.plotters['performance'].plot(ax=ax_perf,seperate_by='contrast')
        
        ax_lick = ax_perf.twinx()
        ax_lick = self.plotters['licktotal'].plot(ax=ax_lick)
        ax_lick.grid(False)
        
        ax_resp = self.fig.add_subplot(gs_in1[0,0])
        ax_resp = self.plotters['resphist'].plot(ax=ax_resp)
        
        ax_resp2 = self.fig.add_subplot(gs[0,1])
        ax_resp2 = self.plotters['responsepertype'].plot(ax=ax_resp2)
        
        ax_type = self.fig.add_subplot(gs[0,2])
        ax_type = self.plotters['resptype'].plot(ax=ax_type)
        
        ax_scatter = self.fig.add_subplot(gs[1,1])
        ax_scatter = self.plotters['respscatter'].plot(ax=ax_scatter)
        
        ax_ldist = self.fig.add_subplot(gs[1,2])
        ax_ldist = self.plotters['lickdist'].plot(ax=ax_ldist)
        
        self.fig.tight_layout()
    
    
    def save(self,saveloc,date,animalid):
        if self.fig is not None:
            saveloc = pjoin(saveloc,'figures')
            if not os.path.exists(saveloc):
                os.mkdir(saveloc)
            savename = f'{date}_sessionSummary_{animalid}.pdf'
            saveloc = pjoin(saveloc,savename)
            self.fig.savefig(saveloc,bbox_inches='tight')
            display(f'Saved {savename} plot')