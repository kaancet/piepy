from typing import Any
from ..basePlotters import *
from ...detection.wheelDetectionSilencing import *
from ...detection.wheelDetectionAnalysis import *


# constant animal colors
ANIMAL_COLORS = {'KC139' : '#332288',
                 'KC141' : '#117733',
                 'KC142' : '#DDCC77',
                 'KC143' : '#AA4499',
                 'KC144' : '#882255',
                 'KC145' : '#88CCEE',
                 'KC146' : '#275D6D',
                 'KC147' : '#F57A6C',
                 'KC148' : '#ADFA9A',
                 'KC149' : '#A45414'}


class ComparisonLinePlotter:
    """This plotter expects data of a single experiment type:
        multiple animals, same area, same stim count(type)"""
    def __init__(self,data) -> None:
        self.data = data # get area filtered data
        
        self.plot_data = self.make_plot_data()
        
    def make_plot_data(self) -> pl.DataFrame:
        """ Returns a dataframe to later loop while plotting
        This automatically groups together two of the same experiment sessions"""
        q = (
            self.data.lazy()
            .groupby(["animalid","stim_type","stim_side","contrast","opto"])
            .agg(
                [
                    pl.count().alias("trial_count"),
                    (pl.col("outcome")==1).sum().alias("correct_count"),
                    (pl.col("outcome")==0).sum().alias("miss_count"),
                    (pl.col("response_latency").alias("response_times")),
                    (pl.col("response_latency").median().alias("median_response_time")),
                    (pl.col("opto_pattern").first()),
                    (pl.col("stimkey").first()),
                    (pl.col("stim_label").first()),
                ]
            ).drop_nulls()
            .sort(["animalid","stim_type","stim_side","contrast","opto"])
        )

        q = q.with_columns((pl.col("correct_count") / pl.col("trial_count")).alias("hit_rate"))
        q = q.with_columns((1.96 * np.sqrt((pl.col("hit_rate")*(100.0 - pl.col("hit_rate"))) / pl.col("trial_count"))).alias("confs"))
        q = q.with_columns((100*pl.col("hit_rate")).alias("hit_rate"))
        q = q.with_columns(pl.when(pl.col("stim_side")=="ipsi").then((pl.col("contrast")*-1)).otherwise(pl.col("contrast")).alias("signed_contrast"))
        q = q.with_columns(pl.when((pl.col("contrast")>0) & (pl.col("contrast")<25)).then(pl.lit("hard")).when(pl.col("contrast")>25).then(pl.lit("easy")).otherwise(pl.lit("catch")).alias("contrast_difficulty"))

        # reorder stim_label to last column
        cols = q.columns
        del cols[-6]
        del cols[-5]
        cols.extend(['stimkey','stim_label'])
        q = q.select(cols)
        df = q.collect()
        return df
    
    def plot_hit_rates(self,**kwargs) -> plt.figure:
        
        fontsize = kwargs.pop('fontsize',30)
        linewidth = kwargs.pop('linewidth',3)
    
        uniq_contrast = self.plot_data['contrast_difficulty'].unique().sort(descending=True).to_numpy()
        uniq_contrast = np.roll(uniq_contrast,1) # roll to have the order [catch, hard, easy]
        n_contrast = len(uniq_contrast) - 1 # remove 0
        uniq_stims = self.plot_data['stim_type'].unique().sort().to_numpy()
        n_stim = len(uniq_stims)
        
        self.fig, axes = plt.subplots(ncols=n_stim*n_contrast,
                                      nrows=1,
                                      constrained_layout=True,
                                      figsize=kwargs.pop('figsize',(12,10)))

        self.p_values_hit_rate = {}
        
        j=-1
        for c in uniq_contrast:
            
            contrast_df = self.plot_data.filter((pl.col("contrast_difficulty")==c)) 

            self.p_values_hit_rate[c] = {}
            
            for i,k in enumerate(uniq_stims):
                stim_df = contrast_df.filter(pl.col("stim_type")==k)
                
                if c=='catch':
                    base_df = stim_df.filter(pl.col('opto')==0)
                    baseline_avg = np.mean(base_df['hit_rate'].to_numpy())
                    baseline_sem = stats.sem(base_df['hit_rate'].to_numpy())
                    continue 
                elif c!='catch' and i==0:
                    j+=1
                
                ax = axes[n_stim*j+i]
                uniq_animals = stim_df['animalid'].unique().to_numpy()
                
                for a_id in uniq_animals:
                    animal_df = stim_df.filter((pl.col("animalid")==a_id) &
                                               (pl.col("stim_side")=="contra"))
                    
                    ax.plot(animal_df['opto'].to_list(),
                            animal_df['hit_rate'].to_list(),
                            marker='o',
                            markersize=20,
                            markeredgewidth=0,
                            linewidth=linewidth,
                            c=ANIMAL_COLORS[a_id],
                            alpha=0.5,
                            label=a_id,
                            zorder=2)
                    
                avg_df = (
                            stim_df.filter(pl.col("stim_side")=="contra")
                            .groupby(["opto"])
                            .agg(
                                [
                                    pl.count().alias("animal_count"),
                                    pl.col('hit_rate').mean().alias('avg_hitrate'),
                                    pl.col('hit_rate')
                                ]
                            ).sort(["opto"])
                        )
                avg_df = avg_df.with_columns(pl.col('hit_rate').apply(lambda x: stats.sem(x)).alias("animal_confs"))
                
                ax.errorbar(avg_df['opto'].to_list(),
                            avg_df['avg_hitrate'].to_list(),
                            avg_df['animal_confs'].to_list(),
                            marker='o',
                            markersize=20,
                            linewidth=linewidth*2,
                            c='k',
                            zorder=2)
                
                ax.axhline(y=baseline_avg,linestyle=':',c='k',alpha=0.4,zorder=1)
                ax.axhspan(baseline_avg+baseline_sem,
                           baseline_avg-baseline_sem,
                           color='gray',alpha=0.05,linewidth=0,zorder=1)
                
                
                # do p-values with mann-whitney-u
                non_opto = stim_df.filter((pl.col('opto')==0)&(pl.col('stim_side')=="contra"))['hit_rate'].to_numpy()
                opto = stim_df.filter((pl.col('opto')==1)&(pl.col('stim_side')=="contra"))['hit_rate'].to_numpy()
                # _,p = mannwhitneyu(non_opto,opto)
                _,p = mannwhitneyu(non_opto,opto)
                self.p_values_hit_rate[c][k] = p
                
                stars = ''
                if p < 0.001:
                    stars = '***'
                elif 0.001 < p < 0.01:
                    stars = '**'
                elif 0.01 < p < 0.05:
                    stars = '*'
                ax.text(0.4, 101, stars,color='k', fontsize=30)
                
                ax.set_ylim([0,105])
                ax.set_yticks([0,25,50,75,100])
                ax.set_xticks([0,1])
                ax.set_xticklabels(['Non\nOpto','Opto'])
                ax.set_title(k,fontsize=fontsize-5)
                
                ax.set_xlabel(f'c={c}',fontsize=fontsize,labelpad=10)
                
                ax.tick_params(axis='x', labelsize=fontsize-5,length=10, width=linewidth, which='major',color='k')
                ax.grid(True,axis='y',alpha=0.4)
                if n_stim*j+i==0:
                    ax.set_ylabel('Hit Rate',fontsize=fontsize)
                    ax.tick_params(axis='y', labelsize=fontsize,length=10, width=linewidth, which='major',color='k')
                else:
                    ax.tick_params(axis='y', labelsize=0,length=0, width=0, which='major',color='k')
                
                ax.spines['bottom'].set_position(('outward', 20))
                ax.spines['bottom'].set_linewidth(2)
                ax.spines['left'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                
        ax.legend(loc='center left',bbox_to_anchor=(1,0.5),fontsize=fontsize,frameon=False)
        return ax
    
    def plot_response(self,remove_miss:bool=True,**kwargs) -> plt.figure:
        
        fontsize = kwargs.pop('fontsize',30)
        linewidth = kwargs.pop('linewidth',3)
    
        uniq_contrast = self.plot_data['contrast_difficulty'].unique().sort(descending=True).to_numpy()
        uniq_contrast = np.roll(uniq_contrast,1) # roll to have the order [catch, hard, easy]
        n_contrast = len(uniq_contrast) - 1 # remove 0
        uniq_stims = self.plot_data['stim_type'].unique().sort().to_numpy()
        n_stim = len(uniq_stims)
        
        # main axes for stim keys
        # self.fig, axes = plt.subplots(ncols=n_stim*n_contrast,
        #                               nrows=1,
        #                               constrained_layout=True,
        #                               figsize=(2*(n_stim*n_contrast)+1,(n_stim*n_contrast)+2))
        self.fig, axes = plt.subplots(ncols=n_stim*n_contrast,
                                      nrows=1,
                                      constrained_layout=True,
                                      figsize=kwargs.pop('figsize',(20,10)))

        self.p_values_resp = {}
        
        j=-1
        for c in uniq_contrast:
            
            contrast_df = self.plot_data.filter((pl.col("contrast_difficulty")==c)) 

            self.p_values_resp[c] = {}
            
            for i,k in enumerate(uniq_stims):
                stim_df = contrast_df.filter(pl.col("stim_type")==k)
                
                if c=='catch':
                    base_df = stim_df.filter(pl.col('opto')==0)
                    baseline_avg = np.mean(base_df['hit_rate'].to_numpy())
                    baseline_sem = stats.sem(base_df['hit_rate'].to_numpy())
                    continue
                elif c!='catch' and i==0:
                    j+=1
                
                ax = axes[n_stim*j+i]
                uniq_animals = stim_df['animalid'].unique().to_numpy()
                
                mean_o_medians = []
                for a_id in uniq_animals:
                    animal_df = stim_df.filter((pl.col("animalid")==a_id) &
                                                (pl.col("stim_side")=="contra"))
                    
                    if remove_miss:
                        tmp = animal_df['response_times'].to_numpy()
                        median = [np.median(i[i<1000]) for i in tmp]
                        mean_o_medians.append(median)
                        ax.plot(animal_df['opto'].to_list(),
                                median,
                                marker='o',
                                markersize=20,
                                markeredgewidth=0,
                                linewidth=linewidth,
                                c=ANIMAL_COLORS[a_id],
                                label=a_id,
                                alpha=0.5,
                                zorder=2)
                    else:
                        ax.plot(animal_df['opto'].to_list(),
                                animal_df['median_response_time'].to_list(),
                                marker='o',
                                linewidth=linewidth,
                                c=ANIMAL_COLORS[a_id],
                                label=a_id,
                                alpha=0.5,
                                zorder=2)
                    
                avg_df = (
                            stim_df.filter(pl.col("stim_side")=="contra")
                            .groupby(["opto"])
                            .agg(
                                [
                                    pl.count().alias("animal_count"),
                                    pl.col('median_response_time').mean().alias('avg_resp_time'),
                                    pl.col('median_response_time')
                                ]
                            ).sort(["opto"])
                        )
                avg_df = avg_df.with_columns(pl.col('median_response_time').apply(lambda x: stats.sem(x)).alias("animal_confs"))
                
                if remove_miss:
                    mean_o_medians = np.array(mean_o_medians)
                    mean = np.mean(mean_o_medians,axis=0)
                    conf = stats.sem(mean_o_medians,axis=0)
                    
                    ax.errorbar(avg_df['opto'].to_list(),
                                mean,
                                conf,
                                marker='o',
                                linewidth=linewidth*2,
                                c='k',
                                zorder=2)
                    
                else:
                    ax.errorbar(avg_df['opto'].to_list(),
                                avg_df['avg_resp_time'].to_list(),
                                avg_df['animal_confs'].to_list(),
                                marker='o',
                                markersize=20,
                                linewidth=linewidth*2,
                                c='k',
                                zorder=2)
                
                # ax.axhline(y=baseline_avg,linestyle=':',c='k',alpha=0.4,zorder=1)
                # ax.axhspan(baseline_avg+baseline_sem,
                #            baseline_avg-baseline_sem,
                #            color='gray',alpha=0.05,linewidth=0,zorder=1)
                
                
                # do p-values with mann-whitney-u
                non_opto = stim_df.filter((pl.col('opto')==0)&(pl.col('stim_side')=="contra"))['median_response_time'].to_numpy()
                opto = stim_df.filter((pl.col('opto')==1)&(pl.col('stim_side')=="contra"))['median_response_time'].to_numpy()
                _,p = mannwhitneyu(non_opto,opto)
                self.p_values_resp[c][k] = p
                
                stars = ''
                if p < 0.001:
                    stars = '***'
                elif 0.001 < p < 0.01:
                    stars = '**'
                elif 0.01 < p < 0.05:
                    stars = '*'
                ax.text(0.4, 1100, stars,color='k', fontsize=30)
                
                ax.set_yscale('symlog')
                minor_locs = [200,400,600,800,2000,4000,6000,8000]
                
                ax.set_ylim([190,1300])
                ax.set_xticks([0,1])
                ax.set_xticklabels(['Non-Opto','Opto'])
                ax.set_title(k,fontsize=fontsize-5)
                
                ax.set_xlabel(f'c={c}',fontsize=fontsize,labelpad=10)
                
                ax.tick_params(axis='x', labelsize=fontsize-5,length=10, width=linewidth, which='major',color='k')
                
                ax.grid(True,axis='y',alpha=0.4,which='minor')
                if n_stim*j+i==0:
                    ax.set_ylabel('Response Times (ms)',fontsize=fontsize)
                    ax.yaxis.set_minor_locator(plt.FixedLocator(minor_locs))
                    ax.yaxis.set_minor_formatter(plt.FormatStrFormatter('%d'))
                    ax.yaxis.set_major_locator(ticker.FixedLocator([10,100,1000,10000]))
                    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
                    ax.tick_params(axis='y', labelsize=fontsize,length=10, width=linewidth, which='major',color='k')
                    ax.tick_params(axis='y', labelsize=fontsize-3,length=10, width=linewidth, which='minor',color='k')
                else:
                    ax.yaxis.set_minor_locator(plt.FixedLocator(minor_locs))
                    ax.yaxis.set_minor_formatter(plt.FormatStrFormatter('%d'))
                    ax.tick_params(axis='y', labelsize=0,length=0, width=0, which='major',color='k')
                    ax.tick_params(axis='y', labelsize=0,length=0, width=0, which='minor',color='k')
                
                ax.spines['bottom'].set_position(('outward', 20))
                ax.spines['bottom'].set_linewidth(2)
                ax.spines['left'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
        
        ax.legend(loc='center left',bbox_to_anchor=(1,0.5),fontsize=fontsize,frameon=False)
        return ax
    

class AllAreasPlotter:
    def __init__(self,data) -> None:
        self.data = data
        
    def make_plot_data(self) -> pl.DataFrame:
        """ Makes a dataframe that has all animals and areas"""
        q = (
            self.data.lazy()
            .groupby(["stim_type","area","animalid","stim_side","contrast","opto"])
            .agg(
                [
                    pl.count().alias("trial_count"),
                    (pl.col("answer")==1).sum().alias("correct_count"),
                    (pl.col("answer")==0).sum().alias("miss_count"),
                    (pl.col("response_latency").alias("response_times")),
                    (pl.col("response_latency").median().alias("median_response_time")),
                    (pl.col("opto_pattern").first()),
                    (pl.col("stimkey").first()),
                    (pl.col("stim_label").first()),
                ]
            ).drop_nulls()
            .sort(["stim_type","area","animalid","stim_type","stim_side","contrast","opto"])
        )

        q = q.with_columns((pl.col("correct_count") / pl.col("trial_count")).alias("hit_rate"))
        q = q.with_columns((1.96 * np.sqrt((pl.col("hit_rate")*(100.0 - pl.col("hit_rate"))) / pl.col("trial_count"))).alias("confs"))
        q = q.with_columns((100*pl.col("hit_rate")).alias("hit_rate"))
        q = q.with_columns(pl.when(pl.col("stim_side")=="ipsi").then((pl.col("contrast")*-1)).otherwise(pl.col("contrast")).alias("signed_contrast"))

        # reorder stim_label to last column
        cols = q.columns
        del cols[-4]
        del cols [-4]
        cols.extend(['stimkey','stim_label'])
        q = q.select(cols)

        df = q.collect()

        return df
    

class ExperimentsSummaryPlotter:
    def __init__(self,data:pl.DataFrame) -> None:
        pass