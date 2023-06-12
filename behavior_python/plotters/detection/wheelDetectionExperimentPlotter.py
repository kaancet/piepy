from ..basePlotters import *
from ...detection.wheelDetectionSilencing import *
from ...detection.wheelDetectionAnalysis import *


# constant animal colors
ANIMAL_COLORS = {'KC139' : '#332288',
                 'KC141' : '#117733',
                 'KC142' : '#DDCC77',
                 'KC143' : '#AA4499',
                 'KC144' : '#882255',
                 'KC145' : '#88CCEE'}


class HitRateLinePlot:
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
                    (pl.col("answer")==1).sum().alias("correct_count"),
                    (pl.col("answer")==0).sum().alias("miss_count"),
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

        # reorder stim_label to last column
        cols = q.columns
        del cols[-4]
        del cols [-4]
        cols.extend(['stimkey','stim_label'])
        q = q.select(cols)

        df = q.collect()

        return df
    
    def plot_indiv(self,**kwargs) -> plt.figure:
        
        fontsize = kwargs.pop('fontsize',30)
        linewidth = kwargs.pop('linewidth',1.5)
    
        uniq_contrast = self.plot_data['contrast'].unique().to_numpy()
        n_contrast = len(uniq_contrast) - 1 # remove 0
        uniq_stims = self.plot_data['stim_type'].unique().to_numpy()
        n_stim = len(uniq_stims)
        
        # main axes for stim keys
        # self.fig, axes = plt.subplots(ncols=n_stim*n_contrast,
        #                               nrows=1,
        #                               constrained_layout=True,
        #                               figsize=(2*(n_stim*n_contrast)+1,(n_stim*n_contrast)+2))
        self.fig, axes = plt.subplots(ncols=n_stim*n_contrast,
                                      nrows=1,
                                      constrained_layout=True,
                                      figsize=kwargs.pop('figsize',(12,10)))

        for i,k in enumerate(uniq_stims):
            stim_df = self.plot_data.filter(pl.col("stim_type")==k)

            j=-1
            for c in uniq_contrast:
                
                contrast_df = stim_df.filter((pl.col("contrast")==c)) 
                
                if c==0:
                    base_df = contrast_df.filter(pl.col('opto')==0)
                    baseline_avg = np.mean(base_df['hit_rate'].to_numpy())
                    baseline_sem = stats.sem(base_df['hit_rate'].to_numpy())
                    continue
                else:
                    j+=1
                
                ax = axes[i+j+(i*1)]
                uniq_animals = contrast_df['animalid'].unique().to_numpy()
                
                for a_id in uniq_animals:
                    animal_df = contrast_df.filter((pl.col("animalid")==a_id) &
                                                   (pl.col("stim_side")=="contra"))
                    
                    ax.plot(animal_df['opto'].to_list(),
                            animal_df['hit_rate'].to_list(),
                            marker='o',
                            linewidth=linewidth,
                            c=ANIMAL_COLORS[a_id],
                            alpha=0.5,
                            zorder=2)
                    
                avg_df = (
                            contrast_df.groupby(["opto"])
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
                            linewidth=linewidth*2,
                            c='dimgray',
                            zorder=2)
                
                ax.axhline(y=baseline_avg,linestyle=':',c='k',alpha=0.4,zorder=1)
                ax.axhspan(baseline_avg+baseline_sem,
                           baseline_avg-baseline_sem,
                           color='gray',alpha=0.05,linewidth=0,zorder=1)
                
                
                # do p-values with mann-whitney-u
                non_opto = contrast_df.filter((pl.col('opto')==0)&(pl.col('stim_side')=="contra"))['hit_rate'].to_numpy()
                opto = contrast_df.filter((pl.col('opto')==1)&(pl.col('stim_side')=="contra"))['hit_rate'].to_numpy()
                _,p = mannwhitneyu(non_opto,opto)
                
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
                if i+j+(i*1)==0:
                    ax.set_ylabel('Hit Rate',fontsize=fontsize)
                    ax.tick_params(axis='y', labelsize=fontsize,length=10, width=linewidth, which='major',color='k')
                else:
                    ax.tick_params(axis='y', labelsize=0,length=0, width=0, which='major',color='k')
                
                ax.spines['bottom'].set_position(('outward', 20))
                ax.spines['bottom'].set_linewidth(2)
                ax.spines['left'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
            

    # def plot_pooled(self, ax:plt.Axes=None, **kwargs) -> plt.Axes:
        
    #     fontsize = kwargs.pop('fontsize',30)
        
    #     if ax is None:
    #         self.fig = plt.figure(figsize = kwargs.get('figsize',(15,10)))
    #         ax = self.fig.add_subplot(1,1,1)