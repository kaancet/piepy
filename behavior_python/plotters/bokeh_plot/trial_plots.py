from bokeh.layouts import column
from .bokeh_base import *
from ...wheelUtils import interpolate_position
from ...utils import find_nearest


class TrialGraph(Graph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cds_wheel = None
        self.cds_areas = None
        self.cds_lines = None
        self.cds_licks = None
        self.cds_interp = None
        self.cds_mov = None
        
    @staticmethod    
    def morph_data(data:pl.DataFrame) -> dict:
        """ Morphs the data to be column data source compatible and changes some column names """
        if data[0,'outcome'] == -1:
            time_anchor = 't_trialstart'
        else:
            time_anchor = 't_stimstart_rig'
        
        data = data.with_columns([
            (pl.col('t_trialstart') - pl.col(time_anchor)).alias('trial_start'),
            (pl.when(pl.col('t_stimstart_rig').is_not_null()).then(pl.col('t_stimstart_rig') - pl.col(time_anchor)).otherwise(pl.col('t_blank_dur'))).alias('stimstart_rig'),
            (pl.col('t_stimend_rig') - pl.col(time_anchor)).alias('stimend_rig')
            ])
        data = data.with_columns((pl.col("trial_start")+pl.col("t_quiescence_dur")).alias("quiescence_end"))
        
        return data.to_dict(as_series=False)
        
    def set_cds(self,data:pl.DataFrame) -> None:
        """ Creates a column data source(cds)"""
        self.dict_data = self.morph_data(data)
        
        temp_wheel = {'wheel_time' : self.dict_data['wheel_time'][0],
                      'wheel_pos' : self.dict_data['wheel_pos'][0]
                     }
        
        temp_areas = {'trial_start' : self.dict_data['trial_start'],
                      'quiescence_end' : self.dict_data['quiescence_end'],
                      'stimstart_rig' : self.dict_data['stimstart_rig'],
                      'stimend_rig' : self.dict_data['stimend_rig'],
                      'opto_start' : self.dict_data['stimstart_rig'] if self.dict_data['opto'] else [],
                      'opto_end' :self.dict_data['stimend_rig'] if self.dict_data['opto'] else [] }
        # temp_licks = {'lick_time':None}
        # temp_lines = {'response_latency' : None}
        
        # interpolate the wheels
        pos,t = interpolate_position(self.dict_data['wheel_time'][0],self.dict_data['wheel_pos'][0],freq=5)
        temp_interp = {'t':t,
                       'pos':pos}
        
        temp_mov = {'onsets' : self.dict_data['wheel_onsets'][0],
                    'offsets' : self.dict_data['wheel_offsets'][0]}
        
        if self.cds_areas is None:
            self.cds_wheel = ColumnDataSource(data=temp_wheel)
            self.cds_areas = ColumnDataSource(data=temp_areas)
            # self.cds_lines = ColumnDataSource(data=temp_lines)
            # self.cds_licks = ColumnDataSource(data=temp_licks)
            self.cds_interp = ColumnDataSource(data=temp_interp)
            self.cds_mov = ColumnDataSource(data=temp_mov)
        else:
            self.cds_wheel.data = temp_wheel
            self.cds_areas.data = temp_areas
            # self.cds_lines.data = temp_lines
            # self.cds_licks.data = temp_licks
            self.cds_interp.data = temp_interp
            self.cds_mov.data = temp_mov
    
    def plot(self,**kwargs):
        
        f_top = figure(title="", width=700,height=80,x_axis_location="above")
        f_top.toolbar.logo = None
        f_top.toolbar_location = None
        f_top.xgrid.grid_line_color = None
        f_top.ygrid.grid_line_color = None
        f_top.axis.visible = False
        
        f_top.vstrip(x0='opto_start',x1='opto_end',source=self.cds_areas,
                     color='#2b3cfc',fill_alpha=0.5,line_alpha=0)
        
        f = figure(title=None,width=700, height=400,x_range=f_top.x_range, y_range=f_top.y_range,
                   x_axis_label='Time(ms)', y_axis_label='Wheel Position')
    
        # quiescence
        f.vstrip(x0='trial_start',
                x1='quiescence_end',
                line_alpha=0,
                fill_alpha=0.3,
                fill_color='#d1d1d1',
                hatch_pattern='/',
                hatch_alpha=0.1,
                source=self.cds_areas,
                legend_label=f'Trial Prep. ({round(self.dict_data["t_quiescence_dur"][0],1)}ms)')
        
        #blank
        f.vstrip(x0='quiescence_end',
                x1='stimstart_rig',
                line_alpha=0,
                fill_alpha=0.3,
                fill_color='#c22121',
                hatch_pattern='x',
                hatch_alpha=0.1,
                source=self.cds_areas,
                legend_label=f'Wait for Stim ({round(self.dict_data["t_blank_dur"][0],1)}ms)')
        
        # response window
        f.vstrip(x0='stimstart_rig',
                x1='stimend_rig',
                line_alpha=0,
                fill_alpha=0.3,
                fill_color='#009912',
                hatch_pattern='x',
                hatch_alpha=0.1,
                source=self.cds_areas,
                legend_label='Response Window')
        
        # if trial_data[0,'opto_pattern']!=-1 and trial_data[0,'outcome']!=-1:
        #     pass
        #     # opto trial that wasn't early
        #     # f.block(x=,
        #     #         y=,
        #     #         )
        #     #     ax.barh(ax.get_ylim()[1],trial_data[0,'response_latency'],left=0,height=10,color='aqua')   
            
        # plot wheel traces        
        f.line('t', 'pos', color="#878787", source=self.cds_interp, line_width=3)
        
        f.circle('wheel_time', 'wheel_pos', source=self.cds_wheel, legend_label="Wheel Trace", color="#000000", size=5)
        
        # plot movements
        # for i,o in enumerate(self.dict_data['wheel_onsets']):
        #     on_idx, on_t = find_nearest(t,o)
        #     off_idx, off_t = find_nearest(t,offsets[i])
        #     f.triangle(on_t,pos[on_idx], color="#075900", size=5, line_alpha=0)
        #     f.triangle_dot(t[on_idx:off_idx],pos[on_idx:off_idx], color="#9c0902", size=5, line_alpha=0)
        #     f.line(t[on_idx:off_idx],pos[on_idx:off_idx],color="#1d02c9",line_width=3,line_dash='dotted')

        # # plot the reward
        # reward = trial_data[0,'reward']
        # if reward is not None:
        #     reward = reward[0] - trial_data[0,time_anchor]
        #     f.image_url(value("https://images.emojiterra.com/google/noto-emoji/unicode-15.1/color/1024px/1f4a7.png"), #droplet emoji
        #                 x=reward,y=0,w=10,h=10, w_units="screen", h_units="screen")
        
        # # plot the lick
        # lick_arr = trial_data['lick'].explode().to_numpy()
        # if len(lick_arr):
        #     lick_arr = [l - trial_data[0,time_anchor] for l in lick_arr]
        #     f.diamond(x=lick_arr,y=[0]*len(lick_arr),color="#1fd2ff",size=7)
        
        self.fig = column(f_top,f)
