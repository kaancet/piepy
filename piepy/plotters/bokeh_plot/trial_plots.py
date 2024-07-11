from bokeh.layouts import column
from .bokeh_base import *
from ...wheelTrace import WheelTrace


class TrialGraph(Graph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # trial epoch areas
        self.cds_areas = None

        # lines for reactions and reward
        self.cds_lines = None

        # licks
        self.cds_licks = None

        # wheel related
        self.cds_wheel = None
        self.cds_interp = None
        self.cds_region = None
        self.cds_move = None
        self.cds_reactions = None

        self.thresh_in_ticks = kwargs.get("tick_thresh", 10)
        self.speed_thresh = self.thresh_in_ticks / kwargs.get(
            "time_thresh", 17
        )  # 17 is the avg ms for a loop
        self.traj = WheelTrace()

    def set_cds(self, data: pl.DataFrame) -> None:
        """Creates a column data source(cds)"""
        regions_df = data.select([c for c in data.columns if c.startswith("t_")])

        if data[0, "state_outcome"] == -1:
            # early
            anchor = "t_trialinit"
            regions_df = regions_df.with_columns(
                [
                    (pl.col("t_trialstart") - pl.col("t_trialinit")).alias("trialstart"),
                    (pl.col("t_trialinit") - pl.col("t_trialinit")).alias("trialinit"),
                    (pl.col("t_blank_dur")).alias("stimstart"),
                    (pl.col("t_blank_dur")).alias("stimend"),
                ]
            )

            time_anchor = data[0, anchor]
            time_window = [time_anchor, time_anchor + regions_df[0, "stimend"]]
        else:
            if data[0, "t_stimstart_rig"] is not None:
                anchor = "t_stimstart_rig"
                end = "t_stimend_rig"
            else:
                anchor = "t_stimstart"
                end = "t_stimend"
            regions_df = regions_df.with_columns(
                [
                    (pl.col("t_trialstart") - pl.col(anchor)).alias("trialstart"),
                    (pl.col("t_trialinit") - pl.col(anchor)).alias("trialinit"),
                    (pl.col("t_stimstart") - pl.col(anchor)).alias("stimstart"),
                    (pl.col(end) - pl.col(anchor)).alias("stimend"),
                ]
            )

            time_anchor = data[0, anchor]
            time_window = [time_anchor, data[0, end]]  # from timeanchor to

        if data[0, "opto"] == 1:
            regions_df = regions_df.with_columns(
                [
                    pl.col("stimstart").alias("optostart"),
                    pl.col("stimend").alias("optoend"),
                ]
            )
        else:
            regions_df = regions_df.with_columns(
                [pl.lit(None).alias("optostart"), pl.lit(None).alias("optoend")]
            )

        temp_areas = regions_df.to_pandas()

        # temp_licks = {'lick_time':None}
        # temp_lines = {'response_latency' : None}

        self.traj.set_trace_data(
            data[0, "wheel_time"].to_list(), data[0, "wheel_pos"].to_list()
        )

        # initialize the trace
        self.traj.init_trace(time_anchor=time_anchor)
        # get the movements from interpolated positions
        self.traj.get_movements(pos_thresh=0.03, t_thresh=0.5)
        print(time_window, flush=True)
        interval_mask = self.traj.make_interval_mask(time_window=time_window)
        self.traj.select_trace_interval(mask=interval_mask)

        # get all the reaction times and outcomes here:
        self.traj.get_speed_reactions(speed_threshold=self.speed_thresh)
        self.traj.get_tick_reactions(tick_threshold=self.thresh_in_ticks)

        temp_interp = {"t": self.traj.tick_t_interp, "pos": self.traj.tick_pos_interp}

        colors = ["#17e800", "#074500", "#0096ed", "#004973", "#910011"]
        reactions = [
            self.traj.pos_reaction_t,
            self.traj.pos_decision_t,
            self.traj.speed_reaction_t,
            self.traj.speed_decision_t,
            data[0, "response_latency"],
        ]

        temp_reactions = {"reactions": reactions, "colors": colors}

        temp_wheel = {"wheel_time": self.traj.tick_t, "wheel_pos": self.traj.tick_pos}

        temp_region = {
            "t_region": self.traj.trace_interval["t"],
            "pos_region": self.traj.trace_interval["pos"],
            "speed_region": self.traj.trace_interval["velo"],
        }

        temp_move = {
            "t_move": self.traj.trace_interval["t_movements"],
            "pos_move": self.traj.trace_interval["pos_movements"],
            "speed_move": self.traj.trace_interval["velo_movements"],
        }

        if self.cds_areas is None:
            self.cds_wheel = ColumnDataSource(data=temp_wheel)
            self.cds_areas = ColumnDataSource(data=temp_areas)
            self.cds_interp = ColumnDataSource(data=temp_interp)
            self.cds_region = ColumnDataSource(data=temp_region)
            self.cds_move = ColumnDataSource(data=temp_move)
            self.cds_reactions = ColumnDataSource(data=temp_reactions)
        else:
            self.cds_wheel.data = temp_wheel
            self.cds_areas.data = temp_areas
            self.cds_interp.data = temp_interp
            self.cds_region.data = temp_region
            self.cds_move.data = temp_move
            self.cds_reactions.data = temp_reactions

    def plot(self, **kwargs):
        """ """
        f_top = figure(title="", width=700, height=80, x_axis_location="above")
        f_top.toolbar.logo = None
        f_top.toolbar_location = None
        f_top.xgrid.grid_line_color = None
        f_top.ygrid.grid_line_color = None
        f_top.axis.visible = False

        f_top.vstrip(
            x0="optostart",
            x1="optoend",
            source=self.cds_areas,
            color="#2b3cfc",
            fill_alpha=0.5,
            line_alpha=0,
        )

        f = figure(
            title=None,
            width=700,
            height=400,
            x_range=f_top.x_range,
            y_range=f_top.y_range,
            x_axis_label="Time(ms)",
            y_axis_label="Wheel Position",
        )

        # quiescence
        f.vstrip(
            x0="trialstart",
            x1="trialinit",
            line_alpha=0,
            fill_alpha=0.3,
            fill_color="#d1d1d1",
            hatch_pattern="/",
            hatch_alpha=0.1,
            source=self.cds_areas,
        )

        # blank
        f.vstrip(
            x0="trialinit",
            x1="stimstart",
            line_alpha=0,
            fill_alpha=0.3,
            fill_color="#c22121",
            hatch_pattern="x",
            hatch_alpha=0.1,
            source=self.cds_areas,
        )

        # response window
        f.vstrip(
            x0="stimstart",
            x1="stimend",
            line_alpha=0,
            fill_alpha=0.3,
            fill_color="#009912",
            hatch_pattern="x",
            hatch_alpha=0.1,
            source=self.cds_areas,
        )

        # plot wheel traces
        # data points
        f.circle(
            "wheel_time", "wheel_pos", color="#000000", size=5, source=self.cds_wheel
        )

        # interp
        f.line(
            "t",
            "pos",
            color="#8c8c8c",
            line_dash="dashed",
            line_width=2,
            source=self.cds_interp,
        )

        # analysis region
        f.line(
            "t_region",
            "pos_region",
            color="#c4a8f0",
            line_width=4,
            source=self.cds_region,
        )

        # movements
        f.multi_line(
            "t_move", "pos_move", color="#4a326e", line_width=5, source=self.cds_move
        )

        f.vspan("reactions", line_width=2, line_color="colors", source=self.cds_reactions)

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

        self.fig = column(f_top, f)
