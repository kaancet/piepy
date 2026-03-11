import numpy as np
import temporaldata as td

from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, Range1d, LinearAxis

from piepy.core.run import Run
from piepy.psychophysics.wheel.wheelTrace import WheelTrace


STIM_SIDE_MAP = {"catch": -0.5, "ipsi": -1, "contra": 0, None: np.nan}

STIM_TYPE_MAP = {"0.04cpd_8.0Hz": "#FF7F0F", "0.16cpd_0.5Hz": "#0099C2", None: ""}

OUTCOME_MAP = {"early": "#ACACAC", "miss": "#990000", "hit": "#009900"}


class LogMovieMaker:
    def __init__(self) -> None:
        self.run_mat = None

    def extract_run(self, run: Run) -> td.Data:
        """Generates a n x t matrix where n is all the events and t is the time points

        Returns:
            tuple[np.ndarray, np.ndarray]: _description_
        """
        trace = WheelTrace()
        # loop through the keys and put all the columns of the dataframes into the matrix
        temp = {}
        for k, v in run.rawdata.items():
            if not v.is_empty():
                if k == "vstim":
                    pass

                elif k == "statemachine":
                    pass

                elif k in ["imaging", "eyecam", "facecam", "onepcam"]:
                    # calculate sampling rate first
                    _sampling_rate = 1 / np.mean(
                        np.diff(v["duinotime"].to_numpy() / 1000)
                    )
                    temp[k] = td.RegularTimeSeries(
                        sampling_rate=round(_sampling_rate, 2),
                        frame_no=v["value"].to_numpy(),
                        domain="auto",
                    )

                elif k == "position":
                    t = v["duinotime"].to_numpy()
                    pos = v["value"].to_numpy()
                    t, pos = trace.fix_trace_timing(t, pos)

                    _, _, t_interp, tick_interp = trace.reset_and_interpolate(
                        t, pos, 0, 1
                    )
                    pos_interp = trace.cm_to_rad(trace.ticks_to_cm(tick_interp))

                    temp["wheel_position"] = td.IrregularTimeSeries(
                        timestamps=t_interp / 1000,  # time needs to be in seconds
                        count=pos_interp,
                        timekeys=["timestamps"],
                        domain="auto",
                    )

                    speed = (
                        np.abs(trace.get_filtered_velocity(pos_interp, 1, 0.4)) * 1000
                    )

                    temp["wheel_speed"] = td.IrregularTimeSeries(
                        timestamps=t_interp / 1000,  # time needs to be in seconds
                        count=speed,
                        timekeys=["timestamps"],
                        domain="auto",
                    )

                elif k in ["screen", "lick", "reward", "opto"]:
                    temp[k] = td.IrregularTimeSeries(
                        timestamps=v["duinotime"].to_numpy()
                        / 1000,  # time needs to be in seconds
                        count=v["value"].to_numpy(),
                        timekeys=["timestamps"],
                        domain="auto",
                    )

        temp["trial"] = td.Interval(
            start=run.data.data["t_trialinit"].to_numpy() / 1000,
            end=run.data.data["t_trialend"].to_numpy() / 1000,
            stim_start=run.data.data["t_vstimstart"].to_numpy() / 1000,
            stim_end=run.data.data["t_vstimend"].to_numpy() / 1000,
            contrast=run.data.data["contrast"].to_numpy(),
            stim_type=run.data.data["stim_type"].to_numpy(),
            stim_side=run.data.data["stim_side"].to_numpy(),
            opto_pattern=run.data.data["opto_pattern"].to_numpy(),
            outcome=run.data.data["outcome"].to_numpy(),
            timekeys=["start", "end", "stim_start", "stim_end"],
        )

        self.run_mat = td.Data(**temp, domain="auto")
        return self.run_mat

    def make_cds(
        self, t_start: float | None = None, t_end: float | None = None, **kwargs
    ) -> None:
        """_summary_

        Args:
            t_start (float | None, optional): _description_. Defaults to None.
            t_end (float | None, optional): _description_. Defaults to None.
        """
        if t_start is None:
            t_start = self.run_mat.absolute_start

        if t_end is None:
            t_end = self.run_mat.end

        data = self.run_mat.slice(t_start, t_end, reset_origin=False)

        # pad the event arrays
        padded = {}

        for kk in ["lick", "reward"]:
            temp_t = np.zeros_like(data.wheel_speed.timestamps)
            temp_t[:] = np.nan
            _subset = data.__dict__[kk].timestamps
            _idxs = np.where(np.isin(data.wheel_speed.timestamps, _subset))[0]
            temp_t[_idxs] = data.wheel_speed.timestamps[_idxs]
            padded[kk] = temp_t

        # rig events
        self.cds_rig = ColumnDataSource(
            data={
                "t": data.wheel_speed.timestamps,
                "wheel_speed": data.wheel_speed.count,
                "wheel_pos": data.wheel_position.count,
                "lick": padded["lick"],
                "reward": padded["reward"],
            }
        )

        # intervals
        _intervals = {
            k: data.trial.__dict__[k]
            for k in ["start", "end", "stim_start", "stim_end"]
        }

        tt = np.zeros_like(data.trial.stim_start)
        tt[:] = np.nan

        _idxs = np.where(np.isin(data.trial.stim_start, data.opto.timestamps))
        tt[_idxs] = data.trial.stim_start[_idxs]

        _opto_pulse = {
            "opto_start": tt,
            "opto_end": data.trial.__dict__["stim_end"],
        }

        _trial_props = {
            "contrast": data.trial.contrast * 2,
            "stim_type": np.array(
                [STIM_TYPE_MAP[k] for k in data.trial.stim_type], dtype=object
            ),
            "stim_side": np.array([STIM_SIDE_MAP[k] for k in data.trial.stim_side]),
            "stim_side_h": np.array(
                [STIM_SIDE_MAP[k] + 1 for k in data.trial.stim_side],
            ),
            "outcome": np.array(
                [OUTCOME_MAP[k] for k in data.trial.outcome], dtype=object
            ),
        }

        self.cds_interval = ColumnDataSource(
            data={**_intervals, **_opto_pulse, **_trial_props}
        )

    def animate_run(
        self, t_start: float | None = None, t_end: float | None = None, **kwargs
    ) -> None:
        """Animate the run

        Args:
            t_start (float | None, optional): _description_. Defaults to None.
            t_end (float | None, optional): _description_. Defaults to None.
        """
        if self.run_mat is None:
            self.extract_run()

        # make cds's
        self.make_cds(t_start=t_start, t_end=t_end, **kwargs)

        # making the bokeh
        self.fig = figure(title=None, width=2000, height=600, y_range=(-1, 1))

        # trial strips
        self.fig.quad(
            top="stim_side_h",
            bottom="stim_side",
            left="start",
            right="end",
            fill_color="#AFAFAF",
            fill_alpha=0.2,
            line_width=0,
            line_alpha=0,
            source=self.cds_interval,
        )

        # stim quads
        self.fig.quad(
            top="stim_side_h",
            bottom="stim_side",
            left="stim_start",
            right="stim_end",
            color="stim_type",
            alpha="contrast",
            source=self.cds_interval,
        )

        # opto markers
        self.fig.hbar(
            y="stim_side_h",
            height=0.15,
            left="opto_start",
            right="opto_end",
            color="#4477FF",
            source=self.cds_interval,
        )

        # outcome markers
        self.fig.scatter(
            x="end",
            y="stim_side_h",
            color="outcome",
            size=20,
            marker="inverted_triangle",
            source=self.cds_interval,
        )

        # lick
        self.fig.scatter(
            x="lick",
            y=0,
            marker="circle",
            size=10,
            color="#11DDFF",
            source=self.cds_rig,
        )

        # reward
        self.fig.scatter(
            x="reward",
            y=0,
            marker="diamond_dot",
            size=20,
            color="#E3038C",
            source=self.cds_rig,
        )

        self.fig.extra_y_ranges["wheel"] = Range1d(0, 10)
        self.fig.extra_y_ranges["wheel_p"] = Range1d(-20, 20)

        # wheel traces
        self.fig.line(
            x="t",
            y="wheel_speed",
            line_color="#010101",
            line_alpha=0.6,
            source=self.cds_rig,
            y_range_name="wheel",
        )

        self.fig.line(
            x="t",
            y="wheel_pos",
            line_color="#FF0000",
            source=self.cds_rig,
            y_range_name="wheel_p",
        )

        ax2 = LinearAxis(y_range_name="wheel", axis_label="Wheel speed (rad/s)")
        self.fig.add_layout(ax2, "left")

        ax3 = LinearAxis(y_range_name="wheel_p", axis_label="Wheel speed (rad)")
        self.fig.add_layout(ax3, "right")

        show(self.fig)

    def make_movie(self) -> None:
        """_summary_"""
        pass
