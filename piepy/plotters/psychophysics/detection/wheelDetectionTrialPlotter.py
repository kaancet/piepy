from ..basePlotters import *
from ...psychophysics.wheelTrace import WheelTrace


class DetectionTrialPlotter:
    def __init__(self, data: pl.DataFrame) -> None:
        self.data = data
        self.trace = WheelTrace()

    @staticmethod
    def get_trial_variables(trial_data: pl.DataFrame) -> dict:
        """Extracts the static trial variables and puts them in a dict"""
        # side, answer, sftf, contrast
        return {
            "stim_side": trial_data[0, "stim_side"],
            "contrast": trial_data[0, "contrast"],
            "outcome": trial_data[0, "state_outcome"],
            "sf": trial_data[0, "spatial_freq"],
            "tf": trial_data[0, "temporal_freq"],
        }

    def plot(self, ax: plt.Axes = None, trial_no: int = 3, t_lim: list = None, **kwargs):
        fontsize = kwargs.pop("fontsize", 25)
        if ax is None:
            self.fig = plt.figure(figsize=kwargs.pop("figsize", (15, 8)))
            ax = self.fig.add_subplot(1, 1, 1)

        if t_lim is None:
            t_lim = [-500, 1200]

        tick_thresh = kwargs.pop("tick_thresh", 20)

        self.trial_data = self.data.filter(pl.col("trial_no") == trial_no)
        if self.trial_data[0, "outcome"] == -1:
            time_anchor = "t_trialinit"
        else:
            time_anchor = "t_stimstart_rig"

        self.trial_data = self.trial_data.with_columns(
            [
                (pl.col("t_trialstart") - pl.col(time_anchor)).name.suffix("_reset"),
                (pl.col("t_trialinit") - pl.col(time_anchor)).name.suffix("_reset"),
                (
                    pl.when(pl.col("t_stimstart_rig").is_not_null())
                    .then(pl.col("t_stimstart_rig") - pl.col(time_anchor))
                    .otherwise(pl.col("t_blank_dur"))
                ).alias("t_stimstart_reset"),
                (pl.col("t_stimend") - pl.col(time_anchor)).name.suffix("_reset"),
            ]
        )

        # plot the regions
        # quiescence
        ax.axvspan(
            self.trial_data[0, "t_trialstart_reset"],
            self.trial_data[0, "t_trialinit_reset"],
            color="#494949",
            alpha=0.3,
            label=f'Trial Prep. ({round(self.trial_data[0,"t_quiescence_dur"],1)}ms)',
        )
        # blank
        ax.axvspan(
            self.trial_data[0, "t_trialinit_reset"],
            self.trial_data[0, "t_stimstart_reset"],
            color="#a7a7a7",
            alpha=0.3,
            label=f'Wait for Stim ({round(self.trial_data[0,"t_blank_dur"],1)}ms)',
        )

        if self.trial_data[0, "outcome"] != -1:
            ax.axvspan(
                self.trial_data[0, "t_stimstart_reset"],
                self.trial_data[0, "t_stimend_reset"],
                color="green",
                alpha=0.3,
                label="Response Window",
            )

        # if opto plot a opto pulse
        if (
            self.trial_data[0, "opto_pattern"] != -1
            and self.trial_data[0, "outcome"] != -1
        ):
            _opto_start = (
                self.trial_data[0, "opto_pulse"] - self.trial_data[0, time_anchor]
            )
            ax.barh(
                y=4.75,
                width=self.trial_data[0, "t_stimend_reset"] - _opto_start,
                left=_opto_start,
                height=0.5,
                color="#0800f5",
                alpha=0.5,
            )

        # plot the wheels
        wheel_time = self.trial_data[0, "wheel_time"].to_numpy()
        wheel_pos = self.trial_data[0, "wheel_pos"].to_numpy()

        self.trace.set_trace_data(wheel_time, wheel_pos)
        # initialize the trace
        time_anchor_ms = self.trial_data[0, time_anchor]
        self.trace.init_trace(time_anchor=time_anchor_ms)
        # get the movements from interpolated positions
        self.trace.get_movements(
            pos_thresh=kwargs.pop("pos_thresh", 0.02),
            t_thresh=kwargs.pop("t_thresh", 0.5),
        )

        time_window = [time_anchor_ms, time_anchor_ms + 5000]
        # these has to be run before calculating reaction times to constrain the region of traces we are interested in
        interval_mask = self.trace.make_interval_mask(time_window=time_window)
        self.trace.select_trace_interval(mask=interval_mask)

        # get all the reaction times and outcomes here:
        # self.trace.get_speed_reactions(speed_threshold=speed_thresh)
        # self.trace.get_tick_reactions(tick_threshold=thresh_in_ticks)

        onsets = self.trial_data[0, "onsets"].to_list()
        offsets = self.trial_data[0, "offsets"].to_list()

        ax2 = ax.twinx()
        wheel_pos_rad = self.trace.cm_to_rad(
            self.trace.ticks_to_cm(np.array(self.trace.tick_pos))
        )
        ax2.plot(
            self.trace.tick_t,
            wheel_pos_rad,
            color="#b03d04",
            linewidth=2,
            marker="+",
            markersize=15,
            label="Wheel Trace",
        )

        # # speed
        wheel_speed = np.abs(self.trace.velo_interp) * 1000  # make it seconds

        ax.plot(
            self.trace.tick_t_interp,
            wheel_speed,
            color="k",
            linewidth=2.5,
            label="Wheel Speed",
        )

        # ax.plot(self.trace.tick_t_interp, self.trace.tick_pos_interp, "k+", linewidth=3, label="Wheel Trace inter")

        # # plot movements
        # for i, o in enumerate(onsets):
        #     on_idx, on_t = find_nearest(t, o)
        #     off_idx, off_t = find_nearest(t, offsets[i])
        #     ax.scatter(on_t, pos[on_idx], s=50, color="g")
        #     ax.plot(
        #         t[on_idx:off_idx],
        #         pos[on_idx:off_idx],
        #         "b",
        #         label="Detected Movements" if i == 0 else "_",
        #     )
        #     ax.scatter(off_t, pos[off_idx], s=50, color="r")

        # # delta tick
        # if self.trial_data[0, "wheel_reaction_time"] is not None:
        #     ax.axvline(
        #         self.trial_data[0, "wheel_reaction_time"],
        #         color="purple",
        #         linestyle="-.",
        #         linewidth=2,
        #         label=f"Wheel Tick Reaction({round(self.trial_data[0,'wheel_reaction_time'],1)}ms)",
        #     )

        #     past_thresh_idx, _ = find_nearest(
        #         t, self.trial_data[0, "wheel_reaction_time"]
        #     )
        #     ax.axhline(
        #         pos[past_thresh_idx],
        #         color="purple",
        #         linestyle=":",
        #         linewidth=2,
        #         label=f" Wheel Tick Threshold ({round(pos[past_thresh_idx],2)} cm)",
        #     )

        # # speed
        # if self.trial_data[0, "wheel_speed_reaction_time"] is not None:
        #     ax2.axvline(
        #         self.trial_data[0, "wheel_speed_reaction_time"],
        #         color="dodgerblue",
        #         linestyle="-.",
        #         linewidth=2,
        #         label=f"Wheel Speed Reaction({round(self.trial_data[0,'wheel_speed_reaction_time'],1)}ms)",
        #     )

        #     speed_past_thresh = interp1d(wheel_time[:-1], wheel_speed)(
        #         self.trial_data[0, "wheel_speed_reaction_time"]
        #     )
        #     ax2.axhline(
        #         speed_past_thresh,
        #         color="dodgerblue",
        #         linestyle=":",
        #         linewidth=2,
        #         label=f"Speed Threshold ({round(speed_past_thresh.tolist(),4)} cm/ms)",
        #     )

        # response times
        if self.trial_data[0, "outcome"] == 1:
            ax.axvline(
                self.trial_data[0, "response_latency"],
                color="r",
                linestyle=":",
                linewidth=2,
                label=f"State Response Time({round(self.trial_data[0,'response_latency'],3)}ms)",
            )

            ax.axvline(
                self.trial_data[0, "rig_reaction_time"],
                color="#1d544d",
                linestyle="-",
                linewidth=2,
                label=f"Rig reaction time({round(self.trial_data[0,'rig_reaction_time'],3)}ms)",
            )

            ax.axvline(
                self.trial_data[0, "speed_reaction_time"],
                color="#004f0c",
                linestyle="-",
                linewidth=2,
                label=f"post-hoc speed reaction time({round(self.trial_data[0,'speed_reaction_time'],3)}ms)",
            )

            ax.axvline(
                self.trial_data[0, "speed_decision_time"],
                color="#960094",
                linestyle="-",
                linewidth=2,
                label=f"post-hoc speed decision time({round(self.trial_data[0,'speed_decision_time'],3)}ms)",
            )

        if self.trial_data[0, "outcome"] != -1:
            speed_thresh = tick_thresh / (
                self.trial_data[0, "median_loop_time"] * 5
            )  # 20 is thresholdStim, 5 is wheelbuffer
            _deg_thresh = self.trace.cm_to_rad(self.trace.ticks_to_cm(speed_thresh))

            ax.axhline(_deg_thresh * 1000, color="red")  # *1000 to make it per seconds

        # plot the reward
        reward = self.trial_data[0, "reward"]
        if reward is not None:
            reward = reward[0] - self.trial_data[0, time_anchor]
            # ax.axvline(reward, color="darkgreen", linestyle="--", label="Reward")
            ax.vlines(reward, 0, 0.5, linewidth=5, color="darkblue")

        # plot the lick
        lick_arr = self.trial_data["lick"].explode().to_numpy()
        if len(lick_arr):
            lick_arr = [l - self.trial_data[0, time_anchor] for l in lick_arr]
            ax.vlines(lick_arr, 0.1, 0.4, linewidth=3, color="#1ca4ed")
            # ax.scatter(lick_arr, [0] * len(lick_arr), marker="|", c="#", s=250)

        # prettify
        trial_vars = self.get_trial_variables(self.trial_data)
        title = f"Trial No : {trial_no}  "
        for k, v in trial_vars.items():
            title += f"{k}={v}, "
            if k == "contrast":
                title += "\n"
        ax.set_title(title, fontsize=fontsize)
        if self.trial_data[0, "outcome"] != -1:
            ax.set_xlabel("Time from Stim onset (ms)", fontsize=fontsize)
        else:
            ax.set_xlabel("Time from Trial Start (ms)", fontsize=fontsize)

        ax.set_ylabel("Wheel Speed (rad/s)", fontsize=fontsize)
        ax2.set_ylabel("wheel Movement (rad)", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax2.tick_params(labelsize=fontsize)
        ax2.tick_params(axis="y", colors="#b03d04")

        ax.set_xlim(t_lim)

        ax.spines["bottom"].set_position(("outward", 10))
        ax.spines["left"].set_position(("outward", 10))
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        ax2.spines["left"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.spines["bottom"].set_visible(False)
        ax2.spines["right"].set_position(("outward", 10))
        ax2.spines["right"].set_color("#b03d04")
        # ax2.spines["left"].set_visible(False)
        # ax2.spines["top"].set_visible(False)
        # ax2.spines["bottom"].set_visible(False)

        ax.legend(
            loc="center left", bbox_to_anchor=(1.1, 0.5), fontsize=fontsize, frameon=False
        )
        ax2.legend(
            loc="center left",
            bbox_to_anchor=(1.1, 0.02),
            fontsize=fontsize,
            frameon=False,
        )

        return ax
