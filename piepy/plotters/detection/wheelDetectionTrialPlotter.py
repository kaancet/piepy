from ..basePlotters import *
from ...psychophysics.wheelTrace import WheelTrace


class DetectionTrialPlotter:
    def __init__(self, data: pl.DataFrame) -> None:
        self.data = data

    @staticmethod
    def get_trial_variables(trial_data: pl.DataFrame) -> dict:
        """Extracts the static trial variables and puts them in a dict"""
        # side, answer, sftf, contrast
        return {
            "stim_side": trial_data[0, "stim_side"],
            "contrast": trial_data[0, "contrast"],
            "state_outcome": trial_data[0, "state_outcome"],
            "wheel_outcome": trial_data[0, "wheel_outcome"],
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

        self.trial_data = self.data.filter(pl.col("trial_no") == trial_no)
        if self.trial_data[0, "outcome"] == -1:
            time_anchor = "t_trialstart"
        else:
            time_anchor = "t_stimstart_rig"

        self.trial_data = self.trial_data.with_columns(
            [
                (pl.col("t_trialstart") - pl.col(time_anchor)).suffix("_reset"),
                (pl.col("t_stimstart_absolute") - pl.col(time_anchor)).suffix("_reset"),
                (
                    pl.when(pl.col("t_stimstart_rig").is_not_null())
                    .then(pl.col("t_stimstart_rig") - pl.col(time_anchor))
                    .otherwise(pl.col("t_blank_dur"))
                ).alias("t_stimstart_rig_reset"),
                (pl.col("t_stimend_rig") - pl.col(time_anchor)).suffix("_reset"),
            ]
        )

        # plot the regions
        # quiescence
        ax.axvspan(
            self.trial_data[0, "t_trialstart_reset"],
            self.trial_data[0, "t_trialstart_reset"]
            + self.trial_data[0, "t_quiescence_dur"],
            color="gray",
            alpha=0.3,
            label=f'Trial Prep. ({round(self.trial_data[0,"t_quiescence_dur"],1)}ms)',
        )
        # blank
        ax.axvspan(
            self.trial_data[0, "t_trialstart_reset"]
            + self.trial_data[0, "t_quiescence_dur"],
            self.trial_data[0, "t_stimstart_rig_reset"],
            color="orange",
            alpha=0.3,
            label=f'Wait for Stim ({round(self.trial_data[0,"t_blank_dur"],1)}ms)',
        )

        ax.axvspan(
            self.trial_data[0, "t_stimstart_rig_reset"],
            self.trial_data[0, "t_stimend_rig_reset"],
            color="green",
            alpha=0.3,
            label="Response Window",
        )

        if (
            self.trial_data[0, "opto_pattern"] != -1
            and self.trial_data[0, "outcome"] != -1
        ):
            ax.barh(
                ax.get_ylim()[1],
                self.trial_data[0, "response_latency"],
                left=0,
                height=10,
                color="aqua",
            )

        # plot the wheels
        wheel_time = self.trial_data[0, "wheel_time"].to_list()
        wheel_pos = self.trial_data[0, "wheel_pos"].to_list()

        onsets = self.trial_data[0, "wheel_onsets"].to_list()
        offsets = self.trial_data[0, "wheel_offsets"].to_list()

        # interpolate the wheels
        pos, t = interpolate_position(wheel_time, wheel_pos, freq=20)

        ax.plot(t, pos, color="gray", linewidth=3, label="Wheel Trace interp.")

        ax.plot(wheel_time, wheel_pos, "k+", linewidth=3, label="Wheel Trace")

        # speed
        wheel_speed = self.trial_data[0, "wheel_speed"].to_list()
        ax2 = ax.twinx()
        ax2.plot(wheel_time[:-1], wheel_speed, color="tab:orange", label="Wheel Speed")
        ax2.set_ylabel("Wheel Speed (cm/s)", fontsize=fontsize)
        ax2.tick_params(labelsize=fontsize)

        # plot movements
        for i, o in enumerate(onsets):
            on_idx, on_t = find_nearest(t, o)
            off_idx, off_t = find_nearest(t, offsets[i])
            ax.scatter(on_t, pos[on_idx], s=50, color="g")
            ax.plot(
                t[on_idx:off_idx],
                pos[on_idx:off_idx],
                "b",
                label="Detected Movements" if i == 0 else "_",
            )
            ax.scatter(off_t, pos[off_idx], s=50, color="r")

        # delta tick
        if self.trial_data[0, "wheel_reaction_time"] is not None:
            ax.axvline(
                self.trial_data[0, "wheel_reaction_time"],
                color="purple",
                linestyle="-.",
                linewidth=2,
                label=f"Wheel Tick Reaction({round(self.trial_data[0,'wheel_reaction_time'],1)}ms)",
            )

            past_thresh_idx, _ = find_nearest(
                t, self.trial_data[0, "wheel_reaction_time"]
            )
            ax.axhline(
                pos[past_thresh_idx],
                color="purple",
                linestyle=":",
                linewidth=2,
                label=f" Wheel Tick Threshold ({round(pos[past_thresh_idx],2)} cm)",
            )

        # speed
        if self.trial_data[0, "wheel_speed_reaction_time"] is not None:
            ax2.axvline(
                self.trial_data[0, "wheel_speed_reaction_time"],
                color="dodgerblue",
                linestyle="-.",
                linewidth=2,
                label=f"Wheel Speed Reaction({round(self.trial_data[0,'wheel_speed_reaction_time'],1)}ms)",
            )

            speed_past_thresh = interp1d(wheel_time[:-1], wheel_speed)(
                self.trial_data[0, "wheel_speed_reaction_time"]
            )
            ax2.axhline(
                speed_past_thresh,
                color="dodgerblue",
                linestyle=":",
                linewidth=2,
                label=f"Speed Threshold ({round(speed_past_thresh.tolist(),4)} cm/ms)",
            )

        ax.axvline(
            self.trial_data[0, "response_latency"],
            color="r",
            linestyle=":",
            linewidth=2,
            label=f"State Response Time({self.trial_data[0,'response_latency']}ms)",
        )

        # plot the reward
        reward = self.trial_data[0, "reward"]
        if reward is not None:
            reward = reward[0] - self.trial_data[0, time_anchor]
            ax.axvline(reward, color="darkgreen", linestyle="--", label="Reward")

        # plot the lick
        lick_arr = self.trial_data["lick"].explode().to_numpy()
        if len(lick_arr):
            lick_arr = [l - self.trial_data[0, time_anchor] for l in lick_arr]
            ax.scatter(lick_arr, [0] * len(lick_arr), marker="|", c="darkblue", s=50)

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

        ax.set_ylabel("Wheel Position (cm)", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)

        ax.set_xlim(t_lim)

        ax.spines["bottom"].set_position(("outward", 10))
        ax.spines["left"].set_position(("outward", 10))
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        ax2.spines["right"].set_position(("outward", 10))
        ax2.spines["left"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.spines["bottom"].set_visible(False)

        ax.legend(
            loc="center left", bbox_to_anchor=(1.2, 0.5), fontsize=fontsize, frameon=False
        )
        ax2.legend(
            loc="center left",
            bbox_to_anchor=(1.2, 0.02),
            fontsize=fontsize,
            frameon=False,
        )

        return ax
