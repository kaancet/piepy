from ..wheelUtils import *
from ..core.trial import *


class WheelTrial(Trial):
    __slots__ = ["answertime", "wheelgain", "rewardsize"]

    def __init__(self, trial_no: int, log_column_keys: dict, meta) -> None:
        super().__init__(trial_no, log_column_keys, meta)

    def get_vstim_props(self) -> dict:
        """Extracts the necessary properties from vstim data"""
        ignore = ["iTrial", "photo", "code", "presentTime"]
        vstim_dict = {}
        vstim = self.data["vstim"]

        # this is an offline fix for a vstim logging issue where time increment messes up vstim logging
        vstim = vstim[:-1]

        # fix for swapped left right logging(for data before 26/01/2021)
        vstim_columns_tofix = list(vstim.columns)
        idx_posl = vstim_columns_tofix.index("posx_l")
        idx_posr = vstim_columns_tofix.index("posx_r")
        if idx_posr < idx_posl:
            vstim_columns_tofix[idx_posr], vstim_columns_tofix[idx_posl] = (
                vstim_columns_tofix[idx_posl],
                vstim_columns_tofix[idx_posr],
            )
            vstim.columns = vstim_columns_tofix

        for col in vstim.columns:
            if col in ignore:
                continue
            temp_col = vstim[col]
            # if a column has all the same values, take the first entry of the column as the value
            # sf, tf, contrast, stim_side, correct, opto, opto_pattern should run through here
            uniq = np.unique(temp_col)
            uniq = uniq[~np.isnan(uniq)]

            # failsafe for animal not moving the wheel
            # TODO: maybe not do this to save memory and time, and keep non-moved trials' values as a single value?
            if len(uniq) == 1 and col != "posx_r" and col != "posx_l":
                vstim_dict[col] = temp_col.iloc[0]
            # if different values exist in the column, take it as a list
            # stim_pos runs through here
            else:
                vstim_dict[col] = np.array(temp_col)

        # this is to make things easier later in the analysis, maybe not the best way
        vstim_dict["contrast"] = (
            vstim_dict["contrast_r"]
            if vstim_dict["correct"]
            else vstim_dict["contrast_l"]
        )
        vstim_dict["stim_pos"] = (
            vstim_dict["posx_r"] if vstim_dict["correct"] else vstim_dict["posx_l"]
        )
        vstim_dict["spatial_freq"] = (
            vstim_dict["sf_r"] if vstim_dict["correct"] else vstim_dict["sf_l"]
        )
        vstim_dict["temporal_freq"] = (
            vstim_dict["tf_r"] if vstim_dict["correct"] else vstim_dict["tf_l"]
        )
        try:
            vstim_dict["stim_side"] = vstim_dict["stim_pos"][0]
        except:
            display(
                f' >>>>> WARNING <<<<< Trial {self.trial_no} has an issue: {vstim_dict["stim_pos"]}'
            )
            vstim_dict["stim_side"] = 1 if vstim_dict["correct"] else -1
        return vstim_dict

    def get_wheel_pos(self, anchor: float) -> np.ndarray:
        """Extracts the wheel trajectories and resets the positions according to anchor"""
        wheel_deg_per_tick = (self.meta.wheelGain * WHEEL_CIRCUM) / WHEEL_TICKS_PER_REV

        wheel_data = self.data["position"]
        wheel_arr = np.array(wheel_data[["duinotime", "value"]])

        # resetting the wheel position so the 0 point is aligned with anchor point and converting encoder ticks into degrees
        # also resetting the time frame into the trial itself rather than the whole session
        if len(wheel_arr):
            reset_idx = find_nearest(wheel_arr[:, 0], anchor)[0]
            wheel_arr[:, 1] = (
                np.apply_along_axis(reset_wheel_pos, 0, wheel_arr[:, 1], reset_idx)
                * wheel_deg_per_tick
            )

            wheel_arr[:, 0] = np.apply_along_axis(
                lambda x: x - anchor, 0, wheel_arr[:, 0]
            )

        return wheel_arr

    def trial_data_from_logs(self) -> list:
        """Iterates over each state change in a DataFrame slice that belongs to one trial(and corrections for pyvstim)
        Returns a list of dictionaries that have data parsed from stimlog and riglog
        """
        trial_log_data = {"trial_no": int(self.trial_no)}
        # iterrows is faster for small DataFrames
        for _, row in self.data["state"].iterrows():
            curr_trans = row["transition"]
            if curr_trans is None:
                # this is a failsafe for non-existant state transitions,
                # mostly for the bug where the state tries to transition right after entering a new state(220222)
                continue
            # trial start
            if curr_trans == "trialstart":
                trial_log_data["trial_start"] = row[self.column_keys["elapsed"]]

            # stim start
            elif curr_trans == "openloopstart":
                trial_log_data["openloop_start"] = row[self.column_keys["elapsed"]]
                trial_log_data["correction"] = row[self.column_keys["trialType"]]
                trial_log_data["stim_start"] = row[self.column_keys["elapsed"]]
                trial_log_data["openstart_absolute"] = row[self.column_keys["elapsed"]]
            # move start
            elif curr_trans == "closedloopstart":
                trial_log_data["closedloop_start"] = row[self.column_keys["elapsed"]]

            # correct
            elif curr_trans == "correct":
                trial_log_data["closedloop_end"] = row[self.column_keys["elapsed"]]
                trial_log_data["response_latency"] = row[self.column_keys["stateElapsed"]]
                trial_log_data["answer"] = 1

            # incorrect
            elif curr_trans == "incorrect":
                trial_log_data["closedloop_end"] = row[self.column_keys["elapsed"]]
                trial_log_data["response_latency"] = row[self.column_keys["stateElapsed"]]
                # stimpy has no seperate state for no answer (maybe should be added?) so it is extracted from stateElapsed time
                # longer than given time to answer so a no answer
                if (
                    trial_log_data["response_latency"]
                    >= float(self.meta.closedStimDuration) * 1000
                ):
                    trial_log_data["answer"] = 0
                else:
                    trial_log_data["answer"] = -1

            # no answer(only enters in pyvstim log)
            elif curr_trans == "nonanswer":
                trial_log_data["response_latency"] = row[self.column_keys["stateElapsed"]]
                trial_log_data["answer"] = 0
                trial_log_data["closedloop_end"] = row[self.column_keys["elapsed"]]

            # stim dissappear
            elif curr_trans == "stimendcorrect" or curr_trans == "stimendincorrect":
                trial_log_data["stim_end"] = row[self.column_keys["elapsed"]]

            # correction or trial end
            elif curr_trans == "correction" or curr_trans == "trialend":
                trial_log_data["trial_end"] = row[self.column_keys["elapsed"]]

                vstim_log = self.get_vstim_props()

                rig_logs = {}
                rig_logs["wheel"] = self.get_wheel_pos(trial_log_data["closedloop_start"])
                rig_logs["lick"] = self.get_licks()
                rig_logs["reward"] = self.get_reward()
                rig_logs["opto_pulse"] = self.get_opto()

                vstim_log["opto"] = vstim_log.get("opto", 0) or len(
                    rig_logs["opto_pulse"]
                )

                trial_log_data = {**trial_log_data, **vstim_log, **rig_logs}

                # right choice(CCW wheel rotation) is 1
                # left choice(CW wheel rotation) is -1
                # No go is 0
                trial_log_data["choice"] = int(
                    np.sign(trial_log_data["stim_side"]) * trial_log_data["answer"]
                )

                self.trial_data = trial_log_data

        return self.trial_data

    def calc_primary_metrics(self, row, **kwargs):
        """Calculates the primary metrics from trial_log data (mostly wheel trajectories)
        :param row : trial_row to be analyzed
        :type row  : dict
        :return    : dictionary with analyzed trial parameters
        :rtype     : dict
        """
        analyzed_row = {}
        temp_wheel = row["wheel"][:]
        temp_lick = row["lick"][:]

        early_t = kwargs.get("early_t", 500)
        reward_t = kwargs.get("reward_t", 2000)

        # Get indexes of different time onsets
        stim_onset_idx = find_nearest(temp_wheel[:, 0], row["openloopstart"])[0]
        early_onset_idx = find_nearest(temp_wheel[:, 0], row["openloopstart"] - early_t)[
            0
        ]
        closedloop_end_idx = find_nearest(temp_wheel[:, 0], row["closedloopdur"][1])[0]

        # Get intervals
        stim_wheel_interval = temp_wheel[stim_onset_idx:closedloop_end_idx, :]
        early_wheel_interval = temp_wheel[early_onset_idx:closedloop_end_idx, :]

        # Get delta_x and delta_t
        deltas_stim = np.diff(stim_wheel_interval, axis=0)
        deltas_early = np.diff(early_wheel_interval, axis=0)

        # (7) calculate the avg speed
        inst_speed = np.abs(np.divide(deltas_stim[:, 1], deltas_stim[:, 0]))
        analyzed_row["avg_speed"] = float(np.mean(inst_speed))

        # (5) calculate the biggest acceleration change
        # get center points for start and end of analysis(stim onset - 500ms to end of closedloop)
        early_onset_idx = find_nearest(temp_wheel[:, 0], row["openloopstart"] - 500)[0]
        closedloop_end_idx = find_nearest(temp_wheel[:, 0], row["closedloopdur"][1])[0]
        point = find_reaction_point(temp_wheel[:, 1], early_onset_idx, closedloop_end_idx)
        # reaction time is always relative to openloop start aka stim appearance
        analyzed_row["reaction_t"] = temp_wheel[point[0], 0] - row["openloopstart"]

        # inst_velocity = np.hstack((early_wheel_interval[1:,0].reshape(-1,1),np.divide(deltas_early[:,1],deltas_early[:,0]).reshape(-1,1)))
        # delta2s_early = np.diff(inst_velocity,axis=0)
        # inst_acc = np.hstack((early_wheel_interval[2:,0].reshape(-1,1),np.abs(np.divide(delta2s_early[:,1],delta2s_early[:,0])).reshape(-1,1)))
        # analyzed_row['acc_reaction_t'] = inst_acc[inst_acc[:,1].argmax(),0]

        # (4) calculate the path surplus
        target_choice_idx = point[0]
        path_wheel_interval = temp_wheel[target_choice_idx:closedloop_end_idx, :]
        if len(path_wheel_interval):
            path_real = np.sum(np.abs(np.diff(path_wheel_interval[:, 1], axis=0)), axis=0)
            path_ideal = np.abs(row["stim_side"]) - np.abs(path_wheel_interval[0, 1])
            analyzed_row["path_surplus"] = float((path_real / path_ideal) - 1)
        else:
            analyzed_row["path_surplus"] = np.nan

        # (6) calculate the average lick time relative to
        # If negative more anticipatory licks
        if len(temp_lick):
            analyzed_row["first_lick_t"] = temp_lick[0, 0]
            lick_t_diffs = np.subtract(
                temp_lick[stim_onset_idx:, 0], row["openloopstart"]
            )
            if len(lick_t_diffs):
                analyzed_row["avg_lick_t_diff"] = float(np.mean(lick_t_diffs))
            else:
                analyzed_row["avg_lick_t_diff"] = np.nan
        else:
            # if no lick data
            analyzed_row["first_lick_t"] = np.nan
            analyzed_row["avg_lick_t_diff"] = np.nan

        return analyzed_row

    def analyze_trial_data(self):
        """Trial data with analyzed parameters and relevant behavioral indexes"""
        cols_to_reset = [
            "closedloopdur",
            "openloopstart",
            "stimdur",
            "wheel",
            "lick",
            "trialdur",
            "reward",
        ]
        log_data = self.trial_data_from_logs()
        if len(log_data):
            row_analyzed = {}
            for row_log in log_data:
                # analyze here
                # if len(row_log['wheel']):
                #     row_analyzed = self.calc_primary_metrics(row_log)
                row_final = {**row_log, **row_analyzed}

                # reset the times in each trial so that they have their own time frames locked into stimulus start
                row_final = reset_times(row_final, cols_to_reset, anchor="openloopstart")

            self.trial_data.append(row_final)

    def save_array_to_db(self, array_name: str) -> None:
        """Saves array values as single entries in a database
        !! This might overload the database??"""

        if array_name not in self.trial_data.keys():
            raise ValueError(f"{array_name} is not a key in trial data dictionary")

        temp_arr = self.trial_data[array_name]
        for val in temp_arr:
            temp_dict = {"duinotime": val[0], array_name: val[1]}
            self.save_to_db(temp_dict, array_name)

    def save_trial_db(self, **kwargs):
        """Saves the trial into a database"""

        temp_dict = {}
        ignore = [
            "correct",
            "trial_no",
            "wheel",
            "stim_pos",
            "posx_l",
            "posx_r",
            "opto_pulse",
            "lick",
        ]

        for k, v in self.trial_data.items():
            if k == "reward":
                if len(v):
                    v = v[0][0]  # get only the time
                else:
                    v = -1
            temp_dict[k] = v

        db_dict = {k: v for k, v in temp_dict.items() if k not in ignore}
        self.save_to_db(db_dict, **kwargs)
