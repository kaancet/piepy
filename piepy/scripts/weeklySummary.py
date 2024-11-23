import os
import glob
from datetime import datetime as dt
from datetime import timedelta as dt_delta
from dataclasses import dataclass, field

from ..detection.wheelDetectionBehavior import WheelDetectionBehavior
from ..detection.wheelDetectionSession import WheelDetectionSession
from ..wheel.wheelBehavior import WheelBehavior
from ..wheel.wheelSession import WheelSession
from ..core.utils import *

weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


@dataclass
class TrainingWeek:
    animalid: str
    task_type: str
    day_end: str
    day_depth: int = field(default=5)
    day_range: list = field(default_factory=lambda: [])
    day_range_as_dt: list = field(default_factory=lambda: [])
    day_range_nice_str: list = field(default_factory=lambda: [])
    days_of_week: list = field(default_factory=lambda: [])
    session_list: list = field(default_factory=lambda: [])
    week_data: dict = field(default_factory=lambda: {})

    def __post_init__(self):
        self.init_week()

    def init_week(self) -> None:
        """Initializes the weekdays and session list for those days"""
        try:
            day_end_dt = dt.strptime(self.day_end, "%y%m%d").date()
        except:
            raise ValueError(
                f"Date string {self.day_end} not recognized, use YYMMDD, e.g. 220918"
            )

        # get the weekdays as dt
        d = 0
        while len(self.session_list) < self.day_depth:
            day = day_end_dt - dt_delta(d)

            sesh = self.get_session(dt.strftime(day, "%y%m%d"))
            if len(sesh):
                display(f"Adding session {sesh[0]}")
                self.day_range_as_dt.insert(0, day)
                self.day_range.insert(0, dt.strftime(day, "%y%m%d"))
                self.day_range_nice_str.insert(0, dt.strftime(day, "%d %b %y"))

                self.session_list.insert(0, sesh[0])
            d += 1

        self.days_of_week = [weekdays[d.weekday()] for d in self.day_range_as_dt]

    def get_session(self, date: str) -> list:
        """Checks and returns name if a session exists"""
        sesh = glob.glob(f"J:\\analysis\\{date}_{self.animalid}_{self.task_type}*")
        if not len(sesh):
            display(f"No analyzed session, looking at training folder")
            sesh = glob.glob(f"J:\\training\\{date}_{self.animalid}_{self.task_type}*")
            if not len(sesh):
                display(
                    f"No session in training directory, looking at presentation folder"
                )
                sesh = glob.glob(
                    f"J:\\presentation\\{date}_{self.animalid}_{self.task_type}*"
                )

        return sesh

    def get_week_data(self, do_load: bool = True) -> dict:
        """Reads the data from the last x sessions and returns a dict containing individual dates(dict) and pooled data(pd.dataframe)"""
        self.week_data["week_pooled"] = {}
        for i, sesh in enumerate(self.session_list):
            exp_name = sesh.split(os.sep)[-1]
            if self.task_type == "detect":
                temp = WheelDetectionSession(exp_name, load_flag=do_load)
            elif self.task_type == "wheel":
                temp = WheelSession(exp_name, load_flag=do_load)

            self.week_data[self.day_range[i]] = temp

            for k, v in temp.data.stim_data.items():
                if k not in self.week_data["week_pooled"].keys():
                    self.week_data["week_pooled"][k] = v
                else:
                    self.week_data["week_pooled"][k] = pd.concat(
                        [self.week_data["week_pooled"][k], v]
                    )

    def __plot_detect__(self):
        """Plots the week summary for detection task"""
        pass


@dataclass
class TrainingDay:
    day: str = field(default=None)
    day_as_dt: dt.date = field(default=None)
    day_nice_str: str = field(default=None)
    day_of_week: str = field(default=None)

    def __post_init__(self):
        self.init_day()

    def init_day(self):
        if self.day is None:
            self.day_as_dt = dt.today()
            self.day = dt.strftime(self.day_as_dt, "%y%m%d")
        else:
            try:
                self.day_as_dt = dt.strptime(self.day, "%y%m%d")
            except:
                raise ValueError(
                    "Date string not recognized, format should be YYMMDD, e.g. 210926"
                )

        self.day_nice_str = dt.strftime(self.day_as_dt, "%d %b %y")
        self.day_of_week = weekdays[self.day_as_dt.weekday()]


def get_week_data(animalid: str, task: str):
    """Gets the week data of a given animal on a given task"""

    # get the wheel experiments/trainings done today
    config = getConfig()
    presentation_folder = config["presentationPath"]
    training_folder = config["trainingPath"]
    today_experiments = glob.glob(
        "{0}/{1}*__no_cam_*/".format(presentation_folder, today.day)
    )
    today_trainings = glob.glob("{0}/{1}*__no_cam_*/".format(training_folder, today.day))


def weeklySummary():

    # if end of the week do weekly summary (pooled psychometric and behavior analysis)
    if day_of_week in ["Friday", "Saturday", "Sunday"]:
        display("It's {0}! Doing end of week analysis".format(day_of_week))
        week_start = dt.strftime(day_as_dt - delta(day_as_dt.weekday()), "%y%m%d")
        week_end = day_str

        week_analysed_behave = {}
        for i, animalid in enumerate(analysed_wheel.keys()):
            week_analysed_behave[animalid] = WheelBehavior(
                animalid,
                dateinterval=[week_start, week_end],
                load_behave=True,
                load_data=True,
            )

        plot_cnt = len(to_plot)
        fig = plt.figure(figsize=(20, 30))
        for i, animalid in enumerate(to_plot):
            # g = week_analysed_behave[animalid].plot('behaviorSummary')
            if plot_cnt <= 3:
                ax_in1 = fig.add_subplot(1 * 3, plot_cnt, i + 1)
                ax_in2 = fig.add_subplot(1 * 3, plot_cnt, i + 4, sharex=ax_in1)
                ax_in3 = fig.add_subplot(1 * 3, plot_cnt, i + 7, sharex=ax_in2)
            elif plot_cnt > 3 and plot_cnt <= 6:
                ax_in1 = fig_add_subplot(2 * 3, 3, i + 1)
                ax_in2 = fig_add_subplot(2 * 3, 3, i + 4, sharex=ax_in1)
                ax_in3 = fig_add_subplot(2 * 3, 3, i + 7, sharex=ax_in2)
            else:
                ax_in1 = fig.add_subplot(3 * 3, 3, i + 1)
                ax_in2 = fig.add_subplot(3 * 3, 3, i + 4, sharex=ax_in1)
                ax_in3 = fig.add_subplot(3 * 3, 3, i + 7, sharex=ax_in2)

            week_analysed_behave[animalid].plot("performance", ax=ax_in1)
            week_analysed_behave[animalid].plot("weight", ax=ax_in2)
            week_analysed_behave[animalid].plot("trialDistributions", ax=ax_in3)

            ax_in1.set_title(animalid, fontsize=18)
        fig.tight_layout()

        display("Saving weeklySummary plot")
        fig.savefig(
            "J:\\data\\analysis\\behavior_results\\python_figures\\weeklyPlots\\{0}_summary.pdf".format(
                day_str
            ),
            dpi=100,
            bbox_inches="tight",
        )
