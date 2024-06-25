import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime as dt
from datetime import timedelta as delta
from dataclasses import dataclass, field

from .wheel.wheelSession import *
from ...utils import display, parsePref

weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


@dataclass
class parsedDay:
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


def dailySummary(day: str = None, to_plot: list = None) -> None:
    """Analyses all the sessions in a given day"""

    today = parsedDay(day)

    display(
        "Producing daily summaries {0} {1}".format(today.day_nice_str, today.day_of_week)
    )

    # get the wheel experiments/trainings done today
    config = getConfig()
    presentation_folder = config["presentationPath"]
    training_folder = config["trainingPath"]
    today_experiments = glob.glob(
        "{0}/{1}*__no_cam_*/".format(presentation_folder, today.day)
    )
    today_trainings = glob.glob("{0}/{1}*__no_cam_*/".format(training_folder, today.day))

    today_sessions = today_experiments + today_trainings

    wheel_experiments = []
    for exp in today_sessions:
        for _, _, contents in os.walk(exp):
            for f in contents:
                if "level" in f or "opto" in f:
                    if exp not in wheel_experiments:
                        wheel_experiments.append(exp)

    if len(wheel_experiments) == 0:
        display("No sessions!")
        return 0
    else:
        display("Session List:")
        print(wheel_experiments)

    # analyse the experiments/trainings
    analysed_wheel = {}
    for wheel_exp in wheel_experiments:
        exp_name = wheel_exp.split("\\")[-2]
        temp_w = WheelSession(exp_name, load_flag=True)
        analysed_wheel[temp_w.session["meta"]["animalid"]] = temp_w

    plot_types = ["performance", "responsePerStim", "psychometric"]
    if to_plot is None:
        to_plot = list(analysed_wheel.keys())

    # giving 10 length units per row
    row_length = 5
    col_width = 8
    fig = plt.figure(
        figsize=(int(col_width * len(plot_types)), int(row_length * len(to_plot)))
    )

    for row, animalid in enumerate(to_plot):
        # columns for different plot panels
        for col, pt in enumerate(plot_types):
            ax = fig.add_subplot(
                len(to_plot), len(plot_types), int(col + row * (len(plot_types))) + 1
            )
            analysed_wheel[animalid].plot(pt, ax=ax, savefig=False, notitle=True)

            if col == 0:
                ax.annotate(
                    animalid,
                    xy=(0, 0.5),
                    xytext=(0.1, 0),
                    xycoords=("subfigure fraction", "axes fraction"),
                    textcoords="offset points",
                    rotation="vertical",
                    fontweight="bold",
                    size=20,
                    ha="left",
                    va="center",
                )
        # handles,_ = ax.get_legend_handles_labels()
        # patch = mpatches.Patch(color='grey', label="answered_trial_count = {0}\nperformance = {1}%".format(analysed_wheel[animalid].session['summaries']['overall']['answered_trials'],analysed_wheel[animalid].session['summaries']['overall']['answered_correct_pct']))
        # handles.append(patch)
    fig.tight_layout()

    display("Saving dailySummary plot")
    fig.savefig(
        "J:\\data\\analysis\\behavior_results\\python_figures\\dailyPlots\\{0}_summary.pdf".format(
            today.day
        ),
        dpi=100,
        bbox_inches="tight",
    )


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Wheel Behavior Daily Summary Script")

    parser.add_argument(
        "day", metavar="summaryday", type=str, help="Day to be summarized (e.g. 191124)"
    )

    opts = parser.parse_args()
    day = opts.day
    dailySummary(day)


if __name__ == "__main__":
    main()
