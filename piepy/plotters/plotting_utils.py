import os  # noqa: F401
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from datetime import datetime as dt

cm = 1 / 2.54
mplstyledict = {}

# styledict for putting in presentations
mplstyledict["presentation"] = {
    "figure.dpi": 300,
    "figure.edgecolor": "white",
    "figure.facecolor": "white",
    "figure.figsize": (12, 12),
    "figure.frameon": False,
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 2,
    "axes.titlesize": 30,
    "axes.titleweight": "bold",
    "axes.titlepad": 6,
    "axes.labelsize": 24,
    "axes.labelcolor": "black",
    "axes.axisbelow": True,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "ytick.major.size": 10,
    "xtick.major.size": 10,
    "ytick.minor.size": 7,
    "xtick.minor.size": 7,
    "xtick.major.width": 2,
    "ytick.major.width": 2,
    "xtick.minor.width": 1.5,
    "ytick.minor.width": 1.5,
    "xtick.color": "black",
    "ytick.color": "black",
    "grid.linewidth": 1.5,
    "grid.color": "black",
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
    "text.color": "black",
    "font.size": 24,
    "font.family": ["sans-serif"],
    "font.sans-serif": [
        "Helvetica",
        "Arial",
        "DejaVu Sans",
        "Bitstream Vera Sans",
        "Computer Modern Sans Serif",
        "Lucida Grande",
        "Verdana",
        "Geneva",
        "Lucid",
        "Avant Garde",
        "sans-serif",
    ],
    "lines.linewidth": 4,
    "lines.markersize": 24,
    "lines.markerfacecolor": "auto",
    "lines.markeredgecolor": "white",
    "lines.markeredgewidth": 2,
    "scatter.edgecolors": "face",
    "errorbar.capsize": 0,
    "image.interpolation": "none",
    "image.resample": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# dark background presentation
mplstyledict["presentation_dark"] = {
    "figure.dpi": 300,
    "figure.edgecolor": "#1E1E1E",
    "figure.facecolor": "#1E1E1E",
    "figure.figsize": (12, 12),
    "figure.frameon": False,
    "axes.facecolor": "#1E1E1E",
    "axes.edgecolor": "#F1F1F1",
    "axes.linewidth": 2,
    "axes.titlesize": 30,
    "axes.titleweight": "bold",
    "axes.titlepad": 6,
    "axes.labelsize": 24,
    "axes.labelcolor": "#F1F1F1",
    "axes.axisbelow": True,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "ytick.major.size": 10,
    "xtick.major.size": 10,
    "ytick.minor.size": 7,
    "xtick.minor.size": 7,
    "xtick.major.width": 2,
    "ytick.major.width": 2,
    "xtick.minor.width": 1.5,
    "ytick.minor.width": 1.5,
    "xtick.color": "#F1F1F1",
    "ytick.color": "#F1F1F1",
    "grid.linewidth": 1.5,
    "grid.color": "#F1F1F1",
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
    "text.color": "#F1F1F1",
    "font.size": 24,
    "font.family": ["sans-serif"],
    "font.sans-serif": [
        "Helvetica",
        "Arial",
        "DejaVu Sans",
        "Bitstream Vera Sans",
        "Computer Modern Sans Serif",
        "Lucida Grande",
        "Verdana",
        "Geneva",
        "Lucid",
        "Avant Garde",
        "sans-serif",
    ],
    "lines.linewidth": 4,
    "lines.markersize": 24,
    "lines.markerfacecolor": "auto",
    "lines.markeredgecolor": "#1E1E1E",
    "lines.markeredgewidth": 2,
    "scatter.edgecolors": "face",
    "errorbar.capsize": 0,
    "image.interpolation": "none",
    "image.resample": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# styledict for putting in word etc.
mplstyledict["print"] = {
    "figure.dpi": 300,
    "figure.edgecolor": "white",
    "figure.facecolor": "white",
    "figure.figsize": (8 * cm, 8 * cm),
    "figure.frameon": False,
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.titlepad": 6,
    "axes.labelsize": 14,
    "axes.labelcolor": "black",
    "axes.axisbelow": True,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "ytick.major.size": 8,
    "xtick.major.size": 8,
    "ytick.minor.size": 6,
    "xtick.minor.size": 6,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "xtick.color": "black",
    "ytick.color": "black",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.4,
    "grid.color": "black",
    "grid.linestyle": "--",
    "text.color": "black",
    "font.size": 15,
    "font.family": ["sans-serif"],
    "font.sans-serif": [
        "Helvetica",
        "Arial",
        "DejaVu Sans",
        "Bitstream Vera Sans",
        "Computer Modern Sans Serif",
        "Lucida Grande",
        "Verdana",
        "Geneva",
        "Lucid",
        "Avant Garde",
        "sans-serif",
    ],
    "lines.linewidth": 2,
    "lines.markersize": 10,
    "lines.markerfacecolor": "auto",
    "lines.markeredgecolor": "w",
    "lines.markeredgewidth": 1,
    "scatter.edgecolors": "face",
    "errorbar.capsize": 0,
    "image.interpolation": "none",
    "image.resample": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def make_label(name: np.ndarray, count: np.ndarray) -> str:
    ret = """\nN=["""
    for i, n in enumerate(name):
        ret += rf"""{float(n)}:$\bf{count[i]}$, """
    ret = ret[:-2]  # remove final space and comma
    ret += """]"""
    return ret

def make_linear_axis(data: pl.DataFrame, column_name: str, mid_value: float = 0) -> dict:
    """Returns a dictionary where keys are contrast values and values are linearly seperated locations in the axis"""
    if column_name not in data.columns:
        raise ValueError(f"{column_name} is not a valid column")

    dep_lst = data[column_name].unique().drop_nulls().sort().to_list()
    pos = np.arange(1, len([c for c in dep_lst if c > mid_value]) + 1)
    neg = np.arange(1, len([c for c in dep_lst if c < mid_value]) + 1)[::-1] * -1
    if 0 in dep_lst:
        ax_list = neg.tolist() + [0] + pos.tolist()
    else:
        ax_list = neg.tolist() + pos.tolist()

    return {k: v for k, v in zip(dep_lst, ax_list)}


def set_style(styledict: str = "presentation") -> None:
    """Sets the styleof"""
    if styledict in ["presentation", "print", "presentation_dark"]:
        plt.style.use(mplstyledict[styledict])
    else:
        try:
            plt.style.use(styledict)
        except KeyError:
            plt.style.use("default")


def dates_to_deltadays(date_arr: list, start_date=dt.date):
    """Converts the date to days from first start"""
    date_diff = [(day - start_date).days for day in date_arr]
    return date_diff
