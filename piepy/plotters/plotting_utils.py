import os  # noqa: F401
import copy
import inspect
import functools
import numpy as np
import polars as pl
import matplotlib.axes
import matplotlib.pyplot as plt
from collections.abc import Iterable
from matplotlib.artist import ArtistInspector
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


def make_named_axes(data:pl.DataFrame,col_names:list[str],**kwargs) -> tuple[plt.Figure,dict]:
    """ Generates a figure and a dictinary to acess the axes with combined unique values of column names as keys
    The col names will be used for row and column generation, respectively 
    
    Args:
        data:
        col_names:
    """
    assert len(col_names) <=2, "Can't have more than 2 column names(row and column of axes)"
    
    uniq_rows = data[col_names[0]].drop_nulls().unique().sort().to_list()
    
    if len(col_names)==2:
        uniq_cols = data[col_names[1]].drop_nulls().unique().sort().to_list()
    else:
        uniq_cols = ['']
        
    f,axes = plt.subplots(nrows=len(uniq_rows),
                          ncols=len(uniq_cols),
                          **kwargs)
    axes = axes.reshape(len(uniq_rows),len(uniq_cols))

    axes_dict = {}
    for i,row in enumerate(uniq_rows):
        for j,col in enumerate(uniq_cols):
            key = f"{row}_{col}"
            axes_dict[key] = axes[i,j]

    return f,axes_dict


def get_valid_mpl_kwargs(plot_type:str, mpl_kwargs:dict) -> dict:
    """ Returns the subset of values that are valid for the given plot type 
    
    Args:
        plot_type: Type of the plotting function (e.g. "plot", "scatter")
        mpl_kwargs: dictionary with all the matplotlib keeyword arguments
    """
    # sometimes matplotlib returns a tuple/list of objects; just get the first
    def get_root(obj):
        if isinstance(obj, Iterable):
            return obj[0]
        return obj
        
    dummy_f,dummy_ax = plt.subplots(1,1)
    func = getattr(matplotlib.axes.Axes, plot_type)  # Get the function dynamically (e.g., plt.plot, plt.scatter, etc.)
    
    dummy_ax.remove()
    dummy_f.clear()
    plt.close(dummy_f)
    
    sig = inspect.signature(func)
    
    valid_kwds = ArtistInspector(get_root(func(dummy_ax,[0], [0]))).get_setters() + list(sig.parameters.keys())
    # Filter mpl_kwargs to only include valid parameters
    return {k: v for k, v in mpl_kwargs.items() if k in valid_kwds}


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


def override_plots(methods_to_override:list[str]|None=None) -> None:
    # wrapping matplotlib functions to allow 
    if methods_to_override is None:
        methods_to_override = ["plot","errorbar","scatter","bar","step","fill_between"]
    
    original_methods = {}
    for meth in methods_to_override:
        original_methods[meth] = getattr(matplotlib.axes.Axes, meth)

        def create_custom_method(name,method):
            @functools.wraps(original_methods[meth])
            def custom_method(self,*args,**kwargs):
                valid_kwargs = get_valid_mpl_kwargs(plot_type=name,
                                                    mpl_kwargs=kwargs.get("mpl_kwargs",{}))
                good_kwargs = copy.deepcopy({**kwargs,**valid_kwargs})
                good_kwargs.pop("mpl_kwargs",{})
                
                return method(self,*args,**good_kwargs)
            return custom_method
        
        cust_meth = create_custom_method(meth,original_methods[meth])
        setattr(matplotlib.axes.Axes, f"_{meth}", cust_meth)
