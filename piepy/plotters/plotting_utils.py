import os  # noqa: F401
import copy
import inspect
import functools
import numpy as np
import polars as pl
import matplotlib.axes
from typing import Literal
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
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
    "ytick.major.size": 5,
    "xtick.major.size": 5,
    "ytick.minor.size": 3,
    "xtick.minor.size": 3,
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
        data: DataFrame to get the unique values from
        col_names: Names of the columns
        
    Returns:
        tuple[plt.Figure,dict]: figure that contains the axes, dictionary of named axes
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
        
    Returns:
        dict: Valid kwargs for given plot type
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
    """ Makes a label given an array of names and counts
    
    Args:
        name: ...
        count: ...
        
    Returns:
        str: string to be used as a label
    """
    ret = """\nN=["""
    for i, n in enumerate(name):
        ret += rf"""{float(n)}:$\bf{count[i]}$, """
    ret = ret[:-2]  # remove final space and comma
    ret += """]"""
    return ret


def make_linear_axis(data: pl.DataFrame, column_name: str, mid_value: float = 0) -> dict:
    """ Returns a dictionary where keys are contrast values and values are linearly seperated locations in the axis

    Args:
        data (pl.DataFrame): Session or multi session data
        column_name (str): name of the column to use making the linear axis
        mid_value (float, optional): Middle value. Defaults to 0.

    Raises:
        ValueError: If column name doesn't exist

    Returns:
        dict: a dictionary with values in the "column" as keys and new linear values as values
    """
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


def make_dot_cloud(
    y_points: ArrayLike,
    center: float = 0,
    bin_width: float = 50,  # ms
    width: float = 0.5,
) -> np.ndarray:
    """Turns the data points into a cloud by dispersing them horizontally depending on their distribution,
    The more points in a bin, the wider the dispersion
    Returns x-coordinates of the dispersed points

    Args:
        y_points (ArrayLike): values that will be dispersed in a cloud
        center (float, optional): Center value that the cloud will be located at. Defaults to 0.
        bin_width (float, optional): width of bins in ms. Defaults to 50.

    Returns:
        np.ndarray: new dispersed x_points corresponding to y_points
    """

    if not isinstance(y_points,np.ndarray):
        y_points = np.array(y_points)
    
    if len(y_points) > 1:
        bin_edges = np.arange(
            np.nanmin(y_points), np.nanmax(y_points) + bin_width, bin_width
        )

        # Get upper bounds of bins
        counts, bin_edges = np.histogram(y_points, bins=bin_edges)
        if not len(counts):
            counts = np.array([0])
        
        # get the indices that correspond to points inside the bin edges
        idx_in_bin = []
        for ymin, ymax in zip(bin_edges[:-1], bin_edges[1:]):
            i = np.nonzero((y_points >= ymin) * (y_points < ymax))[0]
            idx_in_bin.append(i)

        x_coords = np.zeros(len(y_points))
        b = np.nanmax(counts)//2
        dx = b and width / b or 0

        for i in idx_in_bin:
            _points = y_points[i]  # value of points that fall into the bin
            # if less then 2, leave untouched, will put it in the mid line
            if len(i) > 1:
                j = len(i) % 2
                i = i[np.argsort(_points)]
                # if even numbers of points, j will be 0, which will allocate the points equally to left and right
                # if odd, j will be 1, then, below lines will leave idx 0 at the midline and start from idx 1
                a = i[j::2]
                b = i[j + 1 :: 2]
                x_coords[a] = (0.5 + j / 3 + np.arange(len(a))) * dx
                x_coords[b] = (0.5 + j / 3 + np.arange(len(b))) * -dx
    else:
        x_coords = np.array([0])

    return x_coords + center


def set_style(styledict:Literal["presentation","print"]) -> None:
    """ Sets the styling of the plot

    Args:
        styledict (Literal["presentation","print","presentation_dark"]): key to custom matplotlib styledicts(above)
    """
    if styledict in ["presentation", "print", "presentation_dark"]:
        plt.style.use(mplstyledict[styledict])
    else:
        try:
            plt.style.use(styledict)
        except KeyError:
            plt.style.use("default")


def dates_to_deltadays(date_arr: list[dt.date], start_date:dt.date) -> list:
    """Converts the date to days from first start

    Args:
        date_arr (list[dt.date]): Array of dates
        start_date (dt.date): Start date (day 0).

    Returns:
        list: List of delta days from start date
    """
    date_diff = [(day - start_date).days for day in date_arr]
    return date_diff


def pval_plotter(
    ax:plt.Axes,
    p_val:float,
    pos:list[float,float],
    loc:float,
    tail_height:float=0.05,
    **kwargs) -> plt.Axes:
    """ Annotates the p-val between two locations
    
    Args:
        ax (plt.Axes): axes object to draw the annotation on
        p_val (float): the text to be written
        pos (list[float,float]): the position of the stars in the independent axis
        loc (float): location of the stars in the dependent axis
        tail_height (float, optional): height of annotation line tails as a proprtion of loc. Defaults to 0.05.

    Returns:
        plt.Axes: Axes object the stars were plotted to
    """
    x1,x2 = pos
    h = loc*tail_height
    
    stars = "ns"
    if p_val < 0.0001:
        stars = "****"
    elif 0.0001 <= p_val < 0.001:
        stars = "***"
    elif 0.001 <= p_val < 0.01:
        stars = "**"
    elif 0.01 <= p_val < 0.05:
        stars = "*"
    if stars != "ns":
        ax.plot([x1, x1, x2, x2], [loc, loc+h, loc+h, loc], lw=1, c=kwargs.get("color","k"))
        ax.text((x1+x2)*.5, loc+h, stars, ha='center', va='center', color=kwargs.get("color","k"))
    return ax


def override_plots(methods_to_override:list[str]|None=None) -> None:
    """Overrides matplotlib.axes plots with "_" prepended to the name,
    Currently the overriding function checks and filters the valid kwargs for the overriden plot

    Args:
        methods_to_override (list[str] | None, optional): List of matplotlib plots to override. Defaults to None.
    """
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
