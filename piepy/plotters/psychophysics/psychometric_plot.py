import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from ..plotting_utils import make_subsets, set_style
from ..color import Color


def plot_psychometric(
    data: pl.DataFrame,
    ax: plt.Axes,
    mpl_kwargs: dict,
    xaxis_type: str = "linear_spaced",
    *kwargs,
) -> plt.Axes:
    """Plots the hit rates with 95% confidence intervals

    Parameters:
    data (pl.DataFrame) : run data
    ax (plt.axes) : An axes object to place to plot,default is None, which creates the axes
    xaxis_type (str): The type of xaxis, mainly adjusts spacing

    Returns:
    plt.axes: Axes object
    """

    def make_label(name: np.ndarray, count: np.ndarray) -> str:
        ret = """\nN=["""
        for i, n in enumerate(name):
            ret += rf"""{float(n)}:$\bf{count[i]}$, """
        ret = ret[:-2]  # remove final space and comma
        ret += """]"""
        return ret

    if ax is None:
        fig = plt.figure(figsize=kwargs.pop("figsize", (8, 8)))
        ax = fig.add_subplot(1, 1, 1)

    for filt_tup in make_subsets(q, ["stimkey", "stim_side"]):
        pass
