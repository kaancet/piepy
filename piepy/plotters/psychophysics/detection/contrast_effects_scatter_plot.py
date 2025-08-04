import numpy as np
import polars as pl
from typing import Literal
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import wilcoxon, linregress, pearsonr  # noqa: F401
from scipy.optimize import curve_fit  # noqa: F401

from ....core.data_functions import make_subsets
from ...plotting_utils import set_style, override_plots
from ...colors.color import Color
from ....psychophysics.wheel.detection.wheelDetectionGroupedAggregator import (
    WheelDetectionGroupedAggregator,
)


ANIMAL_COLORS = {
    "KC139": "#332288",
    "KC141": "#117733",
    "KC142": "#DDCC77",
    "KC143": "#AA4499",
    "KC144": "#882255",
    "KC145": "#88CCEE",
    "KC146": "#275D6D",
    "KC147": "#F57A6C",
    "KC148": "#ADFA9A",
    "KC149": "#A45414",
    "KC150": "#0000FF",
    "KC151": "#00FF11",
    "KC152": "#FFAA33",
}


X_AXIS_LABEL = {-1: "Non\nOpto", 0: "Opto", 1: "Opto\nOff"}

AREA_MARKERS = {"V1": "o", "HVA": "s", "dorsal": "d", "ventralPM": "P"}


def m_line(x, m, b):
    return m * x + b


@np.vectorize
def lenient_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """returns the numerator if denominator is zero, otherwise does regular division

    Args:
        num (Number | np.ndarray): Numerator
        den (Number | np.ndarray): Denominator

    Returns:
        Number | np.ndarray: Division result
    """
    return float(den and num / den or num)


def plot_contrast_effect_scatter_plot(
    data: pl.DataFrame,
    effect_on: Literal["hit_rate", "reaction", "response"],
    effect_metric: Literal["delta", "BSI", "BSI_base"] = "delta",
    ax: plt.Axes | None = None,
    include_misses: bool = False,
    polarity: Literal[-1, 1] = 1,
    mpl_kwargs: dict | None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Plots a correlation scatter plot of the given "effect_on" readouts of two contrast values,
    seperates the stimulus types

    Args:
        data (pl.DataFrame): Experimental data
        effect_on (Literal["hit_rate","reaction","response"]): The behavioral readout to plot.
        ax (plt.Axes | None, optional): Precreated axis to plot to. Defaults to None.
        include_misses (bool, optional): Whether or not to include miss trials. Defaults to False. Doesn't make sense if effect_on is "hit_rate".
        polarity (Literal[-1,1], optional): Polarity of subtraction. 1 is nonopto - opto, -1 is opto - nonopto. Defaults to 1.
        mpl_kwargs (dict | None, optional): kwargs for styling matplotlib plots. Defaults to None.

    Returns:
        tuple[plt.Figure, plt.Axes]: Figure and axes of plot
    """
    if effect_on != "hit_rate":
        if not include_misses:
            effect_on = f"hit_{effect_on}"
        effect_on = f"median_{effect_on}_times"
    polarity = float(polarity)
    clr = Color(task="detection")
    override_plots()
    set_style(kwargs.get("style", "presentation"))
    if mpl_kwargs is None:
        mpl_kwargs = {}

    if ax is None:
        fig = plt.figure(figsize=mpl_kwargs.pop("figsize", (8, 8)))
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = ax.get_figure()

    aggregator = WheelDetectionGroupedAggregator()
    aggregator.set_data(data=data)
    aggregator.group_data(
        group_by=[
            "animalid",
            "session_id",
            "area",
            "stim_type",
            "stim_side",
            "contrast",
            "opto_pattern",
        ]
    )
    aggregator.calculate_hit_rates()
    aggregator.calculate_opto_pvalues()

    plot_data = aggregator.grouped_data.drop_nulls("contrast").filter(
        pl.col("stim_side") != "ipsi"
    )

    diffs_values = np.zeros(
        (
            plot_data["session_id"].n_unique(),
            plot_data["contrast"].n_unique(),
            plot_data["stim_type"].n_unique(),
        )
    )
    diffs_values[:] = np.nan
    areas = np.empty((plot_data["session_id"].n_unique()), dtype=object)
    for filt_tup in make_subsets(plot_data, ["session_id"], start_enumerate=0):
        filt_df = filt_tup[-1]
        if not filt_df.is_empty():
            q = (
                filt_df.group_by(["stim_type", "opto_pattern"])
                .agg(pl.col("contrast"), pl.col(effect_on))
                .sort(["stim_type", "opto_pattern"])
            )

            for stim_tup in make_subsets(q, ["stim_type"], start_enumerate=0):
                stim_df = stim_tup[-1]
                c_nonopto = stim_df[0, "contrast"].to_numpy()
                c_opto = stim_df[1, "contrast"].to_numpy()
                e1 = np.zeros_like(c_nonopto)
                e1[:] = np.nan
                e2 = np.copy(e1)
                paired_idx = [i for i, c in enumerate(c_nonopto) if c in c_opto]

                e1[paired_idx] = stim_df[0, effect_on].to_numpy()[paired_idx]
                e2[paired_idx] = stim_df[1, effect_on].to_numpy()
                if effect_on == "hit_rate":
                    if effect_metric == "delta":
                        # simple delta
                        temp = e1 - e2
                    elif effect_metric == "BSI":
                        # BSI: (nonopto - opto) / nonopto
                        # what percentage is delta hit rates of nonopto
                        _top = e1 - e2
                        temp = lenient_div(_top, e1)
                    elif effect_metric == "BSI_base":
                        pass
                        # BSI_base: (|nonopto - baseline| - |opto-baseline|) / (|nonopto - baseline|)
                        # with baseline normalization
                        # _opto = np.abs(e2 - session_baseline_hr)
                        # _nonopto = np.abs(e1 - session_baseline_hr)
                        # _top = _nonopto - _opto
                        # temp = lenient_div(_top, _nonopto)
                elif "time" in effect_on:
                    temp = e2 - e1

                diffs = temp
                diffs = polarity * diffs

                diffs_values[filt_tup[0], :, stim_tup[0]] = diffs
            areas[filt_tup[0]] = filt_df[0, "area"]
        else:
            print("soknfdksndf")

    s_type = ["0.04cpd_8.0Hz", "0.16cpd_0.5Hz"]

    for s in range(diffs_values.shape[2]):
        v = diffs_values[:, 1:, s]  # 0th column is contrast 0
        if effect_on == "hit_rate":
            v *= 100

        for a in range(v.shape[0]):
            _marker = AREA_MARKERS[areas[a]]
            ax.scatter(
                v[a, 0],
                v[a, 1],
                c=clr.stim_keys[f"{s_type[s]}_-1"]["color"],
                marker=_marker,
            )

        x = v[:, 0]
        y = v[:, 1]
        x_nan = np.isnan(x)
        y_nan = np.isnan(y)

        y = y[~np.logical_or(x_nan, y_nan)]
        x = x[~np.logical_or(x_nan, y_nan)]

        res = pearsonr(x, y)
        # x = x[:,np.newaxis]
        # m, _, _, _ = np.linalg.lstsq(x, y)
        # y_pred = m * x

        # Compute R²
        # SS_res = np.sum((y - y_pred) ** 2)  # Residual sum of squares
        # SS_tot = np.sum((y - np.mean(y)) ** 2)  # Total sum of squares
        # r2 = 1 - (SS_res / SS_tot)

        # ax.plot(np.arange(0,600,100), m_line(np.arange(0,600,100), *popt))
        ax.text(
            0,
            100 * s,
            f"p_r={res.statistic:.4f}",
            c=clr.stim_keys[f"{s_type[s]}_-1"]["color"],
        )

    ax.plot([0, 100], [0, 100], linestyle="--", color="k", alpha=0.5)
    if "time" in effect_on:
        _y_ = np.arange(-400, 800, 200)
        y_ticks = [_t * -1 * polarity for _t in _y_]
    else:
        _y_ = [0, 25, 50, 75, 100]
        y_ticks = [_t * polarity for _t in _y_]

    # ax.set_ylim([min(y_ticks) - 10, max(y_ticks) + 10])
    # ax.set_xlim([min(y_ticks) - 10, max(y_ticks) + 10])
    ax.set_yticks(y_ticks)
    ax.set_xticks(y_ticks)
    ax.set_ylabel(r"$\Delta$ Hit Rate 50 (%)")
    ax.set_xlabel(r"$\Delta$ Hit Rate 12.5 (%)")

    return fig, ax
