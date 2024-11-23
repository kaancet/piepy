import os
import json
import numpy as np
import pandas as pd
from cycler import cycler
import matplotlib.pyplot as plt
from datetime import timedelta
from datetime import datetime as dt
from collections import defaultdict
from matplotlib import colors as mcolors
from piepy.core.io import display
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ..core.config import config as cfg
import colorsys
import warnings

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


def set_style(styledict="presentation"):
    if styledict in ["presentation", "print", "presentation_dark"]:
        plt.style.use(mplstyledict[styledict])
    else:
        try:
            plt.style.use(styledict)
        except KeyError:
            plt.style.use("default")
            display(f"Matplotlib {styledict} style is nonexistant, using default style")
    display(f"Changed plotting style to {styledict}")


def dates_to_deltadays(date_arr: list, start_date=dt.date):
    """Converts the date to days from first start"""
    date_diff = [(day - start_date).days for day in date_arr]
    return date_diff


class Color:
    __slots__ = ["colorkey_path", "stim_keys", "contrast_keys", "outcome_keys"]

    def __init__(self):
        # this is hardcoded for now, fix this
        self.colorkey_path = cfg.paths["colors"][0]
        self.read_colors()

    def read_colors(self) -> None:
        """Reads the colorkey.json and returns a dict of color keys for different sftf and contrast values"""
        with open(self.colorkey_path, "r") as f:
            keys = json.load(f)
            self.stim_keys = keys["spatiotemporal"]
            self.contrast_keys = keys["contrast"]
            self.outcome_keys = keys["outcome"]

    def check_stim_colors(self, keys):
        """Checks if the stim key has a corresponding color value, if not adds a randomly selected color the key"""
        new_colors = {}
        for k in keys:
            if k not in self.stim_keys:
                print(f"Stim key {k} not present in colors, generating random color...")
                colors = {**mcolors.BASE_COLORS, **mcolors.CSS4_COLORS}
                new_colors[k] = {
                    "color": np.random.choice(list(colors.keys()), replace=False, size=1)[
                        0
                    ],
                    "linestyle": np.random.choice(
                        [":", "--", "-.", "-"], replace=False, size=1
                    )[0],
                }
        if len(new_colors):
            self.stim_keys = {**self.stim_keys, **new_colors}
        else:
            print("Stimulus colors checkout!!")

    def check_contrast_colors(self, contrasts):
        """Checks if the contrast key has a corresponding color value, if not adds a randomly selected color the key"""
        new_colors = {}
        for c in contrasts:
            str_key = str(c)
            if str_key not in self.contrast_keys:
                print(
                    f"Contrast key {str_key} not present in colors, generating random color..."
                )
                colors = {**mcolors.BASE_COLORS, **mcolors.CSS4_COLORS}
                new_colors[str_key] = {
                    "color": np.random.choice(list(colors.keys()), replace=False, size=1)[
                        0
                    ]
                }
        if len(new_colors):
            self.contrast_keys = {**self.contrast_keys, **new_colors}
        else:
            print("Contrast colors checkout!!")

    @staticmethod
    def name2hsv(color_name: str) -> tuple:
        rgb = mcolors.to_rgb(color_name)
        return Color.rgb2hsv(
            rgb, normalize=False
        )  # no need to normalize here, already 0-1 range from mcolors method

    @staticmethod
    def hex2rgb(hex_code):
        hex_code = hex_code.lstrip("#")
        return tuple(int(hex_code[i : i + 2], 16) for i in (0, 2, 4))

    @staticmethod
    def rgb2hex(rgb_tuple):
        def clamp(x):
            return max(0, min(x, 255))

        rgb_tuple = tuple(int(clamp(i) * 255) for i in rgb_tuple)
        r, g, b = rgb_tuple
        return f"#{r:02x}{g:02x}{b:02x}"

    @staticmethod
    def normalize_rgb(rgb_tuple):
        return tuple(i / 255.0 for i in rgb_tuple)

    @staticmethod
    def rgb2hsv(rgb_tuple, normalize: bool = True):
        if normalize:
            rgb_tuple = Color.normalize_rgb(rgb_tuple)
        r, g, b = rgb_tuple
        return colorsys.rgb_to_hsv(r, g, b)

    @staticmethod
    def hsv2rgb(hsv_tuple):
        h, s, v = hsv_tuple
        return colorsys.hsv_to_rgb(h, s, v)

    @staticmethod
    def lighten(hsv_tuple, l_coeff: float = 0.33):
        """Lightens the hsv_tuple by l_coeff percent, aka from S subtracts l_coeff percent of the S value"""
        if not l_coeff <= 1 and l_coeff >= 0:
            raise ValueError(
                f"The l_coeff value needs to be 0<=l_coeff<= 1, got {l_coeff} instead"
            )
        h, s, v = hsv_tuple
        s_new = s - (s * l_coeff)
        return Color.rgb2hex(Color.hsv2rgb((h, s_new, v)))

    @staticmethod
    def make_color_range(
        start_color: str, rng: int, s_limit: list = [20, 100], v_limit: list = [20, 100]
    ) -> list:
        """Returns a list of hex colors ranging from start color to specific limit values"""
        rgb = Color.hex2rgb(start_color)
        hsv = Color.rgb2hsv(rgb)
        h, s, v = hsv

        # limit the saturation and value
        # s= 20%-100% v=100%

        s_steps = np.linspace(s, 1, rng)
        v_steps = np.linspace(v, 1, rng)
        v_steps = v_steps[::-1]  # reverse values to go from light to darker color

        color_range = []
        for i, s in enumerate(s_steps):

            new_hsv = (h, s_steps[i], v_steps[i])
            new_rgb = Color.hsv2rgb(new_hsv)
            color_range.append(Color.rgb2hex(new_rgb))

        return color_range

    @staticmethod
    def gen_color(cmap, n, reverse=False):
        """Generates n distinct color from a given colormap. From https://github.com/binodbhttr/mycolorpy/tree/master

        Args:
            cmap(str): The name of the colormap you want to use.
                Refer https://matplotlib.org/stable/tutorials/colors/colormaps.html to choose
                Suggestions:
                For Metallicity in Astrophysics: Use coolwarm, bwr, seismic in reverse
                For distinct objects: Use gnuplot, brg, jet,turbo.

            n(int): Number of colors you want from the cmap you entered.

            reverse(bool): False by default. Set it to True if you want the cmap result to be reversed.

        Returns:
            colorlist(list): A list with hex values of colors.
        """
        c_map = plt.cm.get_cmap(str(cmap))  # select the desired cmap
        arr = np.linspace(0, 1, n)  # create a list with numbers from 0 to 1 with n items
        colorlist = []
        for c in arr:
            rgba = c_map(
                c
            )  # select the rgba value of the cmap at point c which is a number between 0 to 1
            clr = mcolors.rgb2hex(rgba)  # convert to hex
            colorlist.append(str(clr))  # create a list of these colors

        if reverse == True:
            colorlist.reverse()
        return colorlist

    @staticmethod
    def gen_color_normalized(
        cmap,
        data_arr,
        reverse: bool = False,
        vmin=0,
        vmax=0,
        return_with_data: bool = True,
    ):
        """Generates n distinct color from a given colormap for an array of desired data.
        Args:
            cmap(str): The name of the colormap you want to use.
                Refer https://matplotlib.org/stable/tutorials/colors/colormaps.html to choose

                Some suggestions:
                For Metallicity in Astrophysics: use coolwarm, bwr, seismic in reverse
                For distinct objects: Use gnuplot, brg, jet,turbo.

            data_arr(numpy.ndarray): The numpy array of data for which you want these distinct colors.

            vmin(float): 0 by default which sets vmin=minimum value in the data.
                When vmin is assigned a non zero value it normalizes the color based on this minimum value


            vmax(float): 0 by default which set vmax=maximum value in the data.
                When vmax is assigned a non zero value it normalizes the color based on this maximum value

            return_with_data(bool): True by default, returns a list of tuples (data_i, hex_color_i)

        Returns:
            colorlist_normalized(list): A normalized list of colors with hex values for the given array.
        """

        if (vmin == 0) and (vmax == 0):
            data_min = np.min(data_arr)
            data_max = np.max(data_arr)

        else:
            if vmin > np.min(data_arr):
                warn_string = (
                    "vmin you entered is greater than the minimum value in the data array "
                    + str(np.min(data_arr))
                )
                warnings.warn(warn_string)

            if vmax < np.max(data_arr):
                warn_string = (
                    "vmax you entered is smaller than the maximum value in the data array "
                    + str(np.max(data_arr))
                )
                warnings.warn(warn_string)

            data_arr = np.append(data_arr, [vmin, vmax])
            data_min = np.min(data_arr)
            data_max = np.max(data_arr)

        c_map = plt.get_cmap(str(cmap))  # select the desired cmap

        colorlist_normalized = []
        for c in data_arr:
            norm = (c - data_min) / (data_max - data_min) * 0.99
            if reverse:
                rgba = c_map(1 - norm)
            else:
                rgba = c_map(
                    norm
                )  # select the rgba value of the cmap at point c which is a number between 0 to 1
            clr = mcolors.rgb2hex(rgba)  # convert to hex

            if return_with_data:
                colorlist_normalized.append((c, str(clr)))
            else:
                colorlist_normalized.append(str(clr))  # create a list of these mcolors

        if (vmin == 0) and (vmax == 0):
            return colorlist_normalized
        else:
            colorlist_normalized = colorlist_normalized[:-2]
            return colorlist_normalized

    # def add_colorkey(self) -> None:
    #     """ Add a colorkey to the json file """
    #     pass

    # @staticmethod
    # def make_new_key(colorkey_dict) -> str:
    #     """ Returns a hex color code, putting it equidistant from other keys"""
    #     # Get the colors and make them hsv
    #     h_values = []

    #     for k,v in colorkey_dict.items():
    #         for key,color in v.items:
    #             rgb = Color.hex2rgb(mcolors.cnames[color['color']])
    #             h,_,_ = Color.rgb2hsv(rgb)
    #             h_values.append(h)
