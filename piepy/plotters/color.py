import colorsys
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors

from ..core.config import config as cfg


class Color:
    def __init__(self):
        # this is hardcoded for now, fix this
        self.colorkey_path = cfg.paths["colors"][0]
        self.read_colors()

    @staticmethod
    def check_hex(hex_code: str) -> None:
        """Makes sure hex code is in correct form"""
        if not (hex_code.startswith("#") and len(hex_code) == 7):
            raise ValueError("Both colors must be in the format '#RRGGBB'.")

    def read_colors(self) -> None:
        """Reads the colorkey.json and returns a dict of color keys for different sftf and contrast values"""
        with open(self.colorkey_path, "r") as f:
            keys = json.load(f)
            self.stim_keys = keys["spatiotemporal"]
            self.contrast_keys = keys["contrast"]
            self.outcome_keys = keys["outcome"]

    def check_stim_colors(self, keys: list[str]) -> None:
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

    def check_contrast_colors(self, contrasts: list[str]) -> None:
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

    def make_key_mixed_color(self, stim_key: str, contrast_key: str) -> str:
        """ """
        s_color = self.stim_keys[stim_key]["color"]
        c_color = self.contrast_keys[contrast_key]["color"]
        return self.mix_colors(s_color, c_color)

    @classmethod
    def name2hsv(cls, color_name: str) -> tuple[float, float, float]:
        rgb = mcolors.to_rgb(color_name)
        return Color.rgb2hsv(
            rgb, normalize=False
        )  # no need to normalize here, already 0-1 range from mcolors method

    @classmethod
    def hex2rgb(cls, hex_code: str) -> tuple[float, float, float]:
        hex_code = hex_code.lstrip("#")
        return tuple(int(hex_code[i : i + 2], 16) for i in (0, 2, 4))

    @classmethod
    def rgb2hex(cls, rgb_tuple: tuple[float, float, float]) -> str:
        def clamp(x):
            return max(0, min(x, 255))

        rgb_tuple = tuple(int(clamp(i)) for i in rgb_tuple)
        r, g, b = rgb_tuple
        return f"#{r:02x}{g:02x}{b:02x}"

    @classmethod
    def normalize_rgb(
        cls, rgb_tuple: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        return tuple(i / 255.0 for i in rgb_tuple)

    @classmethod
    def rgb2hsv(
        cls, rgb_tuple: tuple[float, float, float], normalize: bool = True
    ) -> tuple[float, float, float]:
        if normalize:
            rgb_tuple = Color.normalize_rgb(rgb_tuple)
        r, g, b = rgb_tuple
        return colorsys.rgb_to_hsv(r, g, b)

    @classmethod
    def hsv2rgb(cls, hsv_tuple: tuple[float, float, float]) -> tuple[float, float, float]:
        h, s, v = hsv_tuple
        return colorsys.hsv_to_rgb(h, s, v)

    @classmethod
    def lighten(cls, hex_color: str, l_coeff: float = 0.33) -> str:
        """Lightens the hsv_tuple by l_coeff percent, aka from S subtracts l_coeff percent of the S value"""

        hsv_tuple = cls.rgb2hsv(cls.hex2rgb(hex_color))

        if not l_coeff <= 1 and l_coeff >= 0:
            raise ValueError(
                f"The l_coeff value needs to be 0<=l_coeff<= 1, got {l_coeff} instead"
            )
        h, s, v = hsv_tuple
        s_new = s - (s * l_coeff)
        return cls.rgb2hex(cls.hsv2rgb((h, s_new, v)))

    @classmethod
    def make_color_range(
        cls,
        start_color: str,
        rng: int,
        s_limit: list = [20, 100],
        v_limit: list = [20, 100],
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

    @classmethod
    def mix_colors(cls, hex_color1: str, hex_color2: str) -> str:
        """Mixes the two input hex colors"""

        cls.check_hex(hex_color1)
        cls.check_hex(hex_color2)

        r1, g1, b1 = cls.hex2rgb(hex_color1)
        r2, g2, b2 = cls.hex2rgb(hex_color2)

        # Calculate the average for each channel
        r_mix = round((r1 + r2) / 2)
        g_mix = round((g1 + g2) / 2)
        b_mix = round((b1 + b2) / 2)

        return cls.rgb2hex(rgb_tuple=(r_mix, g_mix, b_mix))

    @classmethod
    def gen_color(cls, cmap, n, reverse=False):
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

        if reverse:
            colorlist.reverse()
        return colorlist

    @classmethod
    def gen_color_normalized(
        cls,
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
