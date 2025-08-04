import os
import numpy as np
import skimage as ski
import tifffile as tf
import matplotlib.pyplot as plt
from ..plotters.plotting_utils import override_plots


def get_pattern_edges(
    pattern_dir: str,
    ax: plt.Axes | None = None,
    gamma: float = 1.5,
    gauss_std: int = 5,
    adaptive_histogram_kernel: int = 127,
    closing_footprint: int = 30,
    binary_threshold: float = 0.5,
    do_convex_hull: bool = False,
    iso_close_radius: int = 100,
    binary_dilate_footprint: int = 10,
    mpl_kwargs: dict | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """_summary_

    Args:
        pattern_dir (str): _description_
        gamma (float, optional): _description_. Defaults to 1.5.
        gauss_std (int, optional): _description_. Defaults to 5.
        adaptive_histogram_kernel (int, optional): _description_. Defaults to 127.
        closing_footprint (int, optional): _description_. Defaults to 30.
        binary_threshold (float, optional): _description_. Defaults to 0.6.
    """
    override_plots()
    # read the images
    for f in os.listdir(pattern_dir):
        if f.endswith(".tif"):
            if f.endswith("_-1.tif"):
                background = tf.imread(os.path.join(pattern_dir, f))
                # normalize
                background = background / np.max(background)
            elif f.endswith("_0.tif"):
                pattern = tf.imread(os.path.join(pattern_dir, f))
                # normalize
                pattern = pattern / np.max(pattern)

    # gamma correct
    pattern_gamma = ski.exposure.adjust_gamma(pattern, gamma=gamma)
    back_gamm = ski.exposure.adjust_gamma(background, gamma=gamma)

    # gaussian blur
    pattern_gauss = ski.filters.gaussian(pattern_gamma, gauss_std)

    # enhance contrast (adaptive histogram)
    pattern_clahe = ski.exposure.equalize_adapthist(
        pattern_gauss, kernel_size=adaptive_histogram_kernel
    )

    # closing
    footprint = ski.morphology.disk(closing_footprint)
    pattern_clahe_closed = ski.morphology.closing(pattern_clahe, footprint)

    # binary filtering
    pattern_clahe_binary = np.zeros_like(pattern_clahe_closed)
    pattern_clahe_binary[pattern_clahe_closed >= binary_threshold] = 1

    if do_convex_hull:
        # do convex hull for individual area
        pattern_closed = ski.morphology.convex_hull_image(pattern_clahe_binary)
    else:
        # otherwise do isoclose and dilate
        iso_closed = ski.morphology.isotropic_closing(
            pattern_clahe_binary, radius=iso_close_radius
        )
        pattern_closed = ski.morphology.binary_dilation(
            iso_closed, footprint=ski.morphology.disk(binary_dilate_footprint)
        )

    edges = ski.feature.canny(pattern_closed)

    # plot on top
    if ax is None:
        f, ax = plt.subplots(1, 1)
    else:
        f = ax.get_figure()

    # make rgb image and put pattern in the blue channel
    bck_color = ski.color.gray2rgb(back_gamm)
    pattern_clr = np.zeros_like(bck_color)

    pattern_clr[:, :, 2] = ski.exposure.adjust_gamma(pattern, gamma=gamma * 2)

    ax.imshow(bck_color + pattern_clr)

    borders = np.zeros_like(pattern)
    borders[:] = np.nan

    x, y = np.where(edges)
    ax._scatter(y, x, s=1, c="k", mpl_kwargs=mpl_kwargs)
    ax.set_axis_off()

    return f, ax
