"""Optogenetics silencing-pattern columns, shared by opto-capable psychophysics tasks.

A task whose sessions can include opto trials mixes :class:`OptoPatternMixin` into its RunData to
derive ``opto_region``/``stimkey``/``stim_label`` from the per-session silencing-pattern images,
and to read those images. Used by both detection and discrimination.
"""

from __future__ import annotations

import os
from os.path import join as pjoin

import numpy as np
import polars as pl
import tifffile as tf
from PIL import Image

from piepy.core.errors import OptoPatternError


class OptoPattern:
    """Mixin adding opto silencing-pattern columns + image reading to a RunData."""

    def add_pattern_related_columns(self, pattern_path: str | None) -> None:
        """Add ``opto_region``/``stimkey``/``stim_label`` from the session's silencing patterns.

        Non-opto sessions (a single ``opto`` value) get placeholder columns; opto sessions map
        each logged ``opto_pattern`` id to a region read from the pattern image filenames.

        Raises:
            OptoPatternError: opto session with no valid pattern dir, or an id with no image.
        """
        if len(self.data["opto"].unique()) == 1:
            # regular (non-opto) session: placeholder columns
            self.data = self.data.with_columns(pl.lit(None).alias("opto_region"))
            self.data = self.data.with_columns((pl.col("stim_type") + "_-1").alias("stimkey"))
            self.data = self.data.with_columns(pl.col("stim_type").alias("stim_label"))
            return

        if pattern_path is None or not os.path.exists(pattern_path):
            raise OptoPatternError(
                "This session has opto trials (more than one 'opto' value) but no valid "
                "opto-pattern directory was found.",
                where=pattern_path,
                fix="Provide the session's opto-pattern image directory (the .tif/.bmp files "
                "named '<region>_<id>'), or mark the session non-opto. Check "
                "config.paths['opto_pattern'].",
            )

        pattern_names = {}
        for im in os.listdir(pattern_path):
            if im.endswith(".tif"):
                pattern_id = int(im[:-4].split("_")[-1])
                if pattern_id == -1:
                    pattern_names[pattern_id] = "nonopto"
                else:
                    pattern_names[pattern_id] = im[:-4].split("_")[-2]

        try:
            self.data = self.data.with_columns(
                pl.struct(["opto_pattern", "state_outcome"])
                .map_elements(
                    lambda x: pattern_names[x["opto_pattern"]] if x["state_outcome"] != -1 else None,
                    return_dtype=str,
                )
                .alias("opto_region")
            )
        except KeyError:
            raise OptoPatternError(
                "An 'opto_pattern' id logged in the data has no matching pattern image.",
                where=pattern_path,
                fix="Rename each opto-pattern image so it ends with its integer id (e.g. "
                "'V1_0.tif', 'nonopto_-1.tif'); the ids must match the 'opto_pattern' values "
                "logged in the session.",
            ) from None

        self.data = self.data.with_columns(
            (pl.col("stim_type") + "_" + pl.col("opto_pattern").cast(int).cast(str)).alias("stimkey")
        )
        self.data = self.data.with_columns(
            (pl.col("stim_type") + "_" + pl.col("opto_region").cast(str)).alias("stim_label")
        )

    @staticmethod
    def read_pattern_images(pattern_path: str) -> dict:
        """Read the run's pattern/window images from ``pattern_path``, as ``{name: image}``."""
        imgs = {}
        for im in os.listdir(pattern_path):
            if im.endswith(".tif"):
                pattern_id = int(im[:-4].split("_")[-1])
                read_img = tf.imread(pjoin(pattern_path, im))
                if pattern_id == -1:
                    imgs["window"] = read_img
                else:
                    imgs[im[:-4].split("_")[-2]] = read_img
            elif im.endswith(".bmp"):
                name = im.split("_")[1]
                read_bmp = np.array(Image.open(pjoin(pattern_path, im)).convert("L"))
                imgs[f"pattern_{name}"] = read_bmp
        return imgs
