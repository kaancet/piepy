"""RunData is a pure store; a task derives columns by composing pure df -> df transforms in its
Run.augment_data (one phase, run-context available on self). These cover the store contract and
the detection transforms in isolation.
"""

from __future__ import annotations

import polars as pl

from piepy.psychophysics.tasks.wheel_detection.wheelDetectionSession import (
    add_contrast_descriptors,
    add_sftf_descriptor,
)
from piepy.psychophysics.transforms import add_stim_side


def test_detection_transforms_compose_in_order():
    df = pl.DataFrame(
        {
            "stim_pos": [1, -1, 0],
            "isCatch": [0, 0, 1],
            "sf": [0.1, 0.1, 0.1],
            "tf": [4.0, 4.0, 4.0],
            "contrast": [0.5, 0.0625, 0.0],
        }
    )
    df = add_stim_side(df)
    df = add_sftf_descriptor(df)
    assert df["stim_side"].to_list() == ["contra", "ipsi", "catch"]
    assert "stim_type" in df.columns

    df = add_contrast_descriptors(df)  # depends on stim_side from the previous transform
    assert df["signed_contrast"].to_list() == [0.5, -0.0625, 0.0]
    assert df["contrast_type"].to_list() == ["easy", "hard", "catch"]
