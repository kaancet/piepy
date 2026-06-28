"""Pure trial-table transforms shared across psychophysics tasks (detection, discrimination, ...).

Each is a ``pl.DataFrame -> pl.DataFrame`` (or a small reporter); a task composes the ones it
needs into its RunData ``augmenters`` pipeline (and the run-context ones into its ``augment_data``
hook) instead of inheriting behavior classes. This keeps capabilities explicit, individually
needs, in order, inside its ``Run.augment_data`` (where run context like paths/meta is on
``self``) instead of inheriting behavior classes. This keeps capabilities explicit, individually
testable, and free of mixin/MRO coupling.
"""

from __future__ import annotations

import polars as pl
from scipy.optimize import curve_fit

from piepy.core.errors import ParsingError
from piepy.core.io import display


def add_rig_response_time(df: pl.DataFrame) -> pl.DataFrame:
    """Interpolate a rig-based ``response_time`` for trials lacking ``rig_response_time``.

    Fits ``rig = state + a`` on the trials that do have a rig time, applies it to all hit trials,
    and keeps the state time for misses (state_outcome == 0).
    """
    with_rig = df.drop_nulls("rig_response_time")
    if len(with_rig) == 0:
        display(
            "No rig response time to interpolate; copying state response time.",
            color="yellow",
        )
        return df.with_columns(pl.col("state_response_time").alias("response_time"))

    def _line(x, a):
        return x + a  # slope fixed at 1; a is the state->rig time offset

    popt, _ = curve_fit(_line, with_rig["state_response_time"], with_rig["rig_response_time"])
    new_times = _line(df["state_response_time"], *popt)
    return (
        df.with_columns(pl.Series("temp_response_times", new_times))
        .with_columns(
            pl.when(pl.col("state_outcome") != 0)
            .then(pl.col("temp_response_times"))
            .otherwise(pl.col("state_response_time"))
            .alias("response_time")
        )
        .drop("temp_response_times")
    )


def add_runno(df: pl.DataFrame, runno: int) -> pl.DataFrame:
    """Adds the tun no as a column"""
    return df.with_columns(pl.lit(runno).cast(pl.UInt8).alias("run_no"))


def set_outcome(df: pl.DataFrame, outcome_type: str = "state") -> pl.DataFrame:
    """Return ``df`` with ``outcome`` pointed at the ``<outcome_type>_outcome`` column."""
    col = f"{outcome_type}_outcome"
    if col not in df.columns:
        available = sorted(c[: -len("_outcome")] for c in df.columns if c.endswith("_outcome"))
        raise ParsingError(
            f"{outcome_type!r} is not a valid outcome type.",
            fix=f"Pass one of {available} (the outcome variants present in this data).",
        )
    return df.with_columns(pl.col(col).alias(col))


def add_stim_side(df: pl.DataFrame) -> pl.DataFrame:
    """Stimulus identity columns: ``stim_side`"""
    return df.with_columns(
        pl.when(pl.col("stim_pos") > 0)
        .then(pl.lit("contra"))
        .when(pl.col("stim_pos") < 0)
        .then(pl.lit("ipsi"))
        .when((pl.col("stim_pos") == 0) | (pl.col("isCatch") == 1))
        .then(pl.lit("catch"))
        .otherwise(None)
        .alias("stim_side")
    )


def add_sftf_descriptor(df: pl.DataFrame) -> pl.DataFrame:
    """Rounded ``sf``/``tf``, and a ``stim_type`` key"""
    df = df.with_columns(pl.col("sf").round(2).alias("sf"), pl.col("tf").round(1).alias("tf"))

    return df.with_columns(
        (pl.col("sf").round(2).cast(str) + "cpd_" + pl.col("tf").cast(str) + "Hz").alias("stim_type")
    )
