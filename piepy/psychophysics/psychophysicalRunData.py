"""Behavioral RunData methods shared across psychophysics tasks (detection, discrimination, ...).

These operate on a parsed trial table (one row per trial) and are task-agnostic *within*
psychophysics: switching/auditing the ``outcome`` column and interpolating a rig-based response
time. A task's RunData subclasses :class:`PsychophysicalRunData` (often together with
:class:`piepy.psychophysics.opto.OptoPatternMixin`) and adds only its task-specific
quality-of-life columns.
"""

from __future__ import annotations

import polars as pl
from scipy.optimize import curve_fit
from tabulate import tabulate

from piepy.core.errors import ParsingError
from piepy.core.io import display
from piepy.core.run import RunData


class PsychophysicalRunData(RunData):
    """Base RunData for psychophysics tasks: outcome handling + rig response-time interpolation."""

    def set_outcome(self, outcome_type: str = "state") -> None:
        """Point the ``outcome`` column at a chosen ``<outcome_type>_outcome`` variant."""
        display(f"Setting outcome to {outcome_type}")
        col_name = f"{outcome_type}_outcome"
        if col_name not in self.data.columns:
            available = sorted(
                c[: -len("_outcome")] for c in self.data.columns if c.endswith("_outcome")
            )
            raise ParsingError(
                f"{outcome_type!r} is not a valid outcome type.",
                fix=f"Pass one of {available} (the outcome variants present in this data).",
            )
        self.data = self.data.with_columns(pl.col(col_name).alias("outcome"))

    def compare_outcomes(self) -> None:
        """Print a small table comparing the different ``*_outcome`` columns."""
        out_cols = [c for c in self.data.columns if "_outcome" in c]
        q = (
            self.data.group_by(out_cols)
            .agg([pl.count().alias("count")])
            .sort(["state_outcome"])
        )
        print(tabulate(q.to_pandas(), headers=q.columns))

    def add_rig_response_time(self) -> None:
        """Interpolate a rig-based ``response_time`` for trials lacking ``rig_response_time``."""
        with_rig_time = self.data.drop_nulls("rig_response_time")
        resp_time = with_rig_time["state_response_time"]
        rig_time = with_rig_time["rig_response_time"]

        def m1_func(x, a):
            m = 1
            return m * x + a

        if len(rig_time):
            popt, _ = curve_fit(
                m1_func, resp_time, rig_time
            )  # popt[0] = time-diff intercept
            new_rig_times = m1_func(self.data["state_response_time"], *popt)
            self.data = self.data.with_columns(
                pl.Series("temp_response_times", new_rig_times)
            )
            # don't change miss, they become hits
            self.data = self.data.with_columns(
                pl.when(pl.col("state_outcome") != 0)
                .then(pl.col("temp_response_times"))
                .otherwise(pl.col("state_response_time"))
                .alias("response_time")
            ).drop("temp_response_times")
        else:
            display(
                "No rig response time to interpolate; copying state response time.",
                color="yellow",
            )
            self.data = self.data.with_columns(
                pl.col("state_response_time").alias("response_time")
            )
