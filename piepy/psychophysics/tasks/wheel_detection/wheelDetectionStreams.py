"""Wheel-detection session streams: wheel (+position), licks, reward, stim windows.

Subclasses the generic :class:`piepy.temporal.base.SessionStreams`. The base builds the universal
``trials`` domain; this adds the detection-specific streams in :meth:`extra_streams`, where the
two-clock Stimpy detail lives: wheel/lick/reward are on the **rig** clock and are shifted onto the
**state** clock by the per-trial ``t_vstimstart - t_vstimstart_rig`` offset.
"""

from __future__ import annotations

import polars as pl
from temporaldata import Interval, IrregularTimeSeries

from piepy.temporal.base import SessionStreams


class WheelDetectionStreams(SessionStreams):
    trial_attrs = (
        "trial_no",
        "outcome",
        "contrast",
        "signed_contrast",
        "stim_side",
        "opto",
    )

    def extra_streams(self, trials: Interval) -> dict:
        df = self._with_rig_offset()
        out = {"wheel": self._wheel(df, trials)}
        for name, col in (("licks", "lick_session"), ("reward", "reward_session")):
            s = self._train(df, col, trials, first_only=(name == "reward"))
            if s is not None:
                out[name] = s
        if {"t_vstimstart_session", "t_vstimend_session"} <= set(df.columns):
            s = df.drop_nulls(["t_vstimstart_session", "t_vstimend_session"])
            if s.height:
                out["stim"] = Interval(
                    start=s["t_vstimstart_session"].cast(pl.Float64).to_numpy(),
                    end=s["t_vstimend_session"].cast(pl.Float64).to_numpy(),
                )
        return out

    def _with_rig_offset(self) -> pl.DataFrame:
        """Add ``_off`` = state-minus-rig stim-onset offset per trial (0 when there is no rig clock)."""
        df = self.df
        if {"t_vstimstart_session", "t_vstimstart_rig_session"} <= set(df.columns):
            off = (
                pl.col("t_vstimstart_session") - pl.col("t_vstimstart_rig_session")
            ).cast(pl.Float64)
            df = df.with_columns(off.alias("_off"))
            return df.with_columns(pl.col("_off").fill_null(df["_off"].median() or 0.0))
        return df.with_columns(pl.lit(0.0).alias("_off"))

    def _wheel(self, df: pl.DataFrame, trials: Interval) -> IrregularTimeSeries:
        w = (
            df.select("wheel_t_session", "wheel_pos", "_off")
            .explode(["wheel_t_session", "wheel_pos"])
            .drop_nulls("wheel_t_session")
            .with_columns((pl.col("wheel_t_session") + pl.col("_off")).alias("t"))
            .sort("t")
        )
        return IrregularTimeSeries(
            timestamps=w["t"].cast(pl.Float64).to_numpy(),
            position=w["wheel_pos"].cast(pl.Float64).to_numpy(),
            domain=trials,
        )

    def _train(self, df, col, trials, *, first_only=False) -> IrregularTimeSeries | None:
        """Rig-clock event train -> state clock. ``first_only`` keeps just each list's leading time
        (reward = ``[time, amount]``)."""
        if col not in df.columns:
            return None
        time = pl.col(col).list.first() if first_only else pl.col(col)
        sel = df.select(time.alias("ts"), "_off")
        if (
            not first_only
        ):  # a full train is a list per trial -> explode; first_only is already scalar
            sel = sel.explode("ts")
        t = (
            sel.drop_nulls("ts")
            .with_columns((pl.col("ts") + pl.col("_off")).alias("t"))
            .sort("t")
        )
        return (
            IrregularTimeSeries(
                timestamps=t["t"].cast(pl.Float64).to_numpy(), domain=trials
            )
            if t.height
            else None
        )
