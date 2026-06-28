"""Base builder for a session's temporaldata ``Data``, from the trial table.

The base owns the **universal** part -- the ``trials`` Interval (trial start/end, which is the
data's domain) -- plus a declarative ``series`` map so a task can expose simple extra columns as
regular/irregular streams with one line (e.g. ``series = {"2p_frame": "regular"}``). Anything that
needs pairing or clock alignment (a value paired with its timestamps, the rig clock, ...) goes in a
task subclass's :meth:`extra_streams`.

A task's stream class lives next to its Session/Trial (``tasks/<task>/streams.py``) and subclasses
this. Everything here reads only the canonical session-clock columns, so it is not tied to any one
acquisition system -- only to the trial-table contract.
"""

from __future__ import annotations

from abc import ABC

import numpy as np
import polars as pl
from temporaldata import Data, Interval, IrregularTimeSeries, RegularTimeSeries


class SessionStreams(ABC):
    """Build a temporaldata ``Data`` of one session's streams. Subclass per task.

    Class attributes a subclass sets:
        series: extra columns to expose as ``{name: "irregular" | "regular"}``. ``"irregular"`` reads
            ``<name>_session`` (a per-trial list of session-clock times); ``"regular"`` reads ``name``
            at :attr:`sampling_rate`.
        trial_attrs: trial-table columns attached to the ``trials`` Interval as attributes.
        sampling_rate: Hz for any ``"regular"`` series.
    """

    series: dict[str, str] = {}
    trial_attrs: tuple[str, ...] = ("trial_no",)
    sampling_rate: float | None = None

    def __init__(self, data) -> None:
        self.df = data if isinstance(data, pl.DataFrame) else data.concatenate_runs()
        if "t_trialstart_session" not in self.df.columns:
            raise ValueError("session-clock columns missing; build them with Session.concatenate_runs().")

    def build(self) -> Data:
        """The session's ``Data``: the ``trials`` domain + the declared ``series`` + task extras."""
        trials = self._trials()
        domain = trials.coalesce()
        streams: dict = {"trials": trials}
        for name, kind in self.series.items():
            s = self._regular(name) if kind == "regular" else self._irregular(name, domain)
            if s is not None:
                streams[name] = s
        streams.update(self.extra_streams(domain))
        return Data(domain=domain, **streams)

    # -- universal commonality ------------------------------------------------------------------ #
    def _trials(self) -> Interval:
        df = self.df
        attrs = [c for c in self.trial_attrs if c in df.columns]
        return Interval(
            start=df["t_trialstart_session"].cast(pl.Float64).to_numpy(),
            end=df["t_trialend_session"].cast(pl.Float64).to_numpy(),
            **{c: df[c].to_numpy() for c in attrs},
        )

    # -- generic builders for the declarative `series` map ------------------------------------- #
    def _irregular(self, name: str, domain: Interval) -> IrregularTimeSeries | None:
        """Irregular series from a per-trial list of session-clock times (``<name>_session``)."""
        col = f"{name}_session" if f"{name}_session" in self.df.columns else name
        if col not in self.df.columns:
            return None
        t = self.df.select(col).explode(col).drop_nulls(col).sort(col)
        if not t.height:
            return None
        return IrregularTimeSeries(timestamps=t[col].cast(pl.Float64).to_numpy(), domain=domain)

    def _regular(self, name: str) -> RegularTimeSeries | None:
        """Regular (evenly sampled) series from column ``name`` at :attr:`sampling_rate`."""
        if name not in self.df.columns or self.sampling_rate is None:
            return None
        vals = self.df.select(name).explode(name).drop_nulls(name)[name].cast(pl.Float64).to_numpy()
        return RegularTimeSeries(sampling_rate=self.sampling_rate, **{name: vals})

    # -- task hook ------------------------------------------------------------------------------ #
    def extra_streams(self, domain: Interval) -> dict:
        """Override to add streams that need pairing / clock alignment (``domain`` for the streams).
        Default: none."""
        return {}


def trial_slice(session_data: Data, trial_no: int) -> Data:
    """One trial as a slice of the session ``Data`` (relative time). A trial is a slice, not a build."""
    nos = np.asarray(session_data.trials.trial_no)
    idx = np.flatnonzero(nos == trial_no)
    if not idx.size:
        raise ValueError(f"trial_no {trial_no} not in this session.")
    i = int(idx[0])
    return session_data.slice(float(session_data.trials.start[i]), float(session_data.trials.end[i]))
