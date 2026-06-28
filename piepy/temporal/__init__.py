"""temporaldata layer: a session's behavioural streams on one clock, built from the trial table.

``SessionStreams`` is the base builder (universal ``trials`` domain + a declarative ``series`` map);
each task subclasses it next to its Session/Trial (e.g. ``WheelDetectionStreams``). A single trial
is a slice of the resulting ``Data`` (``trial_slice`` / ``Data.slice``), not a separate build.
"""

from .base import SessionStreams, trial_slice

__all__ = ["SessionStreams", "trial_slice"]
