"""Session.analyze = concatenate_runs + enrich. Base enrich is a no-op; a paradigm overrides it
to join per-run/cohort columns. Data-free -- a tiny Session subclass stands in for a parsed one.
"""

from __future__ import annotations

from types import SimpleNamespace

import polars as pl

from piepy.core.session import Session


def _run(trial_df, meta):
    return SimpleNamespace(data=SimpleNamespace(data=trial_df), meta=meta)


class _FakeSession(Session):
    """Bypass __init__/IO: just supply runs + a canned concatenate_runs."""

    def __init__(self, base, runs):
        self._base = base
        self.runs = runs

    def concatenate_runs(self, paradigm=None):
        return self._base


def test_base_analyze_is_just_concatenate():
    base = pl.DataFrame({"run_no": [1, 2]}, schema_overrides={"run_no": pl.UInt32})
    out = _FakeSession(base, []).analyze()  # base enrich is a no-op
    assert out.columns == ["run_no"] and out.height == 2
