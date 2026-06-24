"""psychometric() wires stats + fitting + behaviz into a PlotResult.

behaviz is the drawing dep; the draw path is skipped when it isn't installed. The behaviz-free
shaping (aggregate / subject_average / compare_by_x) is covered in the stats test modules.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from piepy.viz import PlotResult, psychometric


def _toy_detection(n_per: int = 200, compare: bool = False) -> pl.DataFrame:
    """Trials whose hit probability follows a logistic in signed_contrast."""
    rng = np.random.default_rng(0)
    levels = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]
    rows = []
    groups = [0, 1] if compare else [None]
    for g in groups:
        shift = 0.0 if not g else -0.2  # opto shifts the curve
        for c in levels:
            p = 1 / (1 + np.exp(-4 * (c + shift)))
            for hit in rng.random(n_per) < p:
                row = {"signed_contrast": c, "outcome": "hit" if hit else "miss"}
                if compare:
                    row["opto"] = g
                rows.append(row)
    return pl.DataFrame(rows)


def test_psychometric_returns_populated_plotresult():
    pytest.importorskip("behaviz")
    res = psychometric(_toy_detection(), x="signed_contrast")
    assert isinstance(res, PlotResult)
    # one tidy row per stimulus level, Wilson CIs bracketing the estimate
    assert res.data.height == 7
    assert (res.data["ci_low"] <= res.data["value"]).all()
    assert (res.data["value"] <= res.data["ci_high"]).all()
    assert res.figure is not None
