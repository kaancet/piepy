"""Regression: parsing must return a *plain* polars DataFrame, not a patito one whose
dynamically-expanded model class can't be pickled -- otherwise Hub's spawn workers fail to
ship their result back to the parent (the 'ExpandedExpanded...DataFrame' PicklingError).
"""

from __future__ import annotations

import pickle

import patito as pt
import polars as pl
import pytest

from piepy.core.run import to_plain_polars


class _M(pt.Model):
    a: int


def _patito_frame_with_dynamic_model():
    expanded = _M.with_fields(
        b=(int, None)
    )  # dynamically-created, not importable by name
    return pt.DataFrame({"a": [1, 2], "b": [3, 4]}).set_model(expanded)


def test_patito_expanded_model_frame_is_unpicklable():
    # documents the failure mode the fix targets
    with pytest.raises(Exception):
        pickle.dumps(_patito_frame_with_dynamic_model())


def test_to_plain_polars_yields_picklable_base_frame():
    plain = to_plain_polars(_patito_frame_with_dynamic_model())
    assert type(plain) is pl.DataFrame  # base class, no patito model attached
    assert pickle.loads(pickle.dumps(plain)).to_dict(as_series=False) == {
        "a": [1, 2],
        "b": [3, 4],
    }


def test_with_columns_keeps_a_plain_frame_picklable():
    # the RunData augmenter pipeline runs with_columns; it must not re-introduce patito
    plain = to_plain_polars(_patito_frame_with_dynamic_model())
    out = plain.with_columns(pl.lit(1).alias("c"))
    assert type(out) is pl.DataFrame
    pickle.dumps(out)  # must not raise
