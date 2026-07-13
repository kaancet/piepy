"""viz infra: _resolve accepts Run/Session/Hub/DataFrame uniformly; _need guards columns."""

from __future__ import annotations

import types

import polars as pl
import pytest

from piepy.core.errors import SchemaError
from piepy.viz.base import _need, _resolve

DF = pl.DataFrame({"a": [1, 2]})


def test_resolve_dataframe_passthrough():
    assert _resolve(DF) is DF


def test_resolve_session_concatenates():
    session = types.SimpleNamespace(concatenate_runs=lambda: DF)
    assert _resolve(session) is DF


def test_resolve_hub_data_attr():
    hub = types.SimpleNamespace(data=DF)  # Hub.data is the cohort DataFrame
    assert _resolve(hub) is DF


def test_resolve_run_data_data():
    run = types.SimpleNamespace(
        data=types.SimpleNamespace(data=DF)
    )  # Run.data (RunData).data
    assert _resolve(run) is DF


def test_resolve_unsupported_raises_structured():
    with pytest.raises(SchemaError):
        _resolve(object())


def test_need_raises_naming_missing_column():
    with pytest.raises(SchemaError, match="missing column"):
        _need(DF, ["a", "nope"], plot="x")
    _need(DF, ["a"], plot="x")  # present -> no raise
