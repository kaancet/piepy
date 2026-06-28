"""obj.viz.psychometric() forwards to the free function with scope-appropriate defaults.

behaviz-free: we monkeypatch plots.psychometric to capture how the accessor calls it, so this
tests the delegation + the Hub-subject / Session-pool defaults without drawing anything.
"""

from __future__ import annotations

import polars as pl

from piepy.viz import Viz
from piepy.viz import plots


def _capture(monkeypatch):
    calls = {}

    def fake(data, **kwargs):
        calls["data"], calls["kwargs"] = data, kwargs
        return "PLOTRESULT"

    monkeypatch.setattr(plots, "psychometric", fake)
    return calls


def test_session_scope_pools(monkeypatch):
    calls = _capture(monkeypatch)
    out = Viz("SESSION_OBJ").psychometric(x="signed_contrast")
    assert out == "PLOTRESULT"
    assert calls["data"] == "SESSION_OBJ"
    assert calls["kwargs"]["average_over"] is None  # single scope -> pooled


def test_hub_scope_subject_averages(monkeypatch):
    calls = _capture(monkeypatch)
    Viz("HUB_OBJ", subject="animalid").psychometric(compare="opto")
    assert calls["kwargs"]["average_over"] == "animalid"  # cohort scope default
    assert calls["kwargs"]["compare"] == "opto"  # other kwargs pass straight through


def test_explicit_average_over_overrides_scope_default(monkeypatch):
    calls = _capture(monkeypatch)
    Viz("HUB_OBJ", subject="animalid").psychometric(average_over="mouse")
    assert calls["kwargs"]["average_over"] == "mouse"  # caller wins over the scope default


def test_filter_scalar_keeps_matching_rows(monkeypatch):
    calls = _capture(monkeypatch)
    df = pl.DataFrame({"opto": [0, 1, 1], "area": ["V1", "V1", "LM"], "x": [1, 2, 3]})
    Viz(df).psychometric(filterer={"opto": 1})
    assert calls["data"]["x"].to_list() == [2, 3]  # only opto == 1


def test_filter_list_and_multiple_keys_anded(monkeypatch):
    calls = _capture(monkeypatch)
    df = pl.DataFrame({"opto": [0, 1, 1], "area": ["V1", "V1", "LM"], "x": [1, 2, 3]})
    Viz(df).psychometric(filterer={"opto": [0, 1], "area": ["LM"]})
    assert calls["data"]["x"].to_list() == [3]  # opto in {0,1} AND area in {LM}


def test_no_filter_passes_object_through(monkeypatch):
    calls = _capture(monkeypatch)
    Viz("OBJ").psychometric()
    assert calls["data"] == "OBJ"  # unresolved -> the plot resolves it itself


def test_property_attached_to_core_objects():
    # the property exists and binds the object (no parsing needed for this check)
    from piepy.core.hub import Hub

    hub = Hub("wheel_detection")
    hub.data = pl.DataFrame({"animalid": ["a"]})
    assert isinstance(hub.viz, Viz)
    assert hub.viz._subject == "animalid"  # Hub -> cohort scope
