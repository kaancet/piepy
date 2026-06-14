"""Unit tests for piepy.stats.aggregate (Phase 3 workhorse). Data-free -> CI."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from piepy.stats import Count, Median, Rate, aggregate, group_arrays


@pytest.fixture
def trials():
    # two conditions x a known outcome / reaction-time structure
    return pl.DataFrame(
        {
            "cond": ["a"] * 6 + ["b"] * 4,
            "is_hit": [1, 1, 1, 0, 0, 0, 1, 1, 1, 1],  # a: 3/6=0.5, b: 4/4=1.0
            "outcome": [
                "hit",
                "hit",
                "hit",
                "miss",
                "miss",
                "miss",
                "hit",
                "hit",
                "hit",
                "hit",
            ],
            "rt": [10.0, 20, 30, 40, 50, 60, 100, 200, 300, 400],
        }
    )


def test_rate_shorthand_tidy_shape(trials):
    out = aggregate(trials, group="cond", rate="is_hit")
    assert out.columns == ["cond", "metric", "value", "ci_low", "ci_high", "n"]
    by = {r["cond"]: r for r in out.iter_rows(named=True)}
    assert by["a"]["value"] == 0.5 and by["a"]["n"] == 6
    assert by["b"]["value"] == 1.0 and by["b"]["n"] == 4
    # Wilson CI stays inside [0, 1] even at the 100% group
    assert 0.0 <= by["b"]["ci_low"] < 1.0 and by["b"]["ci_high"] <= 1.0
    assert by["a"]["metric"] == "rate[is_hit]"


def test_rate_with_categorical_success(trials):
    out = aggregate(trials, group="cond", metrics=[Rate("outcome", success="hit")])
    by = {r["cond"]: r["value"] for r in out.iter_rows(named=True)}
    assert by["a"] == 0.5 and by["b"] == 1.0


def test_median_order_statistic_ci(trials):
    out = aggregate(trials, group="cond", value="rt", stat="median").sort("cond")
    a = out.filter(pl.col("cond") == "a").row(0, named=True)
    assert a["value"] == 35.0  # median of 10..60
    assert a["ci_low"] <= 35.0 <= a["ci_high"]
    assert a["metric"] == "median[rt]"


def test_mean_t_ci(trials):
    out = aggregate(trials, group="cond", value="rt", stat="mean")
    a = out.filter(pl.col("cond") == "a").row(0, named=True)
    assert math.isclose(a["value"], 35.0)
    assert a["ci_low"] < 35.0 < a["ci_high"]


def test_multi_metric_one_pass(trials):
    out = aggregate(
        trials,
        group="cond",
        metrics=[Rate("is_hit"), Median("rt"), Count()],
    )
    # 2 groups x 3 metrics = 6 rows
    assert out.height == 6
    assert set(out["metric"].unique()) == {"rate[is_hit]", "median[rt]", "count"}
    cnt = out.filter((pl.col("cond") == "a") & (pl.col("metric") == "count")).row(
        0, named=True
    )
    assert cnt["value"] == 6.0


def test_multi_column_group(trials):
    out = aggregate(trials, group=["cond"], rate="is_hit")
    assert "cond" in out.columns
    # grouping is order-independent / deterministic via sort
    assert out["cond"].to_list() == ["a", "b"]


def test_empty_group_is_nan_not_crash():
    df = pl.DataFrame(
        {"cond": ["a", "a"], "rt": [None, None]},
        schema={"cond": pl.Utf8, "rt": pl.Float64},
    )
    out = aggregate(df, group="cond", value="rt", stat="median")
    r = out.row(0, named=True)
    assert r["n"] == 0
    assert r["value"] is None or math.isnan(r["value"])


def test_missing_column_raises():
    df = pl.DataFrame({"cond": ["a"], "rt": [1.0]})
    with pytest.raises(ValueError, match="not in the dataframe"):
        aggregate(df, group="cond", rate="nope")
    with pytest.raises(ValueError, match="not in the dataframe"):
        aggregate(df, group="missing", value="rt")


def test_bootstrap_median_option(trials):
    out = aggregate(trials, group="cond", metrics=[Median("rt", ci="bootstrap", seed=0)])
    a = out.filter(pl.col("cond") == "a").row(0, named=True)
    assert a["ci_low"] <= a["value"] <= a["ci_high"]


def test_rate_ci_always_brackets_value_at_extremes():
    # all-hit and all-miss groups: ci must bracket value exactly (no float overshoot that
    # would make value > ci_high / value < ci_low and break errorbar plots downstream).
    df = pl.DataFrame(
        {"cond": ["all_hit"] * 38 + ["all_miss"] * 20, "is_hit": [1] * 38 + [0] * 20}
    )
    out = aggregate(df, group="cond", rate="is_hit")
    for r in out.iter_rows(named=True):
        assert r["ci_low"] <= r["value"] <= r["ci_high"]
    hi = out.filter(pl.col("cond") == "all_hit").row(0, named=True)
    assert hi["value"] == 1.0 and hi["ci_high"] == 1.0  # exactly, not 0.999...


def test_group_arrays_bridges_to_compare(trials):
    arrs = group_arrays(trials, group="cond", value="rt")
    assert set(arrs) == {"a", "b"}
    assert np.array_equal(arrs["a"], np.array([10.0, 20, 30, 40, 50, 60]))
    assert np.array_equal(arrs["b"], np.array([100.0, 200, 300, 400]))
