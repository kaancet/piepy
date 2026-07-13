"""Unit tests for piepy.stats.aggregate (Phase 3 workhorse). Data-free -> CI."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from piepy.stats import Count, Median, Rate, aggregate, group_arrays, subject_average


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
    out = aggregate(trials, group="cond", rate="outcome", rate_of="hit")
    assert out.columns == ["cond", "metric", "value", "ci_low", "ci_high", "n"]
    by = {r["cond"]: r for r in out.iter_rows(named=True)}
    assert by["a"]["value"] == 0.5 and by["a"]["n"] == 6
    assert by["b"]["value"] == 1.0 and by["b"]["n"] == 4
    # Wilson CI stays inside [0, 1] even at the 100% group
    assert 0.0 <= by["b"]["ci_low"] < 1.0 and by["b"]["ci_high"] <= 1.0
    assert by["a"]["metric"] == "rate[outcome=hit]"


def test_rate_with_categorical_success(trials):
    out = aggregate(trials, group="cond", metrics=[Rate("outcome", rate_of="hit")])
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
    out = aggregate(trials, group=["cond"], rate="outcome", rate_of="hit")
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


def test_bootstrap_median_option(trials):
    out = aggregate(trials, group="cond", metrics=[Median("rt", ci="bootstrap", seed=0)])
    a = out.filter(pl.col("cond") == "a").row(0, named=True)
    assert a["ci_low"] <= a["value"] <= a["ci_high"]


def test_rate_ci_always_brackets_value_at_extremes():
    # all-hit and all-miss groups: ci must bracket value exactly (no float overshoot that
    # would make value > ci_high / value < ci_low and break errorbar plots downstream).
    df = pl.DataFrame(
        {"cond": ["all_hit"] * 38 + ["all_miss"] * 20, "outcome": [1] * 38 + [0] * 20}
    )
    out = aggregate(df, group="cond", rate="outcome", rate_of=1)
    for r in out.iter_rows(named=True):
        assert r["ci_low"] <= r["value"] <= r["ci_high"]
    hi = out.filter(pl.col("cond") == "all_hit").row(0, named=True)
    assert hi["value"] == 1.0 and hi["ci_high"] == 1.0  # exactly, not 0.999...


def test_group_arrays_bridges_to_compare(trials):
    arrs = group_arrays(trials, group="cond", value="rt")
    assert set(arrs) == {"a", "b"}
    assert np.array_equal(arrs["a"], np.array([10.0, 20, 30, 40, 50, 60]))
    assert np.array_equal(arrs["b"], np.array([100.0, 200, 300, 400]))


def test_points_attaches_raw_values_for_value_metrics(trials):
    out = aggregate(trials, group="cond", value="rt", stat="median", points=True)
    assert "points" in out.columns
    by = {r["cond"]: r for r in out.iter_rows(named=True)}
    assert sorted(by["a"]["points"]) == [10.0, 20, 30, 40, 50, 60]
    assert sorted(by["b"]["points"]) == [100.0, 200, 300, 400]
    # list length matches the reported n
    assert all(len(r["points"]) == r["n"] for r in out.iter_rows(named=True))


def test_points_left_null_for_rate_metric(trials):
    # Rate has no underlying value distribution -> points stays null even when requested
    out = aggregate(
        trials, group="cond", metrics=[Rate("is_hit"), Median("rt")], points=True
    )
    rate_rows = out.filter(pl.col("metric").str.starts_with("rate"))
    assert rate_rows["points"].is_null().all()
    med_rows = out.filter(pl.col("metric").str.starts_with("median"))
    assert med_rows["points"].is_not_null().all()


def test_subject_average_averages_across_subjects_equally():
    # subject A: 2 trials at x=0 (1 hit -> 0.5); subject B: 100 trials at x=0 (90 hits -> 0.9).
    # pooled aggregate is dominated by B (~0.89); subject_rate weights A and B equally -> ~0.70.
    rows = [{"x": 0, "animal": "A", "out": "hit" if i == 0 else "miss"} for i in range(2)]
    rows += [
        {"x": 0, "animal": "B", "out": "hit" if i < 90 else "miss"} for i in range(100)
    ]
    df = pl.DataFrame(rows)

    pooled = aggregate(df, group="x", rate="out", rate_of="hit")["value"][0]
    subj = subject_average(df, x="x", subject="animal", rate="out", rate_of="hit")
    assert subj.height == 1
    assert subj["n"][0] == 2  # n is subjects, not trials
    assert abs(subj["value"][0] - 0.70) < 1e-9  # mean(0.5, 0.9)
    assert pooled > 0.85  # pooling would have over-weighted subject B


def test_subject_average_with_compare_keeps_group_column():
    rng = np.random.default_rng(0)
    rows = []
    for animal in ("A", "B", "C"):
        for opto in (0, 1):
            for c in (-0.5, 0.5):
                for hit in rng.random(40) < (0.7 if opto == 0 else 0.4):
                    rows.append(
                        {
                            "x": c,
                            "animal": animal,
                            "opto": opto,
                            "out": "hit" if hit else "miss",
                        }
                    )
    df = pl.DataFrame(rows)
    out = subject_average(
        df, x="x", subject="animal", rate="out", rate_of="hit", compare="opto"
    )
    assert "opto" in out.columns
    assert out.height == 4  # 2 x-levels x 2 opto
    assert (out["n"] == 3).all()  # 3 subjects per cell
