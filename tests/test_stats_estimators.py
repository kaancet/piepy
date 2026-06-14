"""Unit tests for piepy.stats estimators (Phase 3, chunk 1). Data-free -> CI."""

from __future__ import annotations

import math

import numpy as np

from piepy.stats import Estimate, bootstrap_ci, mean_ci, median_ci, proportion_ci


def test_proportion_ci_wilson_basic():
    e = proportion_ci(8, 10)
    assert e.value == 0.8
    assert e.n == 10
    assert 0.0 <= e.ci_low < 0.8 < e.ci_high <= 1.0
    assert "wilson" in e.method


def test_proportion_ci_extremes_stay_in_unit_interval():
    # the case the naive normal interval breaks on
    hi = proportion_ci(10, 10)  # 100%
    lo = proportion_ci(0, 10)  # 0%
    assert hi.value == 1.0 and hi.ci_high <= 1.0 and hi.ci_low < 1.0
    assert lo.value == 0.0 and lo.ci_low >= 0.0 and lo.ci_high > 0.0


def test_proportion_ci_empty():
    e = proportion_ci(0, 0)
    assert e.n == 0 and math.isnan(e.value)


def test_mean_ci_brackets_mean_and_err():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    e = mean_ci(x)
    assert e.value == 3.0
    assert e.ci_low < 3.0 < e.ci_high
    neg, pos = e.err
    assert math.isclose(neg, pos)  # symmetric for a t-interval
    assert math.isclose(e.value - neg, e.ci_low)


def test_mean_ci_single_and_empty():
    assert mean_ci([7.0]).value == 7.0  # n=1 -> degenerate interval
    assert math.isnan(mean_ci([]).value)


def test_bootstrap_ci_is_deterministic_with_seed():
    x = np.arange(100.0)
    a = bootstrap_ci(x, np.mean, seed=0, n_boot=500)
    b = bootstrap_ci(x, np.mean, seed=0, n_boot=500)
    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)
    assert a.ci_low < a.value < a.ci_high


def test_median_ci_brackets_median():
    x = np.arange(101.0)  # median 50
    e = median_ci(x, seed=1)
    assert e.value == 50.0
    assert e.ci_low <= 50.0 <= e.ci_high
    assert e.method == "median"


def test_estimate_err_asymmetric():
    e = Estimate(value=0.8, ci_low=0.5, ci_high=0.9, n=10, method="x")
    neg, pos = e.err
    assert math.isclose(neg, 0.3) and math.isclose(pos, 0.1)
