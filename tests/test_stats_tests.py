"""Unit tests for piepy.stats.tests (compare + 2-D tests). Data-free -> CI."""

from __future__ import annotations

import math

import numpy as np
import pytest

from piepy.stats import compare, compare_by_x, energy_2d, ks_2d


def test_compare_mannu_detects_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 200)
    b = rng.normal(1.5, 1, 200)
    res = compare(a, b, method="mannu")
    assert res.method == "mannu"
    assert res.n1 == 200 and res.n2 == 200
    assert res.pvalue < 1e-10
    assert res.significant
    assert res.effect_size is not None and abs(res.effect_size) > 0.4  # rank-biserial


def test_compare_no_difference_is_not_significant():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 300)
    b = rng.normal(0, 1, 300)
    assert not compare(a, b, method="mannu").significant


def test_compare_ttest_and_welch_effect_is_cohens_d():
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, 500)
    b = rng.normal(0.8, 1, 500)
    for method in ("ttest", "welch"):
        res = compare(a, b, method=method)
        assert res.pvalue < 1e-10
        assert 0.6 < abs(res.effect_size) < 1.0  # d ~ 0.8


def test_compare_paired_requires_equal_length():
    with pytest.raises(ValueError, match="paired"):
        compare([1, 2, 3], [1, 2], method="wilcoxon")


def test_compare_paired_wilcoxon():
    rng = np.random.default_rng(3)
    base = rng.normal(10, 2, 100)
    after = base + rng.normal(1.0, 0.5, 100)  # consistent within-pair increase
    res = compare(base, after, method="ttest_paired")
    assert res.n1 == res.n2 == 100
    assert res.pvalue < 1e-10
    assert res.effect_size < 0  # base < after


def test_compare_drops_nans():
    res = compare([1.0, 2.0, np.nan, 4.0], [10.0, np.nan, 30.0], method="mannu")
    assert res.n1 == 3 and res.n2 == 2


def test_compare_empty_is_nan():
    res = compare([], [1, 2, 3], method="mannu")
    assert res.n1 == 0 and math.isnan(res.pvalue)


def test_compare_unknown_method():
    with pytest.raises(ValueError, match="Unknown method"):
        compare([1, 2], [3, 4], method="nope")


def test_ks_2d_and_energy_2d_return_testresult():
    rng = np.random.default_rng(4)
    a = rng.normal(0, 1, (150, 2))
    b = rng.normal(1.2, 1, (150, 2))
    ks = ks_2d(a, b)
    en = energy_2d(a, b, n_boot=200)
    assert ks.method == "ks_2d" and 0.0 <= ks.pvalue <= 1.0
    assert en.method == "energy_2d" and 0.0 <= en.pvalue <= 1.0
    # clearly different distributions -> small p
    assert ks.pvalue < 0.05
