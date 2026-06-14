"""Tests for piepy.fitting (models + fit). Data-free -> CI."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from piepy.fitting import Erf, Logistic, Weibull, fit, get_model


def test_logistic_predict_shape_and_asymptotes():
    m = Logistic()
    p = m.predict([-100, 0, 100], (0.0, 5.0, 0.05, 0.07))
    assert np.isclose(p[0], 0.05, atol=1e-3)  # lower lapse
    assert np.isclose(p[-1], 1 - 0.07, atol=1e-3)  # upper lapse
    assert p[0] < p[1] < p[2]  # monotonic


def test_weibull_uses_absolute_x():
    m = Weibull()
    p = m.predict([-0.5, 0.5], (0.25, 2.0, 0.0, 0.0))
    assert np.isclose(p[0], p[1])  # detection: depends on |contrast|


def test_lsq_recovers_params_on_exact_data():
    m = Logistic()
    true = (0.1, 6.0, 0.03, 0.04)
    x = np.array([-1, -0.5, -0.25, 0, 0.25, 0.5, 1.0])
    y = m.predict(x, true)  # exact (no noise)
    res = fit("logistic", x, y)  # -> least squares
    assert res.method == "lsq"
    for name, t in zip(m.param_names, true):
        assert abs(res.params[name] - t) < 0.05
    assert res.gof["r2"] > 0.999


def test_mle_runs_with_counts_and_reports_loglik():
    m = Logistic()
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    y = m.predict(x, (0.0, 5.0, 0.05, 0.05))
    n = np.full(x.size, 100)
    res = fit("logistic", x, y, n=n)
    assert res.method == "mle"
    assert "log_likelihood" in res.gof
    assert res.param_ci is not None and "x0" in res.param_ci


def test_parameter_recovery_sample_then_fit():
    # the payoff of predict + sample living together: generate from known params, fit them back
    m = Logistic()
    true = np.array([0.0, 5.0, 0.03, 0.03])
    levels = np.array([-1, -0.5, -0.25, -0.125, 0.125, 0.25, 0.5, 1.0])
    x_rep = np.repeat(levels, 400)
    outcomes = m.sample(x_rep, true, rng=0).astype(int)

    df = pl.DataFrame({"x": x_rep, "hit": outcomes})
    agg = (
        df.group_by("x")
        .agg(pl.col("hit").mean().alias("rate"), pl.len().alias("n"))
        .sort("x")
    )
    res = fit("logistic", agg["x"], agg["rate"], n=agg["n"], ci="none")

    assert abs(res.params["x0"] - 0.0) < 0.15
    assert 3.0 < res.params["k"] < 8.0  # true slope 5
    assert res.params["lapse_low"] < 0.12 and res.params["lapse_high"] < 0.12


def test_fitresult_curve_is_plotting_ready():
    res = fit(
        "erf",
        [-1, -0.5, 0, 0.5, 1.0],
        Erf().predict([-1, -0.5, 0, 0.5, 1.0], (0, 0.4, 0.05)),
    )
    xx, yy = res.curve(n=50)
    assert xx.shape == (50,) and yy.shape == (50,)
    assert yy.min() >= 0.0 and yy.max() <= 1.0


def test_sample_is_deterministic_with_seed():
    m = Weibull()
    a = m.sample([0.1, 0.5, 1.0] * 50, (0.25, 2.0, 0.05, 0.05), rng=1)
    b = m.sample([0.1, 0.5, 1.0] * 50, (0.25, 2.0, 0.05, 0.05), rng=1)
    assert np.array_equal(a, b)


def test_get_model_unknown_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        get_model("nope")
