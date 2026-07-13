"""Tests for piepy.simulations.simulate_session. Data-free -> CI."""

from __future__ import annotations

import polars as pl
import pytest

from piepy.simulations import simulate_session
from piepy.stats import Rate, aggregate, group_arrays


def test_detection_shape_and_identity():
    df = simulate_session(paradigm="wheel_detection", n_trials=400, seed=0)
    assert df.height == 400
    # canonical identity is stamped on
    for c in [
        "session_uid",
        "run_uid",
        "run_no",
        "paradigm",
        "animalid",
        "baredate",
        "date",
    ]:
        assert c in df.columns
    assert df["paradigm"].unique().to_list() == ["wheel_detection"]
    # outcomes look like a detection session
    assert set(df["outcome"].unique()) <= {"hit", "miss", "early", "catch"}


def test_detection_reaction_time_null_for_non_hits():
    df = simulate_session(paradigm="wheel_detection", n_trials=500, seed=1)
    n_hit = df.filter(pl.col("outcome") == "hit").height
    # reaction_time is set only for hits (null elsewhere), like a real parse
    assert df["reaction_time"].drop_nulls().len() == n_hit


def test_opto_ratio_is_respected():
    df = simulate_session(
        paradigm="wheel_detection", n_trials=4000, opto_ratio=0.3, seed=2
    )
    frac = df["opto"].mean()
    assert 0.26 < frac < 0.34  # ~0.3


def test_default_psychometric_is_monotonic():
    df = simulate_session(
        paradigm="wheel_detection", n_trials=6000, catch_ratio=0.0, early_rate=0.0, seed=3
    )
    rate = aggregate(df, group="contrast", metrics=[Rate("outcome", rate_of="hit")]).sort(
        "contrast"
    )
    vals = rate["value"].to_list()
    assert vals[0] < vals[-1]  # hit rate rises with contrast


def test_is_a_drop_in_for_aggregate():
    df = simulate_session(
        paradigm="wheel_detection", n_trials=600, opto_ratio=0.4, seed=4
    )
    # the exact psychometric-plot call from the notebook works unchanged
    psych = aggregate(
        df, group="signed_contrast", metrics=[Rate("outcome", rate_of="hit")]
    )
    assert {"value", "ci_low", "ci_high", "n"} <= set(psych.columns)
    # and the compare bridge
    arrs = group_arrays(df, group="opto", value="reaction_time")
    assert set(arrs) <= {True, False}


def test_determinism_with_seed():
    a = simulate_session(n_trials=200, seed=7)
    b = simulate_session(n_trials=200, seed=7)
    assert a.equals(b)


@pytest.mark.parametrize(
    "rate", [0.8, {0.25: 0.6, 0.5: 0.9}, lambda c: 0.5 + 0.4 * (c > 0.2)]
)
def test_hit_rate_accepts_scalar_mapping_callable(rate):
    df = simulate_session(paradigm="wheel_detection", n_trials=300, hit_rate=rate, seed=0)
    assert df.height == 300
    assert df.filter(pl.col("outcome") == "hit").height > 0


def test_discrimination():
    df = simulate_session(paradigm="wheel_discrimination", n_trials=400, seed=0)
    assert set(df["outcome"].unique()) <= {"correct", "incorrect", "early"}
    assert "right_choice" in df.columns and "target_side" in df.columns
    assert df["paradigm"].unique().to_list() == ["wheel_discrimination"]


def test_unknown_paradigm_raises():
    with pytest.raises(ValueError, match="Unknown paradigm"):
        simulate_session(paradigm="nope")
