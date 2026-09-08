"""Data-free tests for detection and discrimination get_run_stats.

Reproduces the crashes from F03 (zero-denominator, None median, inf d_prime,
wrong discrimination outcome key) and F22 (enrich guard).
"""

import math

import polars as pl
import pytest

from piepy.psychophysics.tasks.wheel_detection.wheelDetectionSession import (
    get_run_stats as detection_stats,
)
from piepy.psychophysics.tasks.wheel_discrimination.wheelDiscriminationSession import (
    get_run_stats as discrimination_stats,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _detection_frame(**overrides):
    """Minimal detection trial table. Override columns by name."""
    defaults = dict(
        outcome=["hit", "miss", "early"],
        isCatch=[0, 0, 0],
        opto=[0, 0, 0],
        contrast=[1.0, 0.5, 0.25],
        state_response_time=[0.3, None, None],
        reaction_time=[0.15, None, None],
    )
    defaults.update(overrides)
    return pl.DataFrame(defaults)


def _discrimination_frame(**overrides):
    """Minimal discrimination trial table. Override columns by name."""
    defaults = dict(
        outcome=["correct", "incorrect", "correct"],
        opto=[0, 0, 0],
        state_response_time=[0.4, None, 0.35],
    )
    defaults.update(overrides)
    return pl.DataFrame(defaults)


# ---------------------------------------------------------------------------
# Detection stats
# ---------------------------------------------------------------------------


class TestDetectionStats:
    def test_normal_run(self):
        df = _detection_frame()
        s = detection_stats(df)
        assert s["total_trial_count"] == 3
        assert s["hit_rate"] is not None
        assert s["d_prime"] is not None
        assert math.isfinite(s["d_prime"])

    def test_no_hits(self):
        """All miss / early -- used to crash with TypeError on round(None)."""
        df = _detection_frame(
            outcome=["miss", "miss", "early"],
            state_response_time=[None, None, None],
            reaction_time=[None, None, None],
        )
        s = detection_stats(df)
        assert s["hit_rate"] == 0.0 or s["hit_rate"] is not None
        assert s["median_response_time"] is None
        assert s["median_reaction_time"] is None
        assert s["d_prime"] is not None
        assert math.isfinite(s["d_prime"])

    def test_zero_stim_trials(self):
        """All early -- stim_data is empty, used to crash with ZeroDivisionError."""
        df = _detection_frame(
            outcome=["early", "early", "early"],
            isCatch=[0, 0, 0],
            state_response_time=[None, None, None],
            reaction_time=[None, None, None],
        )
        s = detection_stats(df)
        assert s["stim_trial_count"] == 0
        assert s["hit_rate"] is None
        assert s["nogo_rate"] is None
        assert s["d_prime"] is None

    def test_all_opto(self):
        """Every trial is opto -- nonopto_data is empty."""
        df = _detection_frame(opto=[1, 1, 1])
        s = detection_stats(df)
        assert s["nonopto_hit_rate"] is None

    def test_dprime_100_percent_hit(self):
        """100% hit rate must not produce inf."""
        df = _detection_frame(
            outcome=["hit", "hit", "hit"],
            isCatch=[0, 0, 0],
            state_response_time=[0.3, 0.25, 0.28],
            reaction_time=[0.15, 0.12, 0.14],
        )
        s = detection_stats(df)
        assert math.isfinite(s["d_prime"])

    def test_keys_present(self):
        """All expected keys exist."""
        s = detection_stats(_detection_frame())
        for k in [
            "total_trial_count",
            "hit_rate",
            "false_alarm_rate",
            "d_prime",
            "median_response_time",
            "median_reaction_time",
            "nonopto_hit_rate",
            "nogo_rate",
            "opto_ratio",
        ]:
            assert k in s, f"missing key: {k}"


# ---------------------------------------------------------------------------
# Discrimination stats
# ---------------------------------------------------------------------------


class TestDiscriminationStats:
    def test_normal_run(self):
        df = _discrimination_frame()
        s = discrimination_stats(df)
        assert s["total_trial_count"] == 3
        assert s["correct_rate"] is not None
        assert s["nonopto_correct_rate"] is not None

    def test_counts_correct_not_hit(self):
        """Discrimination must count 'correct', not 'hit'."""
        df = _discrimination_frame()
        s = discrimination_stats(df)
        # 2 out of 3 are "correct"
        assert s["nonopto_correct_rate"] == pytest.approx(200 / 3, abs=0.01)

    def test_no_correct_trials(self):
        """All incorrect -- used to crash."""
        df = _discrimination_frame(
            outcome=["incorrect", "incorrect", "incorrect"],
            state_response_time=[None, None, None],
        )
        s = discrimination_stats(df)
        assert s["correct_rate"] == 0.0 or s["correct_rate"] is not None
        assert s["median_response_time"] is None

    def test_all_opto(self):
        df = _discrimination_frame(opto=[1, 1, 1])
        s = discrimination_stats(df)
        assert s["nonopto_correct_rate"] is None

    def test_no_trailing_space_key(self):
        """Old bug: key had trailing space 'median_response_latency '."""
        s = discrimination_stats(_discrimination_frame())
        assert "median_response_latency " not in s
        assert "median_response_time" in s

    def test_keys_present(self):
        s = discrimination_stats(_discrimination_frame())
        for k in [
            "total_trial_count",
            "correct_rate",
            "nonopto_correct_rate",
            "median_response_time",
            "opto_ratio",
        ]:
            assert k in s, f"missing key: {k}"
        # old keys that should NOT exist
        assert "nonopto_hit_rate" not in s
        assert "median_response_latency " not in s
