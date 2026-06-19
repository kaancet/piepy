"""The per-paradigm rawdata repair steps must not assume visual channels, so a non-visual
StimPy task (statemachine only, no vstim/screen) runs the base path unmodified.
"""

from __future__ import annotations

import polars as pl

from piepy.core.log_repair_functions import add_total_iStim, extract_trial_count


def test_add_total_istim_is_noop_without_vstim():
    raw = {"statemachine": pl.DataFrame({"trialNo": [1, 2]})}
    out = add_total_iStim(raw)  # was a KeyError on rawdata["vstim"] before the guard
    assert out is raw
    assert "vstim" not in out  # nothing fabricated


def test_extract_trial_count_needs_only_statemachine():
    # multiple trialNos -> skips the re-derivation branch; must not touch vstim/screen
    raw = {"statemachine": pl.DataFrame({"trialNo": [1, 2, 3]})}
    out = extract_trial_count(raw)
    assert set(out) == {"statemachine"}
