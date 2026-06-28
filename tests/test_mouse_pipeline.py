"""Integration test for the rebuilt Mouse multi-session pipeline (needs_data).

Mouse.gather_data was stale (it called Session.data/.get_meta/.stats, none of which exist on the
current API). It now composes the Phase-1/2 primitives: SessionLocator -> per-session
concatenate_runs -> align_and_concat across sessions. This runs one animal over a tight date
range so it stays fast; analysis writes go to a temp dir via redirect_analysis.
"""

from __future__ import annotations

import pytest

ANIMAL = "KC150"
DATE_RANGE = ["240810", "240810"]  # a single, clean 2-run detection session


@pytest.mark.needs_data
def test_mouse_gather_data_real_animal(redirect_analysis):
    from piepy.core.mouse import Mouse

    m = Mouse(ANIMAL, paradigm="wheel_detection", dateinterval=DATE_RANGE)
    if m.session_list.is_empty():
        pytest.skip(f"no {ANIMAL} detection sessions in {DATE_RANGE}")

    m.gather_data(load_type="no_load")

    cumul = m.data.cumul_data
    summary = m.data.summary_data
    assert cumul is not None and cumul.height > 0
    assert summary is not None and summary.height >= 1

    # per-trial cumulative table carries the Phase-1 identity + Mouse's session bookkeeping
    assert {
        "session_uid",
        "run_uid",
        "session_no",
        "session_type",
        "cumul_trial_no",
    } <= set(cumul.columns)

    # per-session summary carries the rebuilt fields (from runs[0].meta + stats + session table)
    assert {"date", "level", "task", "sf", "tf", "session_no"} <= set(summary.columns)

    # the two tables agree on how many sessions were aggregated
    assert summary.height == cumul["session_no"].n_unique()
