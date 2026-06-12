"""Integration test: Session.concatenate_runs() on a REAL multi-run session.

Marked ``needs_data`` -> skipped in CI and whenever the session isn't present locally.
Uses ``redirect_analysis`` so the re-parse never touches the real analysis dir.
"""

from __future__ import annotations

import polars as pl
import pytest

# a detection session with two real run sub-directories (resolves cleanly in one location)
MULTI_RUN_DETECTION = "240810_KC150_detect__no_cam_KC"


@pytest.mark.needs_data
def test_concatenate_runs_on_real_session(redirect_analysis):
    from conftest import SessionUnavailable, build_session

    try:
        sess = build_session("detection", MULTI_RUN_DETECTION)
    except SessionUnavailable as exc:
        pytest.skip(str(exc))

    if len(sess.runs) < 2:
        pytest.skip(f"{MULTI_RUN_DETECTION} has <2 runs; nothing to concatenate")

    out = sess.concatenate_runs(paradigm="wheel_detection")

    # identity columns present and at the front
    assert out.columns[:4] == ["session_uid", "run_uid", "run_no", "paradigm"]
    assert out["paradigm"].unique().to_list() == ["wheel_detection"]

    # one session_uid, one run_uid per run, run_no in {1, 2}
    assert out["session_uid"].n_unique() == 1
    assert out["run_no"].unique().sort().to_list() == [1, 2]
    assert out["run_uid"].n_unique() == 2

    # row count == sum of per-run rows
    expected_rows = sum(r.data.data.height for r in sess.runs)
    assert out.height == expected_rows

    # session_trial_no is a contiguous 1..N over the whole session
    assert out["session_trial_no"].sort().to_list() == list(range(1, expected_rows + 1))

    # offset math: run 1 offset is 0; run 2 offset is run 1's max t_trialend
    run1_max_end = sess.runs[0].data.data["t_trialend"].max()
    offsets = out.group_by("run_no").agg(pl.col("run_time_offset").first()).sort("run_no")
    assert offsets["run_time_offset"].to_list() == [0, int(run1_max_end)]

    # keep-both: originals untouched, session clock = original + offset
    check = out.with_columns(
        (pl.col("t_trialstart").cast(pl.Int64) + pl.col("run_time_offset")).alias(
            "expected"
        )
    )
    assert check["t_trialstart_session"].to_list() == check["expected"].to_list()
