"""Golden-master tests: parse a real session and assert the trial table is
byte-for-byte stable against a stored snapshot.

These are the safety net for the refactor. Run ``pytest --update-golden`` once
to capture the current behavior, then plain ``pytest`` on every change to catch
any drift in parsed output.

Marked ``needs_data`` -> excluded in CI (``pytest -m "not needs_data"``) and
skipped locally whenever the session is not resolvable via ~/.piepy/config.json.
"""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal


@pytest.mark.needs_data
def test_parse_matches_golden(
    paradigm, session_dir, parsed_session, snapshot_dir, update_golden
):
    runs = parsed_session.runs
    assert runs, f"{paradigm}:{session_dir} parsed to zero runs"

    for i, run in enumerate(runs):
        df = run.data.data
        assert df is not None, f"run {i} has no data"

        snap = snapshot_dir / paradigm / f"{session_dir}__run{i:02d}.parquet"

        if update_golden:
            snap.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(snap)
            continue

        if not snap.exists():
            pytest.skip(f"no golden snapshot at {snap} - run once with --update-golden")

        expected = pl.read_parquet(snap)
        assert_frame_equal(
            df,
            expected,
            check_column_order=True,
            check_dtypes=True,
        )
