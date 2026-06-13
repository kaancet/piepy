"""Tests for the concat-mechanism migration onto align_and_concat.

These cover the two aggregation paths that previously used hand-rolled dtype-reconciliation
loops: ``TaskHub._combine_session_data`` (across sessions) and ``MouseData.append`` (across
an animal's sessions). Data-free, so they run in CI -- this path had no coverage before.
"""

from __future__ import annotations

import datetime

import polars as pl


# --------------------------------------------------------------------------- #
# TaskHub._combine_session_data
# --------------------------------------------------------------------------- #
def test_combine_session_data_unions_sorts_numbers():
    from piepy.core.hub import TaskHub

    # s1 is a later date; s2 is earlier and has an extra column.
    s1 = pl.DataFrame(
        {"date": [datetime.date(2024, 1, 2)], "animalid": ["A"], "run_no": [1], "x": [1]}
    )
    s2 = pl.DataFrame(
        {
            "date": [datetime.date(2024, 1, 1)],
            "animalid": ["A"],
            "run_no": [1],
            "x": [2],
            "extra": [9],
        }
    )
    out = TaskHub._combine_session_data([s1, pl.DataFrame(), s2])

    # total_trial_no is the first column and counts every row
    assert out.columns[0] == "total_trial_no"
    assert out["total_trial_no"].to_list() == [1, 2]
    # sorted ascending by date -> s2 (Jan 1) comes first
    assert out["x"].to_list() == [2, 1]
    # union of columns: 'extra' is present, null for the s1 row
    assert out["extra"].to_list() == [9, None]


def test_combine_session_data_empty_inputs():
    from piepy.core.hub import TaskHub

    assert TaskHub._combine_session_data([None, pl.DataFrame()]).is_empty()


def test_combine_session_data_coerces_supertype():
    from piepy.core.hub import TaskHub

    a = pl.DataFrame(
        {
            "date": [datetime.date(2024, 1, 1)],
            "animalid": ["A"],
            "run_no": [1],
            "v": pl.Series([1], dtype=pl.Int64),
        }
    )
    b = pl.DataFrame(
        {
            "date": [datetime.date(2024, 1, 2)],
            "animalid": ["A"],
            "run_no": [1],
            "v": pl.Series([2.5], dtype=pl.Float64),
        }
    )
    out = TaskHub._combine_session_data([a, b])
    assert out["v"].dtype == pl.Float64
    assert out["v"].to_list() == [1.0, 2.5]


# --------------------------------------------------------------------------- #
# MouseData.append
# --------------------------------------------------------------------------- #
def test_mousedata_append_accumulates_and_unions():
    from piepy.core.mouse import MouseData

    md = MouseData()

    c1 = pl.DataFrame(
        {
            "date": [datetime.date(2024, 1, 1), datetime.date(2024, 1, 1)],
            "trial_no": [1, 2],
            "v": [10, 20],
        }
    )
    md.append([c1], {"date": [datetime.date(2024, 1, 1)], "s": [100]})

    assert md.cumul_data.height == 2
    assert md.summary_data.height == 1
    assert md.cumul_data["cumul_trial_no"].to_list() == [1, 2]

    # second session: a new column 'w' should union in
    c2 = pl.DataFrame(
        {"date": [datetime.date(2024, 1, 2)], "trial_no": [1], "v": [30], "w": [7]}
    )
    md.append([c2], {"date": [datetime.date(2024, 1, 2)], "s": [200]})

    assert md.cumul_data.height == 3
    assert md.summary_data.height == 2
    # running counter is recomputed 1..N after re-sorting by (date, trial_no)
    assert md.cumul_data.sort(["date", "trial_no"])["cumul_trial_no"].to_list() == [
        1,
        2,
        3,
    ]
    # union column w is null for the first session's rows
    assert md.cumul_data.sort(["date", "trial_no"])["w"].to_list() == [None, None, 7]
