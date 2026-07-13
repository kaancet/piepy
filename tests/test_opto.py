"""OptoPatternMixin: shared opto silencing-pattern columns + structured error handling.
Data-free -- a tiny RunData-like holder stands in for a parsed table.
"""

from __future__ import annotations

import polars as pl
import pytest

from piepy.core.errors import OptoPatternError
from piepy.psychophysics.opto import add_opto_pattern_columns


def test_non_opto_session_gets_placeholder_columns():
    df = pl.DataFrame({"opto": [-1, -1], "stim_type": ["a", "a"]})

    df = add_opto_pattern_columns(
        df, None
    )  # no pattern dir needed for a non-opto session
    assert df["opto_region"].to_list() == [None, None]
    assert df["stimkey"].to_list() == ["a_-1", "a_-1"]
    assert df["stim_label"].to_list() == ["a", "a"]


def test_opto_session_without_pattern_dir_raises_structured_error():
    df = pl.DataFrame(
        {
            "opto": [0, 1],
            "stim_type": ["a", "a"],
            "opto_pattern": [-1, 0],
            "state_outcome": [1, 1],
        }
    )
    with pytest.raises(OptoPatternError) as exc:
        df = add_opto_pattern_columns(df, None)
    msg = str(exc.value)
    # structured PiepyError rendering: problem + actionable fix
    assert "opto" in msg
    assert "fix:" in msg
    # it is a piepy error (caught by `except Exception`), parsing category
    from piepy.core.errors import ParsingError, PiepyError

    assert isinstance(exc.value, (ParsingError, PiepyError))
