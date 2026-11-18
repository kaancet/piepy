"""M3: ``save_mat=True`` writes a loadable .mat for every column type; ``Session.save_session``
no longer reads an undefined ``self.save_mat``. Data-free.
"""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import polars as pl
import scipy.io as sio

from piepy.core.run import RunData
from piepy.core.session import Session


def test_save_as_mat_roundtrip(tmp_path):
    rd = RunData()
    rd.set_data(
        pl.DataFrame(
            {
                "trial_no": pl.Series([1, 2, 3], dtype=pl.UInt64),
                "reaction_time": [200.0, None, 150.0],
                "hit": [True, None, False],
                "outcome": ["hit", None, "miss"],
                "date": [dt.date(2025, 6, 18)] * 3,
                "wheel_t": [[0.1, 0.2], None, []],
            }
        )
    )
    rd.save_data(str(tmp_path), save_mat=True)
    m = sio.loadmat(tmp_path / "sessionData.mat")
    assert m["trial_no"].ravel().tolist() == [1, 2, 3]
    assert (
        m["reaction_time"].ravel()[0] == 200.0 and m["reaction_time"].ravel()[1] != m["reaction_time"].ravel()[1]
    )  # NaN
    assert m["wheel_t"].shape[-1] == 3 or m["wheel_t"].size == 3
    assert (tmp_path / "runData.parquet").exists()


def test_save_session_passes_save_mat():
    calls = []
    s = object.__new__(Session)
    s.runs = [SimpleNamespace(save_run=calls.append)]
    s.save_session(save_mat=True)
    s.save_session()
    assert calls == [True, False]
