"""Tests for SessionLocator (Phase 2, chunk 2).

The layout-rule logic is exercised against synthetic directory trees (data-free -> CI); a
single needs_data test resolves a real session through the configured roots.
"""

from __future__ import annotations

import pytest

from piepy.core.errors import (
    AmbiguousSessionError,
    MalformedSessionError,
    SessionNotFoundError,
)
from piepy.core.paths import SessionLocator

SESSION = "240810_KC150_detect__no_cam_KC"


def _write_run(run_dir, *, n_stim=1, with_pair=True):
    """Create a run directory with n_stim stimlogs (+ matching rig/prot/prefs)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_stim):
        (run_dir / f"run0{i}_x.stimlog").write_text("")
        if with_pair:
            (run_dir / f"run0{i}_x.riglog").write_text("")
            (run_dir / f"run0{i}_x.prot").write_text("")
            (run_dir / f"run0{i}_x.prefs").write_text("")


def _locator(tmp_path):
    pres = tmp_path / "presentation"
    train = tmp_path / "training"
    pres.mkdir()
    train.mkdir()
    return (
        SessionLocator(
            behaviour_roots=[("presentation", str(pres)), ("training", str(train))],
            analysis_roots=[str(tmp_path / "analysis")],
        ),
        pres,
        train,
    )


def test_single_flat_run(tmp_path):
    loc, pres, _ = _locator(tmp_path)
    _write_run(pres / SESSION)  # logs loose in the session dir
    m = loc.locate(SESSION)

    assert m.run_count == 1
    assert m.root_kind == "presentation"
    assert m.runs[0].run_no == 1
    assert m.runs[0].stimlog.endswith(".stimlog")
    assert m.runs[0].riglog.endswith(".riglog")
    # single flat run saves at the session level (no run sub-dir)
    assert m.runs[0].save_dirs == [str(tmp_path / "analysis" / SESSION)]
    assert m.name.animalid == "KC150"


def test_multi_run_subdirs(tmp_path):
    loc, pres, _ = _locator(tmp_path)
    _write_run(pres / SESSION / "run00")
    _write_run(pres / SESSION / "run01")
    m = loc.locate(SESSION)

    assert m.run_count == 2
    assert [r.run_no for r in m.runs] == [1, 2]
    # save dirs mirror the run sub-dir layout
    assert m.runs[0].save_dirs == [str(tmp_path / "analysis" / SESSION / "run00")]
    assert m.runs[1].save_dirs == [str(tmp_path / "analysis" / SESSION / "run01")]


def test_flat_multiple_stimlogs_is_malformed(tmp_path):
    loc, pres, _ = _locator(tmp_path)
    _write_run(pres / SESSION, n_stim=2)  # two loose stimlogs -> the case you outlawed
    with pytest.raises(MalformedSessionError) as exc:
        loc.locate(SESSION)
    assert "run<NN>" in str(exc.value)  # the fix tells them to use sub-dirs


def test_no_stimlog_is_malformed(tmp_path):
    loc, pres, _ = _locator(tmp_path)
    (pres / SESSION).mkdir()  # empty session dir
    with pytest.raises(MalformedSessionError):
        loc.locate(SESSION)


def test_not_found(tmp_path):
    loc, _, _ = _locator(tmp_path)
    with pytest.raises(SessionNotFoundError):
        loc.locate(SESSION)


def test_ambiguous_two_roots(tmp_path):
    loc, pres, train = _locator(tmp_path)
    _write_run(pres / SESSION)
    _write_run(train / SESSION)  # same session in BOTH roots
    with pytest.raises(AmbiguousSessionError):
        loc.locate(SESSION)


@pytest.mark.needs_data
def test_locate_real_session():
    # uses the configured presentation/training roots from ~/.piepy/config.json
    loc = SessionLocator()
    try:
        m = loc.locate(SESSION)
    except SessionNotFoundError as exc:
        pytest.skip(str(exc))
    assert m.run_count == 2  # this session has two real run sub-directories
    for run in m.runs:
        assert run.stimlog and run.riglog
        assert run.run_name
