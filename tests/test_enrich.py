"""build_enrich: declarative enrich hook (run_stats + meta/opts maps + per_run) joined onto
concatenate_runs. Data-free -- a duck-typed fake session stands in for a parsed one.
"""

from __future__ import annotations

from types import SimpleNamespace

import polars as pl

from piepy.core.enrich import build_enrich


def _run(trial_df, meta):
    return SimpleNamespace(data=SimpleNamespace(data=trial_df), meta=meta)


def _session(base, runs):
    return SimpleNamespace(runs=runs, concatenate_runs=lambda paradigm=None: base)


def test_build_enrich_joins_per_run_columns():
    base = pl.DataFrame(
        {"run_no": [1, 1, 2], "trial": [1, 2, 1]},
        schema_overrides={"run_no": pl.UInt32},
    )
    runs = [
        _run(pl.DataFrame({"a": [1, 2]}), {"level": 3, "opts": {"controller": "foo"}}),
        _run(pl.DataFrame({"a": [9]}), {"level": 5, "opts": {"controller": "bar"}}),
    ]
    enrich = build_enrich(
        "toy",
        run_stats=lambda d: {"n": d.height},
        meta_map={"lvl": "level"},
        opts_map={"task": "controller"},
        per_run=lambda run, d, session: {"derived": d.height * 10},
    )

    out = enrich(_session(base, runs))

    assert out.height == 3  # base rows preserved
    assert {"stat_n", "lvl", "task", "derived"}.issubset(out.columns)

    r1 = out.filter(pl.col("run_no") == 1).row(0, named=True)
    assert (r1["stat_n"], r1["lvl"], r1["task"], r1["derived"]) == (2, 3, "foo", 20)
    r2 = out.filter(pl.col("run_no") == 2).row(0, named=True)
    assert (r2["stat_n"], r2["lvl"], r2["task"], r2["derived"]) == (1, 5, "bar", 10)


def test_build_enrich_skips_empty_runs():
    base = pl.DataFrame({"run_no": [2]}, schema_overrides={"run_no": pl.UInt32})
    runs = [
        _run(pl.DataFrame({"a": []}), {"opts": {}}),  # empty -> skipped
        _run(pl.DataFrame({"a": [7]}), {"opts": {}}),
    ]
    out = build_enrich("toy", run_stats=lambda d: {"n": d.height})(_session(base, runs))
    assert out.height == 1
    assert out.row(0, named=True)["stat_n"] == 1


def test_build_enrich_empty_base_returns_base():
    base = pl.DataFrame({"run_no": []}, schema={"run_no": pl.UInt32})
    out = build_enrich("toy", run_stats=lambda d: {"n": 1})(_session(base, []))
    assert out.is_empty()
