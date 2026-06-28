"""The canonical data contract for piepy trial tables. Important for:

1. **Stable identity** (`session_uid`, `run_uid`, `run_no`, `paradigm`, ...): who/when/which
   run a trial belongs to, computed deterministically so they're reproducible across
   re-parses and machines (unlike Python's builtin ``hash``).
2. **Schema-aligned concatenation** (`align_and_concat`): one tested code path for stacking
   frames whose columns/dtypes differ, replacing the bespoke reconciliation loops in
   ``hub.gather_sessions`` and ``MouseData.append``.
3. **Within-session run concatenation** (`concat_session_runs`): stacks a session's runs onto
   a single session-wide clock while keeping the original per-run times intact.

Design note: this module is **additive**. It does not change how individual trials are
parsed or validated (that stays with the patito ``Trial`` models). It formalizes the layer
*above* the per-trial schema (identity + how frames combine).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import polars as pl

__all__ = [
    "SchemaContractError",
    "UID_MODULUS",
    "IDENTITY_COLUMNS",
    "PURE_TIME_LIST_COLUMNS",
    "TIME_AT_INDEX0_LIST_COLUMNS",
    "stable_uid",
    "session_uid",
    "run_uid",
    "attach_run_identity",
    "align_and_concat",
    "concat_session_runs",
]


class SchemaContractError(Exception):
    """Raised when frames cannot be reconciled into one trial table."""


# 10**18 < UInt64 max (~1.8e19); leaves ample headroom while keeping collisions negligible.
UID_MODULUS = 10**18

# Canonical identity columns, in the order they should appear at the front of a trial table.
# (dtype is what we cast to when we create them; columns parsed elsewhere keep their dtype.)
IDENTITY_COLUMNS: dict[str, pl.DataType] = {
    # "session_uid": pl.UInt64,
    # "run_uid": pl.UInt64,
    # "run_no": pl.UInt32,
    # "paradigm": pl.Utf8,
    "animalid": pl.Utf8,
    "baredate": pl.Utf8,
    "date": pl.Date,
}

# List-valued columns that hold ABSOLUTE event times (run-clock ms) end to end -> every
# element is offset onto the session clock.
PURE_TIME_LIST_COLUMNS: tuple[str, ...] = ("wheel_t", "lick", "opto_pulse")

# List-valued columns whose element 0 is an absolute time and the rest are NOT times.
# ``reward`` is stored as ``[reward_time, reward_value]`` -- only element 0 may be offset.
TIME_AT_INDEX0_LIST_COLUMNS: tuple[str, ...] = ("reward",)


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #
def stable_uid(*parts: object, modulus: int = UID_MODULUS) -> int:
    """A deterministic unsigned id from the given parts.

    Uses SHA-256 (stable across processes/machines, unlike ``hash``), then reduces to an
    int that fits a ``UInt64`` column.
    """
    combined = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    return int(digest, 16) % modulus


def session_uid(sessiondir: str) -> int:
    """Stable id for a session, derived from the full session directory name.

    Using the full ``sessiondir`` (not just date+animal) means two same-day sessions of the
    same animal -- e.g. ``..._ISI`` and ``..._ISI_ket`` -- do not collide.
    """
    return stable_uid(sessiondir)


def run_uid(sessiondir: str, run_no: int, run_name: str) -> int:
    """Stable id for a single run within a session."""
    return stable_uid(sessiondir, run_no, run_name)


def attach_run_identity(
    df: pl.DataFrame,
    *,
    sessiondir: str,
    run_no: int,
    run_name: str,
    paradigm: str | None,
    animalid: str | None = None,
    baredate: str | None = None,
) -> pl.DataFrame:
    """Prepend the canonical identity columns to a single run's trial table.

    ``animalid``/``baredate``/``date`` are only added if absent (``RunData`` already adds
    them during parsing); the rest are always (re)computed. Identity columns are moved to
    the front in :data:`IDENTITY_COLUMNS` order.
    """
    out = df.with_columns(
        pl.lit(session_uid(sessiondir)).cast(pl.UInt64).alias("session_uid"),
        pl.lit(run_uid(sessiondir, run_no, run_name)).cast(pl.UInt64).alias("run_uid"),
        pl.lit(run_no).cast(pl.UInt32).alias("run_no"),
        pl.lit(paradigm).cast(pl.Utf8).alias("paradigm"),
    )

    if "animalid" not in out.columns and animalid is not None:
        out = out.with_columns(pl.lit(animalid).cast(pl.Utf8).alias("animalid"))
    if "baredate" not in out.columns and baredate is not None:
        out = out.with_columns(pl.lit(baredate).cast(pl.Utf8).alias("baredate"))
    if "date" not in out.columns and "baredate" in out.columns:
        out = out.with_columns(pl.col("baredate").str.strptime(pl.Date, "%y%m%d", strict=False).alias("date"))

    front = [c for c in IDENTITY_COLUMNS if c in out.columns]
    rest = [c for c in out.columns if c not in front]
    return out.select(front + rest)


# --------------------------------------------------------------------------- #
# Schema-aligned concatenation
# --------------------------------------------------------------------------- #
def align_and_concat(frames: Sequence[pl.DataFrame | None]) -> pl.DataFrame:
    """Vertically concatenate frames that may differ in columns and/or dtypes.

    - ``None`` and zero-column frames are dropped.
    - Columns are unioned by name in first-seen order; columns missing from a frame are
      filled with nulls.
    - Differing dtypes for the same column are coerced to their common supertype.

    Returns an empty ``DataFrame`` if nothing concatenable is given. Raises
    :class:`SchemaContractError` if polars cannot reconcile the frames.

    This is the single replacement for the hand-rolled dtype-reconciliation loops in
    ``hub.gather_sessions`` and ``MouseData.append``.
    """
    real = [f for f in frames if f is not None and f.width > 0]
    if not real:
        return pl.DataFrame()
    if len(real) == 1:
        return real[0]

    ordered_cols: list[str] = []
    for f in real:
        for c in f.columns:
            if c not in ordered_cols:
                ordered_cols.append(c)

    try:
        # diagonal_relaxed: union columns + null-fill missing + coerce to supertypes.
        out = pl.concat(real, how="diagonal_relaxed")
    except Exception as exc:  # noqa: BLE001 - re-raised as a domain error with context
        raise SchemaContractError(f"Could not align {len(real)} frames into one table: {exc}") from exc

    return out.select(ordered_cols)


# --------------------------------------------------------------------------- #
# Within-session run concatenation (session clock)
# --------------------------------------------------------------------------- #
def _scalar_time_columns(df: pl.DataFrame) -> list[str]:
    """Absolute scalar time columns: numeric and named with the ``t_`` convention.

    This correctly excludes durations/diffs such as ``reaction_time``, ``peak_speed`` and
    ``vstim_time_diff`` (none start with ``t_``).
    """
    return [c for c in df.columns if c.startswith("t_") and df.schema[c].is_numeric()]


def _session_clock_exprs(
    df: pl.DataFrame,
    offset: int,
    pure_time_list_columns: Sequence[str],
    time_at_index0_list_columns: Sequence[str],
) -> list[pl.Expr]:
    """Build the ``<col>_session`` expressions for one run given its ``offset`` (ms)."""
    exprs: list[pl.Expr] = [pl.lit(offset).cast(pl.Int64).alias("run_time_offset")]

    for c in _scalar_time_columns(df):
        exprs.append((pl.col(c).cast(pl.Int64) + offset).alias(f"{c}_session"))

    for c in pure_time_list_columns:
        if c in df.columns:
            exprs.append(pl.col(c).list.eval(pl.element() + offset).alias(f"{c}_session"))

    for c in time_at_index0_list_columns:
        if c in df.columns:
            exprs.append(
                pl.col(c)
                .list.eval(pl.when(pl.int_range(0, pl.len()) == 0).then(pl.element() + offset).otherwise(pl.element()))
                .alias(f"{c}_session")
            )

    return exprs


def concat_session_runs(
    run_frames: Sequence[pl.DataFrame],
    *,
    end_time_column: str = "t_trialend",
    pure_time_list_columns: Sequence[str] = PURE_TIME_LIST_COLUMNS,
    time_at_index0_list_columns: Sequence[str] = TIME_AT_INDEX0_LIST_COLUMNS,
) -> pl.DataFrame:
    """Concatenate the per-run trial tables of ONE session onto a session-wide clock.

    Time policy is **keep both**: the original per-run time columns are never modified.
    For each run we add:

    - ``run_time_offset``  : the offset (ms) applied to reach the session clock.
    - ``<col>_session``    : session-clock copies of every absolute-time column -- the scalar
      ``t_*`` columns, the pure-time list columns, and element 0 of the
      ``[time, value]`` list columns (e.g. ``reward``).
    - ``session_trial_no`` : 1-based cumulative trial index across the whole session.

    Offsets are cumulative: run ``k`` starts after run ``k-1``'s last ``end_time_column``.
    ``run_frames`` must be in run order and already carry identity columns.
    """
    frames = [f for f in run_frames if f is not None and f.width > 0]
    if not frames:
        return pl.DataFrame()

    # cumulative offsets: offset[k] = sum of prior runs' max end-time.
    offsets: list[int] = []
    cumulative = 0
    for f in frames:
        offsets.append(cumulative)
        if end_time_column in f.columns and f.height:
            run_end = f[end_time_column].max()
            cumulative += int(run_end) if run_end is not None else 0

    adjusted: list[pl.DataFrame] = []
    trials_so_far = 0
    for f, off in zip(frames, offsets):
        f2 = f.with_columns(_session_clock_exprs(f, off, pure_time_list_columns, time_at_index0_list_columns))
        f2 = f2.with_columns((pl.int_range(1, pl.len() + 1, dtype=pl.UInt64) + trials_so_far).alias("session_trial_no"))
        trials_so_far += f2.height
        adjusted.append(f2)

    return align_and_concat(adjusted)
