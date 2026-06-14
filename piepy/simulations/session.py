"""Simulate a parsed-session trial table.

``simulate_session`` produces a polars DataFrame that *looks like* the output of parsing a real
session -- the analysis-relevant columns of the detection / discrimination trial table, with the
canonical identity columns stamped on (so it drops straight into ``piepy.stats.aggregate``,
plotting, fitting, etc.). It is **not** a byte-exact replica of every internal parser column;
it's a controllable stand-in for trying modelling approaches and testing the analysis stack.

You drive it with the knobs you'd actually want to vary::

    simulate_session(paradigm="detection", n_trials=600, opto_ratio=0.3,
                     rt_mean=320, rt_std=130, seed=0)

The hit/choice rate can be a constant, a per-contrast mapping, a callable, or (default) a
built-in psychometric, so the simulated psychometric curve is realistic and fittable.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import polars as pl

from ..core.schema import attach_run_identity

__all__ = ["simulate_session"]

# default stimulus contrasts (the catch/0 condition is generated separately via catch_ratio)
_DEFAULT_CONTRASTS = (0.0625, 0.125, 0.25, 0.5, 1.0)


def _resolve_rate(
    rate: float | Mapping | Callable | None,
    *,
    lapse: float,
    slope: float,
    threshold: float,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a function abs_contrast -> probability.

    None -> a logistic psychometric in |contrast|; a float -> constant; a mapping -> lookup by
    |contrast|; a callable -> used directly (vectorized over |contrast|).
    """
    if rate is None:
        return lambda c: lapse + (1 - 2 * lapse) / (1 + np.exp(-slope * (c - threshold)))
    if callable(rate):
        return lambda c: np.asarray(rate(c), dtype=float)
    if isinstance(rate, Mapping):
        return lambda c: np.array([rate.get(float(ci), 0.5) for ci in c], dtype=float)
    return lambda c: np.full(np.shape(c), float(rate), dtype=float)


def _gamma_rt(n: int, mean: float, std: float, rng: np.random.Generator) -> np.ndarray:
    """Right-skewed reaction times (ms) with the requested mean/std via a gamma."""
    if mean <= 0 or std <= 0:
        return np.full(n, float(mean))
    shape = (mean / std) ** 2
    scale = std**2 / mean
    return rng.gamma(shape, scale, n)


def _time_axis(n: int, rng: np.random.Generator):
    """Plausible, monotonically increasing per-trial timestamps (ms)."""
    durations = rng.integers(3000, 8000, n)
    t_trialstart = 1000 + np.cumsum(durations) - durations  # first starts at 1000
    blank = rng.integers(400, 1000, n)
    t_vstimstart = t_trialstart + blank
    t_vstimend = t_vstimstart + 1000  # response window
    t_trialend = t_vstimend + rng.integers(200, 800, n)
    return (
        t_trialstart.astype(np.uint64),
        t_vstimstart.astype(np.uint64),
        t_vstimend.astype(np.uint64),
        t_trialend.astype(np.uint64),
    )


def _event_lists(responded: np.ndarray, t_resp_abs: np.ndarray, reward_size: float):
    """Per-trial reward [time, value] and lick [times] lists (empty when no response)."""
    reward, lick = [], []
    for ok, t in zip(responded, t_resp_abs):
        if ok and np.isfinite(t):
            reward.append([float(t), float(reward_size)])
            lick.append([float(t) - 20.0, float(t) + 90.0])
        else:
            reward.append([])
            lick.append([])
    return reward, lick


def simulate_session(
    *,
    paradigm: str = "detection",
    n_trials: int = 500,
    contrasts: tuple[float, ...] | list[float] | None = None,
    hit_rate: float | Mapping | Callable | None = None,
    lapse: float = 0.05,
    slope: float = 10.0,
    threshold: float = 0.15,
    rt_mean: float = 300.0,
    rt_std: float = 120.0,
    opto_ratio: float = 0.0,
    opto_rt_shift: float = 80.0,
    opto_rate_factor: float = 1.0,
    early_rate: float = 0.1,
    catch_ratio: float = 0.1,
    sf: float = 0.1,
    tf: float = 4.0,
    reward_size: float = 30.0,
    animalid: str = "SM000",
    baredate: str = "240101",
    run_no: int = 1,
    seed: int | None = None,
) -> pl.DataFrame:
    """Generate a synthetic parsed-session trial table.

    Args:
        paradigm: "detection" or "discrimination".
        n_trials: number of trials.
        contrasts: stimulus contrasts to draw from (the 0/catch condition is added separately).
        hit_rate: success probability vs |contrast| -- constant float, ``{contrast: p}`` mapping,
            callable ``f(|contrast|)->p``, or None for a built-in logistic psychometric
            (controlled by ``lapse``/``slope``/``threshold``).
        rt_mean / rt_std: reaction-time distribution (ms; gamma, right-skewed).
        opto_ratio: fraction of opto trials; ``opto_rt_shift`` slows their RT and
            ``opto_rate_factor`` scales their success probability (1.0 = no effect).
        early_rate / catch_ratio: fraction of early (impulsive) and catch (no-stim) trials.
        sf / tf / reward_size / animalid / baredate / run_no / seed: session knobs.

    Returns:
        pl.DataFrame with the canonical identity columns plus the paradigm's analysis-relevant
        trial columns (one row per trial).
    """
    rng = np.random.default_rng(seed)
    contrasts = tuple(_DEFAULT_CONTRASTS if contrasts is None else contrasts)
    prob = _resolve_rate(hit_rate, lapse=lapse, slope=slope, threshold=threshold)

    if paradigm == "detection":
        df = _simulate_detection(
            n_trials,
            contrasts,
            prob,
            rt_mean,
            rt_std,
            opto_ratio,
            opto_rt_shift,
            opto_rate_factor,
            early_rate,
            catch_ratio,
            sf,
            tf,
            reward_size,
            rng,
        )
    elif paradigm == "discrimination":
        df = _simulate_discrimination(
            n_trials,
            contrasts,
            prob,
            rt_mean,
            rt_std,
            opto_ratio,
            opto_rt_shift,
            opto_rate_factor,
            early_rate,
            sf,
            tf,
            reward_size,
            rng,
        )
    else:
        raise ValueError(
            f"Unknown paradigm {paradigm!r}; use 'detection' or 'discrimination'."
        )

    sessiondir = f"{baredate}_{animalid}_{paradigm}__no_cam_SIM"
    return attach_run_identity(
        df,
        sessiondir=sessiondir,
        run_no=run_no,
        run_name=f"run{run_no:02d}_sim",
        paradigm=paradigm,
        animalid=animalid,
        baredate=baredate,
    )


def _simulate_detection(
    n,
    contrasts,
    prob,
    rt_mean,
    rt_std,
    opto_ratio,
    opto_rt_shift,
    opto_rate_factor,
    early_rate,
    catch_ratio,
    sf,
    tf,
    reward_size,
    rng,
) -> pl.DataFrame:
    # trial types
    is_early = rng.random(n) < early_rate
    is_catch = (~is_early) & (rng.random(n) < catch_ratio)
    is_stim = ~is_early & ~is_catch

    contrast = np.where(is_stim, rng.choice(contrasts, n), 0.0)
    side = rng.choice([-1, 1], n)  # +1 contra, -1 ipsi
    opto = rng.random(n) < opto_ratio

    p = prob(contrast) * np.where(opto, opto_rate_factor, 1.0)
    hit = is_stim & (rng.random(n) < p)

    outcome = np.where(
        is_early, "early", np.where(is_catch, "catch", np.where(hit, "hit", "miss"))
    )
    state_outcome = np.where(is_early, -1, np.where(hit, 1, 0)).astype(np.int64)

    # times
    t_trialstart, t_vstimstart, t_vstimend, t_trialend = _time_axis(n, rng)
    rt = _gamma_rt(n, rt_mean, rt_std, rng) + np.where(opto, opto_rt_shift, 0.0)
    # state_response_time: defined for all (timeout for miss, short for early)
    state_rt = np.where(hit, rt, np.where(is_early, rng.uniform(50, 150, n), 1000.0))
    reaction = np.where(hit, rt, np.nan)  # null for non-hits (set below)
    t_resp_abs = t_vstimstart.astype(float) + np.where(hit, rt, np.nan)

    reward, lick = _event_lists(hit, t_resp_abs, reward_size)
    signed = contrast * side
    stim_side = np.where(is_stim, np.where(side > 0, "contra", "ipsi"), "catch")
    contrast_type = np.where(
        contrast == 0, "catch", np.where(contrast >= 0.25, "easy", "hard")
    )
    stim_type = f"{round(sf, 2)}cpd_{tf}Hz"

    df = pl.DataFrame(
        {
            "trial_no": np.arange(1, n + 1, dtype=np.uint64),
            "t_trialstart": t_trialstart,
            "t_vstimstart": t_vstimstart,
            "t_vstimend": t_vstimend,
            "t_trialend": t_trialend,
            "outcome": outcome,
            "state_outcome": state_outcome,
            "isCatch": is_catch,
            "contrast": contrast.astype(np.float64),
            "signed_contrast": signed.astype(np.float64),
            "stim_side": stim_side,
            "stim_pos": np.where(is_stim, side, 0).astype(np.int64),
            "sf": np.full(n, round(sf, 2)),
            "tf": np.full(n, float(tf)),
            "stim_type": np.full(n, stim_type),
            "contrast_type": contrast_type,
            "opto": opto,
            "opto_pattern": np.where(opto, 0, -1).astype(np.int64),
            "opto_region": pl.Series(
                ["sim_region" if o else None for o in opto], dtype=pl.Utf8
            ),
            "reaction_time": reaction,
            "response_time": state_rt.astype(np.float64),
            "state_response_time": state_rt.astype(np.float64),
            "reward": reward,
            "lick": lick,
            "stimkey": np.array([f"{stim_type}_{op}" for op in np.where(opto, 0, -1)]),
            "stim_label": np.full(n, stim_type),
        }
    )
    return df.with_columns(pl.col("reaction_time").fill_nan(None))


def _simulate_discrimination(
    n,
    contrasts,
    prob,
    rt_mean,
    rt_std,
    opto_ratio,
    opto_rt_shift,
    opto_rate_factor,
    early_rate,
    sf,
    tf,
    reward_size,
    rng,
) -> pl.DataFrame:
    is_early = rng.random(n) < early_rate
    is_trial = ~is_early

    contrast = rng.choice(contrasts, n)
    side = rng.choice([-1, 1], n)  # which side the target (stronger stimulus) is on
    signed = contrast * side
    opto = rng.random(n) < opto_ratio

    # probability of choosing the correct side rises with |contrast|
    p_correct = prob(contrast) * np.where(opto, opto_rate_factor, 1.0)
    correct = is_trial & (rng.random(n) < p_correct)
    # right_choice: target on right (+1) and correct, or target left and incorrect
    chose_right = np.where(side > 0, correct, ~correct) & is_trial

    outcome = np.where(is_early, "early", np.where(correct, "correct", "incorrect"))
    state_outcome = np.where(is_early, -1, correct.astype(int)).astype(np.int64)

    t_trialstart, t_vstimstart, t_vstimend, t_trialend = _time_axis(n, rng)
    rt = _gamma_rt(n, rt_mean, rt_std, rng) + np.where(opto, opto_rt_shift, 0.0)
    state_rt = np.where(is_early, rng.uniform(50, 150, n), rt)
    reaction = np.where(is_trial, rt, np.nan)
    t_resp_abs = t_vstimstart.astype(float) + np.where(is_trial, rt, np.nan)

    reward, lick = _event_lists(correct, t_resp_abs, reward_size)
    target_side = np.where(side > 0, "right", "left")
    stim_type = f"{round(sf, 2)}cpd_{tf}Hz"

    df = pl.DataFrame(
        {
            "trial_no": np.arange(1, n + 1, dtype=np.uint64),
            "t_trialstart": t_trialstart,
            "t_vstimstart": t_vstimstart,
            "t_vstimend": t_vstimend,
            "t_trialend": t_trialend,
            "outcome": outcome,
            "state_outcome": state_outcome,
            "signed_contrast": signed.astype(np.float64),
            "target_contrast": contrast.astype(np.float64),
            "target_side": target_side,
            "right_choice": pl.Series(
                [int(cr) if t else None for cr, t in zip(chose_right, is_trial)],
                dtype=pl.Int64,
            ),
            "sf": np.full(n, round(sf, 2)),
            "tf": np.full(n, float(tf)),
            "stim_type": np.full(n, stim_type),
            "opto": opto,
            "opto_pattern": np.where(opto, 0, -1).astype(np.int64),
            "opto_region": pl.Series(
                ["sim_region" if o else None for o in opto], dtype=pl.Utf8
            ),
            "reaction_time": reaction,
            "response_time": state_rt.astype(np.float64),
            "state_response_time": state_rt.astype(np.float64),
            "reward": reward,
            "lick": lick,
            "stimkey": np.array([f"{stim_type}_{op}" for op in np.where(opto, 0, -1)]),
            "stim_label": np.full(n, stim_type),
        }
    )
    return df.with_columns(pl.col("reaction_time").fill_nan(None))
