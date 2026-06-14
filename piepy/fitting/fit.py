"""Fit a psychometric model to data.

``fit(model, x, y, n=...)`` returns a :class:`FitResult` with named parameters, confidence
intervals, goodness-of-fit, and a ``curve()`` ready to hand to a plotter.

Method is automatic: if per-level trial counts ``n`` are given, it does a **binomial MLE**
(statistically correct for proportions); otherwise it does **least-squares** on the rates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats as sps
from scipy.optimize import curve_fit, minimize

from .models import Model, get_model

__all__ = ["FitResult", "fit"]

_EPS = 1e-9


@dataclass
class FitResult:
    """A fitted model: parameters + uncertainty + goodness-of-fit, with a ready-to-plot curve."""

    model: Model
    params: dict[str, float]
    param_ci: dict[str, tuple[float, float]] | None
    gof: dict[str, float]
    method: str
    x: np.ndarray
    y: np.ndarray

    def _param_array(self) -> np.ndarray:
        return np.array([self.params[name] for name in self.model.param_names])

    def predict(self, x: ArrayLike) -> np.ndarray:
        """Model prediction at ``x`` using the fitted parameters."""
        return self.model.predict(x, self._param_array())

    def curve(self, n: int = 200, x_range: tuple[float, float] | None = None):
        """A dense ``(x, y)`` curve over the data range (or ``x_range``) for plotting."""
        lo, hi = x_range or (float(np.min(self.x)), float(np.max(self.x)))
        xx = np.linspace(lo, hi, n)
        return xx, self.predict(xx)


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _neg_loglik(model: Model, params, x, k, n) -> float:
    p = np.clip(model.predict(x, params), _EPS, 1 - _EPS)
    return -float(np.sum(k * np.log(p) + (n - k) * np.log(1 - p)))


def fit(
    model: str | Model,
    x: ArrayLike,
    y: ArrayLike,
    *,
    n: ArrayLike | None = None,
    method: str = "auto",
    ci: str = "auto",
    confidence: float = 0.95,
    n_boot: int = 500,
    seed: int | None = None,
    maxfev: int = 10000,
) -> FitResult:
    """Fit ``model`` to ``(x, y)``.

    Args:
        model: a :class:`Model` instance or a registered name ("logistic"/"weibull"/"erf").
        x: stimulus levels.
        y: observed probability/rate at each level (0..1).
        n: per-level trial counts. If given, enables binomial MLE (and bootstrap CIs).
        method: "auto" (MLE when ``n`` is given, else LSQ), "lsq", or "mle".
        ci: "auto" (covariance for LSQ, bootstrap for MLE), "covariance", "bootstrap", "none".
        confidence / n_boot / seed / maxfev: CI level, bootstrap resamples, RNG seed, optimizer iters.

    Returns:
        FitResult with named ``params``, ``param_ci``, ``gof`` (r2, plus log_likelihood for MLE),
        and ``predict``/``curve`` helpers.
    """
    model = get_model(model)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]
    counts = None if n is None else np.asarray(n, dtype=float)[keep]

    if method == "auto":
        method = "mle" if counts is not None else "lsq"
    if method == "mle" and counts is None:
        raise ValueError("method='mle' needs per-level trial counts via n=...")

    lower, upper = model.bounds(x, y)
    p0 = model.guess(x, y)

    if method == "lsq":
        popt, pcov = curve_fit(
            lambda xx, *p: model.predict(xx, p),
            x,
            y,
            p0=p0,
            bounds=(lower, upper),
            maxfev=maxfev,
        )
        param_ci = (
            _covariance_ci(model, popt, pcov, confidence)
            if ci in ("auto", "covariance")
            else None
        )
    else:  # mle
        k = np.round(y * counts)
        res = minimize(
            lambda p: _neg_loglik(model, p, x, k, counts),
            p0,
            bounds=list(zip(lower, upper)),
            method="L-BFGS-B",
        )
        popt = res.x
        param_ci = None
        if ci in ("auto", "bootstrap"):
            param_ci = _bootstrap_ci(
                model, popt, x, counts, lower, upper, confidence, n_boot, seed
            )
        elif ci == "covariance":
            raise ValueError("ci='covariance' is only available for least-squares fits.")

    if ci == "none":
        param_ci = None

    params = dict(zip(model.param_names, (float(v) for v in popt)))
    gof = {"r2": _r2(y, model.predict(x, popt)), "n_points": int(x.size)}
    if method == "mle":
        gof["log_likelihood"] = -_neg_loglik(model, popt, x, np.round(y * counts), counts)

    return FitResult(model, params, param_ci, gof, method, x, y)


def _covariance_ci(model, popt, pcov, confidence) -> dict[str, tuple[float, float]]:
    z = float(sps.norm.ppf(1 - (1 - confidence) / 2))
    se = np.sqrt(np.diag(pcov))
    return {
        name: (float(popt[i] - z * se[i]), float(popt[i] + z * se[i]))
        for i, name in enumerate(model.param_names)
    }


def _bootstrap_ci(model, popt, x, counts, lower, upper, confidence, n_boot, seed):
    rng = np.random.default_rng(seed)
    p_hat = np.clip(model.predict(x, popt), 0.0, 1.0)
    n_int = counts.astype(int)
    boots = np.empty((n_boot, model.n_params))
    for b in range(n_boot):
        k_b = rng.binomial(n_int, p_hat)
        try:
            res = minimize(
                lambda p: _neg_loglik(model, p, x, k_b.astype(float), counts),
                popt,
                bounds=list(zip(lower, upper)),
                method="L-BFGS-B",
            )
            boots[b] = res.x
        except Exception:  # noqa: BLE001 - a failed resample shouldn't sink the whole CI
            boots[b] = np.nan
    alpha = (1 - confidence) / 2
    lo = np.nanquantile(boots, alpha, axis=0)
    hi = np.nanquantile(boots, 1 - alpha, axis=0)
    return {
        name: (float(lo[i]), float(hi[i])) for i, name in enumerate(model.param_names)
    }
