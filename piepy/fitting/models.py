"""Psychometric models.

Each model maps a stimulus level ``x`` to a probability (0..1) given parameters. Every model
exposes both:

* ``predict(x, params)`` -- the curve, used for fitting and plotting;
* ``sample(x, params, rng=)`` -- Bernoulli draws from that curve, so you can *generate* data
  from a model and fit it back (parameter recovery), and so a future simulator can sit on top.

Models are registered by name (:data:`MODELS`) so a paradigm picks one with a string.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import erf as _erf

__all__ = ["Model", "Logistic", "Weibull", "Erf", "MODELS", "get_model"]


class Model(ABC):
    """A psychometric model: parameters -> probability curve."""

    name: str
    param_names: tuple[str, ...]

    @abstractmethod
    def predict(self, x: ArrayLike, params: ArrayLike) -> np.ndarray:
        """Probability at each ``x`` for the given parameter vector."""

    @abstractmethod
    def guess(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """A reasonable initial parameter vector from data."""

    @abstractmethod
    def bounds(self, x: np.ndarray, y: np.ndarray) -> tuple[list[float], list[float]]:
        """``(lower, upper)`` parameter bounds for the optimizer."""

    def sample(self, x: ArrayLike, params: ArrayLike, *, rng=None) -> np.ndarray:
        """Bernoulli outcomes (one per element of ``x``) from the model curve."""
        rng = np.random.default_rng(rng)
        p = np.clip(self.predict(x, params), 0.0, 1.0)
        return rng.random(np.shape(p)) < p

    @property
    def n_params(self) -> int:
        return len(self.param_names)


class Logistic(Model):
    """4-parameter logistic with lower/upper lapses; good general choice (e.g. P(right) vs x).

    ``p(x) = lapse_low + (1 - lapse_low - lapse_high) / (1 + exp(-k (x - x0)))``
    """

    name = "logistic"
    param_names = ("x0", "k", "lapse_low", "lapse_high")

    def predict(self, x, params):
        x0, k, lo, hi = params
        x = np.asarray(x, dtype=float)
        return lo + (1 - lo - hi) / (1 + np.exp(-k * (x - x0)))

    def guess(self, x, y):
        span = max(float(x.max() - x.min()), 1e-9)
        return np.array([float(np.median(x)), 4.0 / span, 0.02, 0.02])

    def bounds(self, x, y):
        span = max(float(x.max() - x.min()), 1e-9)
        return (
            [float(x.min()) - span, 0.0, 0.0, 0.0],
            [float(x.max()) + span, np.inf, 0.5, 0.5],
        )


class Weibull(Model):
    """Weibull on ``|x|`` with guess/lapse rates; the detection shape (hit rate vs |contrast|).

    ``p(x) = guess + (1 - guess - lapse) * (1 - exp(-(|x| / alpha) ** beta))``
    """

    name = "weibull"
    param_names = ("alpha", "beta", "guess", "lapse")

    def predict(self, x, params):
        alpha, beta, guess, lapse = params
        x = np.abs(np.asarray(x, dtype=float))
        alpha = max(alpha, 1e-9)
        return guess + (1 - guess - lapse) * (1 - np.exp(-((x / alpha) ** beta)))

    def guess(self, x, y):
        nz = np.abs(x[np.abs(x) > 0])
        alpha = float(np.median(nz)) if nz.size else 0.5
        return np.array([alpha, 2.0, float(np.clip(y.min(), 0, 0.5)), 0.02])

    def bounds(self, x, y):
        return ([1e-6, 0.0, 0.0, 0.0], [float(np.abs(x).max()) * 4 + 1, np.inf, 0.5, 0.5])


class Erf(Model):
    """Cumulative-Gaussian (erf) with a symmetric lapse; the discrimination shape (P(right) vs x).

    ``p(x) = lapse + (1 - 2 lapse) * 0.5 * (1 + erf((x - mu) / (sqrt(2) sigma)))``
    """

    name = "erf"
    param_names = ("mu", "sigma", "lapse")

    def predict(self, x, params):
        mu, sigma, lapse = params
        x = np.asarray(x, dtype=float)
        sigma = max(sigma, 1e-9)
        return lapse + (1 - 2 * lapse) * 0.5 * (1 + _erf((x - mu) / (np.sqrt(2) * sigma)))

    def guess(self, x, y):
        span = max(float(x.max() - x.min()), 1e-9)
        return np.array([float(np.median(x)), span / 4, 0.02])

    def bounds(self, x, y):
        span = max(float(x.max() - x.min()), 1e-9)
        return ([float(x.min()) - span, 1e-6, 0.0], [float(x.max()) + span, np.inf, 0.5])


MODELS: dict[str, type[Model]] = {m.name: m for m in (Logistic, Weibull, Erf)}


def get_model(model: str | Model) -> Model:
    """Resolve a model name (or pass-through a Model instance)."""
    if isinstance(model, Model):
        return model
    try:
        return MODELS[model]()
    except KeyError:
        raise ValueError(
            f"Unknown model {model!r}; registered: {sorted(MODELS)}"
        ) from None
