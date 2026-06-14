"""Psychometric model fitting for piepy.

Pure scipy/numpy. Models expose ``predict`` (for fitting/plotting) and ``sample`` (for
generating data and parameter recovery). ``fit()`` auto-selects binomial MLE (when trial counts
are given) or least-squares, and returns a plotting-ready :class:`FitResult`.
"""

from .fit import FitResult, fit
from .models import MODELS, Erf, Logistic, Model, Weibull, get_model

__all__ = [
    "fit",
    "FitResult",
    "Model",
    "Logistic",
    "Weibull",
    "Erf",
    "MODELS",
    "get_model",
]
