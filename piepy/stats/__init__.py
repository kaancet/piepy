"""Statistics for piepy: consistent, plotting-ready estimates and tests.

Pure numpy/scipy. Estimators return :class:`Estimate` (value + CI + n); tests return
``TestResult``; :func:`aggregate` turns a trial table into a tidy per-condition estimate frame.
These are the compute layer used by the plotting (behaviz) and per-paradigm analysis pipelines.
"""

from .aggregate import Count, Mean, Median, Rate, aggregate, group_arrays
from .estimators import Estimate, bootstrap_ci, mean_ci, median_ci, proportion_ci
from .tests import TestResult, compare, energy_2d, ks_2d, mantel_haenszel

__all__ = [
    "Estimate",
    "proportion_ci",
    "mean_ci",
    "median_ci",
    "bootstrap_ci",
    "aggregate",
    "group_arrays",
    "Rate",
    "Mean",
    "Median",
    "Count",
    "TestResult",
    "compare",
    "ks_2d",
    "energy_2d",
    "mantel_haenszel",
]
