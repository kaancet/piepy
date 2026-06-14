"""Synthetic data generators for piepy.

Useful for trying modelling approaches and exercising the analysis stack (stats, fitting,
plotting) without needing real recordings. :func:`simulate_session` produces a DataFrame shaped
like a parsed session.
"""

from .session import simulate_session

__all__ = ["simulate_session"]
