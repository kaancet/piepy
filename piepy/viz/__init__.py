"""piepy plotting: thin behaviz wrappers that return a PlotResult (data + stats + figure).

behaviz is imported lazily inside each plot, so importing ``piepy.viz`` never requires it -- only
calling a plot does.
"""

from .base import PlotResult
from .accessor import Viz
from .plots import psychometric, reaction_time_cloud, reaction_time_dist


__all__ = ["PlotResult", "Viz", "psychometric", "reaction_time_cloud", "reaction_time_dist"]
