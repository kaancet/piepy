from importlib.metadata import version as _version, PackageNotFoundError

try:
    __version__ = _version("piepy-neuro")
except PackageNotFoundError:  # not installed (e.g. running from source tree)
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
