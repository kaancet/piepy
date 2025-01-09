import numpy as np
import time
from .io import display
from numpy.typing import ArrayLike


def unique_except(x: ArrayLike, exceptions: list) -> np.ndarray:
    """Returns the unique values in an array except the given list"""
    uniq = np.unique(x)
    ret = [i for i in uniq if i not in exceptions]
    return np.asarray(ret)


def nonan_unique(x: ArrayLike, sort: bool = False) -> np.ndarray:
    """Returns the unique list without nan values"""
    x = np.array(x)
    u = np.unique(x[~np.isnan(x)])
    if sort:
        return np.sort(u)
    else:
        return u


def get_fraction(
    data_in: np.ndarray, fraction_of, window_size: int = 10, min_period: int = None
) -> np.ndarray:
    """Returns the fraction of values in data_in"""
    if min_period is None:
        min_period = window_size

    fraction = []
    for i in range(len(data_in)):
        window_start = int(i - window_size / 2)
        if window_start < 0:
            window_start = 0
        window_end = int(i + window_size / 2)
        window = data_in[window_start:window_end]

        if len(window) < min_period:
            to_append = np.nan
        else:
            tmp = []
            for i in window:
                tmp.append(1 if i == fraction_of else 0)
                to_append = float(np.mean(tmp))
        fraction.append(to_append * 100)

    return np.array(fraction)


def timeit(msg):
    def decorator(func):
        def wrapper(*args, **kwargs):
            ts = time.time()
            result = func(*args, **kwargs)
            te = time.time()
            display(f"{msg} : {te-ts:.3}s")
            return result

        return wrapper

    return decorator
