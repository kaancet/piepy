import numpy as np
import time
from .io import display
from numpy.typing import ArrayLike


def unique_except(x: ArrayLike, exceptions: list) -> np.ndarray:
    """Returns the unique values in an array except the given list

    Args:
        x: The array to be searched
        exceptions: The values that should be excluded froom unique search
        
    Returns:
        np.ndarray: An array of unique values except given values
    """
    uniq = np.unique(x)
    ret = [i for i in uniq if i not in exceptions]
    return np.asarray(ret)


def nonan_unique(x: ArrayLike, sort: bool = False) -> np.ndarray:
    """Returns the unique list without nan values

    Args:
        x: The array to be searched
        sort: Flag to sort the resulting unique array
        
    Returns:
        np.ndarray: An aray of unique values that are not nan
    """
    x = np.array(x)
    u = np.unique(x[~np.isnan(x)])
    if sort:
        return np.sort(u)
    else:
        return u


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
