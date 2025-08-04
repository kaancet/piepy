import time
import numpy as np
import polars as pl

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


def clean_string(str_in: str) -> str:
    """Cleans the string by removing surrounding whitespaces and making it lowercase

    Args:
        str_in (str): Input string to be cleaned

    Returns:
        str: Cleaned string
    """
    # clean whitespace around and make lowercase
    return str_in.strip().lower()


def timeit(msg):
    def decorator(func):
        def wrapper(*args, **kwargs):
            ts = time.time()
            result = func(*args, **kwargs)
            te = time.time()
            display(f"{msg} : {te - ts:.3}s")
            return result

        return wrapper

    return decorator


#######################
# POLARS EXPRESSIONS
#######################


def pl_weighted_mean(
    value_col: str, weight_col: str, ignore_nulls: bool = True
) -> pl.Expr:
    """Generate a Polars aggregation expression to take a weighted mean
    https://github.com/pola-rs/polars/issues/7499

    Args:
        value_col (str): _description_
        weight_col (str): _description_
        ignore_nulls (bool, optional): _description_. Defaults to True.

    Returns:
        pl.Expr: _description_
    """
    values = pl.col(value_col)

    if ignore_nulls:
        weights = pl.when(values.is_not_null()).then(weight_col)
    else:
        weights = pl.col(value_col)

    return weights.dot(values).truediv(weights.sum()).fill_nan(None)
