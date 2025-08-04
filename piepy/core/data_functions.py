import itertools
import numpy as np
import polars as pl
from collections.abc import Generator


def make_subsets(
    data: pl.DataFrame,
    col_name: str | list[str],
    no_nan: bool = True,
    do_sort: bool = True,
    start_enumerate: int | None = None,
) -> Generator:
    """Generates subsets of the data given the col names
    NOTE: The length of returned Generator depends on the length of columns you provide

    Args:
        data: DataFrame to be filtered in subsets
        col_name: Name(s) of the column(s)
        no_nan: Flag to drop the null values of the columns
        do_sort: Flag to sort the data
    """
    if isinstance(col_name, str):
        col_name = [col_name]

    temp = []
    for c in col_name:
        if c not in data.columns:
            raise ValueError(f"No column name {c} in data")

        col_data = data[c]
        if no_nan:
            col_data = col_data.drop_nulls()
        uniq_col = col_data.unique()

        if do_sort:
            uniq_col = uniq_col.sort()

        temp.append(uniq_col.to_list())
    if start_enumerate is None:
        for u in list(itertools.product(*temp)):
            yield (*u, data.filter([pl.col(col_name[i]) == j for i, j in enumerate(u)]))
    else:
        for i, u in enumerate(list(itertools.product(*temp)), start=start_enumerate):
            yield (
                i,
                *u,
                data.filter([pl.col(col_name[i]) == j for i, j in enumerate(u)]),
            )


def get_baseline_hr(data: pl.DataFrame) -> float:
    """_summary_

    Args:
        data (pl.DataFrame): _description_

    Returns:
        float: _description_
    """
    # assert only single session in the data
    if "session_id" in data.columns:
        assert data.n_unique("session_id") == 1, (
            "Multiple sessions exit, can only do in one session"
        )

    catch_trials = data.filter(
        (pl.col("opto_pattern") == -1)
        & (pl.col("contrast") == 0)
        & (pl.col("stim_side") == "catch")
    )
    if "outcome" in catch_trials.columns:
        return len(catch_trials.filter(pl.col("outcome") == "hit")) / len(catch_trials)
    else:
        # explode the count columns
        return np.sum(catch_trials["hit_count"].explode().to_numpy()) / np.sum(
            catch_trials["count"].explode().to_numpy()
        )
