"""How the trial groups are run: one at a time now, on many workers later.

Trial averaging splits the trials into groups and processes each group independently. An
"executor" just says *where* those groups run. Keeping this behind a tiny interface means the
averaging code never changes when you move from your laptop to a compute cluster: swap the
executor, get the same numbers.

Two executors are provided: a local one (all in this process) and a parallel one (several worker
processes on this machine). Because every group produces its own running total that is added up at
the end, and the results are kept in order, the choice of executor never changes the result.

On a compute cluster you usually don't need a special executor at all: run one job per group of
trials, have each write its running total to a file, and add the files together with
:func:`piepy.imaging.average.combine` in a final job. Same result, no extra library.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Iterable


class LocalExecutor:
    """Run each group of trials one after another, in this process."""

    def map(self, fn: Callable, items: Iterable) -> list:
        """Apply ``fn`` to every item and return the results in the same order."""
        return [fn(x) for x in items]


class ProcessExecutor:
    """Run the groups of trials on several worker processes on this machine.

    The work sent to each worker (the frame source, the windows, and the trial indices) must be
    picklable, since Python copies it to the workers. Plain in-memory arrays and file-backed stacks
    that only hold file paths are fine; a stack holding open file handles is not.
    """

    def __init__(self, n_workers: int | None = None):
        """``n_workers``: how many processes to use (``None`` lets Python pick, usually one per CPU)."""
        self.n_workers = n_workers

    def map(self, fn: Callable, items: Iterable) -> list:
        """Apply ``fn`` to every item on the worker processes and return the results in order."""
        items = list(items)
        with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
            return list(pool.map(fn, items))
