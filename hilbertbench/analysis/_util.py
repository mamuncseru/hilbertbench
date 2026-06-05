#!/usr/bin/env python
#
# file: hilbertbench/analysis/_util.py
#
# revision history:
#  20260604 (am): cleaned up to project coding standards
#
# Shared helpers for the built-in analysis functions.
#------------------------------------------------------------------------------

# future imports must come first
#
from __future__ import annotations

# import system modules
#
from pathlib import Path
from typing import Callable, Optional, Union

# import third-party modules
#
import numpy as np

# import hilbertbench modules
#
from hilbertbench.trace import HilbertTrace

#------------------------------------------------------------------------------
#
# global variables are listed here
#
#------------------------------------------------------------------------------

# define a type alias for anything that resolves to a trace
#
TraceLike = Union[HilbertTrace, str, Path]

#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------

def as_trace(trace: TraceLike) -> HilbertTrace:
    """
    function: as_trace

    arguments:
     trace: a HilbertTrace object or a path to a run directory

    return:
     a HilbertTrace instance

    description:
     Accepts either a HilbertTrace or a run-directory path and returns
     a HilbertTrace. Used as a normalizer at the entry point of every
     built-in analysis function.
    """

    # return as-is if already a HilbertTrace
    #
    if isinstance(trace, HilbertTrace):
        return trace

    # exit gracefully — construct from path
    #
    return HilbertTrace(trace)
#
# end of function


def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: Optional[int] = None,
) -> tuple:
    """
    function: bootstrap_ci

    arguments:
     data:      1-D array of observations
     statistic: function mapping a sample array to a scalar
     n_boot:    number of bootstrap resamples (default 1000)
     ci:        central confidence level in (0, 1) (default 0.95)
     seed:      RNG seed for reproducibility

    return:
     (low, high) percentile confidence interval, or (None, None) if
     there are fewer than two observations or n_boot <= 0

    description:
     Non-parametric percentile bootstrap. Resamples `data` with
     replacement `n_boot` times, recomputes `statistic` on each
     resample, and returns the central `ci` percentile interval. Makes
     no distributional assumptions, which suits the heavy-tailed,
     bounded outcome distributions seen in QML traces.
    """

    # guard against degenerate inputs
    #
    arr = np.asarray(data, dtype=float)
    n = arr.size
    if n < 2 or n_boot <= 0:
        return (None, None)

    # resample and recompute the statistic
    #
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = arr[rng.integers(0, n, n)]
        estimates[i] = statistic(sample)

    # take the central percentile interval
    #
    alpha = (1.0 - ci) / 2.0
    low = float(np.quantile(estimates, alpha))
    high = float(np.quantile(estimates, 1.0 - alpha))

    # exit gracefully
    #
    return (low, high)
#
# end of function

#
# end of file
