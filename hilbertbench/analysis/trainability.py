#!/usr/bin/env python
#
# file: hilbertbench/analysis/trainability.py
#
# revision history:
#  20260604 (am): cleaned up to project coding standards
#
# Trainability diagnostics (Diagnostic Axis: Ansatz). The signature of
# a barren plateau is an exponentially vanishing variance in the cost
# landscape: as the ansatz becomes untrainable, expectation values
# concentrate and their variance across the trajectory collapses
# toward zero.
#
# Plain function over a HilbertTrace — compose freely or call it
# standalone. Returns a plain dict (no hidden state).
#
#   from hilbertbench.analysis import detect_barren_plateau
#   result = detect_barren_plateau("runs/20260605_xxx")
#   # {"status": "Trainable", "variance": 0.21, ...}
#------------------------------------------------------------------------------

# future imports must come first
#
from __future__ import annotations

# import system modules
#
import os
from typing import Any

# import third-party modules
#
import numpy as np

# import hilbertbench modules
#
from hilbertbench.analysis._util import TraceLike, as_trace, bootstrap_ci

#------------------------------------------------------------------------------
#
# global variables are listed here
#
#------------------------------------------------------------------------------

# set the filename using basename
#
__FILE__ = os.path.basename(__file__)

# variance below this threshold is treated as a barren plateau;
# heuristic value — tune per study if needed
#
DEFAULT_PLATEAU_THRESHOLD = 0.005

#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------

def detect_barren_plateau(
    trace: TraceLike,
    threshold: float = DEFAULT_PLATEAU_THRESHOLD,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    function: detect_barren_plateau

    arguments:
     trace:     a HilbertTrace or run-directory path
     threshold: variance below this value is classified as a barren
                plateau (default: DEFAULT_PLATEAU_THRESHOLD)
     n_boot:    bootstrap resamples for the variance CI (0 disables)
     ci:        confidence level for the interval (default 0.95)
     seed:      RNG seed for the bootstrap

    return:
     a dict with keys:
      status             'Trainable' | 'Barren Plateau Detected'
                         | 'Insufficient Data'
      variance           variance of the outcome distribution, or None
      std_dev            standard deviation, or None
      num_evaluations    number of numeric outcome values considered
      threshold          the threshold used for classification
      variance_ci        [low, high] bootstrap CI on the variance
      confidence_level   the CI level used (e.g. 0.95)
      verdict_confidence 'high' if the CI is wholly one side of the
                         threshold, 'low' if it straddles it, else None

    description:
     Computes the variance of all numeric execution outcomes and
     classifies ansatz trainability. A bootstrap confidence interval is
     attached to the variance, and the verdict confidence is reported
     as low when that interval straddles the decision threshold —
     transparency over definitive attribution (proposal Section 2.6).
    """

    # resolve the trace object
    #
    t = as_trace(trace)

    # collect all numeric outcomes from completed spans
    #
    outcomes = t.numeric_outcomes()

    # return an insufficient-data sentinel when no outcomes exist
    #
    if outcomes.size == 0:
        return {
            "status":             "Insufficient Data",
            "variance":           None,
            "std_dev":            None,
            "num_evaluations":    0,
            "threshold":          threshold,
            "variance_ci":        [None, None],
            "confidence_level":   ci,
            "verdict_confidence": None,
        }

    # compute variance and classify
    #
    variance = float(np.var(outcomes))
    status = (
        "Trainable"
        if variance > threshold
        else "Barren Plateau Detected"
    )

    # bootstrap a confidence interval on the variance
    #
    low, high = bootstrap_ci(outcomes, np.var, n_boot=n_boot, ci=ci, seed=seed)

    # report verdict confidence: low if the CI straddles the threshold
    #
    verdict_confidence = None
    if low is not None and high is not None:
        straddles = low < threshold < high
        verdict_confidence = "low" if straddles else "high"

    # exit gracefully
    #
    return {
        "status":             status,
        "variance":           variance,
        "std_dev":            float(np.std(outcomes)),
        "num_evaluations":    int(outcomes.size),
        "threshold":          threshold,
        "variance_ci":        [low, high],
        "confidence_level":   ci,
        "verdict_confidence": verdict_confidence,
    }
#
# end of function

#
# end of file
