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
from hilbertbench.analysis._util import TraceLike, as_trace

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
) -> dict[str, Any]:
    """
    function: detect_barren_plateau

    arguments:
     trace:     a HilbertTrace or run-directory path
     threshold: variance below this value is classified as a barren
                plateau (default: DEFAULT_PLATEAU_THRESHOLD)

    return:
     a dict with keys:
      status           'Trainable' | 'Barren Plateau Detected'
                       | 'Insufficient Data'
      variance         variance of the outcome distribution, or None
      std_dev          standard deviation, or None
      num_evaluations  number of numeric outcome values considered
      threshold        the threshold used for classification

    description:
     Computes the variance of all numeric execution outcomes in the
     trace and classifies ansatz trainability. A variance below the
     threshold signals exponential concentration of the gradient
     landscape (a barren plateau).
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
            "status":          "Insufficient Data",
            "variance":        None,
            "std_dev":         None,
            "num_evaluations": 0,
            "threshold":       threshold,
        }

    # compute variance and classify
    #
    variance = float(np.var(outcomes))
    status = (
        "Trainable"
        if variance > threshold
        else "Barren Plateau Detected"
    )

    # exit gracefully
    #
    return {
        "status":          status,
        "variance":        variance,
        "std_dev":         float(np.std(outcomes)),
        "num_evaluations": int(outcomes.size),
        "threshold":       threshold,
    }
#
# end of function

#
# end of file
