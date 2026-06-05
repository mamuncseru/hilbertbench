#!/usr/bin/env python
#
# file: hilbertbench/analysis/__init__.py
#
# revision history:
#  20260604 (am): cleaned up to project coding standards
#
# Public surface for the hilbertbench.analysis package. Every built-in
# diagnostic is a plain function: it accepts a HilbertTrace (or a run-
# directory path) and returns a plain dict. Functions are composable
# and hold no state.
#
#   from hilbertbench.analysis import detect_barren_plateau
#   from hilbertbench.analysis import shot_noise_ratio
#   detect_barren_plateau("runs/20260605_xxx")
#   shot_noise_ratio("runs/20260605_xxx")
#
# summary(trace) runs all built-in axes and returns a combined report.
#------------------------------------------------------------------------------

# future imports must come first
#
from __future__ import annotations

# import system modules
#
import os
from typing import Any

# import hilbertbench modules
#
from hilbertbench.analysis._util import TraceLike, as_trace
from hilbertbench.analysis.circuit import circuit_structure
from hilbertbench.analysis.expressibility import kl_expressibility
from hilbertbench.analysis.measurement import shot_noise_ratio
from hilbertbench.analysis.optimization import optimization_convergence
from hilbertbench.analysis.trainability import detect_barren_plateau

#------------------------------------------------------------------------------
#
# global variables are listed here
#
#------------------------------------------------------------------------------

# set the filename using basename
#
__FILE__ = os.path.basename(__file__)

__all__ = [
    "detect_barren_plateau",
    "shot_noise_ratio",
    "optimization_convergence",
    "circuit_structure",
    "kl_expressibility",
    "summary",
]

#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------

def summary(trace: TraceLike) -> dict[str, Any]:
    """
    function: summary

    arguments:
     trace: a HilbertTrace or run-directory path

    return:
     a combined diagnostic report dict keyed by analysis axis

    description:
     Runs every built-in analyzer and returns one combined report.
     Top-level keys: trace, trainability, measurement, optimization,
     circuit. Use individual functions for finer control.
    """

    # resolve the trace object
    #
    t = as_trace(trace)

    # run all built-in analyzers and combine into one report
    #
    return {
        "trace": {
            "status":    t.status,
            "mode":      t.mode,
            "num_spans": len(t),
            "tags":      t.tags,
        },
        "trainability": detect_barren_plateau(t),
        "measurement":  shot_noise_ratio(t),
        "optimization": optimization_convergence(t),
        "circuit":      circuit_structure(t),
    }
#
# end of function

#
# end of file
