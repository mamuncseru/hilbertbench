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
from typing import Union

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

#
# end of file
