#!/usr/bin/env python
#
# file: hilbertbench/analysis/optimization.py
#
# revision history:
#  20260604 (am): cleaned up to project coding standards
#
# Classical optimization-loop diagnostics (Diagnostic Axis:
# Optimization). Reads the ordered trajectory of outcomes and bound
# parameters to characterise how the optimizer is behaving: is the
# cost landscape still moving, has it converged, or has it stalled?
#
#   from hilbertbench.analysis import optimization_convergence
#   optimization_convergence("runs/20260605_xxx")
#
# SCOPE NOTE: this analyzer treats each span as one trajectory point.
# That is exact for optimizers where one span == one objective
# evaluation. For batched QML (VQC/QNN) or parameter-shift gradients,
# consecutive spans differ by data sample or shift probe, so
# movement_ratio stays near 1 and the convergence verdict is
# uninformative. Outcome envelope fields remain valid in all cases.
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

#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------

def _ordered_completed(trace) -> list:
    """
    function: _ordered_completed

    arguments:
     trace: a resolved HilbertTrace object

    return:
     completed spans sorted ascending by sequence number

    description:
     Returns all COMPLETED spans in the order they were recorded.
     Spans with a None sequence number sort to position 0.
    """

    # sort completed spans by sequence number
    #
    spans = trace.completed()
    return sorted(
        spans,
        key=lambda s: (
            s.sequence_number if s.sequence_number is not None else 0
        ),
    )
#
# end of function


def _parameter_movements(spans) -> np.ndarray:
    """
    function: _parameter_movements

    arguments:
     spans: an ordered list of SpanView objects

    return:
     1-D float array of L2 norms of consecutive parameter changes

    description:
     Computes the L2 distance between consecutive parameter vectors.
     Spans with no parameters or a different vector width from the
     previous span reset the chain — their gap is skipped rather than
     zero-filled to avoid artificial stall readings.
    """

    # accumulate L2 norms of consecutive parameter deltas
    #
    moves: list[float] = []
    prev = None
    for span in spans:
        p = span.parameters
        if p is None:
            prev = None
            continue
        vec = np.ravel(np.asarray(p, dtype=float))
        if prev is not None and prev.shape == vec.shape:
            moves.append(float(np.linalg.norm(vec - prev)))
        prev = vec

    # exit gracefully
    #
    return np.asarray(moves, dtype=float)
#
# end of function


def optimization_convergence(trace: TraceLike) -> dict[str, Any]:
    """
    function: optimization_convergence

    arguments:
     trace: a HilbertTrace or run-directory path

    return:
     a dict with keys:
      status             'Converged' | 'Converging' | 'Still Improving'
                         | 'Stalled' | 'Insufficient Data'
      num_steps          number of completed spans considered
      outcome_initial    mean outcome in the first 10% of the run
      outcome_final      mean outcome in the last 10% of the run
      outcome_min        minimum outcome value
      outcome_max        maximum outcome value
      outcome_trend      outcome_final - outcome_initial
      total_path_length  summed L2 parameter movement over the run
      early_movement     mean parameter movement in the first half
      late_movement      mean parameter movement in the second half
      movement_ratio     late / early (0 converged, 1 still moving)

    description:
     Characterises the optimization trajectory using two signals:
      1. Outcome envelope — min/max/trend of the cost trajectory.
      2. Parameter movement — how much parameters shift between
         consecutive spans (proxies for optimizer step size).
     Classification uses the late/early movement ratio. See the
     module-level scope note for caveats on batched QML runs.
    """

    # resolve the trace and sort completed spans
    #
    t = as_trace(trace)
    spans = _ordered_completed(t)

    # return an insufficient-data sentinel for very short traces
    #
    if len(spans) < 4:
        return {
            "status":            "Insufficient Data",
            "num_steps":         len(spans),
            "outcome_initial":   None,
            "outcome_final":     None,
            "outcome_min":       None,
            "outcome_max":       None,
            "outcome_trend":     None,
            "total_path_length": None,
            "early_movement":    None,
            "late_movement":     None,
            "movement_ratio":    None,
        }

    # compute outcome-envelope statistics with 10% window
    #
    outcomes = t.numeric_outcomes()
    out_stats: dict[str, Any] = {
        "outcome_initial": None,
        "outcome_final":   None,
        "outcome_min":     None,
        "outcome_max":     None,
        "outcome_trend":   None,
    }
    if outcomes.size >= 2:
        w = max(1, outcomes.size // 10)
        out_stats = {
            "outcome_initial": float(np.mean(outcomes[:w])),
            "outcome_final":   float(np.mean(outcomes[-w:])),
            "outcome_min":     float(outcomes.min()),
            "outcome_max":     float(outcomes.max()),
            "outcome_trend":   float(
                np.mean(outcomes[-w:]) - np.mean(outcomes[:w])
            ),
        }

    # compute parameter movement deltas
    #
    moves = _parameter_movements(spans)

    # fall back to outcome-only verdict when parameter data is absent
    #
    if moves.size < 2:
        tail = outcomes[-max(1, outcomes.size // 4):]
        status = (
            "Converged"
            if outcomes.size and float(np.var(tail)) < 1e-6
            else "Still Improving"
        )
        return {
            "status":            status,
            "num_steps":         len(spans),
            **out_stats,
            "total_path_length": None,
            "early_movement":    None,
            "late_movement":     None,
            "movement_ratio":    None,
        }

    # compute early/late movement ratio
    #
    half = moves.size // 2
    early = float(np.mean(moves[:half])) if half > 0 else 0.0
    late = float(np.mean(moves[half:]))
    if early > 1e-12:
        ratio = late / early
    else:
        ratio = 0.0 if late < 1e-12 else float("inf")

    # classify convergence based on movement ratio
    #
    if early < 1e-9 and late < 1e-9:
        status = "Stalled"
    elif ratio < 0.1:
        status = "Converged"
    elif ratio < 0.5:
        status = "Converging"
    else:
        status = "Still Improving"

    # exit gracefully
    #
    return {
        "status":            status,
        "num_steps":         len(spans),
        **out_stats,
        "total_path_length": float(np.sum(moves)),
        "early_movement":    early,
        "late_movement":     late,
        "movement_ratio":    float(ratio),
    }
#
# end of function

#
# end of file
