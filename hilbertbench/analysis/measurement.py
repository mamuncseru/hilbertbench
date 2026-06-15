#!/usr/bin/env python
#
# file: hilbertbench/analysis/measurement.py
#
# revision history:
#  20260610 (am): derive the noise floor from recorded target
#                 precision when no shot counts were recorded
#  20260604 (am): cleaned up to project coding standards
#
# Measurement-strategy diagnostics (Diagnostic Axis: Measurement).
# Answers the question: is the optimization signal buried in shot noise?
#
# An expectation value estimated from 'shots' samples has an estimator
# variance bounded below by ~1/shots (the shot-noise floor). If the
# variance of the outcome trajectory is comparable to that floor, the
# optimizer is chasing noise rather than signal.
#
#   from hilbertbench.analysis import shot_noise_ratio
#   result = shot_noise_ratio("runs/20260605_xxx")
#   # {"status": "Signal Clear ...", "estimated_snr": 8.3, ...}
#------------------------------------------------------------------------------

# future imports must come first
#
from __future__ import annotations

# import system modules
#
import os
from typing import Any, Optional

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

#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------

def _collect_shots(trace) -> list[int]:
    """
    function: _collect_shots

    arguments:
     trace: a resolved HilbertTrace object

    return:
     list of shot counts extracted from EXECUTION_COMPLETED events

    description:
     Reads the shot count from every EXECUTION_COMPLETED event that
     carries one. Returns an empty list when no shots are recorded.
    """

    # collect shot counts from all completed spans
    #
    shots: list[int] = []
    for span in trace.completed():
        attrs = span.event_attributes("EXECUTION_COMPLETED")
        if attrs and attrs.get("shots"):
            shots.append(int(attrs["shots"]))

    # exit gracefully
    #
    return shots
#
# end of function


def _collect_precisions(trace) -> list[float]:
    """
    function: _collect_precisions

    arguments:
     trace: a resolved HilbertTrace object

    return:
     list of target precisions from EXECUTION_COMPLETED events

    description:
     Estimator runs request a target precision p rather than a shot
     count; the corresponding shot-noise floor is p^2 (shots ~ 1/p^2).
     Returns an empty list when no precisions are recorded.
    """

    # collect precisions from all completed spans
    #
    precisions: list[float] = []
    for span in trace.completed():
        attrs = span.event_attributes("EXECUTION_COMPLETED")
        if attrs and attrs.get("precision"):
            p = float(attrs["precision"])
            if p > 0:
                precisions.append(p)

    # exit gracefully
    #
    return precisions
#
# end of function


def shot_noise_ratio(
    trace: TraceLike,
    default_shots: Optional[int] = None,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    """
    function: shot_noise_ratio

    arguments:
     trace:         a HilbertTrace or run-directory path
     default_shots: shot count to assume when none was recorded; if
                    None and no shots are recorded, the status
                    message says so rather than guessing
     n_boot:        bootstrap resamples for the variance confidence
                    interval (0 disables the interval)
     ci:            confidence level for the bootstrap interval
                    (0.95 = 95%)
     seed:          optional RNG seed for the bootstrap

    return:
     a dict with keys:
      status              human-readable classification
      empirical_variance  variance of the outcome trajectory
      theoretical_floor   mean 1/shots across spans (None if unknown)
      estimated_snr       empirical_variance / theoretical_floor
                          (None if unknown)
      mean_shots          mean shot count used (None if unknown)
      shots_source        'recorded' | 'default' | 'precision' | None
      num_evaluations     number of numeric outcomes considered

    description:
     Compares the variance of the outcome trajectory against the
     shot-noise floor (1/shots). When no shot counts were recorded
     but target precisions were (estimator runs), the floor is
     derived as precision^2. Classification thresholds:
      SNR < 1.5  -> Shot Noise Dominated
      SNR < 5.0  -> Marginal
      SNR >= 5.0 -> Signal Clear
    """

    # resolve the trace object
    #
    t = as_trace(trace)

    # collect all numeric outcomes
    #
    outcomes = t.numeric_outcomes()

    # return an insufficient-data sentinel for very small traces
    #
    if outcomes.size < 2:
        return {
            "status":                "Insufficient Data",
            "empirical_variance":    None,
            "empirical_variance_ci": [None, None],
            "theoretical_floor":     None,
            "estimated_snr":         None,
            "mean_shots":            None,
            "shots_source":          None,
            "num_evaluations":       int(outcomes.size),
            "confidence_level":      ci,
        }

    # compute empirical variance of the outcome trajectory with a CI
    #
    empirical_variance = float(np.var(outcomes))
    var_low, var_high = bootstrap_ci(
        outcomes, np.var, n_boot=n_boot, ci=ci, seed=seed
    )

    # collect recorded shot counts; apply default when provided
    #
    shots = _collect_shots(t)
    shots_source = "recorded" if shots else None
    if not shots and default_shots is not None:
        shots = [default_shots]
        shots_source = "default"

    # establish the shot-noise floor: from shot counts when known,
    # otherwise from the recorded target precision (floor = p^2)
    #
    theoretical_floor = None
    mean_shots = None
    if shots:
        mean_shots = float(np.mean(shots))
        theoretical_floor = 1.0 / mean_shots
    else:
        precisions = _collect_precisions(t)
        if precisions:
            shots_source = "precision"
            theoretical_floor = float(
                np.mean([p * p for p in precisions])
            )
            mean_shots = float(
                np.mean([1.0 / (p * p) for p in precisions])
            )

    # return when the measurement budget is unknown
    #
    if theoretical_floor is None:
        return {
            "status": (
                "Shot count not recorded "
                "(pass default_shots to estimate SNR)"
            ),
            "empirical_variance":    empirical_variance,
            "empirical_variance_ci": [var_low, var_high],
            "theoretical_floor":     None,
            "estimated_snr":         None,
            "mean_shots":            None,
            "shots_source":          None,
            "num_evaluations":       int(outcomes.size),
            "confidence_level":      ci,
        }

    # compute the signal-to-noise ratio against the floor
    #
    snr = (
        empirical_variance / theoretical_floor
        if theoretical_floor > 0
        else float("inf")
    )

    # classify the signal quality by SNR
    #
    if snr < 1.5:
        status = "Shot Noise Dominated (signal buried in variance)"
    elif snr < 5.0:
        status = "Marginal (signal comparable to shot noise)"
    else:
        status = "Signal Clear (not limited by shot noise)"

    # exit gracefully
    #
    return {
        "status":                status,
        "empirical_variance":    empirical_variance,
        "empirical_variance_ci": [var_low, var_high],
        "theoretical_floor":     theoretical_floor,
        "estimated_snr":         float(snr),
        "mean_shots":            mean_shots,
        "shots_source":          shots_source,
        "num_evaluations":       int(outcomes.size),
        "confidence_level":      ci,
    }
#
# end of function

#
# end of file
