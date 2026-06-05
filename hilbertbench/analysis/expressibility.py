#!/usr/bin/env python
#
# file: hilbertbench/analysis/expressibility.py
#
# revision history:
#  20260604 (am): cleaned up to project coding standards
#
# Ansatz expressibility diagnostics (Diagnostic Axis: Ansatz,
# expressibility sub-axis).
#
# Expressibility measures how uniformly an ansatz's output states cover
# the Hilbert space. The standard estimator compares the distribution
# of fidelities between randomly-parameterized output states against
# the Haar-random fidelity distribution. A small KL divergence means
# the ansatz is highly expressive; a large one means it is confined to
# a small manifold.
#
# Requires an Active Mode trace (statevectors under uniformly-sampled
# parameters) — see hilbertbench.active. Applying it to a passive
# training trace is physically meaningless because training parameters
# are not Haar-random.
#
#   from hilbertbench.active import active_probe_qiskit
#   from hilbertbench.analysis import kl_expressibility
#   run = active_probe_qiskit(ansatz, num_samples=2000,
#                             output_root="runs")
#   kl_expressibility(run)
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

def _to_statevector(outcome: Any) -> Optional[np.ndarray]:
    """
    function: _to_statevector

    arguments:
     outcome: a raw span outcome value (any type)

    return:
     a complex 1-D numpy array, or None if not a statevector

    description:
     Reconstructs a complex statevector from a span outcome. Handles
     two storage formats produced by Active Mode:
      - JSON [[re, im], ...] form (inline artifact, small states)
      - .npy complex array form (file-store artifact, large states)
    """

    # return None for missing outcomes
    #
    if outcome is None:
        return None

    # decode the JSON [[re, im], ...] Active Mode form
    #
    if (
        isinstance(outcome, list)
        and outcome
        and isinstance(outcome[0], (list, tuple))
        and len(outcome[0]) == 2
    ):
        arr = np.asarray(outcome, dtype=float)
        return arr[:, 0] + 1j * arr[:, 1]

    # decode .npy / array form for large statevectors
    #
    arr = np.asarray(outcome)
    if np.iscomplexobj(arr):
        return arr.ravel()

    # exit gracefully — not a recognizable statevector
    #
    return None
#
# end of function


def _haar_probabilities(
    bin_edges: np.ndarray,
    N: int,
) -> np.ndarray:
    """
    function: _haar_probabilities

    arguments:
     bin_edges: 1-D array of bin boundary values in [0, 1]
     N:         Hilbert space dimension (2 ** num_qubits)

    return:
     probability mass per bin under the Haar fidelity distribution

    description:
     Integrates the Haar fidelity CDF F(x) = 1 - (1 - x)^(N-1)
     over each bin to obtain the exact Haar probability mass. Used
     as the reference distribution in the KL divergence computation.
    """

    # integrate the Haar CDF over each bin
    #
    probs = []
    for i in range(len(bin_edges) - 1):
        cdf_hi = 1.0 - (1.0 - bin_edges[i + 1]) ** (N - 1)
        cdf_lo = 1.0 - (1.0 - bin_edges[i]) ** (N - 1)
        probs.append(cdf_hi - cdf_lo)

    # exit gracefully
    #
    return np.asarray(probs)
#
# end of function


def kl_expressibility(
    trace: TraceLike,
    num_bins: int = 75,
    max_pairs: int = 5000,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    """
    function: kl_expressibility

    arguments:
     trace:     a HilbertTrace or run-directory path; must be an
                Active Mode trace (mode == 'active')
     num_bins:  histogram bins for the fidelity distribution
     max_pairs: maximum number of state pairs to sample
     seed:      optional RNG seed for reproducibility

    return:
     a dict with keys:
      status        classification, or a guard message
      kl_divergence KL(P_ansatz || P_Haar), or None
      num_states    number of statevectors used
      num_pairs     fidelity pairs sampled
      num_qubits    inferred from the statevector dimension

    description:
     Estimates ansatz expressibility as the KL divergence of the
     empirical state-fidelity distribution against the Haar measure.
     Thresholds:
      KL < 0.1  -> Highly Expressive (matches Haar)
      KL < 0.5  -> Moderately Expressive
      KL >= 0.5 -> Low Expressibility (rigid ansatz)

     Guards: requires Active Mode trace; at least 2 statevectors.
     Returns a status-only dict when guards are not met.
    """

    # resolve the trace object
    #
    t = as_trace(trace)

    # guard: expressibility requires an Active Mode trace
    #
    if t.mode != "active":
        return {
            "status": (
                "Requires Active Mode trace (mode != 'active')"
            ),
            "kl_divergence": None,
            "num_states":    0,
            "num_pairs":     0,
            "num_qubits":    None,
        }

    # collect statevectors from all completed spans
    #
    states = []
    for span in t.completed():
        sv = _to_statevector(span.outcome)
        if sv is not None:
            states.append(sv)

    # guard: need at least two statevectors to form fidelity pairs
    #
    if len(states) < 2:
        return {
            "status": (
                "Insufficient Data (need >= 2 statevectors)"
            ),
            "kl_divergence": None,
            "num_states":    len(states),
            "num_pairs":     0,
            "num_qubits":    None,
        }

    # infer qubit count from the statevector dimension
    #
    dim = len(states[0])
    num_qubits = int(round(np.log2(dim))) if dim > 0 else None

    # sample random pairs and compute pairwise fidelities
    #
    rng = np.random.default_rng(seed)
    n = len(states)
    total_possible = n * (n - 1) // 2
    pairs = min(max_pairs, total_possible)

    fidelities = np.empty(pairs)
    for k in range(pairs):
        i, j = rng.choice(n, 2, replace=False)
        overlap = np.vdot(states[i], states[j])
        fidelities[k] = float(np.abs(overlap) ** 2)

    # histogram the empirical fidelity distribution
    #
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    p_ansatz, _ = np.histogram(
        fidelities, bins=bin_edges, density=False
    )
    p_ansatz = p_ansatz / np.sum(p_ansatz)

    # compute Haar reference probabilities for the same bins
    #
    p_haar = _haar_probabilities(bin_edges, dim)

    # compute KL divergence with epsilon smoothing to avoid log(0)
    #
    eps = 1e-10
    p_a = np.where(p_ansatz == 0, eps, p_ansatz)
    p_h = np.where(p_haar == 0, eps, p_haar)
    kl = float(np.sum(p_a * np.log(p_a / p_h)))

    # classify expressibility by KL divergence
    #
    if kl < 0.1:
        status = "Highly Expressive (matches Haar)"
    elif kl < 0.5:
        status = "Moderately Expressive"
    else:
        status = "Low Expressibility (rigid ansatz)"

    # exit gracefully
    #
    return {
        "status":        status,
        "kl_divergence": kl,
        "num_states":    n,
        "num_pairs":     pairs,
        "num_qubits":    num_qubits,
    }
#
# end of function

#
# end of file
