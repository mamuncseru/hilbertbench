#!/usr/bin/env python
#
# file: hilbertbench/analysis/noise.py
#
# revision history:
#  20260605 (am): initial version
#
# Noise and error-mitigation diagnostics (Diagnostic Axis: Noise).
# Reads the calibration snapshot recorded at execution time (T1, T2,
# readout error, gate errors) and, combined with the recorded circuit
# structure, estimates how much hardware noise the run was exposed to.
#
# This is evidence plus a coarse, standard estimate — not a precise
# simulation. The estimated circuit fidelity is the usual NISQ product
# of per-operation success probabilities; it lets a researcher correlate
# a loss spike with device calibration rather than the model alone.
#
#   from hilbertbench.analysis import noise_profile
#   noise_profile("runs/20260605_xxx")
#
# Returns None-valued fields for ideal-simulator traces, which carry no
# calibration snapshot (degraded mode, not failure).
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
from hilbertbench.analysis.circuit import circuit_structure

#------------------------------------------------------------------------------
#
# global variables are listed here
#
#------------------------------------------------------------------------------

# set the filename using basename
#
__FILE__ = os.path.basename(__file__)

# estimated-fidelity bands for the noise verdict
#
FIDELITY_LOW_NOISE = 0.90
FIDELITY_MODERATE = 0.50

#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------

def _stats(values: list) -> dict:
    """
    function: _stats

    arguments:
     values: a list of numeric values

    return:
     dict with mean/min/max, or None values if the list is empty

    description:
     Small helper to summarise a list of calibration values.
    """

    # summarise or return empty sentinel
    #
    if not values:
        return {"mean": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
#
# end of function


def _parse_calibration(cal: dict) -> dict:
    """
    function: _parse_calibration

    arguments:
     cal: the calibration snapshot dict (backend.properties().to_dict())

    return:
     dict of per-qubit lists (t1, t2, readout) and gate-error lists
     split into single-qubit and multi-qubit

    description:
     Flattens the BackendProperties structure into plain lists. T1/T2
     are reported in microseconds by Qiskit's to_dict().
    """

    # collect per-qubit coherence and readout values
    #
    t1, t2, readout = [], [], []
    for q in cal.get("qubits", []):
        params = {p["name"]: p["value"] for p in q}
        if "T1" in params:
            t1.append(params["T1"])
        if "T2" in params:
            t2.append(params["T2"])
        if "readout_error" in params:
            readout.append(params["readout_error"])

    # collect gate errors, split by arity
    #
    err_1q, err_2q = [], []
    for g in cal.get("gates", []):
        params = {p["name"]: p["value"] for p in g.get("parameters", [])}
        gate_error = params.get("gate_error")
        if gate_error is None:
            continue
        arity = len(g.get("qubits", []))
        if arity == 1:
            err_1q.append(gate_error)
        elif arity >= 2:
            err_2q.append(gate_error)

    # exit gracefully
    #
    return {
        "t1": t1, "t2": t2, "readout": readout,
        "err_1q": err_1q, "err_2q": err_2q,
        "backend_name": cal.get("backend_name"),
    }
#
# end of function


def noise_profile(trace: TraceLike) -> dict[str, Any]:
    """
    function: noise_profile

    arguments:
     trace: a HilbertTrace or run-directory path

    return:
     a dict with keys:
      status                    noise verdict or guard message
      backend_name              device name from the snapshot
      num_qubits_calibrated     qubits present in the snapshot
      t1_us / t2_us             coherence-time stats (mean/min/max, us)
      readout_error             readout-error stats
      gate_error_1q_mean        mean single-qubit gate error
      gate_error_2q_mean        mean two-qubit gate error
      circuit                   recorded circuit structure summary
      estimated_circuit_fidelity  product of per-op success probs
      dominant_error_source     largest infidelity contributor

    description:
     Summarises the recorded device calibration and, using the circuit
     structure, estimates the run's circuit fidelity as
       (1-e1q)^n1q * (1-e2q)^n2q * (1-readout)^n_measured
     classifying the result into low / moderate / dominated noise. The
     dominant error source is whichever factor removes the most fidelity.
    """

    # resolve the trace and fetch the calibration snapshot
    #
    t = as_trace(trace)
    cal = t.calibration()

    # degraded mode: ideal simulators carry no calibration
    #
    if cal is None:
        return {
            "status": "No calibration recorded (ideal simulator)",
            "backend_name": None,
            "num_qubits_calibrated": 0,
            "t1_us": _stats([]), "t2_us": _stats([]),
            "readout_error": _stats([]),
            "gate_error_1q_mean": None, "gate_error_2q_mean": None,
            "circuit": None,
            "estimated_circuit_fidelity": None,
            "dominant_error_source": None,
        }

    # parse calibration and summarise the device
    #
    p = _parse_calibration(cal)
    mean_1q = float(np.mean(p["err_1q"])) if p["err_1q"] else 0.0
    mean_2q = float(np.mean(p["err_2q"])) if p["err_2q"] else 0.0
    mean_ro = float(np.mean(p["readout"])) if p["readout"] else 0.0

    # pull circuit structure to weight the errors by usage
    #
    cs = circuit_structure(t)
    prim = cs.get("primary")
    fidelity = None
    dominant = None
    circuit_summary = None

    if prim is not None:
        n_1q = prim["single_qubit_gates"]
        n_2q = prim["entangling_gates"]
        n_meas = prim["num_measurements"] or prim["num_qubits"]
        circuit_summary = {
            "num_qubits": prim["num_qubits"],
            "depth": prim["depth"],
            "single_qubit_gates": n_1q,
            "entangling_gates": n_2q,
            "num_measurements": n_meas,
        }

        # per-channel survival probabilities
        #
        s_1q = (1.0 - mean_1q) ** n_1q
        s_2q = (1.0 - mean_2q) ** n_2q
        s_ro = (1.0 - mean_ro) ** n_meas
        fidelity = float(s_1q * s_2q * s_ro)

        # dominant infidelity contributor
        #
        contributions = {
            "single_qubit_gates": 1.0 - s_1q,
            "two_qubit_gates": 1.0 - s_2q,
            "readout": 1.0 - s_ro,
        }
        dominant = max(contributions, key=contributions.get)

    # classify the noise level from the estimated fidelity
    #
    if fidelity is None:
        status = "Calibration recorded (no circuit to weight errors)"
    elif fidelity >= FIDELITY_LOW_NOISE:
        status = "Low Noise (high estimated fidelity)"
    elif fidelity >= FIDELITY_MODERATE:
        status = "Moderate Noise"
    else:
        status = "Noise Dominated (low estimated fidelity)"

    # exit gracefully
    #
    return {
        "status": status,
        "backend_name": p["backend_name"],
        "num_qubits_calibrated": len(p["t1"]),
        "t1_us": _stats(p["t1"]),
        "t2_us": _stats(p["t2"]),
        "readout_error": _stats(p["readout"]),
        "gate_error_1q_mean": mean_1q,
        "gate_error_2q_mean": mean_2q,
        "circuit": circuit_summary,
        "estimated_circuit_fidelity": fidelity,
        "dominant_error_source": dominant,
    }
#
# end of function

#
# end of file
