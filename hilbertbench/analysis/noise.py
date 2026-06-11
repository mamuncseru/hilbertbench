#!/usr/bin/env python
#
# file: hilbertbench/analysis/noise.py
#
# revision history:
#  20260610 (am): scope calibration to the qubits the circuit actually
#                 used; device-wide means wildly mispredict on large
#                 devices (found in the ibm_marrakesh smoke test)
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
     dict with per-qubit value maps (t1, t2, readout keyed by qubit
     index) and gate-error lists of (qubit_tuple, error) split into
     single-qubit and multi-qubit

    description:
     Flattens the BackendProperties structure while preserving which
     qubit each value belongs to, so callers can scope statistics to
     the qubits a circuit actually used. T1/T2 are in microseconds.
    """

    # collect per-qubit coherence and readout values by index
    #
    t1, t2, readout = {}, {}, {}
    for idx, q in enumerate(cal.get("qubits", [])):
        params = {p["name"]: p["value"] for p in q}
        if "T1" in params:
            t1[idx] = params["T1"]
        if "T2" in params:
            t2[idx] = params["T2"]
        if "readout_error" in params:
            readout[idx] = params["readout_error"]

    # collect gate errors with their qubit tuples, split by arity
    #
    err_1q, err_2q = [], []
    for g in cal.get("gates", []):
        params = {p["name"]: p["value"] for p in g.get("parameters", [])}
        gate_error = params.get("gate_error")
        if gate_error is None:
            continue
        qubits = tuple(g.get("qubits", []))
        if len(qubits) == 1:
            err_1q.append((qubits, gate_error))
        elif len(qubits) >= 2:
            err_2q.append((qubits, gate_error))

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
      scope                     'active_qubits' | 'device_wide' —
                                which qubits the statistics cover
      active_qubits             qubit indices the circuit used
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
     classifying the result into low / moderate / dominated noise.
     All statistics are scoped to the qubits the recorded circuit
     actually used when those are known — device-wide averages over a
     large device (including its worst edges) mispredict the run's
     exposure by orders of magnitude. Falls back to device-wide
     values when the active qubits cannot be determined.
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
            "scope": None,
            "active_qubits": [],
            "t1_us": _stats([]), "t2_us": _stats([]),
            "readout_error": _stats([]),
            "gate_error_1q_mean": None, "gate_error_2q_mean": None,
            "circuit": None,
            "estimated_circuit_fidelity": None,
            "dominant_error_source": None,
        }

    # pull circuit structure first: it tells us which qubits the run
    # actually touched, which sets the calibration scope
    #
    cs = circuit_structure(t)
    prim = cs.get("primary")
    active = set(prim.get("active_qubits") or []) if prim else set()

    # scope helpers: prefer values on the active qubits, fall back to
    # the whole device when the filter would leave nothing
    #
    def _scoped_qubit_values(values: dict) -> list:
        if active:
            scoped = [v for k, v in values.items() if k in active]
            if scoped:
                return scoped
        return list(values.values())

    def _scoped_gate_errors(pairs: list) -> list:
        if active:
            scoped = [e for qs, e in pairs if set(qs) <= active]
            if scoped:
                return scoped
        return [e for _, e in pairs]

    # parse calibration and compute scoped statistics
    #
    p = _parse_calibration(cal)
    t1_vals = _scoped_qubit_values(p["t1"])
    t2_vals = _scoped_qubit_values(p["t2"])
    ro_vals = _scoped_qubit_values(p["readout"])
    e1_vals = _scoped_gate_errors(p["err_1q"])
    e2_vals = _scoped_gate_errors(p["err_2q"])
    mean_1q = float(np.mean(e1_vals)) if e1_vals else 0.0
    mean_2q = float(np.mean(e2_vals)) if e2_vals else 0.0
    mean_ro = float(np.mean(ro_vals)) if ro_vals else 0.0
    scope = "active_qubits" if active else "device_wide"

    # weight the errors by circuit usage
    #
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
        "scope": scope,
        "active_qubits": sorted(active),
        "t1_us": _stats(t1_vals),
        "t2_us": _stats(t2_vals),
        "readout_error": _stats(ro_vals),
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
