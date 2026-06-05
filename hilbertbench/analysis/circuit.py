#!/usr/bin/env python
#
# file: hilbertbench/analysis/circuit.py
#
# revision history:
#  20260604 (am): cleaned up to project coding standards
#
# Circuit-structure diagnostics. Parses the stored OpenQASM circuit(s)
# to report the structural facts that drive hardware cost and
# expressibility: qubit count, circuit depth, gate composition,
# entangling-gate fraction, and the number of trainable parameters.
#
# This is evidence, not interpretation — it reports what the circuit
# IS. It forms the structural foundation for both encoding (Axis 1)
# and ansatz (Axis 2) reasoning. A faithful encoding-injectivity
# analysis additionally needs the raw classical inputs, which the
# recorder does not yet capture.
#
#   from hilbertbench.analysis import circuit_structure
#   circuit_structure("runs/20260605_xxx")
#------------------------------------------------------------------------------

# future imports must come first
#
from __future__ import annotations

# import system modules
#
import os
import re
from typing import Any

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

# matches qubit index references of the form q[N]
#
_WIRE_RE = re.compile(r"q\[(\d+)\]")

# matches positional parameter placeholders from _qasm_to_template
#
_PLACEHOLDER_RE = re.compile(r"_p\d+")

# matches qubit-register declarations in QASM2 and QASM3 syntax
#
_NUM_QUBITS_RE = re.compile(
    r"(?:qreg\s+\w+\[(\d+)\]|qubit\[(\d+)\])"
)

# line prefixes that are not compute gates and must be skipped
#
_SKIP_PREFIXES = (
    "openqasm", "include", "qreg", "creg", "qubit", "bit",
    "input", "output", "gate ", "//", "measure", "barrier", "reset",
)

#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------

def _parse_qasm(qasm: str) -> dict[str, Any]:
    """
    function: _parse_qasm

    arguments:
     qasm: an OpenQASM string (concrete or templated)

    return:
     a dict of structural metrics:
      num_qubits          qubit count from the register declaration
      depth               circuit depth (greedy ASAP layer assignment)
      total_gates         single + entangling gate count
      single_qubit_gates  gates acting on exactly one qubit
      entangling_gates    gates acting on two or more qubits
      entangling_fraction entangling / total (0 if no gates)
      gate_counts         dict of gate_name -> count
      num_parameters      distinct parameter placeholders
      num_measurements    measure instructions

    description:
     Parses one OpenQASM string line-by-line. Non-gate lines are
     skipped. Circuit depth is computed using greedy ASAP layering:
     each gate is placed in the earliest layer where all its qubits
     are free.
    """

    # initialise counters and layering state
    #
    num_qubits = 0
    m = _NUM_QUBITS_RE.search(qasm)
    if m:
        num_qubits = int(m.group(1) or m.group(2))

    gate_counts: dict[str, int] = {}
    entangling = 0
    single = 0
    measurements = 0
    qubit_layer: dict[int, int] = {}
    depth = 0

    # iterate over lines and classify each gate
    #
    for raw in qasm.splitlines():
        line = raw.strip()
        if not line:
            continue

        # count measurement instructions separately
        #
        low = line.lower()
        if low.startswith("measure"):
            measurements += 1
            continue

        # skip non-gate lines (headers, declarations, barriers)
        #
        if low.startswith(_SKIP_PREFIXES):
            continue

        # extract qubit indices from the line
        #
        wires = [int(w) for w in _WIRE_RE.findall(line)]
        if not wires:
            continue

        # tally the gate name
        #
        gate_name = re.split(r"[\s(]", line, 1)[0].lower()
        gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1

        # classify as single-qubit or entangling
        #
        distinct = set(wires)
        if len(distinct) >= 2:
            entangling += 1
        else:
            single += 1

        # place the gate in the earliest available layer (ASAP)
        #
        layer = 1 + max(
            (qubit_layer.get(w, 0) for w in distinct), default=0
        )
        for w in distinct:
            qubit_layer[w] = layer
        depth = max(depth, layer)

    # count distinct parameter placeholders in the circuit
    #
    total_gates = single + entangling
    num_params = len(set(_PLACEHOLDER_RE.findall(qasm)))

    # Qiskit QASM3 uses 'input float[..] name;' for free parameters
    #
    if num_params == 0:
        num_params = sum(
            1 for ln in qasm.splitlines()
            if ln.strip().lower().startswith("input")
        )

    # exit gracefully
    #
    return {
        "num_qubits":          num_qubits,
        "depth":               depth,
        "total_gates":         total_gates,
        "single_qubit_gates":  single,
        "entangling_gates":    entangling,
        "entangling_fraction": (
            entangling / total_gates if total_gates else 0.0
        ),
        "gate_counts":         gate_counts,
        "num_parameters":      num_params,
        "num_measurements":    measurements,
    }
#
# end of function


def circuit_structure(trace: TraceLike) -> dict[str, Any]:
    """
    function: circuit_structure

    arguments:
     trace: a HilbertTrace or run-directory path

    return:
     a dict with keys:
      status        'OK' | 'No QASM circuit recorded'
      num_circuits  number of distinct circuit_qasm artifacts found
      circuits      list of per-circuit structure dicts (_parse_qasm)
      primary       structure of the largest circuit by gate count,
                    or None if no circuits were found

    description:
     Resolves all circuit_qasm artifacts from the trace catalog and
     parses each with _parse_qasm. The 'primary' field surfaces the
     dominant circuit for callers that expect a single circuit per
     trace (the common case).
    """

    # resolve the trace object
    #
    t = as_trace(trace)

    # find all circuit_qasm artifact refs in the catalog
    #
    qasm_refs = [
        ref for ref, meta in t.catalog.items()
        if meta.get("kind") == "circuit_qasm"
    ]

    # return early when no QASM artifacts exist
    #
    if not qasm_refs:
        return {
            "status":       "No QASM circuit recorded",
            "num_circuits": 0,
            "circuits":     [],
            "primary":      None,
        }

    # parse every QASM artifact
    #
    structures = []
    for ref in qasm_refs:
        qasm = t._resolve_ref({}, ref)
        if isinstance(qasm, str):
            structures.append(_parse_qasm(qasm))

    # return early when all resolves failed
    #
    if not structures:
        return {
            "status":       "No QASM circuit recorded",
            "num_circuits": 0,
            "circuits":     [],
            "primary":      None,
        }

    # surface the largest circuit as 'primary'
    #
    primary = max(structures, key=lambda s: s["total_gates"])

    # exit gracefully
    #
    return {
        "status":       "OK",
        "num_circuits": len(structures),
        "circuits":     structures,
        "primary":      primary,
    }
#
# end of function

#
# end of file
