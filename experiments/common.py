#!/usr/bin/env python
#
# file: experiments/common.py
#
# revision history:
#  20260610 (am): initial version
#
# Shared infrastructure for the HilbertBench paper experiments.
# Provides the ansatz family builders, the batched cost-landscape
# recorder, and result-file conventions used by every study script.
#
# Design rules:
#  - every random draw is seeded and the seed is stored in the result
#  - parameter sets are batched into single PUBs (broadcasting), so
#    the same code path is cheap on simulators and on paid hardware
#  - each study writes one JSON result file that its figure script
#    consumes; traces remain the primary evidence
#------------------------------------------------------------------------------

# future imports must come first
#
from __future__ import annotations

# import system modules
#
import json
import os
from pathlib import Path
from typing import Any, Optional

# import third-party modules
#
import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import SparsePauliOp

# import hilbertbench modules
#
from hilbertbench.integrations.qiskit import HilbertEstimatorProxy
from hilbertbench.models import Mode
from hilbertbench.recorder.tape import HilbertTape

#------------------------------------------------------------------------------
#
# global variables are listed here
#
#------------------------------------------------------------------------------

# set the filename using basename
#
__FILE__ = os.path.basename(__file__)

# root for experiment outputs (traces + result JSON files)
#
RESULTS_ROOT = Path(__file__).parent / "results"
TRACES_ROOT = Path(__file__).parent / "traces"

# parameter sets per PUB; amortises overhead on sims and hardware
#
DEFAULT_BATCH = 50

#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------

def build_ansatz(
    n_qubits: int,
    n_layers: int,
    entanglement: str = "linear",
) -> tuple:
    """
    function: build_ansatz

    arguments:
     n_qubits:     circuit width
     n_layers:     number of rotation+entangling layers
     entanglement: 'linear' | 'ring' | 'full' CNOT topology

    return:
     (circuit, n_params) — a hardware-efficient ansatz with RY+RZ
     rotations per qubit per layer and the requested entangling map

    description:
     The three ansatz families of Study A. 'linear' is the standard
     nearest-neighbour ladder, 'ring' closes the ladder, and 'full'
     applies all-to-all CNOTs (deepest, most expressive).
    """

    # build the parameter vector and circuit
    #
    n_params = n_layers * n_qubits * 2
    theta = ParameterVector("t", n_params)
    qc = QuantumCircuit(n_qubits)

    # resolve the entangling pairs for one layer
    #
    if entanglement == "linear":
        pairs = [(q, q + 1) for q in range(n_qubits - 1)]
    elif entanglement == "ring":
        pairs = [(q, (q + 1) % n_qubits) for q in range(n_qubits)]
        if n_qubits <= 2:
            pairs = [(0, 1)] if n_qubits == 2 else []
    elif entanglement == "full":
        pairs = [
            (a, b)
            for a in range(n_qubits)
            for b in range(a + 1, n_qubits)
        ]
    else:
        raise ValueError(f"unknown entanglement '{entanglement}'")

    # lay down rotation + entangling layers
    #
    idx = 0
    for _ in range(n_layers):
        for q in range(n_qubits):
            qc.ry(theta[idx], q)
            idx += 1
            qc.rz(theta[idx], q)
            idx += 1
        for a, b in pairs:
            qc.cx(a, b)

    # exit gracefully
    #
    return qc, n_params
#
# end of function


def pair_observable(n_qubits: int) -> SparsePauliOp:
    """
    function: pair_observable

    arguments:
     n_qubits: circuit width

    return:
     the ZZ observable on the first qubit pair (identity elsewhere)

    description:
     The cost observable used across studies; ZZ on a fixed pair
     exhibits the barren plateau sharply (matches demo 13).
    """

    # exit gracefully
    #
    if n_qubits < 2:
        return SparsePauliOp("Z")
    return SparsePauliOp("Z" * 2 + "I" * (n_qubits - 2))
#
# end of function


def sample_landscape(
    out_root: Path,
    n_qubits: int,
    n_layers: int,
    entanglement: str,
    n_samples: int,
    seed: int,
    tags: Optional[dict] = None,
    estimator: Any = None,
    precision: Optional[float] = None,
    batch_size: int = DEFAULT_BATCH,
) -> Path:
    """
    function: sample_landscape

    arguments:
     out_root:     directory under which the run directory is created
     n_qubits:     circuit width
     n_layers:     circuit depth (ansatz layers)
     entanglement: ansatz family ('linear' | 'ring' | 'full')
     n_samples:    number of uniform random parameter points
     seed:         RNG seed for the parameter draws
     tags:         trace tags (merged with the defaults)
     estimator:    optional real V2 estimator (None = statevector)
     precision:    optional target precision recorded in each PUB
     batch_size:   parameter sets per PUB (broadcasting)

    return:
     the run directory of the sealed trace

    description:
     Records the cost landscape of the ansatz at uniformly random
     parameter points through HilbertEstimatorProxy. This is the
     active-mode random sampling required for variance / barren-
     plateau characterisation (McClean et al. 2018). Samples are
     batched into PUBs so the recording is hardware-affordable.
    """

    # build the ansatz, observable, and parameter draws
    #
    qc, n_params = build_ansatz(n_qubits, n_layers, entanglement)
    observable = pair_observable(n_qubits)
    rng = np.random.default_rng(seed)
    params = rng.uniform(0.0, 2.0 * np.pi, (n_samples, n_params))

    # record the landscape batch by batch
    #
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    run_tags = {"protocol": "landscape_v1", "seed": str(seed)}
    run_tags.update(tags or {})
    with HilbertTape(out_root, mode=Mode.active, tags=run_tags) as tape:
        proxy = HilbertEstimatorProxy(tape, real_estimator=estimator)
        for start in range(0, n_samples, batch_size):
            batch = params[start:start + batch_size]
            pub = (
                (qc, observable, batch)
                if precision is None
                else (qc, observable, batch, precision)
            )
            proxy.run([pub]).result()

    # exit gracefully
    #
    return tape.dir_path
#
# end of function


def save_result(study: str, payload: dict) -> Path:
    """
    function: save_result

    arguments:
     study:   study name (e.g. 'study_a'); names the result file
     payload: JSON-serialisable result dictionary

    return:
     the path of the written result file

    description:
     Writes results/<study>/results.json, creating directories as
     needed. Figure scripts read these files; nothing else does.
    """

    # write the result file
    #
    out_dir = RESULTS_ROOT / study
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.json"
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")

    # exit gracefully
    #
    return path
#
# end of function

#
# end of file
