#!/usr/bin/env python
#
# file: demo/03_qiskit_ibm.py
#
# revision history:
#  20260605 (am): initial version
#
# VQE on IBM Quantum real hardware + HilbertBench passive recording.
#
# Same 2-qubit VQE as 01_qiskit_estimator.py, but targets a real IBM
# Quantum backend via qiskit_ibm_runtime. The circuit is transpiled
# to the hardware's ISA (instruction set architecture) before being
# handed to the proxy.
#
# HilbertEstimatorProxy wraps the IBM EstimatorV2 transparently —
# no changes to the VQE loop are needed.  After the tape closes the
# trace is sealed and converted to Parquet, exactly like the simulator
# demo.
#
# Prerequisites:
#   pip install hilbertbench[qiskit,storage] qiskit-ibm-runtime scipy
#
# Setup:
#   Set IBM_TOKEN below to your IBM Quantum API token.
#   Generate one at: https://quantum.ibm.com/
#
# Usage:
#   python demo/03_qiskit_ibm.py
#------------------------------------------------------------------------------

# import system modules
#
import os
import sys

# import third-party modules
#
import numpy as np
from scipy.optimize import minimize

# import qiskit modules
#
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import (
    generate_preset_pass_manager,
)

# import IBM runtime modules
#
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2

# import hilbertbench modules
#
from hilbertbench import HilbertTrace
from hilbertbench.integrations.qiskit import HilbertEstimatorProxy
from hilbertbench.recorder.tape import HilbertTape
from hilbertbench.recorder.storage.writer import convert_trace_to_parquet

#------------------------------------------------------------------------------
#
# global variables are listed here
#
#------------------------------------------------------------------------------

# set the filename using basename
#
__FILE__ = os.path.basename(__file__)

# ── SET YOUR IBM QUANTUM TOKEN HERE ──────────────────────────────────────────
IBM_TOKEN = "YOUR_IBM_QUANTUM_TOKEN_HERE"
# ─────────────────────────────────────────────────────────────────────────────

# output root — timestamped run directory is created inside
#
RUNS_DIR = "runs/ibm_vqe"

# number of VQE steps (keep low on real hardware to limit queue time)
#
N_ITER = 5

#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------

def build_abstract_ansatz() -> tuple:
    """
    function: build_abstract_ansatz

    arguments:
     none

    return:
     (circuit, observable) — abstract (un-transpiled) circuit and
     SparsePauliOp

    description:
     Same hardware-efficient ansatz as the simulator demo:
       RY(θ₀) ⊗ RY(θ₁) → CNOT(0→1)
     Observable: ZZ.  Ground-state energy = -1.

     The circuit is returned in abstract form — transpilation to the
     hardware ISA happens in main() after the backend is selected.
    """

    # build the abstract ansatz
    #
    theta = ParameterVector("θ", 2)
    qc = QuantumCircuit(2)
    qc.ry(theta[0], 0)
    qc.ry(theta[1], 1)
    qc.cx(0, 1)

    # exit gracefully
    #
    return qc, SparsePauliOp("ZZ")
#
# end of function


def main() -> None:
    """
    function: main

    arguments:
     none

    return:
     none

    description:
     Authenticates with IBM Quantum, selects the least-busy real
     backend with at least 5 qubits, and transpiles the abstract VQE
     circuit to the hardware ISA.

     A HilbertTape is opened, the IBM EstimatorV2 is wrapped in
     HilbertEstimatorProxy, and N_ITER steps of scipy COBYLA are
     run.  Every hardware job result is intercepted, the ISA-QASM is
     stored once in the file store (content-addressed, deduplicated),
     and the expectation value is stored inline.

     After the tape closes the trace is sealed, converted to Parquet,
     and the span count and final energy are printed.
    """

    # guard: require the user to set their token
    #
    if IBM_TOKEN == "YOUR_IBM_QUANTUM_TOKEN_HERE":
        print(
            f"[{__FILE__}] ERROR: set IBM_TOKEN at the top of this file."
        )
        sys.exit(1)

    sep = "-" * 60
    print(f"\n[{__FILE__}]  HilbertBench — Qiskit IBM Real Hardware (VQE)")
    print(sep)

    # authenticate and select the least-busy real backend
    #
    print("  Authenticating with IBM Quantum ...")
    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=IBM_TOKEN,
    )
    backend = service.least_busy(
        simulator=False,
        operational=True,
        min_num_qubits=5,
    )
    print(f"  Backend        : {backend.name}")
    print(f"  Pending jobs   : {backend.status().pending_jobs}")

    # build abstract circuit and transpile to hardware ISA
    #
    abstract_qc, abstract_obs = build_abstract_ansatz()
    pm = generate_preset_pass_manager(
        optimization_level=1,
        backend=backend,
    )
    isa_qc = pm.run(abstract_qc)
    isa_obs = abstract_obs.apply_layout(isa_qc.layout)

    print(f"  Abstract depth : {abstract_qc.depth()}")
    print(f"  ISA depth      : {isa_qc.depth()}")
    print(f"  Physical qubits: {isa_qc.layout.final_index_layout()}")
    print(f"  Optimizer      : COBYLA, {N_ITER} iterations")
    print(sep)

    # open the tape and run the VQE
    #
    x0 = np.random.uniform(0.0, 2.0 * np.pi, 2)
    energies: list[float] = []

    with HilbertTape(
        RUNS_DIR,
        tags={
            "demo":    "qiskit_ibm",
            "backend": backend.name,
            "algorithm": "vqe",
        },
    ) as tape:

        # wrap IBM EstimatorV2 — no other code changes required
        #
        runtime_estimator = EstimatorV2(mode=backend)
        estimator = HilbertEstimatorProxy(
            tape,
            real_estimator=runtime_estimator,
        )

        def cost(x: np.ndarray) -> float:
            pv = x.reshape(1, -1)
            job = estimator.run([(isa_qc, isa_obs, pv)])
            energy = float(np.ravel(job.result()[0].data.evs)[0])
            energies.append(energy)
            step = len(energies)
            print(
                f"  step {step:2d}/{N_ITER}"
                f"  energy = {energy:+.6f}"
            )
            return energy

        minimize(
            cost,
            x0,
            method="COBYLA",
            options={"maxiter": N_ITER, "rhobeg": 0.5},
        )

    # convert to Parquet and report
    #
    parquet_path = convert_trace_to_parquet(tape.dir_path)
    trace = HilbertTrace(tape.dir_path)

    print(sep)
    print(f"  Trace dir     : {tape.dir_path}")
    print(f"  Parquet       : {parquet_path.name}")
    print(f"  Trace status  : {trace.status}")
    print(f"  Spans recorded: {len(trace)}")
    if energies:
        print(f"  Final energy  : {energies[-1]:+.6f}")
    print(sep)
#
# end of function

#------------------------------------------------------------------------------
#
# main entry point
#
#------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#
# end of file
