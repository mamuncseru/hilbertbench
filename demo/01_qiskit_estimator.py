#!/usr/bin/env python
#
# file: demo/01_qiskit_estimator.py
#
# revision history:
#  20260605 (am): initial version
#
# VQE with Qiskit V2 Estimator + HilbertBench passive recording.
#
# Runs a 20-step gradient-free VQE on a 2-qubit hardware-efficient
# ansatz (RY+CNOT) minimising the ZZ expectation value. Every circuit
# evaluation is intercepted and recorded by HilbertEstimatorProxy
# without any changes to the optimizer or circuit code.
#
# After the tape closes the JSONL trace is converted to Parquet and
# the built-in barren-plateau analyzer is run on the recorded outcomes.
#
# Prerequisites:
#   pip install hilbertbench[qiskit,storage] scipy
#
# Usage:
#   python demo/01_qiskit_estimator.py
#------------------------------------------------------------------------------

# import system modules
#
import os

# import third-party modules
#
import numpy as np
from scipy.optimize import minimize

# import qiskit modules
#
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import SparsePauliOp

# import hilbertbench modules
#
from hilbertbench import HilbertTrace
from hilbertbench.integrations.qiskit import HilbertEstimatorProxy
from hilbertbench.recorder.tape import HilbertTape
from hilbertbench.recorder.storage.writer import convert_trace_to_parquet
from hilbertbench.analysis import detect_barren_plateau

#------------------------------------------------------------------------------
#
# global variables are listed here
#
#------------------------------------------------------------------------------

# set the filename using basename
#
__FILE__ = os.path.basename(__file__)

# output root — timestamped run directory is created inside
#
RUNS_DIR = "runs/estimator_vqe"

# COBYLA iteration budget (each iteration = one recorded span)
#
N_ITER = 20

#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------

def build_ansatz() -> tuple:
    """
    function: build_ansatz

    arguments:
     none

    return:
     (circuit, observable) — a parameterized QuantumCircuit and a
     SparsePauliOp ready for Estimator.run()

    description:
     Hardware-efficient 2-qubit ansatz:
       RY(θ₀) ⊗ RY(θ₁) → CNOT(0→1)
     Observable: ZZ. Ground-state energy is -1 at θ = [π, 0].
    """

    # build the parameterized ansatz
    #
    theta = ParameterVector("θ", 2)
    qc = QuantumCircuit(2)
    qc.ry(theta[0], 0)
    qc.ry(theta[1], 1)
    qc.cx(0, 1)

    # ZZ observable — minimize for antiferromagnetic ground state
    #
    observable = SparsePauliOp("ZZ")

    # exit gracefully
    #
    return qc, observable
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
     Opens a HilbertTape, wraps StatevectorEstimator in
     HilbertEstimatorProxy, then runs scipy COBYLA for N_ITER steps.
     Each cost() call executes normally on the real simulator — the
     proxy intercepts the result and records a span containing the
     circuit QASM, expectation value, parameters, and observable.

     After the tape closes, the trace is sealed, converted to Parquet,
     and a trainability report is printed from the recorded outcomes.
    """

    # print run header
    #
    sep = "-" * 60
    print(f"\n[{__FILE__}]  HilbertBench — Qiskit Estimator (VQE)")
    print(sep)

    # build circuit and observable
    #
    circuit, observable = build_ansatz()
    x0 = np.random.uniform(0.0, 2.0 * np.pi, 2)
    print(f"  Ansatz     : 2-qubit RY+CNOT, 2 parameters")
    print(f"  Observable : ZZ  (ground state energy = -1.0)")
    print(f"  Optimizer  : COBYLA, {N_ITER} iterations")
    print(f"  Initial θ  : {x0.round(4)}")
    print(sep)

    # open the tape and run the optimizer
    #
    energies: list[float] = []

    with HilbertTape(
        RUNS_DIR,
        tags={"demo": "qiskit_estimator", "algorithm": "vqe"},
    ) as tape:

        # wrap StatevectorEstimator — no other code changes required
        #
        estimator = HilbertEstimatorProxy(tape)

        def cost(x: np.ndarray) -> float:

            # run one evaluation; proxy records the span automatically
            #
            pv = x.reshape(1, -1)
            job = estimator.run([(circuit, observable, pv)])
            energy = float(np.ravel(job.result()[0].data.evs)[0])
            energies.append(energy)

            # progress report every 5 steps
            #
            step = len(energies)
            if step == 1 or step % 5 == 0 or step == N_ITER:
                print(
                    f"  step {step:3d}/{N_ITER}"
                    f"  energy = {energy:+.6f}"
                )
            return energy

        # run COBYLA — every call to cost() is intercepted and recorded
        #
        result = minimize(
            cost,
            x0,
            method="COBYLA",
            options={"maxiter": N_ITER, "rhobeg": 0.5},
        )

    # convert the JSONL trace to Parquet for analysis tools
    #
    parquet_path = convert_trace_to_parquet(tape.dir_path)

    # load the sealed trace and run the trainability analyzer
    #
    trace = HilbertTrace(tape.dir_path)
    bp = detect_barren_plateau(trace)

    print(sep)
    print(f"  Trace dir     : {tape.dir_path}")
    print(f"  Parquet       : {parquet_path.name}")
    print(f"  Trace status  : {trace.status}")
    print(f"  Spans recorded: {len(trace)}")
    print(f"  Final energy  : {energies[-1]:+.6f}  (target -1.0)")
    print(f"  Optimal θ     : {result.x.round(4)}")
    print(f"  Trainability  : {bp['status']}")
    print(f"  Variance      : {bp['variance']:.6f}")
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
