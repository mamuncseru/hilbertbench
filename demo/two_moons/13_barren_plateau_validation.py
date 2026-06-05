#!/usr/bin/env python
#
# file: demo/two_moons/13_barren_plateau_validation.py
#
# revision history:
#  20260605 (am): initial version
#
# Phenomenological validation (proposal Section 2.6): plant a known QML
# phenomenon — the barren plateau — in synthetic ground-truth circuits
# and confirm HilbertBench's detector attributes it correctly.
#
# McClean et al. (2018): for random hardware-efficient ansatze the
# variance of the cost landscape vanishes exponentially with qubit
# count. We sweep qubit width, record each cost landscape via active
# random sampling (HilbertEstimatorProxy), then run the built-in
# detect_barren_plateau analyzer on each sealed trace and compare its
# verdict against the planted ground truth.
#
# Success criterion: the detector flags the wide/deep circuits as
# barren and the narrow/shallow ones as trainable, with no knowledge of
# how the traces were produced (blind diagnosis).
#
# Usage:
#   python demo/two_moons/13_barren_plateau_validation.py
#------------------------------------------------------------------------------

# import system modules
#
import os
import warnings
from pathlib import Path

# import third-party modules
#
import numpy as np

# import qiskit modules
#
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import SparsePauliOp

# import hilbertbench modules
#
from hilbertbench.recorder.tape import HilbertTape
from hilbertbench.integrations.qiskit import HilbertEstimatorProxy
from hilbertbench.models import Mode
from hilbertbench.analysis import detect_barren_plateau

# silence framework deprecation chatter for a clean report
#
warnings.filterwarnings("ignore")

#------------------------------------------------------------------------------
#
# global variables are listed here
#
#------------------------------------------------------------------------------

# set the filename using basename
#
__FILE__ = os.path.basename(__file__)

# output root for the recorded landscape traces
#
RUNS_DIR = Path(__file__).parent / "runs"

# random parameter samples per landscape
#
N_SAMPLES = 120

# qubit widths to sweep; depth scales with width to deepen the plateau
#
WIDTHS = [2, 4, 6, 8]

# section separator
#
SEP = "=" * 70

#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------

def build_hea(n_qubits: int, n_layers: int) -> tuple:
    """
    function: build_hea

    arguments:
     n_qubits: number of qubits (circuit width)
     n_layers: number of rotation+entangling layers (circuit depth)

    return:
     (circuit, observable, n_params) — a parameterized hardware-
     efficient ansatz, the ZZ observable on the first two qubits, and
     the parameter count

    description:
     Standard hardware-efficient ansatz: each layer applies RY and RZ
     on every qubit followed by a linear CNOT ladder. A global-style ZZ
     observable is used because it exhibits the barren plateau most
     sharply.
    """

    # build the parameter vector and circuit
    #
    n_params = n_layers * n_qubits * 2
    theta = ParameterVector("t", n_params)
    qc = QuantumCircuit(n_qubits)

    # lay down rotation + entangling layers
    #
    idx = 0
    for _ in range(n_layers):
        for q in range(n_qubits):
            qc.ry(theta[idx], q)
            idx += 1
            qc.rz(theta[idx], q)
            idx += 1
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)

    # ZZ observable on the first two qubits
    #
    observable = SparsePauliOp("Z" * 2 + "I" * (n_qubits - 2))

    # exit gracefully
    #
    return qc, observable, n_params


def record_landscape(n_qubits: int, n_layers: int, seed: int) -> Path:
    """
    function: record_landscape

    arguments:
     n_qubits: circuit width
     n_layers: circuit depth
     seed:     RNG seed for parameter sampling

    return:
     the run directory of the sealed trace

    description:
     Opens an active-mode tape, wraps the estimator with
     HilbertEstimatorProxy, and records the cost (ZZ expectation value)
     at N_SAMPLES uniformly-random parameter points. This is a
     controlled, opt-in active diagnostic — the random sampling needed
     to characterise the cost landscape.
    """

    # build the ansatz and prepare the sampler
    #
    qc, observable, n_params = build_hea(n_qubits, n_layers)
    rng = np.random.default_rng(seed)
    planted = "barren_plateau" if n_qubits >= 8 else "trainable"

    # record the cost landscape under random parameters
    #
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with HilbertTape(
        RUNS_DIR,
        mode=Mode.active,
        tags={"experiment": "barren_plateau", "planted": planted,
              "n_qubits": str(n_qubits)},
    ) as tape:
        estimator = HilbertEstimatorProxy(tape)
        for _ in range(N_SAMPLES):
            params = rng.uniform(0.0, 2.0 * np.pi, (1, n_params))
            estimator.run([(qc, observable, params)]).result()

    # exit gracefully
    #
    return tape.dir_path


def main() -> None:
    """
    function: main

    arguments:
     none

    return:
     none

    description:
     Sweeps qubit width, records each cost landscape, runs the
     detector blind on the sealed traces, and tabulates the verdict
     against the planted ground truth. Reports whether the instrument
     correctly attributed the barren plateau.
    """

    # print the experiment header
    #
    print("\n" + SEP)
    print("  HILBERTBENCH - PHENOMENOLOGICAL VALIDATION")
    print("  planting a barren plateau and confirming blind detection")
    print(SEP)
    print("\n  Random hardware-efficient ansatze; cost = <Z0 Z1>.")
    print("  Ground truth: variance collapses exponentially with width,")
    print("  so wide/deep circuits are barren and narrow ones trainable.\n")

    # sweep widths, record landscapes, and collect detector verdicts
    #
    print(f"  {'qubits':>6} | {'depth':>5} | {'planted':>13} | "
          f"{'variance':>9} | {'detector verdict':>22} | match")
    print("  " + "-" * 78)

    all_correct = True
    for n in WIDTHS:
        n_layers = n * 3
        planted = "Barren Plateau" if n >= 8 else "Trainable"
        run_dir = record_landscape(n, n_layers, seed=n)

        # the detector sees only the sealed trace, not the ground truth
        #
        result = detect_barren_plateau(run_dir)
        verdict = result["status"]
        correct = (planted == "Barren Plateau") == \
                  (verdict == "Barren Plateau Detected")
        all_correct = all_correct and correct

        print(f"  {n:>6} | {n_layers:>5} | {planted:>13} | "
              f"{result['variance']:>9.6f} | {verdict:>22} | "
              f"{'YES' if correct else 'NO'}")

    # print the conclusion
    #
    print("\n" + SEP)
    if all_correct:
        print("  RESULT: the detector attributed every planted case correctly.")
        print("  HilbertBench reproduces the barren-plateau phenomenon from")
        print("  trace evidence alone, with no access to the generating code.")
    else:
        print("  RESULT: one or more cases were misattributed - investigate.")
    print(SEP + "\n")

#------------------------------------------------------------------------------
#
# main entry point
#
#------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#
# end of file
