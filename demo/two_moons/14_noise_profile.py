#!/usr/bin/env python
#
# file: demo/two_moons/14_noise_profile.py
#
# revision history:
#  20260605 (am): initial version
#
# Noise-axis diagnostics (proposal Section 2.5, Axis 5). HilbertBench
# snapshots backend calibration (T1, T2, readout error, gate errors) at
# execution time; the noise_profile analyzer reads that snapshot and,
# weighted by the recorded circuit structure, estimates the run's
# circuit fidelity and identifies the dominant error source.
#
# This demo runs circuits of increasing depth on a noisy fake backend
# (FakeManilaV2) and shows how the estimated fidelity falls and the
# dominant error source shifts from readout to two-qubit gates as depth
# grows — the "interaction with circuit depth" the proposal calls for.
#
# Requires qiskit-ibm-runtime (for the fake backend with calibration).
#
# Usage:
#   python demo/two_moons/14_noise_profile.py
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
from qiskit.primitives import BackendEstimatorV2
from qiskit_ibm_runtime.fake_provider import FakeManilaV2

# import hilbertbench modules
#
from hilbertbench.recorder.tape import HilbertTape
from hilbertbench.integrations.qiskit import HilbertEstimatorProxy
from hilbertbench.analysis import noise_profile

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

# output root for the recorded traces
#
RUNS_DIR = Path(__file__).parent / "runs"

# qubit width and the circuit depths (rep counts) to sweep
#
N_QUBITS = 3
DEPTHS = [1, 3, 7, 15]

# section separator
#
SEP = "=" * 70

#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------

def record_circuit(reps: int, seed: int) -> Path:
    """
    function: record_circuit

    arguments:
     reps: number of rotation+entangling layers (circuit depth)
     seed: RNG seed for the random parameter point

    return:
     the run directory of the sealed trace

    description:
     Records one execution of a depth-`reps` hardware-efficient ansatz
     on FakeManilaV2 via HilbertEstimatorProxy. The calibration snapshot
     is captured automatically on the first run() call.
    """

    # build the ansatz of the requested depth
    #
    n_params = N_QUBITS * reps
    theta = ParameterVector("t", n_params)
    qc = QuantumCircuit(N_QUBITS)
    idx = 0
    for _ in range(reps):
        for q in range(N_QUBITS):
            qc.ry(theta[idx], q)
            idx += 1
        for q in range(N_QUBITS - 1):
            qc.cx(q, q + 1)
    obs = SparsePauliOp("Z" + "I" * (N_QUBITS - 1))

    # record one execution on the noisy fake backend
    #
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    est = BackendEstimatorV2(backend=FakeManilaV2())
    with HilbertTape(
        RUNS_DIR,
        tags={"demo": "noise_profile", "depth": str(reps)},
    ) as tape:
        proxy = HilbertEstimatorProxy(tape, real_estimator=est)
        params = np.array([rng.random(n_params)])
        proxy.run([(qc, obs, params)]).result()

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
     Records the device calibration once, prints its summary, then
     sweeps circuit depth and tabulates the estimated fidelity and
     dominant error source for each depth.
    """

    # print the header
    #
    print("\n" + SEP)
    print("  HILBERTBENCH - NOISE PROFILE (Axis 5)")
    print("  correlating circuit fidelity with device calibration")
    print(SEP)

    # record a shallow run and print the device calibration summary
    #
    first = noise_profile(record_circuit(DEPTHS[0], seed=DEPTHS[0]))
    print(f"\n  Device: {first['backend_name']}  "
          f"({first['num_qubits_calibrated']} qubits calibrated)")
    print(f"    T1 (us)        : mean {first['t1_us']['mean']:.1f}  "
          f"min {first['t1_us']['min']:.1f}")
    print(f"    T2 (us)        : mean {first['t2_us']['mean']:.1f}")
    print(f"    readout error  : mean {first['readout_error']['mean']:.4f}")
    print(f"    1q gate error  : {first['gate_error_1q_mean']:.5f}")
    print(f"    2q gate error  : {first['gate_error_2q_mean']:.5f}")

    # sweep depth and tabulate the noise estimate
    #
    print(f"\n  {'depth':>5} | {'2q gates':>8} | {'est. fidelity':>13} | "
          f"{'dominant error':>16} | verdict")
    print("  " + "-" * 78)
    for reps in DEPTHS:
        r = noise_profile(record_circuit(reps, seed=reps))
        c = r["circuit"]
        short = r["status"].split(" (")[0]
        print(f"  {c['depth']:>5} | {c['entangling_gates']:>8} | "
              f"{r['estimated_circuit_fidelity']:>13.4f} | "
              f"{r['dominant_error_source']:>16} | {short}")

    # print the conclusion
    #
    print("\n" + SEP)
    print("  As depth grows, estimated fidelity falls and the dominant")
    print("  error source shifts from readout to two-qubit gates. A loss")
    print("  spike can now be correlated with device noise, not just the")
    print("  model parameters.")
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
