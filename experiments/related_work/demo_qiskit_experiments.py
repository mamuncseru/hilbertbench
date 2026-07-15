#!/usr/bin/env python
#
# file: experiments/related_work/demo_qiskit_experiments.py
#
# What Qiskit Experiments gives you for post-hoc failure attribution —
# and what it is actually for, done properly.
#
# Qiskit Experiments CHARACTERIZES a processor: randomized
# benchmarking, T1/T2, calibrations. It answers "how good is this
# device?" by running DEDICATED circuits designed for that question —
# extra quantum executions that have nothing to do with your
# algorithm. It does not ingest your run, and it cannot say why your
# training failed; it can only tell you properties of the hardware it
# ran its own circuits on.
#
#   .venv/bin/python demo_qiskit_experiments.py     (isolated env)
#------------------------------------------------------------------------------
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_experiments.library import StandardRB


def main() -> int:
    # a noisy device stand-in
    #
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(
        depolarizing_error(0.002, 1), ["sx", "x", "rz"])
    backend = AerSimulator(noise_model=noise)

    # single-qubit randomized benchmarking, exactly as intended
    #
    exp = StandardRB(physical_qubits=(0,),
                     lengths=[10, 50, 100, 200],
                     num_samples=3, seed=20260715)
    n_circuits = len(exp.circuits())
    print(f"StandardRB prepared {n_circuits} DEDICATED circuits "
          "(none of them is your algorithm)")

    data = exp.run(backend, shots=1024).block_for_results()
    epc = data.analysis_results("EPC").value
    print(f"measured error per Clifford: {epc}")

    print(f"""
WHAT THIS SHOWS
  - Qiskit Experiments did its job: {n_circuits} purpose-built
    circuits were executed and a device property (EPC) was estimated
  - every one of those executions is EXTRA quantum cost, spent on
    characterization circuits, not on your workload
  - the result describes the PROCESSOR, statistically, at
    measurement time; it has no notion of your run, your cost
    function, your shot budget, or your optimizer
  - if a training run failed yesterday, RB today cannot attribute
    it: the run left no record for this tool to read
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
