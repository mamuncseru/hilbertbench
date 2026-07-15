#!/usr/bin/env python
#
# file: experiments/related_work/demo_mitiq.py
#
# What Mitiq gives you for post-hoc failure attribution — and what it
# is actually for, done properly.
#
# Mitiq performs error MITIGATION: it improves an expectation value by
# executing MODIFIED versions of the circuit (here, zero-noise
# extrapolation over noise-scaled copies) and extrapolating. That is:
# (1) it must be wired in BEFORE the run, (2) it multiplies the number
# of circuit executions, and (3) it changes the computation whose
# behaviour you might have wanted to diagnose. There is no API that
# accepts a finished run and returns a cause.
#
#   .venv/bin/python demo_mitiq.py       (isolated env)
#------------------------------------------------------------------------------
# NOTE: no `from __future__ import annotations` here -- mitiq detects
# the executor's return type from its live annotation, and stringized
# annotations break that detection.

import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from mitiq import zne

N_EXECUTIONS = {"count": 0}


def make_circuit() -> QuantumCircuit:
    rng = np.random.default_rng(20260715)
    qc = QuantumCircuit(2)
    for _ in range(6):
        qc.ry(rng.uniform(0, 2 * np.pi), 0)
        qc.ry(rng.uniform(0, 2 * np.pi), 1)
        qc.cx(0, 1)
    return qc


def executor(circuit: QuantumCircuit) -> float:
    """Noisy <Z0 Z1> — every call is one more quantum execution."""
    N_EXECUTIONS["count"] += 1
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(
        depolarizing_error(0.02, 2), ["cx"])
    backend = AerSimulator(noise_model=noise, seed_simulator=7)
    qc = circuit.copy()
    qc.measure_all()
    # optimization_level=0: Mitiq's noise scaling works by inserting
    # G G^dagger G folds, which an optimizing transpiler would cancel
    counts = backend.run(
        transpile(qc, backend, optimization_level=0),
        shots=4096).result().get_counts()
    total = sum(counts.values())
    return sum(
        (+1 if k.replace(" ", "")[-2:].count("1") % 2 == 0 else -1) * v
        for k, v in counts.items()) / total


def main() -> int:
    qc = make_circuit()

    unmitigated = executor(qc)
    n_before = N_EXECUTIONS["count"]
    mitigated = zne.execute_with_zne(qc, executor)
    n_zne = N_EXECUTIONS["count"] - n_before

    print(f"unmitigated <ZZ> : {unmitigated:+.4f}  (1 execution)")
    print(f"ZNE-mitigated    : {mitigated:+.4f}  "
          f"({n_zne} executions of noise-scaled circuit variants)")

    print("""
WHAT THIS SHOWS
  - Mitiq did its job: the mitigated value moved (it suppresses noise
    in a value you are about to compute)
  - to do it, Mitiq executed the circuit {n} more times, in modified
    (noise-scaled) forms — the computation under study was changed
    and the quantum cost multiplied
  - nothing here diagnoses a FINISHED run: Mitiq's input is a circuit
    and an executor, not a run record; if your training failed last
    week, Mitiq has no object to operate on
  - mitigation and diagnosis are complements, not competitors: one
    changes the run in advance, the other explains it afterwards
""".replace("{n}", str(n_zne)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
