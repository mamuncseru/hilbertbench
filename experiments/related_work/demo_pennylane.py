#!/usr/bin/env python
#
# file: experiments/related_work/demo_pennylane.py
#
# What PennyLane's inspection utilities give you for post-hoc failure
# attribution — and what they cannot, by construction.
#
# PennyLane is a first-rate autodiff framework and this demo uses it
# as intended: qml.specs() reports circuit resources, qml.Tracker
# counts executions during a run. Both are IN-PROCESS: their outputs
# are Python objects that vanish with the interpreter, record no
# device calibration, and cannot be produced for a run that has
# already finished.
#
#   python demo_pennylane.py
#------------------------------------------------------------------------------
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pennylane as qml


def main() -> int:
    dev = qml.device("default.qubit", wires=4, shots=100)

    @qml.qnode(dev)
    def circuit(theta):
        for i in range(4):
            qml.RY(theta[i], wires=i)
            qml.RZ(theta[4 + i], wires=i)
        for i in range(3):
            qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    rng = np.random.default_rng(20260715)
    theta = rng.uniform(0, 2 * np.pi, 8)

    # 1. qml.specs: circuit resources — depth, gate counts
    #
    specs = qml.specs(circuit)(theta)
    res = specs["resources"]
    print("qml.specs (available DURING the run):")
    print(f"  depth={res.depth}  gates={res.num_gates}  "
          f"gate_types={dict(res.gate_types)}")

    # 2. qml.Tracker: execution counting — also during the run
    #
    with qml.Tracker(dev) as tracker:
        for _ in range(5):
            circuit(theta)
    print(f"qml.Tracker: executions={tracker.totals['executions']}, "
          f"shots={tracker.totals.get('shots')}")

    print("""
WHAT THIS SHOWS
  - specs/Tracker answer resource questions, live, in-process; they
    are genuinely useful for what they are for
  - neither output persists: when this interpreter exits, nothing
    remains that a third party (or your future self) could analyse
  - neither records WHAT the estimates were, the device calibration
    in force, or the shot-level evidence a failure diagnosis needs
  - there is no API that takes "a finished run" as input at all —
    the object of study does not exist in this model
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
