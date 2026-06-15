#!/usr/bin/env python
#
# file: experiments/study_d_fidelity.py
#
# revision history:
#  20260610 (am): initial version
#
# Study D — fidelity-prediction cross-validation (Claim 3, Figure 4).
#
# Question: how well does the product-formula fidelity estimate in
# noise_profile predict the observed signal attenuation?
#
# Protocol (pre-registered), per depth in {1, 3, 5, 7, 10, 14}:
#  - build the 4-qubit linear HEA at that depth, ISA-transpile it for
#    the (fake or real) device, fix one random parameter point
#  - observed attenuation = <O>_noisy / <O>_ideal for O = Z0 Z1
#    (the standard cheap fidelity proxy for Pauli observables)
#  - predicted fidelity = noise_profile(...)['estimated_circuit_
#    fidelity'] computed from the trace's calibration snapshot
#  - report (predicted, observed) pairs; the figure shows R2 and RMSE
#
# The simulator arm uses a calibrated fake device. The hardware arm
# runs the identical script with a qiskit_ibm_runtime EstimatorV2 —
# only the estimator/backend construction changes (week 2 submission).
#
# Usage:
#   python experiments/study_d_fidelity.py [--quick]
#------------------------------------------------------------------------------

# future imports must come first
#
from __future__ import annotations

# import system modules
#
import argparse
import os
import warnings

# import third-party modules
#
import numpy as np
from qiskit.primitives import BackendEstimatorV2
from qiskit.quantum_info import Statevector
from qiskit.transpiler.preset_passmanagers import (
    generate_preset_pass_manager,
)

# import hilbertbench modules
#
from hilbertbench.integrations.qiskit import HilbertEstimatorProxy
from hilbertbench.models import Mode
from hilbertbench.recorder.tape import HilbertTape
from hilbertbench.analysis import noise_profile

# import experiment infrastructure
#
from common import TRACES_ROOT, build_ansatz, pair_observable, save_result

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

# pre-registered configuration
#
N_QUBITS = 4
DEPTHS = [1, 3, 5, 7, 10, 14]
REPEATS = 5
SEED_BASE = 20260613

#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------

def main() -> None:
    """
    function: main

    arguments:
     none

    return:
     none

    description:
     For each depth and repeat: records one noisy execution at a fixed
     random parameter point, computes the ideal expectation by
     statevector, and compares the trace-predicted fidelity with the
     observed attenuation. Writes results/study_d/results.json.
    """

    # parse the CLI
    #
    parser = argparse.ArgumentParser(description="Study D: fidelity")
    parser.add_argument("--quick", action="store_true",
                        help="reduced sweep for smoke testing")
    args = parser.parse_args()

    depths = [1, 7] if args.quick else DEPTHS
    repeats = 2 if args.quick else REPEATS

    # the calibrated fake device for the simulator arm
    #
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2
    backend = FakeManilaV2()
    pm = generate_preset_pass_manager(
        backend=backend, optimization_level=1, seed_transpiler=SEED_BASE,
    )

    # sweep depth x repeat
    #
    records = []
    out_root = TRACES_ROOT / "study_d"
    for depth in depths:
        for rep in range(repeats):
            seed = SEED_BASE + depth * 100 + rep
            rng = np.random.default_rng(seed)
            qc, n_params = build_ansatz(N_QUBITS, depth, "linear")
            theta = rng.uniform(0.0, 2.0 * np.pi, n_params)
            observable = pair_observable(N_QUBITS)

            # ideal expectation from the statevector
            #
            bound = qc.assign_parameters(theta)
            ideal = float(
                Statevector(bound).expectation_value(observable).real
            )

            # noisy execution through the proxy on the ISA circuit
            #
            isa = pm.run(qc)
            isa_obs = observable.apply_layout(isa.layout)
            est = BackendEstimatorV2(backend=backend)
            out_root.mkdir(parents=True, exist_ok=True)
            with HilbertTape(
                out_root, mode=Mode.active,
                tags={"study": "d", "depth": str(depth)},
            ) as tape:
                proxy = HilbertEstimatorProxy(tape, real_estimator=est)
                res = proxy.run(
                    [(isa, isa_obs, [theta], 0.02)]
                ).result()
            noisy = float(np.asarray(res[0].data.evs).ravel()[0])

            # predicted fidelity from the sealed trace alone
            #
            profile = noise_profile(tape.dir_path)
            predicted = profile["estimated_circuit_fidelity"]
            observed = noisy / ideal if abs(ideal) > 0.05 else None

            records.append({
                "depth": depth,
                "repeat": rep,
                "seed": seed,
                "ideal": ideal,
                "noisy": noisy,
                "predicted_fidelity": predicted,
                "observed_attenuation": observed,
                "dominant_error_source":
                    profile["dominant_error_source"],
                "run_dir": str(tape.dir_path),
            })
            obs_str = f"{observed:.3f}" if observed else "  n/a"
            print(f"  depth={depth:<3} rep={rep} "
                  f"predicted={predicted:.3f} observed={obs_str}")

    # write the result file for the figure script
    #
    path = save_result("study_d", {
        "protocol": "study_d_fidelity_v1",
        "quick": args.quick,
        "backend": "FakeManilaV2",
        "records": records,
    })
    print(f"\n  wrote {path}")

#------------------------------------------------------------------------------
#
# main entry point
#
#------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#
# end of file
