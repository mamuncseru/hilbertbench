#!/usr/bin/env python
#
# file: experiments/study_c_expressibility.py
#
# revision history:
#  20260610 (am): initial version
#
# Study C — expressibility vs trainability (paper Claim 2, Figure 3).
#
# Question: where does the ansatz family sit on the expressibility-
# trainability frontier (Holmes et al. 2022), and how does hardware
# noise shift that frontier?
#
# Protocol (pre-registered), per depth 1..8 of the 4-qubit linear HEA:
#  - Active Mode probe: 2000 statevector samples -> kl_expressibility
#  - Passive landscape: 500 random points -> detect_barren_plateau
#  - the (KL, variance) pairs trace the frontier; depth is the colour
#    axis of Figure 3
#
# The hardware arm (depths 1,2,4,6 on an IBM backend, reduced samples)
# reuses sample_landscape with a runtime EstimatorV2 — see --hardware.
# Submit it in week 2; the queue is the bottleneck, not this script.
#
# Usage:
#   python experiments/study_c_expressibility.py [--quick]
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
from qiskit.quantum_info import Statevector

# import hilbertbench modules
#
from hilbertbench.active.probe import probe_expressibility
from hilbertbench.analysis import detect_barren_plateau, kl_expressibility

# import experiment infrastructure
#
from common import TRACES_ROOT, build_ansatz, sample_landscape, save_result

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
DEPTHS = [1, 2, 3, 4, 5, 6, 7, 8]
N_PROBE = 2000
N_LANDSCAPE = 500
SEED_BASE = 20260611

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
     For each depth: runs the Active Mode expressibility probe and the
     passive landscape recording, analyzes both sealed traces, and
     writes the (KL, variance) frontier to results/study_c.
    """

    # parse the CLI
    #
    parser = argparse.ArgumentParser(description="Study C: frontier")
    parser.add_argument("--quick", action="store_true",
                        help="reduced sweep for smoke testing")
    parser.add_argument("--hardware", action="store_true",
                        help="print the hardware-arm instructions")
    args = parser.parse_args()

    if args.hardware:
        print("hardware arm: wrap a qiskit_ibm_runtime EstimatorV2 in")
        print("sample_landscape(estimator=...) for depths 1,2,4,6 with")
        print("n_samples=100 and precision=0.05; submit in batch mode.")
        print("KL on hardware uses the compute-uncompute probe (wk 2).")
        return

    depths = [1, 3] if args.quick else DEPTHS
    n_probe = 200 if args.quick else N_PROBE
    n_landscape = 60 if args.quick else N_LANDSCAPE

    # sweep depths: active probe + passive landscape per depth
    #
    records = []
    out_root = TRACES_ROOT / "study_c"
    for depth in depths:
        seed = SEED_BASE + depth
        qc, n_params = build_ansatz(N_QUBITS, depth, "linear")

        # active probe: uniform parameter draws -> statevectors
        #
        def state_fn(theta: np.ndarray) -> np.ndarray:
            return Statevector(qc.assign_parameters(theta)).data

        probe_dir = probe_expressibility(
            state_fn, n_params, n_probe, out_root,
            seed=seed, tags={"study": "c", "depth": str(depth)},
        )
        expr = kl_expressibility(probe_dir, seed=seed)

        # passive landscape: variance at random points
        #
        land_dir = sample_landscape(
            out_root, N_QUBITS, depth, "linear", n_landscape, seed,
            tags={"study": "c", "depth": str(depth)},
        )
        train = detect_barren_plateau(land_dir, seed=seed)

        records.append({
            "depth": depth,
            "n_qubits": N_QUBITS,
            "seed": seed,
            "kl_divergence": expr["kl_divergence"],
            "expressibility_status": expr["status"],
            "variance": train["variance"],
            "variance_ci": train["variance_ci"],
            "trainability_status": train["status"],
            "probe_dir": str(probe_dir),
            "landscape_dir": str(land_dir),
        })
        print(
            f"  depth={depth} KL={expr['kl_divergence']:.4f} "
            f"var={train['variance']:.3e}  "
            f"({expr['status']} / {train['status']})"
        )

    # write the result file for the figure script
    #
    path = save_result("study_c", {
        "protocol": "study_c_frontier_v1",
        "quick": args.quick,
        "backend": "statevector",
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
