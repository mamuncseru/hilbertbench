#!/usr/bin/env python
#
# file: experiments/study_b_shot_noise.py
#
# revision history:
#  20260610 (am): initial version
#
# Study B — shot-noise budget (supplementary figure).
#
# Question: at what measurement budget does the diagnostic SNR cross
# the usable-signal threshold for a fixed 4-qubit/2-layer ansatz?
#
# Protocol (pre-registered):
#  - shot budgets 32..4096 (precision = 1/sqrt(shots) per PUB)
#  - 50 random parameter points per budget on a shot-based simulator
#  - metric: shot_noise_ratio per sealed trace; the threshold is the
#    budget where status first leaves 'Shot Noise Dominated'
#
# Usage:
#   python experiments/study_b_shot_noise.py [--quick]
#------------------------------------------------------------------------------

# future imports must come first
#
from __future__ import annotations

# import system modules
#
import argparse
import math
import os
import warnings

# import third-party modules
#
from qiskit.primitives import BackendEstimatorV2
from qiskit_aer import AerSimulator

# import hilbertbench modules
#
from hilbertbench.analysis import shot_noise_ratio

# import experiment infrastructure
#
from common import TRACES_ROOT, sample_landscape, save_result

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
N_LAYERS = 2
SHOT_BUDGETS = [32, 64, 128, 256, 512, 1024, 2048, 4096]
N_POINTS = 50
SEED_BASE = 20260612

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
     Records one landscape per shot budget on a shot-based simulator,
     analyzes each sealed trace with shot_noise_ratio, and writes the
     SNR-vs-budget curve to results/study_b.
    """

    # parse the CLI
    #
    parser = argparse.ArgumentParser(description="Study B: shot noise")
    parser.add_argument("--quick", action="store_true",
                        help="reduced sweep for smoke testing")
    args = parser.parse_args()

    budgets = [32, 512] if args.quick else SHOT_BUDGETS
    n_points = 12 if args.quick else N_POINTS

    # sweep the measurement budget
    #
    records = []
    out_root = TRACES_ROOT / "study_b"
    for shots in budgets:
        seed = SEED_BASE + shots
        precision = 1.0 / math.sqrt(shots)
        estimator = BackendEstimatorV2(
            backend=AerSimulator(seed_simulator=seed),
        )
        run_dir = sample_landscape(
            out_root, N_QUBITS, N_LAYERS, "linear", n_points, seed,
            tags={"study": "b", "shots": str(shots)},
            estimator=estimator, precision=precision, batch_size=n_points,
        )

        # the analyzer sees only the sealed trace
        #
        result = shot_noise_ratio(run_dir, seed=seed)
        records.append({
            "shots": shots,
            "precision": precision,
            "seed": seed,
            "empirical_variance": result["empirical_variance"],
            "theoretical_floor": result["theoretical_floor"],
            "estimated_snr": result["estimated_snr"],
            "shots_source": result["shots_source"],
            "status": result["status"],
            "run_dir": str(run_dir),
        })
        print(f"  shots={shots:<5} snr={result['estimated_snr']:.2f}  "
              f"{result['status']}")

    # write the result file for the figure script
    #
    path = save_result("study_b", {
        "protocol": "study_b_shot_noise_v1",
        "quick": args.quick,
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
