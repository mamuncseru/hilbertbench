#!/usr/bin/env python
#
# file: experiments/study_a_bp_scaling.py
#
# revision history:
#  20260610 (am): initial version
#
# Study A — barren-plateau scaling (paper Claim 3, Figure 2).
#
# Question: does the instrument recover the McClean et al. (2018)
# exponential variance decay from trace evidence alone?
#
# Protocol (pre-registered):
#  - widths 2..12 qubits, three depth rules (1, n/2, n layers),
#    three ansatz families (linear, ring, full entanglement)
#  - 500 uniform random parameter points per landscape
#  - metric: Var[<Z0 Z1>] with 95% bootstrap CI from the built-in
#    detect_barren_plateau analyzer, computed on the sealed trace
#  - fit: log2(variance) vs n per (family, depth rule); McClean
#    predicts a slope near -2 (variance ~ 4^-n) for deep circuits
#
# Usage:
#   python experiments/study_a_bp_scaling.py [--quick]
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

# import hilbertbench modules
#
from hilbertbench.analysis import detect_barren_plateau

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

# pre-registered sweep configuration
#
WIDTHS = [2, 4, 6, 8, 10, 12]
FAMILIES = ["linear", "ring", "full"]
DEPTH_RULES = {
    "shallow": lambda n: 1,
    "half":    lambda n: max(1, n // 2),
    "linear":  lambda n: n,
}
N_SAMPLES = 500
SEED_BASE = 20260610

#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------

def fit_scaling(records: list, family: str, rule: str) -> dict:
    """
    function: fit_scaling

    arguments:
     records: the per-landscape result records
     family:  ansatz family to fit
     rule:    depth rule to fit

    return:
     dict with the fitted log2(variance)-vs-n slope and intercept,
     or None values when fewer than three points are available

    description:
     Least-squares fit of log2(variance) against qubit count. The
     slope is the empirical scaling exponent; McClean predicts ~-2
     for deep random circuits (variance ~ 4^-n).
    """

    # select the matching records with positive variance
    #
    points = [
        (r["n_qubits"], r["variance"])
        for r in records
        if r["family"] == family and r["depth_rule"] == rule
        and r["variance"] and r["variance"] > 0
    ]
    if len(points) < 3:
        return {"slope": None, "intercept": None, "n_points": len(points)}

    # fit in log2 space
    #
    ns = np.array([p[0] for p in points], dtype=float)
    log_var = np.log2([p[1] for p in points])
    slope, intercept = np.polyfit(ns, log_var, 1)

    # exit gracefully
    #
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "n_points": len(points),
    }
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
     Runs the full sweep, analyzes each sealed trace blind, fits the
     scaling exponents, and writes results/study_a/results.json.
    """

    # parse the CLI
    #
    parser = argparse.ArgumentParser(description="Study A: BP scaling")
    parser.add_argument(
        "--quick", action="store_true",
        help="reduced sweep for smoke testing",
    )
    args = parser.parse_args()

    widths = [2, 4] if args.quick else WIDTHS
    families = ["linear"] if args.quick else FAMILIES
    n_samples = 60 if args.quick else N_SAMPLES

    # sweep: record each landscape, then analyze the sealed trace
    #
    records = []
    out_root = TRACES_ROOT / "study_a"
    total = len(widths) * len(families) * len(DEPTH_RULES)
    done = 0
    for fam_idx, family in enumerate(families):
        for rule_idx, (rule, depth_fn) in enumerate(DEPTH_RULES.items()):
            for n in widths:
                depth = depth_fn(n)

                # deterministic seed (python hash() is randomised
                # per process and must not be used here)
                #
                seed = SEED_BASE + fam_idx * 1000 + rule_idx * 100 + n
                run_dir = sample_landscape(
                    out_root, n, depth, family, n_samples, seed,
                    tags={"study": "a", "family": family,
                          "depth_rule": rule},
                )

                # the analyzer sees only the sealed trace
                #
                result = detect_barren_plateau(run_dir, seed=seed)
                records.append({
                    "n_qubits": n,
                    "depth": depth,
                    "depth_rule": rule,
                    "family": family,
                    "n_samples": n_samples,
                    "seed": seed,
                    "variance": result["variance"],
                    "variance_ci": result["variance_ci"],
                    "status": result["status"],
                    "verdict_confidence": result["verdict_confidence"],
                    "run_dir": str(run_dir),
                })
                done += 1
                print(
                    f"  [{done:>2}/{total}] {family:<6} {rule:<7} "
                    f"n={n:<2} depth={depth:<2} "
                    f"var={result['variance']:.3e} {result['status']}"
                )

    # fit the scaling exponent per (family, depth rule)
    #
    fits = {
        f"{family}/{rule}": fit_scaling(records, family, rule)
        for family in families for rule in DEPTH_RULES
    }
    for name, fit in fits.items():
        if fit["slope"] is not None:
            print(f"  fit {name:<16} slope={fit['slope']:+.3f} "
                  f"(McClean deep-circuit prediction ~ -2)")

    # write the result file for the figure script
    #
    path = save_result("study_a", {
        "protocol": "study_a_bp_scaling_v1",
        "quick": args.quick,
        "records": records,
        "fits": fits,
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
