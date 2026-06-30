#!/usr/bin/env python
#
# file: experiments/recipe_pilot.py
#
# Hardware recipe pilot for the Study E hardware subset.
#
# Generates ONE pilot run per class with the candidate RECIPES from
# study_e_hardware.py, then runs the analyzers to check that each
# planted mechanism is the DOMINANT signature on the real device:
#
#   healthy          none dominant  (var high, SNR high, fidelity high)
#   shot_starved     SNR < 1.5      (and fidelity NOT < 0.5)
#   noise_dominated  fidelity < 0.5
#   barren_plateau   variance < 0.005   (noise co-occurring is expected
#                    -- this is the registered H2 case)
#
# These are PILOT/calibration runs (a separate directory), NOT the test
# corpus, so running the analyzers on them is allowed. Iterate the
# RECIPES in study_e_hardware.py until every class reads OK, then freeze.
#
#   python recipe_pilot.py --sim                          # logic, 0 QPU
#   python recipe_pilot.py --hw --token-file ~/.qiskit/hb_ibm_token_10
#------------------------------------------------------------------------------
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

from hilbertbench.analysis import (
    detect_barren_plateau, noise_profile, shot_noise_ratio,
)

from common import TRACES_ROOT
from study_e_hardware import (
    LABELS, RECIPES, connect, generate_run, job_qpu_seconds,
    make_estimator,
)

warnings.filterwarnings("ignore")

__FILE__ = os.path.basename(__file__)

# pilot seeds, disjoint from dev (1000+), test (2000+), hw subset (3000+)
#
PILOT_SEED = 9000

# frozen analyzer thresholds (must match the pre-registration)
#
T_VARIANCE, T_SNR, T_FIDELITY = 0.005, 1.5, 0.50


def measure(run_dir) -> dict:
    """Read variance / SNR / fidelity from a pilot run, best-effort."""
    def safe(fn, key):
        try:
            return fn(str(run_dir)).get(key)
        except Exception:
            return None
    return {
        "variance": safe(detect_barren_plateau, "variance"),
        "snr": safe(shot_noise_ratio, "estimated_snr"),
        "fidelity": safe(noise_profile, "estimated_circuit_fidelity"),
    }


def verdict(label: str, m: dict) -> str:
    """
    Does the recipe yield this class's intended dominant signature?
    A None metric (e.g. fidelity on a noiseless sim) returns "n/a" --
    that class can only be judged on hardware, not a failure.
    """
    var, snr, fid = m["variance"], m["snr"], m["fidelity"]
    barren = var is not None and var < T_VARIANCE
    shot = snr is not None and snr < T_SNR
    noise = fid is not None and fid < T_FIDELITY

    if label == "barren_plateau":
        if var is None:
            return "n/a"
        return "OK (barren; +noise = H2)" if barren else "NOT barren"
    if label == "noise_dominated":
        if fid is None:
            return "n/a (needs hardware)"
        return "OK (noise dominant)" if noise else "NOT noise-dominated"
    if label == "shot_starved":
        if snr is None:
            return "n/a (needs hardware)"
        if shot and not noise:
            return "OK (shot dominant)"
        return "shot+noise (ambiguous)" if shot else "NOT shot-starved"
    if label == "healthy":
        if snr is None or fid is None:
            return ("barren fired" if barren
                    else "n/a (var clean; SNR/fid need hardware)")
        return ("OK (none dominant)"
                if not (barren or shot or noise) else "a mechanism fired")
    return "?"


def main() -> int:
    p = argparse.ArgumentParser(description="Study E hardware recipe pilot")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--hw", action="store_true", help="run on real hardware")
    g.add_argument("--sim", action="store_true",
                   help="statevector logic check (0 QPU; fidelity n/a)")
    p.add_argument("--token-file", default="~/.qiskit/hb_ibm_token_10")
    p.add_argument("--backend", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--classes", nargs="+", default=LABELS, choices=LABELS,
                   help="only pilot these classes (e.g. to re-test one)")
    args = p.parse_args()

    out = Path(args.out) if args.out else TRACES_ROOT / "recipe_pilot"
    out.mkdir(parents=True, exist_ok=True)

    estimator = pass_manager = None
    if args.hw:
        from qiskit.transpiler.preset_passmanagers import (
            generate_preset_pass_manager,
        )
        _, backend = connect(args.token_file, args.backend)
        estimator = make_estimator(backend)
        pass_manager = generate_preset_pass_manager(
            backend=backend, optimization_level=1, seed_transpiler=PILOT_SEED,
        )

    # one pilot run per class, then measure + judge
    #
    qpu, t0, rows = 0.0, time.time(), []
    for label in args.classes:
        rec = RECIPES[label]
        run_dir, job = generate_run(out, label,
                                    PILOT_SEED + LABELS.index(label),
                                    estimator, pass_manager)
        qpu += job_qpu_seconds(job)
        m = measure(run_dir)
        rows.append((label, rec, m, verdict(label, m)))
        print(f"  {label:<16} recorded ({run_dir.name})")

    # report table
    #
    def fmt(v):
        return "n/a" if v is None else f"{v:.4g}"
    print("\n  recipe pilot results")
    print("  " + "-" * 78)
    print(f"  {'class':<16}{'recipe':<14}{'variance':<11}{'SNR':<9}"
          f"{'fidelity':<10}verdict")
    for label, rec, m, v in rows:
        recipe = f"{rec['w']}q/{rec['d']} p{rec['precision']}"
        print(f"  {label:<16}{recipe:<14}{fmt(m['variance']):<11}"
              f"{fmt(m['snr']):<9}{fmt(m['fidelity']):<10}{v}")

    print(f"\n  QPU used: {qpu:.0f}s | wall {time.time() - t0:.0f}s")
    bad = [lbl for lbl, _, _, v in rows
           if not v.startswith(("OK", "n/a"))]
    nas = [lbl for lbl, _, _, v in rows if v.startswith("n/a")]
    if bad:
        print(f"  ADJUST recipes for: {bad} (edit RECIPES in "
              f"study_e_hardware.py and re-run)")
    elif nas:
        print(f"  generation OK; run --hw to assess {nas} "
              f"(need real device noise / calibration)")
    else:
        print("  all classes read OK -- recipes can be frozen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
