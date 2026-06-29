#!/usr/bin/env python
#
# file: experiments/threshold_sensitivity.py
#
# Registered supporting analysis for the pre-registration. Shows the
# frozen default thresholds (variance 0.005 / SNR 1.5 / fidelity 0.50)
# sit inside a STABLE separation gap on the 12-run DEVELOPMENT corpus --
# i.e. each class verdict is robust to the exact threshold, which is
# what justifies freezing the library defaults. Uses development data
# ONLY (never the test corpus). Zero QPU.
#
# Usage:  python threshold_sensitivity.py
#------------------------------------------------------------------------------
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from hilbertbench.analysis import (
    detect_barren_plateau, noise_profile, shot_noise_ratio,
)

from common import TRACES_ROOT

__FILE__ = os.path.basename(__file__)

DEV = TRACES_ROOT / "corpus_dev"


def variance_of(run: str):
    try:
        return detect_barren_plateau(run).get("variance")
    except Exception:
        return None


def snr_of(run: str):
    try:
        return shot_noise_ratio(run).get("estimated_snr")
    except Exception:
        return None


def fidelity_of(run: str):
    try:
        return noise_profile(run).get("estimated_circuit_fidelity")
    except Exception:
        return None


# (mechanism, target label, metric, default threshold, sweep range)
# convention for all three: metric < threshold => flagged as the failure
#
MECHANISMS = [
    ("barren_plateau", variance_of, 0.005, np.linspace(0.002, 0.02, 19)),
    ("shot_starved",   snr_of,      1.5,   np.linspace(1.0, 3.0, 21)),
    ("noise_dominated", fidelity_of, 0.50, np.linspace(0.30, 0.70, 21)),
]


def load_dev() -> list:
    """Return [(run_dir, label), ...] from the dev manifest."""
    manifest = json.loads((DEV / "manifest.json").read_text())
    return [(str(DEV / rid), meta["label"]) for rid, meta in manifest.items()]


def analyse(name, metric, default, sweep, runs) -> None:
    vals = [(label, metric(run)) for run, label in runs]
    vals = [(l, v) for l, v in vals if v is not None]
    pos = [v for l, v in vals if l == name]
    neg = [v for l, v in vals if l != name]

    print(f"== {name}  (metric < t => flagged; default t = {default}) ==")
    if not pos or not neg:
        print("  insufficient data for this mechanism\n")
        return

    # the failure class should sit LOW, everything else HIGH
    gap_lo, gap_hi = max(pos), min(neg)
    print(f"  {name:<16} n={len(pos)}  metric in "
          f"[{min(pos):.4g}, {max(pos):.4g}]")
    print(f"  {'others':<16} n={len(neg)}  metric in "
          f"[{min(neg):.4g}, {max(neg):.4g}]")
    if gap_lo < gap_hi:
        inside = gap_lo < default < gap_hi
        print(f"  separating gap: ({gap_lo:.4g}, {gap_hi:.4g})  "
              f"width {gap_hi - gap_lo:.4g}")
        print(f"  default {default}: "
              f"{'INSIDE the gap (robust)' if inside else 'OUTSIDE the gap'}")
    else:
        print(f"  NO clean single-metric gap "
              f"(max {name} {gap_lo:.4g} >= min others {gap_hi:.4g}) "
              f"-- expected where classes share a signature; the full "
              f"summary() disambiguates")

    # threshold sweep: fraction of dev runs classified correctly
    rows = []
    for t in sweep:
        correct = sum(1 for l, v in vals if (v < t) == (l == name))
        rows.append((float(t), correct / len(vals)))
    perfect = [t for t, a in rows if a == 1.0]
    if perfect:
        print(f"  100%-accuracy threshold band on dev: "
              f"[{min(perfect):.4g}, {max(perfect):.4g}]")
    print()


def main() -> int:
    runs = load_dev()
    print(f"\nthreshold sensitivity on the development corpus "
          f"({len(runs)} runs)\n")
    for name, metric, default, sweep in MECHANISMS:
        analyse(name, metric, default, sweep, runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
