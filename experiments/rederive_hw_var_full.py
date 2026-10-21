#!/usr/bin/env python
#
# file: experiments/rederive_hw_var_full.py
#
# Reconstructs results/hw_var_full/results.json for the full depth range
# 1-18 from evidence. The depths-1-12 record is re-derived from the
# preserved sealed trace (hw variance and bootstrap CI from the recorded
# outcomes; the matched noiseless-sim variance recomputed from the
# recorded parameters); the depths 14/16/18 record is read from the
# current results file (written by the deep hw_frontier run). This
# repairs the file after the deep run overwrote the 1-12 entries, and
# keeps the figure regenerable strictly from evidence.
#
#   python rederive_hw_var_full.py
#------------------------------------------------------------------------------
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Statevector

from hilbertbench import HilbertTrace
from hilbertbench.analysis._util import bootstrap_ci

from common import RESULTS_ROOT, build_ansatz, pair_observable

warnings.filterwarnings("ignore")

TRACE_1_12 = ("traces/hw_var_full/20260630_143721_587f61e5")
DEPTHS_1_12 = [1, 2, 3, 4, 6, 8, 10, 12]   # PUB/span order of the run
SHOT_FLOOR = 0.0009734399999999999          # precision 0.0312 -> ~1024 shots
SEED = 20260615


def sim_variance(depth: int, params: np.ndarray) -> float:
    """Noiseless landscape variance for the same parameter points."""
    qc, _ = build_ansatz(4, depth, "linear")
    obs = pair_observable(4)
    vals = [
        float(Statevector(qc.assign_parameters(p))
              .expectation_value(obs).real)
        for p in params
    ]
    return float(np.var(vals))


def main() -> int:
    trace = HilbertTrace(TRACE_1_12)
    spans = list(trace.completed())

    records = []
    for i, depth in enumerate(DEPTHS_1_12):
        sp = spans[i]
        evs = np.asarray(sp.outcome, dtype=float)
        params = np.asarray(sp.parameters, dtype=float)
        hw_var = float(np.var(evs))
        lo, hi = bootstrap_ci(evs, np.var, n_boot=1000, ci=0.95,
                              seed=SEED + depth)
        records.append({
            "depth": depth,
            "hw_variance": hw_var,
            "hw_variance_ci": [lo, hi],
            "sim_variance": sim_variance(depth, params),
            "shot_floor": SHOT_FLOOR,
            "n_points": int(evs.size),
        })
        print(f"  depth {depth:>2}: hw {hw_var:.4f} "
              f"[{lo:.4f}, {hi:.4f}]  sim {records[-1]['sim_variance']:.4f}")

    # append the deep depths from the current (post-deep-run) file
    cur = json.loads(
        (RESULTS_ROOT / "hw_var_full" / "results.json").read_text())
    for r in cur["records"]:
        if r["depth"] in (14, 16, 18):
            records.append({
                "depth": r["depth"],
                "hw_variance": r["hw_variance"],
                "hw_variance_ci": r["hw_variance_ci"],
                "sim_variance": r["sim_variance"],
                "shot_floor": r.get("shot_floor", SHOT_FLOOR),
                "n_points": r.get("n_points", 100),
            })
            print(f"  depth {r['depth']:>2}: hw {r['hw_variance']:.4f} "
                  f"(from deep run)  sim {r['sim_variance']:.4f}")

    records.sort(key=lambda r: r["depth"])
    out = {
        "protocol": "hw_variance_full_v1",
        "backend": "ibm_marrakesh",
        "seed": SEED,
        "note": ("depths 1-12 re-derived from the sealed trace "
                 f"{TRACE_1_12.split('/')[-1]}; depths 14-18 from the "
                 "deep hw_frontier run. See rederive_hw_var_full.py."),
        "records": records,
    }
    path = RESULTS_ROOT / "hw_var_full" / "results.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {path} with depths "
          f"{[r['depth'] for r in records]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
