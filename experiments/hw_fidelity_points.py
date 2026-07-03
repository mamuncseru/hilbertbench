#!/usr/bin/env python
#
# file: experiments/hw_fidelity_points.py
#
# Clean hardware fidelity points for Figure 4 (predicted vs observed).
#
# The single-point fidelity fill-ins in the variance job were unusable
# where the ideal expectation was near zero (the attenuation ratio
# C_noisy/C_ideal blows up). This script fixes that by SEARCHING, on a
# free local statevector, for one parameter point per depth with
# |ideal| >= 0.3, and only then spending QPU: one tape per depth (same
# per-run semantics as study_d, so noise_profile gives the per-circuit
# prediction), submitted inside a Batch so all jobs share a queue slot.
#
# Cost: ~6 single-point PUBs x ~1k shots -- a few QPU seconds total.
#
#   python hw_fidelity_points.py --dry                  # search only, 0 QPU
#   python hw_fidelity_points.py --hw \
#       --token-file ~/.qiskit/hb_ibm_token_10 --backend ibm_marrakesh
#------------------------------------------------------------------------------
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Statevector
from qiskit.transpiler.preset_passmanagers import (
    generate_preset_pass_manager,
)

from hilbertbench.integrations.qiskit import HilbertEstimatorProxy
from hilbertbench.models import Mode
from hilbertbench.recorder.tape import HilbertTape
from hilbertbench.analysis import noise_profile

from common import TRACES_ROOT, build_ansatz, pair_observable, save_result

warnings.filterwarnings("ignore")

__FILE__ = os.path.basename(__file__)

N_QUBITS = 4
DEPTHS = [1, 2, 3, 5, 7, 10]
MIN_IDEAL = 0.30                 # |ideal| floor so attenuation is defined
PRECISION = 0.0316               # ~1000 shots per point
SEED_BASE = 20260702


def find_point(depth: int) -> tuple:
    """Search seeds for a parameter point with |ideal| >= MIN_IDEAL."""
    qc, n_params = build_ansatz(N_QUBITS, depth, "linear")
    obs = pair_observable(N_QUBITS)
    for k in range(200):
        rng = np.random.default_rng(SEED_BASE + depth * 1000 + k)
        theta = rng.uniform(0.0, 2.0 * np.pi, n_params)
        ideal = float(
            Statevector(qc.assign_parameters(theta))
            .expectation_value(obs).real
        )
        if abs(ideal) >= MIN_IDEAL:
            return theta, ideal, SEED_BASE + depth * 1000 + k
    raise RuntimeError(f"no |ideal|>={MIN_IDEAL} point at depth {depth}")


def main() -> int:
    p = argparse.ArgumentParser(description="hardware fidelity points")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry", action="store_true",
                   help="search the ideal points only (0 QPU)")
    g.add_argument("--hw", action="store_true", help="run on hardware")
    p.add_argument("--token-file", default="~/.qiskit/hb_ibm_token_10")
    p.add_argument("--backend", default=None)
    p.add_argument("--instance", default=None,
                   help="quantum-computing service CRN if multi-service")
    args = p.parse_args()

    # free phase: pick one well-conditioned point per depth
    #
    points = {}
    print("  searching |ideal| >= %.2f points (statevector, free)"
          % MIN_IDEAL)
    for d in DEPTHS:
        theta, ideal, seed = find_point(d)
        points[d] = (theta, ideal, seed)
        print(f"    depth {d:>2}: ideal={ideal:+.3f} (seed {seed})")
    if args.dry:
        return 0

    # connect and pin the device
    #
    from qiskit_ibm_runtime import Batch, EstimatorV2, QiskitRuntimeService
    token = Path(args.token_file).expanduser().read_text().strip()
    kwargs = {"channel": "ibm_cloud", "token": token}
    if args.instance:
        kwargs["instance"] = args.instance
    service = QiskitRuntimeService(**kwargs)
    backend = (service.backend(args.backend) if args.backend
               else service.least_busy(simulator=False, operational=True))
    print(f"  device: {backend.name} "
          f"({backend.status().pending_jobs} pending)")

    pm = generate_preset_pass_manager(
        backend=backend, optimization_level=1, seed_transpiler=SEED_BASE,
    )
    out = TRACES_ROOT / "hw_fidelity"
    out.mkdir(parents=True, exist_ok=True)

    # one tape per depth (per-run noise_profile), one shared Batch
    #
    records, qpu, t0 = [], 0.0, time.time()
    with Batch(backend=backend) as batch:
        est = EstimatorV2(mode=batch)
        est.options.resilience_level = 0
        try:
            est.options.twirling.enable_gates = False
            est.options.twirling.enable_measure = False
        except Exception:
            pass

        for d in DEPTHS:
            theta, ideal, seed = points[d]
            qc, _ = build_ansatz(N_QUBITS, d, "linear")
            obs = pair_observable(N_QUBITS)
            isa = pm.run(qc)
            isa_obs = obs.apply_layout(isa.layout)
            with HilbertTape(
                out, mode=Mode.active,
                tags={"experiment": "hw_fidelity", "depth": str(d),
                      "backend": backend.name},
            ) as tape:
                proxy = HilbertEstimatorProxy(tape, real_estimator=est)
                job = proxy.run(
                    [(isa, isa_obs, theta.reshape(1, -1), PRECISION)]
                )
                res = job.result()
            noisy = float(np.asarray(res[0].data.evs).ravel()[0])
            try:
                usage = job.metrics().get("usage", {})
                qpu += float(usage.get("quantum_seconds", 0) or 0)
            except Exception:
                pass
            predicted = noise_profile(tape.dir_path)[
                "estimated_circuit_fidelity"]
            observed = noisy / ideal
            records.append({
                "depth": d, "seed": seed, "ideal": ideal, "noisy": noisy,
                "predicted_fidelity": predicted,
                "observed_attenuation": observed,
                "run_dir": str(tape.dir_path),
            })
            pf = "n/a" if predicted is None else f"{predicted:.3f}"
            print(f"    depth {d:>2}: predicted={pf} "
                  f"observed={observed:+.3f}")

    save_result("hw_fidelity", {
        "protocol": "hw_fidelity_points_v1",
        "backend": backend.name,
        "min_ideal": MIN_IDEAL,
        "precision": PRECISION,
        "records": records,
    })
    print(f"\n  {len(records)} hardware fidelity points | "
          f"QPU ~{qpu:.0f}s | wall {time.time() - t0:.0f}s")
    print("  re-run paper/draft_arxiv/figures/make_figures.py to "
          "refresh Fig. 4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
