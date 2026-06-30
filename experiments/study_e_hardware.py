#!/usr/bin/env python
#
# file: experiments/study_e_hardware.py
#
# Study E hardware subset generator.
#
# Generates the real-hardware stratum of the blinded corpus: balanced
# rounds (one run per class per round) of the four planted classes on a
# SINGLE pinned IBM device, with a QPU budget guard (no pay-as-you-go).
# Each run is a broadcast cost-landscape recorded through the proxy in
# ONE job (sequential VQE trajectories are wall-clock-prohibitive on a
# shared queue). Stratum lives in the MANIFEST, never in the trace tags,
# so Researcher B cannot read it from a blinded run.
#
# !!! The per-class RECIPES below are PLACEHOLDERS. Freeze them on the
#     hardware recipe pilot before the real run (each planted mechanism
#     must be the dominant signature; barren is an H2 co-occurrence
#     case). See paper/PREREGISTRATION.md and blind-study notes.
#
# Append-safe: re-running adds rounds toward the target and keeps the
# classes balanced; runs land in the same corpus dir as the simulated
# stratum so a single blind_corpus.py blind covers both.
#
#   python study_e_hardware.py --sim --rounds 2          # validate, 0 QPU
#   python study_e_hardware.py --hw  --rounds 9 \
#       --budget-seconds 1000 --token-file ~/.qiskit/hb_ibm_token_10
#------------------------------------------------------------------------------
from __future__ import annotations

import argparse
import json
import os
import random
import secrets
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from hilbertbench.integrations.qiskit import HilbertEstimatorProxy
from hilbertbench.models import Mode
from hilbertbench.recorder.tape import HilbertTape

from common import TRACES_ROOT, build_ansatz, pair_observable

__FILE__ = os.path.basename(__file__)

# the four planted classes
#
LABELS = ["healthy", "barren_plateau", "shot_starved", "noise_dominated"]

# PLACEHOLDER per-class hardware recipes (width, depth, precision,
# landscape size). Freeze on the pilot. Rationale for the starting
# guesses: healthy = shallow + many shots (clean); shot_starved =
# shallow + ~5 shots (precision 0.45); noise_dominated = deep (fidelity
# collapses); barren_plateau = wide+deep (flat AND noisy -> H2).
#
RECIPES = {
    "healthy":         {"w": 4, "d": 2,  "precision": 0.05, "samples": 40},
    "shot_starved":    {"w": 4, "d": 2,  "precision": 0.45, "samples": 40},
    "noise_dominated": {"w": 4, "d": 16, "precision": 0.10, "samples": 40},
    "barren_plateau":  {"w": 8, "d": 8,  "precision": 0.03, "samples": 60},
}

# hardware-subset seeds: disjoint from dev (1000+) and sim test (2000+)
#
SEED_BASE = 3000


def connect(token_file: str, backend_name: str | None):
    """
    function: connect

    arguments:
     token_file:   path to the private IBM token file
     backend_name: pin this backend, or None for the least busy

    return:
     (service, backend) -- the runtime service and chosen device

    description:
     Opens the runtime service with the token from a file outside the
     repo and pins one device. Listing backends costs no QPU.
    """

    # open the service and pick the device
    #
    from qiskit_ibm_runtime import QiskitRuntimeService
    token = Path(token_file).expanduser().read_text().strip()
    service = QiskitRuntimeService(channel="ibm_cloud", token=token)
    if backend_name:
        backend = service.backend(backend_name)
    else:
        backend = service.least_busy(simulator=False, operational=True)
    print(f"  device: {backend.name} ({backend.num_qubits}q, "
          f"{backend.status().pending_jobs} pending)")
    return service, backend


def make_estimator(backend):
    """
    function: make_estimator

    arguments:
     backend: the IBM backend to run on

    return:
     an EstimatorV2 with raw-noise settings (resilience 0, no twirling)

    description:
     Unmitigated device behaviour, so the recorded trace reflects raw
     noise -- matching the rest of the hardware study.
    """

    # build the estimator with mitigation disabled
    #
    from qiskit_ibm_runtime import EstimatorV2
    est = EstimatorV2(mode=backend)
    est.options.resilience_level = 0
    try:
        est.options.twirling.enable_gates = False
        est.options.twirling.enable_measure = False
    except Exception:
        pass
    return est


def job_qpu_seconds(job) -> float:
    """
    function: job_qpu_seconds

    arguments:
     job: the runtime job handle (or None for a simulator run)

    return:
     the QPU seconds the job billed, best-effort (0.0 if unavailable)

    description:
     Reads the usage from the runtime metrics; tolerant of API shape
     differences across runtime versions. Never raises.
    """

    # try the metrics endpoint, then the usage() fallback
    #
    if job is None:
        return 0.0
    try:
        usage = job.metrics().get("usage", {})
        if isinstance(usage, dict):
            return float(
                usage.get("quantum_seconds")
                or usage.get("seconds") or 0.0
            )
        return float(usage)
    except Exception:
        try:
            return float(job.usage())
        except Exception:
            return 0.0


def generate_run(out_dir, label, seed, estimator, pass_manager):
    """
    function: generate_run

    arguments:
     out_dir:      corpus directory the run is written under
     label:        the planted class for this run
     seed:         RNG seed for the parameter draw
     estimator:    a real EstimatorV2 (hardware), or None (statevector)
     pass_manager: an ISA pass manager (hardware), or None (no transpile)

    return:
     (run_dir, job) -- the sealed run directory and the runtime job
     handle (None on the simulator path)

    description:
     Records one broadcast cost-landscape for the class recipe in a
     single job. Tags carry only a neutral corpus_id -- the stratum is
     NOT written into the trace (it goes in the manifest), so a blinded
     run reveals nothing about being hardware.
    """

    # build the ansatz + observable for this recipe
    #
    rec = RECIPES[label]
    qc, n_params = build_ansatz(rec["w"], rec["d"], "linear")
    obs = pair_observable(rec["w"])

    # ISA-transpile for the device when running on hardware
    #
    if pass_manager is not None:
        isa = pass_manager.run(qc)
        isa_obs = obs.apply_layout(isa.layout)
        pub = (isa, isa_obs,
               _draw(seed, rec["samples"], n_params), rec["precision"])
    else:
        pub = (qc, obs, _draw(seed, rec["samples"], n_params))

    # record the landscape in one job; neutral tags only
    #
    tags = {"corpus_id": secrets.token_hex(4)}
    holder = {}
    with HilbertTape(out_dir, mode=Mode.active, tags=tags) as tape:
        proxy = HilbertEstimatorProxy(tape, real_estimator=estimator)
        job = proxy.run([pub])
        holder["job"] = job
        job.result()
    return tape.dir_path, holder.get("job")


def _draw(seed: int, samples: int, n_params: int) -> np.ndarray:
    """Uniform random parameter batch for one landscape (seeded)."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 2.0 * np.pi, (samples, n_params))


def main() -> int:
    p = argparse.ArgumentParser(description="Study E hardware subset")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--hw", action="store_true",
                      help="generate on real hardware")
    mode.add_argument("--sim", action="store_true",
                      help="validate the logic on a statevector sim (0 QPU)")
    p.add_argument("--rounds", type=int, default=9,
                   help="target runs per class (balanced rounds)")
    p.add_argument("--budget-seconds", type=float, default=None,
                   help="stop after the round that reaches this QPU budget")
    p.add_argument("--token-file", default="~/.qiskit/hb_ibm_token_10")
    p.add_argument("--backend", default=None,
                   help="pin this device (default: least busy)")
    p.add_argument("--out", default=None,
                   help="corpus directory (default depends on mode)")
    args = p.parse_args()

    # the hardware subset lands in the same corpus dir as the simulated
    # stratum so one blind covers both; --sim writes to a throwaway dir
    #
    out = Path(args.out) if args.out else (
        TRACES_ROOT / ("corpus_test" if args.hw else "corpus_hw_simcheck")
    )
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = (json.loads(manifest_path.read_text())
                if manifest_path.exists() else {})
    done = Counter(
        v["label"] for v in manifest.values()
        if v.get("stratum") == "hardware"
    )

    # connect (hardware only)
    #
    estimator = pass_manager = None
    if args.hw:
        from qiskit.transpiler.preset_passmanagers import (
            generate_preset_pass_manager,
        )
        _, backend = connect(args.token_file, args.backend)
        estimator = make_estimator(backend)
        pass_manager = generate_preset_pass_manager(
            backend=backend, optimization_level=1,
            seed_transpiler=SEED_BASE,
        )

    # balanced rounds with a budget guard
    #
    qpu_used = 0.0
    seed = SEED_BASE + sum(done.values())
    t0 = time.time()
    for rnd in range(args.rounds):
        if all(done[c] >= args.rounds for c in LABELS):
            print("target reached for every class")
            break
        if (args.budget_seconds is not None
                and qpu_used >= args.budget_seconds):
            print(f"budget reached ({qpu_used:.0f}s); stopping balanced "
                  f"at {dict(done)}")
            break
        order = random.sample(LABELS, len(LABELS))   # shuffle within round
        print(f"\n-- round {rnd + 1} (order {order}) --")
        for label in order:
            if done[label] >= args.rounds:
                continue
            run_dir, job = generate_run(out, label, seed, estimator,
                                        pass_manager)
            used = job_qpu_seconds(job)
            qpu_used += used
            manifest[run_dir.name] = {
                "label": label, "stratum": "hardware", "seed": seed,
            }
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            done[label] += 1
            seed += 1
            print(f"  {label:<16} {run_dir.name}  "
                  f"+{used:.1f}s QPU (total {qpu_used:.0f}s)")

    # summary
    #
    hw_total = sum(done.values())
    print(f"\nhardware subset: {hw_total} runs  {dict(done)}")
    print(f"QPU used: {qpu_used:.0f}s | wall {time.time() - t0:.0f}s")
    print(f"manifest: {manifest_path}")
    if args.hw:
        print("next: audit, then blind the COMBINED corpus (sim + hw):")
        print(f"  python ../tools/blind_corpus.py audit --corpus {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
