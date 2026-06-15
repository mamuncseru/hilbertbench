#!/usr/bin/env python
#
# file: experiments/hw_smoke_test.py
#
# revision history:
#  20260610 (am): initial version
#
# Real-hardware smoke test + first science points (open-access QPU).
#
# Two subcommands, each ONE runtime job in job mode:
#
#  smoke     ~6k shots (~10-15 s QPU). One 30-point landscape PUB plus
#            three single-point fidelity PUBs (depths 1/3/7). Then a
#            full verification report on the sealed trace:
#             - calibration snapshot captured on the runtime path
#               (bug fix #1) with real T1/T2/error data
#             - ISA circuit parsed with non-zero structure (bug fix #2)
#             - precision evidence recorded -> SNR floor available
#             - noise_profile fidelity sane; trace verify() passes
#            The fidelity PUBs double as real-hardware points for the
#            paper's Figure 4 (predicted vs observed attenuation).
#
#  variance  ~102k shots (~40-80 s QPU). Depths {1,2,4,6} x 50 random
#            parameter points x ~512 shots: the hardware trainability
#            points for Figure 3 (variance vs depth under real noise).
#            Run ONLY after smoke passes.
#
# Raw-noise settings on purpose: resilience_level=0, no twirling —
# the paper compares predictions against unmitigated device behavior.
#
# Usage:
#   python experiments/hw_smoke_test.py smoke
#   python experiments/hw_smoke_test.py variance
#------------------------------------------------------------------------------

# future imports must come first
#
from __future__ import annotations

# import system modules
#
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

# import third-party modules
#
import numpy as np
from qiskit.quantum_info import Statevector
from qiskit.transpiler.preset_passmanagers import (
    generate_preset_pass_manager,
)
from qiskit_ibm_runtime import EstimatorV2, QiskitRuntimeService

# import hilbertbench modules
#
from hilbertbench.integrations.qiskit import HilbertEstimatorProxy
from hilbertbench.models import Mode
from hilbertbench.recorder.tape import HilbertTape
from hilbertbench.analysis import (
    circuit_structure, noise_profile, shot_noise_ratio,
)
from hilbertbench.analysis._util import bootstrap_ci

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

# private token file (outside the repo; never hardcode credentials)
#
TOKEN_FILE = Path.home() / ".qiskit" / "hb_ibm_token"

# experiment geometry
#
N_QUBITS = 4
SEED = 20260614

# smoke-job design (~6k shots total)
#
SMOKE_DEPTH = 2
SMOKE_POINTS = 30
SMOKE_PRECISION = 0.1            # ~100 shots per point
FIDELITY_DEPTHS = [1, 3, 7]
FIDELITY_PRECISION = 0.0316      # ~1000 shots per point

# variance-job design (~102k shots total)
#
VAR_DEPTHS = [1, 2, 4, 6]
VAR_POINTS = 50
VAR_PRECISION = 0.0442           # ~512 shots per point

#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------

def connect() -> tuple:
    """
    function: connect

    arguments:
     none

    return:
     (service, backend) — the runtime service and least-busy device

    description:
     Connects with the token from the private token file and selects
     the least-busy operational non-simulator backend. Listing
     backends costs no QPU time.
    """

    # read the token and open the service
    #
    token = TOKEN_FILE.read_text().strip()
    service = QiskitRuntimeService(channel="ibm_cloud", token=token)

    # pick the least busy real device
    #
    backend = service.least_busy(simulator=False, operational=True)
    print(f"  backend: {backend.name} "
          f"({backend.num_qubits} qubits, "
          f"{backend.status().pending_jobs} pending)")

    # exit gracefully
    #
    return service, backend
#
# end of function


def make_estimator(backend) -> EstimatorV2:
    """
    function: make_estimator

    arguments:
     backend: the IBM backend to run on

    return:
     an EstimatorV2 in job mode with raw-noise settings

    description:
     resilience_level=0 and twirling off so the paper compares its
     fidelity predictions against unmitigated device behavior.
    """

    # build the estimator with mitigation disabled
    #
    est = EstimatorV2(mode=backend)
    est.options.resilience_level = 0
    try:
        est.options.twirling.enable_gates = False
        est.options.twirling.enable_measure = False
    except Exception:
        pass

    # exit gracefully
    #
    return est
#
# end of function


def isa_pubs(backend, depths_points: list, seed: int) -> tuple:
    """
    function: isa_pubs

    arguments:
     backend:       the target backend (for the pass manager)
     depths_points: list of (depth, params_array, precision) requests
     seed:          transpiler seed for reproducibility

    return:
     (pubs, ideals) — ISA-transpiled PUBs and, for single-point PUBs,
     the ideal statevector expectation of each (None for batches)

    description:
     Builds the 4-qubit linear HEA at each requested depth, transpiles
     it for the device, maps the ZZ-pair observable through the
     layout, and computes ideal expectations locally where cheap.
    """

    # one pass manager for all circuits
    #
    pm = generate_preset_pass_manager(
        backend=backend, optimization_level=1, seed_transpiler=seed,
    )

    # build each PUB
    #
    pubs, ideals = [], []
    for depth, params, precision in depths_points:
        qc, _ = build_ansatz(N_QUBITS, depth, "linear")
        obs = pair_observable(N_QUBITS)
        isa = pm.run(qc)
        isa_obs = obs.apply_layout(isa.layout)
        pubs.append((isa, isa_obs, params, precision))

        # ideal value only for single-point PUBs (fidelity arm)
        #
        if params.shape[0] == 1:
            bound = qc.assign_parameters(params[0])
            ideals.append(float(
                Statevector(bound).expectation_value(obs).real
            ))
        else:
            ideals.append(None)

    # exit gracefully
    #
    return pubs, ideals
#
# end of function


def report_usage(proxy_job_holder: dict) -> None:
    """
    function: report_usage

    arguments:
     proxy_job_holder: dict that may carry the runtime job handle

    return:
     none

    description:
     Best-effort print of the job's QPU usage so the open-access
     budget can be tracked. Never raises.
    """

    # try the metrics endpoint
    #
    job = proxy_job_holder.get("job")
    if job is None:
        return
    try:
        usage = job.metrics().get("usage", {})
        print(f"  QPU usage: {json.dumps(usage)}")
    except Exception:
        try:
            print(f"  QPU usage (s): {job.usage()}")
        except Exception:
            print("  QPU usage: unavailable from this runtime version")
#
# end of function


def cmd_smoke() -> int:
    """
    function: cmd_smoke

    arguments:
     none

    return:
     process exit code (0 = all checks passed)

    description:
     Submits the combined smoke + fidelity job through the recording
     proxy, waits for the result, then verifies both bug fixes and
     the full evidence chain on the sealed trace.
    """

    # connect and build the job
    #
    service, backend = connect()
    rng = np.random.default_rng(SEED)
    requests = [(
        SMOKE_DEPTH,
        rng.uniform(0.0, 2 * np.pi, (SMOKE_POINTS, SMOKE_DEPTH * N_QUBITS * 2)),
        SMOKE_PRECISION,
    )]
    for d in FIDELITY_DEPTHS:
        requests.append((
            d,
            rng.uniform(0.0, 2 * np.pi, (1, d * N_QUBITS * 2)),
            FIDELITY_PRECISION,
        ))
    pubs, ideals = isa_pubs(backend, requests, SEED)
    est = make_estimator(backend)

    # record the job through the proxy (blocks until the result)
    #
    out_root = TRACES_ROOT / "hw_smoke"
    out_root.mkdir(parents=True, exist_ok=True)
    holder = {}
    print(f"  submitting 1 job, {len(pubs)} PUBs "
          f"(~6k shots) ...", flush=True)
    t0 = time.time()
    with HilbertTape(
        out_root, mode=Mode.active,
        tags={"experiment": "hw_smoke", "backend": backend.name},
    ) as tape:
        proxy = HilbertEstimatorProxy(tape, real_estimator=est)
        job = proxy.run(pubs)
        holder["job"] = job
        res = job.result()
    print(f"  done in {time.time() - t0:.0f}s wall time")
    report_usage(holder)

    # ---- verification report on the sealed trace --------------------
    #
    run_dir = tape.dir_path
    checks = {}

    # bug fix #1: calibration captured on the runtime path
    #
    from hilbertbench import HilbertTrace
    trace = HilbertTrace(run_dir)
    hist = trace.calibration_history()
    npf = noise_profile(run_dir)
    checks["calibration_captured"] = (
        len(hist) >= 1
        and npf["backend_name"] is not None
        and npf["t1_us"]["mean"] is not None
    )

    # bug fix #2: ISA circuit parses with non-zero structure
    #
    cs = circuit_structure(run_dir)
    prim = cs.get("primary") or {}
    checks["isa_circuit_parsed"] = (
        prim.get("num_qubits", 0) >= N_QUBITS
        and prim.get("total_gates", 0) > 0
    )

    # precision evidence -> SNR floor available
    #
    snr = shot_noise_ratio(run_dir)
    checks["shot_evidence_recorded"] = snr["shots_source"] is not None

    # noise profile gives a sane fidelity
    #
    fid = npf["estimated_circuit_fidelity"]
    checks["fidelity_estimate_sane"] = fid is not None and 0.0 < fid <= 1.0

    # cryptographic seal verifies
    #
    checks["trace_verifies"] = bool(trace.verify())

    # fidelity science points: predicted vs observed
    #
    fidelity_points = []
    for i, d in enumerate(FIDELITY_DEPTHS, start=1):
        noisy = float(np.asarray(res[i].data.evs).ravel()[0])
        ideal = ideals[i]
        observed = noisy / ideal if abs(ideal) > 0.05 else None
        fidelity_points.append({
            "depth": d, "ideal": ideal, "noisy": noisy,
            "observed_attenuation": observed,
        })

    # print the report
    #
    print("\n  verification report")
    print("  " + "-" * 50)
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL':<6} {name}")
    print(f"\n  backend: {npf['backend_name']}")
    print(f"  T1 mean: {npf['t1_us']['mean']:.1f} us | "
          f"2q gate err: {npf['gate_error_2q_mean']:.4f}")
    print(f"  est. fidelity (primary circuit): {fid}")
    print(f"  SNR: {snr['estimated_snr']:.2f} "
          f"({snr['shots_source']}) -> {snr['status']}")
    for p in fidelity_points:
        obs_str = (f"{p['observed_attenuation']:.3f}"
                   if p["observed_attenuation"] else "n/a")
        print(f"  fidelity arm depth={p['depth']}: "
              f"ideal={p['ideal']:+.3f} noisy={p['noisy']:+.3f} "
              f"observed={obs_str}")

    # persist everything for the paper
    #
    save_result("hw_smoke", {
        "protocol": "hw_smoke_v1",
        "backend": backend.name,
        "seed": SEED,
        "checks": checks,
        "noise_profile": npf,
        "shot_noise": {k: snr[k] for k in
                       ("estimated_snr", "shots_source", "status",
                        "theoretical_floor")},
        "fidelity_points": fidelity_points,
        "run_dir": str(run_dir),
    })
    all_ok = all(checks.values())
    print(f"\n  {'ALL CHECKS PASSED' if all_ok else 'CHECKS FAILED'}")

    # exit gracefully
    #
    return 0 if all_ok else 1
#
# end of function


def cmd_variance() -> int:
    """
    function: cmd_variance

    arguments:
     none

    return:
     process exit code

    description:
     Submits the depth-sweep variance job (one job, four PUBs) and
     computes per-depth landscape variance with bootstrap CIs — the
     hardware trainability points for Figure 3. Run after smoke
     passes.
    """

    # connect and build the job
    #
    service, backend = connect()
    rng = np.random.default_rng(SEED + 1)
    requests = [
        (d, rng.uniform(0.0, 2 * np.pi, (VAR_POINTS, d * N_QUBITS * 2)),
         VAR_PRECISION)
        for d in VAR_DEPTHS
    ]
    pubs, _ = isa_pubs(backend, requests, SEED)
    est = make_estimator(backend)

    # record through the proxy
    #
    out_root = TRACES_ROOT / "hw_variance"
    out_root.mkdir(parents=True, exist_ok=True)
    holder = {}
    print(f"  submitting 1 job, {len(pubs)} PUBs "
          f"(~102k shots) ...", flush=True)
    t0 = time.time()
    with HilbertTape(
        out_root, mode=Mode.active,
        tags={"experiment": "hw_variance", "backend": backend.name},
    ) as tape:
        proxy = HilbertEstimatorProxy(tape, real_estimator=est)
        job = proxy.run(pubs)
        holder["job"] = job
        res = job.result()
    print(f"  done in {time.time() - t0:.0f}s wall time")
    report_usage(holder)

    # per-depth variance with bootstrap CI (span i <-> depth i)
    #
    records = []
    for i, d in enumerate(VAR_DEPTHS):
        evs = np.asarray(res[i].data.evs, dtype=float).ravel()
        lo, hi = bootstrap_ci(evs, np.var, n_boot=1000, ci=0.95,
                              seed=SEED + d)
        records.append({
            "depth": d, "n_points": int(evs.size),
            "variance": float(np.var(evs)),
            "variance_ci": [lo, hi],
            "shots_per_point": int(round(1 / VAR_PRECISION ** 2)),
        })
        print(f"  depth={d}: var={np.var(evs):.4f} "
              f"CI=[{lo:.4f}, {hi:.4f}]")

    # persist for the figure
    #
    save_result("hw_variance", {
        "protocol": "hw_variance_v1",
        "backend": backend.name,
        "seed": SEED + 1,
        "records": records,
        "run_dir": str(tape.dir_path),
    })

    # exit gracefully
    #
    return 0
#
# end of function


def main() -> int:
    """
    function: main

    arguments:
     none

    return:
     process exit code

    description:
     Dispatches the smoke or variance subcommand.
    """

    # parse and dispatch
    #
    parser = argparse.ArgumentParser(description="hardware smoke test")
    parser.add_argument("command", choices=["smoke", "variance"])
    args = parser.parse_args()
    return cmd_smoke() if args.command == "smoke" else cmd_variance()
#
# end of function


# begin gracefully
#
if __name__ == "__main__":
    sys.exit(main())
#
# end of file
