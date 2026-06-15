#!/usr/bin/env python
#
# file: experiments/hw_frontier.py
#
# revision history:
#  20260611 (am): initial version
#
# Hardware arm of the frontier measurement (paper Claim 2, Figure 3)
# plus fidelity fill-in points (Figure 4). Each subcommand is ONE
# runtime job; --pilot is sized for the free tier and --full is the
# week-2 paid batch — the same code path either way, so the pilot
# de-risks the paid campaign.
#
#  variance        deep landscape variance: the NIBP-onset hunt. The
#                  free-tier data showed no suppression at depth <= 6,
#                  so the pilot probes depths 8-16. Adds fidelity
#                  fill-in PUBs in the same job. A matched noiseless
#                  simulator baseline (same parameter draws) is
#                  computed locally for a direct hardware-vs-ideal
#                  delta per depth.
#
#  expressibility  KL expressibility on hardware via compute-uncompute
#                  pairs: run U(a) U(b)^-1 and estimate the state
#                  fidelity as P(all zeros). The SAME parameter pairs
#                  are evaluated on a local statevector, giving a
#                  matched-sim KL and the first measured KL noise
#                  shift.
#
# Raw-noise settings (no mitigation, no twirling) on purpose.
#
# Usage:
#   python experiments/hw_frontier.py variance --pilot
#   python experiments/hw_frontier.py expressibility --pilot
#   python experiments/hw_frontier.py variance --full --yes   # week 2
#------------------------------------------------------------------------------

# future imports must come first
#
from __future__ import annotations

# import system modules
#
import argparse
import os
import sys
import time
import warnings
from pathlib import Path

# import third-party modules
#
import numpy as np
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.transpiler.preset_passmanagers import (
    generate_preset_pass_manager,
)
from qiskit_ibm_runtime import (
    Batch, EstimatorV2, QiskitRuntimeService, SamplerV2,
)

# import hilbertbench modules
#
from hilbertbench.integrations.qiskit import (
    HilbertEstimatorProxy, HilbertSamplerProxy,
)
from hilbertbench.models import Mode
from hilbertbench.recorder.tape import HilbertTape
from hilbertbench.analysis._util import bootstrap_ci
from hilbertbench.analysis.expressibility import _haar_probabilities

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

# geometry and seeds
#
N_QUBITS = 4
SEED = 20260615

# variance arm: pilot (free tier) vs full (week-2 paid batch)
#
VAR_CFG = {
    "pilot": {"depths": [8, 10, 12, 16], "points": 50,
              "precision": 0.0442,          # ~512 shots
              "fid_depths": [2, 5, 10, 14],
              "fid_precision": 0.0316},     # ~1000 shots
    "full":  {"depths": [1, 2, 3, 4, 6, 8, 10, 12], "points": 100,
              "precision": 0.0312,          # ~1024 shots
              "fid_depths": [1, 3, 5, 7, 10, 14],
              "fid_precision": 0.0316},
}

# expressibility arm
#
EXPR_CFG = {
    "pilot": {"depths": [1, 4], "pairs": 120, "shots": 256},
    "full":  {"depths": [1, 2, 4, 6], "pairs": 500, "shots": 256},
}

# KL histogram bins; pilots use coarser bins to keep the estimate
# stable at low pair counts (matched-sim uses the same bins, so the
# hardware-vs-sim delta stays apples-to-apples)
#
KL_BINS = {"pilot": 20, "full": 75}

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
     (service, backend) — runtime service and least-busy real device
    """

    # open the service with the private token and pick a device
    #
    token = TOKEN_FILE.read_text().strip()
    service = QiskitRuntimeService(channel="ibm_cloud", token=token)
    backend = service.least_busy(simulator=False, operational=True)
    print(f"  backend: {backend.name} "
          f"({backend.status().pending_jobs} pending)")

    # exit gracefully
    #
    return service, backend
#
# end of function


def raw_options(primitive) -> None:
    """
    function: raw_options

    arguments:
     primitive: an EstimatorV2 or SamplerV2

    return:
     none

    description:
     Disables mitigation and twirling so the device's raw behavior is
     what the trace records.
    """

    # best-effort: option names vary slightly across runtime versions
    #
    try:
        primitive.options.resilience_level = 0
    except Exception:
        pass
    try:
        primitive.options.twirling.enable_gates = False
        primitive.options.twirling.enable_measure = False
    except Exception:
        pass
#
# end of function


def kl_vs_haar(fidelities: np.ndarray, num_qubits: int,
               num_bins: int, seed: int) -> dict:
    """
    function: kl_vs_haar

    arguments:
     fidelities: array of pairwise state fidelities
     num_qubits: circuit width (sets the Haar reference)
     num_bins:   histogram bins
     seed:       bootstrap seed

    return:
     dict with kl, kl_ci, num_pairs

    description:
     KL(P_empirical || P_Haar) with a bootstrap CI — the same binned
     estimator as the library's kl_expressibility, applied to an
     externally measured fidelity sample (hardware counts or matched
     statevector overlaps).
    """

    # Haar reference over shared bin edges
    #
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    p_haar = _haar_probabilities(bin_edges, 2 ** num_qubits)
    eps = 1e-10
    p_h = np.where(p_haar == 0, eps, p_haar)

    def kl_of(fids: np.ndarray) -> float:
        p, _ = np.histogram(np.clip(fids, 0.0, 1.0), bins=bin_edges)
        p = p / np.sum(p)
        p_a = np.where(p == 0, eps, p)
        return float(np.sum(p_a * np.log(p_a / p_h)))

    # point estimate plus bootstrap interval
    #
    kl = kl_of(np.asarray(fidelities, dtype=float))
    lo, hi = bootstrap_ci(np.asarray(fidelities, dtype=float), kl_of,
                          n_boot=500, ci=0.95, seed=seed)

    # exit gracefully
    #
    return {"kl": kl, "kl_ci": [lo, hi],
            "num_pairs": int(len(fidelities))}
#
# end of function


def cmd_variance(cfg: dict, tier: str, backend, est_mode) -> int:
    """
    function: cmd_variance

    arguments:
     cfg:      the tier configuration dict
     tier:     'pilot' | 'full'
     backend:  the target device
     est_mode: execution mode for EstimatorV2 (backend or Batch)

    return:
     process exit code

    description:
     One job: deep landscape PUBs plus single-point fidelity PUBs.
     Hardware variance per depth is compared against a matched
     noiseless statevector baseline built from the same draws.
    """

    # build all PUBs and the matched-sim baselines
    #
    rng = np.random.default_rng(SEED)
    pm = generate_preset_pass_manager(
        backend=backend, optimization_level=1, seed_transpiler=SEED,
    )
    pubs, sim_var, fid_ideals = [], {}, []
    for d in cfg["depths"]:
        qc, n_params = build_ansatz(N_QUBITS, d, "linear")
        obs = pair_observable(N_QUBITS)
        params = rng.uniform(0, 2 * np.pi, (cfg["points"], n_params))
        isa = pm.run(qc)
        pubs.append((isa, obs.apply_layout(isa.layout), params,
                     cfg["precision"]))

        # matched noiseless baseline from the identical draws
        #
        ideal = [
            float(Statevector(qc.assign_parameters(p))
                  .expectation_value(obs).real)
            for p in params
        ]
        sim_var[d] = float(np.var(ideal))

    for d in cfg["fid_depths"]:
        qc, n_params = build_ansatz(N_QUBITS, d, "linear")
        obs = pair_observable(N_QUBITS)
        theta = rng.uniform(0, 2 * np.pi, (1, n_params))
        isa = pm.run(qc)
        pubs.append((isa, obs.apply_layout(isa.layout), theta,
                     cfg["fid_precision"]))
        fid_ideals.append(float(
            Statevector(qc.assign_parameters(theta[0]))
            .expectation_value(obs).real
        ))

    # submit through the recording proxy
    #
    out_root = TRACES_ROOT / f"hw_var_{tier}"
    out_root.mkdir(parents=True, exist_ok=True)
    est = EstimatorV2(mode=est_mode)
    raw_options(est)
    print(f"  submitting 1 job, {len(pubs)} PUBs ...", flush=True)
    t0 = time.time()
    with HilbertTape(
        out_root, mode=Mode.active,
        tags={"experiment": f"hw_var_{tier}", "backend": backend.name},
    ) as tape:
        proxy = HilbertEstimatorProxy(tape, real_estimator=est)
        job = proxy.run(pubs)
        res = job.result()
    print(f"  done in {time.time() - t0:.0f}s wall")
    try:
        print(f"  QPU usage: {job.metrics().get('usage', {})}")
    except Exception:
        pass

    # per-depth variance vs the matched baseline
    #
    records = []
    for i, d in enumerate(cfg["depths"]):
        evs = np.asarray(res[i].data.evs, dtype=float).ravel()
        lo, hi = bootstrap_ci(evs, np.var, n_boot=1000, ci=0.95,
                              seed=SEED + d)
        shot_floor = cfg["precision"] ** 2
        records.append({
            "depth": d, "hw_variance": float(np.var(evs)),
            "hw_variance_ci": [lo, hi], "sim_variance": sim_var[d],
            "shot_floor": shot_floor, "n_points": int(evs.size),
        })
        print(f"  depth={d:<3} hw={np.var(evs):.4f} "
              f"[{lo:.4f},{hi:.4f}]  sim={sim_var[d]:.4f}")

    # fidelity fill-in points
    #
    fid_records = []
    base = len(cfg["depths"])
    for k, d in enumerate(cfg["fid_depths"]):
        noisy = float(np.asarray(res[base + k].data.evs).ravel()[0])
        ideal = fid_ideals[k]
        observed = noisy / ideal if abs(ideal) > 0.05 else None
        fid_records.append({"depth": d, "ideal": ideal, "noisy": noisy,
                            "observed_attenuation": observed})
        obs_str = f"{observed:.3f}" if observed else "n/a"
        print(f"  fidelity depth={d:<3} observed={obs_str}")

    # persist for the figures
    #
    save_result(f"hw_var_{tier}", {
        "protocol": f"hw_variance_{tier}_v1", "backend": backend.name,
        "seed": SEED, "records": records,
        "fidelity_points": fid_records,
        "run_dir": str(tape.dir_path),
    })

    # exit gracefully
    #
    return 0
#
# end of function


def cmd_expressibility(cfg: dict, tier: str, backend, mode) -> int:
    """
    function: cmd_expressibility

    arguments:
     cfg:     the tier configuration dict
     tier:    'pilot' | 'full'
     backend: the target device
     mode:    execution mode for SamplerV2 (backend or Batch)

    return:
     process exit code

    description:
     Compute-uncompute KL expressibility on hardware: each PUB row
     runs U(a) U(b)^-1 and the all-zeros probability estimates the
     pair fidelity |<psi(b)|psi(a)>|^2. The same draws are evaluated
     on a local statevector for the matched-sim KL, so the output is
     the measured KL noise shift per depth.
    """

    # build one compute-uncompute PUB per depth
    #
    rng = np.random.default_rng(SEED + 7)
    pm = generate_preset_pass_manager(
        backend=backend, optimization_level=1, seed_transpiler=SEED,
    )
    bins = KL_BINS[tier]
    pubs, sim_fids = [], []
    for d in cfg["depths"]:
        qc, n_params = build_ansatz(N_QUBITS, d, "linear")
        theta_b = ParameterVector("u", n_params)
        qc_b = qc.assign_parameters(dict(zip(qc.parameters, theta_b)))
        probe = qc.compose(qc_b.inverse())
        probe.measure_all()
        isa = pm.run(probe)

        # parameter rows: [t-values | u-values] (sorted by vector name)
        #
        a = rng.uniform(0, 2 * np.pi, (cfg["pairs"], n_params))
        b = rng.uniform(0, 2 * np.pi, (cfg["pairs"], n_params))
        pubs.append((isa, np.hstack([a, b]), cfg["shots"]))

        # matched statevector fidelities from the identical draws
        #
        fids = [
            float(np.abs(
                Statevector(qc.assign_parameters(bb)).inner(
                    Statevector(qc.assign_parameters(aa))
                )
            ) ** 2)
            for aa, bb in zip(a, b)
        ]
        sim_fids.append(np.array(fids))

    # submit through the recording sampler proxy
    #
    out_root = TRACES_ROOT / f"hw_expr_{tier}"
    out_root.mkdir(parents=True, exist_ok=True)
    sampler = SamplerV2(mode=mode)
    raw_options(sampler)
    print(f"  submitting 1 job, {len(pubs)} PUBs ...", flush=True)
    t0 = time.time()
    with HilbertTape(
        out_root, mode=Mode.active,
        tags={"experiment": f"hw_expr_{tier}", "backend": backend.name},
    ) as tape:
        proxy = HilbertSamplerProxy(tape, real_sampler=sampler)
        job = proxy.run(pubs)
        res = job.result()
    print(f"  done in {time.time() - t0:.0f}s wall")
    try:
        print(f"  QPU usage: {job.metrics().get('usage', {})}")
    except Exception:
        pass

    # per-depth: hardware KL vs matched-sim KL
    #
    zeros = "0" * N_QUBITS
    records = []
    for i, d in enumerate(cfg["depths"]):
        bits = res[i].data.meas
        hw_fids = np.array([
            bits.get_counts(k).get(zeros, 0) / cfg["shots"]
            for k in range(cfg["pairs"])
        ])
        hw = kl_vs_haar(hw_fids, N_QUBITS, bins, SEED + d)
        sim = kl_vs_haar(sim_fids[i], N_QUBITS, bins, SEED + d)
        records.append({
            "depth": d, "kl_hw": hw["kl"], "kl_hw_ci": hw["kl_ci"],
            "kl_sim": sim["kl"], "kl_sim_ci": sim["kl_ci"],
            "kl_shift": hw["kl"] - sim["kl"],
            "mean_fid_hw": float(hw_fids.mean()),
            "mean_fid_sim": float(sim_fids[i].mean()),
            "pairs": cfg["pairs"], "shots": cfg["shots"],
            "num_bins": bins,
        })
        print(f"  depth={d}: KL_hw={hw['kl']:.3f} "
              f"[{hw['kl_ci'][0]:.3f},{hw['kl_ci'][1]:.3f}]  "
              f"KL_sim={sim['kl']:.3f}  "
              f"shift={hw['kl'] - sim['kl']:+.3f}")

    # persist for the figures
    #
    save_result(f"hw_expr_{tier}", {
        "protocol": f"hw_expressibility_{tier}_v1",
        "backend": backend.name, "seed": SEED + 7,
        "records": records, "run_dir": str(tape.dir_path),
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
     Parses the arm and tier, estimates the shot budget, and runs.
     --full requires --yes as a guard against accidental submission
     of the paid batch.
    """

    # parse the CLI
    #
    parser = argparse.ArgumentParser(description="hardware frontier arm")
    parser.add_argument("arm", choices=["variance", "expressibility"])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot", action="store_true")
    group.add_argument("--full", action="store_true")
    parser.add_argument("--yes", action="store_true",
                        help="required to submit the --full batch")
    args = parser.parse_args()
    tier = "pilot" if args.pilot else "full"

    # estimate the shot budget up front
    #
    if args.arm == "variance":
        cfg = VAR_CFG[tier]
        shots = (len(cfg["depths"]) * cfg["points"]
                 * round(1 / cfg["precision"] ** 2)
                 + len(cfg["fid_depths"])
                 * round(1 / cfg["fid_precision"] ** 2))
    else:
        cfg = EXPR_CFG[tier]
        shots = len(cfg["depths"]) * cfg["pairs"] * cfg["shots"]
    print(f"  {args.arm} [{tier}]: ~{shots:,} shots planned")

    if tier == "full" and not args.yes:
        print("  --full is the PAID week-2 batch; re-run with --yes")
        return 1

    # connect and dispatch (Batch mode for the paid tier)
    #
    service, backend = connect()
    if tier == "full":
        with Batch(backend=backend) as batch:
            if args.arm == "variance":
                return cmd_variance(cfg, tier, backend, batch)
            return cmd_expressibility(cfg, tier, backend, batch)
    if args.arm == "variance":
        return cmd_variance(cfg, tier, backend, backend)
    return cmd_expressibility(cfg, tier, backend, backend)
#
# end of function


# begin gracefully
#
if __name__ == "__main__":
    sys.exit(main())
#
# end of file
