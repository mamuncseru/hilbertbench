#!/usr/bin/env python
#
# file: experiments/hw_expr_final.py
#
# revision history:
#  20260615 (am): initial version
#
# Definitive single-device measurement of the expressibility axis of
# the frontier (paper Claim 2, Figure 3). Companion to
# hw_onset_final.py, which measured the trainability (variance) axis on
# the same device. Together they give a complete, single-device
# expressibility-trainability frontier under real hardware noise.
#
# Method: compute-uncompute. For each pair of parameter vectors (a, b),
# run U(a) U(b)^-1 and read the all-zeros probability, which estimates
# the state fidelity |<psi(b)|psi(a)>|^2. The empirical fidelity
# distribution is compared (KL divergence) against the Haar reference,
# both for the hardware run and for a matched noiseless statevector
# baseline built from the identical parameter draws. The hardware-minus-
# simulator KL shift is the measured noise effect on expressibility.
#
# Sized for the free tier (~70-80 s QPU). One job, one pinned device.
#
# Usage:
#   python experiments/hw_expr_final.py
#------------------------------------------------------------------------------

# future imports must come first
#
from __future__ import annotations

# import system modules
#
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
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

# import hilbertbench modules
#
from hilbertbench.integrations.qiskit import HilbertSamplerProxy
from hilbertbench.models import Mode
from hilbertbench.recorder.tape import HilbertTape

# import experiment infrastructure
#
from common import TRACES_ROOT, build_ansatz, save_result
from hw_frontier import kl_vs_haar

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

# measurement design: 5 depths spanning the frontier, one device
#
N_QUBITS = 4
DEPTHS = [1, 2, 4, 6, 8]
N_PAIRS = 180
SHOTS = 256
KL_BINS = 30
SEED = 20260615

#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------

def main() -> int:
    """
    function: main

    arguments:
     none

    return:
     process exit code

    description:
     Submits one compute-uncompute job sweeping five depths on a single
     pinned device, then reports the hardware KL expressibility against
     the matched noiseless baseline per depth.
    """

    # connect and pin the least-busy device for the whole run
    #
    token = TOKEN_FILE.read_text().strip()
    service = QiskitRuntimeService(channel="ibm_cloud", token=token)
    backend = service.least_busy(simulator=False, operational=True)
    print(f"  device: {backend.name} "
          f"({backend.status().pending_jobs} pending)", flush=True)

    # build one compute-uncompute PUB per depth plus matched-sim draws
    #
    rng = np.random.default_rng(SEED)
    pm = generate_preset_pass_manager(
        backend=backend, optimization_level=1, seed_transpiler=SEED,
    )
    pubs, sim_fids = [], []
    for d in DEPTHS:
        qc, n_params = build_ansatz(N_QUBITS, d, "linear")
        theta_b = ParameterVector("u", n_params)
        qc_b = qc.assign_parameters(dict(zip(qc.parameters, theta_b)))
        probe = qc.compose(qc_b.inverse())
        probe.measure_all()
        isa = pm.run(probe)

        a = rng.uniform(0, 2 * np.pi, (N_PAIRS, n_params))
        b = rng.uniform(0, 2 * np.pi, (N_PAIRS, n_params))
        pubs.append((isa, np.hstack([a, b]), SHOTS))

        # matched statevector fidelities from the identical draws
        #
        sim_fids.append(np.array([
            float(np.abs(
                Statevector(qc.assign_parameters(bb)).inner(
                    Statevector(qc.assign_parameters(aa))
                )
            ) ** 2)
            for aa, bb in zip(a, b)
        ]))

    # run raw (no mitigation) through the recording sampler proxy
    #
    sampler = SamplerV2(mode=backend)
    try:
        sampler.options.twirling.enable_gates = False
        sampler.options.twirling.enable_measure = False
    except Exception:
        pass

    out_root = TRACES_ROOT / "hw_expr_final"
    out_root.mkdir(parents=True, exist_ok=True)
    total_shots = len(DEPTHS) * N_PAIRS * SHOTS
    print(f"  submitting 1 job, {len(pubs)} PUBs "
          f"(~{total_shots // 1000}k shots) ...", flush=True)
    t0 = time.time()
    with HilbertTape(
        out_root, mode=Mode.active,
        tags={"experiment": "hw_expr_final", "backend": backend.name},
    ) as tape:
        proxy = HilbertSamplerProxy(tape, real_sampler=sampler)
        job = proxy.run(pubs)
        res = job.result()
    print(f"  done in {time.time() - t0:.0f}s wall", flush=True)
    try:
        print(f"  QPU usage: {job.metrics().get('usage', {})}", flush=True)
    except Exception:
        pass

    # per-depth hardware KL vs matched-sim KL
    #
    zeros = "0" * N_QUBITS
    records = []
    for i, d in enumerate(DEPTHS):
        bits = res[i].data.meas
        hw_fids = np.array([
            bits.get_counts(k).get(zeros, 0) / SHOTS
            for k in range(N_PAIRS)
        ])
        hw = kl_vs_haar(hw_fids, N_QUBITS, KL_BINS, SEED + d)
        sim = kl_vs_haar(sim_fids[i], N_QUBITS, KL_BINS, SEED + d)
        records.append({
            "depth": d,
            "kl_hw": hw["kl"], "kl_hw_ci": hw["kl_ci"],
            "kl_sim": sim["kl"], "kl_sim_ci": sim["kl_ci"],
            "kl_shift": hw["kl"] - sim["kl"],
            "mean_fid_hw": float(hw_fids.mean()),
            "mean_fid_sim": float(sim_fids[i].mean()),
            "pairs": N_PAIRS, "shots": SHOTS, "num_bins": KL_BINS,
        })
        print(f"  depth={d}: KL_hw={hw['kl']:.3f} "
              f"[{hw['kl_ci'][0]:.3f},{hw['kl_ci'][1]:.3f}]  "
              f"KL_sim={sim['kl']:.3f}  "
              f"shift={hw['kl'] - sim['kl']:+.3f}", flush=True)

    # persist for the figure
    #
    path = save_result("hw_expr_final", {
        "protocol": "hw_expr_final_v1",
        "backend": backend.name, "seed": SEED,
        "depths": DEPTHS, "pairs": N_PAIRS, "shots": SHOTS,
        "records": records, "run_dir": str(tape.dir_path),
    })
    print(f"\n  wrote {path}", flush=True)

    # exit gracefully
    #
    return 0
#
# end of function


# begin gracefully
#
if __name__ == "__main__":
    sys.exit(main())
#
# end of file
