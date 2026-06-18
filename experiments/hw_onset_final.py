#!/usr/bin/env python
#
# file: experiments/hw_onset_final.py
#
# revision history:
#  20260615 (am): initial version
#
# Definitive single-device measurement of the noise-induced variance
# suppression onset (paper Claim 2, Figure 3 hardware arm).
#
# The pilot runs measured shallow depths on one device (ibm_marrakesh)
# and deep depths on another (ibm_fez), which left a two-device caveat
# on the onset bracket. This script removes it: one job, one pinned
# device, the full depth span 1..12 that crosses the onset, with a
# matched noiseless statevector baseline from the identical parameter
# draws. Sized for the free tier (~50 s QPU).
#
# Usage:
#   python experiments/hw_onset_final.py
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

# measurement design: full span crossing the onset, one device
#
N_QUBITS = 4
DEPTHS = [1, 2, 3, 4, 6, 8, 10, 12]
N_POINTS = 40
PRECISION = 0.0442           # ~512 shots per estimate
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
     Submits one job sweeping the full depth span on a single pinned
     device, then reports hardware variance vs the matched noiseless
     baseline per depth, bracketing the suppression onset.
    """

    # connect and pin the least-busy device for the whole run
    #
    token = TOKEN_FILE.read_text().strip()
    service = QiskitRuntimeService(channel="ibm_cloud", token=token)
    backend = service.least_busy(simulator=False, operational=True)
    print(f"  device: {backend.name} "
          f"({backend.status().pending_jobs} pending)", flush=True)

    # build every PUB and its matched noiseless baseline
    #
    rng = np.random.default_rng(SEED)
    pm = generate_preset_pass_manager(
        backend=backend, optimization_level=1, seed_transpiler=SEED,
    )
    pubs, sim_var = [], {}
    for d in DEPTHS:
        qc, n_params = build_ansatz(N_QUBITS, d, "linear")
        obs = pair_observable(N_QUBITS)
        params = rng.uniform(0, 2 * np.pi, (N_POINTS, n_params))
        isa = pm.run(qc)
        pubs.append((isa, obs.apply_layout(isa.layout), params, PRECISION))
        sim_var[d] = float(np.var([
            float(Statevector(qc.assign_parameters(p))
                  .expectation_value(obs).real)
            for p in params
        ]))

    # run raw (no mitigation) through the recording proxy
    #
    est = EstimatorV2(mode=backend)
    est.options.resilience_level = 0
    try:
        est.options.twirling.enable_gates = False
        est.options.twirling.enable_measure = False
    except Exception:
        pass

    out_root = TRACES_ROOT / "hw_onset_final"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"  submitting 1 job, {len(pubs)} PUBs "
          f"(~{len(DEPTHS) * N_POINTS * 512 // 1000}k shots) ...",
          flush=True)
    t0 = time.time()
    with HilbertTape(
        out_root, mode=Mode.active,
        tags={"experiment": "hw_onset_final", "backend": backend.name},
    ) as tape:
        proxy = HilbertEstimatorProxy(tape, real_estimator=est)
        job = proxy.run(pubs)
        res = job.result()
    print(f"  done in {time.time() - t0:.0f}s wall", flush=True)
    try:
        print(f"  QPU usage: {job.metrics().get('usage', {})}", flush=True)
    except Exception:
        pass

    # per-depth hardware variance vs the matched simulator baseline
    #
    records = []
    for i, d in enumerate(DEPTHS):
        evs = np.asarray(res[i].data.evs, dtype=float).ravel()
        lo, hi = bootstrap_ci(evs, np.var, n_boot=1000, ci=0.95,
                              seed=SEED + d)
        hw = float(np.var(evs))
        ratio = hw / sim_var[d] if sim_var[d] > 0 else None
        records.append({
            "depth": d, "hw_variance": hw, "hw_variance_ci": [lo, hi],
            "sim_variance": sim_var[d], "ratio": ratio,
            "n_points": int(evs.size),
        })
        rstr = f"{ratio:.2f}" if ratio is not None else "n/a"
        print(f"  depth={d:<3} hw={hw:.4f} [{lo:.4f},{hi:.4f}]  "
              f"sim={sim_var[d]:.4f}  ratio={rstr}", flush=True)

    # persist for the figure
    #
    path = save_result("hw_onset_final", {
        "protocol": "hw_onset_final_v1",
        "backend": backend.name, "seed": SEED,
        "depths": DEPTHS, "n_points": N_POINTS, "precision": PRECISION,
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
