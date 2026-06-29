#!/usr/bin/env python
#
# file: experiments/hw_separation_pilot.py
#
# Pilot: does barren_plateau separate from noise_dominated on HARDWARE?
#
# The blind corpus needs each planted class to be the DOMINANT mechanism
# on the real device. The risk: a barren_plateau circuit that is also
# noise-dominated (low fidelity) is ambiguous ground truth. This pilot
# tests whether a WIDE+SHALLOW circuit can be barren (flat ideal
# landscape) while keeping fidelity high enough that noise is NOT
# dominant -- the separation the corpus design relies on.
#
#  flatness  : measured on a STATEVECTOR simulator (exact, 0 QPU) --
#              a barren plateau is a property of the ideal circuit.
#  fidelity  : measured on REAL hardware with a tiny job -- the
#              calibration-based estimate the noise_profile analyzer
#              (and Researcher B) would actually see.
#
# Usage:
#   python hw_separation_pilot.py --sim    # flatness sweep only (free)
#   python hw_separation_pilot.py --hw     # + tiny hardware fidelity
#------------------------------------------------------------------------------
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from qiskit.transpiler.preset_passmanagers import (
    generate_preset_pass_manager,
)

from hilbertbench.integrations.qiskit import HilbertEstimatorProxy
from hilbertbench.models import Mode
from hilbertbench.recorder.tape import HilbertTape
from hilbertbench.analysis import (
    circuit_structure, detect_barren_plateau, noise_profile,
)

from common import (
    TRACES_ROOT, build_ansatz, pair_observable, sample_landscape,
)

warnings.filterwarnings("ignore")

__FILE__ = os.path.basename(__file__)

# the user's 10-minute token (outside the repo; never hardcode)
#
TOKEN_FILE = Path.home() / ".qiskit" / "hb_ibm_token_10"

# (width, depth) grid spanning the barren onset, plus the narrow-deep
# noise reference at the end
#
CANDIDATES = [
    (8, 4), (8, 8), (8, 12),
    (10, 4), (10, 8), (10, 12),
    (12, 4), (12, 8), (12, 12), (12, 16),
]
NOISE_REF = (4, 12)

# representative ibm_marrakesh error rates (from the prior hw_smoke
# noise profile) for a FREE, transpile-free fidelity upper bound
#
ERR_1Q, ERR_2Q = 0.0219, 0.0539


def estimate_fidelity(w: int, d: int) -> float:
    """Optimistic (logical, no-SWAP) product-formula fidelity bound."""
    n2q = d * (w - 1)            # linear entangler, pre-transpile
    n1q = d * w * 2             # RY+RZ per qubit per layer
    return (1 - ERR_2Q) ** n2q * (1 - ERR_1Q) ** n1q

SIM_POINTS = 200        # statevector landscape size (exact, free)
HW_POINTS = 2           # points per hardware fidelity PUB (tiny)
HW_PRECISION = 0.1      # ~100 shots/point
SEED = 7
FIDELITY_FLOOR = 0.5    # above this, noise is NOT dominant


def flatness_sim(configs: list) -> dict:
    """Exact ideal-landscape variance per config on a statevector sim."""
    out = TRACES_ROOT / "sep_pilot_sim"
    rows = {}
    for (w, d) in configs:
        run = sample_landscape(
            out, w, d, "linear", SIM_POINTS, SEED,
            tags={"cfg": f"{w}x{d}"},
        )
        bp = detect_barren_plateau(run)
        rows[(w, d)] = {
            "variance": bp["variance"],
            "status": bp["status"],
            "barren": bp["status"].lower().startswith("barren"),
        }
        print(f"  {w:>2}q x {d:<2}layer  var={bp['variance']:.5f}  "
              f"{bp['status']}")
    return rows


def connect():
    """Least-busy real device on the 10-minute token (no QPU to list)."""
    from qiskit_ibm_runtime import QiskitRuntimeService
    token = TOKEN_FILE.read_text().strip()
    service = QiskitRuntimeService(channel="ibm_cloud", token=token)
    backend = service.least_busy(simulator=False, operational=True)
    print(f"  backend: {backend.name} ({backend.num_qubits}q, "
          f"{backend.status().pending_jobs} pending)")
    return service, backend


def hw_fidelity(configs: list, backend) -> dict:
    """Tiny per-config hardware run -> calibration-based fidelity."""
    from qiskit_ibm_runtime import EstimatorV2
    est = EstimatorV2(mode=backend)
    est.options.resilience_level = 0
    try:
        est.options.twirling.enable_gates = False
        est.options.twirling.enable_measure = False
    except Exception:
        pass

    pm = generate_preset_pass_manager(
        backend=backend, optimization_level=1, seed_transpiler=SEED,
    )
    rng = np.random.default_rng(SEED)
    out = TRACES_ROOT / "sep_pilot_hw"
    out.mkdir(parents=True, exist_ok=True)

    rows = {}
    for (w, d) in configs:
        qc, _ = build_ansatz(w, d, "linear")
        obs = pair_observable(w)
        isa = pm.run(qc)
        isa_obs = obs.apply_layout(isa.layout)
        pts = rng.uniform(0.0, 2 * np.pi, (HW_POINTS, d * w * 2))
        with HilbertTape(
            out, mode=Mode.active,
            tags={"cfg": f"{w}x{d}", "backend": backend.name},
        ) as tape:
            proxy = HilbertEstimatorProxy(tape, real_estimator=est)
            proxy.run([(isa, isa_obs, pts, HW_PRECISION)]).result()
        npf = noise_profile(tape.dir_path)
        cs = circuit_structure(tape.dir_path).get("primary") or {}
        rows[(w, d)] = {
            "fidelity": npf["estimated_circuit_fidelity"],
            "isa_depth": cs.get("depth"),
            "two_qubit_gates": cs.get("two_qubit_gates"),
        }
        print(f"  {w:>2}q x {d:<2}layer  isa_depth={cs.get('depth')}  "
              f"fidelity={npf['estimated_circuit_fidelity']}")
    return rows


def report(flat: dict, fid: dict | None) -> None:
    print("\n  separation table")
    print("  " + "-" * 60)
    print(f"  {'config':<12}{'ideal var':<12}{'barren?':<10}"
          f"{'fidelity':<14}{'noise dom?':<10}")
    for cfg in CANDIDATES + [NOISE_REF]:
        f = flat.get(cfg, {})
        v = f.get("variance")
        barren = "yes" if f.get("barren") else "no"
        fd = (fid or {}).get(cfg, {}).get("fidelity")
        if fd is None:
            # free upper-bound estimate (logical gate counts)
            fd = estimate_fidelity(*cfg)
            fds = f"~{fd:.4f}"
        else:
            fds = f"{fd:.4f}"
        ndom = "no" if fd >= FIDELITY_FLOOR else "YES"
        vs = f"{v:.5f}" if v is not None else "-"
        print(f"  {f'{cfg[0]}q x {cfg[1]}':<12}{vs:<12}{barren:<10}"
              f"{fds:<14}{ndom:<10}")

    if fid:
        winners = [
            cfg for cfg in CANDIDATES
            if flat.get(cfg, {}).get("barren")
            and (fid.get(cfg, {}).get("fidelity") or 0) >= FIDELITY_FLOOR
        ]
        print("\n  CLEAN barren+high-fidelity configs:",
              winners or "NONE -- caveat NOT resolved")


def main() -> int:
    p = argparse.ArgumentParser(description="hardware separation pilot")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--sim", action="store_true",
                   help="flatness sweep only (free, no QPU)")
    g.add_argument("--hw", action="store_true",
                   help="flatness + tiny hardware fidelity")
    args = p.parse_args()

    print("== flatness (statevector, exact, 0 QPU) ==")
    flat = flatness_sim(CANDIDATES + [NOISE_REF])

    fid = None
    if args.hw:
        # only confirm fidelity for barren candidates + the noise ref
        targets = [c for c in CANDIDATES if flat[c]["barren"]] + [NOISE_REF]
        if not targets:
            print("\n  no barren candidates from sim; skipping hardware")
        else:
            print(f"\n== hardware fidelity for {targets} ==")
            _, backend = connect()
            t0 = time.time()
            fid = hw_fidelity(targets, backend)
            print(f"  hardware phase wall time: {time.time() - t0:.0f}s")

    report(flat, fid)
    return 0


if __name__ == "__main__":
    sys.exit(main())
