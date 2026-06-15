#!/usr/bin/env python
#
# file: experiments/study_e_corpus.py
#
# revision history:
#  20260610 (am): initial version
#
# Study E — blinded-corpus generation (paper Claim 1, Figure 5).
#
# Generates runs with planted failure modes for the blinded diagnosis
# protocol. Four classes:
#
#  healthy          4q/2-layer VQE, ample measurement budget, converges
#  barren_plateau   10q/30-layer random landscape, variance ~ 0
#  shot_starved     4q/2-layer VQE at precision 0.45 (~5 shots): the
#                   optimizer chases shot noise
#  noise_dominated  4q/16-layer VQE on a fake noisy device (calibrated
#                   noise model, ISA-transpiled), fidelity collapses
#
# Blinding rules enforced here:
#  - tags carry only a neutral corpus_id (audited by blind_corpus.py)
#  - generation order is shuffled so trace timestamps carry no label
#    information
#  - the manifest (path -> label) is written next to the corpus and
#    consumed by tools/blind_corpus.py blind
#
# Usage:
#   python experiments/study_e_corpus.py --dev       # 3/class, tuning
#   python experiments/study_e_corpus.py --test      # 9/class, blinded
#------------------------------------------------------------------------------

# future imports must come first
#
from __future__ import annotations

# import system modules
#
import argparse
import json
import os
import secrets
import warnings
from pathlib import Path

# import third-party modules
#
import numpy as np
from qiskit.primitives import BackendEstimatorV2
from qiskit.transpiler.preset_passmanagers import (
    generate_preset_pass_manager,
)
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

# import hilbertbench modules
#
from hilbertbench.integrations.qiskit import HilbertEstimatorProxy
from hilbertbench.models import Mode
from hilbertbench.recorder.tape import HilbertTape

# import experiment infrastructure
#
from common import (
    TRACES_ROOT, build_ansatz, pair_observable, sample_landscape,
)

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

# optimizer iterations for trajectory-style corpus runs
#
MAX_ITER = 60

#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------

def _neutral_tags() -> dict:
    """
    function: _neutral_tags

    arguments:
     none

    return:
     a tags dict carrying only a random corpus identifier

    description:
     Corpus runs must never carry label information in their tags;
     blind_corpus.py audits this before blinding.
    """

    # exit gracefully
    #
    return {"corpus_id": secrets.token_hex(4)}
#
# end of function


def _train(
    out_root: Path,
    n_qubits: int,
    n_layers: int,
    seed: int,
    estimator,
    precision: float | None,
    transpile_backend=None,
) -> Path:
    """
    function: _train

    arguments:
     out_root:           directory for the run directory
     n_qubits:           circuit width
     n_layers:           ansatz layers
     seed:               seed for the initial parameter draw
     estimator:          the V2 estimator to run on
     precision:          target precision per PUB (None = omit)
     transpile_backend:  when given, the circuit is ISA-transpiled
                         for this backend and the observable mapped

    return:
     the run directory of the sealed trace

    description:
     Runs a COBYLA minimisation of <Z0 Z1> through the proxy — a
     realistic passive-mode training trajectory. All failure planting
     happens through the estimator/precision/depth choices; the
     training loop itself is identical across classes.
    """

    # build and (optionally) transpile the ansatz
    #
    qc, n_params = build_ansatz(n_qubits, n_layers, "linear")
    observable = pair_observable(n_qubits)
    if transpile_backend is not None:
        pm = generate_preset_pass_manager(
            backend=transpile_backend, optimization_level=1, seed_transpiler=seed,
        )
        qc = pm.run(qc)
        observable = observable.apply_layout(qc.layout)

    # run the optimization through the recording proxy
    #
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0.0, 2.0 * np.pi, n_params)
    out_root.mkdir(parents=True, exist_ok=True)
    with HilbertTape(
        out_root, mode=Mode.passive, tags=_neutral_tags(),
    ) as tape:
        proxy = HilbertEstimatorProxy(tape, real_estimator=estimator)

        def cost(theta: np.ndarray) -> float:
            pub = (
                (qc, observable, [theta])
                if precision is None
                else (qc, observable, [theta], precision)
            )
            res = proxy.run([pub]).result()
            return float(np.asarray(res[0].data.evs).ravel()[0])

        minimize(
            cost, x0, method="COBYLA",
            options={"maxiter": MAX_ITER, "rhobeg": 0.4},
        )

    # exit gracefully
    #
    return tape.dir_path
#
# end of function


def generate_one(label: str, out_root: Path, seed: int) -> Path:
    """
    function: generate_one

    arguments:
     label:    failure-mode class to plant
     out_root: directory for the run directory
     seed:     RNG seed for this run

    return:
     the run directory of the sealed trace

    description:
     Dispatches to the generator for one corpus class. The classes
     differ only in width/depth/measurement budget/backend — the
     recording pathway is identical, so traces are stylistically
     indistinguishable except through their evidence.
    """

    # healthy: exact-ish budget, shallow circuit, clean simulator
    #
    if label == "healthy":
        est = BackendEstimatorV2(backend=AerSimulator(seed_simulator=seed))
        return _train(out_root, 4, 2, seed, est, precision=0.02)

    # barren plateau: wide deep landscape, variance collapses
    #
    if label == "barren_plateau":
        return sample_landscape(
            out_root, 10, 30, "linear", 300, seed, tags=_neutral_tags(),
        )

    # shot starvation: same circuit as healthy, ~5 shots per estimate
    #
    if label == "shot_starved":
        est = BackendEstimatorV2(backend=AerSimulator(seed_simulator=seed))
        return _train(out_root, 4, 2, seed, est, precision=0.45)

    # noise domination: deep ISA circuit on a calibrated fake device
    #
    if label == "noise_dominated":
        from qiskit_ibm_runtime.fake_provider import FakeManilaV2
        backend = FakeManilaV2()
        est = BackendEstimatorV2(backend=backend)
        return _train(
            out_root, 4, 16, seed, est, precision=0.1,
            transpile_backend=backend,
        )

    raise ValueError(f"unknown corpus label '{label}'")
#
# end of function


def main() -> None:
    """
    function: main

    arguments:
     none

    return:
     none

    description:
     Generates the dev or test corpus with shuffled generation order
     and writes manifest.json (path -> label) for blind_corpus.py.
     The dev corpus tunes analyzer thresholds; the test corpus is
     frozen and only ever touched through the blinding tool.
    """

    # parse the CLI
    #
    parser = argparse.ArgumentParser(description="Study E: corpus")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dev", action="store_true",
                       help="3 runs/class, seeds 1000+, for tuning")
    group.add_argument("--test", action="store_true",
                       help="9 runs/class, seeds 2000+, for blinding")
    args = parser.parse_args()

    per_class = 3 if args.dev else 9
    seed_base = 1000 if args.dev else 2000
    name = "corpus_dev" if args.dev else "corpus_test"
    out_root = TRACES_ROOT / name

    # build the shuffled job list (timestamps must not leak labels)
    #
    labels = ["healthy", "barren_plateau", "shot_starved",
              "noise_dominated"]
    jobs = [
        (label, seed_base + i * len(labels) + j)
        for j, label in enumerate(labels)
        for i in range(per_class)
    ]
    np.random.default_rng(seed_base).shuffle(jobs)

    # generate every run and collect the manifest
    #
    manifest = {}
    for k, (label, seed) in enumerate(jobs, 1):
        run_dir = generate_one(label, out_root, int(seed))
        rel = str(Path(run_dir).relative_to(out_root))
        manifest[rel] = {"label": label, "seed": int(seed)}
        print(f"  [{k:>2}/{len(jobs)}] {label:<16} -> {rel}")

    # write the manifest next to the corpus
    #
    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\n  wrote {manifest_path}")
    if args.test:
        print("  next: python tools/blind_corpus.py blind "
              f"--manifest {manifest_path} --out blinded/")

#------------------------------------------------------------------------------
#
# main entry point
#
#------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#
# end of file
