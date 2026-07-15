#!/usr/bin/env python
#
# file: experiments/study_f_hardmode.py
#
# revision history:
#  20260715 (am): initial version
#
# Study F — detection-boundary mapping (hard-mode algorithmic arm).
#
# Study E validated the human+instrument workflow on a corpus designed
# to be diagnosable. Study F measures the instrument's discriminative
# power alone: each planted mechanism is swept from physically absent
# to fully expressed, and a frozen mechanical decision rule (committed
# in study_f_protocol.json BEFORE execution) classifies every run with
# no human in the loop. Sub-100% detection in the transition region is
# the desired outcome — it locates the failure boundary.
#
# Anti-circularity: the strength knob is always physical (width, depth,
# measurement budget, device noise model), never the diagnostic
# statistic. Generators are imported from study_e_corpus.py so the
# corpus is produced by byte-identical code paths.
#
# Commitment: `score` refuses to run unless study_f_protocol.json still
# hashes to the committed study_f_commitment.sha256 — the rule and grid
# cannot be silently tuned after the corpus exists.
#
#   python study_f_hardmode.py generate           # build the corpus
#   python study_f_hardmode.py score              # rule + metrics
#------------------------------------------------------------------------------

# future imports must come first
#
from __future__ import annotations

# import system modules
#
import argparse
import hashlib
import json
import math
import os
import sys
import warnings
from pathlib import Path

# import third-party modules
#
import numpy as np
from qiskit.primitives import BackendEstimatorV2
from qiskit_aer import AerSimulator

# import hilbertbench modules
#
from hilbertbench import HilbertTrace
from hilbertbench.analysis import summary

# import experiment infrastructure; the generators are the Study E
# generators, imported so the code paths are byte-identical
#
from common import TRACES_ROOT, RESULTS_ROOT, sample_landscape
from study_e_corpus import _neutral_tags, _train

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

# protocol and commitment files (the protocol is committed to git
# before any run is generated; score verifies the hash)
#
HERE = Path(__file__).parent
PROTOCOL_PATH = HERE / "study_f_protocol.json"
COMMITMENT_PATH = HERE / "study_f_commitment.sha256"

# corpus and manifest locations
#
CORPUS_ROOT = TRACES_ROOT / "corpus_hardmode"
MANIFEST_PATH = CORPUS_ROOT / "manifest.json"

# the Study E blinded corpus replayed by the frozen rule
#
REPLAY_ROOT = TRACES_ROOT / "corpus_test"

# canonical mechanism order for reports
#
MECHANISMS = ["barren_plateau", "shot_starved", "noise_dominated"]

#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------

def _protocol() -> dict:
    """
    function: _protocol

    arguments:
     none

    return:
     the parsed protocol dict

    description:
     Loads study_f_protocol.json; every grid and seed below is read
     from it so the committed document is the single source of truth.
    """

    # exit gracefully
    #
    return json.loads(PROTOCOL_PATH.read_text())
#
# end of function


def _verify_commitment() -> str:
    """
    function: _verify_commitment

    arguments:
     none

    return:
     the verified hex digest

    description:
     Hashes the protocol file and compares against the committed
     digest; refuses to proceed on mismatch so the rule and grid
     cannot be modified after the corpus exists.
    """

    # hash the protocol document byte-for-byte
    #
    digest = hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()
    committed = COMMITMENT_PATH.read_text().strip()
    if digest != committed:
        print("error: study_f_protocol.json does NOT match its "
              "commitment hash")
        print(f"  committed: {committed}")
        print(f"  computed:  {digest}")
        sys.exit(1)

    # exit gracefully
    #
    return digest
#
# end of function


def _wilson(successes: int, n: int) -> tuple:
    """
    function: _wilson

    arguments:
     successes: number of positive outcomes
     n:         number of trials

    return:
     (low, high) 95% Wilson score interval

    description:
     Same formula as tools/blind_corpus.py (z = 1.96); well-behaved
     at the small per-level counts used here.
    """

    # guard the empty case
    #
    if n == 0:
        return (0.0, 0.0)

    # standard Wilson score computation
    #
    z = 1.96
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (
        z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    )

    # exit gracefully
    #
    return (max(0.0, centre - half), min(1.0, centre + half))
#
# end of function


def apply_rule(report: dict) -> tuple:
    """
    function: apply_rule

    arguments:
     report: a summary() report dict

    return:
     (flags, primary) — the raised mechanism flags (list, canonical
     order) and the primary label under the committed precedence

    description:
     The frozen decision rule of study_f_protocol.json: a mechanism
     flag is raised by its analyzer's status string, and the primary
     label is the highest-precedence raised flag ('healthy' iff no
     flag is raised). The rule embeds no knowledge of the corpus.
    """

    # read the three analyzer statuses
    #
    tra = (report.get("trainability") or {}).get("status") or ""
    mea = (report.get("measurement") or {}).get("status") or ""
    noi = (report.get("noise") or {}).get("status") or ""

    # raise flags exactly as committed
    #
    flags = []
    if tra == "Barren Plateau Detected":
        flags.append("barren_plateau")
    if mea.startswith("Shot Noise Dominated"):
        flags.append("shot_starved")
    if noi.startswith("Noise Dominated"):
        flags.append("noise_dominated")

    # primary by committed precedence; healthy iff nothing raised
    #
    for label in ("noise_dominated", "barren_plateau", "shot_starved"):
        if label in flags:
            return flags, label

    # exit gracefully
    #
    return flags, "healthy"
#
# end of function


def _achieved_stats(report: dict) -> dict:
    """
    function: _achieved_stats

    arguments:
     report: a summary() report dict

    return:
     the achieved statistics recorded per run (descriptive only)

    description:
     Extracts the measured variance, estimated SNR, and estimated
     circuit fidelity so the transition can also be stated in
     evidence units.
    """

    # exit gracefully
    #
    return {
        "variance": (report.get("trainability") or {}).get("variance"),
        "estimated_snr":
            (report.get("measurement") or {}).get("estimated_snr"),
        "estimated_circuit_fidelity":
            (report.get("noise") or {}).get(
                "estimated_circuit_fidelity"),
    }
#
# end of function


def _generate_run(spec: dict, out_root: Path) -> Path:
    """
    function: _generate_run

    arguments:
     spec:     one manifest entry (arm, knob values, seed)
     out_root: corpus root directory

    return:
     the run directory of the sealed trace

    description:
     Dispatches one corpus run to the Study E generators according to
     its arm. All planting is through physical knobs.
    """

    arm, seed = spec["arm"], spec["seed"]

    # barren sweep: statevector landscape at the depth-n rule
    #
    if arm == "barren_plateau":
        n = spec["n_qubits"]
        return sample_landscape(
            out_root, n, n, "linear", 300, seed, tags=_neutral_tags(),
        )

    # shot sweep: the Study E shot generator with a moving budget
    #
    if arm == "shot_starved":
        est = BackendEstimatorV2(
            backend=AerSimulator(seed_simulator=seed))
        return _train(out_root, 4, 2, seed, est,
                      precision=spec["precision"])

    # noise sweep: the Study E noise generator with a moving depth
    #
    if arm == "noise_dominated":
        from qiskit_ibm_runtime.fake_provider import FakeManilaV2
        backend = FakeManilaV2()
        est = BackendEstimatorV2(backend=backend)
        return _train(out_root, 4, spec["depth"], seed, est,
                      precision=0.1, transpile_backend=backend)

    # healthy controls: the Study E healthy generator
    #
    if arm == "healthy":
        est = BackendEstimatorV2(
            backend=AerSimulator(seed_simulator=seed))
        return _train(out_root, 4, spec["depth"], seed, est,
                      precision=0.02)

    # mixtures at strong settings (advance predictions in protocol)
    #
    if arm == "mix_noise_shot":
        from qiskit_ibm_runtime.fake_provider import FakeManilaV2
        backend = FakeManilaV2()
        est = BackendEstimatorV2(backend=backend)
        return _train(out_root, 4, 16, seed, est,
                      precision=0.45, transpile_backend=backend)

    if arm == "mix_barren_shot":
        est = BackendEstimatorV2(
            backend=AerSimulator(seed_simulator=seed))
        return sample_landscape(
            out_root, 8, 24, "linear", 200, seed,
            tags=_neutral_tags(), estimator=est, precision=0.45,
        )

    if arm == "mix_barren_noise":
        # sample_landscape builds abstract circuits, but a device
        # backend needs ISA form; mirror the Study E noise path
        # (transpile + layout) around the same recording pattern
        from qiskit.transpiler.preset_passmanagers import (
            generate_preset_pass_manager,
        )
        from qiskit_ibm_runtime.fake_provider import FakeGuadalupeV2
        from hilbertbench.integrations.qiskit import (
            HilbertEstimatorProxy,
        )
        from hilbertbench.models import Mode
        from hilbertbench.recorder.tape import HilbertTape
        from common import build_ansatz, pair_observable

        backend = FakeGuadalupeV2()
        qc, n_params = build_ansatz(8, 24, "linear")
        observable = pair_observable(8)
        pm = generate_preset_pass_manager(
            backend=backend, optimization_level=1,
            seed_transpiler=seed,
        )
        qc = pm.run(qc)
        observable = observable.apply_layout(qc.layout)

        rng = np.random.default_rng(seed)
        params = rng.uniform(0.0, 2.0 * np.pi, (200, n_params))
        est = BackendEstimatorV2(backend=backend)
        out_root.mkdir(parents=True, exist_ok=True)
        with HilbertTape(
            out_root, mode=Mode.active, tags=_neutral_tags(),
        ) as tape:
            proxy = HilbertEstimatorProxy(tape, real_estimator=est)
            for start in range(0, 200, 50):
                batch = params[start:start + 50]
                proxy.run(
                    [(qc, observable, batch, 0.045)]
                ).result()
        return tape.dir_path

    raise ValueError(f"unknown arm '{arm}'")
#
# end of function


def build_manifest(proto: dict) -> list:
    """
    function: build_manifest

    arguments:
     proto: the protocol dict

    return:
     the ordered list of run specs (arm, knob, seed, truth flags)

    description:
     Expands the committed grids and seed rules into one spec per
     run. Deterministic: the corpus is fully specified by the
     committed protocol.
    """

    specs = []
    sweeps = proto["sweeps"]

    # barren sweep
    #
    for li, n in enumerate(sweeps["barren_plateau"]["levels_n_qubits"]):
        for r in range(sweeps["barren_plateau"]["replicates_per_level"]):
            specs.append({
                "arm": "barren_plateau", "n_qubits": n, "level": n,
                "seed": 3000 + 10 * li + r,
                "truth_flags": ["barren_plateau"],
            })

    # shot sweep
    #
    for li, p in enumerate(sweeps["shot_starved"]["levels_precision"]):
        for r in range(sweeps["shot_starved"]["replicates_per_level"]):
            specs.append({
                "arm": "shot_starved", "precision": p, "level": p,
                "seed": 3200 + 10 * li + r,
                "truth_flags": ["shot_starved"],
            })

    # noise sweep
    #
    for li, d in enumerate(sweeps["noise_dominated"]["levels_depth"]):
        for r in range(sweeps["noise_dominated"]["replicates_per_level"]):
            specs.append({
                "arm": "noise_dominated", "depth": d, "level": d,
                "seed": 3400 + 10 * li + r,
                "truth_flags": ["noise_dominated"],
            })

    # healthy controls
    #
    for li, d in enumerate(sweeps["healthy_controls"]["levels_depth"]):
        for r in range(sweeps["healthy_controls"]["replicates_per_level"]):
            specs.append({
                "arm": "healthy", "depth": d, "level": d,
                "seed": 3600 + 10 * li + r,
                "truth_flags": [],
            })

    # mixtures
    #
    for r in range(proto["mixtures"]["noise_plus_shot"]["replicates"]):
        specs.append({
            "arm": "mix_noise_shot", "level": "strong",
            "seed": 3800 + r,
            "truth_flags": ["noise_dominated", "shot_starved"],
        })
    for r in range(proto["mixtures"]["barren_plus_shot"]["replicates"]):
        specs.append({
            "arm": "mix_barren_shot", "level": "strong",
            "seed": 3820 + r,
            "truth_flags": ["barren_plateau", "shot_starved"],
        })
    for r in range(proto["mixtures"]["barren_plus_noise"]["replicates"]):
        specs.append({
            "arm": "mix_barren_noise", "level": "strong",
            "seed": 3840 + r,
            "truth_flags": ["barren_plateau", "noise_dominated"],
        })

    # exit gracefully
    #
    return specs
#
# end of function


def cmd_generate(_: argparse.Namespace) -> int:
    """
    function: cmd_generate

    arguments:
     _: parsed CLI arguments (unused)

    return:
     process exit code (0 on success)

    description:
     Verifies the protocol commitment, expands the committed grid,
     generates every run through the Study E generators, and writes
     the corpus manifest. Idempotent per spec: a spec whose run
     directory is already recorded in the manifest is skipped, so an
     interrupted generation resumes where it stopped.
    """

    # commitment first: the grid must predate the corpus
    #
    digest = _verify_commitment()
    print(f"protocol commitment verified: {digest}")

    proto = _protocol()
    specs = build_manifest(proto)
    CORPUS_ROOT.mkdir(parents=True, exist_ok=True)

    # resume support: reload prior manifest entries when present
    #
    done = {}
    if MANIFEST_PATH.exists():
        done = json.loads(MANIFEST_PATH.read_text())

    # generate every missing run
    #
    for k, spec in enumerate(specs, 1):
        key = f"{spec['arm']}_s{spec['seed']}"
        if key in done:
            print(f"  [{k:>3}/{len(specs)}] {key:<28} (already done)")
            continue
        run_dir = _generate_run(spec, CORPUS_ROOT)
        rel = str(Path(run_dir).relative_to(CORPUS_ROOT))
        done[key] = dict(spec, run=rel)
        MANIFEST_PATH.write_text(json.dumps(done, indent=2) + "\n")
        print(f"  [{k:>3}/{len(specs)}] {key:<28} -> {rel}")

    print(f"\ncorpus complete: {len(done)} runs under {CORPUS_ROOT}")

    # exit gracefully
    #
    return 0
#
# end of function


def _score_sweeps(entries: list) -> dict:
    """
    function: _score_sweeps

    arguments:
     entries: manifest entries joined with rule outputs

    return:
     per-arm, per-level detection metrics plus transition report

    description:
     For each sweep arm and knob level: the detection rate of the
     planted mechanism with a Wilson interval, the rates of every
     other flag, and the per-run achieved statistics. The transition
     report brackets the knob interval between <=0.2 and >=0.8
     detection, as committed.
    """

    out = {}
    sweep_arms = ["barren_plateau", "shot_starved", "noise_dominated",
                  "healthy"]
    for arm in sweep_arms:
        rows = [e for e in entries if e["arm"] == arm]
        if not rows:
            continue
        levels = sorted({r["level"] for r in rows})
        per_level = []
        for lv in levels:
            sub = [r for r in rows if r["level"] == lv]
            n = len(sub)
            if arm == "healthy":
                hits = sum(1 for r in sub if r["flags"])
            else:
                hits = sum(1 for r in sub if arm in r["flags"])
            lo, hi = _wilson(hits, n)
            per_level.append({
                "level": lv, "n": n,
                "detected": hits, "rate": hits / n,
                "rate_ci95": [lo, hi],
                "other_flags": sorted({
                    f for r in sub for f in r["flags"] if f != arm
                }),
                "achieved": [r["achieved"] for r in sub],
            })
        entry = {"per_level": per_level}

        # transition bracket in knob units (sweeps only)
        #
        if arm != "healthy":
            low_edge = None
            high_edge = None
            for pl in per_level:
                if pl["rate"] <= 0.2:
                    low_edge = pl["level"]
                if high_edge is None and pl["rate"] >= 0.8:
                    high_edge = pl["level"]
            entry["transition"] = {
                "last_level_rate_le_0.2": low_edge,
                "first_level_rate_ge_0.8": high_edge,
            }
        out[arm] = entry

    # exit gracefully
    #
    return out
#
# end of function


def _score_mixtures(entries: list, proto: dict) -> dict:
    """
    function: _score_mixtures

    arguments:
     entries: manifest entries joined with rule outputs
     proto:   the protocol dict (for the advance predictions)

    return:
     per-mixture per-mechanism recall, exact-set match, predictions

    description:
     Scores the mixture arm as committed: per-mechanism recall from
     the flag set, exact-set match rate, and the advance prediction
     text carried alongside so hits and misses report identically.
    """

    names = {
        "mix_noise_shot": "noise_plus_shot",
        "mix_barren_shot": "barren_plus_shot",
        "mix_barren_noise": "barren_plus_noise",
    }
    out = {}
    for arm, pname in names.items():
        rows = [e for e in entries if e["arm"] == arm]
        if not rows:
            continue
        truth = rows[0]["truth_flags"]
        recall = {}
        for mech in truth:
            hit = sum(1 for r in rows if mech in r["flags"])
            lo, hi = _wilson(hit, len(rows))
            recall[mech] = {
                "detected": hit, "n": len(rows),
                "rate": hit / len(rows), "rate_ci95": [lo, hi],
            }
        exact = sum(
            1 for r in rows if sorted(r["flags"]) == sorted(truth)
        )
        out[pname] = {
            "n": len(rows),
            "truth_flags": truth,
            "per_mechanism_recall": recall,
            "exact_set_match": exact / len(rows),
            "primaries": sorted({r["primary"] for r in rows}),
            "advance_prediction":
                proto["mixtures"][pname]["advance_prediction"],
        }

    # exit gracefully
    #
    return out
#
# end of function


def _score_replay() -> dict:
    """
    function: _score_replay

    arguments:
     none

    return:
     the frozen rule's score on the original Study E corpus

    description:
     Applies the rule, unmodified, to all 72 Study E runs and scores
     primary accuracy (overall and per stratum) plus a flag-set
     top-2 analogue against the manifest ground truth.
    """

    manifest = json.loads((REPLAY_ROOT / "manifest.json").read_text())
    labels = sorted({v["label"] for v in manifest.values()})
    rows = []
    for rel, truth in sorted(manifest.items()):
        report = summary(HilbertTrace(str(REPLAY_ROOT / rel)))
        flags, primary = apply_rule(report)
        rows.append({
            "run": rel,
            "stratum": truth.get("stratum", "simulated"),
            "truth": truth["label"],
            "primary": primary,
            "flags": flags,
        })

    def _block(sub: list) -> dict:
        n = len(sub)
        correct = sum(1 for r in sub if r["primary"] == r["truth"])
        top2 = sum(
            1 for r in sub
            if r["truth"] == r["primary"] or (
                r["truth"] != "healthy" and r["truth"] in r["flags"])
        )
        lo, hi = _wilson(correct, n)
        confusion: dict = {}
        for r in sub:
            confusion.setdefault(r["truth"], {})
            confusion[r["truth"]][r["primary"]] = (
                confusion[r["truth"]].get(r["primary"], 0) + 1
            )
        return {
            "n": n, "correct": correct, "accuracy": correct / n,
            "accuracy_ci95": [lo, hi],
            "flagset_top2_accuracy": top2 / n,
            "confusion": confusion,
        }

    strata = sorted({r["stratum"] for r in rows})
    out = {
        "labels": labels,
        "overall": _block(rows),
        "strata": {
            st: _block([r for r in rows if r["stratum"] == st])
            for st in strata
        },
        "rows": rows,
    }

    # exit gracefully
    #
    return out
#
# end of function


def cmd_score(_: argparse.Namespace) -> int:
    """
    function: cmd_score

    arguments:
     _: parsed CLI arguments (unused)

    return:
     process exit code (0 on success)

    description:
     Verifies the protocol commitment, applies the frozen rule to
     every hard-mode run and to the Study E replay corpus, computes
     the committed metrics, and writes results/study_f/results.json.
    """

    # the rule must still match its commitment
    #
    digest = _verify_commitment()
    print(f"protocol commitment verified: {digest}")

    proto = _protocol()
    manifest = json.loads(MANIFEST_PATH.read_text())

    # run the rule over the hard-mode corpus
    #
    entries = []
    for key, spec in sorted(manifest.items()):
        report = summary(HilbertTrace(str(CORPUS_ROOT / spec["run"])))
        flags, primary = apply_rule(report)
        entries.append(dict(
            spec, key=key, flags=flags, primary=primary,
            achieved=_achieved_stats(report),
        ))
        print(f"  {key:<28} flags={flags or ['-']} "
              f"primary={primary}")

    # committed metrics
    #
    result = {
        "protocol": proto["protocol"],
        "commitment": digest,
        "n_runs": len(entries),
        "sweeps": _score_sweeps(entries),
        "mixtures": _score_mixtures(entries, proto),
        "replay_study_e": _score_replay(),
        "runs": entries,
    }

    out_dir = RESULTS_ROOT / "study_f"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n")

    # console digest of the headline numbers
    #
    print("\n== sweeps: transition brackets ==")
    for arm, blk in result["sweeps"].items():
        if arm == "healthy":
            rates = [pl for pl in blk["per_level"]]
            fp = sum(pl["detected"] for pl in rates)
            n = sum(pl["n"] for pl in rates)
            print(f"  healthy: false-positive {fp}/{n}")
            continue
        tr = blk["transition"]
        print(f"  {arm}: <=0.2 at {tr['last_level_rate_le_0.2']}, "
              f">=0.8 at {tr['first_level_rate_ge_0.8']}")
    print("\n== mixtures ==")
    for name, blk in result["mixtures"].items():
        rec = {m: v["rate"]
               for m, v in blk["per_mechanism_recall"].items()}
        print(f"  {name}: recall={rec} "
              f"exact={blk['exact_set_match']:.2f}")
    rep = result["replay_study_e"]
    print(f"\n== replay of Study E (frozen rule, no human) ==")
    print(f"  overall: {rep['overall']['correct']}/{rep['overall']['n']}"
          f" = {rep['overall']['accuracy']:.3f}, top-2 "
          f"{rep['overall']['flagset_top2_accuracy']:.3f}")
    for st, blk in rep["strata"].items():
        print(f"  {st}: {blk['correct']}/{blk['n']} "
              f"= {blk['accuracy']:.3f}")
    print(f"\nwrote {out_path}")

    # exit gracefully
    #
    return 0
#
# end of function


def main(argv=None) -> int:
    """
    function: main

    arguments:
     argv: CLI arguments (defaults to sys.argv[1:])

    return:
     process exit code

    description:
     Parses the subcommand and dispatches to generate / score.
    """

    parser = argparse.ArgumentParser(
        prog=__FILE__,
        description="Study F: detection-boundary mapping",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("generate", help="build the hard-mode corpus")\
       .set_defaults(func=cmd_generate)
    sub.add_parser("score", help="apply the frozen rule and score")\
       .set_defaults(func=cmd_score)
    args = parser.parse_args(argv)

    # exit gracefully
    #
    return args.func(args)
#
# end of function


# begin gracefully
#
if __name__ == "__main__":
    sys.exit(main())
#
# end of file
