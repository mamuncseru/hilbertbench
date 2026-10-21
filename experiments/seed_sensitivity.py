#!/usr/bin/env python
#
# file: experiments/seed_sensitivity.py
#
# revision history:
#  20261021 (pm): initial version
#  20261021 (pm): add the same-seed verification and run-level
#                 (cluster) summaries
#
# Post-hoc sensitivity analysis of the simulator sampling seed.
#
# The shot-based simulated arms of Studies E, F and G were generated
# with BackendEstimatorV2(AerSimulator(seed_simulator=seed)). A fixed
# seed_simulator makes Aer reuse the same random numbers on EVERY
# estimator call of a run, so the shot noise is common-random-number
# noise rather than independent: identical parameters give identical
# estimates, and a ~5-shot estimate becomes a deterministic step
# function of the parameters. Real hardware draws independent shots.
#
# This script regenerates every affected run with the byte-identical
# generators, changing ONLY the sampler: each estimator call draws a
# fresh simulator seed from a stream seeded by (study, run seed,
# replicate), so the noise is independent across calls yet the whole
# analysis is reproducible. The frozen decision rule (study_f_hardmode.
# apply_rule) is applied to the regenerated traces and compared with
# the archived (seeded) rule outputs.
#
# Affected generators (seeded Aer):
#  Study E (sim stratum): healthy, shot_starved
#  Study F: shot_starved sweep, healthy controls, mix_barren_shot
#  Study G: shot_starved, healthy, mix_barren_shot
# The device-noise arms use FakeManilaV2 / FakeGuadalupeV2, whose
# sampler is unseeded, and the barren arms are exact statevector; both
# are unaffected and are not regenerated.
#
# This is an exploratory, disclosed analysis. It is rule-only: the
# human diagnoses cannot be re-obtained, and the archived corpora,
# scores and v1.0.0 instrument are left untouched.
#
#   python seed_sensitivity.py verify    # same-seed regeneration check
#   python seed_sensitivity.py run       # regenerate (resumable)
#   python seed_sensitivity.py score     # write the results file
#------------------------------------------------------------------------------

# future imports must come first
#
from __future__ import annotations

# import system modules
#
import argparse
import collections
import contextlib
import json
import sys
import warnings
from pathlib import Path

# import third-party modules
#
import numpy as np
from qiskit_aer import AerSimulator

# import hilbertbench modules
#
from hilbertbench import HilbertTrace
from hilbertbench.analysis import summary

# import experiment infrastructure
#
import study_e_corpus
import study_f_hardmode
from common import RESULTS_ROOT, TRACES_ROOT
from study_f_hardmode import _generate_run, apply_rule, build_manifest

# silence framework deprecation chatter for a clean report
#
warnings.filterwarnings("ignore")

#------------------------------------------------------------------------------
#
# global variables are listed here
#
#------------------------------------------------------------------------------

HERE = Path(__file__).parent
OUT_TRACES = TRACES_ROOT / "seed_sensitivity"
OUT_RESULTS = RESULTS_ROOT / "seed_sensitivity" / "results.json"

# independent replicates per affected run
#
N_REPLICATES = 3

# study codes mixed into the sampler seed stream
#
STUDY_CODE = {"E": 1, "F": 2, "G": 3}

# arms whose generator used the seeded AerSimulator
#
AFFECTED_E = ("healthy", "shot_starved")
AFFECTED_FG = ("shot_starved", "healthy", "mix_barren_shot")

#------------------------------------------------------------------------------
#
# classes are listed here
#
#------------------------------------------------------------------------------


class FreshSeedSimulator(AerSimulator):
    """
    class: FreshSeedSimulator

    description:
     An AerSimulator that draws a new seed_simulator for every run()
     call from a seeded stream, so shots are independent across
     estimator calls while the stream itself is reproducible.
    """

    def __init__(self, stream: tuple, **kwargs):
        super().__init__(**kwargs)
        self._rng = np.random.default_rng(np.random.SeedSequence(stream))

    def run(self, circuits, **run_options):
        run_options["seed_simulator"] = int(self._rng.integers(2**31 - 1))
        return super().run(circuits, **run_options)


#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------


@contextlib.contextmanager
def independent_sampling(study: str, replicate: int):
    """
    function: independent_sampling

    arguments:
     study:     'E', 'F' or 'G'
     replicate: replicate index

    description:
     Temporarily replaces the AerSimulator constructor seen by the
     Study E and Study F generator modules, so that the generators
     run unchanged except for the sampler. The run seed passed as
     seed_simulator becomes part of the fresh-seed stream.
    """

    def factory(seed_simulator=None, **kwargs):
        stream = (STUDY_CODE[study], int(seed_simulator or 0), replicate)
        return FreshSeedSimulator(stream, **kwargs)

    saved = (study_e_corpus.AerSimulator, study_f_hardmode.AerSimulator)
    study_e_corpus.AerSimulator = factory
    study_f_hardmode.AerSimulator = factory
    try:
        yield
    finally:
        study_e_corpus.AerSimulator, study_f_hardmode.AerSimulator = saved
#
# end of function


def _rule(run_dir) -> dict:
    """Frozen rule plus the achieved statistics for one trace."""
    rep = summary(HilbertTrace(str(run_dir)))
    flags, primary = apply_rule(rep)
    return {
        "flags": flags,
        "primary": primary,
        "variance": (rep["trainability"] or {}).get("variance"),
        "estimated_snr": (rep["measurement"] or {}).get("estimated_snr"),
    }
#
# end of function


def affected_specs() -> list:
    """
    function: affected_specs

    return:
     one entry per affected archived run: study, arm, level, seed,
     truth flags, and the archived (seeded) run directory

    description:
     Collects the affected runs of the three studies from their
     manifests, keeping the archived run directory so the archived
     rule output is recomputed from the archived trace itself.
    """

    out = []

    # Study E simulated stratum
    #
    root = TRACES_ROOT / "corpus_test"
    for rel, v in json.loads((root / "manifest.json").read_text()).items():
        if v.get("stratum") == "hardware" or v["label"] not in AFFECTED_E:
            continue
        out.append({
            "study": "E", "arm": v["label"], "level": None,
            "seed": v["seed"], "spec": {"label": v["label"]},
            "truth_flags": ([] if v["label"] == "healthy"
                            else [v["label"]]),
            "archived": str(root / rel),
        })

    # Study F (committed grid; archived run directories in manifest)
    #
    root = TRACES_ROOT / "corpus_hardmode"
    manifest = json.loads((root / "manifest.json").read_text())
    by_seed = {v["seed"]: v["run"] for v in manifest.values()}
    proto = json.loads((HERE / "study_f_protocol.json").read_text())
    for spec in build_manifest(proto):
        if spec["arm"] not in AFFECTED_FG:
            continue
        out.append({
            "study": "F", "arm": spec["arm"], "level": spec["level"],
            "seed": spec["seed"], "spec": spec,
            "truth_flags": spec["truth_flags"],
            "archived": str(root / by_seed[spec["seed"]]),
        })

    # Study G (private assignment, now unsealed; corpus manifest)
    #
    root = TRACES_ROOT / "corpus_g"
    manifest = json.loads((root / "manifest.json").read_text())
    by_seed = {v["seed"]: rel for rel, v in manifest.items()}
    specs = json.loads((HERE / "study_g_assignment.json").read_text())
    for spec in specs["specs"]:
        if spec["arm"] not in AFFECTED_FG:
            continue
        out.append({
            "study": "G", "arm": spec["arm"], "level": spec["level"],
            "seed": spec["seed"], "spec": spec,
            "truth_flags": spec["truth_flags"],
            "archived": str(root / by_seed[spec["seed"]]),
        })

    # exit gracefully
    #
    return out
#
# end of function


def _key(entry: dict, replicate: int) -> str:
    return f"{entry['study']}_{entry['arm']}_s{entry['seed']}_r{replicate}"


def cmd_run(_: argparse.Namespace) -> int:
    """
    function: cmd_run

    description:
     Regenerates every affected run N_REPLICATES times under
     independent sampling. Resumable: finished keys are skipped.
    """

    OUT_TRACES.mkdir(parents=True, exist_ok=True)
    index_path = OUT_TRACES / "index.json"
    index = (json.loads(index_path.read_text())
             if index_path.exists() else {})

    entries = affected_specs()
    total = len(entries) * N_REPLICATES
    k = 0
    for e in entries:
        for r in range(N_REPLICATES):
            k += 1
            key = _key(e, r)
            if key in index:
                continue
            out_root = OUT_TRACES / e["study"]
            with independent_sampling(e["study"], r):
                if e["study"] == "E":
                    run_dir = study_e_corpus.generate_one(
                        e["spec"]["label"], out_root, e["seed"])
                else:
                    run_dir = _generate_run(e["spec"], out_root)
            index[key] = str(Path(run_dir).relative_to(OUT_TRACES))
            index_path.write_text(json.dumps(index, indent=1) + "\n")
            print(f"  [{k:>3}/{total}] {key}", flush=True)

    print(f"\ndone: {len(index)} regenerated runs under {OUT_TRACES}")
    return 0
#
# end of function


def _outcomes(run_dir) -> list:
    """Every logged expectation value of a trace, in record order."""
    out = []
    for sp in HilbertTrace(str(run_dir)).completed():
        out.extend(np.asarray(sp.outcome, dtype=float).ravel().tolist())
    return out


def cmd_verify(_: argparse.Namespace) -> int:
    """
    function: cmd_verify

    description:
     Regenerates every affected run with its ORIGINAL generator and
     fixed simulator seed (no patching) and checks that the logged
     values and the frozen rule's output match the archived trace.
     This shows that the archived corpora are the deterministic
     product of the fixed seed, so the rerun differs only in the
     sampler.
    """

    out_root = OUT_TRACES / "verify"
    rows = []
    for e in affected_specs():
        if e["study"] == "E":
            run_dir = study_e_corpus.generate_one(
                e["spec"]["label"], out_root / "E", e["seed"])
        else:
            run_dir = _generate_run(e["spec"], out_root / e["study"])
        same_values = _outcomes(run_dir) == _outcomes(e["archived"])
        a, b = _rule(e["archived"]), _rule(run_dir)
        same_rule = (a["flags"], a["primary"]) == (b["flags"], b["primary"])
        rows.append({"study": e["study"], "arm": e["arm"],
                     "seed": e["seed"], "same_values": same_values,
                     "same_rule_output": same_rule})
        print(f"  {e['study']} {e['arm']:<16} s{e['seed']}: values "
              f"{'identical' if same_values else 'DIFFER'}, rule "
              f"{'identical' if same_rule else 'DIFFERS'}", flush=True)

    summary = {
        "n": len(rows),
        "identical_values": sum(r["same_values"] for r in rows),
        "identical_rule_output": sum(r["same_rule_output"] for r in rows),
    }
    path = OUT_RESULTS.parent / "verify.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"summary": summary, "rows": rows},
                               indent=1) + "\n")
    print(json.dumps(summary), f"\nwrote {path}")
    return 0
#
# end of function


def _correct(primary: str, truth_flags: list) -> bool:
    """The registered Study G correctness rule (also used for E)."""
    if not truth_flags:
        return primary == "healthy"
    return primary in truth_flags


def _detected(arm: str, flags: list) -> bool:
    """The committed Study F detection rule."""
    if arm == "healthy":
        return bool(flags)
    return arm in flags


def cmd_score(_: argparse.Namespace) -> int:
    """
    function: cmd_score

    description:
     Applies the frozen rule to the archived (seeded) traces and to
     the regenerated (independent) traces, and tabulates per study,
     arm and level: primary-label correctness, detection of the
     planted mechanism (Study F definition), false barren flags, and
     the median achieved variance and SNR.
    """

    index = json.loads((OUT_TRACES / "index.json").read_text())
    rows = []
    for e in affected_specs():
        arch = _rule(e["archived"])
        reps = [_rule(OUT_TRACES / index[_key(e, r)])
                for r in range(N_REPLICATES) if _key(e, r) in index]
        rows.append({
            "study": e["study"], "arm": e["arm"], "level": e["level"],
            "seed": e["seed"], "truth_flags": e["truth_flags"],
            "seeded": arch, "independent": reps,
        })

    def _tally(items, arm, truth):
        n = len(items)
        return {
            "n": n,
            "primary_correct": sum(_correct(x["primary"], truth)
                                   for x in items),
            "detected": sum(_detected(arm, x["flags"]) for x in items),
            "barren_flag": sum("barren_plateau" in x["flags"]
                               for x in items),
            "primary_counts": dict(collections.Counter(
                x["primary"] for x in items)),
            "median_variance": float(np.median(
                [x["variance"] for x in items
                 if x["variance"] is not None] or [np.nan])),
            "median_snr": float(np.median(
                [x["estimated_snr"] for x in items
                 if x["estimated_snr"] is not None] or [np.nan])),
        }

    def _runs(sub, arm, truth):
        """Run-level view: a run counts when at least two of its three
        reruns do; the interval resamples runs (cluster bootstrap)."""
        det = [np.mean([_detected(arm, y["flags"]) for y in x["independent"]])
               for x in sub]
        cor = [np.mean([_correct(y["primary"], truth)
                        for y in x["independent"]]) for x in sub]
        rng = np.random.default_rng(len(sub) * 7919 + len(arm))
        boot = [np.mean(rng.choice(det, len(det))) for _ in range(2000)]
        return {
            "runs": len(sub),
            "runs_majority_detected": int(sum(v >= 0.5 for v in det)),
            "runs_majority_correct": int(sum(v >= 0.5 for v in cor)),
            "detection_rate": float(np.mean(det)),
            "detection_rate_cluster_ci95": [
                float(q) for q in np.percentile(boot, [2.5, 97.5])],
            "median_snr_independent": float(np.median(
                [y["estimated_snr"] for x in sub for y in x["independent"]
                 if y["estimated_snr"] is not None] or [np.nan])),
        }

    # aggregate by study / arm / level
    #
    table = []
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["study"], r["arm"], str(r["level"]))].append(r)
    for (study, arm, level), sub in sorted(groups.items()):
        truth = sub[0]["truth_flags"]
        table.append({
            "study": study, "arm": arm, "level": level,
            "seeded": _tally([x["seeded"] for x in sub], arm, truth),
            "independent": _tally(
                [y for x in sub for y in x["independent"]], arm, truth),
            "independent_by_run": _runs(sub, arm, truth),
        })

    # Study G: rule accuracy over all 63 runs with the affected runs
    # replaced by their independent-sampling replicates (expected
    # value over replicates); unaffected runs keep their archived
    # rule label
    #
    g_rows = json.loads(
        (RESULTS_ROOT / "study_g" / "results.json").read_text())["rows"]
    g_affected = [r for r in rows if r["study"] == "G"]
    archived_correct = sum(r["r_ok"] for r in g_rows)
    unaffected_correct = sum(
        r["r_ok"] for r in g_rows if r["arm"] not in AFFECTED_FG)
    exp_affected = sum(
        np.mean([_correct(y["primary"], r["truth_flags"])
                 for y in r["independent"]])
        for r in g_affected)
    g_summary = {
        "n": len(g_rows),
        "rule_correct_archived": archived_correct,
        "rule_correct_unaffected_arms": unaffected_correct,
        "rule_correct_affected_arms_expected_independent":
            float(exp_affected),
        "rule_accuracy_archived": archived_correct / len(g_rows),
        "rule_accuracy_independent_expected":
            float((unaffected_correct + exp_affected) / len(g_rows)),
    }

    OUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    OUT_RESULTS.write_text(json.dumps({
        "protocol": "seed_sensitivity_v1",
        "status": ("post-hoc, exploratory, disclosed; rule-only; "
                   "archived corpora and scores unmodified"),
        "n_replicates": N_REPLICATES,
        "table": table,
        "study_g_rule_accuracy": g_summary,
        "rows": rows,
    }, indent=1) + "\n")

    # print a compact report
    #
    for t in table:
        s, i = t["seeded"], t["independent"]
        print(f"{t['study']} {t['arm']:<16} {t['level']:>6}  "
              f"seeded: correct {s['primary_correct']}/{s['n']} "
              f"det {s['detected']}/{s['n']} bp {s['barren_flag']}  |  "
              f"indep: correct {i['primary_correct']}/{i['n']} "
              f"det {i['detected']}/{i['n']} bp {i['barren_flag']}  "
              f"snr {s['median_snr']:.2f}->{i['median_snr']:.2f}")
    print(json.dumps(g_summary, indent=1))
    print(f"\nwrote {OUT_RESULTS}")
    return 0
#
# end of function


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify").set_defaults(func=cmd_verify)
    sub.add_parser("run").set_defaults(func=cmd_run)
    sub.add_parser("score").set_defaults(func=cmd_score)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
#
# end of file
