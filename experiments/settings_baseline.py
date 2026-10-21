#!/usr/bin/env python
#
# file: experiments/settings_baseline.py
#
# revision history:
#  20261021 (pm): initial version
#
# Settings-only baseline for the two blinded corpora, requested in
# review. It asks whether the planted class can be recovered from the
# run's configuration alone, without any diagnostic statistic.
#
# Features are read from the trace, never from the analyzers' verdicts
# or statistics:
#   mode (active landscape sampling vs passive trajectory logging),
#   circuit width, entangling-gate count of the largest recorded
#   circuit, recorded target precision, and whether a device
#   calibration snapshot is present.
#
# Classifier: leave-one-out 1-nearest-neighbour on standardised
# features, scored with each study's registered correctness rule. A
# high score means the corpus design leaks the class through its
# settings, so the study tests whether the reader used the diagnostic
# evidence, not whether the cause was hidden.
#
#   python settings_baseline.py
#------------------------------------------------------------------------------

# future imports must come first
#
from __future__ import annotations

# import system modules
#
import json
import math
import sys
import warnings
from pathlib import Path

# import third-party modules
#
import numpy as np

# import hilbertbench modules
#
from hilbertbench import HilbertTrace
from hilbertbench.analysis.circuit import circuit_structure

# import experiment infrastructure
#
from common import RESULTS_ROOT, TRACES_ROOT, save_result

# silence framework deprecation chatter for a clean report
#
warnings.filterwarnings("ignore")

#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------


def _precision(trace: HilbertTrace):
    """The target precision recorded on the first completed span."""
    for sp in trace.completed():
        for ev in sp.events:
            attrs = ev.get("attributes") or {}
            if "precision" in attrs:
                return attrs["precision"]
        return None
    return None


def settings(run_dir: Path) -> dict:
    """
    function: settings

    arguments:
     run_dir: a sealed run directory

    return:
     the configuration features of the run

    description:
     Reads only configuration: mode, width, entangling gates,
     precision and calibration presence.
    """

    t = HilbertTrace(str(run_dir))
    prim = circuit_structure(t)["primary"] or {}
    return {
        "active": 1.0 if str(t.mode).endswith("active") else 0.0,
        "n_qubits": prim.get("num_qubits") or 0,
        "entangling_gates": prim.get("entangling_gates") or 0,
        "precision": _precision(t),
        "calibration": 1.0 if t.calibration() is not None else 0.0,
    }


def _vector(s: dict) -> list:
    return [
        s["active"],
        math.log2(max(s["n_qubits"], 1)),
        math.log1p(s["entangling_gates"]),
        math.log(s["precision"]) if s["precision"] else math.log(1e-3),
        s["calibration"],
    ]


def _correct(primary: str, truth_flags: list) -> bool:
    """Registered correctness: healthy iff nothing planted, else any
    planted mechanism counts."""
    if not truth_flags:
        return primary == "healthy"
    return primary in truth_flags


def loo_1nn(items: list) -> dict:
    """
    function: loo_1nn

    arguments:
     items: dicts with 'features', 'label', 'truth_flags'

    return:
     accuracy and per-item predictions

    description:
     Leave-one-out 1-nearest-neighbour on standardised features;
     ties are broken by the lowest index.
    """

    X = np.array([_vector(i["features"]) for i in items], dtype=float)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X - X.mean(axis=0)) / sd
    preds = []
    for k in range(len(items)):
        d = np.sum((Z - Z[k]) ** 2, axis=1)
        d[k] = np.inf
        preds.append(items[int(np.argmin(d))]["label"])
    ok = [_correct(p, i["truth_flags"]) for p, i in zip(preds, items)]
    signatures = {}
    for i in items:
        key = json.dumps(i["features"], sort_keys=True)
        signatures.setdefault(key, set()).add(i["label"])
    shared = sum(1 for v in signatures.values() if len(v) > 1)
    return {
        "n": len(items), "correct": int(sum(ok)),
        "accuracy": float(np.mean(ok)),
        "distinct_signatures": len(signatures),
        "signatures_shared_across_classes": shared,
        "predictions": preds,
    }


def main() -> int:

    # Study E: the 72-run blinded test corpus, per stratum
    #
    root = TRACES_ROOT / "corpus_test"
    e_items = []
    for rel, v in json.loads((root / "manifest.json").read_text()).items():
        lab = v["label"]
        e_items.append({
            "run": rel, "stratum": v.get("stratum") or "simulated",
            "label": lab,
            "truth_flags": [] if lab == "healthy" else [lab],
            "features": settings(root / rel),
        })
    e_all = loo_1nn(e_items)
    e_strata = {s: loo_1nn([i for i in e_items if i["stratum"] == s])
                for s in sorted({i["stratum"] for i in e_items})}

    # Study G: the 63-run blinded corpus; the label used for the
    # neighbour vote is the arm, scoring uses the registered truth
    #
    root = TRACES_ROOT / "corpus_g"
    g_items = []
    for rel, v in json.loads((root / "manifest.json").read_text()).items():
        g_items.append({
            "run": rel, "label": v["label"], "arm": v["arm"],
            "level": v["level"], "truth_flags": v["truth_flags"],
            "features": settings(root / rel),
        })
    g_all = loo_1nn(g_items)

    for name, r in [("E all", e_all)] + [
            (f"E {k}", v) for k, v in e_strata.items()] + [("G", g_all)]:
        print(f"  {name:<14} settings-only LOO 1-NN: {r['correct']}/{r['n']}"
              f" = {r['accuracy']:.3f}  ({r['distinct_signatures']} "
              f"signatures, {r['signatures_shared_across_classes']} shared"
              " across classes)")

    path = save_result("settings_baseline", {
        "protocol": "settings_baseline_v1",
        "status": "post-hoc, exploratory; requested in review",
        "features": ["active mode", "log2 width",
                     "log(1 + entangling gates)", "log precision",
                     "calibration present"],
        "classifier": "leave-one-out 1-nearest-neighbour, standardised",
        "study_e": {"all": e_all, "strata": e_strata},
        "study_g": g_all,
    })
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
#
# end of file
