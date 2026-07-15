#!/usr/bin/env python
#
# file: experiments/study_g_corpus.py
#
# revision history:
#  20260715 (am): initial version
#
# Study G -- second blinded corpus (the detection-boundary exam).
#
# Generates the 63-run round-2 corpus from the PRIVATE assignment file
# (study_g_assignment.json, hash-committed in the public protocol).
# Generation dispatches through the Study F generator, so every run is
# produced by the byte-identical Study E code paths; the assignment
# order is pre-shuffled, so timestamps carry no arm information.
#
# The manifest written here feeds tools/blind_corpus.py blind, exactly
# as in Study E. The assignment file and the manifest are PRIVATE
# until unsealing.
#
#   python study_g_corpus.py
#------------------------------------------------------------------------------

# future imports must come first
#
from __future__ import annotations

# import system modules
#
import hashlib
import json
import sys
import warnings
from pathlib import Path

# import experiment infrastructure (Study F dispatch = Study E
# generators)
#
from common import TRACES_ROOT
from study_f_hardmode import _generate_run

# silence framework deprecation chatter for a clean report
#
warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
ASSIGNMENT_PATH = HERE / "study_g_assignment.json"
ASSIGNMENT_SHA = HERE / "study_g_assignment.sha256"
CORPUS_ROOT = TRACES_ROOT / "corpus_g"


def main() -> int:
    # the assignment must match its published commitment
    #
    text = ASSIGNMENT_PATH.read_text()
    digest = hashlib.sha256(text.encode()).hexdigest()
    committed = ASSIGNMENT_SHA.read_text().strip()
    if digest != committed:
        print("error: assignment does NOT match its commitment")
        return 1
    print(f"assignment commitment verified: {digest}")

    specs = json.loads(text)["specs"]
    CORPUS_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = CORPUS_ROOT / "manifest.json"
    manifest = (json.loads(manifest_path.read_text())
                if manifest_path.exists() else {})
    done_seeds = {v["seed"] for v in manifest.values()}

    # generate in the assignment's pre-shuffled order; resumable
    #
    for k, spec in enumerate(specs, 1):
        if spec["seed"] in done_seeds:
            print(f"  [{k:>2}/{len(specs)}] seed {spec['seed']} "
                  "(already done)")
            continue
        run_dir = _generate_run(spec, CORPUS_ROOT)
        rel = str(Path(run_dir).relative_to(CORPUS_ROOT))
        manifest[rel] = {
            "label": spec["label"],
            "arm": spec["arm"],
            "level": spec["level"],
            "seed": spec["seed"],
            "truth_flags": spec["truth_flags"],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"  [{k:>2}/{len(specs)}] {spec['arm']:<18} -> {rel}")

    print(f"\ncorpus complete: {len(manifest)} runs under {CORPUS_ROOT}")
    print("next: python ../tools/blind_corpus.py blind "
          f"--manifest {manifest_path} --out "
          f"{TRACES_ROOT / 'blinded_g'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
#
# end of file
