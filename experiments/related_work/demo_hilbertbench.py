#!/usr/bin/env python
#
# file: experiments/related_work/demo_hilbertbench.py
#
# The baseline for the comparative analysis: post-hoc mechanism
# attribution of a FINISHED run, from its sealed trace alone.
#
# The run under diagnosis is a development-corpus run whose failure
# was planted by construction (deep transpiled circuit on a calibrated
# noisy device model). Nothing is re-executed here: the process that
# produced the run is long dead, and everything below reads evidence.
#
#   python demo_hilbertbench.py
#------------------------------------------------------------------------------
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from hilbertbench import HilbertTrace
from hilbertbench.analysis import summary

HERE = Path(__file__).parent
CORPUS = HERE.parent / "traces" / "corpus_dev"


def main() -> int:
    # pick the planted noise-domination run from the dev corpus
    #
    manifest = json.loads((CORPUS / "manifest.json").read_text())
    rel = next(r for r, t in sorted(manifest.items())
               if t["label"] == "noise_dominated")
    print(f"diagnosing finished run: {rel} (process long dead)\n")

    report = summary(HilbertTrace(str(CORPUS / rel)))
    for axis in ("trainability", "measurement", "optimization",
                 "noise"):
        d = report[axis]
        print(f"  {axis:<13} {d['status']}")
    fid = report["noise"].get("estimated_circuit_fidelity")
    print(f"\n  estimated circuit fidelity from the recorded "
          f"calibration snapshot: {fid:.3f}")
    print(f"  dominant error source: "
          f"{report['noise'].get('dominant_error_source')}")

    print("""
WHAT THIS SHOWS
  - attribution happened AFTER the run finished, from preserved
    evidence: no re-execution, no extra circuits, no quantum cost
  - the evidence includes the device calibration in force at
    execution time, the transpiled structure, and the shot record
  - the trace is append-only and sealed: a third party can verify
    it was not edited after the fact
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
