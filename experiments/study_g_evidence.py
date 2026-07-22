#!/usr/bin/env python
#
# file: experiments/study_g_evidence.py
#
# revision history:
#  20260722 (am): initial version
#
# Post-hoc descriptive extraction of the achieved evidence statistics
# for the Study G corpus, for the manipulation-check panels of the
# boundary figure.
#
# This is deliberately a SEPARATE output file. results/study_g/
# results.json is the scored record archived on Zenodo
# (10.5281/zenodo.21482518) and is left byte-for-byte untouched so the
# published dataset stays verifiable.
#
# Nothing here is a hypothesis test: it reads, per run, the three
# statistics the frozen analyzers computed (landscape variance,
# estimated SNR, estimated circuit fidelity) so the figure can show
# whether each planted knob actually moved the evidence it was
# supposed to move.
#
#   python study_g_evidence.py
#------------------------------------------------------------------------------
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

from hilbertbench import HilbertTrace
from hilbertbench.analysis import summary

from common import RESULTS_ROOT, TRACES_ROOT

warnings.filterwarnings("ignore")

BLINDED = TRACES_ROOT / "blinded_g"
SCORES = RESULTS_ROOT / "study_g" / "results.json"
OUT = RESULTS_ROOT / "study_g" / "evidence.json"


def main() -> int:
    rows = json.loads(SCORES.read_text())["rows"]

    records = []
    for r in rows:
        rep = summary(HilbertTrace(str(BLINDED / r["id"])))
        records.append({
            "id": r["id"],
            "arm": r["arm"],
            "level": r["level"],
            "truth_flags": r["truth_flags"],
            "variance": (rep["trainability"] or {}).get("variance"),
            "estimated_snr": (rep["measurement"] or {}).get(
                "estimated_snr"),
            "estimated_circuit_fidelity": (rep["noise"] or {}).get(
                "estimated_circuit_fidelity"),
            "trainability_status": rep["trainability"]["status"],
            "measurement_status": rep["measurement"]["status"],
            "noise_status": rep["noise"]["status"],
            "h_primary": r["h_primary"],
            "h_secondary": r["h_secondary"],
            "h_conf": r["h_conf"],
            "r_primary": r["r_primary"],
            "r_flags": r["r_flags"],
        })
        print(f"  {r['id']} {r['arm']:<16} level={r['level']}")

    OUT.write_text(json.dumps({
        "note": ("post-hoc descriptive extraction of achieved evidence "
                 "statistics for the Study G corpus; the scored record "
                 "in results.json is unmodified and matches the "
                 "archived dataset"),
        "n": len(records),
        "records": records,
    }, indent=2) + "\n")
    print(f"\nwrote {OUT} ({len(records)} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
#
# end of file
