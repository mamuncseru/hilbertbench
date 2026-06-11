#!/usr/bin/env python
#
# file: tools/blind_corpus.py
#
# revision history:
#  20260610 (am): initial version
#
# Blinded-corpus protocol tool for the HilbertBench validation study.
#
# The validation protocol: Researcher A generates a corpus of runs with
# planted failure modes (barren plateau, shot starvation, noise
# domination, healthy). Researcher B must diagnose each run from its
# trace alone. This tool implements the blinding machinery:
#
#   blind  copy each run dir verbatim under a random ID, write the
#          private answer key, and emit a SHA-256 commitment of the
#          key that is published BEFORE diagnosis begins
#   audit  scan run dirs for ground-truth leakage (label-like tags)
#   score  after diagnosis, verify the key against its commitment and
#          report a confusion matrix with per-label precision/recall
#
# Traces are sealed (INV-002), so blinding never edits trace contents —
# runs are copied byte-for-byte and only the directory name changes.
# Ground truth must therefore never be written into tags; the audit
# subcommand enforces this before blinding.
#
#   python tools/blind_corpus.py blind --manifest corpus/manifest.json \
#       --out blinded/
#   python tools/blind_corpus.py audit --corpus blinded/
#   python tools/blind_corpus.py score --key answer_key.json \
#       --commitment answer_key.sha256 --diagnosis diagnosis_filled.json
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
import re
import secrets
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

#------------------------------------------------------------------------------
#
# global variables are listed here
#
#------------------------------------------------------------------------------

# set the filename using basename
#
__FILE__ = os.path.basename(__file__)

# tag keys or values matching these patterns indicate that ground
# truth leaked into the trace metadata; blinding refuses to proceed
#
LEAK_PATTERNS = [
    r"plateau", r"barren", r"noise", r"shot", r"healthy",
    r"broken", r"label", r"planted", r"ground.?truth", r"fail",
    r"starv",
]

# canonical failure-mode labels for the validation study
#
VALID_LABELS = [
    "barren_plateau", "shot_starved", "noise_dominated", "healthy",
]

# files that must exist for a directory to be a HilbertBench run
#
TRACE_FILES = ("trace.json", "events.jsonl", "catalog.json")

#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------

def _canonical_json(obj: Any) -> str:
    """
    function: _canonical_json

    arguments:
     obj: any JSON-serialisable object

    return:
     a canonical (sorted-key, minimal-separator) JSON string

    description:
     Canonical serialisation so that the SHA-256 commitment of the
     answer key is reproducible byte-for-byte.
    """

    # exit gracefully
    #
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))
#
# end of function


def _sha256_hex(text: str) -> str:
    """
    function: _sha256_hex

    arguments:
     text: the string to hash

    return:
     the hex SHA-256 digest of the UTF-8 encoding of text
    """

    # exit gracefully
    #
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
#
# end of function


def _is_run_dir(path: Path) -> bool:
    """
    function: _is_run_dir

    arguments:
     path: a candidate directory

    return:
     True if the directory looks like a sealed HilbertBench run
    """

    # exit gracefully
    #
    return all((path / f).is_file() for f in TRACE_FILES)
#
# end of function


def audit_run(run_dir: Path) -> list:
    """
    function: audit_run

    arguments:
     run_dir: a HilbertBench run directory

    return:
     a list of human-readable leakage findings (empty when clean)

    description:
     Scans the trace tags and the directory name for label-like
     strings. Traces are sealed, so leaked tags cannot be stripped —
     a leaky run must be regenerated with neutral tags.
    """

    # compile the denylist once
    #
    leak_re = re.compile("|".join(LEAK_PATTERNS), re.IGNORECASE)
    findings = []

    # the original directory name may itself encode the label
    #
    if leak_re.search(run_dir.name):
        findings.append(
            f"directory name '{run_dir.name}' matches a label pattern"
        )

    # scan trace.json tags (keys and values)
    #
    trace_meta = json.loads((run_dir / "trace.json").read_text())
    for key, value in (trace_meta.get("tags") or {}).items():
        if leak_re.search(str(key)) or leak_re.search(str(value)):
            findings.append(
                f"tag '{key}: {value}' matches a label pattern"
            )

    # exit gracefully
    #
    return findings
#
# end of function


def cmd_blind(args: argparse.Namespace) -> int:
    """
    function: cmd_blind

    arguments:
     args: parsed CLI arguments (manifest, out, force)

    return:
     process exit code (0 on success)

    description:
     Reads the corpus manifest ({run_path: {"label": ...}}), audits
     every run for leakage, copies each run verbatim under a random
     8-hex-character blind ID, and writes three files next to the
     blinded corpus:
      answer_key.json    private — held by Researcher A only
      answer_key.sha256  public commitment, published before diagnosis
      diagnosis_sheet.json  empty template for Researcher B
    """

    # load and validate the manifest
    #
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    base = manifest_path.parent

    runs = []
    for rel_path, truth in sorted(manifest.items()):
        run_dir = (base / rel_path).resolve()
        if not _is_run_dir(run_dir):
            print(f"error: not a sealed run directory: {run_dir}")
            return 1
        label = truth.get("label")
        if label not in VALID_LABELS:
            print(
                f"error: run '{rel_path}' has label '{label}'; "
                f"expected one of {VALID_LABELS}"
            )
            return 1
        runs.append((rel_path, run_dir, truth))

    # refuse to blind a corpus that leaks ground truth
    #
    leaky = False
    for rel_path, run_dir, _ in runs:
        for finding in audit_run(run_dir):
            print(f"LEAK  {rel_path}: {finding}")
            leaky = True
    if leaky and not args.allow_leaky:
        print(
            "error: ground truth leaks into trace metadata; regenerate "
            "the affected runs with neutral tags (traces are sealed "
            "and cannot be edited)"
        )
        return 1

    # copy each run under a fresh random blind ID
    #
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    salt = secrets.token_hex(16)
    key_entries = {}
    for rel_path, run_dir, truth in runs:
        blind_id = secrets.token_hex(4)
        while blind_id in key_entries or (out_dir / blind_id).exists():
            blind_id = secrets.token_hex(4)
        shutil.copytree(run_dir, out_dir / blind_id)
        key_entries[blind_id] = dict(truth, original_path=rel_path)
        print(f"  {blind_id}  <-  {rel_path}")

    # write the private answer key and its public commitment; the salt
    # prevents brute-forcing the key from the commitment alone
    #
    answer_key = {
        "protocol": "hilbertbench-blind-corpus-v1",
        "salt": salt,
        "corpus": key_entries,
    }
    key_text = _canonical_json(answer_key)
    key_path = out_dir / "answer_key.json"
    key_path.write_text(key_text)
    (out_dir / "answer_key.sha256").write_text(
        _sha256_hex(key_text) + "\n"
    )

    # write the empty diagnosis sheet for Researcher B
    #
    sheet = {
        blind_id: {"label": None, "confidence": None, "notes": ""}
        for blind_id in sorted(key_entries)
    }
    (out_dir / "diagnosis_sheet.json").write_text(
        json.dumps(sheet, indent=2) + "\n"
    )

    # print the protocol instructions
    #
    print(f"\nblinded {len(key_entries)} runs into {out_dir}/")
    print(f"commitment: {_sha256_hex(key_text)}")
    print("protocol:")
    print("  1. publish answer_key.sha256 (commit/OSF) NOW")
    print("  2. move answer_key.json OUT of the corpus dir; only")
    print("     Researcher A keeps it")
    print("  3. give the blinded dirs + diagnosis_sheet.json to")
    print("     Researcher B")
    print("  4. score with: blind_corpus.py score ...")

    # exit gracefully
    #
    return 0
#
# end of function


def cmd_audit(args: argparse.Namespace) -> int:
    """
    function: cmd_audit

    arguments:
     args: parsed CLI arguments (corpus)

    return:
     process exit code (0 when clean, 1 when leaky)

    description:
     Runs the leakage audit over every run directory found directly
     under the corpus directory.
    """

    # scan each run directory under the corpus root
    #
    corpus = Path(args.corpus)
    total, leaky = 0, 0
    for child in sorted(corpus.iterdir()):
        if not child.is_dir() or not _is_run_dir(child):
            continue
        total += 1
        findings = audit_run(child)
        if findings:
            leaky += 1
            for finding in findings:
                print(f"LEAK  {child.name}: {finding}")

    # report and exit gracefully
    #
    print(f"audited {total} runs: {leaky} leaky, {total - leaky} clean")
    return 1 if leaky else 0
#
# end of function


def _wilson_interval(successes: int, n: int) -> tuple:
    """
    function: _wilson_interval

    arguments:
     successes: number of correct diagnoses
     n:         total number of diagnoses

    return:
     (low, high) 95% Wilson score interval for the accuracy

    description:
     Wilson interval is well-behaved for the small corpus sizes used
     in the validation study (n ~ 30-40), unlike the normal interval.
    """

    # guard the empty case
    #
    if n == 0:
        return (0.0, 0.0)

    # standard Wilson score computation at z = 1.96
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


def cmd_score(args: argparse.Namespace) -> int:
    """
    function: cmd_score

    arguments:
     args: parsed CLI arguments (key, commitment, diagnosis, out)

    return:
     process exit code (0 on success)

    description:
     Verifies the answer key against its published commitment, joins
     it with the filled diagnosis sheet, and reports accuracy with a
     95% Wilson interval, a confusion matrix, and per-label
     precision/recall/F1. Optionally writes the full result as JSON
     for figure generation.
    """

    # verify the answer key against the published commitment
    #
    key_text = Path(args.key).read_text()
    expected = Path(args.commitment).read_text().strip()
    actual = _sha256_hex(key_text)
    if actual != expected:
        print("error: answer key does NOT match the commitment hash")
        print(f"  committed: {expected}")
        print(f"  computed:  {actual}")
        return 1
    print(f"commitment verified: {actual}")

    # join the key with the diagnosis sheet
    #
    key = json.loads(key_text)["corpus"]
    sheet = json.loads(Path(args.diagnosis).read_text())
    missing = sorted(set(key) - set(sheet))
    if missing:
        print(f"error: diagnosis sheet missing runs: {missing}")
        return 1

    # build the confusion matrix (truth -> predicted -> count)
    #
    labels = sorted(
        set(VALID_LABELS)
        | {entry["label"] for entry in key.values()}
    )
    predicted_labels = set()
    confusion: dict = {t: {} for t in labels}
    correct = 0
    for blind_id, truth in key.items():
        true_label = truth["label"]
        pred = sheet[blind_id].get("label") or "undiagnosed"
        predicted_labels.add(pred)
        row = confusion[true_label]
        row[pred] = row.get(pred, 0) + 1
        if pred == true_label:
            correct += 1

    # per-label precision / recall / F1
    #
    per_label = {}
    for label in labels:
        tp = confusion[label].get(label, 0)
        fn = sum(confusion[label].values()) - tp
        fp = sum(
            row.get(label, 0)
            for truth, row in confusion.items()
            if truth != label
        )
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision and recall else None
        )
        per_label[label] = {
            "precision": precision, "recall": recall, "f1": f1,
            "support": tp + fn,
        }

    # overall accuracy with Wilson interval
    #
    n = len(key)
    accuracy = correct / n if n else 0.0
    ci_low, ci_high = _wilson_interval(correct, n)

    # print the report
    #
    print(f"\nruns scored: {n}")
    print(
        f"accuracy: {accuracy:.3f} "
        f"(95% Wilson CI [{ci_low:.3f}, {ci_high:.3f}])"
    )
    print("\nper-label metrics:")
    for label, metrics in per_label.items():
        if metrics["support"] == 0:
            continue
        fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
        print(
            f"  {label:<18} P={fmt(metrics['precision'])} "
            f"R={fmt(metrics['recall'])} F1={fmt(metrics['f1'])} "
            f"(n={metrics['support']})"
        )
    print("\nconfusion matrix (rows=truth, cols=predicted):")
    cols = sorted(predicted_labels)
    header = " " * 20 + "  ".join(f"{c[:12]:>12}" for c in cols)
    print(header)
    for truth_label in labels:
        row = confusion[truth_label]
        if not row:
            continue
        cells = "  ".join(
            f"{row.get(c, 0):>12}" for c in cols
        )
        print(f"{truth_label:<20}{cells}")

    # optionally write the machine-readable result
    #
    if args.out:
        result = {
            "n": n,
            "accuracy": accuracy,
            "accuracy_ci95": [ci_low, ci_high],
            "per_label": per_label,
            "confusion": confusion,
        }
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nwrote {args.out}")

    # exit gracefully
    #
    return 0
#
# end of function


def main(argv: Optional[list] = None) -> int:
    """
    function: main

    arguments:
     argv: CLI arguments (defaults to sys.argv[1:])

    return:
     process exit code

    description:
     Parses the subcommand and dispatches to blind / audit / score.
    """

    # build the CLI
    #
    parser = argparse.ArgumentParser(
        prog="blind_corpus.py",
        description="blinded-corpus protocol tool for HilbertBench",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_blind = sub.add_parser("blind", help="anonymise a corpus")
    p_blind.add_argument(
        "--manifest", required=True,
        help="JSON manifest: {run_path: {'label': ...}, ...}",
    )
    p_blind.add_argument(
        "--out", required=True, help="output directory for blinded runs",
    )
    p_blind.add_argument(
        "--allow-leaky", action="store_true",
        help="proceed despite leakage findings (NOT for the real study)",
    )
    p_blind.set_defaults(func=cmd_blind)

    p_audit = sub.add_parser("audit", help="scan runs for label leakage")
    p_audit.add_argument(
        "--corpus", required=True,
        help="directory containing run directories",
    )
    p_audit.set_defaults(func=cmd_audit)

    p_score = sub.add_parser("score", help="score a filled diagnosis")
    p_score.add_argument("--key", required=True, help="answer_key.json")
    p_score.add_argument(
        "--commitment", required=True, help="answer_key.sha256",
    )
    p_score.add_argument(
        "--diagnosis", required=True, help="filled diagnosis sheet",
    )
    p_score.add_argument(
        "--out", default=None, help="write JSON result here",
    )
    p_score.set_defaults(func=cmd_score)

    # dispatch and exit gracefully
    #
    args = parser.parse_args(argv)
    return args.func(args)
#
# end of function


# begin gracefully
#
if __name__ == "__main__":
    sys.exit(main())
#
# end of file
