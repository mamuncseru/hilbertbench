#!/usr/bin/env python
#
# file: hilbertbench/reader/verify.py
#
# revision history:
#  20260604 (am): cleaned up to project coding standards
#
# Consumes a HilbertBench trace directory and mathematically proves its
# integrity. Enforces INV-002 (Trace Immutability) and INV-005
# (Causal Ordering). CLI entry point: hb-verify.
#------------------------------------------------------------------------------

# import system modules
#
import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Set

# import hilbertbench modules
#
from hilbertbench.models import (
    HilbertbenchTraceManifest,
    HilbertbenchSpan,
    HilbertbenchArtifactCatalog,
)

#------------------------------------------------------------------------------
#
# global variables are listed here
#
#------------------------------------------------------------------------------

# set the filename using basename
#
__FILE__ = os.path.basename(__file__)

# pre-compiled pattern for validating sha256 artifact keys
#
SHA256_PATTERN = re.compile(r"^sha256:[a-f0-9]{64}$")

#------------------------------------------------------------------------------
#
# classes are listed here
#
#------------------------------------------------------------------------------

class TraceValidationError(Exception):
    """
    Class: TraceValidationError

    description:
     Raised when a trace fails cryptographic or causal verification.
    """
    pass
#
# end of class

#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------

def verify_trace_directory(run_dir: Path | str) -> bool:
    """
    function: verify_trace_directory

    arguments:
     run_dir: path to the trace run directory to verify

    return:
     True if the trace is perfectly intact

    description:
     Cryptographically and logically verifies a HilbertBench trace.
     Raises TraceValidationError if any tampering or corruption is
     detected. Enforces INV-002 (Trace Immutability) and INV-005
     (Causal Ordering).
    """

    # resolve the run directory path
    #
    run_path = Path(run_dir)
    if not run_path.is_dir():
        raise TraceValidationError(
            f"Run directory not found: {run_path}"
        )

    # step 1: parse and validate schema integrity (INV-003)
    #
    trace_path = run_path / "trace.json"
    if not trace_path.exists():
        raise TraceValidationError("Missing trace.json")

    try:
        manifest = HilbertbenchTraceManifest.model_validate_json(
            trace_path.read_text(encoding="utf-8")
        )
    except Exception as e:
        raise TraceValidationError(f"trace.json schema violation: {e}")

    catalog_path = run_path / "catalog.json"
    if not catalog_path.exists():
        raise TraceValidationError("Missing catalog.json")

    try:
        catalog = HilbertbenchArtifactCatalog.model_validate_json(
            catalog_path.read_text(encoding="utf-8")
        )
    except Exception as e:
        raise TraceValidationError(f"catalog.json schema violation: {e}")

    # step 2: cryptographic artifact verification (INV-002)
    #
    _verify_artifacts(run_path, catalog)

    # step 3: causal sequence verification (INV-005)
    #
    events_path = run_path / "events.jsonl"
    if not events_path.exists():
        raise TraceValidationError("Missing events.jsonl")

    _verify_causal_spans(events_path, catalog)

    # step 4: integrity seal verification — tamper detection on the event
    # stream; only enforced for sealed traces that carry a seal (INV-002)
    #
    if manifest.integrity_seal is not None:
        _verify_integrity_seal(events_path, manifest.integrity_seal)

    # exit gracefully
    #
    return True
#
# end of function


def _verify_integrity_seal(events_path: Path, seal) -> None:
    """
    function: _verify_integrity_seal

    arguments:
     events_path: path to the events.jsonl file
     seal:        the IntegritySeal embedded in the trace manifest

    return:
     none

    description:
     Re-hashes events.jsonl in 1 MB chunks and compares the result
     against the sealed checksum. Raises TraceValidationError if the
     checksums do not match.
    """

    # hash the event stream in 1 MB chunks
    #
    hasher = hashlib.sha256()
    with open(events_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    actual = f"sha256:{hasher.hexdigest()}"

    # compare computed hash against the sealed checksum
    #
    if actual != seal.event_stream_checksum:
        raise TraceValidationError(
            "Cryptographic tampering detected! "
            "Event stream checksum mismatch: "
            f"seal expected {seal.event_stream_checksum}, "
            f"but events.jsonl yielded {actual}"
        )
#
# end of function


def _verify_artifacts(
    run_path: Path,
    catalog: HilbertbenchArtifactCatalog,
) -> None:
    """
    function: _verify_artifacts

    arguments:
     run_path: path to the trace run directory
     catalog:  the parsed artifact catalog

    return:
     none

    description:
     Re-hashes all physical artifact files and compares against the
     catalog source-of-truth. Files are streamed in 4 MB chunks to
     handle large NumPy/Parquet arrays without loading into memory.
     Raises TraceValidationError on any mismatch or missing file.
    """

    # iterate over every registered artifact
    #
    for expected_hash, meta in catalog.artifacts.items():

        # verify the physical file exists on disk
        #
        physical_path = run_path / meta.file_path
        if not physical_path.exists():
            raise TraceValidationError(
                "Artifact physically missing from disk: "
                f"{meta.file_path}"
            )

        # hash the file in 4 MB chunks
        #
        hasher = hashlib.sha256()
        with open(physical_path, "rb") as f:
            for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
                hasher.update(chunk)
        actual_hash = f"sha256:{hasher.hexdigest()}"

        # compare computed hash against the catalog entry
        #
        if actual_hash != expected_hash:
            raise TraceValidationError(
                "Cryptographic tampering detected! "
                f"Catalog expected {expected_hash}, but physical file "
                f"yielded {actual_hash} for artifact {meta.file_path}"
            )
#
# end of function


def _verify_causal_spans(
    events_path: Path,
    catalog: HilbertbenchArtifactCatalog,
) -> None:
    """
    function: _verify_causal_spans

    arguments:
     events_path: path to the events.jsonl file
     catalog:     the parsed artifact catalog

    return:
     none

    description:
     Reads events.jsonl in two passes to verify causal correctness.
     Pass 1 validates schema, sequence uniqueness, and artifact refs.
     Pass 2 validates parent-child chronological ordering (INV-005).
     Raises TraceValidationError on any violation.
    """

    all_spans = {}
    seen_seqs: Set[int] = set()

    # pass 1: schema validation, sequence uniqueness, artifact references
    #
    with open(events_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):

            # skip blank lines
            #
            line = line.strip()
            if not line:
                continue

            # parse and schema-validate the span
            #
            try:
                span = HilbertbenchSpan.model_validate_json(line)
            except Exception as e:
                raise TraceValidationError(
                    f"Span schema violation on line {line_num}: {e}"
                )

            # check sequence number uniqueness
            #
            if span.sequence_number in seen_seqs:
                raise TraceValidationError(
                    "Causal violation: Duplicate span sequence number "
                    f"{span.sequence_number}."
                )
            seen_seqs.add(span.sequence_number)

            # gather inline artifact keys; refs resolve against the
            # catalog (file-store) OR the span's inline_artifacts store
            #
            inline_keys: Set[str] = (
                set(span.inline_artifacts.keys())
                if span.inline_artifacts
                else set()
            )

            # verify payload reference resolves
            #
            if span.payload_ref:
                if (
                    span.payload_ref not in catalog.artifacts
                    and span.payload_ref not in inline_keys
                ):
                    raise TraceValidationError(
                        f"Dangling reference: Span payload "
                        f"{span.payload_ref} not found in catalog "
                        f"or inline_artifacts."
                    )

            # verify outcome reference resolves
            #
            if span.outcome_ref:
                if (
                    span.outcome_ref not in catalog.artifacts
                    and span.outcome_ref not in inline_keys
                ):
                    raise TraceValidationError(
                        f"Dangling reference: Span outcome "
                        f"{span.outcome_ref} not found in catalog "
                        f"or inline_artifacts."
                    )

            all_spans[str(span.span_id)] = span

    # pass 2: parent-child causal ordering
    #
    for span_id, span in all_spans.items():
        if not span.parent_span_id:
            continue

        parent_id_str = str(span.parent_span_id)

        # parent must exist in the trace
        #
        if parent_id_str not in all_spans:
            raise TraceValidationError(
                f"Causal violation: Child span {span.span_id} "
                f"references parent {parent_id_str} that does not "
                f"exist in the trace."
            )

        # child must not start before its parent (chronological sanity)
        #
        parent_span = all_spans[parent_id_str]
        if span.timestamp_start < parent_span.timestamp_start:
            raise TraceValidationError(
                f"Causal violation: Child span {span.span_id} "
                f"started before its parent {parent_id_str}."
            )
#
# end of function


def verify_catalog(catalog: HilbertbenchArtifactCatalog) -> list[str]:
    """
    function: verify_catalog

    arguments:
     catalog: the parsed HilbertbenchArtifactCatalog to check

    return:
     list of violation strings; empty list means clean

    description:
     Validates catalog key format and content-address consistency.
     Called by the loader before returning any trace to callers.
    """

    violations = []

    # check each artifact key and its stored hash
    #
    for key, artifact in catalog.artifacts.items():

        # check key format — must be sha256:<64 lowercase hex chars>
        #
        if not SHA256_PATTERN.match(key):
            violations.append(
                f"Artifact key '{key}' is not a valid "
                f"sha256:<hash> identifier"
            )

        # check that the catalog key matches the artifact_hash field
        #
        elif key != artifact.artifact_hash:
            violations.append(
                f"Catalog key '{key}' does not match artifact_hash "
                f"'{artifact.artifact_hash}' — catalog is corrupted"
            )

    # exit gracefully
    #
    return violations
#
# end of function


def main() -> None:
    """
    function: main

    arguments:
     none (reads sys.argv)

    return:
     none

    description:
     CLI entry point for hb-verify. Accepts one run directory as an
     argument and exits with code 0 on success or 1 on failure.
    """

    # check for a run directory argument
    #
    if len(sys.argv) < 2:
        print(f"usage: {os.path.basename(sys.argv[0])} <run_dir>")
        sys.exit(1)

    # verify the trace directory and report result
    #
    run_dir = sys.argv[1]
    try:
        verify_trace_directory(run_dir)
        print(f"OK: {run_dir}")
    except TraceValidationError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
#
# end of function

#
# end of file
