"""
hilbertbench/reader/verify.py

Consumes a HilbertBench trace directory and mathematically proves its integrity.
Enforces INV-002 (Trace Immutability) and INV-005 (Causal Ordering).
"""

import hashlib
from pathlib import Path
from typing import Set

from hilbertbench.models import (
    HilbertbenchTraceManifest,
    HilbertbenchSpan,
    HilbertbenchArtifactCatalog,
)


class TraceValidationError(Exception):
    """Raised when a trace fails cryptographic or causal verification."""
    pass


def verify_trace_directory(run_dir: Path | str) -> bool:
    """
    Cryptographically and logically verifies a HilbertBench trace.
    Returns True if the trace is perfectly intact.
    Raises TraceValidationError if any tampering or corruption is detected.
    """
    run_path = Path(run_dir)
    if not run_path.is_dir():
        raise TraceValidationError(f"Run directory not found: {run_path}")

    # 1. Parse and validate Schema Integrity (INV-003)
    trace_path = run_path / "trace.json"
    if not trace_path.exists():
        raise TraceValidationError("Missing trace.json")
    
    # This automatically validates against the Pydantic schema
    try:
        manifest = HilbertbenchTraceManifest.model_validate_json(trace_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise TraceValidationError(f"trace.json schema violation: {e}")

    catalog_path = run_path / "catalog.json"
    if not catalog_path.exists():
        raise TraceValidationError("Missing catalog.json")
    
    try:
        catalog = HilbertbenchArtifactCatalog.model_validate_json(catalog_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise TraceValidationError(f"catalog.json schema violation: {e}")

    # 2. Cryptographic Verification (INV-002)
    _verify_artifacts(run_path, catalog)

    # 3. Causal Sequence Verification (INV-005)
    events_path = run_path / "events.jsonl"
    if not events_path.exists():
        raise TraceValidationError("Missing events.jsonl")

    _verify_causal_spans(events_path, catalog)

    return True


def _verify_artifacts(run_path: Path, catalog: HilbertbenchArtifactCatalog) -> None:
    """Re-hashes all physical files and compares against the catalog source-of-truth."""
    for expected_hash, meta in catalog.artifacts.items():
        physical_path = run_path / meta.file_path
        
        if not physical_path.exists():
            raise TraceValidationError(f"Artifact physically missing from disk: {meta.file_path}")

        # Stream the file to handle potentially massive NumPy/Parquet arrays
        hasher = hashlib.sha256()
        with open(physical_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096 * 1024), b""): # 4MB chunks
                hasher.update(chunk)
        
        actual_hash = f"sha256:{hasher.hexdigest()}"
        
        if actual_hash != expected_hash:
            raise TraceValidationError(
                f"Cryptographic tampering detected! "
                f"Catalog expected {expected_hash}, but physical file yielded {actual_hash} "
                f"for artifact {meta.file_path}"
            )


def _verify_causal_spans(events_path: Path, catalog: HilbertbenchArtifactCatalog) -> None:
    """Reads events.jsonl to ensure valid references and parent/child logic."""
    all_spans = {}
    seen_seqs = set()

    # --- PASS 1: Parsing and References ---
    with open(events_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            
            try:
                span = HilbertbenchSpan.model_validate_json(line)
            except Exception as e:
                raise TraceValidationError(f"Span schema violation on line {line_num}: {e}")
            
            # A. Check Sequence Uniqueness (Replaces strict monotonicity)
            if span.sequence_number in seen_seqs:
                raise TraceValidationError(
                    f"Causal violation: Duplicate span sequence number {span.sequence_number}."
                )
            seen_seqs.add(span.sequence_number)

            # B. Check Artifact References
            if span.payload_ref and span.payload_ref not in catalog.artifacts:
                raise TraceValidationError(
                    f"Dangling reference: Span payload {span.payload_ref} not found in catalog."
                )
            if span.outcome_ref and span.outcome_ref not in catalog.artifacts:
                raise TraceValidationError(
                    f"Dangling reference: Span outcome {span.outcome_ref} not found in catalog."
                )

            all_spans[str(span.span_id)] = span

    # --- PASS 2: Causal Parent-Child Logic ---
    for span_id, span in all_spans.items():
        if span.parent_span_id:
            parent_id_str = str(span.parent_span_id)
            
            # 1. Parent must exist in the trace
            if parent_id_str not in all_spans:
                raise TraceValidationError(
                    f"Causal violation: Child span {span.span_id} references a parent "
                    f"{parent_id_str} that does not exist in the trace."
                )
            
            # 2. Chronological sanity: Child cannot start before parent
            parent_span = all_spans[parent_id_str]
            if span.timestamp_start < parent_span.timestamp_start:
                raise TraceValidationError(
                    f"Causal violation: Child span {span.span_id} started before "
                    f"its parent {parent_id_str}."
                )

# reader/verify.py
import re
from hilbertbench.models import HilbertbenchArtifactCatalog

SHA256_PATTERN = re.compile(r"^sha256:[a-f0-9]{64}$")

def verify_catalog(catalog: HilbertbenchArtifactCatalog) -> list[str]:
    """
    Returns list of integrity violations. Empty list = clean.
    Called by loader before returning any trace to callers.
    """
    violations = []
    for key, artifact in catalog.artifacts.items():
        # Check 1: key format
        if not SHA256_PATTERN.match(key):
            violations.append(
                f"Artifact key '{key}' is not a valid sha256:<hash> identifier"
            )
        # Check 2: key == artifact_hash (content-addressed integrity)
        elif key != artifact.artifact_hash:
            violations.append(
                f"Catalog key '{key}' does not match artifact_hash "
                f"'{artifact.artifact_hash}' — catalog is corrupted"
            )
    return violations
