"""
tests/reader/test_verify.py

Proves the cryptographic and causal verification engine.
Guarantees that tampered data, missing files, or out-of-order execution
spans are strictly rejected.
"""
import json
from pathlib import Path

import pytest

from hilbertbench.models import Kind, Encoding
from hilbertbench.reader.verify import verify_trace_directory, TraceValidationError
from hilbertbench.recorder.tape import HilbertTape


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_run(tmp_path: Path) -> Path:
    """Generates a mathematically perfect, sealed trace directory."""
    run_root = tmp_path / "valid_run"
    run_root.mkdir()
    
    # Create a dummy quantum circuit file
    dummy_circuit = tmp_path / "circuit.qasm"
    dummy_circuit.write_text("OPENQASM 3.0;\nqubit[2] q;\nh q[0];\ncx q[0], q[1];")

    with HilbertTape(run_root) as tape:
        ref = tape.attach_artifact(dummy_circuit, kind=Kind.circuit_qasm, encoding=Encoding.openqasm)
        
        # Open a root span
        with tape.execution_span(payload_ref=ref) as root_handle:
            # Open a nested child span
            with tape.execution_span(payload_ref=ref):
                pass
            
    return tape.dir_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_verify_valid_trace_passes(valid_run: Path):
    """A perfectly clean trace should pass with True."""
    assert verify_trace_directory(valid_run) is True


def test_verify_detects_tampered_artifact(valid_run: Path):
    """
    Simulates a malicious user altering their result file after the run 
    to make their quantum benchmark look better.
    """
    artifact_dir = valid_run / "artifacts"

    # Artifacts use 2-char sharding: artifacts/<shard>/<hash>.<ext>
    # iterdir() returns shard *directories*, so we must rglob for the actual file.
    artifact_file = next(f for f in artifact_dir.rglob("*") if f.is_file())

    # Tamper with the physical file!
    artifact_file.write_text("OPENQASM 3.0;\nqubit[2] q;\n// I CHEATED AND REMOVED THE NOISE")
    
    with pytest.raises(TraceValidationError, match="Cryptographic tampering detected"):
        verify_trace_directory(valid_run)


def test_verify_detects_missing_events_file(valid_run: Path):
    """If events.jsonl is deleted, the trace is invalid."""
    (valid_run / "events.jsonl").unlink()

    with pytest.raises(TraceValidationError, match="Missing events.jsonl"):
        verify_trace_directory(valid_run)


def test_integrity_seal_present_and_valid(valid_run: Path):
    """A sealed trace carries an integrity_seal that matches events.jsonl."""
    trace = json.loads((valid_run / "trace.json").read_text())
    seal = trace["integrity_seal"]
    assert seal is not None
    assert seal["event_stream_checksum"].startswith("sha256:")
    assert seal["artifact_count"] >= 1
    # Clean trace verifies (seal check is exercised inside verify_trace_directory)
    assert verify_trace_directory(valid_run) is True


def test_verify_detects_event_stream_tampering(valid_run: Path):
    """
    Modifying events.jsonl in a way that still passes causal/reference checks
    (e.g. flipping a backend_id that no check inspects) must still be caught by
    the integrity seal's byte-level checksum.
    """
    events_file = valid_run / "events.jsonl"
    lines = events_file.read_text().splitlines()

    span = json.loads(lines[0])
    span["backend_id"] = "FAKED_BACKEND"  # causally valid, but changes the bytes
    lines[0] = json.dumps(span)
    events_file.write_text("\n".join(lines) + "\n")

    with pytest.raises(TraceValidationError, match="checksum mismatch"):
        verify_trace_directory(valid_run)


def test_verify_detects_causal_sequence_violation(valid_run: Path):
    """
    Simulates a logging error where sequence numbers are duplicated,
    or a user copy-pasting spans to fake execution data.
    """
    events_file = valid_run / "events.jsonl"
    lines = events_file.read_text().splitlines()
    
    # Force the first span to appear twice in a row
    events_file.write_text(lines[0] + "\n" + lines[0])
    
    with pytest.raises(TraceValidationError, match="Duplicate span sequence number"):
        verify_trace_directory(valid_run)


def test_verify_detects_dangling_artifact_references(valid_run: Path):
    """
    Simulates a span pointing to an artifact hash that doesn't exist in the catalog.
    """
    events_file = valid_run / "events.jsonl"
    lines = events_file.read_text().splitlines()
    
    # Load the first span and corrupt its payload reference
    span = json.loads(lines[0])
    span["payload_ref"] = "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    
    lines[0] = json.dumps(span)
    events_file.write_text("\n".join(lines))
    
    with pytest.raises(TraceValidationError, match="Dangling reference: Span payload"):
        verify_trace_directory(valid_run)


def test_verify_detects_child_before_parent_violation(valid_run: Path):
    """
    A child span cannot legally finish and flush to the logs BEFORE its 
    parent span has been created. Causal arrows flow one way.
    """
    events_file = valid_run / "events.jsonl"
    lines = events_file.read_text().splitlines()
    
    # lines[0] is the inner/child span (because it finished first)
    # lines[1] is the outer/parent span (because it finished second)
    # Let's delete the parent span entirely, meaning the child refers to a ghost parent.
    events_file.write_text(lines[0])
    
    with pytest.raises(TraceValidationError, match="Causal violation: Child span"):
        verify_trace_directory(valid_run)