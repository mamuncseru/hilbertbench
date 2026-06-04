"""
Backward compatibility tests.

These fixtures are FROZEN. They represent real v1.0 traces that must
remain parseable for the lifetime of the project.
Never update these dicts — if a model change breaks them, that is
a BREAKING CHANGE and requires a schema version bump.
"""
import pytest
from hilbertbench.models import (
    HilbertbenchTraceManifest,
    HilbertbenchSpan,
    HilbertbenchArtifactMetadata,
)

# ── Frozen v1.0 golden records ────────────────────────────────────────────

GOLDEN_TRACE = {
    "hbtrace_version": "1.0",
    "trace_id": "00000000-0000-0000-0000-000000000001",
    "mode": "passive",
    "timestamp_start": 1711234567000000000,
    "timestamp_end": None,
    "status": "SEALED_SUCCESS",
    "client_environment": {"hilbertbench_version": "0.1.0"},
    "integrity_seal": None,
    "tags": {}
}

GOLDEN_SPAN = {
    "hbspan_version": "1.0",
    "span_id": "00000000-0000-0000-0000-000000000002",
    "trace_id": "00000000-0000-0000-0000-000000000001",
    "sequence_number": 0,
    "timestamp_start": 1711234567000000001,
    "payload_ref": "sha256:" + "a" * 64,
    "events": [{
        "event_id": "00000000-0000-0000-0000-000000000003",
        "event_type": "EXECUTION_REQUEST",
        "timestamp": 1711234567000000002,
        "error_ref": None,
        "attributes": None
    }]
}

GOLDEN_ARTIFACT = {
    "artifact_hash": "sha256:" + "a" * 64,
    "kind": "circuit_qasm",
    "encoding": "openqasm",
    "file_path": "artifacts/aa/" + "a" * 64 + ".qasm",
    "size_bytes": 512,
    "created_at": 1711234567000000003,
    "ref_count": 1
}


class TestGoldenRecords:
    """These must NEVER fail. Failure = breaking change."""

    def test_golden_trace_always_parses(self):
        trace = HilbertbenchTraceManifest.model_validate(GOLDEN_TRACE)
        assert str(trace.trace_id) == "00000000-0000-0000-0000-000000000001"

    def test_golden_span_always_parses(self):
        span = HilbertbenchSpan.model_validate(GOLDEN_SPAN)
        assert span.sequence_number == 0

    def test_golden_artifact_always_parses(self):
        artifact = HilbertbenchArtifactMetadata.model_validate(GOLDEN_ARTIFACT)
        assert artifact.kind.value == "circuit_qasm"
