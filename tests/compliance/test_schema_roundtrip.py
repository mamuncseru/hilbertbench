"""
tests/compliance/test_schema_roundtrip.py

Validates that all Pydantic v2 models correctly enforce schema constraints.
Tests are organized by model. Negative tests confirm bad data is rejected.

Run: pytest tests/compliance/test_schema_roundtrip.py -v
"""

import pytest
from uuid import uuid4
from hilbertbench.models import (
    HilbertbenchTraceManifest,
    HilbertbenchSpan,
    HilbertbenchArtifactMetadata,
    HilbertbenchArtifactCatalog,
    Mode,
    TraceStatus,
    SpanStatus,
    Kind,
    Encoding,
    Compression,
)

# ── Shared test constants ─────────────────────────────────────────────────────

TRACE_ID  = str(uuid4())
SPAN_ID   = str(uuid4())
HASH_KEY  = "sha256:" + "a" * 64
HASH_KEY2 = "sha256:" + "b" * 64


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def valid_trace_dict():
    return {
        "hbtrace_version": "1.0",
        "trace_id": TRACE_ID,
        "mode": "passive",
        "timestamp_start": 1711234567000000000,
        "timestamp_end": 1711234599000000000,
        "status": "SEALED_SUCCESS",
        "client_environment": {
            "hilbertbench_version": "0.1.0",
            "python_version": "3.11.0",
            "os_platform": "linux",
            "frameworks": {"qiskit": "1.0.2"},
        },
        "integrity_seal": {
            "event_stream_checksum": "b" * 64,
            "artifact_count": 3,
        },
        "tags": {"experiment": "vqe_h2"},
    }


@pytest.fixture
def valid_span_dict():
    return {
        "hbspan_version": "1.0",
        "span_id": SPAN_ID,
        "trace_id": TRACE_ID,
        "sequence_number": 0,
        "timestamp_start": 1711234567000000001,
        "status": "COMPLETED",
        "backend_id": "ibm_kyiv",
        "payload_ref": HASH_KEY,
        "outcome_ref": HASH_KEY,
        "parent_span_id": None,
        "events": [
            {
                "event_id": str(uuid4()),
                "event_type": "EXECUTION_REQUEST",
                "timestamp": 1711234567000000002,
                "error_ref": None,
                "attributes": {"shots": 1024, "queue_position": 3},
            },
            {
                "event_id": str(uuid4()),
                "event_type": "EXECUTION_RESULT",
                "timestamp": 1711234568000000000,
                "error_ref": None,
                "attributes": {"backend_execution_ms": 312.4},
            },
        ],
        "tags": None,
    }


@pytest.fixture
def valid_artifact_dict():
    return {
        "artifact_hash": HASH_KEY,
        "kind": "circuit_qasm",
        "encoding": "openqasm",
        "file_path": f"artifacts/aa/{'a' * 64}.qasm",
        "size_bytes": 2048,
        "compression": None,
        "created_at": 1711234567000000003,
        "producer": "qiskit=1.0.2",
        "ref_count": 1,
    }


@pytest.fixture
def valid_catalog_dict(valid_artifact_dict):
    return {
        "hbcatalog_version": "1.0",
        "trace_id": TRACE_ID,
        "created_at": 1711234599000000000,
        "artifacts": {
            HASH_KEY: valid_artifact_dict,
        },
    }


# ── TestTraceManifest ─────────────────────────────────────────────────────────

class TestTraceManifest:

    def test_valid_construction(self, valid_trace_dict):
        trace = HilbertbenchTraceManifest.model_validate(valid_trace_dict)
        assert str(trace.trace_id) == TRACE_ID

    def test_roundtrip(self, valid_trace_dict):
        trace  = HilbertbenchTraceManifest.model_validate(valid_trace_dict)
        trace2 = HilbertbenchTraceManifest.model_validate(trace.model_dump())
        assert trace == trace2

    def test_mode_enum(self, valid_trace_dict):
        trace = HilbertbenchTraceManifest.model_validate(valid_trace_dict)
        assert trace.mode == Mode.passive

    def test_status_enum(self, valid_trace_dict):
        trace = HilbertbenchTraceManifest.model_validate(valid_trace_dict)
        assert trace.status == TraceStatus.SEALED_SUCCESS

    def test_all_status_values_valid(self, valid_trace_dict):
        for status in ["SEALED_SUCCESS", "SEALED_WITH_ERRORS",
                       "CRASHED_IN_FLIGHT", "INITIALIZATION_FAILED"]:
            valid_trace_dict["status"] = status
            trace = HilbertbenchTraceManifest.model_validate(valid_trace_dict)
            assert trace.status == TraceStatus(status)

    def test_all_mode_values_valid(self, valid_trace_dict):
        for mode in ["passive", "active"]:
            valid_trace_dict["mode"] = mode
            trace = HilbertbenchTraceManifest.model_validate(valid_trace_dict)
            assert trace.mode == Mode(mode)

    def test_null_timestamp_end_allowed(self, valid_trace_dict):
        valid_trace_dict["timestamp_end"] = None
        trace = HilbertbenchTraceManifest.model_validate(valid_trace_dict)
        assert trace.timestamp_end is None

    def test_null_integrity_seal_allowed(self, valid_trace_dict):
        valid_trace_dict["integrity_seal"] = None
        trace = HilbertbenchTraceManifest.model_validate(valid_trace_dict)
        assert trace.integrity_seal is None

    def test_optional_fields_absent(self, valid_trace_dict):
        # timestamp_end, integrity_seal, tags are all optional
        del valid_trace_dict["timestamp_end"]
        del valid_trace_dict["integrity_seal"]
        del valid_trace_dict["tags"]
        trace = HilbertbenchTraceManifest.model_validate(valid_trace_dict)
        assert str(trace.trace_id) == TRACE_ID

    def test_rejects_wrong_version(self, valid_trace_dict):
        valid_trace_dict["hbtrace_version"] = "2.0"
        with pytest.raises(Exception):
            HilbertbenchTraceManifest.model_validate(valid_trace_dict)

    def test_rejects_extra_fields(self, valid_trace_dict):
        valid_trace_dict["unknown_field"] = "should_fail"
        with pytest.raises(Exception):
            HilbertbenchTraceManifest.model_validate(valid_trace_dict)

    def test_rejects_invalid_mode(self, valid_trace_dict):
        valid_trace_dict["mode"] = "surveillance"
        with pytest.raises(Exception):
            HilbertbenchTraceManifest.model_validate(valid_trace_dict)

    def test_rejects_invalid_status(self, valid_trace_dict):
        valid_trace_dict["status"] = "RUNNING"
        with pytest.raises(Exception):
            HilbertbenchTraceManifest.model_validate(valid_trace_dict)

    def test_requires_client_environment(self, valid_trace_dict):
        del valid_trace_dict["client_environment"]
        with pytest.raises(Exception):
            HilbertbenchTraceManifest.model_validate(valid_trace_dict)

    def test_client_environment_requires_version(self, valid_trace_dict):
        del valid_trace_dict["client_environment"]["hilbertbench_version"]
        with pytest.raises(Exception):
            HilbertbenchTraceManifest.model_validate(valid_trace_dict)

    def test_tags_arbitrary_strings(self, valid_trace_dict):
        valid_trace_dict["tags"] = {"foo": "bar", "run_id": "42", "note": "vqe"}
        trace = HilbertbenchTraceManifest.model_validate(valid_trace_dict)
        assert trace.tags["run_id"] == "42"


# ── TestSpan ──────────────────────────────────────────────────────────────────

class TestSpan:

    def test_valid_construction(self, valid_span_dict):
        span = HilbertbenchSpan.model_validate(valid_span_dict)
        assert span.sequence_number == 0

    def test_roundtrip(self, valid_span_dict):
        span  = HilbertbenchSpan.model_validate(valid_span_dict)
        span2 = HilbertbenchSpan.model_validate(span.model_dump())
        assert span == span2

    def test_events_preserved(self, valid_span_dict):
        span = HilbertbenchSpan.model_validate(valid_span_dict)
        assert len(span.events) == 2
        assert span.events[0].event_type == "EXECUTION_REQUEST"
        assert span.events[1].event_type == "EXECUTION_RESULT"

    def test_trace_id_matches_parent(self, valid_span_dict):
        span = HilbertbenchSpan.model_validate(valid_span_dict)
        assert str(span.trace_id) == TRACE_ID

    def test_sequence_number_zero_allowed(self, valid_span_dict):
        valid_span_dict["sequence_number"] = 0
        span = HilbertbenchSpan.model_validate(valid_span_dict)
        assert span.sequence_number == 0

    def test_sequence_number_large_value(self, valid_span_dict):
        valid_span_dict["sequence_number"] = 999999
        span = HilbertbenchSpan.model_validate(valid_span_dict)
        assert span.sequence_number == 999999

    def test_all_status_values_valid(self, valid_span_dict):
        for status in ["COMPLETED", "FAILED", "PARTIAL", "IN_FLIGHT"]:
            valid_span_dict["status"] = status
            span = HilbertbenchSpan.model_validate(valid_span_dict)
            assert span.status == SpanStatus(status)

    def test_null_outcome_ref_allowed(self, valid_span_dict):
        valid_span_dict["outcome_ref"] = None
        span = HilbertbenchSpan.model_validate(valid_span_dict)
        assert span.outcome_ref is None

    def test_null_parent_span_id_allowed(self, valid_span_dict):
        # Null parent_span_id = root span
        valid_span_dict["parent_span_id"] = None
        span = HilbertbenchSpan.model_validate(valid_span_dict)
        assert span.parent_span_id is None

    def test_event_type_open_pattern(self, valid_span_dict):
        # event_type is open pattern ^[A-Z_]+$ — custom types must be allowed
        valid_span_dict["events"][0]["event_type"] = "PENNYLANE_GRADIENT_STEP"
        span = HilbertbenchSpan.model_validate(valid_span_dict)
        assert span.events[0].event_type == "PENNYLANE_GRADIENT_STEP"

    def test_event_attributes_allow_arbitrary_scalars(self, valid_span_dict):
        valid_span_dict["events"][0]["attributes"] = {
            "queue_pos": 7,
            "api_latency_ms": 12.3,
            "backend_name": "ibm_kyiv",
        }
        span = HilbertbenchSpan.model_validate(valid_span_dict)
        assert span.events[0].attributes["queue_pos"] == 7

    def test_event_null_attributes_allowed(self, valid_span_dict):
        valid_span_dict["events"][0]["attributes"] = None
        span = HilbertbenchSpan.model_validate(valid_span_dict)
        assert span.events[0].attributes is None

    def test_rejects_negative_sequence_number(self, valid_span_dict):
        valid_span_dict["sequence_number"] = -1
        with pytest.raises(Exception):
            HilbertbenchSpan.model_validate(valid_span_dict)

    def test_rejects_empty_events(self, valid_span_dict):
        # minItems: 1 — a span with no events is invalid
        valid_span_dict["events"] = []
        with pytest.raises(Exception):
            HilbertbenchSpan.model_validate(valid_span_dict)

    def test_rejects_lowercase_event_type(self, valid_span_dict):
        # event_type pattern is ^[A-Z_]+$ — lowercase must be rejected
        valid_span_dict["events"][0]["event_type"] = "execution_request"
        with pytest.raises(Exception):
            HilbertbenchSpan.model_validate(valid_span_dict)

    def test_rejects_extra_fields(self, valid_span_dict):
        valid_span_dict["ghost_field"] = "not_allowed"
        with pytest.raises(Exception):
            HilbertbenchSpan.model_validate(valid_span_dict)


# ── TestArtifact ──────────────────────────────────────────────────────────────

class TestArtifact:

    def test_valid_construction(self, valid_artifact_dict):
        artifact = HilbertbenchArtifactMetadata.model_validate(valid_artifact_dict)
        assert artifact.kind == Kind.circuit_qasm

    def test_roundtrip(self, valid_artifact_dict):
        artifact  = HilbertbenchArtifactMetadata.model_validate(valid_artifact_dict)
        artifact2 = HilbertbenchArtifactMetadata.model_validate(artifact.model_dump())
        assert artifact == artifact2

    def test_all_kind_values_valid(self, valid_artifact_dict):
        for kind in [
            "circuit_qasm", "pulse_schedule", "execution_outcome",
            "execution_error", "calibration_snapshot",
            "parameters", "observables", "generic_blob",
        ]:
            valid_artifact_dict["kind"] = kind
            artifact = HilbertbenchArtifactMetadata.model_validate(valid_artifact_dict)
            assert artifact.kind == Kind(kind)

    def test_all_encoding_values_valid(self, valid_artifact_dict):
        for enc in ["json", "ndjson", "parquet", "numpy_binary", "openqasm", "plaintext"]:
            valid_artifact_dict["encoding"] = enc
            artifact = HilbertbenchArtifactMetadata.model_validate(valid_artifact_dict)
            assert artifact.encoding == Encoding(enc)

    def test_all_compression_values_valid(self, valid_artifact_dict):
        for comp in ["gzip", "zstd", "snappy"]:
            valid_artifact_dict["compression"] = comp
            artifact = HilbertbenchArtifactMetadata.model_validate(valid_artifact_dict)
            assert artifact.compression == Compression(comp)

    def test_compression_null_allowed(self, valid_artifact_dict):
        valid_artifact_dict["compression"] = None
        artifact = HilbertbenchArtifactMetadata.model_validate(valid_artifact_dict)
        assert artifact.compression is None

    def test_size_bytes_zero_allowed(self, valid_artifact_dict):
        valid_artifact_dict["size_bytes"] = 0
        artifact = HilbertbenchArtifactMetadata.model_validate(valid_artifact_dict)
        assert artifact.size_bytes == 0

    def test_producer_null_allowed(self, valid_artifact_dict):
        valid_artifact_dict["producer"] = None
        artifact = HilbertbenchArtifactMetadata.model_validate(valid_artifact_dict)
        assert artifact.producer is None

    def test_hash_pattern_enforced(self, valid_artifact_dict):
        valid_artifact_dict["artifact_hash"] = "md5:abc123"
        with pytest.raises(Exception):
            HilbertbenchArtifactMetadata.model_validate(valid_artifact_dict)

    def test_hash_wrong_length_rejected(self, valid_artifact_dict):
        valid_artifact_dict["artifact_hash"] = "sha256:" + "a" * 32  # too short
        with pytest.raises(Exception):
            HilbertbenchArtifactMetadata.model_validate(valid_artifact_dict)

    def test_rejects_negative_size(self, valid_artifact_dict):
        valid_artifact_dict["size_bytes"] = -1
        with pytest.raises(Exception):
            HilbertbenchArtifactMetadata.model_validate(valid_artifact_dict)

    def test_rejects_ref_count_zero(self, valid_artifact_dict):
        # minimum: 1 — an artifact with zero references is orphaned
        valid_artifact_dict["ref_count"] = 0
        with pytest.raises(Exception):
            HilbertbenchArtifactMetadata.model_validate(valid_artifact_dict)

    def test_rejects_invalid_kind(self, valid_artifact_dict):
        valid_artifact_dict["kind"] = "unknown_type"
        with pytest.raises(Exception):
            HilbertbenchArtifactMetadata.model_validate(valid_artifact_dict)


# ── TestCatalog ───────────────────────────────────────────────────────────────

class TestCatalog:

    def test_valid_construction(self, valid_catalog_dict):
        catalog = HilbertbenchArtifactCatalog.model_validate(valid_catalog_dict)
        assert len(catalog.artifacts) == 1

    def test_roundtrip(self, valid_catalog_dict):
        catalog  = HilbertbenchArtifactCatalog.model_validate(valid_catalog_dict)
        catalog2 = HilbertbenchArtifactCatalog.model_validate(catalog.model_dump())
        assert catalog == catalog2

    def test_multiple_artifacts(self, valid_catalog_dict, valid_artifact_dict):
        second = dict(valid_artifact_dict)
        second["artifact_hash"] = HASH_KEY2
        second["file_path"] = f"artifacts/bb/{'b' * 64}.qasm"
        valid_catalog_dict["artifacts"][HASH_KEY2] = second
        catalog = HilbertbenchArtifactCatalog.model_validate(valid_catalog_dict)
        assert len(catalog.artifacts) == 2

    def test_empty_artifacts_allowed(self):
        catalog = HilbertbenchArtifactCatalog.model_validate({
            "hbcatalog_version": "1.0",
            "trace_id": TRACE_ID,
            "created_at": 1711234599000000000,
            "artifacts": {},
        })
        assert catalog.artifacts == {}

    def test_artifact_values_are_validated(self, valid_catalog_dict):
        # Even though the key is not pattern-validated by Pydantic (see below),
        # the VALUE must be a valid HilbertbenchArtifactMetadata
        valid_catalog_dict["artifacts"][HASH_KEY]["size_bytes"] = -99
        with pytest.raises(Exception):
            HilbertbenchArtifactCatalog.model_validate(valid_catalog_dict)

    def test_artifact_key_format_not_enforced_by_pydantic(self, valid_catalog_dict,
                                                            valid_artifact_dict):
        """
        catalog.json uses additionalProperties (not patternProperties) so Pydantic
        accepts any string key at runtime. Key format and key==artifact_hash
        integrity is the responsibility of reader/verify.py, not the Pydantic model.

        This is a deliberate design tradeoff — see design_decisions/0003.
        This test documents and pins the behaviour. If it starts raising,
        the schema was changed back to patternProperties.
        """
        valid_catalog_dict["artifacts"]["not-a-valid-hash"] = valid_artifact_dict
        catalog = HilbertbenchArtifactCatalog.model_validate(valid_catalog_dict)
        assert "not-a-valid-hash" in catalog.artifacts

    def test_rejects_wrong_version(self, valid_catalog_dict):
        valid_catalog_dict["hbcatalog_version"] = "2.0"
        with pytest.raises(Exception):
            HilbertbenchArtifactCatalog.model_validate(valid_catalog_dict)

    def test_rejects_extra_fields(self, valid_catalog_dict):
        valid_catalog_dict["surprise"] = True
        with pytest.raises(Exception):
            HilbertbenchArtifactCatalog.model_validate(valid_catalog_dict)