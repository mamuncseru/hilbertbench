"""
Tests for hilbertbench.recorder.tape

All I/O is isolated to tmp_path. All model types imported from
hilbertbench.models public interface only — never from v1_0 directly.
Adheres strictly to INV-001, INV-003, INV-004, and INV-007.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

from hilbertbench.models import (
    Kind,
    Encoding,
    Compression,
    TraceStatus,
    SpanStatus,
)
from hilbertbench.recorder.tape import HilbertTape, TapeClosedError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def runs(tmp_path: Path) -> Path:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    return runs_dir


@pytest.fixture
def small_file(tmp_path: Path) -> Path:
    f = tmp_path / "payload.bin"
    f.write_bytes(b"\x00" * 256)
    return f


@pytest.fixture
def dummy_hash() -> str:
    return "sha256:" + "a" * 64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_trace(tape: HilbertTape) -> dict:
    return json.loads((tape.dir_path / "trace.json").read_text())


def read_spans(tape: HilbertTape) -> list[dict]:
    lines = (tape.dir_path / "events.jsonl").read_text().splitlines()
    return [json.loads(l) for l in lines if l.strip()]


def read_catalog(tape: HilbertTape) -> dict:
    return json.loads((tape.dir_path / "catalog.json").read_text())


# ---------------------------------------------------------------------------
# 1. Directory and file creation on open
# ---------------------------------------------------------------------------

class TestOpen:
    def test_creates_run_directory(self, runs: Path) -> None:
        with HilbertTape(runs) as tape:
            assert tape.dir_path.is_dir()

    def test_dir_name_format(self, runs: Path) -> None:
        with HilbertTape(runs) as tape:
            parts = tape.dir_path.name.split("_")
            assert len(parts) == 3
            assert len(parts[0]) == 8    # YYYYMMDD
            assert len(parts[1]) == 6    # HHMMSS
            assert len(parts[2]) == 8    # UUID prefix

    def test_artifacts_subdir_exists(self, runs: Path) -> None:
        with HilbertTape(runs) as tape:
            assert (tape.dir_path / "artifacts").is_dir()

    def test_trace_json_written_on_open(self, runs: Path) -> None:
        with HilbertTape(runs) as tape:
            assert read_trace(tape)["status"] == TraceStatus.CRASHED_IN_FLIGHT.value

    def test_events_jsonl_created_on_open(self, runs: Path) -> None:
        with HilbertTape(runs) as tape:
            assert (tape.dir_path / "events.jsonl").exists()


# ---------------------------------------------------------------------------
# 2. Trace JSON lifecycle
# ---------------------------------------------------------------------------

class TestTraceLifecycle:
    def test_sealed_success_on_clean_exit(self, runs: Path) -> None:
        with HilbertTape(runs) as tape:
            pass
        assert read_trace(tape)["status"] == TraceStatus.SEALED_SUCCESS.value

    def test_sealed_with_errors_on_exception(self, runs: Path) -> None:
        with pytest.raises(ValueError):
            with HilbertTape(runs) as tape:
                raise ValueError("boom")
        assert read_trace(tape)["status"] == TraceStatus.SEALED_WITH_ERRORS.value

    def test_timestamp_end_absent_while_open(self, runs: Path) -> None:
        with HilbertTape(runs) as tape:
            assert read_trace(tape).get("timestamp_end") is None

    def test_timestamp_end_present_after_close(self, runs: Path) -> None:
        with HilbertTape(runs) as tape:
            pass
        assert read_trace(tape)["timestamp_end"] > 0

    def test_tags_persisted(self, runs: Path) -> None:
        with HilbertTape(runs, tags={"env": "test"}) as tape:
            pass
        assert read_trace(tape)["tags"]["env"] == "test"


# ---------------------------------------------------------------------------
# 3. Span recording
# ---------------------------------------------------------------------------

class TestSpans:
    def test_span_flushed_immediately_on_close(self, runs: Path, dummy_hash: str) -> None:
        with HilbertTape(runs) as tape:
            with tape.execution_span(payload_ref=dummy_hash):
                pass
            assert len(read_spans(tape)) == 1

    def test_span_fields_present(self, runs: Path, dummy_hash: str) -> None:
        with HilbertTape(runs) as tape:
            with tape.execution_span(payload_ref=dummy_hash, backend_id="ibm_kyiv"):
                pass
        s = read_spans(tape)[0]
        assert s["payload_ref"] == dummy_hash
        assert s["backend_id"] == "ibm_kyiv"
        assert s["span_id"]
        assert s["timestamp_start"] > 0
        assert s["status"] == SpanStatus.COMPLETED.value

    def test_span_nesting_parent_id(self, runs: Path, dummy_hash: str) -> None:
        with HilbertTape(runs) as tape:
            with tape.execution_span(payload_ref=dummy_hash) as outer:
                with tape.execution_span(payload_ref=dummy_hash):
                    pass
        spans = read_spans(tape)
        assert len(spans) == 2
        # Inner span is closed and flushed first, so it is at index 0
        inner = spans[0]
        outer_span_json = spans[1]
        assert inner["parent_span_id"] == outer_span_json["span_id"]

    def test_root_span_has_no_parent(self, runs: Path, dummy_hash: str) -> None:
        with HilbertTape(runs) as tape:
            with tape.execution_span(payload_ref=dummy_hash):
                pass
        assert read_spans(tape)[0]["parent_span_id"] is None

    def test_sequence_numbers_monotonic_and_unique(self, runs: Path, dummy_hash: str) -> None:
        with HilbertTape(runs) as tape:
            for _ in range(5):
                with tape.execution_span(payload_ref=dummy_hash):
                    pass
        seqs = [s["sequence_number"] for s in read_spans(tape)]
        assert seqs == sorted(seqs)
        assert len(set(seqs)) == len(seqs)

    def test_span_event_recorded(self, runs: Path, dummy_hash: str) -> None:
        with HilbertTape(runs) as tape:
            with tape.execution_span(payload_ref=dummy_hash) as handle:
                handle.add_event("CALIBRATION_CHECK", {"value": 42})
                
        events = read_spans(tape)[0]["events"]
        # Should have REQUEST, CALIBRATION_CHECK, RESULT
        assert len(events) == 3
        assert events[1]["event_type"] == "CALIBRATION_CHECK"
        assert events[1]["attributes"]["value"] == 42


# ---------------------------------------------------------------------------
# 4. Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_parallel_spans_do_not_cross_nest(self, runs: Path, dummy_hash: str) -> None:
        """Per-thread span stack must be independent (threading.local check)."""
        results: dict[str, Any] = {}

        def worker(name: str, tape: HilbertTape) -> None:
            # Capture parent BEFORE opening our span. 
            # If thread-locals are working, this will be None even if the other thread is inside a span.
            parent_before = tape._current_parent_id()
            with tape.execution_span(payload_ref=dummy_hash):
                time.sleep(0.02)
            results[name] = parent_before

        with HilbertTape(runs) as tape:
            t1 = threading.Thread(target=worker, args=("thread-a", tape))
            t2 = threading.Thread(target=worker, args=("thread-b", tape))
            t1.start(); t2.start()
            t1.join(); t2.join()

        assert results["thread-a"] is None
        assert results["thread-b"] is None

    def test_events_jsonl_valid_under_concurrency(self, runs: Path, dummy_hash: str) -> None:
        """Every line must be valid JSON after 10 concurrent span writers."""
        with HilbertTape(runs) as tape:
            threads = []
            for i in range(10):
                def run() -> None:
                    with tape.execution_span(payload_ref=dummy_hash):
                        time.sleep(0.005)
                threads.append(threading.Thread(target=run))
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        lines = (tape.dir_path / "events.jsonl").read_text().splitlines()
        assert len(lines) == 10
        for line in lines:
            json.loads(line)


# ---------------------------------------------------------------------------
# 5. Artifact attachment
# ---------------------------------------------------------------------------

class TestAttach:
    def test_artifact_copied_to_artifacts_dir(
        self, runs: Path, small_file: Path
    ) -> None:
        with HilbertTape(runs) as tape:
            aid = tape.attach_artifact(small_file, kind=Kind.generic_blob, encoding=Encoding.numpy_binary)
        copied = list((tape.dir_path / "artifacts").iterdir())
        assert len(copied) == 1
        # aid is sha256:..., strip the prefix for filename
        assert aid.replace("sha256:", "") in copied[0].name

    def test_catalog_json_written_on_close(
        self, runs: Path, small_file: Path
    ) -> None:
        with HilbertTape(runs) as tape:
            tape.attach_artifact(small_file, kind=Kind.generic_blob, encoding=Encoding.numpy_binary)
        assert len(read_catalog(tape)["artifacts"]) == 1

    def test_sha256_correct(self, runs: Path, small_file: Path) -> None:
        expected = "sha256:" + hashlib.sha256(small_file.read_bytes()).hexdigest()
        with HilbertTape(runs) as tape:
            aid = tape.attach_artifact(small_file, kind=Kind.generic_blob, encoding=Encoding.numpy_binary)
        assert read_catalog(tape)["artifacts"][aid]["artifact_hash"] == expected

    def test_size_bytes_correct(self, runs: Path, small_file: Path) -> None:
        with HilbertTape(runs) as tape:
            aid = tape.attach_artifact(small_file, kind=Kind.generic_blob, encoding=Encoding.numpy_binary)
        assert read_catalog(tape)["artifacts"][aid]["size_bytes"] == 256

    def test_missing_file_raises(self, runs: Path) -> None:
        with HilbertTape(runs) as tape:
            with pytest.raises(FileNotFoundError):
                tape.attach_artifact("/nonexistent/path.bin", kind=Kind.generic_blob, encoding=Encoding.numpy_binary)

    def test_compression_stored(
        self, runs: Path, small_file: Path
    ) -> None:
        with HilbertTape(runs) as tape:
            aid = tape.attach_artifact(
                small_file, kind=Kind.generic_blob, encoding=Encoding.numpy_binary, compression=Compression.gzip
            )
        assert read_catalog(tape)["artifacts"][aid]["compression"] == Compression.gzip.value


# ---------------------------------------------------------------------------
# 6. Freeze-on-close
# ---------------------------------------------------------------------------

class TestFreezeOnClose:
    def test_span_after_close_raises(self, runs: Path, dummy_hash: str) -> None:
        with HilbertTape(runs) as tape:
            pass
        with pytest.raises(TapeClosedError):
            with tape.execution_span(payload_ref=dummy_hash):
                pass

    def test_attach_after_close_raises(
        self, runs: Path, small_file: Path
    ) -> None:
        with HilbertTape(runs) as tape:
            pass
        with pytest.raises(TapeClosedError):
            tape.attach_artifact(small_file, kind=Kind.generic_blob, encoding=Encoding.numpy_binary)

    def test_close_idempotent(self, runs: Path) -> None:
        with HilbertTape(runs) as tape:
            pass
        tape.close()
        tape.close()  # must not raise


# ---------------------------------------------------------------------------
# 7. Exception path (INV-007)
# ---------------------------------------------------------------------------

class TestExceptionPath:
    def test_exception_span_written(self, runs: Path, dummy_hash: str) -> None:
        with pytest.raises(ZeroDivisionError):
            with HilbertTape(runs) as tape:
                with tape.execution_span(payload_ref=dummy_hash):
                    _ = 1 / 0
                    
        # The exception occurred inside the span, so it should be FAILED
        span = read_spans(tape)[0]
        assert span["status"] == SpanStatus.FAILED.value
        # Check that the ERROR event was logged
        event_types = [e["event_type"] for e in span["events"]]
        assert "ERROR" in event_types

    def test_original_exception_propagates(self, runs: Path, dummy_hash: str) -> None:
        with pytest.raises(KeyError):
            with HilbertTape(runs) as tape: # <-- Added 'as tape' here
                with tape.execution_span(payload_ref=dummy_hash):
                    raise KeyError("missing")

    def test_exception_attributes_captured(self, runs: Path, dummy_hash: str) -> None:
        with pytest.raises(ValueError):
            with HilbertTape(runs) as tape:
                with tape.execution_span(payload_ref=dummy_hash):
                    raise ValueError("bad input")
                    
        span = read_spans(tape)[0]
        error_event = next(e for e in span["events"] if e["event_type"] == "ERROR")
        assert error_event["attributes"]["exception_type"] == "ValueError"
        assert "bad input" in error_event["attributes"]["exception_message"]