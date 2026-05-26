"""
hilbertbench/recorder/tape.py

Append-only, crash-safe flight recorder for a single benchmark run.
Strictly adheres to INV-001, INV-003, INV-004, and INV-007.

Thread safety:
  threading.local()  — per-thread span-id stack for Qiskit/PennyLane parallelism
  threading.Lock()   — guards every file.write() call
  itertools.count()  — GIL-atomic monotonic sequence counter
"""

import hashlib
import itertools
import shutil
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from hilbertbench.models import (
    HilbertbenchTraceManifest,
    HilbertbenchSpan,
    HilbertbenchArtifactMetadata,
    HilbertbenchArtifactCatalog,
    ClientEnvironment,
    Event,
    Kind,
    Encoding,
    Compression,
    Mode,
    TraceStatus,
    SpanStatus,
)


class TapeClosedError(RuntimeError):
    """Raised when writing to a Tape after it has been sealed."""


class SpanHandle:
    """
    Returned by ``Tape.execution_span()``. Valid only inside the context manager.
    Allows integrations to attach custom events mid-execution.
    """
    def __init__(self, tape: "Tape") -> None:
        self._tape = tape
        self.events: list[Event] = []
        self.outcome_ref: Optional[str] = None
        self.tags: Optional[dict[str, Any]] = None

    def add_event(self, event_type: str, attributes: Optional[dict[str, Any]] = None) -> None:
        self._tape._assert_open()
        self.events.append(
            Event(
                event_id=uuid.uuid4(),
                event_type=event_type,
                timestamp=time.time_ns(),
                attributes=attributes,
            )
        )


class HilbertTape:
    """
    Context manager owning the full lifecycle of one benchmark trace.
    Writes immediately to disk to prevent data loss on OOM kills.
    """

    def __init__(
        self,
        output_root: Path | str,
        mode: Mode = Mode.passive,
        tags: Optional[dict[str, str]] = None,
    ) -> None:
        self._output_root = Path(output_root)
        self.trace_id = uuid.uuid4()
        self._start_ns = time.time_ns()
        self._mode = mode
        self._tags = tags

        # Prefix with timestamp for easy human sorting
        stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        self.dir_path = self._output_root / f"{stamp}_{str(self.trace_id)[:8]}"
        self.dir_path.mkdir(parents=True, exist_ok=True)
        (self.dir_path / "artifacts").mkdir(exist_ok=True)

        # Concurrency safety
        self._event_lock = threading.Lock()
        self._context = threading.local()
        self._seq = itertools.count()

        self._event_file = None
        self._closed = False
        self._artifacts: dict[str, HilbertbenchArtifactMetadata] = {}

    def _capture_environment(self) -> ClientEnvironment:
        import platform
        return ClientEnvironment(
            hilbertbench_version="0.1.0-dev",
            python_version=platform.python_version(),
            os_platform=platform.system().lower(),
            frameworks={},
        )

    def __enter__(self) -> "HilbertTape":
        """Open events.jsonl and stamp trace.json as CRASHED_IN_FLIGHT."""
        self._event_file = open(
            self.dir_path / "events.jsonl",
            "a",
            encoding="utf-8",
            buffering=1,  # Line-buffered: flush to OS on every newline
        )
        self._write_trace_json(TraceStatus.CRASHED_IN_FLIGHT, end_ns=None)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if exc_type is not None:
            # We don't swallow the error, but we mark the trace as failed
            self.close(TraceStatus.SEALED_WITH_ERRORS)
            return False
        
        self.close(TraceStatus.SEALED_SUCCESS)
        return False

    def close(self, status: TraceStatus = TraceStatus.SEALED_SUCCESS) -> None:
        """Seal the tape. Idempotent — safe to call multiple times."""
        if self._closed:
            return
        self._closed = True
        end_ns = time.time_ns()

        if self._event_file and not self._event_file.closed:
            self._event_file.flush()
            self._event_file.close()

        self._write_trace_json(status, end_ns=end_ns)
        self._write_catalog_json()

    @contextmanager
    def execution_span(
        self,
        payload_ref: str,
        backend_id: Optional[str] = None,
    ) -> Iterator[SpanHandle]:
        """
        Open an execution span. Flushes to events.jsonl on exit.
        Ensures INV-007: Exceptions are recorded and immediately re-raised.
        """
        self._assert_open()

        span_id = uuid.uuid4()
        seq = next(self._seq)
        parent_id = self._current_parent_id()
        start_ns = time.time_ns()

        self._push_span(span_id)
        handle = SpanHandle(tape=self)
        
        # MinItems: 1 enforced by schema
        handle.add_event("EXECUTION_REQUEST")

        status = SpanStatus.IN_FLIGHT

        try:
            yield handle
            
            # Clean exit
            status = SpanStatus.COMPLETED
            handle.add_event("EXECUTION_RESULT")
            
        except Exception as exc:
            # Exception path (INV-007)
            status = SpanStatus.FAILED
            handle.add_event("ERROR", attributes={
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            })
            raise
            
        finally:
            self._pop_span(span_id)
            
            # Flush span to disk immediately
            span = HilbertbenchSpan(
                hbspan_version="1.0",
                span_id=span_id,
                trace_id=self.trace_id,
                parent_span_id=parent_id,
                sequence_number=seq,
                timestamp_start=start_ns,
                status=status,
                backend_id=backend_id,
                payload_ref=payload_ref,
                outcome_ref=handle.outcome_ref,
                events=handle.events,
                tags=handle.tags,
            )
            self._flush_span(span)

    def attach_artifact(
        self,
        src_path: Path | str,
        kind: Kind,
        encoding: Encoding,
        compression: Optional[Compression] = None,
        producer: Optional[str] = None,
    ) -> str:
        """
        Hashes and copies a physical file to the run directory.
        Returns the artifact_hash string (sha256:...).
        """
        self._assert_open()
        src = Path(src_path)
        if not src.is_file():
            raise FileNotFoundError(f"Artifact source not found: {src}")

        data = src.read_bytes()
        sha256_hash = f"sha256:{hashlib.sha256(data).hexdigest()}"
        
        # Write to physical layout
        # Update the physical layout to use 2-character sharding
        hash_hex = sha256_hash.replace('sha256:', '')
        shard = hash_hex[:2]
        shard_dir = self.dir_path / "artifacts" / shard
        shard_dir.mkdir(exist_ok=True)
        
        relative_path = f"artifacts/{shard}/{hash_hex}{src.suffix}"
        dest = self.dir_path / relative_path
        shutil.copy2(src, dest)

        self._artifacts[sha256_hash] = HilbertbenchArtifactMetadata(
            artifact_hash=sha256_hash,
            kind=kind,
            encoding=encoding,
            file_path=relative_path,
            size_bytes=len(data),
            compression=compression,
            created_at=time.time_ns(),
            producer=producer,
            ref_count=1,
        )
        return sha256_hash

    def _assert_open(self) -> None:
        if self._closed:
            raise TapeClosedError(f"Tape {self.trace_id} is sealed.")

    def _current_parent_id(self) -> Optional[uuid.UUID]:
        stack = getattr(self._context, "stack", [])
        return stack[-1] if stack else None

    def _push_span(self, span_id: uuid.UUID) -> None:
        if not hasattr(self._context, "stack"):
            self._context.stack = []
        self._context.stack.append(span_id)

    def _pop_span(self, span_id: uuid.UUID) -> None:
        stack = getattr(self._context, "stack", [])
        if stack and stack[-1] == span_id:
            stack.pop()

    def _flush_span(self, span: HilbertbenchSpan) -> None:
        line = span.model_dump_json() + "\n"
        with self._event_lock:
            if self._event_file and not self._event_file.closed:
                self._event_file.write(line)

    def _write_trace_json(self, status: TraceStatus, end_ns: Optional[int]) -> None:
        manifest = HilbertbenchTraceManifest(
            hbtrace_version="1.0",
            trace_id=self.trace_id,
            mode=self._mode,
            timestamp_start=self._start_ns,
            timestamp_end=end_ns,
            status=status,
            client_environment=self._capture_environment(),
            tags=self._tags,
        )
        (self.dir_path / "trace.json").write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )

    def _write_catalog_json(self) -> None:
        catalog = HilbertbenchArtifactCatalog(
            hbcatalog_version="1.0",
            trace_id=self.trace_id,
            created_at=time.time_ns(),
            artifacts=self._artifacts,
        )
        (self.dir_path / "catalog.json").write_text(
            catalog.model_dump_json(indent=2), encoding="utf-8"
        )