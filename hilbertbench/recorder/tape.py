#!/usr/bin/env python
#
# file: hilbertbench/recorder/tape.py
#
# revision history:
#  20260604 (am): cleaned up to project coding standards
#
# Append-only, crash-safe flight recorder for a single benchmark run.
# Strictly adheres to INV-001, INV-003, INV-004, and INV-007.
#
# Thread safety:
#  threading.local()  — per-thread span-id stack (Qiskit/PennyLane)
#  threading.Lock()   — guards every file.write() call
#  itertools.count()  — GIL-atomic monotonic sequence counter
#------------------------------------------------------------------------------

# import system modules
#
import hashlib
import itertools
import os
import shutil
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

# import hilbertbench modules
#
from hilbertbench.models import (
    HilbertbenchTraceManifest,
    HilbertbenchSpan,
    HilbertbenchArtifactMetadata,
    HilbertbenchArtifactCatalog,
    ClientEnvironment,
    Event,
    InlineArtifact,
    IntegritySeal,
    Kind,
    Encoding,
    Compression,
    Mode,
    TraceStatus,
    SpanStatus,
)

#------------------------------------------------------------------------------
#
# global variables are listed here
#
#------------------------------------------------------------------------------

# set the filename using basename
#
__FILE__ = os.path.basename(__file__)

#------------------------------------------------------------------------------
#
# classes are listed here
#
#------------------------------------------------------------------------------

class TapeClosedError(RuntimeError):
    """
    Class: TapeClosedError

    description:
     Raised when writing to a Tape after it has been sealed.
    """
    pass
#
# end of class


class SpanHandle:
    """
    Class: SpanHandle

    description:
     Returned by HilbertTape.execution_span(). Valid only inside the
     context manager. Allows integrations to attach custom events and
     inline artifacts mid-execution.
    """

    def __init__(self, tape: "HilbertTape") -> None:
        """
        method: constructor

        arguments:
         tape: the parent HilbertTape instance

        return:
         none

        description:
         Initialises the handle with empty events, no outcome ref, no
         tags, and an empty inline artifact store.
        """

        # store reference to the parent tape
        #
        self._tape = tape

        # initialise the mutable event list
        #
        self.events: list[Event] = []

        # initialise outcome ref and tags to none
        #
        self.outcome_ref: Optional[str] = None
        self.tags: Optional[dict[str, Any]] = None

        # initialise inline artifact store
        #
        self.inline_artifacts: dict[str, InlineArtifact] = {}
    #
    # end of method

    def add_event(
        self,
        event_type: str,
        attributes: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        method: add_event

        arguments:
         event_type: string label for the event (e.g. 'ERROR')
         attributes: optional dict of key/value metadata

        return:
         none

        description:
         Appends a timestamped Event record to this span's event list.
         Asserts the tape is still open before writing (INV-007).
        """

        # assert the tape is still accepting writes
        #
        self._tape._assert_open()

        # build and append the event record
        #
        self.events.append(
            Event(
                event_id=uuid.uuid4(),
                event_type=event_type,
                timestamp=time.time_ns(),
                attributes=attributes,
            )
        )
    #
    # end of method

    def attach_inline(
        self,
        data: str,
        kind: str,
        encoding: str = "json",
        producer: Optional[str] = None,
    ) -> str:
        """
        method: attach_inline

        arguments:
         data:     the payload as a UTF-8 string
         kind:     artifact kind label (e.g. 'parameters')
         encoding: encoding label, default 'json'
         producer: optional string identifying the producing component

        return:
         the sha256 content-address hash string

        description:
         Embeds a small artifact directly in this span record rather
         than writing a separate file. Avoids per-span file overhead
         for numeric data (outcomes, parameters, observables). Returns
         the sha256 hash for use as a reference.
        """

        # assert the tape is still accepting writes
        #
        self._tape._assert_open()

        # compute the content-address hash
        #
        data_bytes = data.encode("utf-8")
        sha256_hash = f"sha256:{hashlib.sha256(data_bytes).hexdigest()}"

        # store the inline artifact record
        #
        self.inline_artifacts[sha256_hash] = InlineArtifact(
            kind=kind,
            encoding=encoding,
            data=data,
            size_bytes=len(data_bytes),
            created_at=time.time_ns(),
            producer=producer,
        )

        # exit gracefully
        #
        return sha256_hash
    #
    # end of method
#
# end of class


class HilbertTape:
    """
    Class: HilbertTape

    description:
     Context manager owning the full lifecycle of one benchmark trace.
     Writes immediately to disk to prevent data loss on OOM kills.
    """

    def __init__(
        self,
        output_root: Path | str,
        mode: Mode = Mode.passive,
        tags: Optional[dict[str, str]] = None,
    ) -> None:
        """
        method: constructor

        arguments:
         output_root: root directory under which the run directory is created
         mode:        trace mode (passive or active), default passive
         tags:        optional dict of string tags attached to the trace

        return:
         none

        description:
         Creates the run directory, initialises concurrency primitives,
         and prepares the artifact store. The directory name embeds a
         wall-clock timestamp for human-readable sorting.
        """

        # store configuration
        #
        self._output_root = Path(output_root)
        self.trace_id = uuid.uuid4()
        self._start_ns = time.time_ns()
        self._mode = mode
        self._tags = tags

        # build a timestamped run directory name for easy sorting
        #
        stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        self.dir_path = (
            self._output_root / f"{stamp}_{str(self.trace_id)[:8]}"
        )
        self.dir_path.mkdir(parents=True, exist_ok=True)
        (self.dir_path / "artifacts").mkdir(exist_ok=True)

        # initialise concurrency safety primitives
        #
        self._event_lock = threading.Lock()
        self._context = threading.local()
        self._seq = itertools.count()

        # initialise file handle and closed flag
        #
        self._event_file = None
        self._closed = False
        self._artifacts: dict[str, HilbertbenchArtifactMetadata] = {}

        # running tally of inline artifacts embedded across all spans;
        # combined with len(self._artifacts) for integrity_seal.artifact_count
        #
        self._inline_artifact_count = 0
    #
    # end of method

    def _capture_environment(self) -> ClientEnvironment:
        """
        method: _capture_environment

        arguments:
         none

        return:
         a ClientEnvironment record

        description:
         Captures the current Python and OS environment for embedding
         in the trace manifest.
        """

        # imported here to avoid pulling platform into module-level deps
        #
        import platform

        # exit gracefully
        #
        return ClientEnvironment(
            hilbertbench_version="0.1.0-dev",
            python_version=platform.python_version(),
            os_platform=platform.system().lower(),
            frameworks={},
        )
    #
    # end of method

    def __enter__(self) -> "HilbertTape":
        """
        method: __enter__

        arguments:
         none

        return:
         self

        description:
         Opens events.jsonl for line-buffered append and stamps
         trace.json as CRASHED_IN_FLIGHT so an unexpected process kill
         is detectable.
        """

        # open the event stream file in line-buffered append mode
        #
        self._event_file = open(
            self.dir_path / "events.jsonl",
            "a",
            encoding="utf-8",
            buffering=1,
        )

        # stamp the manifest as in-flight (crash-safe sentinel)
        #
        self._write_trace_json(TraceStatus.CRASHED_IN_FLIGHT, end_ns=None)

        # exit gracefully
        #
        return self
    #
    # end of method

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """
        method: __exit__

        arguments:
         exc_type: exception type, or None on clean exit
         exc_val:  exception value, or None on clean exit
         exc_tb:   exception traceback, or None on clean exit

        return:
         False — exceptions are never suppressed (INV-007)

        description:
         Seals the tape with SEALED_WITH_ERRORS on exception or
         SEALED_SUCCESS on clean exit. Never suppresses exceptions.
        """

        # seal with error status if an exception occurred
        #
        if exc_type is not None:
            self.close(TraceStatus.SEALED_WITH_ERRORS)
            return False

        # exit gracefully — seal as success
        #
        self.close(TraceStatus.SEALED_SUCCESS)
        return False
    #
    # end of method

    def close(
        self,
        status: TraceStatus = TraceStatus.SEALED_SUCCESS,
    ) -> None:
        """
        method: close

        arguments:
         status: the final TraceStatus to stamp; default SEALED_SUCCESS

        return:
         none

        description:
         Seals the tape. Idempotent — safe to call multiple times.
         Flushes and closes events.jsonl, computes the integrity seal,
         then writes trace.json and catalog.json.
        """

        # guard against double-close
        #
        if self._closed:
            return
        self._closed = True
        end_ns = time.time_ns()

        # flush and close the event stream file
        #
        if self._event_file and not self._event_file.closed:
            self._event_file.flush()
            self._event_file.close()

        # compute the integrity seal now that the event stream is flushed
        #
        seal = self._compute_integrity_seal()

        # write the final manifest and artifact catalog
        #
        self._write_trace_json(status, end_ns=end_ns, integrity_seal=seal)
        self._write_catalog_json()
    #
    # end of method

    def _compute_integrity_seal(self) -> Optional[IntegritySeal]:
        """
        method: _compute_integrity_seal

        arguments:
         none

        return:
         an IntegritySeal, or None if the event stream file is missing

        description:
         Builds the integrity seal: a SHA-256 hash of the completed
         event stream combined with the total artifact count (file-store
         + inline). Returns None if events.jsonl is absent, indicating
         abnormal termination before any write occurred.
        """

        # locate the event stream file
        #
        events_path = self.dir_path / "events.jsonl"
        if not events_path.exists():
            return None

        # hash the event stream in 1 MB chunks
        #
        hasher = hashlib.sha256()
        with open(events_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)

        # compute total artifact count (file-store + inline)
        #
        artifact_count = (
            len(self._artifacts) + self._inline_artifact_count
        )

        # exit gracefully
        #
        return IntegritySeal(
            event_stream_checksum=f"sha256:{hasher.hexdigest()}",
            artifact_count=artifact_count,
        )
    #
    # end of method

    @contextmanager
    def execution_span(
        self,
        payload_ref: str,
        backend_id: Optional[str] = None,
    ) -> Iterator[SpanHandle]:
        """
        method: execution_span

        arguments:
         payload_ref: content-address ref for the circuit/payload artifact
         backend_id:  optional identifier for the backend being used

        return:
         a SpanHandle context manager

        description:
         Opens an execution span. Flushes to events.jsonl on exit.
         Ensures INV-007: exceptions are recorded and immediately
         re-raised. Supports nested spans via per-thread parent stacks.
        """

        # assert tape is still accepting writes
        #
        self._assert_open()

        # initialise span metadata
        #
        span_id = uuid.uuid4()
        seq = next(self._seq)
        parent_id = self._current_parent_id()
        start_ns = time.time_ns()

        # push span onto the per-thread parent stack
        #
        self._push_span(span_id)
        handle = SpanHandle(tape=self)

        # emit the initial event (schema requires at least 1 event)
        #
        handle.add_event("EXECUTION_REQUEST")
        status = SpanStatus.IN_FLIGHT

        try:

            # yield control to the caller
            #
            yield handle

            # clean exit — record result event
            #
            status = SpanStatus.COMPLETED
            handle.add_event("EXECUTION_RESULT")

        except Exception as exc:

            # exception path — record error event then re-raise (INV-007)
            #
            status = SpanStatus.FAILED
            handle.add_event("ERROR", attributes={
                "exception_type":    type(exc).__name__,
                "exception_message": str(exc),
            })
            raise

        finally:

            # pop span off the per-thread parent stack
            #
            self._pop_span(span_id)

            # tally inline artifacts for the integrity seal
            #
            with self._event_lock:
                self._inline_artifact_count += len(handle.inline_artifacts)

            # flush span to disk immediately
            #
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
                inline_artifacts=handle.inline_artifacts or None,
            )
            self._flush_span(span)
    #
    # end of method

    def attach_artifact(
        self,
        src_path: Path | str,
        kind: Kind,
        encoding: Encoding,
        compression: Optional[Compression] = None,
        producer: Optional[str] = None,
    ) -> str:
        """
        method: attach_artifact

        arguments:
         src_path:    path to the source file to attach
         kind:        artifact kind (e.g. Kind.circuit, Kind.parameters)
         encoding:    encoding label (e.g. Encoding.qasm3, Encoding.npy)
         compression: optional compression type applied to the file
         producer:    optional string identifying the producing component

        return:
         the sha256 content-address hash string

        description:
         Hashes and copies a physical file into the run directory using
         2-character shard layout (artifacts/{shard}/{full_hash}{ext}).
         Returns the artifact_hash string ('sha256:...').
        """

        # assert tape is still accepting writes
        #
        self._assert_open()

        # verify source file exists
        #
        src = Path(src_path)
        if not src.is_file():
            raise FileNotFoundError(
                f"Artifact source not found: {src}"
            )

        # compute the content-address hash
        #
        data = src.read_bytes()
        sha256_hash = f"sha256:{hashlib.sha256(data).hexdigest()}"

        # resolve the sharded destination path
        #
        hash_hex = sha256_hash.replace("sha256:", "")
        shard = hash_hex[:2]
        shard_dir = self.dir_path / "artifacts" / shard
        shard_dir.mkdir(exist_ok=True)

        relative_path = f"artifacts/{shard}/{hash_hex}{src.suffix}"
        dest = self.dir_path / relative_path
        shutil.copy2(src, dest)

        # register the artifact in the in-memory catalog
        #
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

        # exit gracefully
        #
        return sha256_hash
    #
    # end of method

    def _assert_open(self) -> None:
        """
        method: _assert_open

        arguments:
         none

        return:
         none

        description:
         Raises TapeClosedError if the tape has already been sealed.
        """

        # raise if the tape is sealed
        #
        if self._closed:
            raise TapeClosedError(f"Tape {self.trace_id} is sealed.")
    #
    # end of method

    def _current_parent_id(self) -> Optional[uuid.UUID]:
        """
        method: _current_parent_id

        arguments:
         none

        return:
         the UUID of the innermost enclosing span, or None

        description:
         Reads the top of the per-thread span-id stack to determine
         the parent span for nested execution contexts.
        """

        # fetch the per-thread stack and return its top element
        #
        stack = getattr(self._context, "stack", [])
        return stack[-1] if stack else None
    #
    # end of method

    def _push_span(self, span_id: uuid.UUID) -> None:
        """
        method: _push_span

        arguments:
         span_id: the UUID of the span to push

        return:
         none

        description:
         Pushes a span ID onto the per-thread parent stack.
        """

        # initialise the stack on first use for this thread
        #
        if not hasattr(self._context, "stack"):
            self._context.stack = []

        # push the span id
        #
        self._context.stack.append(span_id)
    #
    # end of method

    def _pop_span(self, span_id: uuid.UUID) -> None:
        """
        method: _pop_span

        arguments:
         span_id: the UUID of the span to pop

        return:
         none

        description:
         Pops the span ID from the top of the per-thread parent stack.
         Guards against mismatched pops by verifying the top matches.
        """

        # pop only if the top of the stack matches the expected span
        #
        stack = getattr(self._context, "stack", [])
        if stack and stack[-1] == span_id:
            stack.pop()
    #
    # end of method

    def _flush_span(self, span: HilbertbenchSpan) -> None:
        """
        method: _flush_span

        arguments:
         span: the HilbertbenchSpan to write

        return:
         none

        description:
         Serialises the span to a single JSONL line and writes it to
         the event stream under the file lock.
        """

        # serialise the span to a newline-terminated JSON line
        #
        line = span.model_dump_json() + "\n"

        # write to disk under the file lock
        #
        with self._event_lock:
            if self._event_file and not self._event_file.closed:
                self._event_file.write(line)
    #
    # end of method

    def _write_trace_json(
        self,
        status: TraceStatus,
        end_ns: Optional[int],
        integrity_seal: Optional[IntegritySeal] = None,
    ) -> None:
        """
        method: _write_trace_json

        arguments:
         status:         the TraceStatus to embed in the manifest
         end_ns:         wall-clock end timestamp in nanoseconds, or None
         integrity_seal: optional IntegritySeal to embed

        return:
         none

        description:
         Writes or overwrites trace.json with the current manifest.
         Called on __enter__ with CRASHED_IN_FLIGHT and on close()
         with the final status.
        """

        # build the manifest object
        #
        manifest = HilbertbenchTraceManifest(
            hbtrace_version="1.0",
            trace_id=self.trace_id,
            mode=self._mode,
            timestamp_start=self._start_ns,
            timestamp_end=end_ns,
            status=status,
            client_environment=self._capture_environment(),
            integrity_seal=integrity_seal,
            tags=self._tags,
        )

        # write to disk
        #
        (self.dir_path / "trace.json").write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )
    #
    # end of method

    def _write_catalog_json(self) -> None:
        """
        method: _write_catalog_json

        arguments:
         none

        return:
         none

        description:
         Writes catalog.json with the complete artifact inventory.
         Called once on close() after all spans are flushed.
        """

        # build the catalog object
        #
        catalog = HilbertbenchArtifactCatalog(
            hbcatalog_version="1.0",
            trace_id=self.trace_id,
            created_at=time.time_ns(),
            artifacts=self._artifacts,
        )

        # write to disk
        #
        (self.dir_path / "catalog.json").write_text(
            catalog.model_dump_json(indent=2), encoding="utf-8"
        )
    #
    # end of method
#
# end of class

#
# end of file
