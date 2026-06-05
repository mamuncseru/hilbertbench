#!/usr/bin/env python
#
# file: hilbertbench/trace/trace.py
#
# revision history:
#  20260604 (am): cleaned up to project coding standards
#
# HilbertTrace — the unified, public data API for a recorded run.
#
# One object, created from a run directory, gives clean access to everything
# in a trace without the caller needing to know about JSONL vs Parquet,
# inline vs file-store artifacts, sharded directories, or content hashes.
#
#   from hilbertbench import HilbertTrace
#
#   trace = HilbertTrace("runs/20260605_xxx")
#   trace.status                 # "SEALED_SUCCESS"
#   trace.tags                   # {"task": "two_moons", ...}
#   trace.spans                  # pandas DataFrame (one row per span)
#
#   for span in trace.completed():
#       span.outcome             # resolved numpy / float / dict
#       span.parameters          # resolved parameter vector
#       span.circuit             # circuit QASM text
#
#   trace.numeric_outcomes()     # flat np.array of every scalar outcome
#   trace.calibration()          # device T1/T2/readout dict, or None
#   trace.verify()               # cryptographic + causal integrity check
#
# Built-in analyzers consume HilbertTrace; so can any user with numpy/pandas.
#------------------------------------------------------------------------------

# future imports must come first
#
from __future__ import annotations

# import system modules
#
import json
import os
from pathlib import Path
from typing import Any, Iterator, Optional

# import third-party modules
#
import numpy as np

# import hilbertbench modules
#
from hilbertbench.trace.span import SpanView

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

class HilbertTrace:
    """
    Class: HilbertTrace

    description:
     Lazy, read-only accessor over a HilbertBench run directory. All
     heavy data (JSONL parsing, numpy loads) is deferred until first
     access and then cached. Thread safety is not guaranteed — construct
     one HilbertTrace per thread if parallel access is needed.
    """

    def __init__(self, run_dir: Path | str) -> None:
        """
        method: constructor

        arguments:
         run_dir: path to the run directory produced by HilbertTape

        return:
         none

        description:
         Validates that the directory exists and contains events.jsonl,
         then initialises lazy-load caches. No file I/O beyond the
         existence checks happens at construction time.
        """

        # resolve and validate the run directory
        #
        self.run_dir = Path(run_dir)
        if not self.run_dir.is_dir():
            raise FileNotFoundError(
                f"Run directory not found: {self.run_dir}"
            )
        if not (self.run_dir / "events.jsonl").exists():
            raise FileNotFoundError(
                f"No events.jsonl in {self.run_dir} "
                f"— not a HilbertBench run directory."
            )

        # initialise lazy-load caches
        #
        self._manifest: Optional[dict] = None
        self._catalog: Optional[dict] = None
        self._spans: Optional[list[dict]] = None
        self._df = None
    #
    # end of method

    # ---- lazy loaders -------------------------------------------------------

    @property
    def manifest(self) -> dict:
        """
        method: manifest

        arguments:
         none

        return:
         the parsed trace.json manifest dict
        """

        # load and cache on first access
        #
        if self._manifest is None:
            self._manifest = json.loads(
                (self.run_dir / "trace.json").read_text()
            )
        return self._manifest
    #
    # end of method

    @property
    def catalog(self) -> dict:
        """
        method: catalog

        arguments:
         none

        return:
         the artifacts dict from catalog.json (keyed by sha256 hash)
        """

        # load and cache on first access
        #
        if self._catalog is None:
            raw = json.loads(
                (self.run_dir / "catalog.json").read_text()
            )
            self._catalog = raw.get("artifacts", {})
        return self._catalog
    #
    # end of method

    def _rows(self) -> list[dict]:
        """
        method: _rows

        arguments:
         none

        return:
         list of raw span dicts parsed from events.jsonl

        description:
         Reads and caches all span rows from events.jsonl. Blank lines
         are skipped. Result is cached for subsequent calls.
        """

        # parse and cache on first access
        #
        if self._spans is None:
            spans: list[dict] = []
            with open(self.run_dir / "events.jsonl", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        spans.append(json.loads(line))
            self._spans = spans
        return self._spans
    #
    # end of method

    # ---- trace-level metadata -----------------------------------------------

    @property
    def trace_id(self) -> Optional[str]:
        """
        method: trace_id

        arguments:
         none

        return:
         the trace UUID string, or None
        """
        return self.manifest.get("trace_id")
    #
    # end of method

    @property
    def status(self) -> Optional[str]:
        """
        method: status

        arguments:
         none

        return:
         the trace status string (e.g. 'SEALED_SUCCESS'), or None
        """
        return self.manifest.get("status")
    #
    # end of method

    @property
    def mode(self) -> Optional[str]:
        """
        method: mode

        arguments:
         none

        return:
         the trace mode string (e.g. 'passive', 'active'), or None
        """
        return self.manifest.get("mode")
    #
    # end of method

    @property
    def tags(self) -> dict:
        """
        method: tags

        arguments:
         none

        return:
         the tags dict attached to the trace; empty dict if none
        """
        return self.manifest.get("tags") or {}
    #
    # end of method

    @property
    def integrity_seal(self) -> Optional[dict]:
        """
        method: integrity_seal

        arguments:
         none

        return:
         the integrity_seal dict from the manifest, or None
        """
        return self.manifest.get("integrity_seal")
    #
    # end of method

    @property
    def environment(self) -> dict:
        """
        method: environment

        arguments:
         none

        return:
         the client_environment dict; empty dict if absent
        """
        return self.manifest.get("client_environment") or {}
    #
    # end of method

    # ---- span access --------------------------------------------------------

    def __len__(self) -> int:
        """
        method: __len__

        arguments:
         none

        return:
         the total number of spans in this trace
        """
        return len(self._rows())
    #
    # end of method

    def __iter__(self) -> Iterator[SpanView]:
        """
        method: __iter__

        arguments:
         none

        return:
         an iterator of SpanView objects

        description:
         Yields one SpanView per span in sequence order.
        """

        # yield a SpanView for each raw row
        #
        for row in self._rows():
            yield SpanView(row, self)
    #
    # end of method

    def filter(
        self,
        status: Optional[str] = None,
        backend_id: Optional[str] = None,
    ) -> list[SpanView]:
        """
        method: filter

        arguments:
         status:     optional status string to match (e.g. 'COMPLETED')
         backend_id: optional backend ID string to match

        return:
         list of SpanViews matching all provided criteria

        description:
         Returns all SpanViews if no criteria are provided. Multiple
         criteria are ANDed together.
        """

        # build initial list of all span views
        #
        views = [SpanView(r, self) for r in self._rows()]

        # apply status filter
        #
        if status is not None:
            views = [v for v in views if v.status == status]

        # apply backend_id filter
        #
        if backend_id is not None:
            views = [v for v in views if v.backend_id == backend_id]

        # exit gracefully
        #
        return views
    #
    # end of method

    def completed(self) -> list[SpanView]:
        """
        method: completed

        arguments:
         none

        return:
         list of SpanViews with status COMPLETED

        description:
         Convenience shortcut — the usual input to analyzers.
        """

        # exit gracefully
        #
        return self.filter(status="COMPLETED")
    #
    # end of method

    @property
    def spans(self):
        """
        method: spans

        arguments:
         none

        return:
         a pandas DataFrame with one scalar row per span

        description:
         Heavy nested data (events, inline artifact contents) is
         summarised by count. Use iteration / SpanView for resolved
         physical data. Requires pandas; raises ImportError if absent.
        """

        # load and cache on first access
        #
        if self._df is None:
            try:
                import pandas as pd
            except ImportError as e:
                raise ImportError(
                    "HilbertTrace.spans requires pandas. "
                    "Install with: pip install pandas"
                ) from e

            # build a flat record per span
            #
            records = []
            for r in self._rows():
                ia = r.get("inline_artifacts")
                if isinstance(ia, str):
                    ia = json.loads(ia) if ia else {}
                records.append({
                    "span_id":           r.get("span_id"),
                    "sequence_number":   r.get("sequence_number"),
                    "status":            r.get("status"),
                    "backend_id":        r.get("backend_id"),
                    "parent_span_id":    r.get("parent_span_id"),
                    "timestamp_start":   r.get("timestamp_start"),
                    "payload_ref":       r.get("payload_ref"),
                    "outcome_ref":       r.get("outcome_ref"),
                    "n_events":          len(r.get("events") or []),
                    "n_inline_artifacts": len(ia or {}),
                })
            self._df = pd.DataFrame.from_records(records)
        return self._df
    #
    # end of method

    # ---- artifact resolution ------------------------------------------------

    def _resolve_ref(self, row: dict, ref: Optional[str]) -> Any:
        """
        method: _resolve_ref

        arguments:
         row: the raw span row dict (used to check inline store)
         ref: the artifact reference string ('sha256:...')

        return:
         the resolved Python value, or None if empty / unresolvable

        description:
         Checks the span's inline store first, then the content-
         addressed file store. Decodes numpy, QASM text, JSON, or
         returns raw bytes depending on the encoding label.
        """

        # return None for empty or missing refs
        #
        if not ref:
            return None

        # check inline store (embedded in the span record)
        #
        ia = row.get("inline_artifacts")
        if isinstance(ia, str):
            ia = json.loads(ia) if ia else {}
        if ia and ref in ia:
            art = ia[ref]
            data = art.get("data")
            if art.get("encoding") == "json":
                return json.loads(data)
            return data

        # check file store (content-addressed, in the catalog)
        #
        if ref in self.catalog:
            meta = self.catalog[ref]
            path = self.run_dir / meta["file_path"]
            enc = meta.get("encoding")
            if enc == "numpy_binary":
                return np.load(path, allow_pickle=True)
            if enc in ("openqasm", "plaintext"):
                return path.read_text(encoding="utf-8")
            if enc == "json":
                return json.loads(path.read_text(encoding="utf-8"))
            return path.read_bytes()

        # exit gracefully — reference did not resolve
        #
        return None
    #
    # end of method

    # ---- bulk physical data -------------------------------------------------

    def outcomes(self) -> list:
        """
        method: outcomes

        arguments:
         none

        return:
         list of raw resolved outcomes for every completed span that
         has an outcome_ref

        description:
         Returns None-free list; spans without an outcome are skipped.
        """

        # collect resolved outcomes from all completed spans
        #
        out = []
        for v in self.completed():
            o = v.outcome
            if o is not None:
                out.append(o)

        # exit gracefully
        #
        return out
    #
    # end of method

    def numeric_outcomes(self) -> np.ndarray:
        """
        method: numeric_outcomes

        arguments:
         none

        return:
         every scalar/array outcome value, flattened into a 1-D float
         numpy array

        description:
         Skips non-numeric outcomes (e.g. Sampler counts dicts). Ideal
         input for variance / barren-plateau statistics.
        """

        # collect and flatten all numeric outcome values
        #
        vals: list[float] = []
        for v in self.completed():
            _flatten_numeric(v.outcome, vals)

        # exit gracefully
        #
        return np.asarray(vals, dtype=float)
    #
    # end of method

    def parameters(self) -> list:
        """
        method: parameters

        arguments:
         none

        return:
         resolved parameter vector per completed span (None where
         unrecorded)
        """
        return [v.parameters for v in self.completed()]
    #
    # end of method

    def observables(self) -> list:
        """
        method: observables

        arguments:
         none

        return:
         resolved observables per completed span (None where unrecorded)
        """
        return [v.observables for v in self.completed()]
    #
    # end of method

    def calibration(self) -> Optional[dict]:
        """
        method: calibration

        arguments:
         none

        return:
         the device calibration snapshot (T1/T2/readout/gate errors),
         or None if no calibration artifact is present

        description:
         Searches the catalog for a 'calibration_snapshot' artifact and
         resolves its content. Ideal sims produce no calibration.
        """

        # search the catalog for a calibration_snapshot artifact
        #
        for ref, meta in self.catalog.items():
            if meta.get("kind") == "calibration_snapshot":
                return self._resolve_ref({}, ref)
        return None
    #
    # end of method

    # ---- integrity ----------------------------------------------------------

    def verify(self) -> bool:
        """
        method: verify

        arguments:
         none

        return:
         True if the trace passes full cryptographic + causal checks

        description:
         Delegates to verify_trace_directory from the reader module.
         Raises TraceValidationError on any violation.
        """

        # import here to avoid a circular dep at module load time
        #
        from hilbertbench.reader.verify import verify_trace_directory

        # exit gracefully
        #
        return verify_trace_directory(self.run_dir)
    #
    # end of method

    def __repr__(self) -> str:
        """
        method: __repr__

        arguments:
         none

        return:
         a compact string representation of this HilbertTrace
        """
        return (
            f"HilbertTrace({self.run_dir.name!r}, "
            f"status={self.status!r}, "
            f"spans={len(self)})"
        )
    #
    # end of method
#
# end of class

#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------

def _flatten_numeric(value: Any, out: list) -> None:
    """
    function: _flatten_numeric

    arguments:
     value: the value to inspect (any type)
     out:   the list to append float leaves to

    return:
     none (mutates out in place)

    description:
     Recursively appends numeric leaves of value to out. Dicts, strings,
     bytes, None, and booleans are skipped. Handles scalars, numpy
     arrays, lists, and tuples.
    """

    # skip None, dicts, strings, and bytes
    #
    if value is None or isinstance(value, (dict, str, bytes)):
        return

    # append plain numeric scalars (exclude bool which is a subclass of int)
    #
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        out.append(float(value))
        return

    # flatten numpy arrays, skipping complex-valued elements
    #
    if isinstance(value, np.ndarray):
        for x in value.ravel():
            if np.isreal(x):
                out.append(float(np.real(x)))
        return

    # recurse into lists and tuples
    #
    if isinstance(value, (list, tuple)):
        for item in value:
            _flatten_numeric(item, out)
#
# end of function

#
# end of file
