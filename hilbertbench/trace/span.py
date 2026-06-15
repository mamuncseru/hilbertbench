#!/usr/bin/env python
#
# file: hilbertbench/trace/span.py
#
# revision history:
#  20260604 (am): cleaned up to project coding standards
#
# SpanView — a read-only, ergonomic view of one recorded span. Physical
# artifacts (circuit, outcome, parameters, observables) are resolved lazily
# on attribute access, transparently handling inline and file-store storage.
#
# Users never construct SpanView directly; instances come from iterating
# a HilbertTrace.
#------------------------------------------------------------------------------

# future imports must come first
#
from __future__ import annotations

# import system modules
#
import json
import os
from typing import TYPE_CHECKING, Any, Optional, cast

# TYPE_CHECKING guard avoids a circular import at runtime
#
if TYPE_CHECKING:
    from hilbertbench.trace.trace import HilbertTrace

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

class SpanView:
    """
    Class: SpanView

    description:
     One span with its physical evidence resolved on demand. Wraps the
     raw row dict from the JSONL or Parquet source and delegates artifact
     resolution to the parent HilbertTrace instance.
    """

    def __init__(self, row: dict, trace: "HilbertTrace") -> None:
        """
        method: constructor

        arguments:
         row:   the raw span row dict from events.jsonl or Parquet
         trace: the parent HilbertTrace that owns the artifact store

        return:
         none

        description:
         Stores the raw row and a back-reference to the parent trace.
         No data is decoded at construction time.
        """

        # store the raw span row
        #
        self._row = row

        # store a back-reference to the parent trace
        #
        self._trace = trace
    #
    # end of method

    # ---- scalar metadata properties ----------------------------------------

    @property
    def span_id(self) -> Optional[str]:
        """
        method: span_id

        arguments:
         none

        return:
         the span UUID string, or None
        """
        return self._row.get("span_id")
    #
    # end of method

    @property
    def trace_id(self) -> Optional[str]:
        """
        method: trace_id

        arguments:
         none

        return:
         the trace UUID string, or None
        """
        return self._row.get("trace_id")
    #
    # end of method

    @property
    def parent_span_id(self) -> Optional[str]:
        """
        method: parent_span_id

        arguments:
         none

        return:
         the parent span UUID string, or None for root spans
        """
        return self._row.get("parent_span_id")
    #
    # end of method

    @property
    def sequence_number(self) -> Optional[int]:
        """
        method: sequence_number

        arguments:
         none

        return:
         the monotonic sequence number, or None
        """
        return self._row.get("sequence_number")
    #
    # end of method

    @property
    def timestamp_start(self) -> Optional[int]:
        """
        method: timestamp_start

        arguments:
         none

        return:
         the wall-clock start timestamp in nanoseconds, or None
        """
        return self._row.get("timestamp_start")
    #
    # end of method

    @property
    def status(self) -> Optional[str]:
        """
        method: status

        arguments:
         none

        return:
         the span status string (e.g. 'COMPLETED', 'FAILED'), or None
        """
        return self._row.get("status")
    #
    # end of method

    @property
    def backend_id(self) -> Optional[str]:
        """
        method: backend_id

        arguments:
         none

        return:
         the backend identifier string, or None
        """
        return self._row.get("backend_id")
    #
    # end of method

    @property
    def tags(self) -> Optional[dict]:
        """
        method: tags

        arguments:
         none

        return:
         the tags dict attached to this span, or None
        """
        return self._row.get("tags")
    #
    # end of method

    @property
    def events(self) -> list:
        """
        method: events

        arguments:
         none

        return:
         the list of event records attached to this span
        """

        # return an empty list if no events are recorded
        #
        evs = self._row.get("events")
        if evs is None:
            return []
        return list(evs)
    #
    # end of method

    # ---- resolved physical evidence properties ------------------------------

    @property
    def outcome(self) -> Any:
        """
        method: outcome

        arguments:
         none

        return:
         the execution outcome (expval float, probs/sample array, or
         counts dict), or None if no outcome was recorded
        """
        return self._trace._resolve_ref(
            self._row, self._row.get("outcome_ref")
        )
    #
    # end of method

    @property
    def circuit(self) -> Any:
        """
        method: circuit

        arguments:
         none

        return:
         the circuit referenced by payload_ref (QASM text or operations
         blob), or None if no payload was recorded
        """
        return self._trace._resolve_ref(
            self._row, self._row.get("payload_ref")
        )
    #
    # end of method

    @property
    def parameters(self) -> Any:
        """
        method: parameters

        arguments:
         none

        return:
         the bound parameter values for this execution, or None
        """
        return self._inline_by_kind("parameters")
    #
    # end of method

    @property
    def observables(self) -> Any:
        """
        method: observables

        arguments:
         none

        return:
         the measurement operators for this execution, or None
        """
        return self._inline_by_kind("observables")
    #
    # end of method

    # ---- non-property accessors ---------------------------------------------

    def inline_artifacts(self) -> dict:
        """
        method: inline_artifacts

        arguments:
         none

        return:
         the raw inline_artifacts map (decoded from JSON if Parquet)

        description:
         Returns the decoded inline artifact dict. When the trace is
         loaded from Parquet, inline_artifacts is stored as a JSON
         string and decoded here.
        """

        # fetch the raw value from the row
        #
        ia = self._row.get("inline_artifacts")
        if ia is None:
            return {}

        # decode JSON string when loaded from Parquet
        #
        if isinstance(ia, str):
            return cast("dict[Any, Any]", json.loads(ia)) if ia else {}
        return cast("dict[Any, Any]", ia)
    #
    # end of method

    def event_attributes(self, event_type: str) -> Optional[dict]:
        """
        method: event_attributes

        arguments:
         event_type: the event type label to search for

        return:
         the attributes dict of the first matching event, or None

        description:
         Searches this span's event list for the first event whose
         event_type matches and returns its attributes. Decodes JSON
         strings when loaded from Parquet.
        """

        # search events for the first matching event_type
        #
        for ev in self.events:
            if not isinstance(ev, dict):
                continue
            if ev.get("event_type") == event_type:
                attrs = ev.get("attributes")

                # decode JSON string when loaded from Parquet
                #
                if isinstance(attrs, str):
                    return cast("dict[Any, Any]", json.loads(attrs))
                return attrs
        return None
    #
    # end of method

    # ---- internal helpers ---------------------------------------------------

    def _inline_by_kind(self, kind: str) -> Any:
        """
        method: _inline_by_kind

        arguments:
         kind: the artifact kind label to search for

        return:
         the decoded artifact data, or None if not found

        description:
         Searches the inline artifact store for the first artifact
         whose kind matches. Decodes JSON-encoded data automatically.
        """

        # search inline artifacts for a matching kind
        #
        for art in self.inline_artifacts().values():
            if art.get("kind") == kind:
                data = art.get("data")

                # decode JSON-encoded data
                #
                if art.get("encoding") == "json":
                    return json.loads(data)
                return data
        return None
    #
    # end of method

    def __repr__(self) -> str:
        """
        method: __repr__

        arguments:
         none

        return:
         a compact string representation of this SpanView
        """
        return (
            f"SpanView(seq={self.sequence_number}, "
            f"status={self.status}, "
            f"backend={self.backend_id!r})"
        )
    #
    # end of method
#
# end of class

#
# end of file
