#!/usr/bin/env python
#
# file: hilbertbench/recorder/storage/writer.py
#
# revision history:
#  20260604 (am): cleaned up to project coding standards
#
# Converts append-only JSONL traces into columnar Parquet files for fast
# offline analysis (e.g., querying millions of parameter updates).
# The original JSONL is preserved as the immutable source of truth
# (INV-002).
#------------------------------------------------------------------------------

# import system modules
#
import json
from pathlib import Path
from typing import Dict, List

# import optional third-party modules
#
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None

#------------------------------------------------------------------------------
#
# classes are listed here
#
#------------------------------------------------------------------------------

class ParquetConversionError(Exception):
    """
    Class: ParquetConversionError

    description:
     Raised when a Parquet write operation fails.
    """
    pass
#
# end of class

#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------

def convert_trace_to_parquet(run_dir: Path | str) -> Path:
    """
    function: convert_trace_to_parquet

    arguments:
     run_dir: path to the run directory containing events.jsonl

    return:
     path to the written events.parquet file

    description:
     Reads events.jsonl and writes events.parquet in the same directory.
     The original JSONL is preserved as the immutable source of truth
     (INV-002). PyArrow ZSTD compression is used for analytical ML
     workloads. Raises ImportError if PyArrow is not installed, and
     ParquetConversionError if the write fails.
    """

    # ensure PyArrow is available
    #
    if pa is None or pq is None:
        raise ImportError(
            "PyArrow is required for Parquet storage. "
            "Install it with: pip install 'hilbertbench[storage]'"
        )

    # resolve source and destination paths
    #
    run_path = Path(run_dir)
    jsonl_path = run_path / "events.jsonl"
    parquet_path = run_path / "events.parquet"

    # verify source file exists
    #
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Source events.jsonl not found in {run_path}"
        )

    # initialise columnar accumulators
    #
    columns: Dict[str, list] = {
        "span_id":          [],
        "trace_id":         [],
        "parent_span_id":   [],
        "sequence_number":  [],
        "timestamp_start":  [],
        "status":           [],
        "backend_id":       [],
        "payload_ref":      [],
        "outcome_ref":      [],
        "events":           [],
        "tags":             [],
        "inline_artifacts": [],
    }

    # read and accumulate every span from the JSONL file
    #
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:

            # skip blank lines
            #
            line = line.strip()
            if not line:
                continue

            # parse the span record
            #
            span = json.loads(line)

            # accumulate scalar fields
            #
            columns["span_id"].append(span.get("span_id"))
            columns["trace_id"].append(span.get("trace_id"))
            columns["parent_span_id"].append(span.get("parent_span_id"))
            columns["sequence_number"].append(span.get("sequence_number"))
            columns["timestamp_start"].append(span.get("timestamp_start"))
            columns["status"].append(span.get("status"))
            columns["backend_id"].append(span.get("backend_id"))
            columns["payload_ref"].append(span.get("payload_ref"))
            columns["outcome_ref"].append(span.get("outcome_ref"))

            # process events: serialise 'attributes' to a JSON string
            # because PyArrow requires uniform struct schemas and event
            # attributes hold framework-dependent, arbitrarily nested dicts
            #
            processed_events = []
            for ev in span.get("events", []):
                attrs = ev.get("attributes")
                processed_events.append({
                    "event_id":   ev.get("event_id"),
                    "event_type": ev.get("event_type"),
                    "timestamp":  ev.get("timestamp"),
                    "error_ref":  ev.get("error_ref"),
                    "attributes": (
                        json.dumps(attrs) if attrs is not None else None
                    ),
                })
            columns["events"].append(processed_events)

            # process tags into PyArrow map format: [(key, val), ...]
            #
            tags = span.get("tags")
            if tags:
                columns["tags"].append(
                    [(k, str(v)) for k, v in tags.items()]
                )
            else:
                columns["tags"].append(None)

            # serialise inline_artifacts as a JSON string — same pattern as
            # event attributes, keeps data queryable via json.loads()
            #
            inline = span.get("inline_artifacts")
            columns["inline_artifacts"].append(
                json.dumps(inline) if inline else None
            )

    # define explicit schema for consistency even on empty traces
    #
    schema = pa.schema([
        ("span_id",          pa.string()),
        ("trace_id",         pa.string()),
        ("parent_span_id",   pa.string()),
        ("sequence_number",  pa.int64()),
        ("timestamp_start",  pa.int64()),
        ("status",           pa.string()),
        ("backend_id",       pa.string()),
        ("payload_ref",      pa.string()),
        ("outcome_ref",      pa.string()),
        ("events", pa.list_(pa.struct([
            ("event_id",   pa.string()),
            ("event_type", pa.string()),
            ("timestamp",  pa.int64()),
            ("error_ref",  pa.string()),
            ("attributes", pa.string()),
        ]))),
        ("tags",             pa.map_(pa.string(), pa.string())),
        ("inline_artifacts", pa.string()),
    ])

    # write the Parquet file using ZSTD compression
    #
    try:
        table = pa.Table.from_pydict(columns, schema=schema)
        pq.write_table(table, parquet_path, compression="ZSTD")
    except Exception as e:
        raise ParquetConversionError(f"Failed to write Parquet: {e}")

    # exit gracefully
    #
    return parquet_path
#
# end of function

#
# end of file
