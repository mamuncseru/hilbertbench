"""
hilbertbench/recorder/storage/writer.py

Converts append-only JSONL traces into highly compressed, columnar Parquet files
for fast offline analysis (e.g., querying millions of parameter updates).
"""
import json
from pathlib import Path
from typing import Dict, Any, List

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None


class ParquetConversionError(Exception):
    pass


def convert_trace_to_parquet(run_dir: Path | str) -> Path:
    """
    Reads events.jsonl and writes events.parquet in the same directory.
    Preserves original JSONL as the immutable source of truth (INV-002).
    """
    if pa is None or pq is None:
        raise ImportError(
            "PyArrow is required for Parquet storage. "
            "Install it with: pip install 'hilbertbench[storage]'"
        )

    run_path = Path(run_dir)
    jsonl_path = run_path / "events.jsonl"
    parquet_path = run_path / "events.parquet"

    if not jsonl_path.exists():
        raise FileNotFoundError(f"Source events.jsonl not found in {run_path}")

    # Columnar accumulators
    columns: Dict[str, list] = {
        "span_id": [],
        "trace_id": [],
        "parent_span_id": [],
        "sequence_number": [],
        "timestamp_start": [],
        "status": [],
        "backend_id": [],
        "payload_ref": [],
        "outcome_ref": [],
        "events": [],
        "tags": []
    }

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            span = json.loads(line)
            
            columns["span_id"].append(span.get("span_id"))
            columns["trace_id"].append(span.get("trace_id"))
            columns["parent_span_id"].append(span.get("parent_span_id"))
            columns["sequence_number"].append(span.get("sequence_number"))
            columns["timestamp_start"].append(span.get("timestamp_start"))
            columns["status"].append(span.get("status"))
            columns["backend_id"].append(span.get("backend_id"))
            columns["payload_ref"].append(span.get("payload_ref"))
            columns["outcome_ref"].append(span.get("outcome_ref"))
            
            # Process events
            processed_events = []
            for ev in span.get("events", []):
                # PyArrow requires uniform struct schemas. Because 'attributes' 
                # can hold arbitrarily nested dicts with different keys depending 
                # on the framework, we safely serialize it to a JSON string.
                attrs = ev.get("attributes")
                attr_str = json.dumps(attrs) if attrs is not None else None
                
                processed_events.append({
                    "event_id": ev.get("event_id"),
                    "event_type": ev.get("event_type"),
                    "timestamp": ev.get("timestamp"),
                    "error_ref": ev.get("error_ref"),
                    "attributes": attr_str
                })
            columns["events"].append(processed_events)
            
            # Process tags into PyArrow map format: [(key, val), ...]
            tags = span.get("tags")
            if tags:
                columns["tags"].append([(k, str(v)) for k, v in tags.items()])
            else:
                columns["tags"].append(None)

    # Define schema explicitly to ensure consistency even with empty traces
    schema = pa.schema([
        ("span_id", pa.string()),
        ("trace_id", pa.string()),
        ("parent_span_id", pa.string()),
        ("sequence_number", pa.int64()),
        ("timestamp_start", pa.int64()),
        ("status", pa.string()),
        ("backend_id", pa.string()),
        ("payload_ref", pa.string()),
        ("outcome_ref", pa.string()),
        ("events", pa.list_(pa.struct([
            ("event_id", pa.string()),
            ("event_type", pa.string()),
            ("timestamp", pa.int64()),
            ("error_ref", pa.string()),
            ("attributes", pa.string())
        ]))),
        ("tags", pa.map_(pa.string(), pa.string()))
    ])

    try:
        table = pa.Table.from_pydict(columns, schema=schema)
        # ZSTD compression is optimal for analytical ML workloads
        pq.write_table(table, parquet_path, compression="ZSTD")
    except Exception as e:
        raise ParquetConversionError(f"Failed to write Parquet: {e}")

    return parquet_path