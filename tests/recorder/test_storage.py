"""
tests/recorder/test_storage.py

Verifies the PyArrow Parquet conversion engine.
Ensures columnar arrays maintain strict integrity against the JSON schema.
"""
import json
from pathlib import Path
import pytest

try:
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

from hilbertbench.recorder.storage.writer import convert_trace_to_parquet, ParquetConversionError


@pytest.fixture
def dummy_run_dir(tmp_path: Path):
    """Creates a mock run directory with a valid events.jsonl file."""
    run_dir = tmp_path / "runs" / "test_run"
    run_dir.mkdir(parents=True)
    
    events_file = run_dir / "events.jsonl"
    
    dummy_span = {
        "hbspan_version": "1.0",
        "span_id": "00000000-0000-0000-0000-000000000001",
        "trace_id": "00000000-0000-0000-0000-000000000000",
        "parent_span_id": None,
        "sequence_number": 0,
        "timestamp_start": 1711234567000000000,
        "status": "COMPLETED",
        "backend_id": "ibm_kyiv",
        "payload_ref": "sha256:abc123payload",
        "outcome_ref": "sha256:def456outcome",
        "events": [
            {
                "event_id": "11111111-1111-1111-1111-111111111111",
                "event_type": "EXECUTION_REQUEST",
                "timestamp": 1711234567000000001,
                "error_ref": None,
                "attributes": {"shots": 1024, "framework": "qiskit"}
            }
        ],
        "tags": {"experiment": "vqe_ansatz_sweep"}
    }
    
    with open(events_file, "w") as f:
        f.write(json.dumps(dummy_span) + "\n")
        
    return run_dir


@pytest.mark.skipif(not HAS_PYARROW, reason="PyArrow not installed")
def test_parquet_conversion_creates_file(dummy_run_dir):
    parquet_path = convert_trace_to_parquet(dummy_run_dir)
    
    assert parquet_path.exists()
    assert parquet_path.suffix == ".parquet"


@pytest.mark.skipif(not HAS_PYARROW, reason="PyArrow not installed")
def test_parquet_schema_and_data_integrity(dummy_run_dir):
    parquet_path = convert_trace_to_parquet(dummy_run_dir)
    
    # Read it back into memory to verify column types
    table = pq.read_table(parquet_path)
    
    assert table.num_rows == 1
    assert "span_id" in table.column_names
    assert "events" in table.column_names
    
    # Convert first row to a standard python dict to verify complex structures
    first_row = table.to_pylist()[0]
    
    assert first_row["span_id"] == "00000000-0000-0000-0000-000000000001"
    assert first_row["sequence_number"] == 0
    assert first_row["payload_ref"] == "sha256:abc123payload"
    
    # Verify the complex events struct and JSON stringification
    event = first_row["events"][0]
    assert event["event_type"] == "EXECUTION_REQUEST"
    
    # The attributes dictionary should be correctly deserializable from JSON
    attrs = json.loads(event["attributes"])
    assert attrs["shots"] == 1024
    assert attrs["framework"] == "qiskit"


def test_missing_jsonl_raises_error(tmp_path):
    empty_dir = tmp_path / "empty_run"
    empty_dir.mkdir()
    
    with pytest.raises(FileNotFoundError, match="Source events.jsonl not found"):
        convert_trace_to_parquet(empty_dir)