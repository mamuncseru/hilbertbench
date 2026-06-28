"""
tests/recorder/test_inline_artifacts.py

Tier 1 unit tests for the two-tier storage system:
  - SpanHandle.attach_inline() correctness
  - Hash integrity (key == sha256 of data)
  - Routing enforcement (structural kinds rejected inline)
  - Inline data appears in JSONL, not in the file store
  - Parquet writer preserves inline_artifacts column

All I/O is isolated to tmp_path. No quantum execution.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from hilbertbench.models import Encoding, Kind, InlineArtifact
from hilbertbench.recorder.tape import HilbertTape, TapeClosedError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_of(s: str) -> str:
    return "sha256:" + hashlib.sha256(s.encode()).hexdigest()


def read_spans(tape: HilbertTape) -> list[dict]:
    lines = (tape.dir_path / "events.jsonl").read_text().splitlines()
    return [json.loads(l) for l in lines if l.strip()]


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    d = tmp_path / "runs"
    d.mkdir()
    return d


@pytest.fixture
def dummy_hash() -> str:
    return "sha256:" + "a" * 64


# ---------------------------------------------------------------------------
# 1. attach_inline basics
# ---------------------------------------------------------------------------

class TestAttachInlineBasics:

    def test_returns_sha256_hash(self, runs, dummy_hash):
        with HilbertTape(runs) as tape:
            with tape.execution_span(payload_ref=dummy_hash) as span:
                ref = span.attach_inline("0.5", kind="execution_outcome", encoding="json")
        assert ref.startswith("sha256:")
        assert len(ref) == len("sha256:") + 64

    def test_hash_matches_data(self, runs, dummy_hash):
        data = json.dumps([0.1, 0.2, 0.3])
        with HilbertTape(runs) as tape:
            with tape.execution_span(payload_ref=dummy_hash) as span:
                ref = span.attach_inline(data, kind="execution_outcome", encoding="json")
        assert ref == sha256_of(data)

    def test_size_bytes_matches_data(self, runs, dummy_hash):
        data = "hello"
        with HilbertTape(runs) as tape:
            with tape.execution_span(payload_ref=dummy_hash) as span:
                ref = span.attach_inline(data, kind="generic_blob", encoding="plaintext")
                art = span.inline_artifacts[ref]
        assert art.size_bytes == len(data.encode("utf-8"))

    def test_all_fields_present(self, runs, dummy_hash):
        with HilbertTape(runs) as tape:
            with tape.execution_span(payload_ref=dummy_hash) as span:
                ref = span.attach_inline(
                    "1.0", kind="execution_outcome", encoding="json", producer="test"
                )
                art = span.inline_artifacts[ref]
        assert art.kind.value == "execution_outcome"
        assert art.encoding.value == "json"
        assert art.data == "1.0"
        assert art.producer == "test"
        assert art.created_at > 0

    def test_same_data_same_hash_idempotent(self, runs, dummy_hash):
        data = json.dumps({"foo": 1})
        with HilbertTape(runs) as tape:
            with tape.execution_span(payload_ref=dummy_hash) as span:
                ref1 = span.attach_inline(data, kind="execution_outcome", encoding="json")
                ref2 = span.attach_inline(data, kind="execution_outcome", encoding="json")
        assert ref1 == ref2
        # dict is keyed by hash — still just one entry
        assert len(span.inline_artifacts) == 1

    def test_raises_after_tape_closed(self, runs, dummy_hash):
        with HilbertTape(runs) as tape:
            with tape.execution_span(payload_ref=dummy_hash) as span:
                pass  # span still in scope, but tape is closing after this
        with pytest.raises(TapeClosedError):
            span.attach_inline("x", kind="execution_outcome", encoding="json")


# ---------------------------------------------------------------------------
# 2. Schema enforcement — structural kinds rejected inline
# ---------------------------------------------------------------------------

class TestInlineKindEnforcement:

    def test_circuit_qasm_rejected(self, runs, dummy_hash):
        from pydantic import ValidationError
        with HilbertTape(runs) as tape:
            with tape.execution_span(payload_ref=dummy_hash) as span:
                with pytest.raises(ValidationError):
                    span.attach_inline("OPENQASM 3.0;", kind="circuit_qasm", encoding="plaintext")

    def test_calibration_snapshot_rejected(self, runs, dummy_hash):
        from pydantic import ValidationError
        with HilbertTape(runs) as tape:
            with tape.execution_span(payload_ref=dummy_hash) as span:
                with pytest.raises(ValidationError):
                    span.attach_inline("{}", kind="calibration_snapshot", encoding="json")

    def test_execution_outcome_allowed(self, runs, dummy_hash):
        with HilbertTape(runs) as tape:
            with tape.execution_span(payload_ref=dummy_hash) as span:
                ref = span.attach_inline("0.9", kind="execution_outcome", encoding="json")
        assert ref.startswith("sha256:")

    def test_parameters_allowed(self, runs, dummy_hash):
        with HilbertTape(runs) as tape:
            with tape.execution_span(payload_ref=dummy_hash) as span:
                ref = span.attach_inline("[0.1, 0.2]", kind="parameters", encoding="json")
        assert ref.startswith("sha256:")

    def test_observables_allowed(self, runs, dummy_hash):
        with HilbertTape(runs) as tape:
            with tape.execution_span(payload_ref=dummy_hash) as span:
                ref = span.attach_inline('["ZZ"]', kind="observables", encoding="json")
        assert ref.startswith("sha256:")

    def test_generic_blob_allowed(self, runs, dummy_hash):
        with HilbertTape(runs) as tape:
            with tape.execution_span(payload_ref=dummy_hash) as span:
                ref = span.attach_inline("some text", kind="generic_blob", encoding="plaintext")
        assert ref.startswith("sha256:")


# ---------------------------------------------------------------------------
# 3. Storage routing — inline stays out of the file store
# ---------------------------------------------------------------------------

class TestStorageRouting:

    def test_inline_artifact_not_written_to_disk(self, runs, dummy_hash):
        with HilbertTape(runs) as tape:
            with tape.execution_span(payload_ref=dummy_hash) as span:
                span.attach_inline("0.42", kind="execution_outcome", encoding="json")
        # artifacts directory must be empty (no files, not even shard dirs with files)
        artifact_files = list((tape.dir_path / "artifacts").rglob("*"))
        file_count = sum(1 for p in artifact_files if p.is_file())
        assert file_count == 0

    def test_inline_artifact_not_in_catalog(self, runs, dummy_hash):
        with HilbertTape(runs) as tape:
            with tape.execution_span(payload_ref=dummy_hash) as span:
                span.attach_inline("0.42", kind="execution_outcome", encoding="json")
        catalog = json.loads((tape.dir_path / "catalog.json").read_text())
        assert len(catalog["artifacts"]) == 0

    def test_inline_appears_in_jsonl(self, runs, dummy_hash):
        data = json.dumps([1.0, 2.0])
        with HilbertTape(runs) as tape:
            with tape.execution_span(payload_ref=dummy_hash) as span:
                ref = span.attach_inline(data, kind="execution_outcome", encoding="json")
        spans = read_spans(tape)
        assert "inline_artifacts" in spans[0]
        assert ref in spans[0]["inline_artifacts"]
        assert spans[0]["inline_artifacts"][ref]["data"] == data

    def test_outcome_ref_resolves_from_inline(self, runs, dummy_hash):
        data = "0.77"
        with HilbertTape(runs) as tape:
            with tape.execution_span(payload_ref=dummy_hash) as span:
                span.outcome_ref = span.attach_inline(
                    data, kind="execution_outcome", encoding="json"
                )
        spans = read_spans(tape)
        s = spans[0]
        assert s["outcome_ref"] in s["inline_artifacts"]
        assert s["inline_artifacts"][s["outcome_ref"]]["data"] == data

    def test_structural_artifact_still_uses_file_store(self, runs, tmp_path):
        qasm_file = tmp_path / "circuit.qasm"
        qasm_file.write_text("OPENQASM 3.0; qubit[1] q;")
        with HilbertTape(runs) as tape:
            ref = tape.attach_artifact(qasm_file, kind=Kind.circuit_qasm, encoding=Encoding.openqasm)
            with tape.execution_span(payload_ref=ref) as span:
                pass
        catalog = json.loads((tape.dir_path / "catalog.json").read_text())
        assert ref in catalog["artifacts"]
        hash_hex = ref.replace("sha256:", "")
        shard_dir = tape.dir_path / "artifacts" / hash_hex[:2]
        assert shard_dir.is_dir()
        assert any(f.stem == hash_hex for f in shard_dir.iterdir())


# ---------------------------------------------------------------------------
# 4. Parquet writer preserves inline_artifacts
# ---------------------------------------------------------------------------

class TestParquetWriterInline:

    def test_inline_artifacts_column_written(self, runs, dummy_hash):
        pytest.importorskip("pyarrow")
        import pyarrow.parquet as pq
        from hilbertbench.recorder.storage.writer import convert_trace_to_parquet

        data = json.dumps([0.5, -0.5])
        with HilbertTape(runs) as tape:
            with tape.execution_span(payload_ref=dummy_hash) as span:
                span.attach_inline(data, kind="execution_outcome", encoding="json")

        parquet_path = convert_trace_to_parquet(tape.dir_path)
        table = pq.read_table(parquet_path)
        assert "inline_artifacts" in table.schema.names

    def test_inline_artifacts_data_round_trips(self, runs, dummy_hash):
        pytest.importorskip("pyarrow")
        import pyarrow.parquet as pq
        from hilbertbench.recorder.storage.writer import convert_trace_to_parquet

        data = json.dumps([1.0, 2.0, 3.0])
        with HilbertTape(runs) as tape:
            with tape.execution_span(payload_ref=dummy_hash) as span:
                ref = span.attach_inline(data, kind="execution_outcome", encoding="json")

        convert_trace_to_parquet(tape.dir_path)
        df = pq.read_table(tape.dir_path / "events.parquet").to_pandas()
        row = df.iloc[0]
        inline = json.loads(row["inline_artifacts"])
        assert ref in inline
        assert inline[ref]["data"] == data
        assert inline[ref]["kind"] == "execution_outcome"

    def test_spans_without_inline_have_null_column(self, runs, tmp_path):
        pytest.importorskip("pyarrow")
        import pyarrow.parquet as pq
        from hilbertbench.recorder.storage.writer import convert_trace_to_parquet

        qasm_file = tmp_path / "c.qasm"
        qasm_file.write_text("OPENQASM 3.0; qubit[1] q;")
        with HilbertTape(runs) as tape:
            ref = tape.attach_artifact(qasm_file, kind=Kind.circuit_qasm, encoding=Encoding.openqasm)
            with tape.execution_span(payload_ref=ref):
                pass  # no inline artifacts on this span

        convert_trace_to_parquet(tape.dir_path)
        df = pq.read_table(tape.dir_path / "events.parquet").to_pandas()
        # a null cell reads back as None or NaN depending on pandas version
        import pandas as pd
        assert pd.isna(df.iloc[0]["inline_artifacts"])

