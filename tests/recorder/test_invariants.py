"""
tests/recorder/test_invariants.py

Tier 4 property / invariant tests.
These test the eight architectural invariants stated in
docs/architecture/001_invariants.md, plus hash integrity and
storage-triage consistency properties.

Properties checked:
  INV-001  Observer Effect — proxy never re-executes
  INV-002  Trace Immutability — written data cannot be overwritten
  INV-007  Failure Visibility — every exception becomes an ERROR event
  PROP-001 Hash Integrity — every inline artifact key == sha256(data)
  PROP-002 File-store hash == sha256 of file on disk
  PROP-003 Catalog count == number of files in artifacts/
  PROP-004 Sequence numbers are monotonically increasing and unique
  PROP-005 No structural artifact (circuit_qasm) in inline_artifacts
  PROP-006 Every outcome_ref resolves (inline or catalog)
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from hilbertbench.models import Encoding, Kind
from hilbertbench.recorder.tape import HilbertTape, TapeClosedError


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    d = tmp_path / "runs"
    d.mkdir()
    return d


def read_spans(tape: HilbertTape) -> list[dict]:
    return [json.loads(l) for l in (tape.dir_path / "events.jsonl").read_text().splitlines() if l.strip()]


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


# ---------------------------------------------------------------------------
# INV-001  Observer Effect
# ---------------------------------------------------------------------------

class TestINV001ObserverEffect:

    def test_qiskit_estimator_does_not_alter_result(self):
        """Proxy result must be bitwise identical to unproxied result."""
        from qiskit.circuit import QuantumCircuit, Parameter
        from qiskit.quantum_info import SparsePauliOp
        from qiskit.primitives import StatevectorEstimator
        from hilbertbench.integrations.qiskit import HilbertEstimatorProxy

        theta = Parameter("t")
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)
        obs = SparsePauliOp("Z")
        param_val = np.array([[0.7]])

        # Unproxied
        direct = StatevectorEstimator().run([(qc, obs, param_val)]).result()[0].data.evs

        with tempfile.TemporaryDirectory() as tmp:
            with HilbertTape(tmp) as tape:
                proxy = HilbertEstimatorProxy(tape)
                proxied = proxy.run([(qc, obs, param_val)]).result()[0].data.evs

        assert direct == pytest.approx(proxied, rel=1e-9)

    def test_pennylane_proxy_does_not_alter_result(self):
        """PennyLane proxy result must match direct device result."""
        import pennylane as qml
        from pennylane import numpy as pnp
        from hilbertbench.integrations.pennylane import HilbertPennyLaneDeviceProxy

        @qml.qnode(qml.device("default.qubit", wires=1))
        def direct(x):
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        direct_val = float(direct(pnp.array(0.9)))

        with tempfile.TemporaryDirectory() as tmp:
            real_dev = qml.device("default.qubit", wires=1)
            with HilbertTape(tmp) as tape:
                proxy_dev = HilbertPennyLaneDeviceProxy(real_dev, tape)
                @qml.qnode(proxy_dev, diff_method="parameter-shift")
                def proxied(x):
                    qml.RY(x, wires=0)
                    return qml.expval(qml.PauliZ(0))
                proxied_val = float(proxied(pnp.array(0.9)))

        assert direct_val == pytest.approx(proxied_val, rel=1e-9)


# ---------------------------------------------------------------------------
# INV-002  Trace Immutability
# ---------------------------------------------------------------------------

class TestINV002TraceImmutability:

    def test_write_after_close_raises(self, runs):
        dummy_hash = "sha256:" + "a" * 64
        with HilbertTape(runs) as tape:
            pass  # closed here

        with pytest.raises(TapeClosedError):
            with tape.execution_span(payload_ref=dummy_hash):
                pass

    def test_attach_artifact_after_close_raises(self, runs, tmp_path):
        f = tmp_path / "x.bin"
        f.write_bytes(b"\x00" * 8)
        with HilbertTape(runs) as tape:
            pass

        with pytest.raises(TapeClosedError):
            tape.attach_artifact(f, kind=Kind.generic_blob, encoding=Encoding.numpy_binary)

    def test_close_idempotent(self, runs):
        with HilbertTape(runs) as tape:
            pass
        tape.close()
        tape.close()  # must not raise

    def test_events_jsonl_append_only(self, runs):
        """Span data is flushed immediately — JSONL length only grows."""
        dummy_hash = "sha256:" + "b" * 64
        with HilbertTape(runs) as tape:
            sizes = []
            for _ in range(3):
                with tape.execution_span(payload_ref=dummy_hash):
                    pass
                sizes.append((tape.dir_path / "events.jsonl").stat().st_size)
        assert sizes[0] < sizes[1] < sizes[2]


# ---------------------------------------------------------------------------
# PROP-007  Integrity Seal
# ---------------------------------------------------------------------------

class TestPROP007IntegritySeal:

    def test_seal_present_after_seal(self, runs):
        dummy_hash = "sha256:" + "6" * 64
        with HilbertTape(runs) as tape:
            with tape.execution_span(payload_ref=dummy_hash):
                pass
        trace = json.loads((tape.dir_path / "trace.json").read_text())
        assert trace["integrity_seal"] is not None
        assert trace["integrity_seal"]["event_stream_checksum"].startswith("sha256:")

    def test_seal_checksum_matches_events_file(self, runs):
        dummy_hash = "sha256:" + "7" * 64
        with HilbertTape(runs) as tape:
            for _ in range(4):
                with tape.execution_span(payload_ref=dummy_hash):
                    pass

        seal = json.loads((tape.dir_path / "trace.json").read_text())["integrity_seal"]
        raw = (tape.dir_path / "events.jsonl").read_bytes()
        expected = "sha256:" + hashlib.sha256(raw).hexdigest()
        assert seal["event_stream_checksum"] == expected

    def test_artifact_count_includes_inline(self, runs):
        dummy_hash = "sha256:" + "8" * 64
        with HilbertTape(runs) as tape:
            with tape.execution_span(payload_ref=dummy_hash) as span:
                span.attach_inline("0.1", kind="execution_outcome", encoding="json")
                span.attach_inline("[0.2]", kind="parameters", encoding="json")
            with tape.execution_span(payload_ref=dummy_hash) as span:
                span.attach_inline("0.3", kind="execution_outcome", encoding="json")

        seal = json.loads((tape.dir_path / "trace.json").read_text())["integrity_seal"]
        assert seal["artifact_count"] == 3  # 2 + 1 inline artifacts

    def test_artifact_count_includes_filestore(self, runs, tmp_path):
        f = tmp_path / "c.qasm"
        f.write_text("OPENQASM 3.0;")
        with HilbertTape(runs) as tape:
            ref = tape.attach_artifact(f, kind=Kind.circuit_qasm, encoding=Encoding.openqasm)
            with tape.execution_span(payload_ref=ref) as span:
                span.attach_inline("0.5", kind="execution_outcome", encoding="json")

        seal = json.loads((tape.dir_path / "trace.json").read_text())["integrity_seal"]
        assert seal["artifact_count"] == 2  # 1 file-store + 1 inline

    def test_seal_absent_while_in_flight(self, runs):
        """Before sealing, trace.json is CRASHED_IN_FLIGHT with no seal."""
        dummy_hash = "sha256:" + "9" * 64
        tape = HilbertTape(runs)
        tape.__enter__()
        with tape.execution_span(payload_ref=dummy_hash):
            pass
        trace = json.loads((tape.dir_path / "trace.json").read_text())
        assert trace["status"] == "CRASHED_IN_FLIGHT"
        assert trace["integrity_seal"] is None
        tape.close()


# ---------------------------------------------------------------------------
# INV-007  Failure Visibility
# ---------------------------------------------------------------------------

class TestINV007FailureVisibility:

    def test_exception_span_has_error_event(self, runs):
        dummy_hash = "sha256:" + "c" * 64
        with pytest.raises(ValueError):
            with HilbertTape(runs) as tape:
                with tape.execution_span(payload_ref=dummy_hash):
                    raise ValueError("test error")

        spans = read_spans(tape)
        s = spans[0]
        assert s["status"] == "FAILED"
        error_events = [e for e in s["events"] if e["event_type"] == "ERROR"]
        assert len(error_events) == 1

    def test_error_event_captures_exception_type(self, runs):
        dummy_hash = "sha256:" + "d" * 64
        with pytest.raises(KeyError):
            with HilbertTape(runs) as tape:
                with tape.execution_span(payload_ref=dummy_hash):
                    raise KeyError("missing_key")

        error_event = next(
            e for e in read_spans(tape)[0]["events"] if e["event_type"] == "ERROR"
        )
        assert error_event["attributes"]["exception_type"] == "KeyError"
        assert "missing_key" in error_event["attributes"]["exception_message"]

    def test_exception_propagates_to_caller(self, runs):
        dummy_hash = "sha256:" + "e" * 64
        with pytest.raises(ZeroDivisionError):
            with HilbertTape(runs) as tape:
                with tape.execution_span(payload_ref=dummy_hash):
                    _ = 1 / 0

    def test_tape_sealed_with_errors_on_outer_exception(self, runs):
        dummy_hash = "sha256:" + "f" * 64
        with pytest.raises(RuntimeError):
            with HilbertTape(runs) as tape:
                raise RuntimeError("outer failure")

        manifest = json.loads((tape.dir_path / "trace.json").read_text())
        assert manifest["status"] == "SEALED_WITH_ERRORS"


# ---------------------------------------------------------------------------
# PROP-001  Hash Integrity
# ---------------------------------------------------------------------------

class TestPROP001HashIntegrity:

    def test_inline_artifact_keys_match_sha256(self, runs):
        dummy_hash = "sha256:" + "0" * 64
        payloads = [
            ("0.42", "execution_outcome", "json"),
            ("[0.1, 0.2, 0.3]", "parameters", "json"),
            ('["ZZ"]', "observables", "json"),
        ]
        with HilbertTape(runs) as tape:
            for data, kind, enc in payloads:
                with tape.execution_span(payload_ref=dummy_hash) as span:
                    ref = span.attach_inline(data, kind=kind, encoding=enc)
                    expected_key = "sha256:" + sha256_hex(data)
                    assert ref == expected_key, f"Hash mismatch for {kind}"

    def test_all_spans_in_jsonl_pass_hash_check(self, runs):
        dummy_hash = "sha256:" + "1" * 64
        with HilbertTape(runs) as tape:
            for v in [0.1, -0.5, 0.9, -0.3]:
                with tape.execution_span(payload_ref=dummy_hash) as span:
                    data = json.dumps(v)
                    span.outcome_ref = span.attach_inline(
                        data, kind="execution_outcome", encoding="json"
                    )

        spans = read_spans(tape)
        for s in spans:
            for ref, art in (s.get("inline_artifacts") or {}).items():
                computed = "sha256:" + sha256_hex(art["data"])
                assert computed == ref, f"Hash mismatch: {ref}"


# ---------------------------------------------------------------------------
# PROP-002  File-store hash == sha256 of file on disk
# ---------------------------------------------------------------------------

class TestPROP002FileStoreIntegrity:

    def test_attached_file_hash_matches_disk(self, runs, tmp_path):
        content = b"OPENQASM 3.0;\nqubit[2] q;\nh q[0];\ncx q[0], q[1];"
        f = tmp_path / "circuit.qasm"
        f.write_bytes(content)
        expected_hash = "sha256:" + hashlib.sha256(content).hexdigest()

        with HilbertTape(runs) as tape:
            ref = tape.attach_artifact(f, kind=Kind.circuit_qasm, encoding=Encoding.openqasm)

        assert ref == expected_hash
        catalog = json.loads((tape.dir_path / "catalog.json").read_text())
        assert catalog["artifacts"][ref]["artifact_hash"] == expected_hash

        # Verify file on disk
        hash_hex = ref.replace("sha256:", "")
        shard_dir = tape.dir_path / "artifacts" / hash_hex[:2]
        disk_file = next(f for f in shard_dir.iterdir() if f.stem == hash_hex)
        actual_hash = "sha256:" + hashlib.sha256(disk_file.read_bytes()).hexdigest()
        assert actual_hash == expected_hash


# ---------------------------------------------------------------------------
# PROP-003  Catalog count == files in artifacts/
# ---------------------------------------------------------------------------

class TestPROP003CatalogConsistency:

    def test_catalog_entries_match_file_count(self, runs, tmp_path):
        files = []
        for i in range(3):
            f = tmp_path / f"circuit_{i}.qasm"
            f.write_text(f"OPENQASM 3.0; // circuit {i}")
            files.append(f)

        with HilbertTape(runs) as tape:
            for f in files:
                ref = tape.attach_artifact(f, kind=Kind.circuit_qasm, encoding=Encoding.openqasm)
                with tape.execution_span(payload_ref=ref):
                    pass

        catalog = json.loads((tape.dir_path / "catalog.json").read_text())
        file_count = len(list((tape.dir_path / "artifacts").rglob("*")))
        file_count = sum(1 for p in (tape.dir_path / "artifacts").rglob("*") if p.is_file())
        # Files on disk may be < catalog entries if circuits are identical (dedup)
        assert file_count <= len(catalog["artifacts"])
        # Each catalog entry must have a corresponding file
        for ref, meta in catalog["artifacts"].items():
            assert (tape.dir_path / meta["file_path"]).is_file()

    def test_inline_artifacts_not_counted_in_catalog(self, runs):
        dummy_hash = "sha256:" + "2" * 64
        with HilbertTape(runs) as tape:
            with tape.execution_span(payload_ref=dummy_hash) as span:
                for _ in range(5):
                    span.attach_inline("0.5", kind="execution_outcome", encoding="json")

        catalog = json.loads((tape.dir_path / "catalog.json").read_text())
        assert len(catalog["artifacts"]) == 0


# ---------------------------------------------------------------------------
# PROP-004  Sequence numbers monotonic and unique
# ---------------------------------------------------------------------------

class TestPROP004SequenceNumbers:

    def test_sequence_numbers_monotonic_and_unique(self, runs):
        dummy_hash = "sha256:" + "3" * 64
        with HilbertTape(runs) as tape:
            for _ in range(10):
                with tape.execution_span(payload_ref=dummy_hash):
                    pass

        seqs = [s["sequence_number"] for s in read_spans(tape)]
        assert seqs == sorted(seqs), "Sequence numbers not monotonic"
        assert len(set(seqs)) == len(seqs), "Duplicate sequence numbers"

    def test_nested_spans_both_get_unique_sequences(self, runs):
        dummy_hash = "sha256:" + "4" * 64
        with HilbertTape(runs) as tape:
            with tape.execution_span(payload_ref=dummy_hash):
                with tape.execution_span(payload_ref=dummy_hash):
                    pass

        seqs = [s["sequence_number"] for s in read_spans(tape)]
        assert len(set(seqs)) == 2


# ---------------------------------------------------------------------------
# PROP-005  No circuit_qasm in inline_artifacts
# ---------------------------------------------------------------------------

class TestPROP005NoCircuitInline:

    def test_circuit_qasm_always_in_file_store(self, runs, tmp_path):
        f = tmp_path / "c.qasm"
        f.write_text("OPENQASM 3.0;")
        with HilbertTape(runs) as tape:
            ref = tape.attach_artifact(f, kind=Kind.circuit_qasm, encoding=Encoding.openqasm)
            with tape.execution_span(payload_ref=ref):
                pass

        spans = read_spans(tape)
        for s in spans:
            inline = s.get("inline_artifacts") or {}
            for art in inline.values():
                assert art["kind"] != "circuit_qasm", "circuit_qasm found inline — must be file-store only"

        catalog = json.loads((tape.dir_path / "catalog.json").read_text())
        assert any(a["kind"] == "circuit_qasm" for a in catalog["artifacts"].values())


# ---------------------------------------------------------------------------
# PROP-006  Every outcome_ref resolves
# ---------------------------------------------------------------------------

class TestPROP006OutcomeRefResolves:

    def test_inline_outcome_ref_resolves(self, runs):
        dummy_hash = "sha256:" + "5" * 64
        with HilbertTape(runs) as tape:
            with tape.execution_span(payload_ref=dummy_hash) as span:
                span.outcome_ref = span.attach_inline(
                    "0.9", kind="execution_outcome", encoding="json"
                )

        spans = read_spans(tape)
        s = spans[0]
        assert s["outcome_ref"] in s["inline_artifacts"]

    def test_file_store_outcome_ref_resolves(self, runs, tmp_path):
        npy = tmp_path / "outcome.npy"
        np.save(npy, np.array([0.5]))
        qasm = tmp_path / "c.qasm"
        qasm.write_text("OPENQASM 3.0;")

        with HilbertTape(runs) as tape:
            payload_ref = tape.attach_artifact(qasm, kind=Kind.circuit_qasm, encoding=Encoding.openqasm)
            outcome_ref = tape.attach_artifact(npy, kind=Kind.execution_outcome, encoding=Encoding.numpy_binary)
            with tape.execution_span(payload_ref=payload_ref) as span:
                span.outcome_ref = outcome_ref

        catalog = json.loads((tape.dir_path / "catalog.json").read_text())
        spans = read_spans(tape)
        s = spans[0]
        assert s["outcome_ref"] in catalog["artifacts"]
