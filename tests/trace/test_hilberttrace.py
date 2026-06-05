"""
tests/trace/test_hilberttrace.py

Tests for the HilbertTrace unified data API. Traces are built with the tape
directly (no quantum execution) so the resolution logic — inline vs file-store,
scalar vs array vs dict outcomes — is exercised deterministically.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hilbertbench.models import Encoding, Kind
from hilbertbench.recorder.tape import HilbertTape
from hilbertbench.trace import HilbertTrace, SpanView


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    d = tmp_path / "runs"
    d.mkdir()
    return d


# ── Builders ──────────────────────────────────────────────────────────────────

def build_inline_trace(runs, outcomes, params=None):
    """Trace with inline scalar outcomes + optional parameters per span."""
    dummy = "sha256:" + "a" * 64
    with HilbertTape(runs, tags={"task": "unit"}) as tape:
        for i, o in enumerate(outcomes):
            with tape.execution_span(payload_ref=dummy, backend_id="sim") as span:
                span.outcome_ref = span.attach_inline(
                    json.dumps(o), kind="execution_outcome", encoding="json"
                )
                if params is not None:
                    span.attach_inline(
                        json.dumps(params[i]), kind="parameters", encoding="json"
                    )
                    span.attach_inline(
                        json.dumps(["ZZ"]), kind="observables", encoding="json"
                    )
    return tape.dir_path


def build_filestore_trace(runs, tmp_path):
    """Trace with a file-store circuit_qasm payload and inline outcome."""
    qasm = tmp_path / "c.qasm"
    qasm.write_text("OPENQASM 3.0;\nqubit[2] q;\nh q[0];")
    with HilbertTape(runs) as tape:
        ref = tape.attach_artifact(qasm, kind=Kind.circuit_qasm, encoding=Encoding.openqasm)
        with tape.execution_span(payload_ref=ref, backend_id="sim") as span:
            span.outcome_ref = span.attach_inline(
                "0.5", kind="execution_outcome", encoding="json"
            )
    return tape.dir_path


# ── Construction ──────────────────────────────────────────────────────────────

class TestConstruction:

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            HilbertTrace(tmp_path / "nope")

    def test_directory_without_events_raises(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        with pytest.raises(FileNotFoundError, match="events.jsonl"):
            HilbertTrace(d)

    def test_repr(self, runs):
        run_dir = build_inline_trace(runs, [0.1, 0.2])
        t = HilbertTrace(run_dir)
        assert "spans=2" in repr(t)
        assert "SEALED_SUCCESS" in repr(t)


# ── Metadata ──────────────────────────────────────────────────────────────────

class TestMetadata:

    def test_status_mode_tags(self, runs):
        t = HilbertTrace(build_inline_trace(runs, [0.1]))
        assert t.status == "SEALED_SUCCESS"
        assert t.mode == "passive"
        assert t.tags == {"task": "unit"}

    def test_integrity_seal_present(self, runs):
        t = HilbertTrace(build_inline_trace(runs, [0.1, 0.2, 0.3]))
        seal = t.integrity_seal
        assert seal is not None
        assert seal["artifact_count"] == 3  # 3 inline outcomes

    def test_environment(self, runs):
        t = HilbertTrace(build_inline_trace(runs, [0.1]))
        assert "hilbertbench_version" in t.environment


# ── Span access ───────────────────────────────────────────────────────────────

class TestSpanAccess:

    def test_len_and_iteration(self, runs):
        t = HilbertTrace(build_inline_trace(runs, [0.1, 0.2, 0.3]))
        assert len(t) == 3
        views = list(t)
        assert all(isinstance(v, SpanView) for v in views)
        assert len(views) == 3

    def test_completed_filter(self, runs):
        t = HilbertTrace(build_inline_trace(runs, [0.1, 0.2]))
        assert len(t.completed()) == 2

    def test_filter_by_backend(self, runs):
        t = HilbertTrace(build_inline_trace(runs, [0.1, 0.2]))
        assert len(t.filter(backend_id="sim")) == 2
        assert len(t.filter(backend_id="nonexistent")) == 0

    def test_dataframe_view(self, runs):
        pytest.importorskip("pandas")
        t = HilbertTrace(build_inline_trace(runs, [0.1, 0.2, 0.3]))
        df = t.spans
        assert len(df) == 3
        assert "sequence_number" in df.columns
        assert "n_inline_artifacts" in df.columns
        assert df["status"].unique().tolist() == ["COMPLETED"]


# ── Resolution: inline ────────────────────────────────────────────────────────

class TestInlineResolution:

    def test_outcome_resolves(self, runs):
        t = HilbertTrace(build_inline_trace(runs, [0.42]))
        assert t.completed()[0].outcome == pytest.approx(0.42)

    def test_parameters_resolve(self, runs):
        t = HilbertTrace(build_inline_trace(runs, [0.1], params=[[1.0, 2.0, 3.0]]))
        assert t.completed()[0].parameters == [1.0, 2.0, 3.0]

    def test_observables_resolve(self, runs):
        t = HilbertTrace(build_inline_trace(runs, [0.1], params=[[1.0]]))
        assert t.completed()[0].observables == ["ZZ"]

    def test_missing_parameters_returns_none(self, runs):
        t = HilbertTrace(build_inline_trace(runs, [0.1]))  # no params
        assert t.completed()[0].parameters is None


# ── Resolution: file store ────────────────────────────────────────────────────

class TestFileStoreResolution:

    def test_circuit_resolves_from_file(self, runs, tmp_path):
        t = HilbertTrace(build_filestore_trace(runs, tmp_path))
        circuit = t.completed()[0].circuit
        assert circuit is not None
        assert "OPENQASM" in circuit

    def test_outcome_inline_circuit_filestore_same_span(self, runs, tmp_path):
        """A span may mix storage tiers: inline outcome + file-store circuit."""
        t = HilbertTrace(build_filestore_trace(runs, tmp_path))
        span = t.completed()[0]
        assert span.outcome == pytest.approx(0.5)         # inline
        assert "OPENQASM" in span.circuit                  # file store


# ── numeric_outcomes flattening ───────────────────────────────────────────────

class TestNumericOutcomes:

    def test_scalars(self, runs):
        t = HilbertTrace(build_inline_trace(runs, [0.1, 0.2, 0.3, 0.4]))
        no = t.numeric_outcomes()
        assert no.shape == (4,)
        assert no.tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4])

    def test_arrays_flattened(self, runs):
        t = HilbertTrace(build_inline_trace(runs, [[0.1, 0.2], [0.3, 0.4]]))
        no = t.numeric_outcomes()
        assert no.shape == (4,)

    def test_counts_dict_skipped(self, runs):
        """Sampler-style counts dicts are not numeric outcomes."""
        dummy = "sha256:" + "b" * 64
        with HilbertTape(runs) as tape:
            with tape.execution_span(payload_ref=dummy) as span:
                span.outcome_ref = span.attach_inline(
                    json.dumps({"meas": {"counts": {"00": 50, "11": 50}}}),
                    kind="execution_outcome", encoding="json",
                )
        t = HilbertTrace(tape.dir_path)
        assert t.numeric_outcomes().shape == (0,)

    def test_variance_matches_manual(self, runs):
        vals = [0.5, -0.3, 0.8, -0.6, 0.1]
        t = HilbertTrace(build_inline_trace(runs, vals))
        assert t.numeric_outcomes().var() == pytest.approx(np.var(vals))


# ── Calibration + verify ──────────────────────────────────────────────────────

class TestCalibrationAndVerify:

    def test_calibration_none_when_absent(self, runs):
        t = HilbertTrace(build_inline_trace(runs, [0.1]))
        assert t.calibration() is None

    def test_calibration_resolves_when_present(self, runs, tmp_path):
        cal = tmp_path / "cal.json"
        cal.write_text(json.dumps({"backend_name": "ibm_test", "qubits": [[{"name": "T1", "value": 1e-4}]]}))
        with HilbertTape(runs) as tape:
            tape.attach_artifact(cal, kind=Kind.calibration_snapshot, encoding=Encoding.json)
            dummy = "sha256:" + "c" * 64
            with tape.execution_span(payload_ref=dummy) as span:
                span.attach_inline("0.1", kind="execution_outcome", encoding="json")

        t = HilbertTrace(tape.dir_path)
        cal_data = t.calibration()
        assert cal_data is not None
        assert cal_data["backend_name"] == "ibm_test"

    def test_verify_passes_on_clean_trace(self, runs, tmp_path):
        t = HilbertTrace(build_filestore_trace(runs, tmp_path))
        assert t.verify() is True


# ── Lazy top-level import ─────────────────────────────────────────────────────

class TestLazyImport:

    def test_top_level_import(self):
        import hilbertbench
        assert hilbertbench.HilbertTrace is HilbertTrace

    def test_unknown_attribute_raises(self):
        import hilbertbench
        with pytest.raises(AttributeError):
            _ = hilbertbench.NoSuchThing
