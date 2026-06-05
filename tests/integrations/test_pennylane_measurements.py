"""
tests/integrations/test_pennylane_measurements.py

Tier 2 integration tests for HilbertPennyLaneDeviceProxy covering all common
PennyLane measurement types: expval, probs, counts, sample, state.
Also verifies exception handling and backend_id propagation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import pennylane as qml
from pennylane import numpy as pnp

from hilbertbench.integrations.pennylane import HilbertPennyLaneDeviceProxy
from hilbertbench.recorder.tape import HilbertTape


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    d = tmp_path / "runs"
    d.mkdir()
    return d


def read_spans(tape: HilbertTape) -> list[dict]:
    return [json.loads(l) for l in (tape.dir_path / "events.jsonl").read_text().splitlines() if l.strip()]


def make_proxy_qnode(measurement_fn, tape, shots=None):
    kwargs = {"shots": shots} if shots else {}
    dev = qml.device("default.qubit", wires=2, **kwargs)
    proxy = HilbertPennyLaneDeviceProxy(dev, tape)

    @qml.qnode(proxy, diff_method="parameter-shift")
    def circuit(x):
        qml.RY(x, wires=0)
        qml.CNOT(wires=[0, 1])
        return measurement_fn()

    return circuit


# ---------------------------------------------------------------------------
# 1. All measurement types serialize correctly
# ---------------------------------------------------------------------------

class TestMeasurementTypes:

    def test_expval_inline(self, runs):
        with HilbertTape(runs) as tape:
            qnode = make_proxy_qnode(lambda: qml.expval(qml.PauliZ(0)), tape)
            qnode(0.0)  # |0> state → expval(Z) = +1

        s = read_spans(tape)[0]
        ref = s["outcome_ref"]
        assert ref in s["inline_artifacts"]
        val = json.loads(s["inline_artifacts"][ref]["data"])
        assert pytest.approx(val, abs=0.01) == 1.0

    def test_probs_inline(self, runs):
        with HilbertTape(runs) as tape:
            qnode = make_proxy_qnode(lambda: qml.probs(wires=[0, 1]), tape)
            qnode(0.0)  # |00> state → probs = [1, 0, 0, 0]

        s = read_spans(tape)[0]
        probs = json.loads(s["inline_artifacts"][s["outcome_ref"]]["data"])
        assert len(probs) == 4
        assert pytest.approx(probs[0], abs=0.01) == 1.0

    def test_counts_inline(self, runs):
        with HilbertTape(runs) as tape:
            qnode = make_proxy_qnode(lambda: qml.counts(), tape, shots=128)
            qnode(0.0)  # deterministic |00> → all shots give "00"

        s = read_spans(tape)[0]
        counts = json.loads(s["inline_artifacts"][s["outcome_ref"]]["data"])
        # counts is a bitstring → int dict; "00" should dominate
        assert isinstance(counts, dict)
        total = sum(int(v) for v in counts.values())
        assert total == 128
        assert counts.get("00", 0) == 128

    def test_sample_inline(self, runs):
        with HilbertTape(runs) as tape:
            qnode = make_proxy_qnode(lambda: qml.sample(), tape, shots=16)
            qnode(0.0)  # deterministic |00>

        s = read_spans(tape)[0]
        samples = json.loads(s["inline_artifacts"][s["outcome_ref"]]["data"])
        assert len(samples) == 16
        assert all(bit == 0 for row in samples for bit in row)

    def test_state_inline_as_complex_pairs(self, runs):
        with HilbertTape(runs) as tape:
            qnode = make_proxy_qnode(lambda: qml.state(), tape)
            qnode(0.0)  # |00> state

        s = read_spans(tape)[0]
        data = json.loads(s["inline_artifacts"][s["outcome_ref"]]["data"])
        # Stored as [[real, imag], ...] — first amplitude should be [1.0, 0.0]
        assert data[0][0] == pytest.approx(1.0, abs=1e-9)
        assert data[0][1] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 2. Span structure
# ---------------------------------------------------------------------------

class TestSpanStructure:

    def test_four_events_per_span(self, runs):
        """EXECUTION_REQUEST + DEVICE_EXECUTE_STARTED + EXECUTION_COMPLETED + EXECUTION_RESULT"""
        with HilbertTape(runs) as tape:
            qnode = make_proxy_qnode(lambda: qml.expval(qml.PauliZ(0)), tape)
            qnode(0.5)

        s = read_spans(tape)[0]
        event_types = [e["event_type"] for e in s["events"]]
        assert "EXECUTION_REQUEST" in event_types
        assert "DEVICE_EXECUTE_STARTED" in event_types
        assert "EXECUTION_COMPLETED" in event_types
        assert "EXECUTION_RESULT" in event_types
        assert len(event_types) == 4

    def test_device_started_event_has_num_tapes(self, runs):
        with HilbertTape(runs) as tape:
            qnode = make_proxy_qnode(lambda: qml.expval(qml.PauliZ(0)), tape)
            qnode(0.5)

        s = read_spans(tape)[0]
        started = next(e for e in s["events"] if e["event_type"] == "DEVICE_EXECUTE_STARTED")
        attrs = started["attributes"]
        if isinstance(attrs, str):
            attrs = json.loads(attrs)
        assert "num_tapes" in attrs
        assert attrs["num_tapes"] >= 1

    def test_parameters_captured_per_span(self, runs):
        with HilbertTape(runs) as tape:
            qnode = make_proxy_qnode(lambda: qml.expval(qml.PauliZ(0)), tape)
            qnode(1.23)

        s = read_spans(tape)[0]
        inline = s["inline_artifacts"]
        param_arts = [a for a in inline.values() if a["kind"] == "parameters"]
        assert len(param_arts) >= 1

    def test_observables_captured(self, runs):
        with HilbertTape(runs) as tape:
            qnode = make_proxy_qnode(lambda: qml.expval(qml.PauliZ(0)), tape)
            qnode(0.5)

        s = read_spans(tape)[0]
        inline = s["inline_artifacts"]
        obs_arts = [a for a in inline.values() if a["kind"] == "observables"]
        assert len(obs_arts) >= 1

    def test_payload_ref_resolves_to_circuit_qasm(self, runs):
        """The circuit is now a templated QASM in the file store (deduplicated),
        so payload_ref must resolve from the catalog, not inline."""
        with HilbertTape(runs) as tape:
            qnode = make_proxy_qnode(lambda: qml.expval(qml.PauliZ(0)), tape)
            qnode(0.5)

        s = read_spans(tape)[0]
        catalog = json.loads((tape.dir_path / "catalog.json").read_text())["artifacts"]
        assert s["payload_ref"] in catalog
        assert catalog[s["payload_ref"]]["kind"] == "circuit_qasm"

    def test_circuit_qasm_deduplicates_across_steps(self, runs):
        """Many evaluations of the same circuit structure produce one QASM file."""
        import pennylane.numpy as pnp
        with HilbertTape(runs) as tape:
            qnode = make_proxy_qnode(lambda: qml.expval(qml.PauliZ(0)), tape)
            for x in [0.1, 0.5, 0.9, 1.3]:
                qnode(pnp.array(x))

        qasm_files = list((tape.dir_path / "artifacts").rglob("*.qasm"))
        assert len(qasm_files) == 1
        # template carries positional placeholders, not baked values
        assert "_p0" in qasm_files[0].read_text()

    def test_backend_id_set(self, runs):
        with HilbertTape(runs) as tape:
            qnode = make_proxy_qnode(lambda: qml.expval(qml.PauliZ(0)), tape)
            qnode(0.5)

        s = read_spans(tape)[0]
        assert s["backend_id"] is not None
        assert s["backend_id"] != ""

    def test_span_status_completed(self, runs):
        with HilbertTape(runs) as tape:
            qnode = make_proxy_qnode(lambda: qml.expval(qml.PauliZ(0)), tape)
            qnode(0.5)

        assert read_spans(tape)[0]["status"] == "COMPLETED"


# ---------------------------------------------------------------------------
# 3. Exception handling (INV-007)
# ---------------------------------------------------------------------------

class TestPennyLaneExceptions:

    def test_device_exception_creates_failed_span(self, runs):
        """When the device raises, a FAILED span with ERROR event is recorded."""
        class CrashingDevice:
            short_name = "crash.qubit"
            def execute(self, tapes, **kw):
                raise RuntimeError("Device overheated")
            def batch_execute(self, tapes, **kw):
                raise RuntimeError("Device overheated")

        with pytest.raises(RuntimeError, match="Device overheated"):
            with HilbertTape(runs) as tape:
                proxy = HilbertPennyLaneDeviceProxy(CrashingDevice(), tape)
                proxy.execute([object()])  # pass dummy tape

        spans = read_spans(tape)
        assert len(spans) == 1
        assert spans[0]["status"] == "FAILED"
        error_event = next(e for e in spans[0]["events"] if e["event_type"] == "ERROR")
        assert error_event["attributes"]["exception_type"] == "RuntimeError"

    def test_exception_propagates_to_caller(self, runs):
        class CrashingDevice:
            short_name = "crash"
            def execute(self, tapes, **kw):
                raise ValueError("bad params")

        with pytest.raises(ValueError, match="bad params"):
            with HilbertTape(runs) as tape:
                proxy = HilbertPennyLaneDeviceProxy(CrashingDevice(), tape)
                proxy.execute([object()])


# ---------------------------------------------------------------------------
# 4. No files on disk for small outcomes
# ---------------------------------------------------------------------------

class TestNoFilePollution:

    def test_all_measurements_stay_inline(self, runs):
        """expval, probs, counts, sample — none should write .npy files."""
        measurements = [
            (lambda: qml.expval(qml.PauliZ(0)), None),
            (lambda: qml.probs(wires=[0]),       None),
            (lambda: qml.counts(),               32),
            (lambda: qml.sample(),               16),
        ]
        for meas_fn, shots in measurements:
            kwargs = {"shots": shots} if shots else {}
            dev = qml.device("default.qubit", wires=1, **kwargs)
            with HilbertTape(runs) as tape:
                proxy = HilbertPennyLaneDeviceProxy(dev, tape)
                @qml.qnode(proxy, diff_method="parameter-shift")
                def circuit():
                    qml.PauliX(wires=0)
                    return meas_fn()
                circuit()

            npy_files = list((tape.dir_path / "artifacts").rglob("*.npy"))
            assert len(npy_files) == 0, f"Got .npy files for {meas_fn()}"
