"""
tests/analysis/test_noise.py

Tests for the noise-profile analyzer (Diagnostic Axis: Noise). Calibration
data only exists on real/fake hardware backends, so these use Qiskit's
FakeManilaV2 (which ships realistic T1/T2/readout/gate-error data) and assert
ideal simulators degrade gracefully to a no-calibration result.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import SparsePauliOp

from hilbertbench.recorder.tape import HilbertTape
from hilbertbench.integrations.qiskit import HilbertEstimatorProxy
from hilbertbench.analysis import noise_profile

fake_provider = pytest.importorskip("qiskit_ibm_runtime.fake_provider")
FakeManilaV2 = fake_provider.FakeManilaV2


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    d = tmp_path / "runs"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def record_depth(runs, n_qubits, reps, seed=0):
    """Record one execution of a depth-`reps` circuit on FakeManilaV2."""
    from qiskit.primitives import BackendEstimatorV2

    n_params = n_qubits * reps
    p = ParameterVector("t", n_params)
    qc = QuantumCircuit(n_qubits)
    idx = 0
    for _ in range(reps):
        for q in range(n_qubits):
            qc.ry(p[idx], q)
            idx += 1
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
    obs = SparsePauliOp("Z" + "I" * (n_qubits - 1))

    rng = np.random.default_rng(seed)
    est = BackendEstimatorV2(backend=FakeManilaV2())
    with HilbertTape(runs) as tape:
        proxy = HilbertEstimatorProxy(tape, real_estimator=est)
        proxy.run([(qc, obs, np.array([rng.random(n_params)]))]).result()
    return tape.dir_path


# ---------------------------------------------------------------------------
# Device summary
# ---------------------------------------------------------------------------

class TestDeviceSummary:

    def test_reports_calibration_stats(self, runs):
        r = noise_profile(record_depth(runs, 2, 1))
        assert r["backend_name"] == "ibmq_manila"
        assert r["num_qubits_calibrated"] == 5
        assert r["t1_us"]["mean"] > 0
        assert r["t2_us"]["mean"] > 0
        assert 0.0 <= r["readout_error"]["mean"] <= 1.0

    def test_gate_errors_present(self, runs):
        r = noise_profile(record_depth(runs, 2, 1))
        assert r["gate_error_1q_mean"] >= 0.0
        assert r["gate_error_2q_mean"] > r["gate_error_1q_mean"]

    def test_estimated_fidelity_in_unit_interval(self, runs):
        r = noise_profile(record_depth(runs, 2, 1))
        assert 0.0 <= r["estimated_circuit_fidelity"] <= 1.0


# ---------------------------------------------------------------------------
# Ideal simulator degrades gracefully
# ---------------------------------------------------------------------------

class TestIdealSimulator:

    def test_no_calibration_status(self, runs):
        from qiskit.primitives import StatevectorEstimator
        qc = QuantumCircuit(2)
        theta = ParameterVector("t", 1)
        qc.ry(theta[0], 0); qc.cx(0, 1)
        obs = SparsePauliOp("ZZ")
        with HilbertTape(runs) as tape:
            proxy = HilbertEstimatorProxy(tape)  # ideal StatevectorEstimator
            proxy.run([(qc, obs, np.array([[0.5]]))]).result()
        r = noise_profile(tape.dir_path)
        assert "No calibration" in r["status"]
        assert r["estimated_circuit_fidelity"] is None


# ---------------------------------------------------------------------------
# Interaction with circuit depth (proposal Axis 5)
# ---------------------------------------------------------------------------

class TestDepthInteraction:

    def test_fidelity_decreases_with_depth(self, runs):
        shallow = noise_profile(record_depth(runs, 3, 1, seed=1))
        deep = noise_profile(record_depth(runs, 3, 15, seed=2))
        assert deep["estimated_circuit_fidelity"] < \
            shallow["estimated_circuit_fidelity"]

    def test_dominant_error_shifts_to_two_qubit_gates(self, runs):
        # shallow circuit: readout dominates the small infidelity
        shallow = noise_profile(record_depth(runs, 3, 1, seed=3))
        assert shallow["dominant_error_source"] == "readout"
        # deep circuit: many CX gates dominate
        deep = noise_profile(record_depth(runs, 3, 15, seed=4))
        assert deep["dominant_error_source"] == "two_qubit_gates"


# ---------------------------------------------------------------------------
# Calibration scoping (found on ibm_marrakesh: device-wide means over a
# 156-qubit device, dead edges included, mispredicted fidelity by ~600x)
# ---------------------------------------------------------------------------

def build_scoped_trace(runs, tmp_path):
    """Trace with a synthetic calibration: qubits 0/1 good, qubit 2 awful.
    The recorded circuit only touches qubits 0 and 1."""
    import json as _json
    from hilbertbench.models import Encoding, Kind

    cal = {
        "backend_name": "synthetic",
        "qubits": [
            [{"name": "T1", "value": 100.0},
             {"name": "T2", "value": 80.0},
             {"name": "readout_error", "value": 0.01}],
            [{"name": "T1", "value": 110.0},
             {"name": "T2", "value": 90.0},
             {"name": "readout_error", "value": 0.01}],
            [{"name": "T1", "value": 5.0},
             {"name": "T2", "value": 3.0},
             {"name": "readout_error", "value": 0.5}],
        ],
        "gates": [
            {"qubits": [0], "parameters":
                [{"name": "gate_error", "value": 1e-4}]},
            {"qubits": [1], "parameters":
                [{"name": "gate_error", "value": 1e-4}]},
            {"qubits": [2], "parameters":
                [{"name": "gate_error", "value": 0.4}]},
            {"qubits": [0, 1], "parameters":
                [{"name": "gate_error", "value": 0.005}]},
            {"qubits": [1, 2], "parameters":
                [{"name": "gate_error", "value": 0.9}]},
        ],
    }
    qasm = ('OPENQASM 3.0;\ninclude "stdgates.inc";\n'
            "sx $0;\ncx $0, $1;\n")

    cal_f = tmp_path / "cal.json"
    cal_f.write_text(_json.dumps(cal))
    qasm_f = tmp_path / "circ.qasm"
    qasm_f.write_text(qasm)
    with HilbertTape(runs) as tape:
        tape.attach_artifact(cal_f, kind=Kind.calibration_snapshot,
                             encoding=Encoding.json)
        ref = tape.attach_artifact(qasm_f, kind=Kind.circuit_qasm,
                                   encoding=Encoding.openqasm)
        with tape.execution_span(payload_ref=ref) as span:
            span.outcome_ref = span.attach_inline(
                "0.5", kind="execution_outcome", encoding="json")
    return tape.dir_path


class TestCalibrationScoping:

    def test_stats_scoped_to_active_qubits(self, runs, tmp_path):
        r = noise_profile(build_scoped_trace(runs, tmp_path))
        assert r["scope"] == "active_qubits"
        assert r["active_qubits"] == [0, 1]
        # the awful qubit 2 must not contaminate any statistic
        assert r["gate_error_2q_mean"] == pytest.approx(0.005)
        assert r["gate_error_1q_mean"] == pytest.approx(1e-4)
        assert r["readout_error"]["mean"] == pytest.approx(0.01)
        assert r["t1_us"]["min"] == pytest.approx(100.0)

    def test_fidelity_uses_scoped_errors(self, runs, tmp_path):
        r = noise_profile(build_scoped_trace(runs, tmp_path))
        # 1 sx + 1 cx on good qubits: fidelity must stay high; the
        # device-wide means (incl. the 0.9-error edge) would crater it
        assert r["estimated_circuit_fidelity"] > 0.9
