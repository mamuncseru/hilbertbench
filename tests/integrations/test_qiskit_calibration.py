"""
tests/integrations/test_qiskit_calibration.py

Tests for calibration-snapshot capture. Calibration data (T1, T2, readout
error, gate errors) only exists on real/fake hardware backends, never on ideal
simulators — so these tests use Qiskit's FakeManilaV2 which ships realistic
calibration data, and assert that ideal simulators produce no snapshot.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import SparsePauliOp

from hilbertbench.integrations.qiskit import (
    _serialize_calibration,
    HilbertEstimatorProxy,
    HilbertSamplerProxy,
)
from hilbertbench.recorder.tape import HilbertTape

# FakeManilaV2 carries realistic calibration; skip the whole module if unavailable.
fake_provider = pytest.importorskip("qiskit_ibm_runtime.fake_provider")
FakeManilaV2 = fake_provider.FakeManilaV2


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    d = tmp_path / "runs"
    d.mkdir()
    return d


def calibration_artifacts(tape: HilbertTape) -> list:
    catalog = json.loads((tape.dir_path / "catalog.json").read_text())["artifacts"]
    return [a for a in catalog.values() if a["kind"] == "calibration_snapshot"]


# ---------------------------------------------------------------------------
# 1. Serialization helper
# ---------------------------------------------------------------------------

class TestSerializeCalibration:

    def test_extracts_t1_t2_readout(self):
        cal = _serialize_calibration(FakeManilaV2())
        assert cal is not None
        d = json.loads(cal)
        assert "qubits" in d and len(d["qubits"]) > 0
        params = {p["name"] for p in d["qubits"][0]}
        assert {"T1", "T2", "readout_error"} <= params

    def test_none_backend_returns_none(self):
        assert _serialize_calibration(None) is None

    def test_backend_without_properties_returns_none(self):
        class NoCalBackend:
            def properties(self):
                return None
        assert _serialize_calibration(NoCalBackend()) is None

    def test_backend_raising_properties_returns_none(self):
        class BrokenBackend:
            def properties(self):
                raise RuntimeError("no calibration endpoint")
        assert _serialize_calibration(BrokenBackend()) is None


# ---------------------------------------------------------------------------
# 2. Estimator proxy capture
# ---------------------------------------------------------------------------

class TestEstimatorCalibrationCapture:

    def _estimator_on_fake(self):
        from qiskit.primitives import BackendEstimatorV2
        return BackendEstimatorV2(backend=FakeManilaV2())

    def test_snapshot_captured(self, runs):
        theta = Parameter("t")
        qc = QuantumCircuit(2); qc.ry(theta, 0); qc.cx(0, 1)
        obs = SparsePauliOp("ZZ")

        with HilbertTape(runs) as tape:
            proxy = HilbertEstimatorProxy(tape, real_estimator=self._estimator_on_fake())
            proxy.run([(qc, obs, np.array([[0.5]]))])

        snaps = calibration_artifacts(tape)
        assert len(snaps) == 1
        assert snaps[0]["encoding"] == "json"

    def test_snapshot_captured_once_across_runs(self, runs):
        theta = Parameter("t")
        qc = QuantumCircuit(2); qc.ry(theta, 0); qc.cx(0, 1)
        obs = SparsePauliOp("ZZ")

        with HilbertTape(runs) as tape:
            proxy = HilbertEstimatorProxy(tape, real_estimator=self._estimator_on_fake())
            for _ in range(3):
                proxy.run([(qc, obs, np.array([[0.5]]))])

        # Content-addressed: identical calibration → one artifact regardless of run count
        assert len(calibration_artifacts(tape)) == 1

    def test_ideal_simulator_produces_no_snapshot(self, runs):
        theta = Parameter("t")
        qc = QuantumCircuit(1); qc.ry(theta, 0)
        obs = SparsePauliOp("Z")

        with HilbertTape(runs) as tape:
            proxy = HilbertEstimatorProxy(tape)  # default StatevectorEstimator
            proxy.run([(qc, obs, np.array([[0.5]]))])

        assert len(calibration_artifacts(tape)) == 0


# ---------------------------------------------------------------------------
# 3. Sampler proxy capture
# ---------------------------------------------------------------------------

class TestSamplerCalibrationCapture:

    def test_snapshot_captured(self, runs):
        from qiskit.primitives import BackendSamplerV2
        qc = QuantumCircuit(2); qc.h(0); qc.cx(0, 1); qc.measure_all()

        with HilbertTape(runs) as tape:
            proxy = HilbertSamplerProxy(
                tape, real_sampler=BackendSamplerV2(backend=FakeManilaV2())
            )
            proxy.run([(qc, None, 32)])

        assert len(calibration_artifacts(tape)) == 1

    def test_ideal_sampler_produces_no_snapshot(self, runs):
        qc = QuantumCircuit(1); qc.h(0); qc.measure_all()
        with HilbertTape(runs) as tape:
            proxy = HilbertSamplerProxy(tape)  # default StatevectorSampler
            proxy.run([(qc, None, 32)])

        assert len(calibration_artifacts(tape)) == 0
