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
    _maybe_capture_calibration,
    _resolve_backend,
    _serialize_calibration,
    HilbertEstimatorProxy,
    HilbertSamplerProxy,
)
from hilbertbench.recorder.tape import HilbertTape
from hilbertbench.trace import HilbertTrace

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


# ---------------------------------------------------------------------------
# 4. Backend resolution across primitive conventions
# ---------------------------------------------------------------------------

class TestResolveBackend:
    """The three conventions in the wild: qiskit BackendEstimatorV2 exposes
    .backend as a property, qiskit-ibm-runtime primitives expose backend()
    as a bound method, and qiskit-aer primitives only hold ._backend."""

    def test_property_style(self):
        from qiskit.primitives import BackendEstimatorV2
        fake = FakeManilaV2()
        est = BackendEstimatorV2(backend=fake)
        assert _resolve_backend(est) is fake

    def test_method_style_runtime_convention(self):
        fake = FakeManilaV2()

        class RuntimeLikeEstimator:
            def backend(self):
                return fake

        assert _resolve_backend(RuntimeLikeEstimator()) is fake

    def test_private_attr_aer_convention(self):
        fake = FakeManilaV2()

        class AerLikeEstimator:
            def __init__(self):
                self._backend = fake

        assert _resolve_backend(AerLikeEstimator()) is fake

    def test_backend_passed_directly(self):
        fake = FakeManilaV2()
        assert _resolve_backend(fake) is fake

    def test_statevector_primitive_resolves_to_none(self):
        from qiskit.primitives import StatevectorEstimator
        assert _resolve_backend(StatevectorEstimator()) is None

    def test_none_resolves_to_none(self):
        assert _resolve_backend(None) is None


# ---------------------------------------------------------------------------
# 5. Drift refresh and rate limiting
# ---------------------------------------------------------------------------

class _DriftingBackend:
    """Backend whose calibration content changes when .version is bumped."""

    def __init__(self):
        self.version = 0

    def properties(self, refresh=False):
        version = self.version

        class _Props:
            def to_dict(self):
                return {"qubits": [], "gates": [], "version": version}

        return _Props()


class _CalState:
    """Minimal proxy stand-in carrying the calibration capture state."""

    def __init__(self, refresh_s):
        self.calibration_refresh_s = refresh_s
        self._cal_next_check = 0.0
        self._cal_last_hash = None


class TestCalibrationRefresh:

    def test_drift_yields_snapshot_history(self, runs):
        backend = _DriftingBackend()
        state = _CalState(refresh_s=0.0)

        with HilbertTape(runs) as tape:
            _maybe_capture_calibration(state, tape, backend)
            backend.version = 1
            _maybe_capture_calibration(state, tape, backend)

        assert len(calibration_artifacts(tape)) == 2

    def test_stable_calibration_attaches_once(self, runs):
        backend = _DriftingBackend()
        state = _CalState(refresh_s=0.0)

        with HilbertTape(runs) as tape:
            for _ in range(3):
                _maybe_capture_calibration(state, tape, backend)

        assert len(calibration_artifacts(tape)) == 1

    def test_rate_limit_skips_query_inside_window(self, runs):
        backend = _DriftingBackend()
        state = _CalState(refresh_s=3600.0)

        with HilbertTape(runs) as tape:
            _maybe_capture_calibration(state, tape, backend)
            backend.version = 1  # drifts, but inside the window
            _maybe_capture_calibration(state, tape, backend)

        assert len(calibration_artifacts(tape)) == 1


# ---------------------------------------------------------------------------
# 6. Trace-side calibration history
# ---------------------------------------------------------------------------

class TestCalibrationHistory:

    def test_single_snapshot_history(self, runs):
        from qiskit.primitives import BackendEstimatorV2
        theta = Parameter("t")
        qc = QuantumCircuit(2); qc.ry(theta, 0); qc.cx(0, 1)
        obs = SparsePauliOp("ZZ")

        with HilbertTape(runs) as tape:
            proxy = HilbertEstimatorProxy(
                tape,
                real_estimator=BackendEstimatorV2(backend=FakeManilaV2()),
            )
            proxy.run([(qc, obs, np.array([[0.5]]))])

        trace = HilbertTrace(tape.dir_path)
        history = trace.calibration_history()
        assert len(history) == 1
        assert isinstance(history[0]["captured_at"], int)
        assert "qubits" in history[0]["calibration"]
        assert trace.calibration() == history[-1]["calibration"]

    def test_drift_history_is_chronological(self, runs):
        backend = _DriftingBackend()
        state = _CalState(refresh_s=0.0)

        with HilbertTape(runs) as tape:
            for version in range(3):
                backend.version = version
                _maybe_capture_calibration(state, tape, backend)

        trace = HilbertTrace(tape.dir_path)
        history = trace.calibration_history()
        assert [h["calibration"]["version"] for h in history] == [0, 1, 2]
        times = [h["captured_at"] for h in history]
        assert times == sorted(times)
        # calibration() returns the newest snapshot
        assert trace.calibration()["version"] == 2

    def test_ideal_trace_has_empty_history(self, runs):
        theta = Parameter("t")
        qc = QuantumCircuit(1); qc.ry(theta, 0)
        obs = SparsePauliOp("Z")

        with HilbertTape(runs) as tape:
            proxy = HilbertEstimatorProxy(tape)
            proxy.run([(qc, obs, np.array([[0.5]]))])

        trace = HilbertTrace(tape.dir_path)
        assert trace.calibration_history() == []
        assert trace.calibration() is None
