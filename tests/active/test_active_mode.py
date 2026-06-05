"""
tests/active/test_active_mode.py

Tests for Active Mode probing and the kl_expressibility analyzer they feed.
Active Mode runs real circuits, so these use small ansätze and modest sample
counts. The headline physics check is directional: a rigid ansatz must score a
larger Haar KL divergence than an expressive one.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hilbertbench import HilbertTrace
from hilbertbench.active import probe_expressibility, active_probe_qiskit, active_probe_pennylane
from hilbertbench.analysis import kl_expressibility, circuit_structure


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    d = tmp_path / "runs"
    d.mkdir()
    return d


# ── Generic probe core ────────────────────────────────────────────────────────

class TestProbeCore:

    def _ry_state(self, theta):
        c, s = np.cos(theta[0] / 2), np.sin(theta[0] / 2)
        return np.array([c, s], dtype=complex)

    def test_records_active_mode_trace(self, runs):
        run = probe_expressibility(self._ry_state, num_params=1, num_samples=50,
                                   output_root=runs, seed=0)
        t = HilbertTrace(run)
        assert t.mode == "active"
        assert len(t) == 50

    def test_trace_verifies(self, runs):
        run = probe_expressibility(self._ry_state, num_params=1, num_samples=30,
                                   output_root=runs, seed=0)
        assert HilbertTrace(run).verify() is True

    def test_seed_reproducible(self, runs):
        r1 = probe_expressibility(self._ry_state, 1, 40, output_root=runs, seed=42)
        r2 = probe_expressibility(self._ry_state, 1, 40, output_root=runs, seed=42)
        # Same seed → identical sampled parameters → identical statevectors
        p1 = [s.parameters for s in HilbertTrace(r1).completed()]
        p2 = [s.parameters for s in HilbertTrace(r2).completed()]
        assert p1 == p2

    def test_statevector_round_trips(self, runs):
        run = probe_expressibility(self._ry_state, 1, 10, output_root=runs, seed=1)
        t = HilbertTrace(run)
        span = t.completed()[0]
        theta = span.parameters
        recorded = span.outcome  # [[re, im], ...]
        sv = np.array([complex(re, im) for re, im in recorded])
        expected = self._ry_state(np.asarray(theta))
        assert np.allclose(sv, expected)


# ── Qiskit probe ──────────────────────────────────────────────────────────────

class TestQiskitProbe:

    def test_qiskit_probe_stores_circuit_qasm(self, runs):
        from qiskit.circuit.library import RealAmplitudes
        run = active_probe_qiskit(RealAmplitudes(2, reps=2), num_samples=50,
                                  output_root=runs, seed=1)
        t = HilbertTrace(run)
        catalog = t.catalog
        assert any(m["kind"] == "circuit_qasm" for m in catalog.values())
        assert t.verify() is True

    def test_qiskit_circuit_structure_visible(self, runs):
        from qiskit.circuit.library import RealAmplitudes
        run = active_probe_qiskit(RealAmplitudes(2, reps=3), num_samples=20,
                                  output_root=runs, seed=1)
        cs = circuit_structure(run)
        assert cs["status"] == "OK"
        # decomposed → real gates visible, depth > 1
        assert cs["primary"]["depth"] > 1
        assert cs["primary"]["entangling_gates"] >= 1


# ── PennyLane probe ───────────────────────────────────────────────────────────

class TestPennyLaneProbe:

    def test_pennylane_probe(self, runs):
        import pennylane as qml
        def ansatz(theta):
            qml.StronglyEntanglingLayers(theta.reshape(1, 2, 3), wires=[0, 1])
        run = active_probe_pennylane(ansatz, num_qubits=2, num_params=6,
                                     num_samples=40, output_root=runs, seed=2)
        t = HilbertTrace(run)
        assert t.mode == "active"
        assert t.verify() is True


# ── kl_expressibility ─────────────────────────────────────────────────────────

class TestExpressibility:

    def test_rigid_more_than_expressive(self, runs):
        """Directional physics: rigid ansatz has larger Haar KL than expressive."""
        from qiskit.circuit import ParameterVector, QuantumCircuit
        from qiskit.circuit.library import RealAmplitudes

        p = ParameterVector("t", 2)
        rigid = QuantumCircuit(2)
        rigid.ry(p[0], 0); rigid.ry(p[1], 1)  # no entanglement, shallow

        rigid_run = active_probe_qiskit(rigid, 800, output_root=runs, seed=1)
        expr_run = active_probe_qiskit(RealAmplitudes(2, reps=4), 800, output_root=runs, seed=2)

        kl_rigid = kl_expressibility(rigid_run, seed=1)["kl_divergence"]
        kl_expr = kl_expressibility(expr_run, seed=2)["kl_divergence"]

        assert kl_rigid > kl_expr

    def test_expressive_ansatz_low_kl(self, runs):
        """StronglyEntanglingLayers is known to be highly expressive."""
        import pennylane as qml
        def ansatz(theta):
            qml.StronglyEntanglingLayers(theta.reshape(2, 2, 3), wires=[0, 1])
        run = active_probe_pennylane(ansatz, 2, 12, 600, output_root=runs, seed=7)
        r = kl_expressibility(run, seed=7)
        assert r["kl_divergence"] < 0.1
        assert "Highly Expressive" in r["status"]

    def test_num_qubits_inferred(self, runs):
        from qiskit.circuit.library import RealAmplitudes
        run = active_probe_qiskit(RealAmplitudes(2, reps=2), 100, output_root=runs, seed=1)
        r = kl_expressibility(run, seed=1)
        assert r["num_qubits"] == 2
        assert r["num_states"] == 100

    def test_passive_trace_guard(self, runs):
        """Expressibility on a passive trace returns a guard, not a number."""
        from hilbertbench.recorder.tape import HilbertTape
        dummy = "sha256:" + "a" * 64
        with HilbertTape(runs) as tape:  # default passive mode
            with tape.execution_span(payload_ref=dummy) as span:
                span.attach_inline("0.5", kind="execution_outcome", encoding="json")
        r = kl_expressibility(tape.dir_path)
        assert r["kl_divergence"] is None
        assert "Active Mode" in r["status"]

    def test_insufficient_states(self, runs):
        from qiskit.circuit import ParameterVector, QuantumCircuit
        p = ParameterVector("t", 1)
        qc = QuantumCircuit(1); qc.ry(p[0], 0)
        run = active_probe_qiskit(qc, 1, output_root=runs, seed=1)  # only 1 state
        r = kl_expressibility(run)
        assert r["kl_divergence"] is None
        assert "Insufficient" in r["status"]
