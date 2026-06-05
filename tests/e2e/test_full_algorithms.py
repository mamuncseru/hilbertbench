"""
tests/e2e/test_full_algorithms.py

Tier 3 end-to-end regression tests.
Each test runs a complete algorithm for a small number of steps and verifies:
  - The trace is sealed and complete
  - The expected number of spans were recorded
  - Inline artifacts contain the correct data kinds

Algorithms covered:
  - VQE on a simple 2-qubit Hamiltonian (Qiskit Estimator)
  - QAOA bitstring sampling (Qiskit Sampler)
  - QNN training on a 4-point dataset (PennyLane)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hilbertbench.recorder.tape import HilbertTape
from hilbertbench.recorder.storage.writer import convert_trace_to_parquet


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    d = tmp_path / "runs"
    d.mkdir()
    return d


def read_spans(tape: HilbertTape) -> list[dict]:
    return [json.loads(l) for l in (tape.dir_path / "events.jsonl").read_text().splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# 1. VQE  (Qiskit EstimatorV2)
# ---------------------------------------------------------------------------

class TestVQERegression:

    def test_vqe_trace_complete(self, runs):
        """
        Runs 5 steps of gradient-based VQE on the simple H = Z⊗Z Hamiltonian.
        Verifies: spans created, outcome + parameters + observables captured.
        """
        from qiskit.circuit import QuantumCircuit, Parameter
        from qiskit.quantum_info import SparsePauliOp
        from qiskit.primitives import StatevectorEstimator
        from hilbertbench.integrations.qiskit import HilbertEstimatorProxy

        theta = Parameter("theta")
        ansatz = QuantumCircuit(2)
        ansatz.ry(theta, 0)
        ansatz.cx(0, 1)

        hamiltonian = SparsePauliOp("ZZ")
        lr = 0.1
        param_val = np.array([[0.5]])

        with HilbertTape(runs, tags={"algorithm": "vqe"}) as tape:
            estimator = HilbertEstimatorProxy(tape)

            for _ in range(5):
                # Compute energy
                job = estimator.run([(ansatz, hamiltonian, param_val)])
                energy = job.result()[0].data.evs.item()
                # Finite-difference gradient
                dp = np.array([[0.01]])
                e_plus  = estimator.run([(ansatz, hamiltonian, param_val + dp)]).result()[0].data.evs.item()
                e_minus = estimator.run([(ansatz, hamiltonian, param_val - dp)]).result()[0].data.evs.item()
                grad = (e_plus - e_minus) / (2 * 0.01)
                param_val = param_val - lr * grad

        spans = read_spans(tape)
        assert len(spans) == 15  # 3 evaluations × 5 steps
        assert all(s["status"] == "COMPLETED" for s in spans)

        # Every span should have outcome + parameters + observables inline
        for s in spans:
            kinds = {a["kind"] for a in (s.get("inline_artifacts") or {}).values()}
            assert "execution_outcome" in kinds
            assert "parameters" in kinds
            assert "observables" in kinds

        # Trace must be sealed
        manifest = json.loads((tape.dir_path / "trace.json").read_text())
        assert manifest["status"] == "SEALED_SUCCESS"


# ---------------------------------------------------------------------------
# 2. QAOA  (Qiskit SamplerV2)
# ---------------------------------------------------------------------------

class TestQAOARegression:

    def test_qaoa_bitstrings_recorded(self, runs):
        """
        Runs a 2-qubit QAOA-like circuit for 3 angles and records bitstring outcomes.
        Verifies that counts are captured inline with proper structure.
        """
        from qiskit.circuit import QuantumCircuit, Parameter
        from hilbertbench.integrations.qiskit import HilbertSamplerProxy

        gamma = Parameter("gamma")
        beta  = Parameter("beta")
        qc = QuantumCircuit(2)
        qc.h([0, 1])
        qc.rzz(gamma, 0, 1)
        qc.rx(beta, 0); qc.rx(beta, 1)
        qc.measure_all()

        angles = np.array([[0.3, 0.5], [0.6, 0.8], [1.0, 1.2]])

        with HilbertTape(runs, tags={"algorithm": "qaoa"}) as tape:
            sampler = HilbertSamplerProxy(tape)
            sampler.run([(qc, angles, 128)])

        spans = read_spans(tape)
        assert len(spans) == 1
        s = spans[0]

        outcome = json.loads(s["inline_artifacts"][s["outcome_ref"]]["data"])
        assert "meas" in outcome
        counts = outcome["meas"]["counts"]
        total = sum(int(v) for v in counts.values())
        # 3 angle sets × 128 shots = 384 total (get_counts aggregates across param sets)
        assert total == 128 * len(angles)
        assert manifest_sealed(tape)

    def test_qaoa_multiple_angle_sets(self, runs):
        """Multiple parameter sets in one PUB → one span total."""
        from qiskit.circuit import QuantumCircuit, Parameter
        from hilbertbench.integrations.qiskit import HilbertSamplerProxy

        g, b = Parameter("g"), Parameter("b")
        qc = QuantumCircuit(2)
        qc.h([0, 1]); qc.rzz(g, 0, 1); qc.rx(b, 0); qc.measure_all()

        N_ANGLES = 5
        angles = np.random.uniform(0, np.pi, (N_ANGLES, 2))

        with HilbertTape(runs) as tape:
            HilbertSamplerProxy(tape).run([(qc, angles, 64)])

        spans = read_spans(tape)
        assert len(spans) == 1  # one PUB = one span


# ---------------------------------------------------------------------------
# 3. PennyLane QNN  (gradient-based training)
# ---------------------------------------------------------------------------

class TestQNNRegression:

    def test_qnn_training_trace_complete(self, runs):
        """
        Trains a 2-qubit QNN for 5 steps on a 4-point dataset.
        Verifies: spans recorded, outcomes + params captured, trace sealed.
        """
        import pennylane as qml
        from pennylane import numpy as pnp
        from hilbertbench.integrations.pennylane import HilbertPennyLaneDeviceProxy

        X = pnp.array([[0.1, 0.2], [0.8, 0.9], [0.2, 0.1], [0.9, 0.8]])
        y = pnp.array([-1.0, 1.0, -1.0, 1.0])

        real_dev = qml.device("default.qubit", wires=2)

        with HilbertTape(runs, tags={"algorithm": "qnn"}) as tape:
            proxy = HilbertPennyLaneDeviceProxy(real_dev, tape)

            @qml.qnode(proxy, diff_method="parameter-shift")
            def circuit(x, weights):
                qml.AngleEmbedding(x, wires=range(2), rotation="Y")
                qml.StronglyEntanglingLayers(weights, wires=range(2))
                return qml.expval(qml.PauliZ(0))

            shape = qml.StronglyEntanglingLayers.shape(n_layers=1, n_wires=2)
            weights = pnp.array(np.random.uniform(0, np.pi, shape), requires_grad=True)
            opt = qml.AdamOptimizer(stepsize=0.05)

            for _ in range(5):
                def loss_fn(w):
                    preds = pnp.array([circuit(x, w) for x in X])
                    return pnp.mean((preds - y) ** 2)
                weights, _ = opt.step_and_cost(loss_fn, weights)

        spans = read_spans(tape)
        assert len(spans) > 0
        assert all(s["status"] == "COMPLETED" for s in spans)
        assert manifest_sealed(tape)

        for s in spans:
            inline = s.get("inline_artifacts") or {}
            kinds = {a["kind"] for a in inline.values()}
            assert "execution_outcome" in kinds


# ---------------------------------------------------------------------------
# 4. Cross-framework: same Hamiltonian, Qiskit vs PennyLane
# ---------------------------------------------------------------------------

class TestCrossFramework:

    def test_both_frameworks_produce_valid_traces(self, runs):
        """
        Evaluate Z⊗Z expectation value using both frameworks.
        Both traces should be valid, sealed, and contain outcome data.
        """
        import pennylane as qml
        from pennylane import numpy as pnp
        from qiskit.circuit import QuantumCircuit, Parameter
        from qiskit.quantum_info import SparsePauliOp
        from hilbertbench.integrations.qiskit import HilbertEstimatorProxy
        from hilbertbench.integrations.pennylane import HilbertPennyLaneDeviceProxy

        # Qiskit trace
        theta_q = Parameter("theta")
        qc = QuantumCircuit(2)
        qc.ry(theta_q, 0); qc.cx(0, 1)
        obs = SparsePauliOp("ZZ")

        runs_qiskit = runs / "qiskit"
        runs_qiskit.mkdir()
        with HilbertTape(runs_qiskit, tags={"framework": "qiskit"}) as tape_q:
            HilbertEstimatorProxy(tape_q).run([(qc, obs, np.array([[0.5]]))])

        # PennyLane trace
        runs_pl = runs / "pennylane"
        runs_pl.mkdir()
        real_dev = qml.device("default.qubit", wires=2)
        with HilbertTape(runs_pl, tags={"framework": "pennylane"}) as tape_pl:
            proxy = HilbertPennyLaneDeviceProxy(real_dev, tape_pl)
            @qml.qnode(proxy, diff_method="parameter-shift")
            def pl_circuit(x):
                qml.RY(x, wires=0); qml.CNOT(wires=[0, 1])
                return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
            pl_circuit(pnp.array(0.5, requires_grad=True))

        for tape in (tape_q, tape_pl):
            manifest = json.loads((tape.dir_path / "trace.json").read_text())
            assert manifest["status"] == "SEALED_SUCCESS"
            spans = read_spans(tape)
            assert len(spans) >= 1
            for s in spans:
                assert s.get("outcome_ref") is not None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def manifest_sealed(tape: HilbertTape) -> bool:
    m = json.loads((tape.dir_path / "trace.json").read_text())
    return m["status"] == "SEALED_SUCCESS"
