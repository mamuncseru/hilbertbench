"""
tests/e2e/test_phenomenology.py

Phenomenological validation (proposal Section 2.6): plant a known QML
phenomenon in synthetic ground-truth circuits and confirm the detector
attributes it correctly from trace evidence alone.

Barren plateau (McClean et al. 2018): the variance of a random hardware-
efficient ansatz's cost landscape vanishes exponentially with qubit count.
We record cost landscapes for a wide/deep circuit (planted barren plateau)
and a narrow/shallow control, then assert detect_barren_plateau classifies
each correctly.

These run real circuits, so widths/samples are kept modest. Seeds are fixed
for deterministic verdicts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import SparsePauliOp

from hilbertbench.recorder.tape import HilbertTape
from hilbertbench.integrations.qiskit import HilbertEstimatorProxy
from hilbertbench.models import Mode
from hilbertbench.analysis import detect_barren_plateau


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    d = tmp_path / "runs"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_hea(n_qubits: int, n_layers: int):
    """Hardware-efficient ansatz with a ZZ observable on qubits 0,1."""
    n_params = n_layers * n_qubits * 2
    theta = ParameterVector("t", n_params)
    qc = QuantumCircuit(n_qubits)
    idx = 0
    for _ in range(n_layers):
        for q in range(n_qubits):
            qc.ry(theta[idx], q)
            idx += 1
            qc.rz(theta[idx], q)
            idx += 1
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
    obs = SparsePauliOp("Z" * 2 + "I" * (n_qubits - 2))
    return qc, obs, n_params


def record_landscape(runs, n_qubits, n_layers, n_samples, seed):
    """Record the cost landscape at random parameter points (active mode)."""
    qc, obs, n_params = build_hea(n_qubits, n_layers)
    rng = np.random.default_rng(seed)
    with HilbertTape(runs, mode=Mode.active) as tape:
        est = HilbertEstimatorProxy(tape)
        for _ in range(n_samples):
            p = rng.uniform(0.0, 2.0 * np.pi, (1, n_params))
            est.run([(qc, obs, p)]).result()
    return tape.dir_path


# ---------------------------------------------------------------------------
# Planted barren plateau is detected
# ---------------------------------------------------------------------------

class TestBarrenPlateauValidation:

    def test_wide_deep_circuit_flagged_barren(self, runs):
        run_dir = record_landscape(runs, n_qubits=8, n_layers=24,
                                   n_samples=100, seed=1)
        result = detect_barren_plateau(run_dir)
        assert result["status"] == "Barren Plateau Detected"
        assert result["variance"] < 0.005

    def test_shallow_control_is_trainable(self, runs):
        run_dir = record_landscape(runs, n_qubits=2, n_layers=6,
                                   n_samples=100, seed=2)
        result = detect_barren_plateau(run_dir)
        assert result["status"] == "Trainable"
        assert result["variance"] > 0.005

    def test_variance_collapses_with_width(self, runs):
        """The planted property: variance must shrink as width grows."""
        narrow = detect_barren_plateau(
            record_landscape(runs, 2, 6, 80, seed=3)
        )["variance"]
        wide = detect_barren_plateau(
            record_landscape(runs, 8, 24, 80, seed=4)
        )["variance"]
        assert wide < narrow

    def test_trace_is_active_mode(self, runs):
        """Landscape probing is a controlled, opt-in active diagnostic."""
        from hilbertbench import HilbertTrace
        run_dir = record_landscape(runs, 2, 4, 20, seed=5)
        assert HilbertTrace(run_dir).mode == "active"
