"""
tests/analysis/test_optimization_and_circuit.py

Tests for the optimization-loop (Axis 4) and circuit-structure analyzers.

optimization_convergence is verified on clean synthetic trajectories where one
span equals one optimizer step — that isolates the analyzer logic from the
batched-evaluation confounding present in real QML training traces.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hilbertbench.models import Encoding, Kind
from hilbertbench.recorder.tape import HilbertTape
from hilbertbench.analysis import optimization_convergence, circuit_structure


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    d = tmp_path / "runs"
    d.mkdir()
    return d


def build_param_trajectory(runs, thetas):
    """One span per step; each carries its parameter vector and a quadratic cost."""
    dummy = "sha256:" + "a" * 64
    with HilbertTape(runs) as tape:
        for theta in thetas:
            theta = np.asarray(theta, dtype=float)
            with tape.execution_span(payload_ref=dummy) as span:
                span.outcome_ref = span.attach_inline(
                    json.dumps(float(np.sum(theta ** 2))),
                    kind="execution_outcome", encoding="json",
                )
                span.attach_inline(
                    json.dumps(theta.tolist()), kind="parameters", encoding="json"
                )
    return tape.dir_path


# ── optimization_convergence ──────────────────────────────────────────────────

class TestOptimizationConvergence:

    def test_converging_trajectory(self, runs):
        thetas = [np.array([3.0, 3.0]) * (0.85 ** k) for k in range(40)]
        r = optimization_convergence(build_param_trajectory(runs, thetas))
        assert r["status"] == "Converged"
        assert r["movement_ratio"] < 0.1
        assert r["outcome_trend"] < 0  # cost decreasing

    def test_constant_step_still_improving(self, runs):
        thetas = [np.array([0.2 * k, 0.2 * k]) for k in range(40)]
        r = optimization_convergence(build_param_trajectory(runs, thetas))
        assert r["status"] == "Still Improving"
        assert r["movement_ratio"] == pytest.approx(1.0, abs=0.05)

    def test_converging_path_length_positive(self, runs):
        thetas = [np.array([2.0, 2.0]) * (0.9 ** k) for k in range(20)]
        r = optimization_convergence(build_param_trajectory(runs, thetas))
        assert r["total_path_length"] > 0

    def test_insufficient_data(self, runs):
        r = optimization_convergence(build_param_trajectory(runs, [[1.0], [0.5]]))
        assert r["status"] == "Insufficient Data"
        assert r["num_steps"] == 2

    def test_outcome_envelope_reported(self, runs):
        thetas = [np.array([3.0, 3.0]) * (0.8 ** k) for k in range(30)]
        r = optimization_convergence(build_param_trajectory(runs, thetas))
        assert r["outcome_initial"] is not None
        assert r["outcome_final"] is not None
        assert r["outcome_min"] <= r["outcome_max"]
        # cost should drop from start to finish
        assert r["outcome_final"] < r["outcome_initial"]

    def test_accepts_path_and_trace(self, runs):
        from hilbertbench.trace import HilbertTrace
        run_dir = build_param_trajectory(runs, [np.array([1.0, 1.0]) * (0.9 ** k) for k in range(10)])
        a = optimization_convergence(run_dir)
        b = optimization_convergence(HilbertTrace(run_dir))
        assert a["total_path_length"] == pytest.approx(b["total_path_length"])


# ── circuit_structure ─────────────────────────────────────────────────────────

def build_qasm_trace(runs, tmp_path, qasm: str):
    f = tmp_path / "c.qasm"
    f.write_text(qasm)
    with HilbertTape(runs) as tape:
        ref = tape.attach_artifact(f, kind=Kind.circuit_qasm, encoding=Encoding.openqasm)
        with tape.execution_span(payload_ref=ref) as span:
            span.attach_inline("0.5", kind="execution_outcome", encoding="json")
    return tape.dir_path


BELL_QASM = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
"""

PARAM_QASM = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
ry(_p0) q[0];
ry(_p1) q[1];
cx q[0],q[1];
rz(_p2) q[0];
measure q[0] -> c[0];
"""

# hardware-transpiled (ISA) circuit: physical $N qubits, no register
# declaration, QASM3 measure-assignment syntax — what real-device and
# runtime local-mode traces actually contain
ISA_QASM = """OPENQASM 3.0;
include "stdgates.inc";
input float[64] t_0;
bit[2] c;
sx $0;
rz(t_0 + pi) $0;
sx $0;
rz(3*pi) $0;
cx $0, $1;
c[0] = measure $0;
c[1] = measure $1;
"""


class TestCircuitStructure:

    def test_bell_circuit(self, runs, tmp_path):
        r = circuit_structure(build_qasm_trace(runs, tmp_path, BELL_QASM))
        assert r["status"] == "OK"
        p = r["primary"]
        assert p["num_qubits"] == 2
        assert p["single_qubit_gates"] == 1   # h
        assert p["entangling_gates"] == 1     # cx
        assert p["total_gates"] == 2
        assert p["num_measurements"] == 2
        assert p["num_parameters"] == 0
        assert p["gate_counts"] == {"h": 1, "cx": 1}

    def test_bell_depth(self, runs, tmp_path):
        r = circuit_structure(build_qasm_trace(runs, tmp_path, BELL_QASM))
        # h on q0 (layer 1) then cx q0,q1 (layer 2) → depth 2
        assert r["primary"]["depth"] == 2

    def test_parametric_circuit(self, runs, tmp_path):
        r = circuit_structure(build_qasm_trace(runs, tmp_path, PARAM_QASM))
        p = r["primary"]
        assert p["num_parameters"] == 3       # _p0, _p1, _p2
        assert p["entangling_gates"] == 1     # cx
        assert p["single_qubit_gates"] == 3   # 2x ry + 1x rz
        assert p["entangling_fraction"] == pytest.approx(1 / 4)

    def test_no_qasm_circuit(self, runs):
        # Trace with only inline outcomes, no circuit_qasm artifact
        dummy = "sha256:" + "b" * 64
        with HilbertTape(runs) as tape:
            with tape.execution_span(payload_ref=dummy) as span:
                span.attach_inline("0.5", kind="execution_outcome", encoding="json")
        r = circuit_structure(tape.dir_path)
        assert r["status"] == "No QASM circuit recorded"
        assert r["num_circuits"] == 0

    def test_entangling_fraction_bounds(self, runs, tmp_path):
        r = circuit_structure(build_qasm_trace(runs, tmp_path, PARAM_QASM))
        frac = r["primary"]["entangling_fraction"]
        assert 0.0 <= frac <= 1.0

    def test_isa_circuit_physical_qubits(self, runs, tmp_path):
        # Regression: hardware ISA circuits previously parsed as empty
        # (0 qubits, 0 gates), making noise_profile report fidelity 1.0
        # on every real-device trace.
        r = circuit_structure(build_qasm_trace(runs, tmp_path, ISA_QASM))
        assert r["status"] == "OK"
        p = r["primary"]
        assert p["num_qubits"] == 2           # distinct $0, $1
        assert p["single_qubit_gates"] == 4   # 2x sx + 2x rz
        assert p["entangling_gates"] == 1     # cx
        assert p["num_measurements"] == 2     # QASM3 assignment form
        assert p["num_parameters"] == 1       # input float t_0
        assert p["gate_counts"] == {"sx": 2, "rz": 2, "cx": 1}

    def test_isa_circuit_depth(self, runs, tmp_path):
        # sx, rz, sx, rz stack on $0 (layers 1-4), cx joins $0/$1 → 5
        r = circuit_structure(build_qasm_trace(runs, tmp_path, ISA_QASM))
        assert r["primary"]["depth"] == 5
