"""
tests/integrations/test_pennylane_qasm_reproducibility.py

Proves that the templated OpenQASM stored for PennyLane traces is *useful*:
template + recorded parameters reconstructs a valid circuit whose outcome
matches what was recorded — verified by re-executing through Qiskit (a
different framework), proving the QASM is portable and complete.

The stored template carries _p{i} placeholders (so it deduplicates across
training steps); these tests confirm the placeholders bind correctly with the
separately-stored parameter values.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
import pennylane as qml
from pennylane import numpy as pnp

from hilbertbench.integrations.pennylane import (
    HilbertPennyLaneDeviceProxy,
    _qasm_to_template,
)
from hilbertbench.recorder.tape import HilbertTape


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    d = tmp_path / "runs"
    d.mkdir()
    return d


def read_spans(tape: HilbertTape) -> list[dict]:
    return [json.loads(l) for l in (tape.dir_path / "events.jsonl").read_text().splitlines() if l.strip()]


# ── Reconstruction helpers (mirror the demo verifier) ─────────────────────────

def bind_template(template: str, params: list[float]) -> str:
    def repl(m: re.Match) -> str:
        return repr(float(params[int(m.group(1))]))
    return re.sub(r"_p(\d+)", repl, template)


def expval_z_from_qasm(concrete_qasm: str, wire: int) -> float:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector, SparsePauliOp

    lines = [l for l in concrete_qasm.splitlines()
             if not l.strip().startswith(("measure", "creg"))]
    qc = QuantumCircuit.from_qasm_str("\n".join(lines))
    sv = Statevector.from_instruction(qc)

    n = qc.num_qubits
    label = ["I"] * n
    label[n - 1 - wire] = "Z"  # qiskit little-endian
    return float(np.real(sv.expectation_value(SparsePauliOp("".join(label)))))


# ── Unit-level: template helper ───────────────────────────────────────────────

class TestTemplateHelper:

    def test_placeholders_replace_numeric_literals(self):
        qasm = "ry(0.5) q[0];\nrz(1.25) q[1];"
        tmpl = _qasm_to_template(qasm)
        assert "_p0" in tmpl and "_p1" in tmpl
        assert "0.5" not in tmpl and "1.25" not in tmpl

    def test_wire_indices_untouched(self):
        qasm = "cx q[0],q[1];\nry(0.5) q[10];"
        tmpl = _qasm_to_template(qasm)
        # wire indices live in [...] not (...) — must not become placeholders
        assert "q[0]" in tmpl and "q[1]" in tmpl and "q[10]" in tmpl

    def test_multi_param_gate(self):
        qasm = "u(0.1,0.2,0.3) q[0];"
        tmpl = _qasm_to_template(qasm)
        assert "_p0" in tmpl and "_p1" in tmpl and "_p2" in tmpl

    def test_template_stable_across_values(self):
        a = _qasm_to_template("ry(0.5) q[0];")
        b = _qasm_to_template("ry(9.9) q[0];")
        assert a == b  # same structure → identical template


# ── End-to-end: full round-trip reproducibility ──────────────────────────────

class TestQASMRoundTrip:

    def _run_training(self, runs, n_steps=4):
        real_dev = qml.device("default.qubit", wires=2)
        with HilbertTape(runs) as tape:
            proxy = HilbertPennyLaneDeviceProxy(real_dev, tape)

            @qml.qnode(proxy, diff_method="parameter-shift")
            def circuit(x, w):
                qml.AngleEmbedding(x, wires=[0, 1], rotation="Y")
                qml.StronglyEntanglingLayers(w, wires=[0, 1])
                return qml.expval(qml.PauliZ(0))

            shape = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=2)
            w = pnp.array(np.random.default_rng(0).uniform(0, np.pi, shape), requires_grad=True)
            opt = qml.AdamOptimizer(0.1)
            x = pnp.array([0.4, 0.8])
            for _ in range(n_steps):
                w, _ = opt.step_and_cost(lambda ww: circuit(x, ww), w)
        return tape

    def test_template_plus_params_reproduces_outcome(self, runs):
        """The core guarantee: bind(template, params) reproduces the recorded expval."""
        tape = self._run_training(runs)
        spans = read_spans(tape)
        catalog = json.loads((tape.dir_path / "catalog.json").read_text())["artifacts"]

        verified = 0
        for span in spans:
            inline = span.get("inline_artifacts") or {}
            payload_ref = span["payload_ref"]
            if payload_ref not in catalog or catalog[payload_ref]["kind"] != "circuit_qasm":
                continue

            template = (tape.dir_path / catalog[payload_ref]["file_path"]).read_text()

            params = obs = recorded = None
            for ref, art in inline.items():
                if art["kind"] == "parameters":
                    params = json.loads(art["data"])
                elif art["kind"] == "observables":
                    obs = json.loads(art["data"])
                if ref == span.get("outcome_ref"):
                    recorded = json.loads(art["data"])
            if params is None or obs is None or recorded is None:
                continue

            # Only verify Z(w) observables
            m = re.match(r"Z\((\d+)\)", obs[0])
            if not m:
                continue
            wire = int(m.group(1))

            recorded_val = recorded[0] if isinstance(recorded, list) else float(recorded)
            concrete = bind_template(template, params)
            recomputed = expval_z_from_qasm(concrete, wire)

            assert abs(recomputed - recorded_val) < 1e-6, (
                f"QASM reproduction mismatch: recorded={recorded_val} "
                f"reconstructed={recomputed}"
            )
            verified += 1

        assert verified > 0, "No QASM spans were verified"

    def test_single_qubit_rotation_round_trip(self, runs):
        """Minimal case: one RY rotation, check exact reproduction."""
        real_dev = qml.device("default.qubit", wires=1)
        with HilbertTape(runs) as tape:
            proxy = HilbertPennyLaneDeviceProxy(real_dev, tape)

            @qml.qnode(proxy, diff_method="parameter-shift")
            def circuit(theta):
                qml.RY(theta, wires=0)
                return qml.expval(qml.PauliZ(0))

            circuit(pnp.array(0.7))

        span = read_spans(tape)[0]
        catalog = json.loads((tape.dir_path / "catalog.json").read_text())["artifacts"]
        template = (tape.dir_path / catalog[span["payload_ref"]]["file_path"]).read_text()

        inline = span["inline_artifacts"]
        params = next(json.loads(a["data"]) for a in inline.values() if a["kind"] == "parameters")
        recorded = json.loads(inline[span["outcome_ref"]]["data"])
        recorded_val = recorded[0] if isinstance(recorded, list) else float(recorded)

        # Analytic: <Z> after RY(theta) on |0> = cos(theta)
        assert abs(recorded_val - np.cos(0.7)) < 1e-6

        concrete = bind_template(template, params)
        recomputed = expval_z_from_qasm(concrete, 0)
        assert abs(recomputed - recorded_val) < 1e-6
