"""
tests/integrations/test_qiskit_sampler.py

Tier 2 integration tests for HilbertSamplerProxy.
Uses real Qiskit circuits but keeps them minimal (1–2 qubits, few shots).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.primitives import StatevectorSampler

from hilbertbench.integrations.qiskit import HilbertSamplerProxy
from hilbertbench.recorder.tape import HilbertTape


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    d = tmp_path / "runs"
    d.mkdir()
    return d


def read_spans(tape: HilbertTape) -> list[dict]:
    return [json.loads(l) for l in (tape.dir_path / "events.jsonl").read_text().splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# 1. Basic sampling
# ---------------------------------------------------------------------------

class TestSamplerBasic:

    def test_one_span_per_pub(self, runs):
        """Each PUB produces exactly one span."""
        bell = QuantumCircuit(2)
        bell.h(0); bell.cx(0, 1); bell.measure_all()

        with HilbertTape(runs) as tape:
            proxy = HilbertSamplerProxy(tape)
            proxy.run([(bell, None, 32), (bell, None, 32)])

        spans = read_spans(tape)
        assert len(spans) == 2

    def test_outcome_inline_with_counts(self, runs):
        """Bitstring counts are stored inline as JSON, not as files."""
        qc = QuantumCircuit(1)
        qc.h(0); qc.measure_all()

        with HilbertTape(runs) as tape:
            proxy = HilbertSamplerProxy(tape)
            proxy.run([(qc, None, 128)])

        spans = read_spans(tape)
        s = spans[0]
        assert s["outcome_ref"] is not None
        inline = s["inline_artifacts"]
        assert s["outcome_ref"] in inline

        outcome = json.loads(inline[s["outcome_ref"]]["data"])
        # Should have register 'meas' with bitstring counts summing to 128
        assert "meas" in outcome
        total = sum(outcome["meas"]["counts"].values())
        assert total == 128

    def test_no_outcome_files_on_disk(self, runs):
        """All data is inline — artifacts/ holds only QASM, not outcomes."""
        qc = QuantumCircuit(2)
        qc.h(0); qc.cx(0, 1); qc.measure_all()

        with HilbertTape(runs) as tape:
            proxy = HilbertSamplerProxy(tape)
            proxy.run([(qc, None, 64)])

        json_files = list((tape.dir_path / "artifacts").rglob("*.json"))
        assert len(json_files) == 0

    def test_circuit_deduplication(self, runs):
        """Same circuit template across many shots produces only one QASM file."""
        qc = QuantumCircuit(2)
        qc.h(0); qc.cx(0, 1); qc.measure_all()

        with HilbertTape(runs) as tape:
            proxy = HilbertSamplerProxy(tape)
            proxy.run([(qc, None, 16) for _ in range(10)])

        qasm_files = list((tape.dir_path / "artifacts").rglob("*.qasm"))
        assert len(qasm_files) == 1  # content-addressed dedup

    def test_span_status_completed(self, runs):
        qc = QuantumCircuit(1); qc.x(0); qc.measure_all()
        with HilbertTape(runs) as tape:
            HilbertSamplerProxy(tape).run([(qc, None, 8)])
        assert read_spans(tape)[0]["status"] == "COMPLETED"


# ---------------------------------------------------------------------------
# 2. Parametric circuits
# ---------------------------------------------------------------------------

class TestSamplerParametric:

    def test_parameter_bindings_captured(self, runs):
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.ry(theta, 0); qc.measure_all()
        params = np.array([[0.5], [1.0], [1.5]])

        with HilbertTape(runs) as tape:
            HilbertSamplerProxy(tape).run([(qc, params, 32)])

        spans = read_spans(tape)
        assert len(spans) == 1
        inline = spans[0]["inline_artifacts"]
        param_arts = [a for a in inline.values() if a["kind"] == "parameters"]
        assert len(param_arts) == 1
        captured = json.loads(param_arts[0]["data"])
        # Should contain the parameter array flattened
        assert len(captured) > 0

    def test_different_params_different_outcomes(self, runs):
        theta = Parameter("theta")
        qc = QuantumCircuit(1); qc.ry(theta, 0); qc.measure_all()

        with HilbertTape(runs) as tape:
            HilbertSamplerProxy(tape).run([
                (qc, np.array([[0.0]]), 256),  # all |0>
                (qc, np.array([[np.pi]]), 256),  # all |1>
            ])

        spans = read_spans(tape)
        def get_counts(s):
            ref = s["outcome_ref"]
            return json.loads(s["inline_artifacts"][ref]["data"])["meas"]["counts"]

        counts_0 = get_counts(spans[0])
        counts_1 = get_counts(spans[1])
        # theta=0: should be almost all '0'
        assert counts_0.get("0", 0) > 200
        # theta=pi: should be almost all '1'
        assert counts_1.get("1", 0) > 200


# ---------------------------------------------------------------------------
# 3. Proxy transparency
# ---------------------------------------------------------------------------

class TestSamplerTransparency:

    def test_job_result_unchanged(self, runs):
        """The job returned by the proxy produces the same result as unproxied."""
        qc = QuantumCircuit(1); qc.h(0); qc.measure_all()

        # Direct run
        direct_job = StatevectorSampler().run([(qc, None, 512)])
        direct_counts = direct_job.result()[0].data.meas.get_counts()

        # Proxied run (same circuit, same shots → statevector sim is deterministic)
        with HilbertTape(runs) as tape:
            job = HilbertSamplerProxy(tape).run([(qc, None, 512)])
        proxy_counts = job.result()[0].data.meas.get_counts()

        # Both should have the same set of observed bitstrings
        assert set(direct_counts.keys()) == set(proxy_counts.keys())

    def test_shots_in_execution_completed_event(self, runs):
        """EXECUTION_COMPLETED event carries the actual shot count."""
        qc = QuantumCircuit(1); qc.h(0); qc.measure_all()
        with HilbertTape(runs) as tape:
            HilbertSamplerProxy(tape).run([(qc, None, 77)])

        span = read_spans(tape)[0]
        events = span["events"]
        completed = next(e for e in events if e["event_type"] == "EXECUTION_COMPLETED")
        attrs = json.loads(completed["attributes"]) if isinstance(completed["attributes"], str) else completed["attributes"]
        assert attrs["shots"] == 77

    def test_tape_closed_skips_recording(self, runs):
        """After tape closes, proxy forwards calls but records nothing."""
        qc = QuantumCircuit(1); qc.h(0); qc.measure_all()
        with HilbertTape(runs) as tape:
            proxy = HilbertSamplerProxy(tape)
        # tape is now closed
        job = proxy.run([(qc, None, 8)])  # should not raise
        result = job.result()             # should still work
        assert result[0].data.meas.num_shots == 8
        assert len(read_spans(tape)) == 0

    def test_deepcopy_preserves_tape(self, runs):
        import copy
        qc = QuantumCircuit(1); qc.h(0); qc.measure_all()
        with HilbertTape(runs) as tape:
            proxy = HilbertSamplerProxy(tape)
            proxy_copy = copy.deepcopy(proxy)
            proxy_copy.run([(qc, None, 8)])
        assert len(read_spans(tape)) == 1
