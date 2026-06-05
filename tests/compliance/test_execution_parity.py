"""
tests/compliance/test_execution_parity.py

Validates the proposal's central "1:1 Execution Parity" claim (§2.1): wrapping a
primitive with HilbertBench must NOT change what reaches the backend — same
number of executions, same circuits, same shots — and must NOT change the
results. This is what makes the recorder a non-confounding observer.

Parity is verified with counting shims around the real primitive/device, so we
measure exactly how many times — and with what — the backend was invoked, with
and without the HilbertBench proxy in the path.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from hilbertbench.recorder.tape import HilbertTape


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    d = tmp_path / "runs"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# Estimator parity
# ---------------------------------------------------------------------------

class TestEstimatorParity:

    def _counting_estimator(self):
        from qiskit.primitives import StatevectorEstimator, BaseEstimatorV2

        class CountingEstimator(BaseEstimatorV2):
            def __init__(self, inner):
                self.inner = inner
                self.run_calls = 0
                self.total_pubs = 0
                self.pub_fingerprints = []

            def run(self, pubs, **kw):
                self.run_calls += 1
                self.total_pubs += len(pubs)
                for p in pubs:
                    if len(p) > 2 and p[2] is not None:
                        self.pub_fingerprints.append(np.array(p[2]).tolist())
                    else:
                        self.pub_fingerprints.append(None)
                return self.inner.run(pubs, **kw)

            @property
            def options(self):
                return self.inner.options

        return CountingEstimator(StatevectorEstimator())

    def _workload(self):
        from qiskit.circuit import QuantumCircuit, Parameter
        from qiskit.quantum_info import SparsePauliOp
        theta = Parameter("t")
        qc = QuantumCircuit(2)
        qc.ry(theta, 0); qc.cx(0, 1)
        obs = SparsePauliOp("ZZ")
        return [[(qc, obs, np.array([[i * 0.1]]))] for i in range(20)]

    def test_backend_called_once_per_user_call(self, runs):
        from hilbertbench.integrations.qiskit import HilbertEstimatorProxy
        work = self._workload()

        base = self._counting_estimator()
        for pubs in work:
            base.run(pubs).result()

        proxied = self._counting_estimator()
        with HilbertTape(runs) as tape:
            proxy = HilbertEstimatorProxy(tape, real_estimator=proxied)
            for pubs in work:
                proxy.run(pubs).result()

        # No silent extra executions: the backend saw exactly the same calls.
        assert proxied.run_calls == base.run_calls == len(work)
        assert proxied.total_pubs == base.total_pubs

    def test_pubs_unchanged(self, runs):
        from hilbertbench.integrations.qiskit import HilbertEstimatorProxy
        work = self._workload()

        base = self._counting_estimator()
        for pubs in work:
            base.run(pubs).result()

        proxied = self._counting_estimator()
        with HilbertTape(runs) as tape:
            proxy = HilbertEstimatorProxy(tape, real_estimator=proxied)
            for pubs in work:
                proxy.run(pubs).result()

        # The parameter bindings submitted to the backend are bit-identical.
        assert proxied.pub_fingerprints == base.pub_fingerprints

    def test_results_identical(self, runs):
        from qiskit.primitives import StatevectorEstimator
        from hilbertbench.integrations.qiskit import HilbertEstimatorProxy
        work = self._workload()

        def evs(job):
            return float(np.ravel(job.result()[0].data.evs)[0])

        base_vals = [evs(StatevectorEstimator().run(p)) for p in work]

        with HilbertTape(runs) as tape:
            proxy = HilbertEstimatorProxy(
                tape, real_estimator=StatevectorEstimator()
            )
            proxy_vals = [evs(proxy.run(p)) for p in work]

        assert proxy_vals == base_vals  # bit-identical, not just close


# ---------------------------------------------------------------------------
# Sampler parity
# ---------------------------------------------------------------------------

class TestSamplerParity:

    def _counting_sampler(self):
        from qiskit.primitives import StatevectorSampler, BaseSamplerV2

        class CountingSampler(BaseSamplerV2):
            def __init__(self, inner):
                self.inner = inner
                self.run_calls = 0
                self.shots_seen = []

            def run(self, pubs, **kw):
                self.run_calls += 1
                for p in pubs:
                    self.shots_seen.append(p[2] if len(p) > 2 else None)
                return self.inner.run(pubs, **kw)

            @property
            def options(self):
                return self.inner.options

        return CountingSampler(StatevectorSampler())

    def _circuit(self):
        from qiskit.circuit import QuantumCircuit
        qc = QuantumCircuit(2)
        qc.h(0); qc.cx(0, 1); qc.measure_all()
        return qc

    def test_backend_called_once_per_user_call(self, runs):
        from hilbertbench.integrations.qiskit import HilbertSamplerProxy
        qc = self._circuit()

        base = self._counting_sampler()
        for _ in range(15):
            base.run([(qc, None, 128)])

        proxied = self._counting_sampler()
        with HilbertTape(runs) as tape:
            proxy = HilbertSamplerProxy(tape, real_sampler=proxied)
            for _ in range(15):
                proxy.run([(qc, None, 128)])

        assert proxied.run_calls == base.run_calls == 15

    def test_shots_unchanged(self, runs):
        from hilbertbench.integrations.qiskit import HilbertSamplerProxy
        qc = self._circuit()

        proxied = self._counting_sampler()
        with HilbertTape(runs) as tape:
            proxy = HilbertSamplerProxy(tape, real_sampler=proxied)
            for _ in range(10):
                proxy.run([(qc, None, 256)])

        # No silent shot inflation: every backed call kept the requested 256.
        assert all(s == 256 for s in proxied.shots_seen)
        assert len(proxied.shots_seen) == 10


# ---------------------------------------------------------------------------
# PennyLane parity
# ---------------------------------------------------------------------------

class TestPennyLaneParity:

    def _count_executions(self, runs, use_proxy: bool):
        import pennylane as qml
        from pennylane import numpy as pnp
        from hilbertbench.integrations.pennylane import (
            HilbertPennyLaneDeviceProxy,
        )

        real = qml.device("default.qubit", wires=2)
        counter = {"n": 0}
        orig = real.execute

        def counting(tapes, *a, **k):
            counter["n"] += 1
            return orig(tapes, *a, **k)

        real.execute = counting

        if use_proxy:
            tape_ctx = HilbertTape(runs)
            tape_ctx.__enter__()
            dev = HilbertPennyLaneDeviceProxy(real, tape_ctx)
        else:
            tape_ctx = None
            dev = real

        @qml.qnode(dev, diff_method="parameter-shift")
        def circ(x):
            qml.RY(x, wires=0); qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        vals = [float(circ(pnp.array(0.1 * i))) for i in range(10)]

        if tape_ctx is not None:
            tape_ctx.__exit__(None, None, None)

        return counter["n"], vals

    def test_device_executed_same_number_of_times(self, runs):
        base_n, base_vals = self._count_executions(runs, use_proxy=False)
        proxy_n, proxy_vals = self._count_executions(runs, use_proxy=True)

        assert proxy_n == base_n  # no extra device executions
        assert np.allclose(base_vals, proxy_vals)


# ---------------------------------------------------------------------------
# Latency overhead
# ---------------------------------------------------------------------------

class TestOverhead:

    def test_estimator_overhead_under_budget(self, runs):
        """
        Per-call recording overhead must stay small (proposal target: <5ms).
        We assert a generous CI-safe ceiling; the demo reports the real number
        (typically well under 1ms on a workstation).
        """
        import time
        from qiskit.circuit import QuantumCircuit, Parameter
        from qiskit.quantum_info import SparsePauliOp
        from qiskit.primitives import StatevectorEstimator
        from hilbertbench.integrations.qiskit import HilbertEstimatorProxy

        theta = Parameter("t")
        qc = QuantumCircuit(2); qc.ry(theta, 0); qc.cx(0, 1)
        obs = SparsePauliOp("ZZ")
        work = [[(qc, obs, np.array([[i * 0.1]]))] for i in range(30)]

        t0 = time.perf_counter()
        for p in work:
            StatevectorEstimator().run(p).result()
        base = time.perf_counter() - t0

        with HilbertTape(runs) as tape:
            proxy = HilbertEstimatorProxy(
                tape, real_estimator=StatevectorEstimator()
            )
            t0 = time.perf_counter()
            for p in work:
                proxy.run(p).result()
            proxied = time.perf_counter() - t0

        overhead_ms = (proxied - base) / len(work) * 1000.0
        # Generous ceiling to stay robust on loaded CI; real value is sub-ms.
        assert overhead_ms < 25.0, (
            f"overhead {overhead_ms:.2f} ms/call exceeds ceiling"
        )
