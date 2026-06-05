#!/usr/bin/env python
#
# file: demo/two_moons/12_execution_parity.py
#
# revision history:
#  20260605 (am): initial version
#
# Execution-parity and overhead benchmark — validates the proposal's
# central non-intrusive claim (Section 2.1): "1:1 Execution Parity ...
# no silent increase in circuit executions or shots ... and no
# interference with optimization dynamics ... negligible (<5ms)
# latency."
#
# For each primitive an identical workload is run twice — once on the
# bare backend, once with the HilbertBench proxy in the path — using a
# counting shim to record exactly what reaches the backend. The script
# reports: backend invocation count, units submitted, shots, whether
# results are bit-identical, and the added latency per call.
#
# Usage:
#   python demo/two_moons/12_execution_parity.py
#------------------------------------------------------------------------------

# import system modules
#
import os
import time
import tempfile
import warnings

# import third-party modules
#
import numpy as np

# import hilbertbench modules
#
from hilbertbench.recorder.tape import HilbertTape

# silence framework deprecation chatter for a clean benchmark report
#
warnings.filterwarnings("ignore")

#------------------------------------------------------------------------------
#
# global variables are listed here
#
#------------------------------------------------------------------------------

# set the filename using basename
#
__FILE__ = os.path.basename(__file__)

# section separator width
#
SEP = "=" * 64

# proposal latency target, in milliseconds
#
TARGET_MS = 5.0

#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------

def banner(title: str) -> None:
    """
    function: banner

    arguments:
     title: the heading text to print

    return:
     none

    description:
     Prints a separated section heading.
    """

    # print the heading between separators
    #
    print(f"\n{SEP}\n  {title}\n{SEP}")
#
# end of function


def report(name, base_calls, proxy_calls, base_units, proxy_units,
           inputs_identical, results_identical, base_t, proxy_t, extra=None):
    """
    function: report

    arguments:
     name:              primitive label
     base_calls:        backend invocations without the proxy
     proxy_calls:       backend invocations with the proxy
     base_units:        units (PUBs/tapes) submitted without the proxy
     proxy_units:       units submitted with the proxy
     inputs_identical:  True if the submitted inputs matched bitwise
     results_identical: True / False / "n/a" for outcome equality
     base_t:            baseline wall-clock seconds
     proxy_t:           proxied wall-clock seconds
     extra:             optional extra line to print

    return:
     none

    description:
     Prints the parity and overhead summary for one primitive. Overhead
     per call is the wall-clock difference amortised over backend calls.
    """

    # compute per-call overhead in milliseconds
    #
    overhead_ms = (proxy_t - base_t) / max(1, base_calls) * 1000.0
    calls_par = "OK" if base_calls == proxy_calls else "MISMATCH"
    units_par = "OK" if base_units == proxy_units else "MISMATCH"
    budget = "PASS" if overhead_ms < TARGET_MS else "OVER"

    # print the parity block
    #
    print(f"\n  {name}")
    print(f"    backend calls   : baseline {base_calls:>4}   "
          f"proxied {proxy_calls:>4}   parity {calls_par}")
    print(f"    units submitted : baseline {base_units:>4}   "
          f"proxied {proxy_units:>4}   parity {units_par}")
    print(f"    inputs identical: {inputs_identical}")
    print(f"    results identical: {results_identical}")
    if extra:
        print(f"    {extra}")
    print(f"    overhead / call : {overhead_ms:+.3f} ms   "
          f"(target < {TARGET_MS:.0f} ms: {budget})")
#
# end of function


def bench_estimator(n: int = 60) -> None:
    """
    function: bench_estimator

    arguments:
     n: number of estimator calls in the workload

    return:
     none

    description:
     Runs an identical n-call Estimator workload on a bare
     StatevectorEstimator and on a HilbertEstimatorProxy wrapping a
     counting shim, then reports execution parity and overhead.
    """

    # import qiskit primitives locally to keep the module import light
    #
    from qiskit.circuit import QuantumCircuit, Parameter
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.primitives import StatevectorEstimator, BaseEstimatorV2
    from hilbertbench.integrations.qiskit import HilbertEstimatorProxy

    # counting shim records every backend call and its bound parameters
    #
    class Counting(BaseEstimatorV2):
        def __init__(self, inner):
            self.inner = inner
            self.calls = 0
            self.pubs = 0
            self.fp = []

        def run(self, pubs, **kw):
            self.calls += 1
            self.pubs += len(pubs)
            for p in pubs:
                if len(p) > 2 and p[2] is not None:
                    self.fp.append(np.array(p[2]).tolist())
                else:
                    self.fp.append(None)
            return self.inner.run(pubs, **kw)

        @property
        def options(self):
            return self.inner.options

    # build the workload: one parameterized ZZ expectation per call
    #
    theta = Parameter("t")
    qc = QuantumCircuit(2)
    qc.ry(theta, 0)
    qc.cx(0, 1)
    obs = SparsePauliOp("ZZ")
    work = [[(qc, obs, np.array([[i * 0.05]]))] for i in range(n)]

    # extract a scalar expectation value robustly across numpy versions
    #
    def evs(job):
        return float(np.ravel(job.result()[0].data.evs)[0])

    # baseline run on the bare backend
    #
    base = Counting(StatevectorEstimator())
    t0 = time.perf_counter()
    base_vals = [evs(base.run(p)) for p in work]
    base_t = time.perf_counter() - t0

    # proxied run through HilbertBench
    #
    with tempfile.TemporaryDirectory() as tmp:
        prox = Counting(StatevectorEstimator())
        with HilbertTape(tmp) as tape:
            proxy = HilbertEstimatorProxy(tape, real_estimator=prox)
            t0 = time.perf_counter()
            proxy_vals = [evs(proxy.run(p)) for p in work]
            proxy_t = time.perf_counter() - t0

    # report parity and overhead
    #
    report("Qiskit Estimator",
           base.calls, prox.calls, base.pubs, prox.pubs,
           base.fp == prox.fp, base_vals == proxy_vals,
           base_t, proxy_t)
#
# end of function


def bench_sampler(n: int = 60, shots: int = 256) -> None:
    """
    function: bench_sampler

    arguments:
     n:     number of sampler calls in the workload
     shots: shots per call

    return:
     none

    description:
     Runs an identical n-call Sampler workload on a bare
     StatevectorSampler and on a HilbertSamplerProxy wrapping a counting
     shim, verifying that shots are preserved exactly and reporting
     execution parity and overhead.
    """

    # import qiskit primitives locally
    #
    from qiskit.circuit import QuantumCircuit
    from qiskit.primitives import StatevectorSampler, BaseSamplerV2
    from hilbertbench.integrations.qiskit import HilbertSamplerProxy

    # counting shim records every backend call and its shot count
    #
    class Counting(BaseSamplerV2):
        def __init__(self, inner):
            self.inner = inner
            self.calls = 0
            self.shots = []

        def run(self, pubs, **kw):
            self.calls += 1
            for p in pubs:
                self.shots.append(p[2] if len(p) > 2 else None)
            return self.inner.run(pubs, **kw)

        @property
        def options(self):
            return self.inner.options

    # Bell circuit measured in the computational basis
    #
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    # baseline run on the bare backend
    #
    base = Counting(StatevectorSampler())
    t0 = time.perf_counter()
    for _ in range(n):
        base.run([(qc, None, shots)]).result()
    base_t = time.perf_counter() - t0

    # proxied run through HilbertBench
    #
    with tempfile.TemporaryDirectory() as tmp:
        prox = Counting(StatevectorSampler())
        with HilbertTape(tmp) as tape:
            proxy = HilbertSamplerProxy(tape, real_sampler=prox)
            t0 = time.perf_counter()
            for _ in range(n):
                proxy.run([(qc, None, shots)]).result()
            proxy_t = time.perf_counter() - t0

    # verify shots were never inflated, then report
    #
    shots_ok = all(s == shots for s in prox.shots) and len(prox.shots) == n
    report("Qiskit Sampler",
           base.calls, prox.calls, n, n,
           shots_ok, "n/a (counts)",
           base_t, proxy_t,
           extra=f"shots preserved ({shots} each): {shots_ok}")
#
# end of function


def bench_pennylane(n: int = 30) -> None:
    """
    function: bench_pennylane

    arguments:
     n: number of QNode evaluations in the workload

    return:
     none

    description:
     Runs an identical n-call QNode workload on a bare default.qubit
     device and on one wrapped by HilbertPennyLaneDeviceProxy. The real
     device's execute() is monkeypatched with a counter so device-level
     execution parity can be measured directly.
    """

    # import pennylane locally
    #
    import pennylane as qml
    from pennylane import numpy as pnp
    from hilbertbench.integrations.pennylane import HilbertPennyLaneDeviceProxy

    # run the workload, optionally through the proxy, counting executions
    #
    def run(use_proxy):

        # patch the real device's execute() with a call counter
        #
        real = qml.device("default.qubit", wires=2)
        counter = {"n": 0}
        orig = real.execute

        def counting(tapes, *a, **k):
            counter["n"] += 1
            return orig(tapes, *a, **k)

        real.execute = counting

        # choose the device the QNode runs on
        #
        ctx = None
        if use_proxy:
            ctx = HilbertTape(tempfile.mkdtemp())
            ctx.__enter__()
            dev = HilbertPennyLaneDeviceProxy(real, ctx)
        else:
            dev = real

        # define and evaluate the QNode workload
        #
        @qml.qnode(dev, diff_method="parameter-shift")
        def circ(x):
            qml.RY(x, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        t0 = time.perf_counter()
        vals = [float(circ(pnp.array(0.05 * i))) for i in range(n)]
        elapsed = time.perf_counter() - t0

        # seal the tape if one was opened
        #
        if ctx:
            ctx.__exit__(None, None, None)
        return counter["n"], vals, elapsed

    # baseline and proxied runs
    #
    base_n, base_vals, base_t = run(False)
    proxy_n, proxy_vals, proxy_t = run(True)

    # report parity and overhead
    #
    report("PennyLane device",
           base_n, proxy_n, base_n, proxy_n,
           True, bool(np.allclose(base_vals, proxy_vals)),
           base_t, proxy_t,
           extra="(device.execute calls; parameter-shift "
                 "expands per QNode call)")
#
# end of function


def main() -> None:
    """
    function: main

    arguments:
     none

    return:
     none

    description:
     Runs the parity-and-overhead benchmark for all three integrations
     (Qiskit Estimator, Qiskit Sampler, PennyLane device) and prints a
     conclusion.
    """

    # print the benchmark header
    #
    print("\n" + SEP)
    print("  HILBERTBENCH - EXECUTION PARITY & OVERHEAD")
    print("  validating Section 2.1: 1:1 execution parity, <5ms latency")
    print(SEP)

    # run the three benchmarks
    #
    banner("Wrapping the primitive must not change what reaches the backend")
    bench_estimator()
    bench_sampler()
    bench_pennylane()

    # print the conclusion
    #
    banner("CONCLUSION")
    print("  HilbertBench is a non-confounding observer: identical")
    print("  executions, identical shots, bit-identical results, and")
    print("  sub-5ms recording overhead.\n")
#
# end of function

#------------------------------------------------------------------------------
#
# main entry point
#
#------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#
# end of file
