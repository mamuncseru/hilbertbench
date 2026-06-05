# HilbertBench

> Diagnose quantum ML experiments without changing a line of algorithm code.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-291%20passing-brightgreen)]()
[![Docs](https://img.shields.io/badge/docs-readthedocs-blueviolet)](https://hilbertbench.readthedocs.io)

HilbertBench is a **non-intrusive diagnostic framework** for quantum machine
learning. You wrap one object — an Estimator, Sampler, or PennyLane device —
and every circuit execution is silently recorded into an append-only,
cryptographically sealed trace. Nothing else in your code changes.

After the run you call built-in analyzers to diagnose trainability, shot
noise, optimization convergence, circuit structure, expressibility, and
hardware noise — all from the evidence already in the trace.

---

## Install

```bash
# Qiskit + Parquet storage
pip install hilbertbench[qiskit,storage]

# PennyLane + Parquet storage
pip install hilbertbench[pennylane,storage]

# Everything
pip install hilbertbench[full]
```

---

## The one-line change

```python
from hilbertbench.integrations.qiskit import HilbertEstimatorProxy
from hilbertbench.recorder.tape import HilbertTape
from hilbertbench.analysis import detect_barren_plateau
from hilbertbench import HilbertTrace

# Wrap the estimator — everything else stays the same
with HilbertTape("runs/my_vqe", tags={"algorithm": "vqe"}) as tape:
    estimator = HilbertEstimatorProxy(tape)   # ← the only change
    # ... your existing optimizer loop, completely unchanged ...

# Analyze the sealed trace
trace = HilbertTrace(tape.dir_path)
print(detect_barren_plateau(trace))
# {'status': 'Trainable', 'variance': 0.215, 'variance_ci': [0.098, 0.371], ...}
```

---

## Built-in analyzers

| Function | What it answers |
|---|---|
| `detect_barren_plateau` | Is the cost-landscape variance exponentially suppressed? |
| `shot_noise_ratio` | Is the optimizer chasing signal or shot noise? |
| `optimization_convergence` | Is the parameter trajectory still moving? |
| `circuit_structure` | How deep, how many qubits, what gate composition? |
| `kl_expressibility` | How uniformly does the ansatz cover Hilbert space? |
| `noise_profile` | What hardware noise levels did the run experience? |

```python
from hilbertbench.analysis import summary

report = summary(trace)   # runs all analyzers, returns one dict
```

---

## Integrations

| Framework | Proxy class | Install extra |
|---|---|---|
| Qiskit V2 Estimator | `HilbertEstimatorProxy` | `hilbertbench[qiskit]` |
| Qiskit V2 Sampler | `HilbertSamplerProxy` | `hilbertbench[qiskit]` |
| IBM Quantum hardware | `HilbertEstimatorProxy` with `EstimatorV2(mode=backend)` | `hilbertbench[qiskit]` |
| PennyLane | `HilbertPennyLaneDeviceProxy` | `hilbertbench[pennylane]` |

---

## Tutorials

Four end-to-end tutorials, each diagnosing a real quantum ML failure mode:

| # | Tutorial | Concepts |
|---|---|---|
| 01 | [Why Isn't My VQE Converging?](https://mamuncseru.github.io/hilbertbench/tutorials/vqe-barren-plateau/) | Barren plateau, cost-landscape variance |
| 02 | [Am I Using Enough Shots?](https://mamuncseru.github.io/hilbertbench/tutorials/shot-noise/) | Shot noise floor, SNR, shot budget |
| 03 | [Expressibility vs Trainability](https://mamuncseru.github.io/hilbertbench/tutorials/expressibility-vs-trainability/) | Active Mode, KL expressibility, Holmes 2022 |
| 04 | [How Hardware Noise Degrades Your Results](https://mamuncseru.github.io/hilbertbench/tutorials/hardware-noise/) | Circuit fidelity, gate error, noise profile |

---

## Documentation

| Resource | Contents |
|---|---|
| **[mamuncseru.github.io/hilbertbench](https://mamuncseru.github.io/hilbertbench/)** | Tutorials, landing page, concept guides |
| **[hilbertbench.readthedocs.io](https://hilbertbench.readthedocs.io)** | API reference, analyzer docs, trace format |

---

## Design guarantees

- **Non-intrusive** — the proxy never re-executes circuits, modifies shot counts, or alters your algorithm (INV-001)
- **Immutable traces** — every trace is append-only and sealed with a SHA-256 hash on close (INV-002)
- **Evidence, not verdicts** — raw execution data is stored; diagnostic conclusions are computed at read time and never written back (INV-006)
- **No silent failures** — every initiated span ends with a `COMPLETED` or `ERROR` event (INV-007)

---

## License

MIT. See [LICENSE](LICENSE).
