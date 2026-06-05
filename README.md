# HilbertBench

> Diagnose quantum ML experiments without changing a line of algorithm code.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-291%20passing-brightgreen)]()

HilbertBench is a **non-intrusive diagnostic framework** for quantum machine
learning. You wrap one object — an Estimator, Sampler, or PennyLane device —
and every circuit execution is silently recorded into an append-only trace.
Nothing else in your code changes.

After the run you call built-in analyzers to diagnose trainability, shot
noise, convergence, circuit structure, expressibility, and hardware noise —
all from the evidence already in the trace.

---

## Install

```bash
# core + Qiskit integration + Parquet storage
pip install hilbertbench[qiskit,storage]

# core + PennyLane integration
pip install hilbertbench[pennylane,storage]

# everything
pip install hilbertbench[full]
```

---

## 30-second example

```python
from qiskit.primitives import StatevectorEstimator
from hilbertbench.integrations.qiskit import HilbertEstimatorProxy
from hilbertbench.recorder.tape import HilbertTape
from hilbertbench.analysis import detect_barren_plateau
from hilbertbench import HilbertTrace

# Step 1 — open a tape and wrap the estimator (the only code change)
with HilbertTape("runs/my_vqe", tags={"algorithm": "vqe"}) as tape:
    estimator = HilbertEstimatorProxy(tape)   # ← swap this in
    # ... your existing VQE / optimizer loop here, completely unchanged ...

# Step 2 — analyze the sealed trace
trace = HilbertTrace(tape.dir_path)
result = detect_barren_plateau(trace)
print(result)
```

```
{'status': 'Trainable', 'variance': 0.215, 'std_dev': 0.464,
 'num_evaluations': 20, 'threshold': 0.005,
 'variance_ci': [0.098, 0.371], 'confidence_level': 0.95,
 'verdict_confidence': 'high'}
```

---

## What gets recorded

Every circuit execution becomes a **span** inside a **trace**. Each span
captures:

- The circuit (OpenQASM, content-addressed so duplicates are stored once)
- The bound parameter values
- The observable and the expectation value (Estimator) or bitstring counts
  (Sampler)
- UTC timestamps and sequence number
- Shots, backend ID, and device calibration snapshot (when available)

Traces are append-only and cryptographically sealed. The recorder never
re-executes circuits, never modifies shot counts, and never swallows
exceptions.

---

## Built-in analyzers

| Function | What it answers |
|---|---|
| `detect_barren_plateau` | Is the cost-landscape variance exponentially suppressed? |
| `shot_noise_ratio` | Is the optimizer chasing signal or shot noise? |
| `optimization_convergence` | Is the parameter trajectory still moving? |
| `circuit_structure` | How deep, how many qubits, what gate mix? |
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
| IBM Quantum (hardware) | `HilbertEstimatorProxy` with `EstimatorV2(mode=backend)` | `hilbertbench[qiskit]` |
| PennyLane | `HilbertPennyLaneDeviceProxy` | `hilbertbench[pennylane]` |

---

## Documentation

Full documentation — concepts, guides, analyzer reference:
**[mamuncseru.github.io/hilbertbench](https://mamuncseru.github.io/hilbertbench/)**
*(or run `mkdocs serve` locally after `pip install -r requirements-docs.txt`)*

---

## License

MIT. See [LICENSE](LICENSE).
