# Getting Started

This page takes you from zero to a complete recorded trace with diagnostic
output in about 10 minutes.

---

## Install

HilbertBench's core has no external dependencies. Choose the extras that
match your stack.

```bash
# Default: runs the Qiskit workflow out of the box
# (trace core + Qiskit integration + Parquet storage)
pip install hilbertbench scipy

# PennyLane + Parquet storage
pip install hilbertbench[pennylane] scikit-learn

# Everything
pip install hilbertbench[full]
```

**Python 3.10 or later is required.**

---

## Step 1 — Write your circuit code as normal

HilbertBench does not require you to restructure your code. Start with
whatever you already have. Here is a minimal VQE:

```python
import numpy as np
from scipy.optimize import minimize
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator

theta = ParameterVector("θ", 2)
qc = QuantumCircuit(2)
qc.ry(theta[0], 0)
qc.ry(theta[1], 1)
qc.cx(0, 1)
observable = SparsePauliOp("ZZ")

estimator = StatevectorEstimator()

def cost(x):
    pv = x.reshape(1, -1)
    job = estimator.run([(qc, observable, pv)])
    return float(job.result()[0].data.evs.ravel()[0])

result = minimize(cost, np.random.uniform(0, 2 * np.pi, 2), method="COBYLA")
```

---

## Step 2 — Add the tape and swap the estimator

Two imports and three lines are the entire integration:

```python hl_lines="4 5 9 10 11"
import numpy as np
from scipy.optimize import minimize
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from hilbertbench.integrations.qiskit import HilbertEstimatorProxy  # new
from hilbertbench.recorder.tape import HilbertTape                  # new

theta = ParameterVector("θ", 2)
qc = QuantumCircuit(2)
qc.ry(theta[0], 0)
qc.ry(theta[1], 1)
qc.cx(0, 1)
observable = SparsePauliOp("ZZ")

with HilbertTape("runs/my_vqe", tags={"algorithm": "vqe"}) as tape:  # new
    estimator = HilbertEstimatorProxy(tape)                           # new

    def cost(x):
        pv = x.reshape(1, -1)
        job = estimator.run([(qc, observable, pv)])  # unchanged
        return float(job.result()[0].data.evs.ravel()[0])

    result = minimize(cost, np.random.uniform(0, 2 * np.pi, 2), method="COBYLA")
```

The `with HilbertTape(...)` block is a context manager. When it exits, the
trace is sealed and ready to read. The `estimator` inside is identical to
`StatevectorEstimator` in every way — it just also records each call.

---

## Step 3 — What was written to disk

After the `with` block closes, a timestamped directory appears under
`runs/my_vqe/`:

```
runs/my_vqe/
└── 20260605_143022_a1b2c3d4/
    ├── events.jsonl        ← append-only event log (one JSON line per event)
    ├── manifest.json       ← trace metadata, tags, integrity seal
    ├── catalog.json        ← content-addressed artifact index
    └── store/              ← file-store for large artifacts (QASM, .npy, ...)
        └── ab12ef34.qasm
```

`events.jsonl` is plain text. You can inspect it directly:

```bash
cat runs/my_vqe/20260605_143022_a1b2c3d4/events.jsonl | python -m json.tool | head -40
```

---

## Step 4 — Load the trace

```python
from hilbertbench import HilbertTrace

trace = HilbertTrace("runs/my_vqe/20260605_143022_a1b2c3d4")

print(trace.status)      # "SEALED_SUCCESS"
print(trace.mode)        # "passive"
print(len(trace))        # number of spans (one per optimizer step)
print(trace.tags)        # {"algorithm": "vqe", ...}
```

Or use `tape.dir_path` directly if you still have the tape object in scope:

```python
trace = HilbertTrace(tape.dir_path)
```

Iterating gives `SpanView` objects — one per recorded circuit execution:

```python
for span in trace.completed():
    print(span.outcome)      # expectation value (float; list if multi-observable)
    print(span.parameters)   # bound parameter vector (list of floats)
    print(span.circuit)      # OpenQASM string
```

---

## Step 5 — Run an analyzer

```python
from hilbertbench.analysis import detect_barren_plateau

result = detect_barren_plateau(trace)
print(result)
```

```python
{
    'status': 'Trainable',
    'variance': 0.215,
    'std_dev': 0.464,
    'num_evaluations': 20,
    'threshold': 0.005,
    'variance_ci': [0.098, 0.371],
    'confidence_level': 0.95,
    'verdict_confidence': 'high'
}
```

The `status` field is the human-readable verdict. All other fields are
the evidence behind it — you can apply your own thresholds or use them
for further analysis.

---

## Step 6 — Convert to Parquet (optional)

For large runs or integration with pandas / Arrow tooling, convert the
JSONL trace to a single Parquet file:

```python
from hilbertbench.recorder.storage.writer import convert_trace_to_parquet

parquet_path = convert_trace_to_parquet(tape.dir_path)
# returns Path("runs/my_vqe/20260605_.../spans.parquet")
```

---

## Run a complete demo

The `demo/` directory contains four ready-to-run scripts:

```bash
python demo/01_qiskit_estimator.py   # VQE with barren-plateau analysis
python demo/02_qiskit_sampler.py     # QAOA bitstring sweep
python demo/03_qiskit_ibm.py         # IBM hardware (needs token)
python demo/04_pennylane.py          # two-moons QNN
```

Each script is self-contained and prints its trace path and diagnostic
results on completion.

---

## Next steps

- **[Concepts](concepts.md)** — understand what a span is, how passive recording works, and the evidence-vs-interpretation principle
- **[Analyzers](analyzers/barren-plateau.md)** — learn what each diagnostic measures and how to interpret every output field
- **[Guides](guides/qiskit-estimator.md)** — full narrative walkthroughs of each integration
