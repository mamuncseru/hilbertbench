# Guide: Qiskit Estimator (VQE)

This guide walks through `demo/01_qiskit_estimator.py` — a 20-step
gradient-free VQE on a 2-qubit RY+CNOT ansatz minimizing the ZZ
expectation value.

Run it directly:

```bash
python demo/01_qiskit_estimator.py
```

---

## What the demo does

1. Builds a 2-qubit hardware-efficient ansatz: `RY(θ₀) ⊗ RY(θ₁) → CNOT`
2. Defines a ZZ observable (ground state energy = -1.0)
3. Opens a `HilbertTape` and wraps `StatevectorEstimator` in
   `HilbertEstimatorProxy`
4. Runs scipy COBYLA for 20 iterations — each `cost()` call is one
   recorded span
5. After the tape closes: converts to Parquet, loads the trace, runs
   `detect_barren_plateau`

---

## The proxy swap

```python
from qiskit.primitives import StatevectorEstimator       # before
from hilbertbench.integrations.qiskit import HilbertEstimatorProxy
from hilbertbench.recorder.tape import HilbertTape

# Before:
# estimator = StatevectorEstimator()

# After (two extra lines):
with HilbertTape("runs/estimator_vqe", tags={"algorithm": "vqe"}) as tape:
    estimator = HilbertEstimatorProxy(tape)
```

`HilbertEstimatorProxy` wraps `StatevectorEstimator` by default. To use a
different backend:

```python
from qiskit.primitives import StatevectorEstimator
estimator = HilbertEstimatorProxy(tape, real_estimator=StatevectorEstimator())
```

---

## The cost function

The cost function is unchanged from a standard VQE:

```python
def cost(x: np.ndarray) -> float:
    pv = x.reshape(1, -1)
    job = estimator.run([(circuit, observable, pv)])
    return float(np.ravel(job.result()[0].data.evs)[0])
```

The proxy intercepts `estimator.run(...)` and records:
- The circuit QASM (once, content-addressed)
- The parameter binding `pv`
- The expectation value from `data.evs`
- Start/end timestamps

---

## Analyzing the trace

```python
from hilbertbench import HilbertTrace
from hilbertbench.analysis import detect_barren_plateau

trace = HilbertTrace(tape.dir_path)
bp = detect_barren_plateau(trace)

print(f"Trainability: {bp['status']}")
print(f"Variance:     {bp['variance']:.6f}")
```

Expected output for a trainable 2-qubit VQE:

```
Trainability: Trainable
Variance:     0.215xxx
```

---

## Extending this example

- Add `optimization_convergence` to see if COBYLA converged
- Add `circuit_structure` to verify the ansatz depth and gate count
- Increase `N_ITER` and observe variance stabilize
- Change the observable to `XX` or `X` and compare trainability
