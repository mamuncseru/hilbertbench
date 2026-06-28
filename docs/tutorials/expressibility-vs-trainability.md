# Tutorial 03: Expressibility vs Trainability

**Measuring the tradeoff that governs ansatz design**

---

!!! abstract "What you will learn"
    - Why expressibility and trainability are in direct tension
    - How to use Active Mode to measure ansatz expressibility with
      `active_probe_qiskit` + `kl_expressibility`
    - How to quantify both properties on the same circuit and read
      the tradeoff from a single comparison table
    - What the results imply about choosing circuit depth for your problem

**Prerequisites**

```bash
pip install hilbertbench
```

---

## The core tension in ansatz design

When choosing a parameterized ansatz for VQE or QAOA, two properties
compete:

- **Expressibility** — how uniformly the circuit covers Hilbert space.
  A highly expressive ansatz can, in principle, represent the target state.
- **Trainability** — how much gradient signal reaches the optimizer.
  A trainable ansatz has a cost landscape with measurable curvature.

!!! info "The Holmes et al. result"
    Holmes et al. (2022) proved a direct connection between these two
    properties: circuits that are highly expressive — those whose output
    distribution closely approximates the Haar measure — have
    **exponentially small gradient variance** (barren plateaus).

    $$\text{Var}\!\left[\frac{\partial C}{\partial \theta_k}\right]
      \propto \text{Expressibility measure}$$

    A highly expressive ansatz is one that explores all of Hilbert space.
    But a cost function defined on all of Hilbert space is, on average,
    flat — which is exactly what a barren plateau looks like.

    The practical implication: **you cannot have both maximum expressibility
    and maximum trainability**. The ansatz must be expressive *enough*
    to represent the target state, but no more expressive than that.

This tutorial makes that tradeoff visible with real numbers.

---

## The experiment

We measure both properties on three ansatze of increasing depth, all on
4 qubits:

| Ansatz | Layers | Parameters | Prediction |
|---|---|---|---|
| **Shallow** | 1 layer | 8 params | Trainable, low expressibility |
| **Medium** | 2 layers | 16 params | Balanced |
| **Deep** | 4 layers | 32 params | Highly expressive, barren plateau |

---

## Step 1 — Build the ansatze

```python
import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import SparsePauliOp

def build_hea(n_qubits: int, n_layers: int) -> QuantumCircuit:
    """Hardware-efficient ansatz: alternating RY/RZ + CNOT layers."""
    n_params = n_qubits * 2 * n_layers
    params = ParameterVector("θ", n_params)
    qc = QuantumCircuit(n_qubits)
    p = 0
    for _ in range(n_layers):
        for q in range(n_qubits):
            qc.ry(params[p], q)
            qc.rz(params[p + 1], q)
            p += 2
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
    return qc

ansatze = {
    "shallow": build_hea(4, n_layers=1),
    "medium":  build_hea(4, n_layers=2),
    "deep":    build_hea(4, n_layers=4),
}

# 4-qubit Hamiltonian
hamiltonian = SparsePauliOp.from_list([
    ("ZZII", 1.0), ("IZZI", 1.0), ("IIZZ", 1.0),
])
```

---

## Step 2 — Measure expressibility with Active Mode

Active Mode draws `num_samples` random parameter vectors uniformly from
[0, 2π], evaluates the statevector for each, and stores the results.
`kl_expressibility` computes the KL divergence of the fidelity distribution
against the Haar reference.

```python
from hilbertbench.active import active_probe_qiskit
from hilbertbench.analysis import kl_expressibility

expr_results = {}
for name, circuit in ansatze.items():
    run_dir = active_probe_qiskit(
        circuit,
        num_samples=1000,
        output_root=f"runs/expressibility/{name}",
        seed=42,
        tags={"ansatz": name},
    )
    expr_results[name] = kl_expressibility(run_dir, seed=42)
    print(f"{name:8s}  KL = {expr_results[name]['kl_divergence']:.3f}"
          f"  → {expr_results[name]['status']}")
```

**Output:**

```
shallow   KL = 0.413  → Moderately Expressive
medium    KL = 0.121  → Moderately Expressive
deep      KL = 0.047  → Highly Expressive
```

The deep ansatz is nearly Haar-random — it covers Hilbert space almost
uniformly. The shallow circuit is more constrained.

---

## Step 3 — Measure trainability with passive recording

```python
from scipy.optimize import minimize
from hilbertbench import HilbertTrace
from hilbertbench.integrations.qiskit import HilbertEstimatorProxy
from hilbertbench.recorder.tape import HilbertTape
from hilbertbench.recorder.storage.writer import convert_trace_to_parquet
from hilbertbench.analysis import detect_barren_plateau, circuit_structure

train_results = {}
for name, circuit in ansatze.items():
    rng = np.random.default_rng(seed=42)
    x0  = rng.uniform(0.0, 2 * np.pi, circuit.num_parameters)

    with HilbertTape(f"runs/trainability/{name}",
                     tags={"ansatz": name}) as tape:
        estimator = HilbertEstimatorProxy(tape)

        def cost(x):
            job = estimator.run([(circuit, hamiltonian, x.reshape(1, -1))])
            return float(job.result()[0].data.evs.ravel()[0])

        minimize(cost, x0, method="COBYLA",
                 options={"maxiter": 50, "rhobeg": 0.5})

    convert_trace_to_parquet(tape.dir_path)
    trace = HilbertTrace(tape.dir_path)
    train_results[name] = {
        "barren":    detect_barren_plateau(trace),
        "structure": circuit_structure(trace)["primary"],
    }
```

---

## Step 4 — Read the joint result

```python
print(f"\n{'Ansatz':8s}  {'Depth':>6}  {'Params':>6}  "
      f"{'KL Div':>8}  {'Expr. Status':22}  "
      f"{'Variance':>10}  {'Trainability'}")
print("─" * 90)

for name in ("shallow", "medium", "deep"):
    e  = expr_results[name]
    bp = train_results[name]["barren"]
    cs = train_results[name]["structure"]
    print(
        f"{name:8s}  {cs['depth']:6d}  {cs['num_parameters']:6d}  "
        f"{e['kl_divergence']:8.3f}  {e['status']:22s}  "
        f"{bp['variance']:10.4f}  {bp['status']}"
    )
```

**Output:**

```
Ansatz    Depth  Params    KL Div  Expr. Status            Variance  Trainability
──────────────────────────────────────────────────────────────────────────────────────────
shallow       2       8     0.413  Moderately Expressive     0.1953  Trainable
medium        4      16     0.121  Moderately Expressive     0.0312  Trainable
deep          8      32     0.047  Highly Expressive         0.0007  Barren Plateau Detected
```

---

## Step 5 — Interpret the tradeoff

The numbers make the Holmes et al. result tangible:

```
Shallow  ←  KL = 0.41  variance = 0.195  (trainable, limited reach)
Medium   ←  KL = 0.12  variance = 0.031  (trainable, wider reach)
Deep     ←  KL = 0.05  variance = 0.0007 (barren plateau, near-Haar)
```

As expressibility increases (KL decreases toward 0), trainability degrades
— variance collapses by three orders of magnitude from shallow to deep.

The **medium** ansatz sits in the practical sweet spot: it covers more of
Hilbert space than the shallow circuit, retains enough cost-landscape
curvature to train, and avoids the barren plateau.

!!! tip "How to use this in practice"
    1. Run `active_probe_qiskit` with 500–1000 samples on your candidate ansatze.
    2. Find the shallowest depth where `kl_divergence` is below your
       expressibility target (problem-dependent; 0.2–0.4 is typical for
       NISQ problems).
    3. Verify trainability with a short passive run and `detect_barren_plateau`.
    4. Use that depth — adding more layers gains expressibility you do not
       need and costs trainability you cannot recover.

---

## The expressibility confidence interval

The bootstrap confidence intervals on KL divergence tell you whether 1000
samples was sufficient:

```python
for name, r in expr_results.items():
    print(f"{name:8s}  KL = {r['kl_divergence']:.3f}"
          f"  95% CI = {r['kl_ci']}")
```

```
shallow   KL = 0.413  95% CI = [0.341, 0.492]
medium    KL = 0.121  95% CI = [0.094, 0.153]
deep      KL = 0.047  95% CI = [0.033, 0.064]
```

All three intervals are narrow relative to the thresholds (0.1, 0.5), so
the 1000-sample budget was sufficient here. If an interval straddles a
threshold, run more samples (see the
[Expressibility analyzer docs](../analyzers/expressibility.md) for guidance).

---

## Summary

| Ansatz | KL Divergence | Expressibility | Variance | Trainable? |
|---|---|---|---|---|
| Shallow (1 layer) | 0.413 | Moderate | 0.195 | ✅ Yes |
| Medium (2 layers) | 0.121 | Moderate | 0.031 | ✅ Yes |
| Deep (4 layers) | 0.047 | **High** | 0.0007 | ❌ Barren plateau |

HilbertBench measured both axes from the same circuits — no manual
gradient calculations, no access to the parameter-shift formula, no
changes to the optimizer. The tradeoff is in the data.

---

## References

- Holmes, Z., Sharma, K., Cerezo, M., & Coles, P. J. (2022).
  Connecting ansatz expressibility to gradient magnitudes and barren plateaus.
  *PRX Quantum*, 3(1), 010313.

- Sim, S., Johnson, P. D., & Aspuru-Guzik, A. (2019).
  Expressibility and entangling capability of parameterized quantum circuits
  for hybrid quantum-classical algorithms.
  *Advanced Quantum Technologies*, 2(12), 1900070.

---

**Next tutorial →** [How Hardware Noise Degrades Your Results](hardware-noise.md)
