# Tutorial 01: Why Isn't My VQE Converging?

**Diagnosing barren plateaus with HilbertBench**

---

!!! abstract "What you will learn"
    - What a barren plateau is and why deep ansatze cause them
    - How to record a VQE run with HilbertBench using one code change
    - How to use `detect_barren_plateau` to distinguish a stuck optimizer
      from a flat landscape
    - How to compare a deep (barren) ansatz against a shallow (trainable) one

**Prerequisites**

```bash
pip install hilbertbench[qiskit,storage] scipy
```

---

## The problem

You have a 4-qubit hardware-efficient ansatz. You run COBYLA for 40 steps.
The energy drops briefly, then flatlines. The optimizer reports convergence,
but the final energy is far from the known ground state.

```
Step   1/40  energy = -0.412
Step   5/40  energy = -0.438
Step  10/40  energy = -0.441
Step  20/40  energy = -0.441
Step  40/40  energy = -0.442   ← stuck, not at ground state (-1.0)
```

Three possible explanations:

1. **Barren plateau** — the cost landscape is nearly flat everywhere; the
   optimizer receives near-zero signal regardless of direction
2. **Local minimum** — the optimizer found a basin but it is not the global one
3. **Insufficient iterations** — the optimizer would converge given more steps

Without instrumentation, these three look identical from the loss curve alone.
HilbertBench distinguishes them.

---

## Background: barren plateaus

!!! info "The science"
    In 2018, McClean et al. showed that for deep parameterized quantum circuits
    with random initialization, the variance of the cost-function gradient
    with respect to any single parameter vanishes **exponentially** with the
    number of qubits *n*:

    $$\text{Var}\left[\frac{\partial C}{\partial \theta_k}\right] \leq \mathcal{O}\!\left(\frac{1}{2^n}\right)$$

    For a 12-qubit circuit this is O(1/4096). Gradient-based optimizers receive
    near-zero updates — they are navigating a landscape that looks flat in
    every measurable direction. This is the **barren plateau**.

    Gradient-free methods (COBYLA, Nelder-Mead) are not immune: the *outcome
    values themselves* have exponentially small variance, so the optimizer
    cannot tell which direction improves the objective.

The observable symptom in a recorded trace: the **variance of the cost
trajectory** is close to zero throughout the entire run.

---

## The experiment

We compare two circuits on the same Hamiltonian:

| Ansatz | Layers | Parameters | Expected behaviour |
|---|---|---|---|
| **Deep HEA** | 4 layers | 32 parameters | Barren plateau (deep random circuit) |
| **Shallow HEA** | 1 layer | 8 parameters | Trainable (shallow, gradient survives) |

The Hamiltonian is a 4-qubit model: **ZZ + ZI + IZ**, a simplified
anti-ferromagnetic interaction with ground-state energy −2.0.

---

## Step 1 — Build the two ansatze

```python
import numpy as np
from scipy.optimize import minimize
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator

def build_hea(n_qubits: int, n_layers: int) -> tuple:
    """Hardware-efficient ansatz: alternating RY/RZ layers + CNOT entanglement."""
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
    return qc, params

# 4-qubit Hamiltonian: ZZ + ZI + IZ
hamiltonian = SparsePauliOp.from_list([
    ("ZZII", 1.0),
    ("IZZI", 1.0),
    ("IIZZ", 1.0),
])

deep_circuit,    _ = build_hea(n_qubits=4, n_layers=4)   # 32 params
shallow_circuit, _ = build_hea(n_qubits=4, n_layers=1)   # 8 params
```

---

## Step 2 — Add HilbertBench (the only change)

```python hl_lines="2 3 7 8"
from hilbertbench import HilbertTrace
from hilbertbench.integrations.qiskit import HilbertEstimatorProxy
from hilbertbench.recorder.tape import HilbertTape
from hilbertbench.recorder.storage.writer import convert_trace_to_parquet
from hilbertbench.analysis import detect_barren_plateau, circuit_structure

def run_vqe(circuit, label: str, n_steps: int = 40) -> dict:
    """Run VQE and return the HilbertBench diagnostic report."""
    rng = np.random.default_rng(seed=42)
    n_params = circuit.num_parameters
    x0 = rng.uniform(0.0, 2 * np.pi, n_params)

    with HilbertTape(f"runs/bp_tutorial/{label}", tags={"ansatz": label}) as tape:
        estimator = HilbertEstimatorProxy(tape)   # ← the only change

        def cost(x):
            pv = x.reshape(1, -1)
            job = estimator.run([(circuit, hamiltonian, pv)])
            return float(job.result()[0].data.evs.ravel()[0])

        minimize(cost, x0, method="COBYLA",
                 options={"maxiter": n_steps, "rhobeg": 0.5})

    convert_trace_to_parquet(tape.dir_path)
    trace = HilbertTrace(tape.dir_path)
    return {
        "trace":     trace,
        "barren":    detect_barren_plateau(trace),
        "structure": circuit_structure(trace),
    }

deep_report    = run_vqe(deep_circuit,    label="deep_4layer")
shallow_report = run_vqe(shallow_circuit, label="shallow_1layer")
```

!!! note "What changed"
    Three lines were added. Everything else — the circuit, the Hamiltonian,
    the optimizer, the cost function — is identical to the original code.

---

## Step 3 — Read the diagnostic output

```python
import json

for label, report in [("Deep (4 layers)", deep_report),
                       ("Shallow (1 layer)", shallow_report)]:
    bp  = report["barren"]
    cs  = report["structure"]["primary"]
    print(f"\n── {label} ──")
    print(f"  Circuit depth    : {cs['depth']}")
    print(f"  Parameters       : {cs['num_parameters']}")
    print(f"  Entangling gates : {cs['entangling_fraction']:.0%}")
    print(f"  Trainability     : {bp['status']}")
    print(f"  Variance         : {bp['variance']:.4f}")
    print(f"  95% CI           : {bp['variance_ci']}")
    print(f"  Confidence       : {bp['verdict_confidence']}")
```

**Output:**

```
── Deep (4 layers) ──
  Circuit depth    : 8
  Parameters       : 32
  Entangling gates : 33%
  Trainability     : Barren Plateau Detected
  Variance         : 0.0007
  95% CI           : [0.0003, 0.0014]
  Confidence       : high

── Shallow (1 layer) ──
  Circuit depth    : 2
  Parameters       : 8
  Entangling gates : 33%
  Trainability     : Trainable
  Variance         : 0.195
  95% CI           : [0.088, 0.319]
  Confidence       : high
```

---

## Step 4 — Understand the result

### Deep ansatz: variance = 0.0007

The outcome variance is three orders of magnitude below the trainability
threshold (0.005). The 95% bootstrap confidence interval `[0.0003, 0.0014]`
lies entirely below the threshold — the verdict confidence is `'high'`.

This is consistent with the barren plateau prediction. The deep 4-layer HEA
on 4 qubits is approaching the 2-design regime: its output distribution
begins to mimic a Haar-random unitary, which by Levy's lemma concentrates
all measurement outcomes near the same constant value.

### Shallow ansatz: variance = 0.195

The outcome variance is 40× higher. The cost landscape has measurable
curvature and the optimizer can follow it. The shallow circuit does not
have enough expressibility to concentrate outcomes.

!!! warning "Important nuance"
    A barren plateau diagnosis from HilbertBench is **cost-landscape variance**,
    not gradient variance. These are correlated but not identical — a circuit
    can have small gradient variance for other reasons (e.g., a good local
    minimum where gradients are naturally small). Always inspect `outcome_min`
    and `outcome_final` from `optimization_convergence` alongside this result.

---

## Step 5 — Confirm with optimization convergence

```python
from hilbertbench.analysis import optimization_convergence

for label, report in [("Deep", deep_report), ("Shallow", shallow_report)]:
    conv = optimization_convergence(report["trace"])
    print(f"{label:7s}  status={conv['status']:16s}  "
          f"trend={conv['outcome_trend']:+.3f}  "
          f"movement_ratio={conv['movement_ratio']:.3f}")
```

```
Deep     status=Stalled            trend=-0.031  movement_ratio=0.021
Shallow  status=Converging         trend=-0.487  movement_ratio=0.143
```

The deep ansatz shows near-zero parameter movement throughout (stalled).
The shallow ansatz shows meaningful cost improvement (trend = -0.487) with
a decelerating movement ratio — it is converging toward a minimum.

---

## What to do about a barren plateau

If HilbertBench confirms a barren plateau, the practical options are:

| Strategy | Why it helps |
|---|---|
| **Reduce circuit depth** | Fewer layers → less expressibility → more gradient signal |
| **Use a local cost function** | Measuring fewer qubits avoids the global-observable plateau (Cerezo et al., 2021) |
| **Layer-by-layer training** | Train one layer at a time, freezing previous parameters |
| **Problem-inspired ansatz** | Hardware-efficient ansatze are general; problem-specific ones can be shallower and still reach the target state |
| **Correlation analysis** | Use Active Mode + `kl_expressibility` to find the shallowest ansatz with sufficient expressibility for your problem |

---

## Summary

| | Deep HEA (4 layers) | Shallow HEA (1 layer) |
|---|---|---|
| Parameters | 32 | 8 |
| Circuit depth | 8 | 2 |
| Outcome variance | **0.0007** | 0.195 |
| Trainability | **Barren Plateau** | Trainable |
| Convergence | Stalled | Converging |
| Cost trend | -0.031 | -0.487 |

HilbertBench turned an ambiguous "optimizer stuck" symptom into a precise,
evidence-backed diagnosis — in three added lines of code.

---

## References

- McClean, J. R., Boixo, S., Smelyanskiy, V. N., Babbush, R., & Neven, H.
  (2018). Barren plateaus in quantum neural network training landscapes.
  *Nature Communications*, 9(1), 4812.

- Cerezo, M., Sone, A., Volkoff, T., Cincio, L., & Coles, P. J. (2021).
  Cost function dependent barren plateaus in shallow parametrized quantum
  circuits. *Nature Communications*, 12(1), 1791.

- Holmes, Z., Sharma, K., Cerezo, M., & Coles, P. J. (2022).
  Connecting ansatz expressibility to gradient magnitudes and barren plateaus.
  *PRX Quantum*, 3(1), 010313.

---

**Next tutorial →** [Am I Using Enough Shots?](shot-noise.md)
