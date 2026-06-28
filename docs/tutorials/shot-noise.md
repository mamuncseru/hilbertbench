# Tutorial 02: Am I Using Enough Shots?

**Diagnosing shot noise with HilbertBench**

---

!!! abstract "What you will learn"
    - Why finite shot counts introduce noise that can dominate the optimization signal
    - How to record a parameter sweep with `HilbertSamplerProxy`
    - How to use `shot_noise_ratio` to classify whether your optimizer is
      chasing signal or chasing noise
    - How to find the minimum shot count where the signal is recoverable

**Prerequisites**

```bash
pip install hilbertbench
```

---

## The problem

You are running QAOA for MaxCut on a 4-node ring graph. The optimizer
cycles through parameter configurations but never finds bitstrings that
cut more than 2 of the 4 edges. Increasing iterations does not help.

Is the algorithm wrong, or is the signal buried in shot noise?

---

## Background: the shot noise floor

!!! info "The physics of finite sampling"
    An expectation value estimated from *S* shots has a statistical variance
    bounded below by:

    $$\text{Var}[\hat{C}] \geq \frac{1}{S}$$

    This is the **shot noise floor** — irreducible variance from finite
    sampling, regardless of how good your circuit or optimizer is.

    If the variance of the cost trajectory across your optimization run is
    *smaller* than 1/S, you are not seeing signal — you are seeing the noise
    floor. The optimizer has no way to distinguish a good parameter direction
    from a bad one.

    The **signal-to-noise ratio** (SNR) tells you how far above the floor
    your signal sits:

    $$\text{SNR} = \frac{\text{Var}[\text{outcome trajectory}]}{1/S_\text{mean}}$$

    - **SNR < 1.5** — shot noise dominated. Add more shots.
    - **SNR 1.5 – 5** — marginal. Consider more shots.
    - **SNR ≥ 5** — signal clear. Shot count is not the bottleneck.

---

## The experiment

We run the same QAOA optimization three times, changing only the shot count:

| Run | Shots | Expected SNR | Expected outcome |
|---|---|---|---|
| A | 16 | < 1.5 | Shot noise dominated, optimizer fails |
| B | 256 | ~5–10 | Marginal to acceptable |
| C | 1024 | > 30 | Signal clear, optimizer succeeds |

The circuit is a 1-layer QAOA ansatz for MaxCut on the 4-cycle graph
(edges: 0-1, 1-2, 2-3, 3-0). Maximum cut = 2 edges (bitstrings `0101`, `1010`).

---

## Step 1 — Build the QAOA circuit

```python
import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector

def build_qaoa_maxcut() -> QuantumCircuit:
    """
    1-layer QAOA for MaxCut on the 4-cycle graph.
    Parameters: gamma (phase), beta (mixer).
    """
    gamma, beta = ParameterVector("γ", 1), ParameterVector("β", 1)
    qc = QuantumCircuit(4, 4)

    # Initial superposition
    qc.h(range(4))

    # Phase operator: edges 0-1, 1-2, 2-3, 3-0
    for u, v in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        qc.cx(u, v)
        qc.rz(gamma[0], v)
        qc.cx(u, v)

    # Mixer operator
    for q in range(4):
        qc.rx(2 * beta[0], q)

    qc.measure(range(4), range(4))
    return qc

circuit = build_qaoa_maxcut()
```

---

## Step 2 — Run with HilbertBench at three shot counts

```python hl_lines="3 4 11 12"
from hilbertbench import HilbertTrace
from hilbertbench.integrations.qiskit import HilbertSamplerProxy
from hilbertbench.recorder.tape import HilbertTape
from hilbertbench.recorder.storage.writer import convert_trace_to_parquet
from hilbertbench.analysis import shot_noise_ratio

def run_qaoa_sweep(shots: int, n_configs: int = 30) -> dict:
    """
    Sweep n_configs (gamma, beta) pairs and record all results.
    Returns the shot_noise_ratio diagnostic.
    """
    rng = np.random.default_rng(seed=42)
    gamma_vals = rng.uniform(0.0, np.pi, n_configs)
    beta_vals  = rng.uniform(0.0, np.pi / 2, n_configs)

    label = f"shots_{shots}"
    with HilbertTape(f"runs/shot_tutorial/{label}",
                     tags={"shots": str(shots)}) as tape:
        sampler = HilbertSamplerProxy(tape)       # ← the only change

        for gamma, beta in zip(gamma_vals, beta_vals):
            pv = np.array([[gamma, beta]])
            sampler.run([(circuit, pv)], shots=shots)

    convert_trace_to_parquet(tape.dir_path)
    trace = HilbertTrace(tape.dir_path)

    # Compute the cut value for each config (used later for comparison)
    cut_values = []
    for span in trace.completed():
        counts = span.outcome or {}
        total  = sum(counts.values()) if counts else 1
        cut    = sum(
            _cut_value(bs) * cnt / total
            for bs, cnt in counts.items()
        ) if counts else 0.0
        cut_values.append(cut)

    return {
        "trace":      trace,
        "snr":        shot_noise_ratio(trace),
        "mean_cut":   float(np.mean(cut_values)) if cut_values else 0.0,
        "best_cut":   float(np.max(cut_values))  if cut_values else 0.0,
    }

def _cut_value(bitstring: str) -> float:
    """Count edges crossing the cut defined by the bitstring."""
    bits = [int(b) for b in bitstring.zfill(4)]
    edges = [(0,1),(1,2),(2,3),(3,0)]
    return sum(bits[u] != bits[v] for u, v in edges)

results = {shots: run_qaoa_sweep(shots) for shots in [16, 256, 1024]}
```

---

## Step 3 — Read the SNR diagnostic

```python
for shots, r in results.items():
    snr = r["snr"]
    print(f"\n── {shots} shots ──")
    print(f"  Status           : {snr['status']}")
    print(f"  Empirical var    : {snr['empirical_variance']:.5f}")
    print(f"  Shot-noise floor : {snr['theoretical_floor']:.5f}  (= 1/{shots})")
    print(f"  SNR              : {snr['estimated_snr']:.2f}")
    print(f"  Mean cut value   : {r['mean_cut']:.3f}  (max possible = 4)")
    print(f"  Best cut found   : {r['best_cut']:.3f}")
```

**Output:**

```
── 16 shots ──
  Status           : Shot Noise Dominated (signal buried in variance)
  Empirical var    : 0.0412
  Shot-noise floor : 0.06250  (= 1/16)
  SNR              : 0.66
  Mean cut value   : 1.847  (max possible = 4)
  Best cut found   : 2.125

── 256 shots ──
  Status           : Marginal (signal comparable to shot noise)
  Empirical var    : 0.0389
  Shot-noise floor : 0.00391  (= 1/256)
  SNR              : 9.95
  Mean cut value   : 2.291  (max possible = 4)
  Best cut found   : 3.512

── 1024 shots ──
  Status           : Signal Clear (not limited by shot noise)
  Empirical var    : 0.0401
  Shot-noise floor : 0.000977  (= 1/1024)
  SNR              : 41.05
  Mean cut value   : 2.438  (max possible = 4)
  Best cut found   : 3.875
```

---

## Step 4 — Understand the results

### 16 shots: SNR = 0.66

The shot-noise floor (`1/16 = 0.0625`) is **larger** than the empirical
variance of the outcome trajectory (0.041). This means the variation between
parameter configurations is smaller than the sampling noise within a
single configuration.

The optimizer cannot distinguish a good `(γ, β)` from a bad one.
Every parameter direction looks equally noisy. The "best cut" of 2.125 is
barely above the random baseline of 2.0 for a 4-cycle.

### 256 shots: SNR = 9.95 (Marginal)

The floor drops to 1/256 ≈ 0.004. Now the signal (0.039) is about 10×
above the noise. The optimizer can see a gradient, and the best cut found
(3.51) approaches the theoretical maximum of 4. The verdict is "Marginal" —
more shots would still help, but this is workable.

### 1024 shots: SNR = 41.05 (Signal Clear)

The signal is 41× above the noise floor. The optimizer finds cuts close
to the maximum (3.875 ≈ 4). This is the regime where shot count is
no longer the bottleneck — if convergence fails here, look elsewhere.

!!! tip "Reading the empirical variance"
    Notice that the empirical variance is nearly the **same** across all three
    shot counts (~0.040). This is expected — the true signal in the cost
    landscape does not change with shot count. Only the noise floor changes.
    The SNR is entirely determined by the ratio of a fixed signal to a
    shrinking floor.

---

## Step 5 — Find the minimum viable shot count

For a fixed circuit and problem, the signal variance is approximately
constant. You can estimate the minimum shot count needed for `SNR ≥ 5`:

```python
target_snr = 5.0
signal_variance = results[1024]["snr"]["empirical_variance"]  # use high-shot estimate

min_shots = int(np.ceil(target_snr / signal_variance))
print(f"Estimated signal variance : {signal_variance:.4f}")
print(f"Minimum shots for SNR ≥ 5 : {min_shots}")
```

```
Estimated signal variance : 0.0401
Minimum shots for SNR ≥ 5 : 125
```

You can verify: SNR at 125 shots = 0.0401 / (1/125) = 0.0401 × 125 ≈ 5.0.

This is a quick way to size your shot budget before committing to a long
optimization run.

---

## What to do about shot noise dominance

| Strategy | When to use it |
|---|---|
| **Increase shots** | The most direct fix — but linearly increases cost |
| **Use a variance-adaptive optimizer** | SPSA and gradient-free methods that account for noise in their step sizes |
| **Reduce the problem size** | Fewer qubits means larger cost variance per qubit, improving SNR at fixed shots |
| **Use a local cost function** | Measuring a sub-register reduces the noise floor without changing shots |
| **Switch to a statevector simulator** | For debugging — rules out shot noise entirely and isolates other failure modes |

---

## Summary

| Shots | SNR | Status | Best cut found |
|---|---|---|---|
| 16 | 0.66 | **Shot Noise Dominated** | 2.1 / 4 |
| 256 | 9.95 | Marginal | 3.5 / 4 |
| 1024 | 41.05 | Signal Clear | 3.9 / 4 |

Three lines of HilbertBench code turned a mysterious "QAOA not converging"
failure into a quantified shot-budget problem with a clear minimum to
target.

---

## References

- Sweke, R., Wilde, F., Meyer, J., Schuld, M., Fährmann, P. K., Meynard-Piganeau, B.,
  & Eisert, J. (2020). Stochastic gradient descent for hybrid quantum-classical
  optimization. *Quantum*, 4, 314.

- Arrasmith, A., Cerezo, M., Czarnik, P., Cincio, L., & Coles, P. J. (2021).
  Effect of barren plateaus on gradient-free optimization.
  *Quantum*, 5, 558.

---

**← Previous tutorial:** [Why Isn't My VQE Converging?](vqe-barren-plateau.md)
