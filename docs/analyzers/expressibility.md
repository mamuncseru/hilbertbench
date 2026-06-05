# Expressibility (KL Divergence)

```python
from hilbertbench.analysis import kl_expressibility

result = kl_expressibility(active_trace)
```

!!! warning "Requires an Active Mode trace"
    This analyzer **cannot** be applied to a passive training trace.
    See [Active Mode](../guides/active-mode.md) for how to generate the
    required trace.

---

## What it measures

**Expressibility** quantifies how uniformly an ansatz's output states cover
the Hilbert space. A highly expressive ansatz can reach nearly any state;
a low-expressibility ansatz is confined to a small submanifold.

The standard estimator (Sim et al., 2019) compares the **fidelity
distribution** of randomly-parameterized output states against the
**Haar-random fidelity distribution** — the reference for a uniform covering
of Hilbert space.

KL divergence measures how far the ansatz's fidelity distribution deviates
from Haar-random:

- **KL ≈ 0** — ansatz behaves like a Haar-random unitary (highly expressive)
- **KL large** — ansatz is confined to a subspace (low expressibility)

---

## Function signature

```python
kl_expressibility(
    trace: HilbertTrace | str | Path,
    num_bins: int = 75,
    max_pairs: int = 5000,
    seed: int | None = None,
    n_boot: int = 500,
    ci: float = 0.95,
) -> dict
```

| Parameter | Default | Meaning |
|---|---|---|
| `trace` | — | An **Active Mode** trace (mode == `'active'`) |
| `num_bins` | `75` | Histogram bins for the fidelity distribution |
| `max_pairs` | `5000` | Maximum state pairs sampled for fidelity computation |
| `seed` | `None` | RNG seed for pair sampling and bootstrap |
| `n_boot` | `500` | Bootstrap resamples for the KL confidence interval |
| `ci` | `0.95` | Confidence level |

---

## Output fields

```python
{
    'status':           'Moderately Expressive',
    'kl_divergence':    0.28,
    'num_states':       1000,
    'num_pairs':        5000,
    'num_qubits':       2,
    'kl_ci':            [0.21, 0.36],
    'confidence_level': 0.95,
}
```

| Field | Meaning |
|---|---|
| `status` | Human-readable expressibility verdict |
| `kl_divergence` | KL divergence of ansatz fidelity distribution vs. Haar |
| `num_states` | Number of statevectors used (= number of Active Mode samples) |
| `num_pairs` | Number of fidelity pairs sampled |
| `num_qubits` | Inferred from the statevector dimension |
| `kl_ci` | `[low, high]` bootstrap CI on KL divergence |
| `confidence_level` | CI level |

---

## Verdicts

| `status` | KL divergence | Meaning |
|---|---|---|
| `Highly Expressive` | `< 0.1` | Distribution closely matches Haar — near-universal coverage |
| `Moderately Expressive` | `0.1 – 0.5` | Partial coverage — reasonable for shallow ansatze |
| `Low Expressibility` | `≥ 0.5` | Ansatz is confined to a small submanifold |
| `Requires Active Mode trace` | — | Guard: called on a passive trace |
| `Insufficient Data` | — | Fewer than 2 statevectors in the trace |

---

## How to generate an Active Mode trace

```python
from hilbertbench.active import active_probe_qiskit

run_dir = active_probe_qiskit(
    ansatz,                     # QuantumCircuit with ParameterVector
    num_samples=1000,           # how many random parameter draws
    output_root="runs/expr",    # where to write the trace
)

result = kl_expressibility(run_dir)
```

A minimum of ~200 samples gives a stable KL estimate for 2-qubit ansatze.
For N-qubit ansatze, the number of required samples grows with 2^N; aim for
at least 10× the Hilbert space dimension.

---

## Limitations

- The KL estimator uses a histogram with fixed bin edges. Very low KL values
  (< 0.01) may be unstable with fewer than 500 samples; widen `kl_ci` will
  be large in that case.
- This measures **state-space** expressibility. It does not measure whether
  the ansatz can represent the specific ground state or target distribution
  relevant to your problem.
- Expressibility and trainability are in tension: highly expressive ansatze
  often suffer barren plateaus. A high KL score is not automatically bad;
  it depends on the problem.

---

## Example

```python
from hilbertbench.active import active_probe_qiskit
from hilbertbench.analysis import kl_expressibility

# run the probe (creates a new trace with 1000 statevectors)
run_dir = active_probe_qiskit(ansatz, num_samples=1000, output_root="runs")

# analyze
result = kl_expressibility(run_dir, seed=42)

print(f"Expressibility: {result['status']}")
print(f"KL divergence:  {result['kl_divergence']:.3f}  "
      f"95% CI: {result['kl_ci']}")
print(f"Samples used:   {result['num_states']}")
```

---

## Reference

Sim, S., Johnson, P. D., & Aspuru-Guzik, A. (2019). Expressibility and
entangling capability of parameterized quantum circuits for hybrid
quantum-classical algorithms. *Advanced Quantum Technologies*, 2(12), 1900070.
