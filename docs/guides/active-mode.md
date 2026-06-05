# Guide: Active Mode (Expressibility Probing)

Passive recording observes whatever circuits your optimizer chooses to run.
That is the right data for trainability and convergence, but it cannot
measure expressibility: to compare an ansatz against the Haar measure you
need output states under parameters drawn **uniformly at random** over the
full parameter space, which a training trajectory never provides.

**Active Mode** does exactly that: given a parameterized circuit and a way
to evaluate it, it draws `num_samples` random parameter vectors, evaluates
each, and records the resulting statevectors into a separate trace. Feed
that trace to `kl_expressibility`.

Active Mode is an **explicit user action** — it runs new circuits. It is
never triggered automatically.

---

## Qiskit

```python
from qiskit.circuit import QuantumCircuit, ParameterVector
from hilbertbench.active import active_probe_qiskit

theta = ParameterVector("θ", 4)
ansatz = QuantumCircuit(2)
ansatz.ry(theta[0], 0)
ansatz.ry(theta[1], 1)
ansatz.cx(0, 1)
ansatz.ry(theta[2], 0)
ansatz.ry(theta[3], 1)

run_dir = active_probe_qiskit(
    ansatz,
    num_samples=1000,
    output_root="runs/expressibility",
    seed=42,
    tags={"ansatz": "ry_cnot_ry"},
)

from hilbertbench.analysis import kl_expressibility
result = kl_expressibility(run_dir)
print(result)
```

---

## PennyLane

```python
import pennylane as qml
import numpy as np
from hilbertbench.active import active_probe_pennylane

def circuit_fn(params):
    qml.StronglyEntanglingLayers(params.reshape(2, 2, 3), wires=[0, 1])
    dev = qml.device("default.qubit", wires=2)
    return qml.state()

run_dir = active_probe_pennylane(
    circuit_fn,
    num_qubits=2,
    num_params=12,          # 2 layers × 2 wires × 3 params
    num_samples=1000,
    output_root="runs/expressibility",
)
```

---

## Function signatures

### `active_probe_qiskit`

```python
active_probe_qiskit(
    circuit: QuantumCircuit,
    num_samples: int,
    output_root: Path | str,
    *,
    param_low: float = 0.0,
    param_high: float = 2 * π,
    seed: int | None = None,
    tags: dict | None = None,
    backend_id: str = "statevector_simulator",
) -> Path
```

### `active_probe_pennylane`

```python
active_probe_pennylane(
    circuit_fn: Callable[[np.ndarray], np.ndarray],
    num_qubits: int,
    num_params: int,
    num_samples: int,
    output_root: Path | str,
    *,
    param_low: float = 0.0,
    param_high: float = 2 * π,
    seed: int | None = None,
    tags: dict | None = None,
) -> Path
```

Both functions return the path to the created run directory, which you can
pass directly to `kl_expressibility`.

---

## How many samples?

| Qubits | Minimum samples | Recommended |
|---|---|---|
| 2 | 200 | 1000 |
| 4 | 500 | 2000 |
| 6 | 1000 | 5000 |
| 8+ | 2000 | 10000+ |

The KL estimator's confidence interval narrows with more samples. Check
`kl_ci` in the output — if the interval is wide relative to the thresholds
(0.1, 0.5), run more samples.

---

## Interpreting the result

```python
{
    'status':        'Moderately Expressive',
    'kl_divergence': 0.28,
    'kl_ci':         [0.21, 0.36],
    ...
}
```

- `kl_divergence < 0.1` — matches the Haar measure closely (highly expressive)
- `0.1 – 0.5` — moderate coverage (typical for shallow NISQ ansatze)
- `≥ 0.5` — confined to a small submanifold

See [Expressibility (KL)](../analyzers/expressibility.md) for the full
output reference.
