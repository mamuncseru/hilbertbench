# Barren Plateau Detection

```python
from hilbertbench.analysis import detect_barren_plateau

result = detect_barren_plateau(trace)
```

---

## What it measures

A **barren plateau** is a region of the cost landscape where the gradient
variance decays exponentially with qubit count. In practice: the optimizer
makes progress at first, then the expectation values cluster tightly around
a constant and no direction looks downhill.

The signature is a collapse in the **variance of the outcome trajectory**.
If the circuit's expectation values barely move across the full training run,
the landscape is flat in every direction the optimizer has explored.

`detect_barren_plateau` computes the variance of all numeric outcomes
recorded in the trace and compares it to a threshold.

---

## Function signature

```python
detect_barren_plateau(
    trace: HilbertTrace | str | Path,
    threshold: float = 0.005,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int | None = None,
) -> dict
```

| Parameter | Default | Meaning |
|---|---|---|
| `trace` | — | `HilbertTrace` or path to a run directory |
| `threshold` | `0.005` | Variance below this is classified as a barren plateau |
| `n_boot` | `1000` | Bootstrap resamples for the confidence interval |
| `ci` | `0.95` | Confidence level for the variance interval |
| `seed` | `None` | RNG seed for reproducibility |

---

## Output fields

```python
{
    'status':             'Trainable',          # or 'Barren Plateau Detected'
    'variance':           0.215,                # variance of outcome trajectory
    'std_dev':            0.464,                # standard deviation
    'num_evaluations':    20,                   # number of numeric outcomes used
    'threshold':          0.005,                # threshold applied
    'variance_ci':        [0.098, 0.371],       # 95% bootstrap CI on variance
    'confidence_level':   0.95,
    'verdict_confidence': 'high',               # 'high' or 'low'
}
```

| Field | Meaning |
|---|---|
| `status` | Human-readable verdict |
| `variance` | Variance of all scalar outcomes in the trace |
| `std_dev` | Square root of variance |
| `num_evaluations` | How many spans had a numeric outcome |
| `threshold` | The `threshold` argument used |
| `variance_ci` | `[low, high]` bootstrap confidence interval on the variance |
| `confidence_level` | CI level (default 0.95) |
| `verdict_confidence` | `'high'` if the CI is wholly on one side of the threshold, `'low'` if it straddles it |

---

## Verdicts

| `status` | Meaning |
|---|---|
| `Trainable` | Variance exceeds the threshold — the cost landscape has measurable curvature |
| `Barren Plateau Detected` | Variance is at or below the threshold — the landscape is flat |
| `Insufficient Data` | No numeric outcomes in the trace |

---

## Verdict confidence

When the bootstrap confidence interval **straddles** the threshold — part of
the interval is above `0.005` and part below — the verdict could go either
way. In this case `verdict_confidence` is `'low'` and you should run a longer
experiment before drawing conclusions.

When the entire CI is on one side, `verdict_confidence` is `'high'`.

---

## Choosing the threshold

The default `0.005` is a practical heuristic for small ansatze (2–8 qubits).
For deeper circuits or higher-qubit systems where variance decays
exponentially, you may want a smaller threshold. If you tune it, report the
threshold alongside the verdict so your result is reproducible.

```python
# more sensitive threshold for a 12-qubit hardware-efficient ansatz
result = detect_barren_plateau(trace, threshold=0.0001)
```

---

## Limitations

- This detects a **cost-landscape** barren plateau — variance suppression in
  the objective value. It is not the same as a **gradient** barren plateau
  (exponentially small parameter-shift gradients). The two are correlated but
  not identical.
- A short run with few evaluations will have high variance uncertainty. Check
  `verdict_confidence` and `variance_ci` before acting on the verdict.
- Works best for optimizers where each evaluation is one span (e.g., COBYLA,
  Nelder-Mead). For batched QNN training each mini-batch contributes many
  spans per parameter update; the variance reflects batch variation rather
  than landscape flatness.

---

## Example

```python
from hilbertbench import HilbertTrace
from hilbertbench.analysis import detect_barren_plateau

trace = HilbertTrace("runs/vqe/20260605_143022_a1b2c3d4")
result = detect_barren_plateau(trace)

if result['status'] == 'Barren Plateau Detected':
    print(f"Barren plateau — variance {result['variance']:.2e} "
          f"below threshold {result['threshold']}")
    print(f"CI: {result['variance_ci']}  confidence: {result['verdict_confidence']}")
elif result['verdict_confidence'] == 'low':
    print("Borderline result — run a longer experiment")
else:
    print(f"Trainable — variance {result['variance']:.4f}")
```
