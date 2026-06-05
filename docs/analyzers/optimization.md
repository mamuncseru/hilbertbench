# Optimization Convergence

```python
from hilbertbench.analysis import optimization_convergence

result = optimization_convergence(trace)
```

---

## What it measures

`optimization_convergence` characterises how the classical optimizer is
behaving by reading two signals from the recorded trace:

1. **Outcome envelope** — how the cost value changes from the start of
   the run to the end: initial mean, final mean, min, max, trend.
2. **Parameter movement** — how much the parameter vector changes between
   consecutive spans, computed as the L2 norm of successive differences.

A well-converged run shows large early parameter movement (the optimizer is
exploring) that shrinks to near-zero in the second half (it has settled).
A stalled run shows near-zero movement throughout.

---

## Function signature

```python
optimization_convergence(
    trace: HilbertTrace | str | Path,
) -> dict
```

---

## Output fields

```python
{
    'status':            'Converging',
    'num_steps':         20,
    'outcome_initial':   -0.312,      # mean of first 10% of outcomes
    'outcome_final':     -0.981,      # mean of last 10% of outcomes
    'outcome_min':       -0.998,
    'outcome_max':        0.143,
    'outcome_trend':     -0.669,      # outcome_final - outcome_initial
    'total_path_length':  4.823,      # summed L2 parameter movement
    'early_movement':     0.412,      # mean movement in first half
    'late_movement':      0.087,      # mean movement in second half
    'movement_ratio':     0.211,      # late / early
}
```

| Field | Meaning |
|---|---|
| `status` | Human-readable convergence verdict |
| `num_steps` | Number of completed spans used |
| `outcome_initial` | Mean outcome in the first 10% of spans |
| `outcome_final` | Mean outcome in the last 10% of spans |
| `outcome_min` | Minimum outcome over the entire run |
| `outcome_max` | Maximum outcome |
| `outcome_trend` | `outcome_final − outcome_initial` (negative = cost decreased) |
| `total_path_length` | Sum of L2 norms of consecutive parameter changes |
| `early_movement` | Mean parameter movement in the first half of spans |
| `late_movement` | Mean parameter movement in the second half |
| `movement_ratio` | `late_movement / early_movement` |

---

## Verdicts

Classification is based on `movement_ratio`:

| `status` | Condition | Meaning |
|---|---|---|
| `Converged` | ratio `< 0.1` | Parameters have settled — optimizer has found a basin |
| `Converging` | `0.1 ≤` ratio `< 0.5` | Movement is slowing — still approaching a minimum |
| `Still Improving` | ratio `≥ 0.5` | Parameters are still moving at the same rate — run longer |
| `Stalled` | both early and late movement `≈ 0` | Optimizer is stuck — likely a flat landscape |
| `Insufficient Data` | `< 4` spans | Not enough points to characterise the trajectory |

---

## Scope note: batched QML runs

This analyzer was designed for **single-objective optimizers** (COBYLA,
Nelder-Mead, SPSA) where one span equals one objective evaluation. In that
setting, parameter movement between consecutive spans is a direct proxy for
optimizer step size.

For **batched QNN training** (PennyLane Adam, Qiskit QNN), each span is one
data sample or one parameter-shift probe. Consecutive spans differ by data
sample, not by optimizer update — `movement_ratio` stays near 1 throughout
and the convergence verdict is uninformative.

In batched runs, use the **outcome envelope fields** (`outcome_trend`,
`outcome_initial`, `outcome_final`) instead — they remain meaningful because
they track the cost value over time.

---

## Example

```python
from hilbertbench.analysis import optimization_convergence

result = optimization_convergence(trace)

print(f"Status:       {result['status']}")
print(f"Cost trend:   {result['outcome_trend']:+.3f}  "
      f"(initial {result['outcome_initial']:.3f} → "
      f"final {result['outcome_final']:.3f})")
print(f"Path length:  {result['total_path_length']:.3f}")
print(f"Movement ratio: {result['movement_ratio']:.3f}")
```
