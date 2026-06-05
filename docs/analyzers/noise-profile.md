# Noise Profile

```python
from hilbertbench.analysis import noise_profile

result = noise_profile(trace)
```

---

## What it measures

`noise_profile` reads the device calibration snapshot recorded at execution
time and estimates the hardware noise exposure of the run. It combines:

- **Coherence times** (T1, T2) from the calibration snapshot
- **Readout error** per qubit
- **Gate error** rates
- **Circuit structure** (depth and gate count from `circuit_structure`)

to produce an estimated circuit fidelity: the probability that a single
execution produces the correct output under the calibrated noise model.

This is a **coarse estimate** using standard NISQ product formulas — not a
full noise simulation. Its value is helping you correlate a loss spike or
accuracy drop with device calibration rather than your model alone.

!!! info "Simulator traces"
    Traces recorded on statevector simulators (`StatevectorEstimator`,
    `default.qubit`) carry no calibration snapshot. The function returns
    `None`-valued noise fields rather than raising an error — **degraded
    mode, not failure**.

---

## Function signature

```python
noise_profile(
    trace: HilbertTrace | str | Path,
) -> dict
```

---

## Output fields

```python
{
    'status':                   'Moderate Noise',
    'has_calibration':          True,
    'estimated_circuit_fidelity': 0.72,
    't1_us': {
        'mean': 85.3, 'min': 72.1, 'max': 98.4
    },
    't2_us': {
        'mean': 61.7, 'min': 44.2, 'max': 79.3
    },
    'readout_error': {
        'mean': 0.021, 'min': 0.008, 'max': 0.038
    },
    'gate_error': {
        'mean': 0.003, 'min': 0.001, 'max': 0.007
    },
    'num_qubits_calibrated': 2,
}
```

### For simulator traces (no calibration):

```python
{
    'status':                     'No Calibration Data',
    'has_calibration':            False,
    'estimated_circuit_fidelity': None,
    't1_us':        None,
    't2_us':        None,
    'readout_error': None,
    'gate_error':    None,
    'num_qubits_calibrated': 0,
}
```

| Field | Meaning |
|---|---|
| `status` | Human-readable noise verdict |
| `has_calibration` | Whether a calibration snapshot was found in the trace |
| `estimated_circuit_fidelity` | Product-formula estimate of single-run fidelity |
| `t1_us` | T1 coherence time statistics (mean/min/max, microseconds) |
| `t2_us` | T2 coherence time statistics (mean/min/max, microseconds) |
| `readout_error` | Readout error statistics (mean/min/max, probability) |
| `gate_error` | Gate error statistics (mean/min/max, probability) |
| `num_qubits_calibrated` | Number of qubits with calibration data |

---

## Verdicts

| `status` | Fidelity | Meaning |
|---|---|---|
| `Low Noise` | `≥ 0.90` | Device was well-calibrated — noise is unlikely to dominate |
| `Moderate Noise` | `0.50 – 0.90` | Meaningful noise exposure — consider error mitigation |
| `High Noise` | `< 0.50` | Fidelity below 50% — results may be dominated by noise |
| `No Calibration Data` | — | Simulator trace or backend did not expose calibration |

---

## Limitations

- The estimated fidelity uses the product approximation:
  `fidelity ≈ ∏ (1 − gate_error_i) × ∏ (1 − readout_error_j)`.
  This is a lower bound under the assumption of independent, Markovian errors.
  Crosstalk and non-Markovian effects can make the actual fidelity worse.
- Calibration data is a snapshot from the time the job was submitted. Device
  calibration drifts; results from long jobs may have been collected under
  different conditions.
- For traces with multiple circuit structures, the dominant circuit (by gate
  count) is used for the fidelity estimate.

---

## Example

```python
from hilbertbench.analysis import noise_profile

result = noise_profile(trace)

if not result['has_calibration']:
    print("Simulator trace — no hardware noise data")
else:
    print(f"Noise verdict:   {result['status']}")
    print(f"Est. fidelity:   {result['estimated_circuit_fidelity']:.1%}")
    print(f"T1 (mean):       {result['t1_us']['mean']:.1f} µs")
    print(f"Readout error:   {result['readout_error']['mean']:.3f}")
```
