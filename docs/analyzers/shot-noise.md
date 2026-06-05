# Shot Noise Ratio

```python
from hilbertbench.analysis import shot_noise_ratio

result = shot_noise_ratio(trace)
```

---

## What it measures

Every expectation value estimated from a finite number of shots carries
**shot noise**: the estimator variance is bounded below by `1/shots`. If the
variance of the outcome trajectory is comparable to — or smaller than — that
floor, the optimizer is chasing noise rather than signal. More shots will
not fix a barren plateau, but they will fix shot-noise dominance.

`shot_noise_ratio` computes the ratio of empirical outcome variance to the
theoretical shot-noise floor:

```
SNR = empirical_variance / (1 / mean_shots)
```

A high SNR means the signal is real. A low SNR means you need more shots.

---

## Function signature

```python
shot_noise_ratio(
    trace: HilbertTrace | str | Path,
    default_shots: int | None = None,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int | None = None,
) -> dict
```

| Parameter | Default | Meaning |
|---|---|---|
| `trace` | — | `HilbertTrace` or path to a run directory |
| `default_shots` | `None` | Shot count to assume when none was recorded. Required for statevector simulators (which record no shot count). |
| `n_boot` | `1000` | Bootstrap resamples for the empirical variance CI |
| `ci` | `0.95` | Confidence level |
| `seed` | `None` | RNG seed |

!!! note "Statevector simulators"
    `StatevectorEstimator` and `default.qubit` compute exact expectation
    values — they do not use shots. If you are testing against a simulator,
    the shot count will not be recorded and you must pass `default_shots` to
    get an SNR estimate. The result is then a projection: "what would the SNR
    be if I ran with this many shots on real hardware?"

---

## Output fields

```python
{
    'status':                'Signal Clear (not limited by shot noise)',
    'empirical_variance':    0.215,
    'empirical_variance_ci': [0.098, 0.371],
    'theoretical_floor':     0.000977,          # 1 / mean_shots
    'estimated_snr':         220.1,
    'mean_shots':            1024.0,
    'num_evaluations':       20,
    'confidence_level':      0.95,
}
```

| Field | Meaning |
|---|---|
| `status` | Human-readable verdict |
| `empirical_variance` | Variance of the numeric outcome trajectory |
| `empirical_variance_ci` | Bootstrap CI on the empirical variance |
| `theoretical_floor` | `1 / mean_shots` — the expected shot-noise variance |
| `estimated_snr` | `empirical_variance / theoretical_floor` |
| `mean_shots` | Mean shot count across all spans |
| `num_evaluations` | Number of spans with a numeric outcome |

---

## Verdicts

| `status` | SNR | Meaning |
|---|---|---|
| `Shot Noise Dominated` | `< 1.5` | The signal is buried — increase shots significantly |
| `Marginal` | `1.5 – 5.0` | Borderline — more shots may improve convergence |
| `Signal Clear` | `≥ 5.0` | Shot noise is not the limiting factor |
| `Shot count not recorded` | — | Simulator run — pass `default_shots` to estimate |
| `Insufficient Data` | — | Fewer than 2 numeric outcomes |

---

## Limitations

- The shot-noise floor `1/shots` is a **lower bound** on estimator variance
  under ideal conditions. Real hardware readout errors and decoherence increase
  the effective noise floor; the SNR from this function will be optimistic on
  real hardware.
- For QNNs trained with mini-batches, `empirical_variance` reflects both
  batch-to-batch variation and shot noise. The SNR will be higher than on a
  single-shot optimizer even with few shots. Interpret accordingly.
- The function does not distinguish between high variance from a rich cost
  landscape (good) and high variance from noise (bad). Pair it with
  `detect_barren_plateau` to distinguish the two.

---

## Example

```python
from hilbertbench.analysis import shot_noise_ratio

# for a trace recorded on real hardware (shots are stored automatically)
result = shot_noise_ratio(trace)

# for a statevector simulator (shots not recorded)
result = shot_noise_ratio(trace, default_shots=1024)

print(result['status'])
print(f"SNR: {result['estimated_snr']:.1f}")
print(f"Mean shots: {result['mean_shots']:.0f}")
```
