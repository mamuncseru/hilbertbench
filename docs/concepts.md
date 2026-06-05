# Concepts

This page explains the core ideas behind HilbertBench. Reading it is not
required to use the library, but it will make the design feel inevitable
rather than arbitrary.

---

## The problem with quantum ML diagnostics

Quantum ML experiments fail for many reasons: barren plateaus suppress
gradients, shot noise buries the optimization signal, hardware decoherence
corrupts outcomes. Diagnosing which problem you have requires evidence —
the actual circuit executions, parameter trajectories, and outcome values
from your real training run.

The naive approach is to add logging calls throughout your code. But this
couples your diagnostic infrastructure to your algorithm code, makes
reproductions depend on which logging calls you remembered to add, and
changes the program you are trying to observe.

HilbertBench takes a different approach: **intercept at the execution
boundary, record everything, analyze later.**

---

## Passive recording

The proxy sits between your optimizer and the quantum backend. When your
code calls `estimator.run(...)`, the proxy:

1. Passes the call through to the real estimator — unchanged
2. Intercepts the result before returning it to your code
3. Writes a span record to the trace file — asynchronously, never blocking

Your optimizer sees exactly the same result it would have seen without the
proxy. The proxy **never** re-executes circuits, modifies shot counts, or
injects any additional quantum jobs.

```
Your optimizer loop
      │
      ▼
HilbertEstimatorProxy.run(circuit, observable, params)
      │
      ├─► StatevectorEstimator.run(...)  ← real execution, unmodified
      │         │
      │         ▼ result
      │
      ├─► record span to events.jsonl   ← non-blocking, background write
      │
      └─► return result to your code    ← same result, same timing
```

This is INV-001, the **execution parity** invariant: the number of circuit
executions, shots, and parameter bindings is identical to what your code
would have produced without HilbertBench.

---

## Traces and spans

A **trace** is the complete record of one benchmark run. It lives in a
timestamped directory:

```
runs/my_experiment/
└── 20260605_143022_a1b2c3d4/
    ├── events.jsonl     ← the raw event log
    ├── manifest.json    ← run metadata and integrity seal
    ├── catalog.json     ← index of stored artifacts
    └── store/           ← file store for large artifacts
```

A **span** is the record of one circuit execution. Every time your
optimizer calls `estimator.run(...)`, one span is written. A span contains:

| Field | What it stores |
|---|---|
| `outcome` | Expectation value (Estimator) or bitstring counts (Sampler) |
| `parameters` | Bound parameter vector |
| `circuit` | OpenQASM reference (deduplicated: stored once per unique structure) |
| `observable` | Pauli string or operator definition |
| `shots` | Shot count, if specified |
| `backend_id` | Backend or simulator identifier |
| `sequence_number` | Monotonically increasing, for ordering |
| `timestamps` | UTC start and end of the execution |
| `status` | `COMPLETED` or `ERROR` |

The circuit QASM is **content-addressed**: the same circuit structure is
stored exactly once, regardless of how many times it is executed. Only the
parameter bindings differ between spans.

---

## Reading a trace

`HilbertTrace` is the single read API. It gives you clean access to every
span without needing to know about JSONL versus Parquet, inline versus
file-store artifacts, or content hashes.

```python
from hilbertbench import HilbertTrace

trace = HilbertTrace("runs/my_experiment/20260605_143022_a1b2c3d4")

# high-level properties
trace.status       # "SEALED_SUCCESS" or "SEALED_ERROR"
trace.mode         # "passive" or "active"
len(trace)         # total span count
trace.tags         # dict of run-level labels

# iterate spans
for span in trace.completed():
    span.outcome      # float, dict, or numpy array — resolved automatically
    span.parameters   # numpy array or None
    span.circuit      # QASM string or None

# batch access
outcomes = trace.numeric_outcomes()   # flat numpy array of all scalar outcomes
```

---

## Append-only traces

Once a span is written it is never modified. When the `with HilbertTape`
block exits, the trace is **sealed**: a cryptographic hash of the full event
log is written into `manifest.json`. Any subsequent tampering — manual edits,
corrupted writes — is detectable.

This immutability guarantee (INV-002) means you can share a trace directory
and a collaborator can verify it has not been altered:

```python
trace.verify()   # raises IntegrityError if the seal is broken
```

---

## Diagnostic axes

HilbertBench organizes diagnostics into five axes. Each axis addresses a
distinct failure mode.

| Axis | Analyzer | Key question |
|---|---|---|
| **Ansatz — trainability** | `detect_barren_plateau` | Is the cost-landscape variance exponentially vanishing? |
| **Measurement** | `shot_noise_ratio` | Is the optimizer signal buried in shot noise? |
| **Optimization** | `optimization_convergence` | Is the parameter trajectory still moving? |
| **Circuit structure** | `circuit_structure` | How deep, how many qubits, what gate composition? |
| **Ansatz — expressibility** | `kl_expressibility` | How uniformly does the ansatz cover Hilbert space? |
| **Noise** | `noise_profile` | What hardware decoherence did the run experience? |

You can run all of them at once:

```python
from hilbertbench.analysis import summary

report = summary(trace)
# top-level keys: trace, trainability, measurement, optimization,
#                 circuit, noise
```

---

## Evidence vs interpretation

HilbertBench records **what happened**, not **what it means**.

A span stores: circuit submitted, parameters bound, outcome observed. It
does not store: "this looks like a barren plateau" or "convergence
achieved". That interpretation is performed by the analyzer functions at
read time, outside the trace.

This separation has a concrete consequence: the same raw trace can be
re-analyzed with different thresholds, different algorithms, or future
analyzers that do not yet exist. The evidence is permanent; the
interpretation is always re-derivable.

---

## Active Mode

Passive recording cannot measure **expressibility** — how uniformly an
ansatz's output states cover Hilbert space. Expressibility requires states
under *uniformly random* parameters, but a training trajectory uses
optimizer-chosen parameters that cluster near the minimum.

**Active Mode** is an explicit user action that samples the ansatz at
random parameters and records the resulting statevectors:

```python
from hilbertbench.active import active_probe_qiskit

run_dir = active_probe_qiskit(
    ansatz,
    num_samples=1000,
    output_root="runs/expressibility_probe",
)

from hilbertbench.analysis import kl_expressibility
result = kl_expressibility(run_dir)
```

Active Mode creates its own trace (marked `mode="active"`) and runs new
circuits explicitly. It is never triggered automatically. Passive recording
remains passive.

---

## What HilbertBench does not do

- It does not modify your circuit or add ancilla qubits.
- It does not re-run executions or add shots.
- It does not write diagnostic conclusions into the trace.
- It does not require you to restructure your code beyond swapping the proxy.
- It does not produce definitive causal attributions — it produces evidence
  and interpretations with explicit confidence intervals and caveats.
