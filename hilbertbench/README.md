# `hilbertbench/` — Package Architecture Map

This is the importable Python package. It is organized as a one-way
dependency stack: the **write path** (recorder) and the **read path**
(reader/trace/analysis) meet only through the trace files on disk, not
through shared runtime state. That separation is what lets you analyze
a trace on a different machine, in a different year, that never touched
the experiment.

```
            ┌─────────────────────────────────────────────┐
  RECORD    │  integrations/  ──►  recorder/  ──►  models/ │   (write path)
            └─────────────────────────────────────────────┘
                              │  sealed trace on disk
                              ▼
            ┌─────────────────────────────────────────────┐
  ANALYZE   │  reader/  ──►  trace/  ──►  analysis/        │   (read path)
            │                          active/  (opt-in)   │
            └─────────────────────────────────────────────┘
```

## The subpackages

| Package | Role | Depends on |
|---------|------|------------|
| **`models/`** | The trace data model (manifest, span, artifact, catalog). **Generated from JSON Schema — do not edit by hand.** | stdlib only (INV-004) |
| **`recorder/`** | The write path: `HilbertTape`, spans, events, and the content-addressed artifact store under `recorder/storage/`. Append-only, then sealed. | `models/`, stdlib only (INV-004) |
| **`reader/`** | Trace verification — recomputes hashes and checks causal integrity; fails loudly on tampering. | `models/`, stdlib only (INV-004) |
| **`trace/`** | The public read API: `HilbertTrace` resolves spans, outcomes, parameters, circuits, and calibration history out of a run directory. | `reader/`, `models/` |
| **`analysis/`** | The six diagnostic analyzers (barren plateau, shot noise, optimization, circuit structure, expressibility, noise profile). Pure functions: trace in, dict out. | `trace/`, numpy |
| **`active/`** | Active Mode — the opt-in interventional probe (expressibility sampling) that records the extra executions structural diagnostics need. | `trace/`, framework libs |
| **`integrations/`** | Transparent proxies for Qiskit (Estimator/Sampler/backend) and PennyLane. The only place heavy quantum libraries are imported. | `recorder/`, qiskit/pennylane |

## The dependency rule (INV-004)

`models/`, `recorder/`, and `reader/` import **only the Python standard
library and each other**. Heavy, volatile third-party libraries
(`qiskit`, `pennylane`, `pyarrow`) are confined to `integrations/`,
`analysis/`, `active/`, and `recorder/storage/`. This keeps the
evidence-handling core small, auditable, and cheap to import — a
recorder-only user never pays for numpy.

The top-level [`__init__.py`](__init__.py) exposes `HilbertTrace` and
`SpanView` lazily (PEP 562), so `import hilbertbench` stays lightweight
until you actually reach for the analysis layer.

## Where to start reading

- **Using the tool:** [`../docs/end-to-end.md`](../docs/end-to-end.md)
  walks the whole thing top-down.
- **The data model:** [`models/README.md`](models/README.md) and
  [`../schemas/README.md`](../schemas/README.md).
- **The guarantees** referenced as `INV-NNN` throughout the source:
  [`../docs/reference/invariants.md`](../docs/reference/invariants.md).
- **Contributing:** [`../CONTRIBUTING.md`](../CONTRIBUTING.md).
