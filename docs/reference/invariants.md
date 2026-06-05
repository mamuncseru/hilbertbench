# Architectural Invariants

These invariants are the non-negotiable guarantees of HilbertBench. They
ensure the scientific validity of every trace and the long-term
trustworthiness of the framework.

Any code or schema change that violates them must be rejected, regardless
of convenience or urgency.

---

## INV-001: Execution Parity

> The recorder must not alter the number of shots, circuit executions,
> parameter bindings, or observable evaluations performed by the user's code.

The proxy operates strictly in passive interception mode. It never
re-executes circuits, re-samples, or injects additional quantum jobs. The
user's optimizer sees exactly the result it would have seen without
HilbertBench.

**Violated by:** any integration hook that calls `backend.run()` outside of
the original user call context, or modifies the `shots` argument of an
intercepted call.

---

## INV-002: Trace Immutability

> A written trace must not be modified after it is finalized.

Traces are append-only during the recording phase. When the `HilbertTape`
context exits, the trace is cryptographically sealed. The `reader` path has
zero write privileges. Schema migrations produce new derived traces rather
than overwriting originals.

**Verified by:** `trace.verify()` — raises `IntegrityError` if the seal is
broken.

---

## INV-003: Schema is the Sole Source of Truth

> Python data models in `hilbertbench/models/` are always generated from
> the JSON schemas in `schemas/v*/`. They are never edited by hand.

If a model field requires modification, the JSON schema changes first.
Manual edits to model files are not permitted.

---

## INV-004: Core Dependency Isolation

> The foundational modules (`recorder/`, `reader/`, `models/`) import only
> from the Python standard library and each other.

Quantum libraries (`qiskit`, `pennylane`) and storage engines (`pyarrow`)
are confined to `hilbertbench/integrations/` and storage modules.
The core can be imported and used without any quantum framework installed.

---

## INV-005: Schema Version Freezing

> Once a major schema version is officially tagged, no file in that directory
> may be modified.

The structure of recorded evidence is permanent. New fields require a new
schema version (`v1.1/`, `v2.0/`). This ensures that traces recorded against
`v1.0` remain readable by any future `v1.0`-compatible reader.

---

## INV-006: Strict Separation of Evidence and Interpretation

> A span captures strictly what happened — circuits submitted, parameters
> bound, timestamps, outcomes observed. It records no interpretation.

Diagnostic conclusions (`"barren plateau detected"`, `"noise level high"`)
do not exist in the trace schema. They are computed dynamically by the
`analysis` layer and never written back into the trace.

This ensures the same raw trace can be re-analyzed with different thresholds
or future analyzers.

**Violated by:** introducing fields like `is_converged`, `error_rate`, or
`quality_score` into the trace schema.

---

## INV-007: Mandatory Failure Visibility

> Silent failures are prohibited. Every initiated span must conclude with
> either a successful `SPAN_END` event, an explicit `ERROR` event, or be
> structurally detectable as a crash.

Unhandled exceptions in the integration adapter are caught, logged to the
trace as an `ERROR` event, and then re-raised. The recorder never swallows
exceptions to "keep the trace clean."

**Violated by:** `try: ... except Exception: pass` blocks in `recorder/` or
`integrations/`.

---

## INV-008: Graceful Schema Degradation

> Analysis tools reading historical traces must not hallucinate data.

If an optional field is missing from an older trace, the reader degrades
gracefully. Missing fields resolve to `None`. Diagnostics that rely on
absent data return `"Insufficient Data"` rather than assuming a default
value of `0` or `False`.

**Violated by:** reader logic that assigns default fallback values to missing
trace fields to force a diagnostic function to run.
