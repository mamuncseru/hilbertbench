# HilbertBench: Architectural Invariants

**Status:** Active  
**Applies to:** All pull requests, schema modifications, and tool development.

These invariants are the non-negotiable laws of the HilbertBench repository. They ensure the scientific validity of the trace data and the long-term maintainability of the codebase. Any code or schema change that violates them MUST be rejected, regardless of convenience or urgency.

---

## INV-001: Execution Parity (The Observer Effect)
The recorder MUST NOT alter the number of shots, circuit executions, parameter bindings, or observable evaluations performed by the user's underlying code.

* **Implication:** The recorder operates strictly in passive interception mode at the execution boundary, writing to a background queue. It never re-executes, re-samples, or injects synthetic circuits into the hardware queue.
* **Violated by:** Any instrumentation hook that calls `backend.run()` outside of the original user call context, or modifies the `shots` argument of an intercepted call.

---

## INV-002: Trace Immutability
A written trace file MUST NOT be modified after it is finalized. Traces are append-only during the recording phase and cryptographically sealed upon closure.

* **Implication:** The `hilbertbench/reader/` path has zero write privileges. Any schema migrations or anomaly corrections MUST produce a new derived trace file or a secondary index; they do not overwrite the original source of truth.
* **Violated by:** Any diagnostic script, reader utility, or CLI tool that opens a `.parquet` or `.json` trace file in `w`, `a`, or `r+` mode.

---

## INV-003: Schema is the Sole Source of Truth
Python data models in `hilbertbench/models/` are ALWAYS auto-generated from the language-agnostic JSON schemas in `schemas/v*/`. They are NEVER edited by hand.

* **Implication:** If a model field requires modification, the JSON schema changes first. The developer must then run `schemas/scripts/generate_python_models.sh` to propagate the change.
* **Violated by:** Any manual Git commit that modifies `hilbertbench/models/v*.py` without a corresponding change to the upstream JSON schema.

---

## INV-004: Core Dependency Isolation
The foundational modules (`hilbertbench/recorder/`, `hilbertbench/reader/`, and `hilbertbench/models/`) MUST import only from the Python standard library and each other.

* **Implication:** Heavy or volatile third-party quantum libraries (e.g., `import qiskit`, `import pennylane`) and storage engines (`import pyarrow`) are strictly confined to `hilbertbench/integrations/` and `hilbertbench/storage/`. 
* **Verification:**
  ```bash
  # This CI check must produce NO output:
  grep -r "^import\|^from" hilbertbench/recorder/ \
    hilbertbench/reader/ hilbertbench/models/ \
    | grep -v "hilbertbench\|typing\|dataclasses\|abc\|uuid\|datetime\|pathlib\|hashlib\|json\|enum\|sys\|os"
  ```

---

## INV-005: Schema Version Freezing
Once a major schema version (e.g., `schemas/v1.0/`) is officially tagged in Git, no file within that directory may be modified. 

* **Implication:** The structure of "Bucket A" (Irreducible Facts) is permanent. If a new field must be added, or an optional field introduced, it requires the creation of `schemas/v1.1/`.
* **Violated by:** Any PR that modifies the contents of `schemas/v1.0/*.json` after the `schema-v1.0` Git tag has been cut.

---

## INV-006: Strict Separation of Evidence and Interpretation
A span recorded in a trace captures strictly WHAT happened (the physical constants: circuits submitted, parameters bound, UTC timestamps, outcomes observed). It records NO interpretation of what the data means.

* **Implication:** Diagnostic conclusions (e.g., "barren plateau detected", "noise level high") do not exist in the trace schema. They are generated dynamically by the `reader` path and stored externally.
* **Violated by:** Introducing fields like `is_converged`, `error_rate`, or `quality_score` into the trace schema.

---

## INV-007: Mandatory Failure Visibility
Silent failures are strictly prohibited. Every initiated execution span MUST conclude with either a successful `SPAN_END` event, an explicit `ERROR` event, or be structurally detectable as a catastrophic crash.

* **Implication:** Unhandled exceptions in the user's script or the integration adapter must be caught, logged to the trace as an `ERROR` event, and then re-raised. The logger must not swallow exceptions to "keep the trace clean."
* **Violated by:** `try...except Exception: pass` blocks within the `recorder` or `integrations` modules.

---

## INV-008: Graceful Schema Degradation
Analysis tools reading historical traces MUST NOT hallucinate data. If an optional field (e.g., `calibration_snapshot`) is missing from an older trace, the reader must degrade gracefully.

* **Implication:** The absence of evidence must never be interpreted as evidence of absence. Missing fields resolve to `Null`. Diagnostics relying on that data must return an `Insufficient Evidence` status, rather than assuming a default value of `0` or `False`.
* **Violated by:** Reader logic that assigns default fallback values to missing trace fields to force a diagnostic function to run.