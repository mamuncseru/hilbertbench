# Architectural Invariants

HilbertBench records scientific evidence, and these eight rules are
what make that evidence trustworthy. They are referenced throughout the
source as `INV-001` … `INV-008`. A change that breaks one is rejected in
review no matter how convenient it is; if an invariant genuinely needs
to change, that is a design discussion, not a patch.

Each entry below states the rule, then explains the failure it exists
to prevent — because the rule only makes sense once you have seen what
goes wrong without it.

---

## INV-001 — Execution Parity

**The recorder must not change the number of shots, circuit executions,
parameter bindings, or observable evaluations the user's code performs.**

This is the one that justifies the word "non-intrusive". On
superconducting hardware, a circuit's noise depends on when it runs, so
a diagnostic that quietly fires an extra job — or pads the shot count
"just to be safe" — has changed the very thing it claims to observe, and
on paid hardware it has also spent the user's money. The proxy therefore
only watches. It forwards each call to the real backend, waits for the
result the user would have gotten anyway, and copies it aside. The
optimizer cannot tell HilbertBench is attached.

*Broken by* an integration hook that calls `backend.run()` outside the
user's original call, or that touches the `shots` argument on the way
through.

---

## INV-002 — Trace Immutability

**A finalized trace is never modified.**

A flight recorder you can edit after the crash is worthless. During
recording the event log only ever grows; when the `HilbertTape` context
closes, the trace is sealed with a SHA-256 checksum over the whole event
stream. Nothing on the read side can write — not the reader, not an
analyzer, not a migration. If a trace must be reshaped, that produces a
new derived trace and leaves the original untouched.

*Checked by* `trace.verify()`, which recomputes the seal and raises
`IntegrityError` if a single byte has moved.

---

## INV-003 — The Schema is the Only Source of Truth

**The Python models in `hilbertbench/models/` are generated from the
JSON schemas in `schemas/`. They are never hand-edited.**

A trace is meant to be read years from now by tools written in languages
that may not exist yet. If the format lived only in Python, the
"specification" would be whatever the code happened to do that week. So
the JSON Schema is authoritative and the Python is compiled output, the
way an object file is compiled from source. Want to change a field? Edit
the schema, run `python schemas/scripts/generate_models.py`, commit both
together.

*Broken by* a commit that edits `hilbertbench/models/v*.py` without a
matching schema change — the tell-tale sign of a hand-edit.

---

## INV-004 — The Core Imports Only the Standard Library

**`recorder/`, `reader/`, and `models/` depend on nothing but the Python
standard library and each other.**

Recording evidence is too important to be hostage to a heavy, fast-moving
quantum stack. Keeping the core dependency-free means it stays small
enough to audit by reading, and a researcher can open and verify someone
else's traces without installing Qiskit at all. Everything volatile —
`qiskit`, `pennylane`, `pyarrow` — lives in `integrations/`, `analysis/`,
and the storage layer, where it belongs.

---

## INV-005 — Tagged Schema Versions Are Frozen

**Once a schema version is tagged in Git, no file in that directory ever
changes again.**

The whole promise of INV-003 collapses if `v1.0` can quietly mean
something different next month. So a released schema version is permanent.
A new or changed field means a new version directory (`v1.1/`, `v2.0/`),
which guarantees that a trace written today still parses against any
future reader that understands its version.

---

## INV-006 — Evidence and Interpretation Stay Separate

**A span records what physically happened — circuits, parameters,
timestamps, outcomes. It records no judgement about what any of it
means.**

The moment a verdict like "barren plateau detected" is written into a
trace, that trace is frozen around one analysis, made with one set of
thresholds, by one version of the tool. Keep interpretation out, and the
same raw trace can be re-diagnosed tomorrow with a sharper analyzer or a
different threshold — which is exactly the workflow the project is built
for. Diagnoses are computed on read, by the `analysis` layer, and never
flow back.

*Broken by* adding fields such as `is_converged`, `error_rate`, or
`quality_score` to the trace schema.

---

## INV-007 — Failures Are Always Visible

**Every span that starts must end in a way you can see: a clean
`SPAN_END`, an explicit `ERROR` event, or a structurally detectable
crash.**

The worst outcome for a diagnostic tool is to make a problem disappear.
If the user's circuit throws, the adapter catches it, writes an `ERROR`
event into the trace, and re-raises — the failure is both surfaced to the
user and preserved as evidence. What the recorder must never do is
swallow an exception to keep the trace looking tidy; a tidy trace that
hides a failure is a lie.

*Broken by* `try: ... except Exception: pass` anywhere in `recorder/` or
`integrations/`.

---

## INV-008 — Missing Data Degrades, Never Hallucinates

**A reader of an older trace must not invent values the trace does not
contain.**

Absence of evidence is not evidence of absence. If an optional field —
say, a calibration snapshot — was never recorded, it resolves to `None`,
and any diagnostic that needed it reports `Insufficient Data` and stops.
What it must not do is substitute a `0` or a `False` and proceed, because
that manufactures a confident answer out of nothing.

*Broken by* reader logic that fills missing fields with defaults just to
keep a diagnostic from short-circuiting.
