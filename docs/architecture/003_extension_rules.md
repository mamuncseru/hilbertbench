# Extending HilbertBench

HilbertBench is built to be extended in three places: new analyzers,
new framework integrations, and new trace-schema fields. Each has a
safe path and a few rules that keep an extension from quietly breaking
the guarantees the framework rests on. The full contributor checklist
is in [CONTRIBUTING.md](https://github.com/mamuncseru/hilbertbench/blob/main/CONTRIBUTING.md);
this page is the architectural how-to.

## Adding an analyzer

An analyzer is the easiest thing to add, because it only *reads*.

A new analyzer is a plain function: it takes a trace (or a run-directory
path) and returns a dictionary. It lives in `hilbertbench/analysis/`,
and it must follow the same contract as the existing six:

- **Read-only.** It never writes to the trace ([INV-002](001_invariants.md)).
  Take a `HilbertTrace`, resolve what you need, compute, return.
- **Evidence in, dictionary out.** Return a `status` string plus the
  quantitative evidence behind it. Do not invent a verdict the numbers
  do not support.
- **Quantify uncertainty.** If you estimate a statistic, attach a
  bootstrap confidence interval (see `analysis/_util.py` for the
  shared helper) and let the verdict report low confidence when the
  interval straddles a threshold.
- **Degrade, don't guess.** If the trace lacks what you need (an old
  trace with no calibration snapshot, say), return an "insufficient
  data" status — never a fabricated default ([INV-008](001_invariants.md)).
- **Document the threshold.** Any decision cutoff is a named constant
  with a comment explaining where the number came from. Expose it as a
  function argument so callers can override it.

Add a test in `tests/analysis/` that plants a known condition in a
constructed trace and asserts the verdict, then regenerate the
[test catalog](../reference/test-catalog.md).

## Adding a framework integration

An integration is a transparent proxy. It is the only place a heavy
quantum library may be imported, and it carries the heaviest
responsibility, because it runs *during* the user's experiment.

- **Parity is sacred.** The proxy must not change the number of shots,
  executions, parameter bindings, or observables ([INV-001](001_invariants.md)).
  Forward the call to the real backend, wait for the result the user
  would have gotten anyway, and copy it aside. Never re-execute.
- **Record after, not instead.** Run the real call first; record from
  its result. Recording must never sit on the critical path in a way
  that could alter timing or fail the user's job.
- **Failures are visible.** Wrap recording so that a problem writing
  the trace cannot break the user's run, but is itself logged as an
  `ERROR` event, not silently dropped ([INV-007](001_invariants.md)).
- **Import locally.** Keep the framework import inside the integration
  module so the core stays dependency-free ([INV-004](001_invariants.md)),
  and add a clear message if an optional dependency is missing.

The existing [Qiskit](https://github.com/mamuncseru/hilbertbench/blob/main/hilbertbench/integrations/qiskit.py)
and [PennyLane](https://github.com/mamuncseru/hilbertbench/blob/main/hilbertbench/integrations/pennylane.py)
proxies are the templates to copy.

## Adding a trace-schema field

This is the one with the strictest rules, because the trace format is
a long-lived contract.

- **Edit the schema, never the models.** Change the JSON Schema in
  `schemas/`, then regenerate the Python models
  ([INV-003](001_invariants.md)). See the
  [schema guide](https://github.com/mamuncseru/hilbertbench/blob/main/schemas/README.md).
- **Respect the version freeze.** Once a schema version is tagged it is
  frozen forever ([INV-005](001_invariants.md)). A new field goes into
  a new version directory (`v1.1/`), not into a released one.
- **New fields are optional and degrade gracefully.** Anything not
  present since the first release must be optional and nullable, so a
  reader of an older trace resolves it to `None`
  ([INV-008](001_invariants.md)).
- **Evidence only.** The new field records *what happened*, never an
  interpretation of it ([INV-006](001_invariants.md)). Fields like
  `is_converged` or `quality_score` do not belong in a trace.

## The rule behind the rules

Every extension rule traces back to one question: *does this change
keep the trace trustworthy?* If an addition could let the recorder
perturb an experiment, let a trace be edited after the fact, or let an
analyzer invent data, it is rejected — regardless of how useful it
seems. The [compliance suite](../reference/test-catalog.md) exists to
catch the cases where good intentions would have broken a guarantee.
