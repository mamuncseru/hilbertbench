# ADR-0002: Traces store evidence; diagnoses are computed on read

**Status:** Accepted · **Enforces:** [INV-006](../reference/invariants.md)

## Context

The obvious way to build a diagnostic tool is to compute the diagnosis
while you have the data in hand and store the conclusion alongside it:
record the run, decide "barren plateau", write `barren_plateau: true`
into the trace. It is convenient, and it is a trap.

Every diagnosis rests on choices that are not facts:

- a **threshold** (variance below 0.005 is "flat") that is a heuristic,
  not a law of nature;
- an **algorithm** (this estimator of expressibility, this convergence
  test) that a better one will someday replace;
- a **version** of the tool, with whatever bugs it had that week.

If we write the verdict into the trace, we freeze the evidence around
those choices forever. A trace recorded today would carry today's
fallible opinion, and a sharper analyzer next year could never revisit
it honestly — the conclusion would already be baked in.

## Decision

A trace records **strictly what happened** — circuits submitted,
parameters bound, timestamps, outcomes observed, device calibration at
execution time. It records **no interpretation.** Diagnostic
conclusions are computed dynamically, on read, by the `analysis` layer,
and are never written back into the trace.

Concretely: fields like `is_converged`, `error_rate`, or
`quality_score` do not exist in the trace schema. The analyzers take a
trace and return their verdicts as separate, ephemeral dictionaries.

## Consequences

**What we gain.** The same raw trace can be re-analyzed any number of
times — with a different threshold, a new analyzer, a corrected
algorithm — and each analysis is honest, because the evidence was never
contaminated by an earlier verdict. Two researchers can disagree about
what a trace *means* while agreeing completely on what it *contains*.
And a trace can be published for others to analyze independently, which
is the whole premise of the project's "data-only research" model.

**What we give up.** Convenience and a little redundancy. A trace alone
does not tell you "this run was healthy" — you have to run an analyzer
to find out, every time, and analyzers must be robust to being pointed
at any trace. There is no cached verdict to read cheaply.

**What it commits us to.** This is [INV-006](../reference/invariants.md),
and it constrains both ends of the system: the recorder and schema must
never grow an interpretation field, and the analysis layer must treat
the trace as immutable, read-only input. It also pairs with
[INV-008](../reference/invariants.md) — when an analyzer meets a
trace missing the evidence it needs, it reports "insufficient data"
rather than inventing a verdict, because a guessed conclusion is exactly
the contamination this decision exists to prevent.
