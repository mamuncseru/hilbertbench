# Design Decisions

This directory holds the architecture decision records (ADRs) for
HilbertBench — short documents that capture *why* a significant choice
was made, what was traded away, and what it commits us to. An ADR is
written once, when the decision is made, and then left as a record. If
a later decision reverses an earlier one, that is a new ADR, not an
edit to the old.

The point of keeping them is simple: six months from now, someone
(possibly the author) will look at a constraint that seems
inconvenient and ask "why on earth is it like this?" The ADR is the
answer, written while the reasoning was fresh.

Where the [invariants](../reference/invariants.md) state the
rules and the [principles](../architecture/002_principles.md) explain
the shape of the system, the ADRs record the *moments of choice* — the
places where a different decision was genuinely possible and we picked
one path on purpose.

## Records

| ADR | Decision | Enforces |
|-----|----------|----------|
| [ADR-0001](0001_trace_atomicity.md) | Traces are append-only during recording and sealed on close | [INV-002](../reference/invariants.md) |
| [ADR-0002](0002_evidence_vs_interpretation.md) | Traces store evidence only; diagnoses are computed on read | [INV-006](../reference/invariants.md) |

## Format

Each record follows the same short structure: **Context** (the forces
at play), **Decision** (what we chose), **Consequences** (what we gain
and what we give up). New ADRs are numbered in sequence and added to
the table above.
