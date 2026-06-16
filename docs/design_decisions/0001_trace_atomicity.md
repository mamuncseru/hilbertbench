# ADR-0001: Traces are append-only and sealed on close

**Status:** Accepted · **Enforces:** [INV-002](../architecture/001_invariants.md)

## Context

A HilbertBench trace is scientific evidence. It may be the basis of a
published result, handed to a second researcher who did not run the
experiment, or re-analyzed years after it was recorded. Two failure
modes would make that evidence worthless:

1. **Silent mutation.** If a trace can be edited after the fact — by a
   migration script, a reader utility, a well-meaning "cleanup" — then
   no one can trust that what they are reading is what actually
   happened.
2. **Crash corruption.** Quantum experiments are long-running and
   submitted to flaky queues. If a crash mid-run could leave the trace
   in a half-written, ambiguous state, every long experiment becomes a
   gamble.

We needed a storage discipline where a finalized trace is provably
untouched, and an interrupted trace still yields all the evidence
gathered up to the moment of failure.

## Decision

Traces are **append-only during recording** and **cryptographically
sealed on close.**

- The event stream is written to `events.jsonl`, one JSON object per
  line, append-only. A record, once written, is never modified.
- When the `HilbertTape` context exits, the tape computes a SHA-256
  checksum over the entire event stream and writes it into
  `trace.json` as the integrity seal.
- The read path has no write privileges. Any tool that needs to reshape
  a trace produces a *new* derived trace and leaves the original
  untouched.
- `trace.verify()` recomputes the seal and the causal ordering, and
  raises if a single byte has moved.

## Consequences

**What we gain.** A sealed trace is tamper-evident: anyone can verify
it has not changed since it was written, which is precisely the
property the [blinded validation protocol](../end-to-end.md) depends on
— nobody, including the tool's authors, can quietly alter recorded
evidence. The append-only log is also crash-resilient: an interrupted
run loses at most the final partial line, and everything before it is
valid, readable evidence. An unsealed trace is detectable as such.

**What we give up.** No in-place edits, ever. Fixing a mistake in a
trace means generating a corrected derivative, not patching the
original — more disk, more discipline. Append-only JSONL is also not
the most compact on-disk form, which is why a separate, *additive*
Parquet export exists ([`hb-export`](../reference/trace-format.md)) for
analytical workloads; it never replaces the JSONL source of truth.

**What it commits us to.** This decision is the foundation of
[INV-002](../architecture/001_invariants.md). It means migrations are
forward-only, the reader stays strictly read-only, and the seal is part
of the trace contract, not an optional add-on.
