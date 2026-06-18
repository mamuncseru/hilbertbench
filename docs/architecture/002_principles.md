# Design Principles

The [invariants](../reference/invariants.md) say what the code must never do.
This page says *why* the framework is shaped the way it is — the
handful of ideas that, once you hold them, make the rest of the
codebase predictable.

## 1. Record at the execution boundary, not in the user's code

Most diagnostic tools ask you to instrument your training loop. We
refused, for one reason: coverage. Whatever framework, optimizer, or
clever trick a user brings, everything they run eventually passes
through one narrow door — the call that submits circuits for execution
(a Qiskit V2 primitive's `run`, a PennyLane device's `execute`).
Intercept that door and you capture *everything*, with the user
changing a single line. Their code stays theirs; we watch the traffic.

A consequence worth stating plainly: HilbertBench knows nothing about
"models". It sees circuits, parameters, observables, and results. That
is a feature — it means the tool works on code it has never seen,
written by people who never heard of it.

## 2. The recorder is passive; the analysis is deferred

The recorder does as little as physically possible while the
experiment runs: serialize what was about to happen anyway, hand it to
an append-only log, return. Every interpretation — variance,
expressibility, fidelity — happens later, offline, against the sealed
trace. This split is what makes [INV-001](../reference/invariants.md)
(execution parity) achievable: a tool that computed diagnostics inline
would change the experiment's timing, and on real hardware, timing is
noise.

It also unlocks the project's larger goal. A sealed trace is portable
evidence. One group with hardware can publish their traces; another
group with none can download them and run every analyzer. The science
separates from the machine.

## 3. Evidence and interpretation never mix

A trace records what happened. It contains no verdict, no
"converged", no "noisy". Diagnoses live outside the trace, computed on
read ([INV-006](../reference/invariants.md)). The reason is humility: today's
threshold for "barren plateau" is a heuristic, and next year's
analyzer will be better. If we baked a verdict into the trace, we
would freeze the evidence around one fallible opinion. Keep them
apart, and the same raw trace can be re-judged forever. The reasoning
behind this one is recorded in full as
[ADR-0002](../design_decisions/0002_evidence_vs_interpretation.md).

## 4. The schema is the contract; the code is an implementation

The trace format is defined by language-agnostic JSON Schema, and the
Python models are generated from it ([INV-003](../reference/invariants.md)). A
trace written today must be readable by a tool written in 2035, in a
language that may not exist yet. That is only possible if the format is
specified independently of any one implementation. The Python you see
in `hilbertbench/models/` is build output, not the source of truth.

## 5. A small, dependency-free core

The parts that handle evidence — the recorder, the reader, the data
models — import nothing but the Python standard library
([INV-004](../reference/invariants.md)). Heavy, fast-moving libraries (Qiskit,
PennyLane, PyArrow) are confined to the integration and storage layers.
The payoff is trust: the evidence-handling core is small enough to
audit by reading, and you can open and verify someone's traces without
installing a quantum framework at all.

## 6. Fail loudly, degrade gracefully

These sound opposed; they are not. When something breaks *now* — a
circuit throws, a span cannot close — the failure is recorded and
re-raised, never swallowed ([INV-007](../reference/invariants.md)). When
something is *absent* — an old trace lacks a field a new analyzer
wants — the reader returns `None` and the analyzer reports
"insufficient data", never a fabricated default
([INV-008](../reference/invariants.md)). A present failure must be visible; an
absent measurement must not be invented. Both rules serve the same
master: the trace never lies.

---

These principles are not aspirations bolted on after the fact — each
is enforced by an invariant and checked by the
[compliance test suite](../reference/test-catalog.md). When a design
question comes up, the answer is usually "which principle is at
stake?"
