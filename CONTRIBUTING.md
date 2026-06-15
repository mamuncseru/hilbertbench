# Contributing to HilbertBench

Thank you for considering a contribution. HilbertBench records
scientific evidence, so the bar for changes is a little higher than a
typical library: a bug in the recorder can silently corrupt someone's
experimental data. This guide explains the rules that keep that from
happening and how to work with the codebase.

---

## The architectural invariants (INV-NNN)

Throughout the source you will see references like `# Adheres to
INV-001` or `INV-007 is honoured at the batch level`. These are the
project's **non-negotiable guarantees**. There are eight of them and
they are numbered INV-001 through INV-008.

**Where they are defined:**

- Canonical, rendered list — **[docs/reference/invariants.md](docs/reference/invariants.md)**
  (also on the [reference site](https://hilbertbench.readthedocs.io)).
- Contributor rationale with verification commands —
  [docs/architecture/001_invariants.md](docs/architecture/001_invariants.md).

When you see an `INV-NNN` tag anywhere in the code, those documents
are where it comes from. The short version:

| ID | Guarantee |
|----|-----------|
| **INV-001** | **Execution parity** — never change shots, executions, bindings, or observables of the user's run |
| **INV-002** | **Trace immutability** — a finalized trace is append-only then sealed; never rewritten |
| **INV-003** | **Schema is the sole source of truth** — Python models are generated from JSON, never hand-edited |
| **INV-004** | **Core dependency isolation** — `recorder/`, `reader/`, `models/` import only the standard library |
| **INV-005** | **Schema version freezing** — a tagged `schemas/vX.Y/` is frozen; new fields mean a new version |
| **INV-006** | **Evidence, not interpretation** — traces store what happened, never a diagnosis |
| **INV-007** | **Mandatory failure visibility** — every span ends with success, an `ERROR` event, or a detectable crash |
| **INV-008** | **Graceful schema degradation** — missing optional fields resolve to `None`, never a hallucinated default |

A change that violates an invariant must be rejected regardless of how
convenient it is. If you believe an invariant itself needs to change,
that is a design discussion (open an issue) — not a code review.

---

## Editing the trace schema (INV-003 / INV-005)

This is the rule new contributors trip over most often, so it gets its
own section.

**Do not edit the Python files in `hilbertbench/models/`.** They are
auto-generated. Your edits will be silently overwritten the next time
anyone regenerates them.

To change the trace data model:

1. Edit the JSON schema in `schemas/v1.0/` (see
   [schemas/README.md](schemas/README.md)).
2. Regenerate the Python models:
   ```bash
   python schemas/scripts/generate_models.py
   ```
3. Commit **both** the schema change and the regenerated models
   together.

If the schema is already tagged (`schema-v1.0`), you cannot modify it
in place (INV-005) — you create `schemas/v1.1/` instead.
See [hilbertbench/models/README.md](hilbertbench/models/README.md) for
the full rationale.

---

## Development setup

```bash
git clone https://github.com/mamuncseru/hilbertbench
cd hilbertbench
pip install -e ".[qiskit,pennylane,storage,dev]"
python -m pytest tests/ -q          # all tests should pass
```

## Before you open a pull request

- **Tests pass:** `python -m pytest tests/ -q`. Add tests for new
  behaviour — see [tests/README.md](tests/README.md) for where each
  kind of test lives.
- **Lint is clean:** `ruff check hilbertbench/ tests/`.
- **Docstrings are complete:** every public function, method, and
  class carries a docstring whose `arguments:` block names every
  parameter. Match the existing heavy format in library code; a
  concise one-liner is fine for nested closures.
- **Invariants hold:** if you touched `recorder/`, `reader/`,
  `models/`, or `integrations/`, confirm the relevant INV checks (the
  `tests/compliance/` suite enforces several of them).
- **Regenerate the test catalog** if you added or renamed tests:
  ```bash
  python tools/gen_test_catalog.py
  ```

## Coding style

The library and demos use a heavy header/docstring format (file
banner, per-function `arguments:`/`return:`/`description:` blocks,
`# end of function` markers). Tests use a lighter, conventional style.
Keep lines within an 80-character soft limit. When in doubt, match the
file you are editing.

## Commit and PR hygiene

- One logical change per PR.
- Reference the relevant `INV-NNN` in the description if your change
  touches an invariant boundary.
- Schema changes and their regenerated models go in the same commit.

---

Questions about any of this are welcome as issues. Thank you for
helping keep the evidence trustworthy.
