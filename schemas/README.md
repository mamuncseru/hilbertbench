# `schemas/` — The Trace Format, Specified

This directory is the source of truth for the HilbertBench trace
format. The JSON Schema files here define what a trace *is*; everything
downstream — the Python data models, the readers, the analyzers —
follows from them.

The rule that follows from that: change the schema here and regenerate
the Python models. Never edit the generated models directly (INV-003).

```
schemas/v1.0/*.json   ──(scripts/generate_models.py)──►   hilbertbench/models/v1_0/*.py
   you edit these                                            generated, do-not-edit
```

## Layout

```
schemas/
├── v1.0/                  # version 1.0 of the format (JSON Schema, draft 2020-12)
│   ├── trace.json         # root manifest: identity, mode, status, integrity seal
│   ├── span.json          # one execution unit: events + physical evidence
│   ├── artifact.json      # metadata for one content-addressed file
│   └── catalog.json       # the hash → artifact-metadata index
└── scripts/
    └── generate_models.py # compiles the schemas into Python models
```

## The four schemas and what they own

| Schema | Title | Owns |
|--------|-------|------|
| `trace.json` | Trace Manifest | trace identity (`trace_id`, version), `mode` (passive/active), lifecycle `status`, `integrity_seal`, `client_environment`, tags. **Not** the events themselves. |
| `span.json` | Span | one causal execution unit: `span_id`, `sequence_number`, timestamps, `payload_ref`, and the immutable `events` sequence. |
| `artifact.json` | Artifact Metadata | one stored file's record: `artifact_hash`, `kind`, `encoding`, `file_path`, `size_bytes`, `compression`, `created_at`. |
| `catalog.json` | Artifact Catalog | the registry mapping SHA-256 hashes → artifact metadata for O(1) lookup. |

The split mirrors the trace directory on disk: `trace.json` is the
manifest, `events.jsonl` is the span stream, `catalog.json` indexes the
`artifacts/` store. See the
[trace format reference](../docs/reference/trace-format.md) for the
on-disk view.

## Why JSON Schema, not Python

A trace is scientific evidence meant to outlive this codebase and be
read by tools in any language (INV-003). A language-agnostic schema:

- **specifies** the format independently of the Python implementation;
- lets a Rust / Julia / JavaScript reader be generated from the same
  source;
- can be **frozen and versioned** so a 2026 trace still parses in 2030
  (INV-005).

## How to change the format

### Adding or changing a field while v1.0 is still unreleased

1. Edit the relevant `schemas/v1.0/*.json`. Keep it valid JSON Schema
   (draft 2020-12). Add a `description` to every new property — those
   descriptions become the docstrings on the generated models and the
   trace-format documentation.
2. Decide required vs optional. **Optional fields must degrade
   gracefully** (INV-008): a reader of an older trace that lacks the
   field resolves it to `None`, never a fabricated default. Prefer
   optional + nullable for anything not present since v1.0.
3. Regenerate the Python models:
   ```bash
   python schemas/scripts/generate_models.py
   ```
   The generator validates enums and root model names, re-applies the
   do-not-edit banner, and overwrites `hilbertbench/models/v1_0/`.
4. Run the tests that exercise the model/reader boundary:
   ```bash
   python -m pytest tests/trace tests/recorder tests/reader -q
   ```
5. Commit the schema change **and** the regenerated models together.

### Changing the format after v1.0 is tagged

Once the `schema-v1.0` Git tag exists, the contents of `schemas/v1.0/`
are **frozen** (INV-005). You do not edit them — you create
`schemas/v1.1/` (copy, then modify) and add a `v1_1` model package.
This guarantees every previously written trace remains valid forever.

## Regeneration: requirements and behaviour

```bash
pip install 'datamodel-code-generator[ruff]' 'pydantic>=2.6.0'
python schemas/scripts/generate_models.py     # run from the repo root
```

The script: guards that it is running from the repo root, checks the
generator tool is installed, compiles each schema to a Pydantic model,
re-applies the `AUTO-GENERATED … DO NOT EDIT` banner, and runs sanity
checks on the expected enums (e.g. `Kind`, `Encoding`, span/trace
`Status`) and root model names. It exits non-zero on any mismatch.

## See also

- [Generated models — why you can't edit them](../hilbertbench/models/README.md)
- [Architectural invariants](../docs/reference/invariants.md) — INV-003, INV-005, INV-008
- [Trace format reference](../docs/reference/trace-format.md)
