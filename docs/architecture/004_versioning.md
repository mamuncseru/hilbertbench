# Versioning

HilbertBench carries two version numbers, and they move independently
on purpose: the **software** version and the **trace-schema** version.
A trace recorded by one software version must stay readable by later
ones, so the schema cannot be hostage to the software's release pace.

## Software version (semantic versioning)

The package follows [semantic versioning](https://semver.org).
Given `MAJOR.MINOR.PATCH`:

- **PATCH** (`1.0.0 → 1.0.1`) — bug fixes, no change to the public API
  or trace format.
- **MINOR** (`1.0.0 → 1.1.0`) — backward-compatible additions: a new
  analyzer, a new integration, a new *optional* schema field shipped as
  `schemas/v1.1/`. Existing code and existing traces keep working.
- **MAJOR** (`1.0.0 → 2.0.0`) — a breaking change to the public API or
  the trace format.

The public API is the surface in `hilbertbench/__init__.py`, the
`HilbertTrace` reader, the six analyzer functions, the integration
proxies, and the `hb-verify` / `hb-export` commands. Anything prefixed
with an underscore is internal and may change in a minor release.

The version lives in one place — `pyproject.toml` — and the Git tag
must match it exactly (`v1.0.0` ↔ `version = "1.0.0"`). The
[release process](https://github.com/mamuncseru/hilbertbench/blob/main/RELEASING.md)
covers the mechanics, including the rule that a version published to
PyPI can never be replaced.

## Trace-schema version

Every trace records the schema version it was written against
(`hbtrace_version`, `hbspan_version`, and so on). This is what lets an
analyzer written years later know exactly how to read an old trace.

The governing rule is [INV-005](../reference/invariants.md): **once a schema
version is tagged in Git, no file in that directory ever changes.** A
released `schemas/v1.0/` is permanent. The trace format is a contract
with every trace ever written under it, and you cannot quietly change
what `v1.0` means.

When the format must change:

- A new *optional* field, added compatibly, goes into a new minor
  schema version (`schemas/v1.1/`) and degrades gracefully on older
  traces ([INV-008](../reference/invariants.md)).
- A change that would break older readers — removing a field, changing
  a type, making an optional field required — requires a new major
  schema version (`schemas/v2.0/`) and a corresponding major software
  release.

Because the Python models are generated from the schema
([INV-003](../reference/invariants.md)), bumping a schema version means adding a
new generated model package (`hilbertbench/models/v1_1/`) and exposing
it through the version-stable re-export surface, so existing importers
are never broken.

## How the two relate

| Change | Software bump | Schema bump |
|---|---|---|
| Fix a bug in an analyzer | patch | none |
| Add an analyzer or integration | minor | none |
| Add an optional trace field | minor | new minor schema (`v1.1`) |
| Break the public API | major | none |
| Break the trace format | major | new major schema (`v2.0`) |

The asymmetry is the point: the software can evolve quickly, but the
record of an experiment is meant to outlive the tool that wrote it.
