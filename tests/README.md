# `tests/` — Test Suite

HilbertBench records scientific evidence, so the test suite is
organized around the **guarantees** it must uphold, not just around
code units. There are **319 tests**; every one is described
individually in the
[Test Catalog](../docs/reference/test-catalog.md).

## Running

```bash
python -m pytest tests/ -q                 # everything (~25s)
python -m pytest tests/analysis -q         # one subsystem
python -m pytest tests/compliance -q       # the invariant checks
python -m pytest tests/ -q -k calibration  # by keyword
```

No `conftest.py` is required — each test file sets up its own
fixtures (typically a `tmp_path` run directory). Tests use a light,
conventional style (not the library's heavy docstring format).

## Layout by guarantee

| Directory | Tests | What it protects |
|-----------|------:|------------------|
| `recorder/` | 78 | The write path: append-only tape, atomic sealing, the content-addressed artifact store (INV-002). |
| `compliance/` | 65 | The architectural invariants themselves (INV-001 …) hold end to end — the contract the docs and paper promise. |
| `integrations/` | 62 | Proxies are transparent to the wrapped framework (execution parity, INV-001) and record faithfully — including calibration capture across backend conventions and hardware ISA circuits. |
| `analysis/` | 49 | Each analyzer's verdict on constructed ground-truth traces, with its quantitative evidence and confidence intervals. |
| `trace/` | 25 | The read API resolves exactly what was written — spans, outcomes, parameters, circuits, calibration history. |
| `active/` | 12 | Active Mode probing, active-trace labeling, and that analyzers needing interventional data refuse passive traces. |
| `tools/` | 10 | The blinded-corpus protocol: leakage audit, SHA-256 commitments, confusion-matrix scoring. |
| `e2e/` | 10 | Full journeys: record → seal → reopen → analyze. |
| `reader/` | 8 | Verification — `trace.verify()` passes on honest traces and fails loudly on any tampering. |

## Adding tests

1. Put the test in the directory matching the guarantee it protects.
2. Name it `test_*.py`, classes `Test*`, functions `test_*`.
3. Give each test a docstring or a leading comment saying **what** it
   verifies — the test catalog is generated from exactly that text.
4. Regenerate the catalog so the docs stay in sync:
   ```bash
   python tools/gen_test_catalog.py
   ```

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the full PR checklist.

## See also

- [Test Catalog](../docs/reference/test-catalog.md) — every test explained
- [Architectural invariants](../docs/reference/invariants.md) — what `compliance/` enforces
