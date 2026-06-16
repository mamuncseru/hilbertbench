# Trace Format

A HilbertBench trace is a directory produced by `HilbertTape`. This page
documents the files inside it so you can inspect, share, or build tooling
on top of traces without using the Python API.

---

## Directory layout

```
runs/my_experiment/
└── 20260605_143022_a1b2c3d4/
    ├── events.jsonl        ← append-only event log
    ├── manifest.json       ← run metadata and integrity seal
    ├── catalog.json        ← content-addressed artifact index
    └── store/              ← file store for large artifacts
        ├── ab12ef34.qasm
        └── cd56gh78.npy
```

The run directory name is `YYYYMMDD_HHMMSS_<8-hex-uuid>`, created when
the `HilbertTape` context opens.

---

## `events.jsonl`

One JSON object per line, appended in real time. Never modified after
writing. Three event types appear:

### `TRACE_START`

Written once when the tape opens.

```json
{"event": "TRACE_START", "trace_id": "a1b2c3d4-...", "ts": "2026-06-05T14:30:22Z",
 "mode": "passive", "tags": {"algorithm": "vqe"}}
```

### `SPAN_END`

Written once per completed circuit execution.

```json
{
  "event": "SPAN_END",
  "span_id": "00000001",
  "seq": 1,
  "ts_start": "2026-06-05T14:30:22.104Z",
  "ts_end":   "2026-06-05T14:30:22.381Z",
  "status": "COMPLETED",
  "backend_id": "statevector_simulator",
  "outcome_ref": "inline:...",
  "payload_ref": "store:ab12ef34.qasm",
  "inline_artifacts": [
    {"kind": "parameters",  "encoding": "json", "data": "[1.23, 0.45]"},
    {"kind": "execution_outcome", "encoding": "json", "data": "-0.981"}
  ],
  "events": [
    {"name": "EXECUTION_STARTED",  "ts": "2026-06-05T14:30:22.104Z", "attrs": {}},
    {"name": "EXECUTION_COMPLETED","ts": "2026-06-05T14:30:22.381Z",
     "attrs": {"shots": 1024}}
  ]
}
```

### `TRACE_END`

Written once when the tape is sealed.

```json
{"event": "TRACE_END", "ts": "2026-06-05T14:30:25Z",
 "status": "SEALED_SUCCESS", "sha256": "e3b0c442..."}
```

The `sha256` field is the SHA-256 hash of the full `events.jsonl` file up
to (but not including) the `TRACE_END` line. This is the integrity seal
checked by `trace.verify()`.

---

## `manifest.json`

Run-level metadata written when the tape seals.

```json
{
  "trace_id":    "a1b2c3d4-...",
  "schema_version": "1.0",
  "mode":        "passive",
  "status":      "SEALED_SUCCESS",
  "tags":        {"algorithm": "vqe"},
  "ts_start":    "2026-06-05T14:30:22Z",
  "ts_end":      "2026-06-05T14:30:25Z",
  "client_environment": {
    "hilbertbench_version": "1.0.0",
    "python_version": "3.11.4"
  },
  "integrity_seal": {
    "algorithm": "sha256",
    "value":     "e3b0c442..."
  }
}
```

---

## `catalog.json`

Content-addressed index of all artifacts stored in `store/`. The same
circuit structure (identical QASM bytes) is stored exactly once, regardless
of how many times it is executed. Repeated executions reference the same
`store/` entry via a content hash.

```json
{
  "ab12ef34": {
    "kind":     "circuit_qasm",
    "encoding": "openqasm",
    "filename": "ab12ef34.qasm",
    "producer": "HilbertEstimatorProxy",
    "sha256":   "ab12ef34..."
  }
}
```

---

## `store/`

Large artifacts (circuit QASM strings, statevectors, calibration snapshots)
that exceed the inline threshold (`65 536` bytes) are written here as files.
Small artifacts are embedded directly in `events.jsonl` as inline JSON.

The filenames are the first 8 hex digits of the SHA-256 hash of the
artifact contents.

---

## Accessing the trace programmatically

You do not need to parse these files directly. `HilbertTrace` handles all
formats, inline/file-store resolution, and hash verification:

```python
from hilbertbench import HilbertTrace

trace = HilbertTrace("runs/my_experiment/20260605_143022_a1b2c3d4")

trace.verify()          # verify the integrity seal
trace.status            # "SEALED_SUCCESS"
trace.tags              # dict from manifest
len(trace)              # span count

for span in trace.completed():
    span.circuit        # QASM string (resolved from store/ or inline)
    span.outcome        # float / dict / numpy array
    span.parameters     # numpy array
```
