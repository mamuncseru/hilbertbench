# HilbertBench

> Diagnose quantum ML experiments without changing a line of algorithm code.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://github.com/mamuncseru/hilbertbench/actions/workflows/tests.yml/badge.svg)](https://github.com/mamuncseru/hilbertbench/actions/workflows/tests.yml)
[![Docs](https://img.shields.io/badge/docs-readthedocs-blueviolet)](https://hilbertbench.readthedocs.io)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21009244.svg)](https://doi.org/10.5281/zenodo.21009244)

HilbertBench is a **non-intrusive diagnostic framework** for quantum machine
learning. You wrap one object — an Estimator, Sampler, or PennyLane device —
and every circuit execution is silently recorded into an append-only,
cryptographically sealed trace. Nothing else in your code changes.

After the run you call built-in analyzers to diagnose trainability, shot
noise, optimization convergence, circuit structure, expressibility, and
hardware noise — all from the evidence already in the trace.

---

## Install

```bash
# Default: runs the Qiskit workflow out of the box
# (trace core + Qiskit integration + Parquet storage)
pip install hilbertbench

# Add the PennyLane integration
pip install hilbertbench[pennylane]

# Everything (Qiskit + PennyLane + storage)
pip install hilbertbench[full]
```

The default install covers the documented Qiskit path. PennyLane is the
one optional integration; if you use it without installing it, the
error tells you exactly which extra to add.

---

## The one-line change

```python
from hilbertbench.integrations.qiskit import HilbertEstimatorProxy
from hilbertbench.recorder.tape import HilbertTape
from hilbertbench.analysis import detect_barren_plateau
from hilbertbench import HilbertTrace

# Wrap the estimator — everything else stays the same
with HilbertTape("runs/my_vqe", tags={"algorithm": "vqe"}) as tape:
    estimator = HilbertEstimatorProxy(tape)   # ← the only change
    # ... your existing optimizer loop, completely unchanged ...

# Analyze the sealed trace
trace = HilbertTrace(tape.dir_path)
print(detect_barren_plateau(trace))
# {'status': 'Trainable', 'variance': 0.215, 'variance_ci': [0.098, 0.371], ...}
```

---

## Built-in analyzers

| Function | What it answers |
|---|---|
| `detect_barren_plateau` | Is the cost-landscape variance exponentially suppressed? |
| `shot_noise_ratio` | Is the optimizer chasing signal or shot noise? |
| `optimization_convergence` | Is the parameter trajectory still moving? |
| `circuit_structure` | How deep, how many qubits, what gate composition? |
| `kl_expressibility` | How uniformly does the ansatz cover Hilbert space? |
| `noise_profile` | What hardware noise levels did the run experience? |

```python
from hilbertbench.analysis import summary

report = summary(trace)   # runs all analyzers, returns one dict
```

---

## Integrations

| Framework | Proxy class | Install extra |
|---|---|---|
| Qiskit V2 Estimator | `HilbertEstimatorProxy` | `hilbertbench[qiskit]` |
| Qiskit V2 Sampler | `HilbertSamplerProxy` | `hilbertbench[qiskit]` |
| IBM Quantum hardware | `HilbertEstimatorProxy` with `EstimatorV2(mode=backend)` | `hilbertbench[qiskit]` |
| PennyLane | `HilbertPennyLaneDeviceProxy` | `hilbertbench[pennylane]` |

---

## Tutorials

Four end-to-end tutorials, each diagnosing a real quantum ML failure mode:

| # | Tutorial | Concepts |
|---|---|---|
| 01 | [Why Isn't My VQE Converging?](https://mamuncseru.github.io/hilbertbench/tutorials/vqe-barren-plateau/) | Barren plateau, cost-landscape variance |
| 02 | [Am I Using Enough Shots?](https://mamuncseru.github.io/hilbertbench/tutorials/shot-noise/) | Shot noise floor, SNR, shot budget |
| 03 | [Expressibility vs Trainability](https://mamuncseru.github.io/hilbertbench/tutorials/expressibility-vs-trainability/) | Active Mode, KL expressibility, Holmes 2022 |
| 04 | [How Hardware Noise Degrades Your Results](https://mamuncseru.github.io/hilbertbench/tutorials/hardware-noise/) | Circuit fidelity, gate error, noise profile |

---

## Documentation

| Resource | Contents |
|---|---|
| **[End-to-End Guide](docs/end-to-end.md)** | The whole tool in one document — top-down, layered, no quantum background assumed |
| **[mamuncseru.github.io/hilbertbench](https://mamuncseru.github.io/hilbertbench/)** | Learning site: getting started, tutorials, concept guides |
| **[hilbertbench.readthedocs.io](https://hilbertbench.readthedocs.io)** | Reference site: analyzer internals, trace format, architecture, and the [test catalog](docs/reference/test-catalog.md) — all 319 tests explained |

The learning site builds from `mkdocs.yml`, the reference site from
`mkdocs-rtd.yml`. After adding tests, refresh the catalog with
`python tools/gen_test_catalog.py`.

---

## Design guarantees

- **Non-intrusive** — the proxy never re-executes circuits, modifies shot counts, or alters your algorithm (INV-001)
- **Immutable traces** — every trace is append-only and sealed with a SHA-256 hash on close (INV-002)
- **Evidence, not verdicts** — raw execution data is stored; diagnostic conclusions are computed at read time and never written back (INV-006)
- **No silent failures** — every initiated span ends with a `COMPLETED` or `ERROR` event (INV-007)

Those `INV-NNN` codes appear throughout the source comments and docstrings.
Each is a numbered, non-negotiable guarantee; the **complete, canonical list
(INV-001 … INV-008)** lives in
**[docs/reference/invariants.md](docs/reference/invariants.md)** (rendered on
the [reference site](https://hilbertbench.readthedocs.io)). When you see an
`INV-NNN` tag in the code, that is where it is defined.

---

## Acknowledgments

HilbertBench was developed by **Md. Abdullah Al Mamun** in the
**[Neural Engineering Data Consortium (NEDC)](https://nedcdata.org/)**,
lab at **Temple University**, under the advisorship of
**Prof. Joseph Picone** (Department of Electrical and Computer
Engineering). The author is grateful to Dr. Picone for his guidance
and for the research environment in which this work was carried out.

## Citing HilbertBench

If you use HilbertBench in your research, please cite it using the
metadata in [`CITATION.cff`](CITATION.cff). On the GitHub project page,
that file makes a **"Cite this repository"** button appear automatically
in the right-hand sidebar.

Each release is archived on Zenodo with a DOI. Cite the **version** you
used (v1.0.0 → [10.5281/zenodo.21009245](https://doi.org/10.5281/zenodo.21009245));
the **concept DOI** [10.5281/zenodo.21009244](https://doi.org/10.5281/zenodo.21009244)
always resolves to the latest version.

## License

MIT. See [LICENSE](LICENSE).
