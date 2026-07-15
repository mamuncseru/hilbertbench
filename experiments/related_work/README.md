# Related-work comparative analysis

Referee-facing evidence for the paper's related-work positioning
(Introduction, competing-tools paragraph). The question every demo
answers is the same one, held fixed:

> **A variational QML run has finished and failed. From what the tool
> preserves, attribute the failure to a mechanism — after the fact,
> without re-running anything and without having modified the run.**

Each demo is honest about what the neighbouring tool IS for and shows,
concretely, what it does and does not give you for this task. The
tools are good at their own jobs; none of them has this job.

## Layout

| File | Tool | Env | What it shows |
|---|---|---|---|
| `demo_hilbertbench.py` | HilbertBench v1.0.0 | main env | the baseline: post-hoc mechanism attribution from a sealed trace of a finished run |
| `demo_pennylane.py` | PennyLane | main env | `qml.specs` / `qml.Tracker` are in-process resource inspection; nothing survives the process, no device evidence |
| `demo_mitiq.py` | Mitiq | `.venv` | error *mitigation*: improves an expectation value by changing the computation (more circuits, noise scaling); no post-hoc diagnosis API |
| `demo_qiskit_experiments.py` | Qiskit Experiments | `.venv` | device *characterization*: dedicated extra circuits measure processor properties, not why your run failed |
| `COMPARISON.md` | all + MLflow | — | the capability matrix with the receipts |

MLflow is analysed in `COMPARISON.md` only: its gap (classical
tracker; no quantum evidence — no shots semantics, no transpiled
structure, no calibration snapshot) is architectural and does not
need a runtime demonstration.

## Running

```bash
# main environment (frozen instrument env, untouched)
python demo_hilbertbench.py
python demo_pennylane.py

# isolated environment (so Mitiq/Qiskit-Experiments deps can never
# perturb the frozen experiment environment)
.venv/bin/python demo_mitiq.py
.venv/bin/python demo_qiskit_experiments.py
```

Each demo prints a `WHAT THIS SHOWS` block at the end; the demos are
deterministic (seeded) where the underlying tool allows it.

## One-paragraph summary (as argued in the paper)

Mitiq *changes* the computation to suppress noise rather than diagnose
why an unmodified run failed. Qiskit Experiments measures device
properties by running *dedicated* circuits — an intrusive protocol
that spends extra device time and characterizes the processor, not the
training run that failed on it. PennyLane's inspection utilities are
in-process: their outputs live and die with the Python process, and
record no device evidence. MLflow-style trackers persist
hyperparameters and metrics but are blind to quantum evidence.
HilbertBench is the complement: record the evidence a diagnosis needs,
at the execution boundary the run already crosses, without altering
the run — and keep the record verifiable after the fact.
