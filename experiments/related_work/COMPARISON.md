# Capability matrix: post-hoc failure attribution of a finished QML run

The fixed task (see README): a variational run has finished and
failed; attribute the failure to a mechanism from what the tool
preserves, without re-running and without having modified the run.

Rows are capabilities the task needs; the demos in this directory are
the receipts for every "no".

| Capability | HilbertBench | Mitiq | Qiskit Experiments | PennyLane (specs/Tracker) | MLflow-style tracker |
|---|---|---|---|---|---|
| Operates on a **finished** run (post hoc) | **yes** — input is the sealed trace | no — input is a circuit + executor, pre-run | no — runs its own circuits | no — in-process only | partly — metrics persist, but see below |
| **Non-intrusive** (algorithm and computation unmodified) | **yes** — transparent proxy at the execution boundary | no — executes noise-scaled circuit variants | no — dedicated characterization circuits | yes (inspection) but in-process | yes |
| **Zero extra quantum cost** | **yes** — records what already runs | no — multiplies executions (see demo: ~5-10x for ZNE) | no — every characterization circuit is extra QPU | yes | yes |
| Evidence **survives the process** | **yes** — append-only trace on disk | no | partly — analysis results, not run evidence | no | yes (metrics/params only) |
| **Quantum-native evidence** (shot budgets, transpiled/ISA structure, calibration snapshot at execution time) | **yes** | no | measures device properties itself; does not capture them per-run of *your* workload | no | no — classical scalars/artifacts |
| **Mechanism-level failure verdicts** (barren / shot / noise / healthy) | **yes** — six analyzers | no | no | no | no |
| **Tamper-evident** record (sealed, hash-verifiable) | **yes** | no | no | no | no |
| What it is actually for (and good at) | evidence + diagnosis | error **mitigation** | device **characterization** | building/differentiating circuits; live resource inspection | experiment bookkeeping |

## The two sentences a referee needs

1. Each neighbouring tool lacks, **by design and not by defect**, at
   least one of: a post-hoc input object, non-intrusiveness, preserved
   quantum evidence, or mechanism-level attribution — the four things
   the diagnosis task is made of.
2. They compose rather than compete: a run recorded by HilbertBench
   can *contain* a Mitiq-mitigated estimator, run on a device
   characterized by Qiskit Experiments, inside a PennyLane or Qiskit
   program, with MLflow logging the classical metadata — and the
   trace still supports the diagnosis none of them can make.

## Version provenance

- Main environment (frozen, instrument v1.0.0): qiskit 1.3.2,
  pennylane 0.44.1.
- Isolated `.venv` (so comparison installs can never perturb the
  frozen experiment environment): mitiq 1.0.0,
  qiskit-experiments 0.14.1, qiskit 2.5.0, qiskit-aer 0.17.2.

## Citations used in the paper

- Mitiq: LaRose et al., Quantum 6, 774 (2022).
- Qiskit Experiments: Kanazawa et al., JOSS 8(84), 5329 (2023).
- PennyLane: Bergholm et al., arXiv:1811.04968.
- MLflow: Zaharia et al., IEEE Data Eng. Bull. 41(4), 39-45 (2018).
