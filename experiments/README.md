# Experiments — Reproducibility Package

Every result in the HilbertBench validation study is produced by a
script in this directory. Each one is self-contained, fully seeded, and
follows the same shape: build circuits, record their executions into
sealed HilbertBench traces, analyze those traces *blind* (the analyzer
sees only the trace, never the generating code), and write a single
JSON result file. Re-running a script reproduces its result exactly.

These are **synthetic ground-truth** experiments, not dataset-driven
machine learning. The "data" is controlled: random parameter samples
drawn from a fixed distribution, or short optimization trajectories,
over parameterized circuits with a *planted* property — a known barren
plateau, a known shot starvation, a known noise level. Validation means
checking that the instrument recovers the property we planted, using
the trace alone.

## The studies at a glance

| Study | Script | Question it answers |
|-------|--------|---------------------|
| A | `study_a_bp_scaling.py` | Does the recorded cost-landscape variance decay exponentially with qubit count, recovering the McClean (2018) barren-plateau scaling — from trace evidence alone? |
| B | `study_b_shot_noise.py` | At what measurement budget does the diagnostic signal-to-noise ratio cross from "buried in shot noise" to "usable"? |
| C | `study_c_expressibility.py` | Where does each circuit depth sit on the expressibility–trainability frontier (Holmes 2022)? |
| D | `study_d_fidelity.py` | Does the fidelity predicted from a trace's calibration snapshot match the attenuation actually observed? |
| E | `study_e_corpus.py` | (Corpus generator) Produces the labelled runs for the blinded-diagnosis study. |
| HW smoke | `hw_smoke_test.py` | Does the whole pipeline work on real IBM hardware, and do the analyzers read a real-device trace correctly? |
| HW frontier | `hw_frontier.py` | The hardware arms of C and D: variance-vs-depth and KL expressibility on a real device. |

## Running

```bash
cd experiments
python study_a_bp_scaling.py --quick    # reduced smoke run (~1 min)
python study_a_bp_scaling.py            # full study
python study_e_corpus.py --dev          # 12-run threshold-tuning corpus
python study_e_corpus.py --test         # 36-run frozen corpus for blinding
```

Every simulator study takes `--quick` for a fast, smaller version; the
hardware scripts take `--pilot` (free-tier sized) versus `--full`
(paid batch, additionally gated behind `--yes`). Outputs go to
`results/<name>/results.json`; sealed traces to `traces/<name>/`. Both
output trees are git-ignored — they are regenerated from the scripts
and archived separately (e.g. on Zenodo).

## Shared infrastructure — `common.py`

All studies build on three helpers, so the circuit construction and
recording path are identical across experiments:

- **`build_ansatz(n_qubits, n_layers, entanglement)`** — a
  hardware-efficient ansatz (RY+RZ on every qubit per layer, followed
  by a CNOT map). `entanglement` selects the `linear`, `ring`, or
  `full` topology, the three families swept in Study A.
- **`pair_observable(n_qubits)`** — the cost observable, a ZZ on the
  first qubit pair. A local two-qubit observable is used throughout so
  results sit in a consistent cost regime.
- **`sample_landscape(...)`** — opens an active-mode tape, wraps the
  estimator in `HilbertEstimatorProxy`, and records the cost at a batch
  of uniformly random parameter points. This is the controlled random
  sampling that characterizes a landscape's variance.

---

## Study A — Barren-plateau scaling

- **Purpose.** Show the instrument independently recovers the McClean
  (2018) exponential variance decay, and flags the plateau exactly
  where it is planted.
- **Data.** 500 uniform random parameter points per landscape, swept
  over qubit widths 2–12, three depth rules (1 layer, n/2, n), and the
  three entanglement families — 54 landscapes in all.
- **Pipeline.** `sample_landscape` records each landscape →
  `detect_barren_plateau` reads variance + a bootstrap CI from the
  sealed trace → a least-squares fit of `log2(variance)` against qubit
  count gives the empirical scaling exponent per family.
- **Output.** Per-landscape variance, CI, verdict, and the fitted
  slopes in `results/study_a/results.json`.

## Study B — Shot-noise budget

- **Purpose.** Locate the measurement budget below which a diagnostic
  cannot distinguish signal from shot noise.
- **Data.** A fixed 4-qubit, 2-layer circuit evaluated at 50 random
  parameter points for each shot budget in {32, 64, …, 4096}, on a
  shot-based Aer simulator (target precision set to `1/√shots`).
- **Pipeline.** `sample_landscape` (with a real shot-based estimator) →
  `shot_noise_ratio`, which compares the trajectory variance against
  the shot-noise floor recorded in the trace.
- **Output.** SNR, noise floor, and verdict per budget in
  `results/study_b/results.json`.

## Study C — Expressibility vs trainability

- **Purpose.** Trace the Holmes (2022) frontier: as depth increases, a
  circuit becomes more expressive but harder to train.
- **Data.** A 4-qubit ansatz at depths 1–8. Two measurements per depth:
  2000 random parameter samples for expressibility (Active Mode), and
  500 for the cost-landscape variance.
- **Pipeline.** `probe_expressibility` records the Active-Mode trace →
  `kl_expressibility` (KL divergence against the Haar distribution);
  separately `sample_landscape` → `detect_barren_plateau`. The
  resulting `(KL, variance)` pair per depth is the frontier point.
- **Output.** KL, variance, and both verdicts per depth in
  `results/study_c/results.json`.

## Study D — Fidelity cross-validation

- **Purpose.** Test whether the product-formula fidelity estimate,
  computed from a trace's calibration snapshot, predicts the signal
  attenuation the device actually produces.
- **Data.** A 4-qubit ansatz at depths {1, 3, 5, 7, 10, 14}, several
  random parameter points each, run on a calibrated noisy backend
  (`FakeManilaV2` for the simulator arm).
- **Pipeline.** The circuit is ISA-transpiled, executed through the
  proxy, and the noisy expectation compared to the exact statevector
  value; `noise_profile` predicts fidelity from the recorded
  calibration alone. Predicted-vs-observed is the result.
- **Output.** Predicted fidelity, observed attenuation, and dominant
  error source per run in `results/study_d/results.json`.

## Study E — Blinded corpus generator

- **Purpose.** Build the labelled corpus for the blinded-diagnosis
  study. Four planted classes: `healthy`, `barren_plateau`,
  `shot_starved`, `noise_dominated`.
- **Data.** `--dev` makes 3 runs per class (seeds 1000+) for fixing
  analyzer thresholds; `--test` makes 9 per class (seeds 2000+) for the
  actual blind study. The classes differ only in width, depth, and
  measurement budget — the recording path is identical, so traces are
  indistinguishable except through their evidence.
- **Pipeline.** Each class is generated by the matching recipe (a deep
  random landscape for the plateau, a precision-starved VQE for shot
  starvation, a transpiled VQE on a noisy backend for noise
  domination, a clean shallow VQE for healthy). Generation order is
  shuffled so trace timestamps leak no label; tags carry only a neutral
  `corpus_id`.
- **Output.** The run directories plus `manifest.json` (path → label),
  which feeds `tools/blind_corpus.py`.

## Hardware scripts

`hw_smoke_test.py` and `hw_frontier.py` connect to IBM Quantum and run
the hardware arms. They read the API token from a private file outside
the repository (`~/.qiskit/hb_ibm_token`); no credentials live in the
code. They pin one device, broadcast many parameter sets per job, and
run with error mitigation disabled so the recorded trace reflects raw
device behaviour. `hw_smoke_test.py smoke` doubles as an end-to-end
verification: after the run it asserts that calibration was captured,
the ISA circuit parsed, shot evidence was recorded, and the seal
verifies.

## The blinded-diagnosis protocol

Study E produces the corpus; the diagnosis itself runs through
`tools/blind_corpus.py`:

```bash
python study_e_corpus.py --test
python ../tools/blind_corpus.py blind \
    --manifest traces/corpus_test/manifest.json --out blinded/
# publish blinded/answer_key.sha256 BEFORE anyone diagnoses;
# the diagnostician fills diagnosis_sheet.json from summary() output only
python ../tools/blind_corpus.py score \
    --key answer_key.json --commitment answer_key.sha256 \
    --diagnosis diagnosis_filled.json --out results/study_e/scores.json
```

The dev/test split matters: thresholds are fixed on the dev corpus and
frozen, and the test corpus is generated only afterwards, so the
detector cannot have been tuned against its own exam.

## Reproducibility

Every random draw is seeded from a fixed module constant, and the seed
is written into both the result record and the trace tags. Nothing here
draws unseeded randomness, so two runs of the same script on the same
machine produce identical traces and results. Hardware runs are the one
exception — device noise is not reproducible — which is why the
calibration snapshot is recorded alongside every hardware trace.
