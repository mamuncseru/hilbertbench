# HilbertBench, End to End

This is the one document to read if you want to understand the whole
tool — what it is, why it is built the way it is, and how every piece
works. It starts at 10,000 feet and descends one level at a time.
Each level is self-contained: stop whenever you have enough depth.

No quantum computing background is assumed. Quantum-specific terms
are explained the first time they appear, and there is a
[glossary](#level-10-glossary-for-non-quantum-readers) at the end.

---

## Level 0 — What is this? (30 seconds)

**HilbertBench is a flight recorder for quantum machine learning
experiments.**

You add one line to your existing code. From then on, every circuit
your program sends to a quantum computer (or simulator) — and every
answer that comes back — is silently recorded into a tamper-evident
file set called a *trace*. Afterwards, you (or anyone you send the
trace to) can run built-in analyzers that diagnose **why** the
experiment behaved the way it did: was the model untrainable? Were
the measurements too noisy? Was the hardware having a bad day?

The experiment itself is never touched. That is the whole point.

```
your code ──► [HilbertBench proxy] ──► quantum computer
                     │
                     ▼ (silently)
              sealed trace on disk ──► analyzers ──► diagnosis
```

---

## Level 1 — The five-minute mental model

### What problem does it solve?

Quantum machine learning (QML) trains models the way classical ML
trains neural networks — an optimizer adjusts parameters to minimize
a loss. The difference: the "model" is a quantum circuit executed on
a quantum processor, and the loss is estimated from repeated
probabilistic measurements.

These experiments fail in ways that look identical from the outside
(the loss just stops improving) but have completely different causes:

| Failure mode | What is actually happening |
|---|---|
| **Barren plateau** | The optimization landscape is exponentially flat — there is no gradient signal to follow |
| **Shot starvation** | Too few measurement repetitions — the optimizer is chasing statistical noise |
| **Noise domination** | Hardware errors destroy the computation before it can be measured |
| **Nothing** | The run is healthy and just needs more iterations |

Today, telling these apart is folklore: rerun with tweaks until
something works. HilbertBench replaces that with evidence.

### Why "non-intrusive" is the core design constraint

Two reasons, one physical and one scientific:

1. **The observer effect is real here.** You cannot print
   intermediate quantum states — measuring a quantum system changes
   it. And inserting heavy diagnostics into a training loop changes
   its *timing*, which on real superconducting hardware changes the
   noise the experiment sees. A diagnostic tool that perturbs the
   experiment becomes a confounding variable.
2. **Paid hardware bills by the second.** A tool that silently
   re-executes circuits or adds measurements costs real money. Users
   must be able to trust that recording is free of side effects.

So HilbertBench enforces **1:1 execution parity**: it never
re-executes circuits, never adds measurements, never changes shot
counts, and adds negligible latency. It records exactly what was
going to happen anyway. This is invariant INV-001, and the test
suite enforces it.

### Why "evidence, not verdicts"

The recorder writes down **facts**: which circuit ran, with which
parameters, what the device answered, what the machine's calibration
was at that moment. It never writes conclusions into the trace.
Diagnoses are computed at *read time* by analyzers, from the
evidence. This separation (invariant INV-006) means:

- a better analyzer written next year can re-diagnose last year's
  traces;
- two people can disagree about interpretation while agreeing on the
  facts;
- a trace can be published, and strangers can do their own analysis —
  the "data-only research" model.

### The three-step workflow

```python
from hilbertbench.recorder.tape import HilbertTape
from hilbertbench.integrations.qiskit import HilbertEstimatorProxy
from hilbertbench.analysis import summary
from hilbertbench import HilbertTrace

# 1. RECORD: wrap your estimator — the only change to your code
with HilbertTape("runs/my_experiment") as tape:
    estimator = HilbertEstimatorProxy(tape)
    ...  # your training loop, completely unchanged

# 2. SEAL: leaving the `with` block seals the trace (SHA-256)

# 3. ANALYZE: anyone, anywhere, any time later
trace = HilbertTrace(tape.dir_path)
print(summary(trace))
```

---

## Level 2 — Quickstart: run it yourself in two minutes

```bash
git clone https://github.com/mamuncseru/hilbertbench
cd hilbertbench
pip install -e ".[dev]"           # base + PennyLane + test tools
python -m pytest tests/ -q        # 319 tests should pass
```

Then run this complete example:

```python
import numpy as np
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import SparsePauliOp

from hilbertbench.recorder.tape import HilbertTape
from hilbertbench.integrations.qiskit import HilbertEstimatorProxy
from hilbertbench.analysis import detect_barren_plateau

theta = Parameter("t")
qc = QuantumCircuit(2)
qc.ry(theta, 0)
qc.cx(0, 1)
obs = SparsePauliOp("IZ")          # measure <Z> on qubit 0

with HilbertTape("runs/demo") as tape:
    est = HilbertEstimatorProxy(tape)        # local ideal simulator
    for val in np.linspace(0, 2 * np.pi, 25):
        est.run([(qc, obs, [[val]])]).result()

print(detect_barren_plateau(tape.dir_path))
# {'status': 'Trainable', 'variance': 0.49..., ...}
```

What just happened: 25 circuit executions were recorded into
`runs/demo/<timestamp>/`, the trace was sealed, and the analyzer
measured the variance of the recorded outcomes — high variance means
a bumpy, trainable landscape.

---

## Level 3 — How recording works (the proxy and the tape)

### What

Two objects cooperate:

- **The proxy** (`HilbertEstimatorProxy`, `HilbertSamplerProxy`,
  `HilbertPennyLaneDeviceProxy`) impersonates the object your code
  already uses. It forwards every call to the real backend untouched
  and copies what it sees to the tape.
- **The tape** (`HilbertTape`) is the writer. It owns a run
  directory and appends records to it. It is append-only: nothing
  recorded is ever modified or deleted (invariant INV-002).

### Why intercept at the *execution boundary*?

Most ML logging tools ask you to instrument your training loop.
HilbertBench refuses, for coverage reasons: whatever framework or
optimizer you use, everything eventually funnels through one narrow
interface — the call that submits circuits for execution (in Qiskit,
the V2 primitive `run()` method). Intercepting there captures
*everything* with one line, and the user's code stays immutable.

### How: what one execution produces

When your code calls `proxy.run(pubs)`:

1. The proxy forwards the call to the real estimator and waits for
   the result — **first**, so timing is unchanged.
2. For each PUB (a "primitive unified bloc" — circuit + observable +
   parameter values), the tape records one **span**: a begin/end
   record with events. Attached to the span as **artifacts**:
   - the circuit, serialized to OpenQASM 3 text;
   - the parameter values;
   - the observable;
   - the outcome (expectation values, or measurement counts for the
     sampler);
   - the requested measurement budget (shots or target precision) —
     evidence the shot-noise analyzer needs later.
3. Rate-limited (at most once per 10 minutes), the proxy also
   snapshots the **device calibration** — the machine's measured
   error rates and coherence times — into the trace. On a drifting
   device, each *distinct* calibration becomes a new snapshot, so a
   trace carries a small history (`trace.calibration_history()`).
   This is what lets an analyst correlate a loss spike with the
   machine recalibrating mid-run, rather than blaming the model.

Failures in recording never break the user's run: a span that cannot
complete is closed with an `ERROR` event (invariant INV-007 — no
silent failures), and the real result is returned regardless.

---

## Level 4 — Anatomy of a trace directory

Every run directory is self-contained and portable — zip it, email
it, publish it:

```
runs/demo/20260611_120000_ab12cd34/
├── trace.json        # identity: version, mode, timestamps, tags,
│                     #   environment, integrity seal
├── events.jsonl      # the append-only event stream (one JSON/line):
│                     #   span begins, events, span ends
├── catalog.json      # index of every artifact: kind, encoding,
│                     #   size, created_at, content hash
└── artifacts/        # content-addressed files, sharded by hash
    ├── 39/39be4a...qasm    # circuits (OpenQASM 3)
    ├── 7f/7fc01a...json    # calibration snapshot
    └── ...                 # outcomes, parameters, observables
```

Three design decisions worth understanding:

- **Content addressing.** Artifacts are stored under the SHA-256 of
  their bytes. A circuit reused 10,000 times is stored once. The
  hash doubles as an integrity check on every read.
- **The seal.** When the tape closes, the SHA-256 checksum of the
  whole event stream is written into `trace.json`.
  `trace.verify()` re-derives everything and fails loudly on any
  modification. This is what makes the blinded validation protocol
  (Level 7) possible: nobody can quietly edit evidence.
- **Append-only JSONL.** A crash mid-run loses at most the final
  line. Everything before it is still valid evidence — flight
  recorders must survive the crash.

Reading is symmetric to writing: `HilbertTrace(run_dir)` gives you
`.spans` (a pandas DataFrame), `.completed()` (span views with
resolved `.outcome`, `.parameters`, `.circuit`),
`.numeric_outcomes()` (a flat numpy array of every recorded value),
`.calibration_history()`, and `.verify()`.

---

## Level 5 — The six analyzers (what / why / how)

All analyzers are plain functions: trace in, dictionary out. They
never modify the trace. Each returns a `status` string plus the
quantitative evidence behind it, with bootstrap confidence intervals
where a statistic is estimated. Full per-analyzer pages are under
*Analyzers* in the navigation; this is the map.

### `detect_barren_plateau` — is there anything to learn from?

- **What:** variance of all recorded outcome values, with a 95%
  bootstrap CI, classified against a threshold (default 0.005).
- **Why:** the most famous QML failure mode (McClean et al., 2018)
  is an exponentially flat loss landscape. Flatness = low variance
  across the visited points. Below the threshold, an optimizer is
  walking on a billiard table.
- **How:** `np.var(trace.numeric_outcomes())` plus a bootstrap. The
  verdict reports its own confidence: if the CI straddles the
  threshold, confidence is "low" — transparency over false certainty.

### `shot_noise_ratio` — signal, or static?

- **What:** compares the empirical variance of the outcome
  trajectory to the *shot-noise floor* — the variance you would see
  from measurement statistics alone.
- **Why:** a quantum measurement is a coin-flip sample; estimating an
  expectation from N shots has variance ~1/N no matter what. If the
  trajectory's variance is comparable to that floor, the optimizer is
  fitting noise.
- **How:** the floor is 1/shots when shot counts were recorded, or
  precision² when the run requested a target precision (the proxy
  records both). SNR < 1.5 → "Shot Noise Dominated".

### `optimization_convergence` — is the trajectory still moving?

- **What:** classifies the recorded parameter/outcome trajectory:
  Converged / Converging / Still Improving / not moving.
- **Why:** distinguishes "stuck" from "slow" — the cheapest question
  to answer before blaming anything quantum.
- **How:** step sizes and loss deltas across the recorded sequence
  of spans.

### `circuit_structure` — what actually ran?

- **What:** qubit count, depth, gate composition, entangling-gate
  fraction, parameter count, measurement count, and *which physical
  qubits were used* — parsed from the recorded OpenQASM.
- **Why:** every other diagnosis needs the structural facts, and on
  real hardware the circuit that ran (post-transpilation) is not the
  circuit you wrote.
- **How:** a line-level QASM parser that handles both virtual
  (`q[0]`) and hardware-native physical (`$0`) qubit syntax, with
  ASAP layering for depth.

### `kl_expressibility` — how much of the space can the model reach?

- **What:** Kullback–Leibler divergence between the model's
  state-fidelity distribution and the Haar (uniform-random) ideal.
  Small KL = the circuit covers state space like a truly random one.
- **Why:** expressibility is the standard capacity measure for
  quantum models (Sim et al., 2019) — and it trades off provably
  against trainability (Holmes et al., 2022). Claiming it from a
  training run is a methodological error: training visits a biased
  path. Measuring it honestly needs *random* sampling — which is why
  this analyzer only accepts Active Mode traces (Level 6).
- **How:** pairwise state fidelities, histogrammed, compared to the
  exact Haar reference, with a bootstrap CI.

### `noise_profile` — how hostile was the machine?

- **What:** summarizes the calibration snapshot (coherence times,
  readout and gate error rates) **scoped to the qubits the circuit
  actually used**, and predicts the circuit's survival probability
  as a product of per-operation success rates.
- **Why:** on NISQ hardware, noise is part of the algorithm. The
  prediction gives a falsifiable number: on a real 156-qubit device
  we measured prediction within ~8% of observed signal attenuation.
- **How:** fidelity ≈ (1−e₁q)^n1q · (1−e₂q)^n2q · (1−e_ro)^n_meas,
  with error rates averaged over the active qubits only.
  (Device-wide averaging — including a big chip's dead edges — was
  found to mispredict by ~600×; the scoped version is the fix.)

### `summary` — all of the above

One call, one dictionary, every analyzer. This is what a blinded
diagnostician uses.

---

## Level 6 — Passive vs Active Mode

**Passive mode** (the default) records what your experiment was
going to do anyway. It can answer trajectory questions: convergence,
shot noise, plateau evidence *along the visited path*.

**Active mode** is for structural questions, which require
executions your experiment would *not* have performed — for example
expressibility, which needs uniformly random parameter samples. The
rules:

- Active diagnostics are explicitly opt-in and run in isolation —
  never silently mixed into your training run.
- The trace records its mode (`trace.json: "mode": "active"`), so an
  analyst always knows whether they are looking at observational or
  interventional data.
- `hilbertbench.active.probe_expressibility(state_fn, ...)` is the
  built-in active probe: it samples random parameters, records the
  resulting states into an active-mode trace, and
  `kl_expressibility` consumes it.

The distinction is scientific, not cosmetic: observational data can
show *correlation along a path*; interventional data supports claims
about the model *as a whole*. Conflating them is how "highly
expressive" claims end up unfounded.

---

## Level 7 — The blinded-validation toolkit

How do you prove a diagnostic instrument can be trusted? The same
way medicine does: blinded, with the answer key sealed in advance.
`tools/blind_corpus.py` implements the machinery:

```bash
# Researcher A generates a corpus with planted failure modes
python experiments/study_e_corpus.py --test
# Blind it: verbatim copies under random IDs + leakage audit
python tools/blind_corpus.py blind --manifest .../manifest.json --out blinded/
#   -> publishes answer_key.sha256 (the commitment) BEFORE diagnosis
# Researcher B fills diagnosis_sheet.json using only analyzer output
python tools/blind_corpus.py score --key ... --commitment ... --diagnosis ...
#   -> confusion matrix, per-class precision/recall, 95% Wilson CI
```

Why each safeguard exists:

| Safeguard | Attack it prevents |
|---|---|
| Traces sealed at creation | editing evidence after the fact |
| SHA-256 commitment of the key, published first | changing answers after seeing diagnoses |
| Leakage audit on tags & names | accidentally shipping the label inside the trace |
| Thresholds frozen on a disjoint dev corpus | tuning the detector to pass its own exam |
| Shuffled generation order | timestamps leaking the label |

---

## Level 8 — The experiments harness

`experiments/` contains one seeded, reproducible script per study in
the validation paper (see `experiments/README.md` for the full map):

| Script | Question |
|---|---|
| `study_a_bp_scaling.py` | does measured variance decay exponentially with qubit count, as theory predicts? |
| `study_b_shot_noise.py` | at what measurement budget does the SNR cross usability? |
| `study_c_expressibility.py` | where does each depth sit on the expressibility–trainability frontier? |
| `study_d_fidelity.py` | does the predicted fidelity match observed attenuation? |
| `study_e_corpus.py` | generates the blinded corpus (dev and test tiers) |
| `hw_smoke_test.py` | end-to-end validation on real IBM hardware (+ first fidelity points) |
| `hw_frontier.py` | hardware frontier measurement; `--pilot` = free tier, `--full` = paid batch |

Every script accepts `--quick` (or `--pilot`) for a smoke-test-sized
run. All randomness is seeded; every result JSON records its seeds
and trace directories.

---

## Level 9 — Running on real IBM hardware

What you need: an IBM Quantum account and its API token.

```python
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2
service = QiskitRuntimeService(channel="ibm_cloud", token=YOUR_TOKEN)
backend = service.least_busy(simulator=False, operational=True)

est = EstimatorV2(mode=backend)
proxy = HilbertEstimatorProxy(tape, real_estimator=est)   # that's it
```

Practical notes, learned the expensive-feeling way:

- **Transpile first.** Real devices require ISA circuits (their
  native gates, physical qubits). Use
  `generate_preset_pass_manager(backend=...)` and
  `observable.apply_layout(isa.layout)`. The recorded QASM will use
  physical-qubit syntax — the analyzers handle it.
- **Batch your parameters.** One PUB can carry hundreds of parameter
  sets. This is the difference between a 30-second job and a
  30-minute one at ~$96/minute on pay-as-you-go.
- **Never hardcode the token.** Keep it in a private file outside
  the repository.
- **Mind mitigation defaults.** If you are studying raw device
  behavior, set `resilience_level = 0` and disable twirling —
  otherwise the platform "helpfully" corrects the noise you are
  trying to measure.
- Calibration snapshots are captured automatically; on long runs the
  proxy refreshes them (rate-limited) so drift is visible in
  `trace.calibration_history()`.

---

## Level 10 — Quality assurance

The suite has **319 tests**, organized by subsystem, and every single
one is described in the [Test Catalog](reference/test-catalog.md) —
what it checks and why it exists. The shape of the suite mirrors the
guarantees:

| Area | What it protects |
|---|---|
| `tests/recorder/` | append-only writing, sealing, artifact store |
| `tests/trace/` | the read API resolves exactly what was written |
| `tests/integrations/` | proxies are transparent (parity!) and record faithfully — incl. calibration capture across all backend conventions and hardware ISA circuits |
| `tests/analysis/` | each analyzer's verdicts on constructed ground-truth traces |
| `tests/active/` | the expressibility probe and active-mode labeling |
| `tests/compliance/` | the architecture invariants (INV-001 …) hold end to end |
| `tests/reader/` | verification: tampered traces must fail loudly |
| `tests/e2e/` | full record→seal→analyze journeys |
| `tests/tools/` | the blinding protocol: leakage audit, commitments, scoring |

The invariants themselves are documented in
[Reference → Invariants](reference/invariants.md).

## Glossary for non-quantum readers {#level-10-glossary-for-non-quantum-readers}

| Term | Meaning |
|---|---|
| **Qubit** | quantum bit; n qubits span a 2ⁿ-dimensional state space |
| **Circuit** | the program: a sequence of gates applied to qubits |
| **Ansatz** | a parameterized circuit template — the "model architecture" of QML |
| **Shot** | one execution-and-measurement of a circuit; outcomes are statistical, so estimates average many shots |
| **Expectation value** | the average of a measured observable — QML's loss is built from these |
| **Transpilation** | compiling an abstract circuit to a device's native gates and physical qubit layout (ISA form) |
| **Barren plateau** | exponentially flat loss landscape; gradients vanish |
| **Expressibility** | how uniformly a parameterized circuit covers state space (KL vs the Haar ideal) |
| **T1 / T2** | how long a qubit retains its state (energy / phase coherence times) |
| **Calibration** | the device's measured error rates and coherence times, refreshed periodically by the provider |
| **Fidelity** | overlap between intended and actual state; 1.0 = perfect |
| **NISQ** | today's noisy, intermediate-scale quantum hardware era |
| **PUB** | primitive unified bloc — Qiskit's (circuit, observable, parameters, precision) work unit |

---

*Where to next:* the four [tutorials](tutorials/index.md) each walk
one failure mode end to end; the [Analyzer pages](analyzers/barren-plateau.md)
go deeper on each diagnostic; the [Trace Format](reference/trace-format.md)
specifies the schema; the [Test Catalog](reference/test-catalog.md)
explains all 319 tests.
