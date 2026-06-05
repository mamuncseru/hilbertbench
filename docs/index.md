# HilbertBench

**Non-intrusive diagnostic framework for quantum machine learning.**

HilbertBench records every circuit execution your code already performs —
without re-running anything — and gives you built-in analyzers to diagnose
the most common failure modes in quantum ML: barren plateaus, shot noise
dominance, stalled optimization, and more.

---

## The one-line change

You swap one object in your code. Everything else stays the same.

=== "Qiskit"

    ```python
    from hilbertbench.integrations.qiskit import HilbertEstimatorProxy
    from hilbertbench.recorder.tape import HilbertTape

    with HilbertTape("runs/my_experiment") as tape:
        estimator = HilbertEstimatorProxy(tape)   # ← the only change
        # your existing VQE / QAOA code here, unchanged
    ```

=== "PennyLane"

    ```python
    import pennylane as qml
    from hilbertbench.integrations.pennylane import HilbertPennyLaneDeviceProxy
    from hilbertbench.recorder.tape import HilbertTape

    with HilbertTape("runs/my_experiment") as tape:
        real_dev = qml.device("default.qubit", wires=2)
        dev = HilbertPennyLaneDeviceProxy(real_dev, tape)  # ← the only change
        # your existing QNode code here, unchanged
    ```

---

## Analyze the recorded trace

```python
from hilbertbench import HilbertTrace
from hilbertbench.analysis import detect_barren_plateau, shot_noise_ratio

trace = HilbertTrace(tape.dir_path)

print(detect_barren_plateau(trace))
# {'status': 'Trainable', 'variance': 0.215, ...}

print(shot_noise_ratio(trace, default_shots=1024))
# {'status': 'Signal Clear (not limited by shot noise)', 'estimated_snr': 8.3, ...}
```

---

## Install

```bash
pip install hilbertbench[qiskit,storage]     # Qiskit + Parquet
pip install hilbertbench[pennylane,storage]  # PennyLane + Parquet
pip install hilbertbench[full]               # everything
```

---

## Where to go next

- **[Getting Started](getting-started.md)** — install, run your first trace, read the output in 10 minutes
- **[Concepts](concepts.md)** — understand how passive recording works, what a span is, and why the trace is trustworthy
- **[Analyzers](analyzers/barren-plateau.md)** — what each diagnostic measures and how to read its output
- **[Guides](guides/qiskit-estimator.md)** — end-to-end walkthroughs for VQE, QAOA, IBM hardware, and PennyLane
