# Guide: PennyLane QNN

This guide walks through `demo/04_pennylane.py` — a 2-qubit
StronglyEntanglingLayers QNN trained on the two-moons dataset with
HilbertBench passive recording.

Run it directly:

```bash
pip install hilbertbench[pennylane,storage] scikit-learn
python demo/04_pennylane.py
```

---

## The device proxy

```python
import pennylane as qml
from hilbertbench.integrations.pennylane import HilbertPennyLaneDeviceProxy
from hilbertbench.recorder.tape import HilbertTape

with HilbertTape("runs/pennylane_qnn", tags={"dataset": "two_moons"}) as tape:
    real_dev = qml.device("default.qubit", wires=2)
    dev = HilbertPennyLaneDeviceProxy(real_dev, tape)   # ← the only change

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(x, weights):
        qml.AngleEmbedding(x, wires=range(2), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(2))
        return qml.expval(qml.PauliZ(0))
```

`HilbertPennyLaneDeviceProxy` wraps any PennyLane device. Every execution
routed through the device — including parameter-shift gradient probes — is
recorded as a separate span.

---

## Tags must use string values

All tag values must be strings. The schema enforces `dict[str, str]`.

```python
# correct
tags={"n_qubits": str(N_QUBITS), "dataset": "two_moons"}

# will raise a validation error
tags={"n_qubits": N_QUBITS}
```

---

## Span count in batched training

For batched Adam training with parameter-shift gradients, the span count
will be large: each forward pass and each shift probe is a separate span.
A single epoch with 300 samples, batch size 16, 2 parameters, and 2 layers
produces O(1000) spans per epoch.

---

## Post-training analysis

```python
from hilbertbench.analysis import detect_barren_plateau, optimization_convergence

trace = HilbertTrace(tape.dir_path)
bp   = detect_barren_plateau(trace)
conv = optimization_convergence(trace)

print(f"Trainability: {bp['status']}")
print(f"Convergence:  {conv['status']}")
```

!!! note "Optimization convergence in batched QNNs"
    The `movement_ratio` verdict is not meaningful for batched QNN training
    (see [Optimization Convergence](../analyzers/optimization.md#scope-note-batched-qml-runs)).
    Use `outcome_trend` to assess whether the loss decreased.

---

## Extending this example

- Add `circuit_structure` to confirm the StronglyEntanglingLayers depth
- Run `active_probe_pennylane` after training to measure expressibility
- Compare barren plateau variance across different numbers of layers
