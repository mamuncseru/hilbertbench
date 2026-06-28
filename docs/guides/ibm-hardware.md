# Guide: IBM Quantum Hardware

This guide walks through `demo/03_qiskit_ibm.py` — running a 5-step VQE
on a real IBM Quantum backend with full HilbertBench recording, including
the hardware calibration snapshot.

Run it directly (IBM token required):

```bash
python demo/03_qiskit_ibm.py
```

---

## Prerequisites

```bash
pip install hilbertbench qiskit-ibm-runtime
```

You need an IBM Quantum account. Set your token in the demo file:

```python
IBM_TOKEN = "YOUR_IBM_QUANTUM_TOKEN_HERE"
```

---

## What changes on hardware

Two additional steps are required compared to the simulator demo:

1. **Transpile to ISA** — the circuit must be compiled to the backend's
   native gate set and topology
2. **Layout alignment** — the observable must be aligned to the
   transpiled circuit's qubit layout

```python
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

service  = QiskitRuntimeService(channel="ibm_quantum_platform", token=IBM_TOKEN)
backend  = service.least_busy(operational=True, simulator=False)

pm       = generate_preset_pass_manager(optimization_level=1, backend=backend)
isa_qc   = pm.run(circuit)
isa_obs  = abstract_obs.apply_layout(isa_qc.layout)   # ← required on hardware
```

The proxy usage is identical to the simulator:

```python
with HilbertTape("runs/ibm_vqe", tags={"backend": backend.name}) as tape:
    estimator = HilbertEstimatorProxy(tape, real_estimator=EstimatorV2(mode=backend))
```

---

## Calibration snapshot

On hardware, HilbertBench automatically records the device calibration
snapshot (T1, T2, readout error, gate error) at the time of the run.
After the trace is sealed:

```python
from hilbertbench.analysis import noise_profile

result = noise_profile(trace)
print(f"Noise verdict: {result['status']}")
print(f"T1 (mean):     {result['t1_us']['mean']:.1f} µs")
```

---

## Note on iteration count

The demo uses `N_ITER = 5` to minimize hardware queue time. For a
production experiment, increase to 50–100 iterations.

---

## Extending this example

- Compare noise profiles across different backends using the same circuit
- Record the calibration at multiple times during a long run to observe drift
- Pair with `circuit_structure` to understand which gates dominate the
  hardware fidelity estimate
