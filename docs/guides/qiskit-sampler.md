# Guide: Qiskit Sampler (QAOA)

This guide walks through `demo/02_qiskit_sampler.py` — a parameter sweep
over 10 (γ, β) configurations for a 2-qubit QAOA circuit targeting MaxCut
on the edge (0, 1).

Run it directly:

```bash
python demo/02_qiskit_sampler.py
```

---

## The Sampler proxy

```python
from hilbertbench.integrations.qiskit import HilbertSamplerProxy
from hilbertbench.recorder.tape import HilbertTape

with HilbertTape("runs/sampler_qaoa", tags={"algorithm": "qaoa"}) as tape:
    sampler = HilbertSamplerProxy(tape)
    # sampler.run(...) is identical to StatevectorSampler.run(...)
```

`HilbertSamplerProxy` records each `sampler.run(...)` call as a span. The
span outcome stores the bitstring count distribution.

---

## Reading bitstring counts

Bitstring counts are stored as a JSON dict in the span outcome:

```python
trace = HilbertTrace(tape.dir_path)
for span in trace.completed():
    counts = span.outcome       # e.g. {'00': 512, '11': 512}
    print(counts)
```

---

## Extending this example

- Run a real MaxCut optimization by looping over configurations and
  selecting the best (γ, β)
- Add more QAOA layers and compare the bitstring distributions
- Use `circuit_structure` to verify the circuit depth matches your QAOA
  layer count
