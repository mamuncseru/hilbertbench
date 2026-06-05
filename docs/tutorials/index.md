# Tutorials

Each tutorial takes a real quantum ML failure mode, adds HilbertBench with
one code change, and walks through the diagnostic output with the science
behind it.

---

| # | Tutorial | Concepts | Time |
|---|---|---|---|
| 01 | [Why Isn't My VQE Converging?](vqe-barren-plateau.md) | Barren plateau, cost-landscape variance, ansatz depth | 20 min |
| 02 | [Am I Using Enough Shots?](shot-noise.md) | Shot noise floor, SNR, shot budget | 15 min |

---

**Prerequisites for all tutorials**

```bash
pip install hilbertbench[qiskit,storage] scipy
```

Python 3.10+, Qiskit 1.0+.
