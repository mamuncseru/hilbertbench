# Tutorials

Each tutorial takes a real quantum ML failure mode or design question,
adds HilbertBench with one code change, and walks through the diagnostic
output with the science behind it.

---

| # | Tutorial | Concepts | Time |
|---|---|---|---|
| 01 | [Why Isn't My VQE Converging?](vqe-barren-plateau.md) | Barren plateau, cost-landscape variance, ansatz depth | 20 min |
| 02 | [Am I Using Enough Shots?](shot-noise.md) | Shot noise floor, SNR, shot budget | 15 min |
| 03 | [Expressibility vs Trainability](expressibility-vs-trainability.md) | Active Mode, KL expressibility, tradeoff analysis | 25 min |
| 04 | [How Hardware Noise Degrades Your Results](hardware-noise.md) | Circuit fidelity, gate error, noise profile | 20 min |

---

**Prerequisites for all tutorials**

```bash
pip install hilbertbench scipy
```

Tutorial 04 additionally requires:

```bash
pip install qiskit-aer
```

Python 3.10+, Qiskit 1.0+.
