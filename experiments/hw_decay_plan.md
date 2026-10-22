# Depth decay of the noise damping on hardware: analysis plan

Written 2026-10-22, before any data. The plan was emailed to Dr. Picone
the same day (he acknowledged it). This file and `hw_decay.py` are
committed before the first job is submitted. Nothing below changes after
data exist.

## Question

On one IBM Heron device, with the qubits fixed, how does the damping of
the cost depend on circuit depth? Does it fall exponentially (the
noise-induced barren plateau picture), and does the calibration predict
the per-layer rate?

## Measurement

- Device: ibm_marrakesh, or ibm_fez only if marrakesh is not operational
  when the session starts. All jobs use the same device.
- Qubits, fixed by hand:
  - chain A: physical qubits 0, 1, 2, 3, as in the earlier jobs;
  - chain B: the chain of four connected qubits, disjoint from chain A,
    with the lowest sum of three two-qubit gate errors, four readout
    errors and four sx errors in the calibration at session start
    (`choose_chain` in `hw_decay.py`). It is chosen once and kept.
- Circuit: the 4-qubit circuit of the paper (RY and RZ on each qubit,
  then a CNOT ladder, per layer); cost ZZ on circuit qubits 2 and 3.
  Depths 1, 4, 8, 12, 16, 24, 32, 48.
- Per depth, bound to the same transpiled circuit: 40 random parameter
  rows (uniform in [0, 2 pi), seed 20261022) and 4 reference rows
  (angles k pi / 2, kept only when the noiseless cost is exactly +1
  or -1). The same rows in every job and on both chains.
- 512 shots per row; no error mitigation, no twirling.
- Three identical jobs, submitted about an hour apart.

## Changes from the emailed plan (made before any data)

- The account has 10 minutes of QPU time, not the 10 to 12 assumed in
  the email. Shots per row are therefore 512 (as in the earlier jobs),
  not 1,000. A job is about 360,000 shots, about 110 s by the earlier
  rate, about 5.5 min for three.
- Budget rule: after submission and before execution, IBM's usage
  estimate is read; the job is cancelled while queued if it exceeds
  180 s or would bring the total above 570 s. If a repeat is cancelled
  for budget, the session stops and the completed repeats are reported.
- A job that fails for a technical reason is resubmitted once and both
  are reported.

## Analysis (implemented in `hw_decay.py analyze`)

For each job, chain and depth:

1. Damping f: slope of the hardware cost on the noiseless cost over the
   40 random rows; paired bootstrap (2,000 resamples) for the standard
   error and 95% interval.
2. Reference damping: measured over noiseless value, averaged over the
   4 reference rows.
3. Variance: (hardware variance - mean shot variance) / noiseless
   variance, reported next to f^2.
4. Correctly priced ESP from the calibration in effect when the job ran
   (the snapshot taken after completion if it was published before
   execution started, else the one taken at submission); the v1.0.0
   value is reported alongside for the record.

Per chain:

5. Fit f against depth with three models, weighted by the bootstrap
   standard errors: exponential f = A q^L, straight line, constant.
   Primary: the three jobs pooled (24 points). Secondary: each job
   alone. All fits are reported; the model with the lowest AIC
   (chi^2 + 2k) is called best. The interval on q is the fit
   covariance, inflated by the reduced chi^2 when that exceeds 1.
6. Calibration-predicted per-layer rate q_pred: slope of log ESP
   against depth, per job; compared with the fitted q.
7. Spread between jobs: range and standard deviation of f per depth.

## What will be reported

Every job, chain and depth, whatever the result.

- If the exponential has the lowest AIC for a chain, the paper states
  that on that chain noise reduced the cost by a factor q per layer, so
  its variance shrank as q^(2L), as predicted for noise-induced barren
  plateaus (Wang et al. 2021), with q, its interval, q_pred, and the
  spread between jobs.
- If not, the paper reports the measured damping as it is and makes no
  barren plateau claim for that chain.
