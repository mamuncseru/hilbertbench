# Deep follow-up to the depth-decay measurement (exploratory)

Written 2026-10-22, after the three registered jobs of `hw_decay_plan.md`
were analysed, and committed before this job is submitted. It is
**exploratory**: it was proposed after seeing those results, because on
chain A the exponential beat the straight line by only 2 AIC units over
depths 1 to 48. It does not change or replace the registered analysis.

## Question

At depths where the two fits disagree strongly, does the damping keep
decaying exponentially, or follow the straight line? The pooled
straight-line fits of the registered data reach zero near 103 layers
(chain A) and 71 layers (chain B); the exponential fits stay positive.

## Measurement

One job on ibm_marrakesh, same two chains (qubits 0-3 and 122, 123, 136,
143), same circuit family, transpiler settings and checks as
`hw_decay.py`. Depths 64, 96, 128. Per depth 40 random rows and 4
reference rows (seed 20261032 for random rows, 20261033 for the
references), 512 shots per row, no mitigation. About 135,000 shots,
about 40 s by the registered jobs' rate. The same budget rule applies
(total including the registered jobs at most 570 s).

## Analysis (`hw_decay_deep.py analyze`)

1. Damping f at each new depth and chain (slope on the noiseless cost,
   paired bootstrap), reference damping, correctly priced ESP.
2. The predictions of the registered pooled exponential and straight-line
   fits at 64, 96, 128, and the difference of each from the measured f
   in units of its standard error.
3. The three models refitted to all depths (registered jobs plus this
   job), with AIC, reported next to the registered result, which stays
   the primary result.
