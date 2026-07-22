# Study G — what we actually learned

Working summary for the paper. Registered outcomes are labelled as
such; anything computed after unsealing is labelled EXPLORATORY.

## 1. The corpus was genuinely hard — the headline deliverable

| | human | frozen rule |
|---|---|---|
| all 63 runs (registered) | 40/63 = 0.635 [0.511, 0.743] | 38/63 = 0.603 [0.480, 0.715] |
| excluding the shot arm (EXPLORATORY) | 36/48 = 0.750 | 35/48 = 0.729 |

Either way, far from the 72/72 ceiling of round 1. The "your exam was
too easy" objection is now answered with data rather than argument.

## 2. Two of three detection boundaries are sharp and reproducible

Pre-registered manipulation check (does the evidence track the knob?):

- **Barren plateau — PASSED.** Median Var[C] falls monotonically
  0.0319 → 0.0085 → 0.0042 → 0.0020 → 0.0005 across n = 5, 7, 8, 9, 11,
  with tight replicate spread. Detection steps 0/3 → 3/3 between n = 7
  and n = 8, exactly where the variance crosses the frozen 0.005
  threshold. **Reproduces the Study F boundary on independently drawn
  seeds.**
- **Noise domination — PASSED.** Median fidelity falls monotonically
  0.663 → 0.521 → 0.481 → 0.444 → 0.379 across depths 6, 12, 14, 16, 20.
  Detection steps 0/3 → 3/3 between depth 12 and 14, exactly where
  fidelity crosses the frozen 0.50 threshold. Study F bracketed this
  at 12–16; Study G localises it to 12–14.

These two boundaries are the study's real scientific product: sharp,
physically interpretable, and replicated across two independent corpora.

## 3. The shot-starvation arm failed its manipulation check

**Pre-registered check, FAILED.** Median SNR by target precision:
15.50 → 9.15 → 9.22 → 1.34 → 0.00, i.e. non-monotonic, and with
enormous within-level spread ([0.95, 46.14] at a single level).

Cause: SNR = empirical variance / shot floor, and along an optimizer
trajectory the empirical variance depends on the path taken, not only
on the shot budget. The precision knob therefore does **not** reliably
control the achieved SNR.

Consequence: at three of five levels (0.05, 0.12, 0.18) most runs
labelled `shot_starved` were reported by the instrument itself as
"Signal Clear." All 8 of the diagnostician's `healthy` calls on this
arm were on runs the instrument graded Signal Clear (7) or Marginal
(1) — she read the evidence correctly and was scored wrong against a
label the run did not exhibit. **The shot arm's contribution to
accuracy is not interpretable as grader skill**, and its detection
curve should be reported with the failed check stated.

## 4. New confusion mechanism found: extreme shot starvation mimics a
   barren plateau (EXPLORATORY)

At target precision 0.45 (~5 shots), the recorded expectation values
collapse toward a constant, so Var[C] falls below the 0.005 plateau
threshold and the trainability analyzer reports "Barren Plateau
Detected" (SNR ≈ 0.000 at the same time). The diagnostician called
2 of 3 such runs `barren_plateau` with confidence 1.0 — she was
reading the tool correctly; the tool was pointing the wrong way.

This is actionable: the instrument already holds the disambiguating
evidence (SNR ≈ 0 alongside collapsed variance implies starvation, not
an intrinsic plateau). A combined reading, or a guard in the
trainability analyzer, would remove this failure mode.

## 5. Registered outcomes

- **H1 (PRIMARY) — NOT SUPPORTED.** Discordant pairs b = 2, c = 0
  (both favouring the human, both in predicted rule-failure classes:
  a depth-1 converged trap, and a strongly starved run the rule read
  as barren). Discordance of 2 is below the pre-registered underpower
  threshold of 6, so reported descriptively with no headline p-value.
- **H2a (marginal-SNR subset)** — not testable, n = 1.
- **H2b (converged-trajectory traps)** — human 3/3, rule 2/3.
  Descriptive (n = 3); direction as predicted.
- **H3 (confidence calibration) — SUPPORTED.** Confidence in
  transition regions 0.728 (n = 24) vs strong anchors 0.904 (n = 9);
  one-sided Mann-Whitney U = 42.0, p = 0.0037, surviving Holm
  correction. The cleanest positive result in the study: she located
  the hard runs from the evidence alone, without being told.
- **Q4 (detection curves)** — see §2 and §3.
- **Q5 (mixtures)** — human and rule identical on all 9: barren+shot
  → both `shot_starved` (barren masked, as predicted in Study F);
  barren+noise → both `barren_plateau`; noise+shot → both
  `noise_dominated`.
- **Q6 (false positives)** — human 0/9, rule 1/9 on mechanism-free
  runs, including 3/3 on the converged-trajectory traps.

## 6. Why the primary outcome had nothing to measure

The human and the rule assigned an identical primary label on 57/63
runs (90.5%). Both read the same analyzer verdicts, so with a single
rater the discordance was structurally near zero. A secondary
asymmetry: the rule emits a flag *set* (up to three mechanisms) while
the diagnostician gives at most two labels (primary + secondary), so
on multi-flag runs the rule can "detect" a mechanism she had no slot
left to name.

## 7. Design lesson, to be stated openly

The study's stated purpose was to locate the boundary where the
instrument's readings stop being conclusive — an *estimation* goal.
The registration nonetheless made a human-versus-rule comparison the
primary outcome and demoted boundary mapping to a descriptive
question. Given that both graders read the same verdicts, low
discordance was foreseeable (we pre-specified the underpower
contingency, which is itself evidence it was foreseeable), so the
competition was a poor choice of primary.

We report H1 as registered and null. We do **not** retrospectively
promote the boundary mapping to primary — that would be exactly the
outcome-switching the commitment apparatus exists to prevent. The
mismatch is reported as a design lesson in the Discussion.

## 8. Confusion matrix (planted truth vs the diagnostician's primary)

```
                        called it →
planted                 healthy  barren  shot  noise
────────────────────────────────────────────────────
healthy (9)                9       —      —      —
barren plateau (15)        6       9      —      —
shot starvation (15)       8       3      4      —
noise domination (15)      —       —      6      9
barren + noise (3)         —       3      —      —
barren + shot (3)          —       —      3      —
noise + shot (3)           —       —      —      3
```

The `healthy` column holds nearly all the misses, and they concentrate
in the weak levels of each sweep — where the planted mechanism was
deliberately dialled toward absent (barren n = 5, 7) or, in the shot
arm, never took at all (§3).

## 9. Provenance

Pre-registration OSF [10.17605/OSF.IO/UYN46](https://doi.org/10.17605/OSF.IO/UYN46)
(embargoed, filed before corpus delivery). Dataset Zenodo
[10.5281/zenodo.21482518](https://doi.org/10.5281/zenodo.21482518).
Instrument frozen at HilbertBench v1.0.0. Unsealing record and the
four verified commitments: `study_g_unsealing_log.md`.
