# Study G — unsealing log

Written record of the Study G unsealing. The Study G protocol
(`study_g_protocol.json`, committed 2026-07-15 in `b35ee2b`) requires
one ordering guarantee for unsealing:

> the diagnosis is a single pass; the sheet is hash-committed by the
> diagnostician before the answer key, the assignment, and the rule
> outputs are unsealed

That guarantee was met. Unlike Study E, the Study G protocol does
**not** require a recorded (video) unsealing session, and none was
held; this written log is the record. This is stated plainly so that
no reader infers a recorded session exists.

## Participants

- **Researcher A** (corpus custodian, answer-key holder): Md. Abdullah
  Al Mamun. Performed the unsealing.
- **Researcher B** (diagnostician): Sadia Afrin Purba. Not present at
  unsealing; her involvement ended when she submitted her sheet hash
  and file.
- **Supervisor**: Joseph Picone. OSF registration contributor.

## Timeline

| When | Event |
|---|---|
| 2026-07-15 | Protocol, corpus generator, and scorer committed (`b35ee2b`) — before any run existed |
| 2026-07-15 | Corpus generated (63 runs) from the private, hash-committed assignment |
| 2026-07-15 | Corpus blinded; answer key moved out of the corpus directory; leakage audit clean (0/63 leaky) |
| 2026-07-15 | Frozen rule applied to the blinded corpus; outputs hash-committed (`215d97d`) |
| 2026-07-16 | Diagnostician package built (`study_g_researcher_b.zip`), audited leak-free |
| before delivery | **OSF registration filed, embargoed**, with the supervisor as contributor. Diagnostician deliberately *not* added as a contributor, so she could not view the registration (which contains the composition and grid). Confirmed by Researcher A. |
| after registration | Package delivered to the diagnostician |
| 2026-07-19 13:15 | Diagnostician's filled sheet written (`diagnosis_sheet_v02.json`) |
| 2026-07-21 | Diagnostician submitted her sheet **and its SHA-256** to Researcher A |
| 2026-07-21 21:17 | **UNSEALING**: sheet hash verified, then answer key and rule outputs verified against their pre-published commitments, then scored |

## Verified commitment chain

Every hash below was published *before* the artifact it protects could
have been influenced. All four verified at unsealing.

| Artifact | SHA-256 | Committed | Verified |
|---|---|---|---|
| Private run-level assignment | `540dfefccdf1d33312a7ab3711c2486040ace46c86c2ef312fb1b1e03f35b08a` | `b35ee2b`, before generation | yes |
| Answer key | `e7dbf167da9a7dc33219e7721a697c28b2abefa6a76a85c5bf3725417baa0139` | `215d97d`, before diagnosis | yes |
| Frozen-rule outputs on the blinded corpus | `6da0c442e2ca143916921f162a6a3e01f90e0fca46964bbc22dd9e5d70a12c49` | `215d97d`, before the diagnostician received the corpus | yes |
| Diagnostician's filled sheet | `f089413296e6207811cde1f0f2e6936119dfc8673f4561e76b728bdb1184ec65` | submitted by the diagnostician with her sheet, verified before any unsealing | yes |

Independently re-verifiable with `sha256sum` on the archived files.

## Diagnostician's stated procedure

Submitted with her sheet, in her own words (recorded here because it
materially explains the primary outcome):

1. Trainability "Barren Plateau Detected" with high confidence →
   primary `barren_plateau`; if estimated fidelity < 0.5, secondary
   `noise_dominated`.
2. If trainable: low SNR and fidelity >= 0.5 → `shot_starved`; low SNR
   and fidelity < 0.5 → primary `noise_dominated`, secondary
   `shot_starved`.
3. Trainable, clear signal, moderate-to-high fidelity → `healthy`.
4. Three runs with low barren-plateau confidence → lower reported
   confidence regardless of label.

## Result at unsealing

- **H1 (primary): not supported.** Human 40/63 (0.635, 95% CI
  [0.511, 0.743]) vs frozen rule 38/63 (0.603, [0.480, 0.715]).
  Discordant pairs b = 2, c = 0. The pre-registered underpower
  contingency applies (discordant count < 6), so the discordance is
  reported descriptively and no p-value is headlined. The two
  discordant runs both fall in the predicted rule-failure classes: a
  depth-1 converged-trajectory trap, and a strongly shot-starved run
  the rule misread as barren.
- **H2: descriptive.** Converged-trap subset (n = 3): human 3/3, rule
  2/3. Marginal-SNR subset (n = 1): not testable.
- **H3: supported.** Human confidence in transition regions
  (mean 0.728, n = 24) below strong anchors (mean 0.904, n = 9);
  one-sided Mann-Whitney U = 42.0, p = 0.0037, surviving Holm
  correction across the secondary tests.
- **Boundaries reproduced on fresh seeds.** Barren detection steps
  0/3 → 3/3 between n = 7 and n = 8; noise detection steps between
  depth 12 and 14 — matching the Study F boundaries on an
  independently generated corpus.
- **False positives on mechanism-free runs:** human 0/9, rule 1/9.
- Human and rule assigned an identical primary label on 57/63 runs
  (90.5%), which is why the paired comparison had little to measure.

## Disclosures

1. **The committed scorer under-implemented a pre-registered test.**
   `study_g_score.py` as committed in `b35ee2b` recorded the H3 group
   means but did not compute the Mann-Whitney U test named in the OSF
   registration. The test itself was specified in the registration
   before any data existed; only the implementation was incomplete.
   The statistic was computed per the registered plan and the scorer
   subsequently patched as a disclosed addendum (see the commit
   following this log). No analysis choice was made after seeing data.
2. **No recorded unsealing session.** Not required by the Study G
   protocol; this log is the record.
3. **Absolute accuracies understate both graders.** At the weakest
   sweep levels the planted mechanism is physically negligible, yet
   ground truth still names it, so a correct reading of "healthy" is
   scored incorrect. The registration anticipated this ("the
   knob-defined label and the trace evidence genuinely diverge"),
   which is why the primary outcome is paired — both graders face the
   identical ambiguity on the identical runs.
4. **OSF DOI pending.** The registration is embargoed; OSF issues the
   DOI when the embargo is lifted. To be recorded here at that time.
