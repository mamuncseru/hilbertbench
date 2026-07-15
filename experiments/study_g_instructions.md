# Study G — diagnostician instructions (Researcher B)

You are receiving a second blinded corpus. The procedure is the same
as Study E, with the differences noted below. Please read these
instructions fully before opening any run.

## What you receive

- A directory of blinded runs (random identifiers, sealed traces).
- `diagnosis_sheet.json` — one entry per run.

## Your task, per run

Using only the instrument's outputs on the trace (`summary()` and the
individual analyzers), fill in:

- `primary`: one of `healthy`, `barren_plateau`, `shot_starved`,
  `noise_dominated`.
- `secondary`: optional second label, when you see evidence of a
  second mechanism (as in Study E).
- `confidence`: your confidence in the primary label, in [0, 1].
- `notes`: brief free text — which evidence drove your call.

## How this corpus differs from Study E

- **It is deliberately harder.** Planted mechanisms range from
  strongly expressed to physically absent. Some runs may express
  **two** mechanisms at once; some may sit near the edge of what the
  evidence can support.
- **`healthy` is a real answer.** Choose it whenever the evidence
  does not support any failure mechanism. A run is not guilty until
  the trace proves it.
- **Class balance is not disclosed** and is not uniform. Do not
  assume equal counts per class.
- **Read all the evidence.** The analyzers can disagree with each
  other; verdict-status lines alone may not tell the whole story.
  Recorded shot counts, convergence behaviour, calibration snapshots,
  and confidence gradings (e.g. "Marginal") are all part of the
  trace.

## Procedure (same as Study E)

1. Single pass; you may take as long as you need, but do not revise
   entries after moving to the commitment step.
2. When done, compute the SHA-256 of your filled sheet and send the
   hash BEFORE sending the sheet itself, exactly as in Study E.
3. The answer key, the corpus composition, and a pre-committed
   mechanical comparator's outputs are unsealed only after your hash
   is received.

Thank you — your diagnosis is the primary outcome of a pre-registered
study, and it is scored exactly as registered, whatever the result.
