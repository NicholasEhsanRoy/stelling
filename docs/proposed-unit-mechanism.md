<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# Can the unit norm be mechanised? — MEASURED, and the answer is mostly NO

**Both instruments this page names live in the campaign repo `stelling-sweeps`
and are written as `stelling-sweeps/<name>`**, so that no reader looks for them
here — the convention [state-0.1.0.md](state-0.1.0.md) states and
[gauge-coverage.md](gauge-coverage.md) now follows. Neither is in this tree and
nothing here re-derives what they measured. *As-of, in `stelling-sweeps` at
`b694d52`.*

*"A figure in a norm states the unit it counts"* has now failed on its own author
**three times**, in the same shape each time. This asks whether a mechanism can
replace the remembering, and the honest answer is that **no proposed mechanism
catches all three, and the one that comes closest is not a mechanism at all.**

## The three instances, and what they actually share

| | published | true | how the number was produced |
|---|---|---|---|
| 1 | *"105 float64 declarations, 9 float32"* | all sites 105/9; **literal sites 99/6** | another agent's report, plus my own reading |
| 2 | *"**24 of 41** `safe_mask` sites in scope"* | **21 of 41** | a hand count by eye over a grep listing |
| 3 | *"silently exempting **9 of the 30** dtypes"* | **16 of 30** | a throwaway script enumerating **23** dtypes |

The structure first proposed for these was *"a count derived over a
hand-assembled list, published against a population the list didn't cover."*
**That fits 1 and 3 and not 2** — case 2's denominator (41) was correct and
machine-derived; the *numerator* was hand-classified.

The structure that fits all three is narrower and less convenient:

> **Every one of the three errors happened at the transcription boundary — the
> step from measurement to prose — not inside the measurement.**

Case 1 welded two correct counts from adjacent populations. Case 2 typed a
classification the machine had not made. Case 3 had a script that said 23 and I
wrote 30. **No mechanism inside the measurement can catch an error committed
after it.** That single observation disposes of most of the candidates.

## The candidates, measured against the instances they must catch

### A doc lint on bare counts — **INFEASIBLE**

Scanned 74 committed `.md` files across both repositories for
`N of M` / `N/M` / `N%` / `A → B`:

```
N of M    89      N/M   357      percent  183      A -> B   44
TOTAL   673 hits
```

**673 candidates, and the sample is overwhelmingly not counts at all** —
`div int32 -7 / 2` in a results table, `1/3` as a literal in prose about
doubles, `uint 0/255` as a dtype range, `width="100%"` in the README's logo tag.
A lint at that signal ratio trains people to silence it.

### A lint restricted to BOLD counts — **feasible in volume, and it MISSES THE FOUNDING INSTANCE**

The campaign's own convention bolds load-bearing figures, so bolding was worth
measuring as a signal. It narrows well — **673 → 145**, roughly 5×, and the
matches really are claims (`**7 of 14 contracts VERIFIED**`, `**21 of 41 in
scope**`, `**35 of 39 transfers gauged**`, `**3 of 9**`).

Then the decisive test — would it have flagged the three **as published**?

| instance | strict pattern | loosened to allow *"N of the M"* |
|---|---|---|
| 1 — node coverage / declaration histogram | **miss** | **miss** |
| 2 — `24 of 41` | FLAG | FLAG |
| 3 — `9 of the 30` | miss | FLAG |

**1 of 3 strictly, 2 of 3 loosened.** It misses case 1 for a reason no
loosening fixes: **that error was not fraction-shaped.** *"105 float64
declarations, 9 float32"* contains no ratio to detect — it is two labelled
counts side by side, and the falsehood is in the labels. A mechanism that misses
the instance the norm was written for is not the mechanism.

### A `(numerator, denominator, population)` value that cannot be printed incomplete — **addresses the wrong step**

Reachable only where the number is produced by code I wrote: **2 of 3** (case 2
was a hand count, so nothing to route through). And for the two it does reach,
it protects **the script's output**, not the prose. Case 3 is the refutation:
the script would have returned `(9, 23, "dtypes I enumerated")` and I would
still have typed *"9 of 30"* into a docstring, a commit message, `SOUNDNESS.md`
and a test. **The error was in re-typing, which is downstream of every helper.**

### "Cite the instrument that produced it" — **catches 3 of 3, and is a convention, not a mechanism**

| instance | why citing would have caught it |
|---|---|
| 1 | `stelling-sweeps/bx_obligation_crosscheck.md` already existed; citing it surfaces both inflations and the unit |
| 2 | no script existed — being required to produce one is what yielded 21 |
| 3 | the script enumerated 23, and citing it makes the 23 visible next to the claim |

This is the only candidate that catches all three, and there is a **demonstrated
success in tree**: `stelling-sweeps/safe_mask_scope.py` was written precisely because a number
needed an instrument, and writing it changed 24 to 21. But enforcing it
mechanically needs a lint that can tell a claim-bearing count from an incidental
number, which the 673-hit measurement says is infeasible in general. **So it
reduces to "the author must remember", which is the thing that failed.**

## The recursion, stated plainly

Every candidate above ends in the same place: *the author must remember to do
something at the moment of writing prose*. That is the norm. **A mechanism that
requires remembering is the norm with extra steps**, and after three failures
the evidence is that remembering is exactly what does not happen — the errors
cluster where the author is most confident, having just measured.

## What actually has a record: the audit gate

**All three were caught by a blinded context re-deriving the number.** Not by a
lint, not by a helper, and not by the author re-reading.

| instance | caught by |
|---|---|
| 1 | a blinded audit re-deriving the corpus histogram |
| 2 | an AST classifier written *because* the number was going to be published |
| 3 | a blinded audit re-deriving over all 30 dtypes — and it had **already told me 16**, which I overrode with my smaller enumeration |

**Measured record: the gate 3 of 3; the best proposed lint 1 of 3.** So the
recommendation is not a fourth mechanism.

## Recommendation

1. **Do not build the lint.** Measured 1 of 3 at a signal ratio that would be
   silenced, and it cannot see a non-fraction error at all.
2. **Do not build the tuple helper.** It guards the step that did not fail.
3. **Route published counts through the audit gate that already exists** —
   `CONTRIBUTING.md`'s blinded-audit gate. Add one line to it: *a count that
   will be published is part of the change under audit, and the auditor
   re-derives it.* That costs nothing new, and it is the only intervention with
   a 3-of-3 record.
4. **Keep the convention where it has worked**: when a number is going to be
   published, write the instrument first and cite it.
   `stelling-sweeps/safe_mask_scope.py` is
   the pattern. This is a convention and is stated as one, not dressed as
   enforcement.
5. **And record the answer to the question asked**, because it is worth knowing
   after three failures: **there is no mechanism here.** The norm plus an
   independent re-derivation is the best available, and the second half is
   doing the work.
