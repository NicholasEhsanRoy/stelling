# Semantics classification of the E2a corpus — registered before the run

**Status:** REGISTRATION, 2026-07-18. Committed before the seven
not-pre-committed hits are classified, and before any of the nine is run.
The one check that fails *green*: a property that is trivially true in ℝ
proves VERIFIED under the stamp's `semantics: real`, discharges its
registered property (the *strong* relation), and counts 1 — on a
statement that is false in the program that ran. `t + dt > t` is that
shape, and it is a 258-day bug (dfx#632) in this project's own corpus.
This is knowable today from artifacts on disk, and unlike every other
failure mode it makes the band *easier* — the asymmetry that justifies
one more pass before the run.

The registered semantics dial is `real` and this does not move it
(`SOUNDNESS.md`). This registration says which properties `real` can
reach.

## The buckets — fixed before classification

| bucket | meaning |
|---|---|
| **ℝ-faithful** | the property has real content; a proof in ℝ **with margin** (slack ≫ ulp) is evidence about the float program |
| **ℝ-vacuous** | the property is **trivially true in ℝ**; its entire content is float. `t + dt > t` for `dt > 0` has no ℝ content whatsoever |
| **ℝ-partial** | the property has real content, but the **incident's own mechanism is float-specific** (e.g. an `isfinite` property where ℝ catches mathematical blow-up but the incident was a float overflow). The ℝ proof is sound and does **not** discharge the incident |

**ℝ-partial is judged against the incident's mechanism, not the
property's wording** — the same discipline as the state/trajectory axis.
The tracker probe's `accuracy/float` shape column (dfx#632, dfx#657) is a
*product-split* label, not a semantics one; suggestive, not decisive. The
classification is done on the properties and their incidents.

## Pre-committed, before the other seven are read

> **dfx#632 → ℝ-vacuous. dfx#657 → ℝ-vacuous.** Recorded now by the
> proposer so neither can be re-classified once the other seven's buckets
> are visible. dfx#632's property is `t_{n+1} > t_n` (trivial in ℝ for
> `dt > 0`); dfx#657's is `realized step times = requested grid`
> (trivial in ℝ; the entire content is StepTo's f32 clipping).

## The handling — reuses existing machinery, adds no relation, moves no band

- **ℝ-vacuous → excluded from the denominator, before the run** — exactly
  what non-reconstructible got. A VERIFIED under ℝ on an ℝ-vacuous
  property is **not a mechanization success**, and deciding that after
  seeing two land green would be the renegotiation this registration
  exists to prevent.
- **ℝ-partial → runs; the relation names the semantics gap**: "discharges
  the ℝ version of the registered property; the incident's mechanism is
  float-specific and out of the registered semantics." That is
  precondition-shaped with the gap named — criterion (ii)'s **existing**
  second relation, applied to a gap it didn't anticipate. No new relation.
- **ℝ-faithful → runs normally.**

If the two pre-commits hold, R drops **9 → 7**. The band is **≥ 4
absolute across ≥ 2 libraries**, so the threshold does not move and the
bar rises — conservative. The ≥ 2-libraries clause is re-checked against
the survivors in the reading.
