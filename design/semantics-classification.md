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

---

# Reading (2026-07-18 — the nine reconstructible hits, on their registered properties)

| hit | registered property | bucket | grounds (judged on the incident's mechanism) |
|---|---|---|---|
| **dfx#632** | `t_{n+1} > t_n` | **ℝ-vacuous** | pre-committed; `t + dt > t` for `dt > 0` is trivially true in ℝ, entire content is float creep |
| **dfx#657** | realized step times = requested grid | **ℝ-vacuous** | pre-committed; equal in ℝ by construction, entire content is StepTo's f32 clipping |
| **npy#249** | `isfinite(step_size) ∧ step_size > 0` throughout warmup | **ℝ-partial** | incident was `step_size → nan` (float); `isfinite` has no ℝ content, and the ℝ half (`> 0`) does not discharge the nan |
| **jmd#339** | neighbor list ⊇ all pairs within cutoff | **ℝ-partial** | real geometric content, but the incident is float rounding in the PBC coordinate wrap (`99.999995` at cell `100`) — the ℝ version is true by construction; the bug is the wrap |
| **bjx#969** | non-finite proposal ⇒ `step_size_max` shrinks | **ℝ-partial** | the antecedent "non-finite" is float-only (vacuous in ℝ); incident is a dead detector. *Also lands `blocked` — implication, inert `assume` — so it counts 0 for that reason first* |
| **bjx#D416** | adapted `step_size ≥ ε` | **ℝ-faithful** | a real-valued invariant with margin ε; the incident's mechanism (RNG key reuse) is not float — it is a killed category (PRNG reuse), out of E2a's scope |
| **dfx#207** | accepted `dt ≥ dt_min` | **ℝ-faithful** | real invariant; incident is genuine stiffness, not float |
| **dfx#417** | `|y_n| ≤ B` over horizon | **ℝ-faithful** | a real bound with margin; incident is solver instability in the real dynamics |
| **npy#1133** | sampler state stays in support | **ℝ-faithful** | a real region invariant; incident is low-density init, not float |

## Counts

**ℝ-vacuous: 2** (dfx#632, dfx#657) — both pre-committed, both confirmed.
**ℝ-partial: 3** (npy#249, jmd#339, bjx#969). **ℝ-faithful: 4** (bjx#D416,
dfx#207, dfx#417, npy#1133).

**Denominator: 9 → 7.** The two ℝ-vacuous hits are excluded before the
run; the seven survivors are the four ℝ-faithful plus the three
ℝ-partial (which run with the semantics gap named in their relation).

**≥ 2-libraries, re-checked on the seven survivors:** diffrax (dfx#207,
dfx#417), blackjax (bjx#D416, bjx#969), jax-md (jmd#339), numpyro
(npy#249, npy#1133) — **four libraries**, so the clause remains
reachable.

Denominator provenance now has three subtractions plus the control:
**13 addressable → 9 reconstructible → 7 in-semantics**, and hit386 the
control that was never a member. Any count is "N of 7" only with that
chain attached.
