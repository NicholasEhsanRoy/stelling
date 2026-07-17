# Re-checking the 7 in-semantics properties against their incidents — registered before reading

**Status:** REGISTRATION, 2026-07-18. The supply/layer probes recorded the
property-quality correction as **forward-looking** ("any future
registration using one-line properties inherits this correction"); the 20
existing properties were never re-checked. That is a known gap. Three
properties examined closely each had a defect — npy#1360 (false over its
region), dfx#368 (conflates flow with solver-internal state), dfx#207 (a
property-incident gap). And **criterion (ii) anchors to these**:
"discharges the registered property," the breakdown's strongest cell, is
worth exactly what the property is worth.

## Scope and what it does NOT touch

Corpus: the **7 in-semantics hits** (`design/semantics-classification.md`)
— bjx#D416, bjx#969, jmd#339, dfx#207, dfx#417, npy#249, npy#1133. This
re-check **does not re-score any count** and **does not re-run the demand
band** — the incidents happened regardless of how anyone worded a property
afterwards. It is a **fact about the anchor**, reported wherever a
relation is reported, exactly like the denominator's provenance chain.

## Buckets — fixed before reading

| bucket | meaning |
|---|---|
| **sound** | constructive, non-circular, a state predicate, **and it reaches the incident's mechanism** |
| **defective — and how** | one of: false over its region / circular / trajectory-predicate / property-incident gap / conflates layers |

The test that decides it: does the registered one-liner **reach the
incident's mechanism** — would proving it have bearing on the failure that
was actually filed?

## Bias, declared

I have been arguing the result is weak, and this check could weaken it
further. What makes it defensible is **symmetry**: if the 7 come back
sound, "discharges the registered property" means exactly what it says and
the breakdown is *stronger* than I have been reading it. The check can
strengthen the anchor as easily as weaken it; the bucket is decided on the
incident, not on which direction I want.

---

# Reading (2026-07-18 — the 7 against their incidents)

| hit | property | incident's mechanism | verdict |
|---|---|---|---|
| **dfx#417** | `|y_n| ≤ B` over horizon | ReversibleHeun instability = `|y_n|` growing unboundedly | **sound** — the property *is* the incident (bounded trajectory vs the growth that unbounds it); reaches the mechanism |
| **npy#249** | `isfinite(step_size) ∧ step_size > 0` through warmup | `step_size → nan` during warmup | **sound** — `nan` violates `isfinite`; the property reaches the mechanism (ℝ-partial semantics note stands, but the property is well-formed) |
| **jmd#339** | neighbour list ⊇ pairs within cutoff | a pair within cutoff is missed at the PBC boundary | **sound** — the incident is precisely `list ⊉ pairs`; reaches the mechanism (ℝ-partial: the miss is a float wrap) |
| **dfx#207** | accepted `dt ≥ dt_min` | step collapse → `max_steps` hit | **defective — property-incident gap.** `dt ≥ dt_min` is a step-*size* floor; the incident is a step-*count* ceiling — you hit `max_steps` taking `dt_min`-sized steps. Proving it does not bear on the failure |
| **npy#1133** | sampler state stays in support | mangled chains from bad init | **defective — property-incident gap.** Mangled chains **stay in support** (bad mixing, not leaving it); the incident *satisfies* the property, so it cannot detect the failure |
| **bjx#969** | non-finite proposal ⇒ `step_size_max` shrinks | the NaN detector never fires (dead guard) | **defective — conflates layers.** The property is about the *response* to non-finite; the bug is in *detecting* it — with the detector dead, the implication is vacuously true. The real obligation (detector completeness) is unstated |
| **bjx#D416** | adapted `step_size ≥ ε` | RNG key reused → near-zero step_size | **defective — conflates layers.** The symptom (`step_size < ε`) is captured, but the cause is PRNG plumbing (a killed category), a different layer; proving the adaptation output says nothing about the reused key |

## Count: **3 sound, 4 defective**

**Sound (3):** dfx#417, npy#249, jmd#339. **Defective (4):** dfx#207
(property-incident gap), npy#1133 (gap — incident satisfies it), bjx#969
(conflates layers — detection vs response), bjx#D416 (conflates layers —
symptom vs RNG cause).

Not the "3 of 4 defective" the tiny earlier sample suggested — it came
back **mixed, 4 of 7 defective (57%)**. The anchor is neither solid nor
rotten; it is half-and-half, and the check was symmetric enough to say so.

## What this does to the counts (nothing) and to the breakdown (this)

No count moves. But wherever "discharges the registered property" is
reported, the anchor quality rides with it. For the **3 hits that
mechanize** in E2a-with-control-flow (`design/affine-disclosure.md`):

- **dfx#417 — sound anchor.** Its precondition-relation discharges a
  property that reaches its incident.
- **npy#249 — sound anchor** (ℝ-partial).
- **dfx#207 — defective anchor** (property-incident gap). Its "discharges
  the registered property" is worth less than the words: the property it
  discharges does not reach the `max_steps` incident.

So the honest form of the E2a re-read: **3 of 7, on two sound anchors and
one defective one.** The re-check strengthened two of the three
mechanized results (their anchors reach their incidents) and confirmed the
third's known gap — the symmetric outcome the bias declaration promised.
