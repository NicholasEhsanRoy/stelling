<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# Gauge coverage, per primitive, per face

**What "gauged" has meant, and why this table exists.** Every parameter-space
gauge in this project wires exactly one gate, so its scope is whichever entry
point its author happened to call — and three different scopes have all been
reported as "gauged" without ever saying which. The word covered the emission
plans in one instrument, the interval transfers in another, and containment in a
third. This table replaces the word with a measurement.

A row's two faces are independent code paths that must agree:

- **emission** — the obligation-face plan that turns an equation into SMT terms
  (`_scatter_set_plan`, `_scatter_add_plan`, …). Getting it wrong mints a false
  model, and on the `VERIFIED` side nothing downstream catches it.
- **transfer** — the interval transfer (`_t_scatter`, `interval.dot_general`, …).
  Getting it wrong produces a box that excludes the truth, on the
  solver-independent leg.

They share an oracle where one exists, but each retains checks beyond it —
`_t_scatter`'s docstring enumerates its retained checks precisely because that is
how the two faces drifted apart once already.

## The table

| primitive | emission gauged | transfer gauged | by what |
|---|---|---|---|
| `scatter` | **yes** | **yes** | `tests/test_scatter_gauge_jax.py` already drove both — `emission-agreement` runs the pipeline through `escalate`, `interval-soundness` and `point-box-exactness` drive the transfer. Now also `param_gauge.py` (both gates) and `scatter_containment.py` |
| `scatter-add` | **yes** | **yes** | as above |
| `dot_general` | **no** | **yes** | `param_gauge_dot.py` drives `TRANSFERS`/`interval_env` only; `tests/test_dot_general_interval.py` is containment |
| `convert_element_type` | **no** | **yes** | `param_gauge_convert.py` drives `interval_env` only |
| `pow` | **yes** | **partly** | `tests/test_pow_row_gauge_jax.py` drives both from one battery, whose size, survivor count and asymmetry count are MEASURED under *The `pow` row's gauge* below rather than restated here — a digit typed in this cell is the class of thing this page got wrong. Emission: both exponent branches through `check`/`escalate` to a replayed witness, eager AND with the `pow` fused inside a `jit`, at a MEASURED exponent arity and with the per-element emission asserted INVARIANT to the array shape over a printed range of element counts. Transfer: `interval-containment-eager-and-jit` drives `interval_env` against the values jax computes on this target — but only over STRICTLY POSITIVE base boxes, which is the whole domain of `interval.pow_`'s corner rule, so the transfer column is **partly** and not **yes**. See the paragraphs below for what else it does not reach |
| `square` | **yes** | **yes** | `tests/test_square_row_gauge_jax.py` drives both from one battery — the emission gates run the pipeline through `check`/`escalate` to a replayed witness, eagerly AND with the `square` fused inside a `jit`; `interval-containment-eager-and-jit` drives `interval_env` against the values jax computes on this target. Its transfer-face mutation is CAUGHT by the containment gate and ADMITTED by every emission gate, so the two faces are visibly independent rather than assumed to be |
| every other member of the emission set | **no** | mostly yes | containment sweep (Run 11), which is transfer-face by construction |
| every transfer with no emission row | n/a | mostly yes | same |

*Those two rows read "the other 29" and "the 6" when the table was written, and
the second had since gone FALSE: it was `39 − 33`, and neither population is
that any more. Measured at `53f9f84`, the transfer registries hold primitives
the emission set does not, and the difference is now well into double figures —
`dynamic_slice` and `dynamic_update_slice` are only the two most recent. The
live sizes and BOTH set differences are on `docs/supported-primitives.md`,
which is generated from the registries and byte-compared by
`tests/test_supported_primitives_doc.py`, so it cannot drift the way a constant
typed here does. Read the counts there. What this table asserts is which FACES
are gauged, and that claim does not depend on how many rows there are.*

**Totals: 39 registered transfers, 33 in the emission set. Two primitives have
both faces gauged.** Those three figures were measured when this table was
first written and are left as they were read. **As of the round that added
`square`'s emission row (2026-08-03) the emission set is 34 and THREE
primitives have both faces gauged** (`scatter`, `scatter-add`, `square`); the
transfer-side figures are a different population at a different date and are
not restated here. `docs/supported-primitives.md` is generated from the live
registries and is the current count of both.

**As of the `pow`-gauge round (2026-08-15, B7) the emission set is 36 and a
FOURTH row is gauged from both faces** — `pow`, with its transfer column
reading **partly** for the reason the table gives. That figure is this run's
(`len(obligation._SUPPORTED)`, printed by the gauge's own registry census);
`docs/supported-primitives.md` is generated from the live registries and is
the count to read rather than this sentence.

**As of the index-bounds round (2026-08-09) the registry is 48 and the
emission set is unchanged** — `dynamic_slice` and `dynamic_update_slice` are
transfer-only rows, so they join the "no emission row" population. Both are
gauged on the TRANSFER face and by the strongest instrument in this table's
sense: a containment sweep that enumerates the whole declared index set and
executes the real primitive at every point, driven in BOTH directions by five
deliberately wrong hulls, two of them committed
(`tests/test_index_bounds.py`, `design/index-bounds-round.md`). Neither face
is gauged for emission, because neither has one.

## The `pow` row's gauge: what it reaches, and what it does not

Added 2026-08-15 (B7). Measured on jax 0.11.0, python 3.12.3,
`/home/nick/venvs/stelling-jax`, z3 wheel + cvc5 wheel, x64 enabled, 30 s
solver budget. The battery and every attribution below are the file's own
run, printed by `python tests/test_pow_row_gauge_jax.py`.

**Why this row needed a gauge and not a bar.** `pow` is discharge-affecting
and un-self-checking in exactly the sense
`verdict.VERIFIED_BARRED_PRIMITIVES`' argument names: a wrong encoding that
produces a spurious witness is caught by exact-rational replay, and one that
MISSES a violation is not. The 0.2.0 audit found three soundness defects in
this row (§ S1, S2, S3) and observed that none of them was expressible as a
mutation, because the row had no named seam at all. It has three now —
`smt._pow_integer_body`, `smt._pow_rational_lines`, `smt._pow_aux_name` —
and the battery below is what they are for. `tests/test_bar_membership_policy.py`
carries the decision not to bar the row, and its cost measured both ways.

**Every figure in this section is RECOMPUTED from the live battery** by
`tests/test_pow_row_gauge_jax.py::test_the_documented_coverage_figures_are_the_MEASURED_ones`,
which parses the two blocks below and compares them against a real gauging
run. It is written that way because the sentence it replaces — *"Sixteen of
the seventeen are caught by more than one gate"* — was contradicted by the
table printed directly beneath it, which already showed five single-covered
entries where the measurement said six. The page argues from *"a gauge with
one single-covered mutation is one edit from a hole"*, and that premise was
true six times over while the prose said once. The defect is not the digit; it
is a digit in prose that nothing recomputes.

<!-- gauge-figures: pow -->
MEASURED: **32 mutations, 0 survivors, 32 face asymmetries, 17 caught by more
than one gate, 15 caught by exactly one.**

<!-- gauge-joint-reach: pow -->
JOINT REACH: **21 (element, n_out) shapes x 3 base spellings x 4 integer
exponents = 252 tuples; 21 x 3 x 3 (p, q) pairs = 189 tuples — the FULL
PRODUCT in both cases.**

That second line is the one the last two rounds exist for, and it is recomputed
by the same test. The axes were once asserted one at a time; all three
equalities held on a tree whose joint reach was 24 of the then-84 tuples and 25
of the then-63. **The BASE SPELLING is a coordinate of it as of this round**,
and it was added for the same reason and after the same demonstration: the
shape and the exponent were measured jointly while the seams' OTHER argument —
the base term — was driven at `input` and nothing else by every fixture in the
battery, and five wrongnesses conditioned on it survived all 22 gates.

**The single-covered set, NAMED rather than counted.** Each of these is caught
by exactly one gate, so deleting that gate deletes the only measurement of
that wrongness. All fifteen carry an `ALONE` marking in the mutation table
below, the test derives that marking from the measurement rather than trusting
it, and the rows are compared as a SET so an alphabetised table is not rejected
for being alphabetised. For thirteen of them the exclusivity is itself the
FINDING rather than a gap: the two exponent-conditional mutations
are seen only by the fixture at their own `(p, q)`, which is what says the rest
of the battery is blind to a denominator past 2 and a numerator past 1; the
three element-conditional mutations are seen only by the invariance gate, which
is what says every solver-driven fixture here is blind past element 1 on BOTH
branches and at every exponent; the four BASE-KIND-conditional mutations
are seen only by that same gate, which is what says every solver-driven fixture
here raises a DECLARED PROGRAM INPUT to a power and is blind to what its base
is spelled as; and each of the four ID-conditional mutations is seen by exactly
one of the two RANGE gates — three by the invariance gate and the name
collision by the freshness one — which is what says every solver-driven fixture
here drives the emitter's variable numbering at the handful of values its own
equations produce, measured at `out_id` `{2, 3}` and base ids `{0, 2}`. The
other two are admission-guard mutations, which only the admission gate can
reach.

**Two rows LEFT this list in the round that added the freshness gate**, and
the reason is the finding rather than an accident.
`emit-rational-aux-shared-across-elements` was seen only by the vector fixture
and `emit-rational-aux-collides-across-two-pow-OUTPUTS` only by the invariance
gate; both are now seen by
`every-auxiliary-is-declared-ONCE-and-named-FRESHLY` as well, because both are
name COLLISIONS and that gate asserts injectivity of the naming seam rather
than sampling it. What the exclusivity used to stand in for — "no solver-driven
fixture here holds two rational `pow`s at all" — is still true and is still
asserted, by the anti-vacuity floor on the measured `out_id` reach in
`test_the_driven_arity_is_MEASURED_at_the_seams_not_asserted_in_prose`. A claim
that moved from an exclusivity to a direct assertion got stronger, not weaker.

<!-- single-covered: BEGIN -->
| single-covered mutation | its only gate |
|---|---|
| `rational-admission-always-yes` | `non-dyadic-exponent-declines` |
| `exponent-rationalised-to-a-nearby-fraction` | `non-dyadic-exponent-declines` |
| `emit-integer-wrong-only-past-the-second-element-at-degree-five` | `emission-is-invariant-to-every-seam-argument-but-the-EXPONENT` |
| `emit-rational-wrong-only-at-a-larger-denominator` | `refutes-the-false-rational-property-at-denominator-four` |
| `emit-rational-wrong-only-at-a-numerator-past-one` | `refutes-the-false-rational-property-at-numerator-three` |
| `emit-rational-wrong-only-past-the-second-element` | `emission-is-invariant-to-every-seam-argument-but-the-EXPONENT` |
| `emit-rational-wrong-only-past-the-second-element-at-q-two` | `emission-is-invariant-to-every-seam-argument-but-the-EXPONENT` |
| `emit-integer-wrong-only-on-a-COMPUTED-base-at-degree-three` | `emission-is-invariant-to-every-seam-argument-but-the-EXPONENT` |
| `emit-integer-wrong-only-on-an-AUXILIARY-base-at-degree-three` | `emission-is-invariant-to-every-seam-argument-but-the-EXPONENT` |
| `emit-rational-wrong-only-on-a-COMPUTED-base-at-q-two` | `emission-is-invariant-to-every-seam-argument-but-the-EXPONENT` |
| `emit-rational-wrong-only-on-an-AUXILIARY-base-at-q-two` | `emission-is-invariant-to-every-seam-argument-but-the-EXPONENT` |
| `emit-rational-wrong-only-at-a-LATER-OUT-ID` | `emission-is-invariant-to-every-seam-argument-but-the-EXPONENT` |
| `emit-rational-aux-collides-only-at-a-LATER-OUT-ID` | `every-auxiliary-is-declared-ONCE-and-named-FRESHLY` |
| `emit-integer-wrong-only-at-a-LATER-BASE-ID` | `emission-is-invariant-to-every-seam-argument-but-the-EXPONENT` |
| `emit-rational-wrong-only-at-a-LATER-BASE-ID` | `emission-is-invariant-to-every-seam-argument-but-the-EXPONENT` |
<!-- single-covered: END -->

**The count did not move; the membership did** — and then the count moved too,
which is the reason the set is compared by NAME and not by size. It has now
happened four times. The exponent widening took two entries off this list —
`emit-integer-loses-the-reciprocal`, also seen by the second-magnitude
reciprocal gate, and `replay-as-the-identity`, also seen by the `p = 3`
positive control — and the two conditional mutations it added replaced them.
The shape round took `emit-rational-one-aux-for-two-elements` off (the
invariance gate sees it too, without a solver) and added
`emit-rational-wrong-only-past-the-second-element`. Six, six and six; three
different sets. A test comparing a count against a count would have passed
through all three rounds unchanged. The joint round was the first to move the
size, because the two mutations it added are conditioned on the PRODUCT of the
element index and the exponent rather than on either alone; the ID round moved
it again in both directions at once — four new single-covered items in, two
freshness items out to the new gate.

**The mutation table.** The middle column is the COMPLETE catch set, checked
for equality against the measurement: every catching gate is named, no gate is
named that does not catch, and the cell may contain nothing else — no
connective, no count, no `and most others`. That rule has no threshold in it,
which is deliberate, because the two weaker rules this column has had both
failed. The bare COUNTS ("11 gates", "(+5)") went stale five times out of
sixteen and were deleted; what replaced them was a HEDGE, which is the same
claim with the digit removed and nothing recomputing it — measured totals
behind `and most others` and `and others` ran from 4 of 21 to 18 of 21. Worse,
a subset satisfied the test, so deleting one gate name from the
`emit-integer-loses-the-reciprocal` row left the suite green while deleting
exactly the fact the paragraph above argues from.

A complete list of up to nineteen gate names per row is not something anyone
should retype, and that is why the hedge was there. So the column is
GENERATED: `python tests/test_pow_row_gauge_jax.py --doc-blocks` prints both
blocks from a live gauging run, and the third column's prose is merged back in
by hand. The test is what confirms the merge.

The third column is prose and cannot be compared, but the arithmetic in it
can: every `A^B = C` anywhere in this section is decided exactly by the same
test, and the CONDITIONAL rows' emitted forms and caps are recomputed from the
live mutations by
`test_the_conditional_mutations_CAP_the_value_below_the_bound_RECOMPUTED`.
That column was previously unchecked in every respect, and the audit falsified
a whole cell of it without the suite noticing.

<!-- mutation-table: BEGIN -->
| mutation | caught by | what the catch looks like |
|---|---|---|
| `row-absent-from-the-emission-set` | `refutes-the-false-integer-property`, `refutes-under-jit`, `discharges-the-true-integer-property`, `refutes-the-false-integer-property-at-degree-five`, `discharges-the-true-integer-property-at-degree-five`, `discharges-the-negative-exponent-identity`, `discharges-the-fourth-power-reciprocal-identity`, `discharges-the-true-rational-upper-bound`, `discharges-the-true-rational-lower-bound`, `refutes-the-false-rational-property`, `refutes-the-false-rational-property-at-numerator-three`, `discharges-the-true-rational-bound-at-numerator-three`, `refutes-the-false-rational-property-at-denominator-four`, `discharges-the-true-rational-bound-at-denominator-four`, `refutes-the-false-vector-property`, `witness-executes-through-jax`, `fragment-is-nonlinear`, `emission-is-invariant-to-every-seam-argument-but-the-EXPONENT`, `every-auxiliary-is-declared-ONCE-and-named-FRESHLY`, `replay-agrees-with-jax` | everything declines |
| `fragment-claims-linear` | `refutes-the-false-integer-property`, `refutes-under-jit`, `discharges-the-true-integer-property`, `refutes-the-false-integer-property-at-degree-five`, `discharges-the-true-integer-property-at-degree-five`, `discharges-the-negative-exponent-identity`, `discharges-the-fourth-power-reciprocal-identity`, `discharges-the-true-rational-upper-bound`, `discharges-the-true-rational-lower-bound`, `refutes-the-false-rational-property`, `refutes-the-false-rational-property-at-numerator-three`, `discharges-the-true-rational-bound-at-numerator-three`, `refutes-the-false-rational-property-at-denominator-four`, `discharges-the-true-rational-bound-at-denominator-four`, `refutes-the-false-vector-property`, `witness-executes-through-jax`, `fragment-is-nonlinear` | the script violates its own declared logic |
| `rational-admission-always-yes` | `non-dyadic-exponent-declines` ALONE | `x ** 0.1` is admitted |
| `exponent-rationalised-to-a-nearby-fraction` | `non-dyadic-exponent-declines` ALONE | audit S1's own defect: `0.1` becomes `1/10` |
| `transfer-is-the-base` | `refutes-the-false-integer-property`, `refutes-under-jit`, `discharges-the-true-integer-property`, `refutes-the-false-integer-property-at-degree-five`, `discharges-the-true-integer-property-at-degree-five`, `refutes-the-false-rational-property-at-numerator-three`, `discharges-the-true-rational-bound-at-numerator-three`, `witness-executes-through-jax`, `interval-containment-eager-and-jit`, `fragment-is-nonlinear` | the box excludes a value jax computes |
| `emit-integer-off-by-one` | `discharges-the-true-integer-property`, `discharges-the-true-integer-property-at-degree-five`, `discharges-the-negative-exponent-identity`, `discharges-the-fourth-power-reciprocal-identity` | a TRUE property comes back refuted; the witness fails replay |
| `emit-integer-exponent-ignored` | `refutes-the-false-integer-property`, `refutes-under-jit`, `refutes-the-false-integer-property-at-degree-five`, `discharges-the-negative-exponent-identity`, `discharges-the-fourth-power-reciprocal-identity`, `witness-executes-through-jax` | the violation disappears — a MISSED violation |
| `emit-integer-loses-the-reciprocal` | `discharges-the-negative-exponent-identity`, `discharges-the-fourth-power-reciprocal-identity` | `x^-2` is emitted as `x^2` |
| `emit-integer-wrong-only-above-degree-three` | `refutes-the-false-integer-property-at-degree-five`, `discharges-the-fourth-power-reciprocal-identity` | **CONDITIONAL.** Correct at every exponent the shipped battery drove and wrong at `abs(exp) >= 4`; emits `x^5` for `x^6`, so `x**6 <= 40` over [1, 2] goes VERIFIED when `2^6 = 64` |
| `emit-integer-wrong-only-past-the-second-element-at-degree-five` | `emission-is-invariant-to-every-seam-argument-but-the-EXPONENT` ALONE | **CONDITIONAL ON THE PRODUCT of the element index and the exponent, which is where the round that closed each axis separately left a hole.** Correct everywhere except at element 2 and beyond AND at exponent 5 — and BOTH of those coordinates are printed as driven above, so this is not covered by the exponent radius disclosed further down. It emits `x^2` for `x^5` past element 1, so on `x[2]**5 - x[1]**5 <= 10` over [1, 2] the reachable value at element 2 is capped at `2^2 = 4` while the truth is `2^5 = 32` against `1^5 = 1`, and a violated obligation comes back `discharged`. The predecessor's shape-invariance gate drove ONE hardwired exponent per branch — 3 on this one — so every other integer exponent was reached at element 0 only and this survived all 22 gates |
| `emit-rational-wrong-only-at-a-larger-denominator` | `refutes-the-false-rational-property-at-denominator-four` ALONE | **CONDITIONAL.** Correct at `q = 2` and wrong at `q >= 4`; emits `aux^6 = x^1` for `x^(1/4)`, which caps the reachable value at `81^(1/6)` — below the bound — so a bound of 2.9 goes VERIFIED where the truth is `81^(1/4) = 3` |
| `emit-rational-wrong-only-at-a-numerator-past-one` | `refutes-the-false-rational-property-at-numerator-three` ALONE | **CONDITIONAL.** Correct across the whole `p == 1` family and wrong outside it; emits `aux^2 = x^1` for `x^(3/2)`, which caps the reachable value at `4^(1/2)` — below the bound — so a bound of 7.9 goes VERIFIED where the truth is `4^(3/2) = 8` |
| `emit-rational-wrong-only-past-the-second-element` | `emission-is-invariant-to-every-seam-argument-but-the-EXPONENT` ALONE | **CONDITIONAL, on the SHAPE axis, and the mutation that survived the round that widened the exponent one.** Correct at elements 0 and 1 — the whole of what the two-element vector fixture drives — and wrong from element 2 on, where it emits `aux^6 = x^1` for `x^(1/4)`. On `x[2]**0.25 - x[1]**0.25 <= 1.9` over [1, 81] it turns a REFUTED into a VERIFIED; the truth is `81^(1/4) = 3` minus `1^(1/4) = 1`, which is 2. Caught by the INVARIANCE gate, so the catch holds for every element index in that gate's range and not only for the index this mutation names |
| `emit-rational-wrong-only-past-the-second-element-at-q-two` | `emission-is-invariant-to-every-seam-argument-but-the-EXPONENT` ALONE | **The same PRODUCT on the rational branch, and the sharper of the two because `q = 2` is the denominator the whole battery is built around.** Correct everywhere except at element 2 and beyond AND at `q == 2`; emits `aux^4 = x^1` for `x^(1/2)` there, so on `x[2]**0.5 - x[1]**0.5 <= 7.9` over [1, 81] the reachable value at element 2 is capped at `81^(1/4) = 3` while the truth is `81^(1/2) = 9` against `1^(1/2) = 1`, and a REFUTED query comes back VERIFIED. Every element index past 1 used to be driven at `(1, 4)` and nowhere else, so this combination of two DRIVEN coordinates was ungauged and — unlike the exponent radius below — undisclosed |
| `emit-integer-wrong-only-on-a-COMPUTED-base-at-degree-three` | `emission-is-invariant-to-every-seam-argument-but-the-EXPONENT` ALONE | **CONDITIONAL ON THE SEAM'S OTHER ARGUMENT — the base TERM, which no round had enumerated.** Correct at every element, every count and every exponent when the base is a declared program input, and wrong when it is an emitted `t...` intermediate at exponent 3: it emits `x^2` for `x^3`. Every fixture in the battery raised a DECLARED INPUT to a power, so this passed all 22 gates and rode out as a false VERIFIED on ordinary jax — `(x+1)**3 <= 23` over [1, 2], where the truth at the extremum is `3^3 = 27` and the mutated script cannot exceed `3^2 = 9`. Real programs almost always `pow` a computed quantity |
| `emit-integer-wrong-only-on-an-AUXILIARY-base-at-degree-three` | `emission-is-invariant-to-every-seam-argument-but-the-EXPONENT` ALONE | **The same axis at the THIRD spelling, which is the one the base-kind finding itself nearly missed.** A rational `pow` feeding an integer `pow` hands the seam an `aux_...` base, not a `t...` one, so a mutation conditioned on `aux` survives a battery that has just closed `t`. It emits `x^2` for `x^3` there, so `(x**0.5)**3 <= 7.9` over [1, 4] goes REFUTED → VERIFIED: the truth at the extremum is `4^(3/2) = 8` and the mutated script cannot exceed `4^(2/2) = 4` |
| `emit-rational-wrong-only-on-a-COMPUTED-base-at-q-two` | `emission-is-invariant-to-every-seam-argument-but-the-EXPONENT` ALONE | **The base-term axis on the RATIONAL branch, at `q = 2` — the denominator the whole battery is built around, so nothing about this is a corner.** Emits `aux^4 = x^1` for `x^(1/2)` when the base is a `t...` intermediate, which caps the reachable value at the fourth root: on `(x+1)**0.5 <= 8.9` over [0, 80] the truth at the extremum is `81^(1/2) = 9` and the mutated script cannot exceed `81^(1/4) = 3` |
| `emit-rational-wrong-only-on-an-AUXILIARY-base-at-q-two` | `emission-is-invariant-to-every-seam-argument-but-the-EXPONENT` ALONE | **The auxiliary spelling on the rational branch, which is a `pow` of a `pow` — `(x**0.5)**0.5`, ordinary code.** Emits `aux^4 = x^1` for `x^(1/2)` when the base is another element's `aux_...`, so over [1, 6561] the truth at the extremum is `6561^(1/4) = 9` while the mutated outer encoding cannot exceed `6561^(1/8) = 3`, and a bound of 8.9 turns REFUTED into VERIFIED. Note that the INNER `pow` of this program is untouched: the mutation fires on the base's spelling, not on the exponent, and both calls carry `(1, 2)` |
| `emit-rational-wrong-only-at-a-LATER-OUT-ID` | `emission-is-invariant-to-every-seam-argument-but-the-EXPONENT` ALONE | **THE RANGE INSIDE AN ENUMERATED ARGUMENT.** Emits `aux^(q+2) = x^p` once the auxiliary's own NAME spells an `out_id` of 4 or more, and the row's own text below that. `out_id` was classified DRIVEN — "reaches the seam at more than one value AND a wrongness conditioned on it is CAUGHT" — on a measured reach of `{2, 3}`, of which only `2` reaches a verdict-producing gate; the universal was false, and this survived all 22 gates on `41329d7` while minting a verdict-level false VERIFIED on three ordinary jax operations. `y = (x+1)-1 ; r = y**0.5 ; r[0]-r[1] <= 7.9` over [1, 81]^2 drives `out_id` 4, the truth is `81^(1/2) = 9` against `1^(1/2) = 1`, and the mutated encoding caps element 0 at `81^(1/4) = 3` |
| `emit-rational-aux-collides-only-at-a-LATER-OUT-ID` | `every-auxiliary-is-declared-ONCE-and-named-FRESHLY` ALONE | **THE FRESHNESS CLAIM AT AN UNSWEPT `out_id`**, and the item that says why the freshness gate is a second gate rather than a branch of the invariance one. `emit-rational-aux-collides-across-two-pow-OUTPUTS` with the un-swept coordinate as its condition: `out_id` dropped from the name once it is 4 or more, so two `pow` OUTPUTS share an auxiliary. The invariance gate CANONICALISES that name to `AUX` and is blind to it by construction. Its failure is a LOST REFUTATION rather than a missed violation — one symbol declared twice is illegal SMT-LIB2, both backends decline, and a nested `pow` over [1, 6561] whose truth is `6561^(1/4) = 9` against `1^(1/4) = 1` returns `unknown` where the baseline refutes |
| `emit-integer-wrong-only-at-a-LATER-BASE-ID` | `emission-is-invariant-to-every-seam-argument-but-the-EXPONENT` ALONE | **THE SAME RANGE ONE ARGUMENT OVER: the id inside the BASE TERM.** A base's KIND is what `_base_spelling` keeps and its ID is what it throws away, so three spellings were swept while the ids reaching the seams were `{0, 2}`. Emits `x^(exp-1)` once the base term's id is 4 or more. Survived all 22 gates on `41329d7`, and turns a `violated-witness` into a `discharged` on `((x+1)-1)**3` over [1, 3], where the truth is `3^3 = 27` and the mutated script cannot exceed `3^2 = 9`. The end-to-end verdict is intercepted by the vacuity widen re-check, which raises `EmissionInfidelityError`, so the demonstration stops at the escalation outcome exactly as its shape-conditioned twin does |
| `emit-rational-wrong-only-at-a-LATER-BASE-ID` | `emission-is-invariant-to-every-seam-argument-but-the-EXPONENT` ALONE | **THE BASE-TERM ID ON THE RATIONAL BRANCH**, a separate item because the two branches are two seams. Emits `aux^(q+2) = x^p` once the base term's id is 4 or more. Survived all 22 gates on `41329d7` and mints a verdict-level false VERIFIED: two padding equations put the base at `t5`, the truth is `81^(1/2) = 9` against `1^(1/2) = 1` under a bound of 7.9, and the mutated encoding caps the reachable value at `81^(1/4) = 3` |
| `emit-rational-sides-swapped` | `discharges-the-true-rational-upper-bound`, `refutes-the-false-rational-property`, `refutes-the-false-rational-property-at-numerator-three`, `refutes-the-false-rational-property-at-denominator-four`, `discharges-the-true-rational-bound-at-denominator-four`, `refutes-the-false-vector-property` | `aux^p = x^q` is `x^(q/p)`, a different function |
| `emit-rational-root-guard-dropped` | `discharges-the-true-rational-lower-bound`, `refutes-the-false-vector-property` | the NEGATIVE root satisfies the negated obligation |
| `emit-rational-constraint-never-asserted` | `discharges-the-true-rational-upper-bound`, `discharges-the-true-rational-lower-bound`, `refutes-the-false-rational-property`, `refutes-the-false-rational-property-at-numerator-three`, `discharges-the-true-rational-bound-at-numerator-three`, `refutes-the-false-rational-property-at-denominator-four`, `discharges-the-true-rational-bound-at-denominator-four`, `refutes-the-false-vector-property` | `aux` is free; every true property becomes sat |
| `emit-rational-denominator-off-by-one` | `refutes-the-false-rational-property`, `refutes-the-false-rational-property-at-numerator-three`, `refutes-the-false-rational-property-at-denominator-four`, `refutes-the-false-vector-property` | `aux^3 = x` caps the value below the bound — a MISSED violation |
| `emit-rational-aux-is-the-base` | `discharges-the-true-rational-upper-bound`, `refutes-the-false-rational-property`, `refutes-the-false-rational-property-at-numerator-three`, `refutes-the-false-rational-property-at-denominator-four`, `discharges-the-true-rational-bound-at-denominator-four`, `refutes-the-false-vector-property` | the encoding collapses to the identity |
| `emit-rational-aux-shared-across-elements` | `refutes-the-false-vector-property`, `every-auxiliary-is-declared-ONCE-and-named-FRESHLY` | **by MALFORMEDNESS**: two `declare-const` of one symbol, both backends refuse, the obligation returns `unknown`. NOT seen by the invariance gate, and that is correct — every element emits the same lines about the same shared symbol, so there is no per-element difference to see; only a verdict can catch this one |
| `emit-rational-aux-collides-across-two-pow-OUTPUTS` | `emission-is-invariant-to-every-seam-argument-but-the-EXPONENT`, `every-auxiliary-is-declared-ONCE-and-named-FRESHLY` | **THE THIRD `_pow_aux_name` PARAMETER, which the signature enumeration turned up and which the same round had already made reachable.** Drops `out_id` from the auxiliary's name: still fresh per ELEMENT of one output, colliding between two DIFFERENT rational `pow`s. Freshness across elements had two battery items and freshness across OUTPUTS had none, because no fixture held two rational `pow`s — until the `auxiliary` arm of the base-spelling sweep, which is `(x**0.5)**e`. Measured: two distinct `out_id` values reach the seam across the battery, and the collision shows up as a per-element block that canonicalises its own base and auxiliary to the same token. Exclusivity is the finding: no solver-driven fixture here chains two rational `pow`s at all |
| `emit-rational-one-aux-for-two-elements` | `refutes-the-false-vector-property`, `emission-is-invariant-to-every-seam-argument-but-the-EXPONENT`, `every-auxiliary-is-declared-ONCE-and-named-FRESHLY` | **the sharpest item.** Well-formed: one declaration, two constraints, so `x0_0 == x0_1` and the difference of the two roots collapses to 0. An obligation false at `x = [4, 1]` comes back `discharged`. A silent missed violation, and nothing in the tree caught it before. Caught three times now — by the vector fixture's verdict and, without a solver, by BOTH text-level gates: the invariance gate, because declaring the auxiliary for one element and not the other IS a per-element difference in the emitted text, and the freshness gate, because one `declare-const` for two elements is a name that is not fresh |
| `replay-exponent-inverted` | `refutes-the-false-rational-property`, `refutes-the-false-rational-property-at-numerator-three`, `refutes-the-false-rational-property-at-denominator-four`, `refutes-the-false-vector-property`, `replay-agrees-with-jax` | the replay disagrees with jax at a grid point |
| `replay-as-the-identity` | `refutes-the-false-rational-property-at-numerator-three`, `replay-agrees-with-jax` | as above |
<!-- mutation-table: END -->

**What the battery's coverage claim may honestly say, and the measurement it
rests on.** "Both exponent branches" is a statement about branches and says
nothing about exponents, and the difference is where three false VERIFIEDs
lived: the shipped battery drove integer exponents `{-2, 3}` and the single
pair `(1, 2)`, an unstated SCOPE, and three wrongnesses conditioned outside it
passed all fourteen gates while each turned a genuinely REFUTED query into
VERIFIED on an exponent the guard admits. The claim is now:

> Both exponent branches of the `pow` row, at integer exponents
> `[-4, -2, 3, 5]` — both signs at both magnitude classes — and at exactly
> the three `(p, q)` pairs `1/2`, `1/4`, `3/2`, which is three of the 448
> admitted pairs and not the admitted space. A wrongness conditioned outside
> those two sets is NOT gauged.
>
> On the SHAPE axis, the `(element, n_out)` pairs of element counts
> `[1, 2, 3, 4, 5, 6]` reach `smt._pow_aux_name`. Of those, `[1, 2]` are driven
> END TO END to a verdict; the whole range is driven through the EMISSION,
> where the row's per-element output is asserted INVARIANT to the shape.
>
> On the BASE-TERM axis — the exponent seams' OTHER argument — the EMISSION is
> driven with the base written in each of the three spellings `smt.emit`
> produces for a symbol: a declared program input (`x...`), an emitted
> intermediate (`t...`) and a rational `pow`'s own auxiliary (`aux_...`). The
> per-element output is asserted invariant to that too. Every NON-emission
> stage is driven with an `input` base and only an `input` base.
>
> On the ID axis — which is the RANGE half of the two arguments above and not
> a fifth axis — the emitted block is asserted INVARIANT to every emitter id
> in `[0, 15]`, at every symbol spelling, every driven shape and every driven
> exponent; and the auxiliary's NAME is asserted FRESH over the same range,
> between the elements of one output and between two outputs. An id past 15 is
> NOT gauged, and an id is unbounded, so that residue is disclosed rather than
> closed. The ids the gates DRIVE are measured and asserted to lie inside the
> range: `out_id` at `{2, 3}`, base ids at `{0, 2}`.
>
> And the axes are claimed JOINTLY, not one at a time: every one of those
> shapes, in every one of those spellings, is driven at every one of the
> integer exponents and at every one of the `(p, q)` pairs — the FULL PRODUCT
> in both cases, asserted as an equality against it, and counted in the
> machine-checked JOINT REACH line above. So an emission wrongness conditioned
> on the element index, on the element count, on a DRIVEN base spelling, on a
> driven exponent, on an id inside the range, or on ANY COMBINATION of the
> five, is caught. A
> non-emission wrongness past two elements or outside an `input` base, or
> anything at all past six elements, or any rank above 1, or anything
> conditioned on an exponent outside the two sets above, on an id past the top
> of the range, or on a base that is
> not a symbol at all, is NOT gauged.

Four and three is better than two and one and is still finite, so the honest
form of the claim is to print the sets rather than describe them. They are not
typed into that paragraph either: they are derived from the gauge file's
fixture table and MEASURED at all three seams by
`test_the_driven_arity_is_MEASURED_at_the_seams_not_asserted_in_prose`, which
instruments `smt._pow_integer_body`, `smt._pow_rational_lines` and
`smt._pow_aux_name`, runs every gate against the baseline and fails if the
reach is not exactly that set. A fixture that drifts moves the measurement and
fails there; it cannot leave this paragraph standing.

**THE SHAPE HALF OF THAT CLAIM WAS PROSE FOR A ROUND, WHICH IS THE SAME DEFECT
ONE LEVEL DOWN.** The round that measured the exponents instrumented two of
the row's three seams and left out the only one that is handed the array
shape, then wrote "any other array shape" into the not-reached list. A
mutation correct at elements 0 and 1 and wrong from element 2 on passed all
twenty-one gates and minted a false VERIFIED — `81^(1/4) = 3` against a
neighbour of 1, under a bound of 1.9. The measured/asserted line ran through
the middle of the instrument.

**AND THEN THE REPAIR MEASURED THREE AXES AS MARGINALS AND WROTE A CLAIM ABOUT
THE PLANE, WHICH IS THE SAME DEFECT ONE DIMENSION UP.** The round that added
the shape axis asserted the integer exponents against their declared set, the
`(p, q)` pairs against theirs and the `(element, n_out)` pairs against theirs —
and all three equalities passed on a tree where every element index `>= 2` was
driven at `(1, 4)` and nowhere else. `(1, 2)` and `(3, 2)`, both printed above
as driven, reached elements 0 and 1; the integer exponents `-4`, `-2` and `5`,
all printed above as driven, reached element 0. The cause was one line down in
the instrument: the shape-invariance gate's probes were hardwired to a single
exponent per branch, because each carried a BOUND tuned by hand to straddle at
that exponent. So the radius moved off the element axis and onto the exponent
axis, where nothing printed it. A wrongness conditioned on `element >= 2 AND
q == 2` — both coordinates inside the printed sets, so NOT covered by the
exponent radius disclosed below — passed all 22 gates and turned `81^(1/2) = 9`
minus `1^(1/2) = 1` into a discharged obligation under a bound of 7.9. **No
mutation was needed to find it: recording `(element, n_out, p, q)` jointly
shows the hole on its own**, and that is what the arity test records now.

The repair is a probe SHAPE rather than more probes. The invariance probes now
compare two ELEMENTS of one array against zero — `r[n-1] - r[0] <= 0` on a
strictly positive box — which lands in an interval symmetric about the bound
under the real transfer AND under `transfer-is-the-base`, at every exponent and
every element count. That symmetry is a SELF-subtraction and not monotonicity:
every element of a declared array carries the identical interval, so
`[a, b] - [a, b] = [a-b, b-a]` whether or not the row is monotone. (Strict
positivity is load-bearing for a different reason — DOMAIN ADMISSIBILITY: a
negative integer exponent is a reciprocal and needs a base excluding 0, and a
fractional exponent on a negative base is declined.) Nothing is tuned per
exponent, so the gate can quantify over the whole driven exponent set instead
of one point in it, and the joint reach becomes the full product rather than a
disclosure. Both branches are swept, negative integer exponents included; two
mutations conditioned on the product (`element >= 2 AND q == 2`,
`element >= 2 AND exp == 5`) are in the battery to keep it that way.

**AND THEN THE RADIUS MOVED ONTO THE SEAM'S OTHER ARGUMENT, WHICH IS THE THIRD
TIME AND THE REASON THE ENUMERATION ITSELF IS NOW DERIVED FROM THE CODE.**
`smt._pow_integer_body(term, exp_val)` takes two arguments and this gauge
measured the joint reach of ONE of them. Every fixture in the battery raised a
DECLARED PROGRAM INPUT to a power — measured across all 22 gates, the
base-term prefix reaching either exponent seam was `x` and nothing else — so a
wrongness conditioned on "the base is not an input" was correct on every
fixture. Five of them survived all 22 gates:

| survived every gate on the previous tree | its oracle |
|---|---|
| integer, wrong only at `exp == 3` AND an intermediate base | `3^3 = 27`, against a bound of 23 |
| integer, wrong only at `exp == 3` AND an auxiliary base | `4^(3/2) = 8`, against a bound of 7.9 |
| integer, wrong only at `exp == 3` AND a numeral base | no oracle measured — this spelling is disclosed, not driven (see the not-reached list) |
| rational, wrong only at `(p, q) == (1, 2)` AND an intermediate base | `81^(1/2) = 9`, against a bound of 8.9 |
| rational, wrong only at `(p, q) == (1, 2)` AND an auxiliary base | `6561^(1/4) = 9`, against a bound of 8.9 |

Each is pinned to a DRIVEN exponent, element 0 and `n_out` 1 — every
coordinate the page prints as driven — so none of them is the disclosed
exponent radius wearing a different hat. **Four of the five are demonstrated
minting a verdict-level false VERIFIED** on programs anyone would write: the
four rows above that carry an oracle, each recomputed end to end by
`test_the_base_kind_conditioned_mutations_are_CAUGHT_by_the_INVARIANCE_gate`,
which asserts REFUTED on the baseline and VERIFIED under the mutation. The
fifth survives every gate too, but its verdict effect was never measured, and
saying otherwise would be quoting a run nobody made. **Real programs almost
always `pow` a computed quantity.**

The repair is the same shape as the last one — the base kind is a COORDINATE of
the joint reach, and the invariance gate sweeps it against a shared reference,
so what is closed is the product of the shape, the spelling and the exponent
rather than three marginals. Four mutations conditioned on the base kind at a
driven exponent are in the battery to keep it that way.

**What is meant to end the pattern is not the fourth axis; it is that the axis
LIST is no longer written from memory.** A seam is a pure function of its
arguments, so its arguments are the complete set of things a wrongness inside
it can be conditioned on.
`test_every_SEAM_ARGUMENT_is_a_gauged_COORDINATE_or_a_named_exemption` reads
all three seams' signatures with `inspect` and requires every parameter to be
classified. A future round that adds an argument to a seam fails that test
instead of shipping an axis nobody enumerated.

**Running the enumeration corrected its own first draft.**
`smt._pow_aux_name`'s `out_id` — the auxiliary's
freshness across two `pow` OUTPUTS rather than across two elements — was
written down as an undriven gap, on the reasoning that no fixture holds two
rational `pow`s. Measured, that was already false: the `auxiliary` arm of the
base-spelling sweep is `(x**0.5)**e`, which is two rational `pow`s and
therefore two `out_id`s. Two distinct values reach the seam across the
battery, and a mutation that drops `out_id` from the name — colliding two
outputs' auxiliaries while keeping them fresh per element — is CAUGHT.

**AND THEN THE LIST WAS CLOSED AND EVERY RANGE WAS STILL OPEN, WHICH IS THE
DEFECT THE ENUMERATION ROUND SHIPPED.** A wrongness is conditioned on an
argument's VALUE, not on its name, so enumerating the parameters closed the
LIST and left every parameter's RANGE exactly as open as before. Two entries in
that table said otherwise and both were wrong in the same way:

- `out_id` was classified DRIVEN, a class defined as *"reaches the seam at more
  than one value AND a wrongness conditioned on it is CAUGHT"*. That universal
  was false. The measured reach was `{2, 3}` — and `{2}` at every
  verdict-producing gate, `3` arriving only through one emission-only gate — so
  what held was the EXISTENTIAL: one particular mutation, conditioned at
  `out_id >= 3`, was caught. Mutations conditioned at `>= 4`, `>= 5` and `>= 6`
  survived all 22 gates, and one mints a verdict-level false VERIFIED on three
  ordinary jax operations: `y = (x+1)-1 ; r = y**0.5 ; r[0]-r[1] <= 7.9` over
  [1, 81]^2 hands the seam `out_id` 4, the truth is `81^(1/2) = 9` against
  `1^(1/2) = 1`, and the mutated encoding cannot exceed `81^(1/4) = 3`.
- `aux_name` was classified DERIVED because it is *"a pure function of
  parameters already classified … SO it carries no axis of its own"*. That `so`
  does not carry. Derived-ness closes the LIST; the RANGES compose. `aux_name`
  is derived from `out_id`, which was classified but not closed, and a product
  over `element` and `n_out` says nothing about `out_id`.

The same defect ran one seam over, on an argument nobody had called an id at
all: both exponent seams read a BASE TERM, and `x0`, `t2`, `aux_2` spell an id
exactly as `aux_2_0` does. The base's KIND was swept and the ID inside it was
`{0, 2}` — so `emit-integer-wrong-only-at-a-LATER-BASE-ID` and its rational
twin survived every gate too, the second minting a false VERIFIED on the same
three-operation shape.

**The repair is an INVARIANCE over a printed range, plus a vocabulary that
forces the choice.** The ids are closed the way the shape is: the emitted block
must be the REFERENCE block with its symbol names substituted, at every id in
`[0, 15]`, for every symbol spelling, at every driven shape and every driven
exponent — evidence about every conditioning function on an id inside the
range rather than about the values a fixture produced. Freshness needs its own
gate, because the invariance comparison canonicalises the auxiliary's name to
`AUX` and is blind to a wrongness that changes only the name;
`every-auxiliary-is-declared-ONCE-and-named-FRESHLY` asserts INJECTIVITY of the
naming seam over the same range and, end to end, that every `declare-const` in
an emitted probe script is declared once.

Each seam parameter now carries a CLOSURE as well as an axis — **SWEPT** (every
value in its range is driven; only `element` is one), **INVARIANT** (a named
gate asserts the text is independent of it over a declared range) or
**DISCLOSED** (a finite set of driven points, with the rest named in the gauge's
`SCOPE`) — and a RESIDUE naming the part of its range that is not closed, whose
phrase must appear in that `SCOPE`. The DISCLOSED class was EMPTY for a round,
and its emptiness was the tell: a vocabulary with nowhere to say *"listed, but
not closed"* had nowhere to put the exponent, which is the most disclosed thing
on this page. `exp_val`, `p` and `q` are in it now. A DERIVED argument is
required to carry its sources' residues, which is the `so` above made
mechanical.

What none of this closes is whether a declared RANGE is wide enough. An id is
unbounded and `[0, 15]` is finite, so a wrongness conditioned past the top
survives — and it has to be wrong ABOVE the range and right inside it, which is
a defect shaped to the instrument. That residue is disclosed in the not-reached
list below, and the ids the gates DRIVE are asserted to lie inside the range
and to be a PROPER subset of it, so a sweep that started recording itself would
fail rather than print its own range as the reach.

**And a note on what that instrument can and cannot do, because "MEASURED"
reads as "sufficient".** The equality between what the fixtures declare and
what the seams see is DRIFT protection: it is real, it is exactly what failed
when a paragraph outlived its fixtures, and it cannot tell anyone that the
declared set is the right one. The genuinely new coverage lives in the test's
ANTI-VACUITY floors, and every floor is typed at the radius of a mutation
someone already wrote — `|exp| >= 4`, `q >= 4`, `p != 1`, element index `>= 2`.
Widen the battery again and the next mutation sits one step outside the new
floor. **That is not a prediction; it is measured — and it was measured again
one round later, on the ID axis, which is why the repair there is an invariance
and not a fifth floor.** Three mutations
conditioned one step outside the exponent floors survive every gate
on this tree today, each minting a false VERIFIED on an exponent the admission
guard admits:

| survives every gate | its oracle |
|---|---|
| integer, wrong only at `abs(exp) >= 6` | `2^6 = 64`, against a bound of 40 |
| rational, wrong only at `q >= 8` | `6561^(1/8) = 3`, against a bound of 2.9 |
| rational, wrong only at `p >= 5` | `4^(5/2) = 32`, against a bound of 31.9 |

They are left alone deliberately. Catching them means moving each floor by one
and writing three more mutations, which is this round again rather than the end
of it; what the row honestly claims is the printed set above, and a wrongness
conditioned outside it is disclosed here rather than covered.

The one part of this row's gauge that is not radius-shaped **on the SHAPE,
BASE-TERM and ID axes** is the invariance gate: emission is text, an
elementwise row's per-element text cannot legitimately depend on the shape, on
what its base is spelled as, or on the emitter's variable numbering, so
asserting that it does not is evidence about every conditioning
function on `(element, n_out)`, on the base's spelling and on an id inside the
printed range rather than about a
sampled point — and, since the gate sweeps the driven exponent set, that
argument now holds at each of those exponents rather than at one per branch.
**It is not an argument about the exponent axis and this page previously
overstated it as one.** The exponent axis has no invariance available — the
emitted text is legitimately different at a different exponent, which is the
whole content of the row — so it remains a finite set of driven points, which
is why they are printed. What is closed is the product of the shape and
base-spelling axes with THAT SET, and a wrongness conditioned on an exponent
outside it is exactly as ungauged as the table above says, whether or not it
also names an element index or a base spelling.

**The base-spelling half of that is also bounded, and by a SAMPLE rather than
by an argument.** The three spellings are driven because `smt.emit` produces
exactly three for a SYMBOL, and the block comparison canonicalises the symbol
away — but the gate is still comparing three points, not quantifying over the
set of strings a base could be. What makes that honest rather than a
re-run of the old defect is that the recorder records the spelling it saw:
a fourth spelling reaching a seam appears in the measured reach or in its
`inconsistent` list, and fails the arity test, instead of sitting in a gap.

**What this gauge does NOT reach, stated as flatly as the table above.**

- **An emitter ID past the top of the printed range** — an `out_id` at
  `smt._pow_aux_name`, or the id inside a base term at either exponent seam.
  The invariance and freshness gates close every id inside `[0, 15]`, which is
  every conditioning function on an id rather than the values a fixture
  happens to produce, and an id is UNBOUNDED so no finite range closes it.
  What is asserted instead is that the reach lies inside the range and nowhere
  near its top — measured, `out_id` at `{2, 3}` and base ids at `{0, 2}` — so a
  program with more equations ahead of its `pow` than the range has room for is
  outside the closure. A wrongness living there has to be right at every id up
  to the top and wrong above it, which is a defect shaped to the instrument
  rather than to the row; widening the range is cheap and would move the line
  rather than remove it, which is why the residue is disclosed instead.
- **The rational branch at an ODD `q` — which no longer exists to reach.**
  This page and the gauge both used to describe the branch as covering `q`
  even *and odd*. The odd half was structurally unreachable:
  `obligation.pow_exponent_rational` is `Fraction` of a binary64, every finite
  binary64 is a dyadic rational, so in lowest terms `q` is a power of two and
  `q == 1` takes the integer branch. Measured: `q` over the whole 448-pair
  admitted set is exactly `{2, 4, 8, 16, 32, 64, 128}`, 0 odd in 500 000
  random draws. An untested branch that READS as covered is worse than no
  branch, so the fix is enforcement rather than a corrected sentence: the root
  guard is unconditional now, the derivation refuses a non-dyadic rational,
  admission DECLINES an odd denominator and the emission REFUSES one.
  `test_the_odd_denominator_branch_is_UNREACHABLE_and_FAILS_CLOSED` pins all
  three on the standard the integer-dtype guard already gets.
- **A base that is not a SYMBOL — the NUMERAL spelling, which is reachable and
  measured surviving the whole battery.** `smt.emit` normally folds a `pow`
  whose base and exponent are both constant, but when `smt._renderable`
  declines the fold (the exact value's denominator crosses CPython's
  `int`→`str` cap — `Fraction(1e-100) ** 64` has a 5000-digit one) the
  literal's TEXT is pasted into the seam call in place of a symbol. Measured on
  this tree: `jnp.power(1e-100, 64.0)` hands `smt._pow_integer_body` the term
  `(/ 492525077454931 4925…)`, a compound s-expression, and a mutation
  conditioned on "the base is not an identifier" survives every gate — 22 of
  them on `41329d7` and all 23 on this tree, re-measured because the two new
  RANGE gates sweep symbol spellings and a numeral is not one. It is
  not DRIVEN because the invariance argument the other three spellings rest on
  presupposes a symbol to canonicalise, and because its element index is not
  recoverable from the text — so the reach recorder reports such a term as an
  INCONSISTENCY rather than counting it as element 0 of an unknown kind, which
  is what keeps this gap from being absorbed into a driven set. Closing it is a
  further round with its own probe, not a sentence.
- **`integer_pow`'s row.** The seams are `pow`'s own, and
  `test_the_pow_seams_do_not_move_the_integer_pow_row` asserts that every
  emission mutation in this battery leaves an `integer_pow` slice's text
  byte-identical. The renderer the two rows share, `smt._repeated_product`, is
  deliberately NOT in the battery: mutating it moves both rows, so a catch
  could not be attributed to either. That renderer is pinned by
  `tests/test_pow_audit_findings.py` (audit S2) and not by mutation here.
- **The transfer face outside strictly positive base boxes, and outside a
  SCALAR shape.** `interval.pow_` raises for a base interval reaching 0 or
  below and the propagator turns that into a noted ⊤ decline, which contains
  everything, so a containment gate there would be vacuous. The row's own
  integer-branch fixtures deliberately straddle zero — that ⊤ is what makes
  them interval-undecidable and is why the emission is what decides them.
  Separately, `interval-containment-eager-and-jit` builds every one of its
  probes at shape `()`, so the containment claim is about one element; the
  transfer is reached at two only through the vector fixture's end-to-end
  verdict, and at no other count at all. The shape-invariance argument does
  not transfer here — there is no emitted text to compare on that face.
- **The libm assumption under the transfer.** `interval.pow_` evaluates its
  corners with the host's `math.pow` bumped one ulp outward, under
  `POW_LIBM_ASSUMPTION`. The containment gate is a spot check at four points
  per box on five boxes; it is not evidence that the bracket holds for every
  operand, and audit 0.2.0 § S9/S11 is the record of that class of assumption
  failing for the compiled libm rather than the host one.
- **`pow`'s integer-dtype guard, which is UNREACHABLE rather than ungauged.**
  `pow` is in `obligation._INT_OVERFLOW_EMITTED`, and no jax program can put
  an integer dtype on it: measured on jax 0.11.0, `lax.pow` raises *"pow does
  not accept dtype int32 at position 0"*, `jnp.power(int32, 2.0)` inserts a
  `convert_element_type` to float64 first, and `jnp.power(int32, 2)` binds
  `integer_pow` instead. The guard is therefore recorded UNCOVERED, per
  `docs/norms.md`, and the unreachability itself is pinned so that a jax
  release admitting it fails loudly.
- **Both emission CAPS** (`INTEGER_POW_EXPANSION_CAP`,
  `RATIONAL_POW_DEGREE_CAP`) and the 448-pair admitted exponent set, which
  `tests/test_pow_audit_findings.py` owns. Mutating the degree cap upward
  cannot be done safely: an admitted non-dyadic exponent renders a product
  with 3.6e16 factors.
- **`ieee` semantics.** `escalate` refuses every ieee propagation, so there is
  no ieee `pow` EMISSION to gauge; the ieee transfer rides a declared libm
  budget and belongs to `tests/test_libm_budget.py`.
- **Rank above 1, element counts above six, and every NON-emission stage above
  two elements.** The emission's shape dependence is closed by invariance over
  `[1, 2, 3, 4, 5, 6]` **at each driven exponent, which is the product and not
  two marginals**; the transfer, the slice, the solver dispatch, the replay and
  the verdict are driven at one element and at two, and that is a sample,
  stated as one. Nothing here is evidence about rank ≥ 2 at all, and the
  scatter rows' history in this file is the reason that sentence is written
  rather than implied.
- **The affine domain**, which does not admit `pow`.

**A gap in the seams, named because naming a residual is not closing one.**
Neither `_pow_rational_lines` nor `_pow_integer_body` sees the element index —
only `_pow_aux_name` does, and it is not called on the integer branch at all —
so a mutation conditioned on the element has to read the index off the base
term's name, which `emit-rational-one-aux-for-two-elements` and all three
`...-past-the-second-element...` items do. They are fixture-shaped mutations
rather than seam-shaped ones, and a seam that saw the index would make them
seam-shaped. The same gap runs through the MEASUREMENT: recording the joint
`(element, n_out, exponent)` reach means recovering the shape at seams that are
not handed it — from the aux call that immediately precedes each rational
element, and on the integer branch from the base term's index plus the fact
that `smt.emit` renders one eqn's elements consecutively from 0. Both
recoveries are cross-checked and a disagreement is reported rather than assumed
away, but a seam that carried its own shape would need neither. The
consequence has narrowed rather than gone: the
shape-invariance gate closes the axis for the emission whether or not the seam
sees the index, because it compares emitted TEXT and does not need the seam to
be parameterised. What is still open is the same gap on the non-emission
stages, where there is no text to compare. That is the next refinement of this
row's seams, not a claim that the space is closed.

**And that RUN RECOVERY can silently OVERSTATE, which is a disclosed follow-up
rather than a closed case.** The recovery's docstring states its guarantee as
an invariant — an out-of-order or gappy run is reported as an inconsistency
instead of being counted — and the guarantee is really a coincidence of today's
gate set. Two SEPARATE scalar `pow` eqns on `x[0]` and `x[1]`, emitted
consecutively, present as elements 0 and 1 of one two-element eqn: that run is
neither out of order nor gappy, nothing fires, and the recorder books an
`n_out` of 2 where the truth is two eqns of 1. Measured against jaxpr ground
truth on this tree it overstates nothing and understates nothing — no gate here
holds two consecutive scalar `pow`s — so the figures above are a reading today
and would stop being one the moment a fixture did. The honest fix keys the run
on the EQN rather than on the element index; it is recorded here rather than
taken, because a guarantee that holds by coincidence and READS as an invariant
is the class of claim this page exists to refuse.

## "Gauged" is a claim about a SPACE, not about a row

A row is gauged over the shapes something measured it on, and a sweep is blind
one step past its bound. This project has now measured five corruptions living
in that step — two of them in the scatter rows, each a `violated-witness`
turned `discharged` with the whole suite green in both columns, and each
keyed on a shape the sweep does not contain (a SET axis of 9, an ADD leading
axis of 4). Raising the bound has failed four times; the fourth escape sat at
exactly the newly declared ceiling.

So for these two rows the table's **yes** now means something stricter than
"a sweep exists": the ADMITTED space equals the GAUGED space. The static-index
`scatter` SET row declines an operand axis longer than 8, and the
`scatter-add` accumulate row declines an operand outside rank ≤ 3 / every axis
≤ 3 / at most 12 elements, with `tests/test_scatter_gauge_jax.py` pinning the
source bounds equal to the sweep's in both directions. Past those bounds the
rows do not run ungauged — they refuse, and the obligation comes back
`unknown`. That costs answers (see `SOUNDNESS.md`, 2026-08-06) and it is the
only reading of **yes** that a bounded sweep can honestly support.

**The INDEX COLUMN is inside the guarded space too, as of the round after
this one.** The paragraph that stood here said the bound guarded the shape and
not the ADD row's column length, that `jax.ops.segment_sum` reaches a column of
4 on an admitted operand, and that the axis was gauged by a mutation battery
rather than an exhaustive sweep. Naming a residual is not closing one, and this
one was then demonstrated: a census of `len(ks)` at the row across the whole
suite reaches `{1, 2, 3, 4, 6, 254, 255}` — 5 absent, 7..253 absent — and a
line-neutral mis-route wrong only at a column of 5 turned a `violated-witness`
into `discharged` with the suite green. The admitted column space is now the
union of three exhaustively swept families and nothing else: one index over
every gauged shape; every column of `range(n)` to the power of the length, for
lengths up to 6, on a RANK-1 operand; and the single-element operand at every
length up to 255, where every index is forced to 0 and the length is the only
free parameter. Outside that the row refuses.

What the column bound gives up, stated because a narrowing that is not stated
is a silent one: a multi-index `segment_sum` onto an operand of rank 2 or 3 —
normal-matrix assembly, say — now declines. Exhausting `n ** length` over every
gauged shape is 12510 traces and 80 seconds against 3 for the rank-1 family,
and the census says nothing **in the pytest-driven tree** reaches the row with
more than one index on a higher-rank operand. That scope is the census's, not
the repository's: `corpus/` is driven by hand, and `corpus/run_census.py`
classifies primitives out of jaxprs without ever reaching the row.

**And "normal-matrix assembly, say" is not hypothetical here.**
`tests/test_scatter_gauge_jax.py`'s own header names "a small normal-matrix
assembly in the segment_sum style" among the programs it gauges, and its
`m-assembly` fixture is that program: `jax.ops.segment_sum` over per-point
(2, 2) blocks, which is a rank-3 operand with an index column of 3. Posed
through the slicing face at the fixture's own declared shapes it now refuses —
*"index column of 3 element(s) on operand (2, 2, 2) is outside the GAUGED
accumulate column space"* — while the same accumulation flattened to a rank-1
operand is admitted. Nothing in the tree fails, because the in-tree
`m-assembly` cases are settled by the interval transfer and never reach the
row. A downstream harness that escalates one gets UNKNOWN, and that is the
shape of program most likely to meet this bound.

## What this table says that the earlier numbers did not

- **"35 of 39 transfers gauged, zero survivors"** was a *transfer-face* figure.
  Containment cannot see an emission gate at all — it compares a box against an
  executed value, and an emission plan produces neither. So that sweep says
  nothing about the emission set — not one row of it, whatever its size, which
  `docs/supported-primitives.md` reports and this sentence deliberately does
  not restate. (It said "33 emission rows"; the emission set is not 33 any
  more, and the *nothing* is what the sentence is for.)
- **The two rows the project built by hand are gauged in opposite directions.**
  `param_gauge.py` was emission-only and `param_gauge_dot.py` is transfer-only,
  and each was quoted as coverage of its row.
- **But "no instrument covered both faces of anything" is too strong**, and the
  table is what shows it. `tests/test_scatter_gauge_jax.py` already drove both
  faces of the scatter rows, through end-to-end pipeline gates rather than by
  calling the emission plan directly. What was emission-only was the
  *parameter-space* gauge — which is the one whose "zero survivors" got quoted
  as an acceptance criterion, so the finding stands and its scope is narrower
  than first stated.
- **`dot_general`'s emission face is the largest ungauged surface**, because it
  is the only hand-built emission row besides the scatter pair and it carries a
  shared oracle whose retained checks are not exercised from the emission side.

## Reading a survivor count correctly

**"Caught" is disjunctive**: any gate declining a mutation is enough. So a
regression confined to ONE face leaves the survivor count at zero, because the
other face still catches the mutation. **A survivor count cannot detect a
one-face regression, by construction.**

What detects one is a mutation that some gates catch and others admit — the two
faces disagreeing about a single program. `param_gauge.py` reports that as a
**face asymmetry** and it is a finding whether or not any survivor exists.
Verified by injection: with the transfers made to stop reading the real index
dtype and the emission face untouched, the survivor count stayed at **0** while
**6 face asymmetries** appeared (three per row: int8 at n=129, uint8 at n=257,
int16 at n=32769). Reverting returned both counts to zero.

## What follows from the table

The work is ordered by what is unrecoverable if wrong, not by what is missing:

1. **`dot_general`'s emission face.** It is in the emission set, it is
   hand-built, and no instrument drives it.
2. **The emission rows with no gauge at all.** Mostly thin arithmetic plans,
   but "mostly" is an assumption this table exists to stop making — and `pow`
   is the row that showed why. It looked like a two-line product expansion; it
   is also an auxiliary-variable encoding stating three separable claims, one
   of which (a fresh constant per element) can be broken WELL-FORMEDLY and
   silently discharge a false obligation. The count of ungauged rows is
   `len(obligation._SUPPORTED)` minus the four in the table above; it is not
   written here as a digit, because the digit that used to be here (31) went
   stale twice.
3. **Nothing about the transfer face is urgent** — it is the better-covered of
   the two, by containment, and a transfer error yields a box that excludes an
   executed value rather than a false `VERIFIED`.
