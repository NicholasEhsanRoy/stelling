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
MEASURED: **21 mutations, 0 survivors, 21 face asymmetries, 15 caught by more
than one gate, 6 caught by exactly one.**

**The single-covered set, NAMED rather than counted.** Each of these is caught
by exactly one gate, so deleting that gate deletes the only measurement of
that wrongness. All six carry an `ALONE` marking in the mutation table below,
the test derives that marking from the measurement rather than trusting it,
and the rows are compared as a SET so an alphabetised table is not rejected
for being alphabetised. For four of them the exclusivity is itself the FINDING
rather than a gap: `emit-rational-aux-shared-across-elements` is seen only by
the vector fixture, which is what says the scalar fixtures are blind to
per-element freshness; the two exponent-conditional mutations are seen only by
the fixture at their own `(p, q)`, which is what says the rest of the battery
is blind to a denominator past 2 and a numerator past 1; and
`emit-rational-wrong-only-past-the-second-element` is seen only by the
shape-invariance gate, which is what says every solver-driven fixture here is
blind past element 1. The other two are admission-guard mutations, which only
the admission gate can reach.

<!-- single-covered: BEGIN -->
| single-covered mutation | its only gate |
|---|---|
| `rational-admission-always-yes` | `non-dyadic-exponent-declines` |
| `exponent-rationalised-to-a-nearby-fraction` | `non-dyadic-exponent-declines` |
| `emit-rational-wrong-only-at-a-larger-denominator` | `refutes-the-false-rational-property-at-denominator-four` |
| `emit-rational-wrong-only-at-a-numerator-past-one` | `refutes-the-false-rational-property-at-numerator-three` |
| `emit-rational-wrong-only-past-the-second-element` | `emission-is-invariant-to-the-array-shape` |
| `emit-rational-aux-shared-across-elements` | `refutes-the-false-vector-property` |
<!-- single-covered: END -->

**The count did not move; the membership did**, which is the reason the set is
compared by NAME and not by size. It has now happened twice. The exponent
widening took two entries off this list — `emit-integer-loses-the-reciprocal`,
also seen by the second-magnitude reciprocal gate, and `replay-as-the-identity`,
also seen by the `p = 3` positive control — and the two conditional mutations
it added replaced them. The shape round took `emit-rational-one-aux-for-two-
elements` off (the invariance gate sees it too, without a solver) and added
`emit-rational-wrong-only-past-the-second-element`. Six, six and six; three
different sets. A test comparing a count against a count would have passed
through both rounds unchanged.

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
| `row-absent-from-the-emission-set` | `refutes-the-false-integer-property`, `refutes-under-jit`, `discharges-the-true-integer-property`, `refutes-the-false-integer-property-at-degree-five`, `discharges-the-true-integer-property-at-degree-five`, `discharges-the-negative-exponent-identity`, `discharges-the-fourth-power-reciprocal-identity`, `discharges-the-true-rational-upper-bound`, `discharges-the-true-rational-lower-bound`, `refutes-the-false-rational-property`, `refutes-the-false-rational-property-at-numerator-three`, `discharges-the-true-rational-bound-at-numerator-three`, `refutes-the-false-rational-property-at-denominator-four`, `discharges-the-true-rational-bound-at-denominator-four`, `refutes-the-false-vector-property`, `witness-executes-through-jax`, `fragment-is-nonlinear`, `emission-is-invariant-to-the-array-shape`, `replay-agrees-with-jax` | everything declines |
| `fragment-claims-linear` | `refutes-the-false-integer-property`, `refutes-under-jit`, `discharges-the-true-integer-property`, `refutes-the-false-integer-property-at-degree-five`, `discharges-the-true-integer-property-at-degree-five`, `discharges-the-negative-exponent-identity`, `discharges-the-fourth-power-reciprocal-identity`, `discharges-the-true-rational-upper-bound`, `discharges-the-true-rational-lower-bound`, `refutes-the-false-rational-property`, `refutes-the-false-rational-property-at-numerator-three`, `discharges-the-true-rational-bound-at-numerator-three`, `refutes-the-false-rational-property-at-denominator-four`, `discharges-the-true-rational-bound-at-denominator-four`, `refutes-the-false-vector-property`, `witness-executes-through-jax`, `fragment-is-nonlinear` | the script violates its own declared logic |
| `rational-admission-always-yes` | `non-dyadic-exponent-declines` ALONE | `x ** 0.1` is admitted |
| `exponent-rationalised-to-a-nearby-fraction` | `non-dyadic-exponent-declines` ALONE | audit S1's own defect: `0.1` becomes `1/10` |
| `transfer-is-the-base` | `refutes-the-false-integer-property`, `refutes-under-jit`, `discharges-the-true-integer-property`, `refutes-the-false-integer-property-at-degree-five`, `discharges-the-true-integer-property-at-degree-five`, `refutes-the-false-rational-property-at-numerator-three`, `discharges-the-true-rational-bound-at-numerator-three`, `witness-executes-through-jax`, `interval-containment-eager-and-jit`, `fragment-is-nonlinear` | the box excludes a value jax computes |
| `emit-integer-off-by-one` | `discharges-the-true-integer-property`, `discharges-the-true-integer-property-at-degree-five`, `discharges-the-negative-exponent-identity`, `discharges-the-fourth-power-reciprocal-identity` | a TRUE property comes back refuted; the witness fails replay |
| `emit-integer-exponent-ignored` | `refutes-the-false-integer-property`, `refutes-under-jit`, `refutes-the-false-integer-property-at-degree-five`, `discharges-the-negative-exponent-identity`, `discharges-the-fourth-power-reciprocal-identity`, `witness-executes-through-jax` | the violation disappears — a MISSED violation |
| `emit-integer-loses-the-reciprocal` | `discharges-the-negative-exponent-identity`, `discharges-the-fourth-power-reciprocal-identity` | `x^-2` is emitted as `x^2` |
| `emit-integer-wrong-only-above-degree-three` | `refutes-the-false-integer-property-at-degree-five`, `discharges-the-fourth-power-reciprocal-identity` | **CONDITIONAL.** Correct at every exponent the shipped battery drove and wrong at `abs(exp) >= 4`; emits `x^5` for `x^6`, so `x**6 <= 40` over [1, 2] goes VERIFIED when `2^6 = 64` |
| `emit-rational-wrong-only-at-a-larger-denominator` | `refutes-the-false-rational-property-at-denominator-four` ALONE | **CONDITIONAL.** Correct at `q = 2` and wrong at `q >= 4`; emits `aux^6 = x^1` for `x^(1/4)`, which caps the reachable value at `81^(1/6)` — below the bound — so a bound of 2.9 goes VERIFIED where the truth is `81^(1/4) = 3` |
| `emit-rational-wrong-only-at-a-numerator-past-one` | `refutes-the-false-rational-property-at-numerator-three` ALONE | **CONDITIONAL.** Correct across the whole `p == 1` family and wrong outside it; emits `aux^2 = x^1` for `x^(3/2)`, which caps the reachable value at `4^(1/2)` — below the bound — so a bound of 7.9 goes VERIFIED where the truth is `4^(3/2) = 8` |
| `emit-rational-wrong-only-past-the-second-element` | `emission-is-invariant-to-the-array-shape` ALONE | **CONDITIONAL, on the SHAPE axis, and the mutation that survived the round that widened the exponent one.** Correct at elements 0 and 1 — the whole of what the two-element vector fixture drives — and wrong from element 2 on, where it emits `aux^6 = x^1` for `x^(1/4)`. On `x[2]**0.25 - x[1]**0.25 <= 1.9` over [1, 81] it turns a REFUTED into a VERIFIED; the truth is `81^(1/4) = 3` minus `1^(1/4) = 1`, which is 2. Caught by the INVARIANCE gate, so the catch holds for every element index in that gate's range and not only for the index this mutation names |
| `emit-rational-sides-swapped` | `discharges-the-true-rational-upper-bound`, `refutes-the-false-rational-property`, `refutes-the-false-rational-property-at-numerator-three`, `refutes-the-false-rational-property-at-denominator-four`, `discharges-the-true-rational-bound-at-denominator-four`, `refutes-the-false-vector-property` | `aux^p = x^q` is `x^(q/p)`, a different function |
| `emit-rational-root-guard-dropped` | `discharges-the-true-rational-lower-bound`, `refutes-the-false-vector-property` | the NEGATIVE root satisfies the negated obligation |
| `emit-rational-constraint-never-asserted` | `discharges-the-true-rational-upper-bound`, `discharges-the-true-rational-lower-bound`, `refutes-the-false-rational-property`, `refutes-the-false-rational-property-at-numerator-three`, `discharges-the-true-rational-bound-at-numerator-three`, `refutes-the-false-rational-property-at-denominator-four`, `discharges-the-true-rational-bound-at-denominator-four`, `refutes-the-false-vector-property` | `aux` is free; every true property becomes sat |
| `emit-rational-denominator-off-by-one` | `refutes-the-false-rational-property`, `refutes-the-false-rational-property-at-numerator-three`, `refutes-the-false-rational-property-at-denominator-four`, `refutes-the-false-vector-property` | `aux^3 = x` caps the value below the bound — a MISSED violation |
| `emit-rational-aux-is-the-base` | `discharges-the-true-rational-upper-bound`, `refutes-the-false-rational-property`, `refutes-the-false-rational-property-at-numerator-three`, `refutes-the-false-rational-property-at-denominator-four`, `discharges-the-true-rational-bound-at-denominator-four`, `refutes-the-false-vector-property` | the encoding collapses to the identity |
| `emit-rational-aux-shared-across-elements` | `refutes-the-false-vector-property` ALONE | **by MALFORMEDNESS**: two `declare-const` of one symbol, both backends refuse, the obligation returns `unknown`. NOT seen by the invariance gate, and that is correct — every element emits the same lines about the same shared symbol, so there is no per-element difference to see; only a verdict can catch this one |
| `emit-rational-one-aux-for-two-elements` | `refutes-the-false-vector-property`, `emission-is-invariant-to-the-array-shape` | **the sharpest item.** Well-formed: one declaration, two constraints, so `x0_0 == x0_1` and the difference of the two roots collapses to 0. An obligation false at `x = [4, 1]` comes back `discharged`. A silent missed violation, and nothing in the tree caught it before. Caught twice now — by the vector fixture's verdict and, without a solver, by the shape-invariance gate, because declaring the auxiliary for one element and not the other IS a per-element difference in the emitted text |
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
> On the SHAPE axis, element counts `[1, 2, 3, 4, 5, 6]` reach
> `smt._pow_aux_name`. Of those, `[1, 2]` are driven END TO END to a verdict;
> the whole range is driven through the EMISSION, where the row's per-element
> output is asserted INVARIANT to the shape. So an emission wrongness
> conditioned on the element index or the element count is caught for every
> conditioning function inside that range. A non-emission wrongness past two
> elements, or anything at all past six, or any rank above 1, is NOT gauged.

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

**And a note on what that instrument can and cannot do, because "MEASURED"
reads as "sufficient".** The equality between what the fixtures declare and
what the seams see is DRIFT protection: it is real, it is exactly what failed
when a paragraph outlived its fixtures, and it cannot tell anyone that the
declared set is the right one. The genuinely new coverage lives in the test's
ANTI-VACUITY floors, and every floor is typed at the radius of a mutation
someone already wrote — `|exp| >= 4`, `q >= 4`, `p != 1`, element index `>= 2`.
Widen the battery again and the next mutation sits one step outside the new
floor. **That is not a prediction; it is measured.** Three mutations
conditioned one step outside the exponent floors survive all twenty-two gates
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
conditioned outside it is disclosed here rather than covered. The one part of
this row's gauge that is not radius-shaped is the shape
INVARIANCE gate: emission is text, an elementwise row's per-element text
cannot legitimately depend on the shape, so asserting that it does not is
evidence about every conditioning function inside a printed range rather than
about a sampled point. The exponent axis has no such argument available and
remains a finite set of driven points, which is why they are printed.

**What this gauge does NOT reach, stated as flatly as the table above.**

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
  `[1, 2, 3, 4, 5, 6]`; the transfer, the slice, the solver dispatch, the
  replay and the verdict are driven at one element and at two, and that is a
  sample, stated as one. Nothing here is evidence about rank ≥ 2 at all, and
  the scatter rows' history in this file is the reason that sentence is
  written rather than implied.
- **The affine domain**, which does not admit `pow`.

**A gap in the seams, named because naming a residual is not closing one.**
`_pow_rational_lines` does not see the element index — only `_pow_aux_name`
does — so a mutation conditioned on the element has to read the index off the
base term's name, which both
`emit-rational-one-aux-for-two-elements` and
`emit-rational-wrong-only-past-the-second-element` do. They are fixture-shaped
mutations rather than seam-shaped ones, and a seam that saw the index would
make them seam-shaped. The consequence has narrowed rather than gone: the
shape-invariance gate closes the axis for the emission whether or not the seam
sees the index, because it compares emitted TEXT and does not need the seam to
be parameterised. What is still open is the same gap on the non-emission
stages, where there is no text to compare. That is the next refinement of this
row's seams, not a claim that the space is closed.

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
