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
| `pow` | **yes** | **partly** | `tests/test_pow_row_gauge_jax.py` drives both from one battery of **17 mutations, 0 survivors, 17 face asymmetries**. Emission: both exponent branches through `check`/`escalate` to a replayed witness, eager AND with the `pow` fused inside a `jit`. Transfer: `interval-containment-eager-and-jit` drives `interval_env` against the values jax computes on this target — but only over STRICTLY POSITIVE base boxes, which is the whole domain of `interval.pow_`'s corner rule, so the transfer column is **partly** and not **yes**. See the two paragraphs below for what else it does not reach |
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

**The mutation table**, each entry with the gate that catches it. Sixteen of
the seventeen are caught by more than one gate; the exclusivity of the last
one is asserted, because that is what says the scalar fixtures are blind to
it.

| mutation | caught by | what the catch looks like |
|---|---|---|
| `row-absent-from-the-emission-set` | 11 gates | everything declines |
| `fragment-claims-linear` | 10 gates | the script violates its own declared logic |
| `rational-admission-always-yes` | `non-dyadic-exponent-declines` | `x ** 0.1` is admitted |
| `exponent-rationalised-to-a-nearby-fraction` | `non-dyadic-exponent-declines` | audit S1's own defect: `0.1` becomes `1/10` |
| `transfer-is-the-base` | `interval-containment-eager-and-jit` (+5) | the box excludes a value jax computes |
| `emit-integer-off-by-one` | `discharges-the-true-integer-property`, `discharges-the-negative-exponent-identity` | a TRUE property comes back refuted; the witness fails replay |
| `emit-integer-exponent-ignored` | `refutes-the-false-integer-property` (+3) | the violation disappears — a MISSED violation |
| `emit-integer-loses-the-reciprocal` | `discharges-the-negative-exponent-identity` | `x^-2` is emitted as `x^2` |
| `emit-rational-sides-swapped` | `discharges-the-true-rational-upper-bound` (+2) | `aux^p = x^q` is `x^(q/p)`, a different function |
| `emit-rational-root-guard-dropped` | `discharges-the-true-rational-lower-bound`, `refutes-the-false-vector-property` | the NEGATIVE root satisfies the negated obligation |
| `emit-rational-constraint-never-asserted` | 4 gates | `aux` is free; every true property becomes sat |
| `emit-rational-denominator-off-by-one` | `refutes-the-false-rational-property`, `refutes-the-false-vector-property` | `aux^3 = x` caps the value below the bound — a MISSED violation |
| `emit-rational-aux-is-the-base` | 3 gates | the encoding collapses to the identity |
| `emit-rational-aux-shared-across-elements` | `refutes-the-false-vector-property` | **by MALFORMEDNESS**: two `declare-const` of one symbol, both backends refuse, the obligation returns `unknown` |
| `emit-rational-one-aux-for-two-elements` | `refutes-the-false-vector-property` ALONE | **the sharpest item.** Well-formed: one declaration, two constraints, so `x0_0 == x0_1` and the difference of the two roots collapses to 0. An obligation false at `x = [4, 1]` comes back `discharged`. A silent missed violation, and nothing in the tree caught it before |
| `replay-exponent-inverted` | `replay-agrees-with-jax` (+2) | the replay disagrees with jax at a grid point |
| `replay-as-the-identity` | `replay-agrees-with-jax` | as above |

**What this gauge does NOT reach, stated as flatly as the table above.**

- **`integer_pow`'s row.** The seams are `pow`'s own, and
  `test_the_pow_seams_do_not_move_the_integer_pow_row` asserts that every
  emission mutation in this battery leaves an `integer_pow` slice's text
  byte-identical. The renderer the two rows share, `smt._repeated_product`, is
  deliberately NOT in the battery: mutating it moves both rows, so a catch
  could not be attributed to either. That renderer is pinned by
  `tests/test_pow_audit_findings.py` (audit S2) and not by mutation here.
- **The transfer face outside strictly positive base boxes.**
  `interval.pow_` raises for a base interval reaching 0 or below and the
  propagator turns that into a noted ⊤ decline, which contains everything, so
  a containment gate there would be vacuous. The row's own integer-branch
  fixtures deliberately straddle zero — that ⊤ is what makes them
  interval-undecidable and is why the emission is what decides them.
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
- **Array shapes past one item.** One two-element rational `pow` is driven,
  for per-element auxiliary freshness. Nothing here is evidence about rank ≥ 2
  or about longer axes, and the scatter rows' history in this file is the
  reason that sentence is written rather than implied.
- **The affine domain**, which does not admit `pow`.

**A gap in the seams, named because naming a residual is not closing one.**
`_pow_rational_lines` does not see the element index, so a mutation that
declares the shared auxiliary exactly once cannot be written against that
seam alone. The battery expresses it by pairing two patches and keying on the
base term's element suffix, which works and is the sharpest item in the table
— but it is a fixture-shaped mutation rather than a seam-shaped one, and a
seam that saw the index would make it seam-shaped. That is the next
refinement of this row's seams, not a claim that the space is closed.

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
