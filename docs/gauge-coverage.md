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
| `square` | **yes** | **yes** | `tests/test_square_row_gauge_jax.py` drives both from one battery — the emission gates run the pipeline through `check`/`escalate` to a replayed witness, eagerly AND with the `square` fused inside a `jit`; `interval-containment-eager-and-jit` drives `interval_env` against the values jax computes on this target. Its transfer-face mutation is CAUGHT by the containment gate and ADMITTED by every emission gate, so the two faces are visibly independent rather than assumed to be |
| the other 29 in the emission set | **no** | mostly yes | containment sweep (Run 11), which is transfer-face by construction |
| the 6 transfers with no emission row | n/a | mostly yes | same |

**Totals: 39 registered transfers, 33 in the emission set. Two primitives have
both faces gauged.** Those three figures were measured when this table was
first written and are left as they were read. **As of the round that added
`square`'s emission row (2026-08-03) the emission set is 34 and THREE
primitives have both faces gauged** (`scatter`, `scatter-add`, `square`); the
transfer-side figures are a different population at a different date and are
not restated here. `docs/supported-primitives.md` is generated from the live
registries and is the current count of both.

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
  nothing about 33 emission rows.
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
2. **The other 31 emission rows.** Mostly thin arithmetic plans, but "mostly" is
   an assumption this table exists to stop making.
3. **Nothing about the transfer face is urgent** — it is the better-covered of
   the two, by containment, and a transfer error yields a box that excludes an
   executed value rather than a false `VERIFIED`.
