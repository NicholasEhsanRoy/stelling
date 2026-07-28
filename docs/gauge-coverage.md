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
| the other 29 in the emission set | **no** | mostly yes | containment sweep (Run 11), which is transfer-face by construction |
| the 6 transfers with no emission row | n/a | mostly yes | same |

**Totals: 39 registered transfers, 33 in the emission set. Two primitives have
both faces gauged.**

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
