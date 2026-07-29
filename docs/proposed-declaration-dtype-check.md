<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# `any_array` should reject bounds its dtype cannot hold — PROPOSED, NOT BUILT

Published surface, so this is the argument and not the change.

## The hole

`any_array` validates shape, bound ordering, and infinite-point emptiness. It does
**not** validate the bounds against the declared dtype. So this is accepted:

```python
x = any_array((1,), "uint8", (-3.0, -1.0))     # no uint8 holds -3
```

A declared box that **no execution of the program can inhabit**, admitted as a
declaration, is then propagated as fact.

## What it cost, measured

`sign` on that box returned `[-1, -1]` — outside uint8 — at **100% coverage, with
no note**, and the obligation came back `violated-over-set`: a **REFUTED**, the
campaign's strongest output shape, minted from a premise the program cannot reach.

Three instruments nominally covered `sign`'s integer safety and **none could
fail**: `_probe_operands` feeds it uint 0/255, the import-time behavioural census
only range-checks those operands, and the guard test is int32-only.

## It is not a `sign` bug — 13 of 21 transfers admit it

Routing `sign` through `_int_overflow_guard` fixed `sign`. It fixed one row.
Driving the same dtype-impossible `uint8` box `(-3, -1)` through every transfer
that accepts integers:

| transfer | returns | |
|---|---|---|
| `lt`, `le`, `ne` | **definite TRUE** | a definite boolean straight into an assert |
| `gt`, `ge`, `eq` | **definite FALSE** | same |
| `neg` | `[1, 3]` | positive values, from negating impossible negatives |
| `square`, `integer_pow` | `[1, 9]` | |
| `min` | `[-3, -1]` | out of range, propagated |
| `copy`, `stop_gradient` | `[-3, -1]` | out of range, propagated |
| `max` | `[0, 255]` | in range, but on a false premise |
| `add`, `sub`, `mul`, `div`, `rem`, `reduce_sum`, `sign` | decline | the overflow guard catches these |

**13 of 21 admit. The six comparisons are the dangerous ones**, because they mint a
definite boolean with nothing downstream to widen it.

## Why the fix belongs at the declaration

This is the same shape as the `from_dict` load-path refusal: **reject at
construction what the primitive could never produce.** Every alternative is worse:

- *Guard each transfer.* That is 21 sites for one invariant, and the count only
  grows. It is also what was tried — `sign`'s comment asserted the invariant
  (*"a uint box has lo >= 0, so the negative cases are unreachable"*) instead of
  enforcing it, and the comment was simply false.
- *Guard at the propagator.* Better, but it fires after the declaration has been
  accepted, so the error names an equation rather than the line the user wrote.
- *Leave it.* The failure mode is a REFUTED with a witness the user cannot
  reproduce, which is the most expensive wrong answer the tool has.

The declaration is the only place that knows both the dtype and the bounds, and
it is where the user can act on the message.

## The fourth way a declaration can be wrong

Recorded in that class deliberately — it is the fourth, after:

1. **deserialization corruption** (the `from_dict` load path),
2. **dropped-assume widening** (an `assume` whose predicate box is ⊤ is dropped,
   and the query answers the unconditional question),
3. **correctly-declared-but-unreachable** (a box the program never occupies),
4. **dtype-impossible bounds** — this one.

**It is the only one of the four that mints a REFUTED.** The other three cost a
VERIFIED that means less than it looks; this one manufactures a counterexample.

## Scope of the proposal

Reject at declaration when a bound lies outside the declared dtype's
representable range. For integers that is exact and cheap — `_INT_DTYPE_BOUNDS`
already holds the table. For floats it is the overflow boundary
(`float32(1e39)` is `inf`), which is the same check and closes the adjacent case
a blinded audit raised: an f32 declaration of `(1e39, 1e40)` currently passes the
dividend guard in `rem` because that guard tests the **binary64** endpoints.

**Not proposed here:** rejecting bounds that are merely *unreachable* rather than
unrepresentable. That is case 3 above, it is undecidable in general, and
conflating the two would make the declaration layer a reachability analysis.
