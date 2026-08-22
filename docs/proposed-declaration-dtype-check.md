<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# `any_array` rejects bounds its dtype cannot hold — **BUILT**

**Status: BUILT and shipped in `89413c2`.** This file was written as a proposal
and kept its `PROPOSED, NOT BUILT` header through the change that implemented
it — a **claim divergence on a DOCUMENT**, which is a new surface for that
class: the norm was written about code being narrower or wider than what it
says, and a status header is the same failure with a shorter path to a reader.
Corrected here rather than quietly retitled, because the divergence is the
point. The argument below is unchanged; only the status was wrong.

Pinned by `tests/test_declaration_dtype.py`, which drives the refusal over
every dtype, asserts the nearest-representable values it names, and carries
its own load-bearing control (`test_the_check_is_load_bearing`) — so this
page's BUILT is a claim about running code, not a word.

## The hole

`any_array` validates shape, bound ordering, and infinite-point emptiness. It does
**not** validate the bounds against the declared dtype. So this is accepted:

<!-- doc-example: illustrative -->
```python
x = any_array((1,), "uint8", (-3.0, -1.0))     # no uint8 holds -3
```

A declared box that **no execution of the program can inhabit**, admitted as a
declaration, is then propagated as fact.

> Tense note, since the argument is preserved as written: that declaration is
> **no longer accepted**. This tree refuses it at declaration time, naming the
> dtype and the empty interval —
> [measured](harness-api.md#any_arrayshape-dtype-bounds). The block above is
> the hole as it stood, not as it stands.

## What it cost, measured

`sign` on that box returned `[-1, -1]` — outside uint8 — at **100% coverage, with
no note**, and the obligation came back `violated-over-set`: a **REFUTED**, the
campaign's strongest output shape, minted from a premise the program cannot reach.

Three instruments nominally covered `sign`'s integer safety and **none could
fail**: `_probe_operands` feeds it uint 0/255, the import-time behavioural census
only range-checks those operands, and the guard test is int32-only.

## It is not a `sign` bug — a MAJORITY of transfers admit it

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

**The six comparisons are the dangerous ones**, because they mint a definite
boolean with nothing downstream to widen it.

**The count is convention-dependent and the first version of this table stated
none**: the 13 above take a valid full-range box as the second operand; driving
the impossible box on *every* operand gives 20 admits of 26 traceable rows. A
blinded audit found that "13 of 21" also used 21 as a denominator when it is
itself an admit count. **What is robust is the shape, not the number.**

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

**Way #2 also mints a REFUTED** — an `assume` whose predicate box is ⊤ is
dropped, and the measured instance returns a replay-confirmed witness violating
the precondition the author wrote. The first version of this line claimed #4 was
the only one, contradicting a measurement recorded twenty lines away in
`SOUNDNESS.md`. **What distinguishes #4 is narrower:** the other three answer a
DIFFERENT question than the author asked; this one answers **no question at
all**, because the declared set is empty and the witness cannot be constructed
at any dtype.

## The rule AS BUILT — and it is not the rule this section first proposed

**A second divergence in this same document, and the more serious one**: the
paragraph here originally read *"reject when a bound lies outside the declared
dtype's representable range."* **That is not what shipped, and it would have
been wrong.** A box wider than the dtype is an OVER-approximation — every
transfer's answer over it still contains the executed value — so rejecting it
would refuse a legitimate envelope, which is the failure this check must not
have.

**What shipped is: reject when the interval contains NO representable value.**

| declared | verdict | why |
|---|---|---|
| `uint8 (-3, -1)` | **REJECT** | holds nothing |
| `uint8 (-3, 10)` | admit | partial overlap → over-approximation, sound |
| `int8 (-200, 200)` | admit | wider than int8, same reason |
| `int8 (0.2, 0.8)` | **REJECT** | contains no integer |
| `float32 (0.0, 1e39)` | admit | clamps into range and holds float32s |
| `float32 (1e-50, 1e-49)` | **REJECT** | inside a representation gap, below the smallest subnormal |
| `float32 (1e39, 1e40)` | **REJECT** | entirely above float32's finite max |
| `float32 (0.1, 0.1)` | **REJECT** | holds no float32 — an empty set like any other |
| `complex*` | admit | undefined what a real box means; not settled by refusing |

Ranges come from `_INT_DTYPE_BOUNDS` first (it carries `bool`) and then from
**jax itself** — `jnp.iinfo` and `jnp.finfo`, which between them cover 29 of
the 30 dtypes jax builds arrays in. That is genuinely a second source, not one
registry: this layer knows `int2`/`uint2` while the transfer layer declines
them, a coverage disagreement in the safe direction (declare-admit →
transfer-decline), with the values agreeing on all 11 shared dtypes.

Measured against every literal declaration in the campaign corpus — **105 of
them, top-level `.py` only; 131 recursively — zero are refused** under the
exact rule.

**Exact representability IS checked** — reversed from this document's first
version, after measurement. Range-only admitted an interval lying wholly inside
a representation gap (`float32 (1e-50, 1e-49)`), which reached a REFUTED at
100% coverage. No rule admits that and rejects `float32 (0.1, 0.1)`: both hold
no value of the dtype. The trade resolves asymmetrically — admitting an empty
point costs nothing, admitting the gap mints a false counterexample — and
refusing a point declaration at a decimal literal is not over-strict, it tells
the caller their declaration does not mean what they think. **The message names
the nearest representable neighbours** so the caller learns what to write
instead. `float64` is untouched: every python float IS a float64.

**Deliberately NOT checked:**

- **bounds that are merely UNREACHABLE rather than unrepresentable.** That is
  case 3 above, it is undecidable in general, and conflating the two would make
  the declaration layer a reachability analysis.
- **complex**, admitted unconditionally — what a real-bounded box means for a
  complex array is undefined, and this check does not settle it by refusing.
