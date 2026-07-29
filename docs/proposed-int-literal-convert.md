<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# `int64→float64` from an integer literal — PROPOSED, NOT BUILT

A fourth member of the fixed-width family, which the family table named as
three. **The family is parked, and `2**53` is a numeric constant, so this is a
proposal.**

## What it blocks, measured

`jax_md/util.py:86`:

```python
def safe_mask(mask, fn, operand, placeholder=0):
  masked = jnp.where(mask, operand, 0)          # <-- the Python int
  return jnp.where(mask, fn(masked), placeholder)
```

The literal `0` promotes through `convert_element_type int64→float64`, which
declines, which ⊤s the enclosing `_where` — and it is in the **body**, so all
**41** `safe_mask` call sites inherit it regardless of the placeholder the
caller passes. Isolated:

```
jnp.where(x > 1.0, x, 0)        ->  TOP          <-- one character
jnp.where(x > 1.0, x, 0.0)      ->  [0, 12]
jnp.where(x > 1.0, x, f64(0.0)) ->  [0, 12]
astype(int64 -> float64)        ->  TOP          (the transfer returns None)
```

`int64→float64` was already on the measured external-terminal list in
[state-0.1.0.md](state-0.1.0.md). This is its mechanism.

## Why the current decline is right, and narrow

`_EXACT_CONVERSIONS` contains `("int32","float64")` and not
`("int64","float64")`, which is **correct**: int64 spans values beyond
float64's exactly-representable integer range, so the conversion is not exact
in general.

But the test is **set membership on the dtype pair**, while the `float→int`
branch immediately below it does a **range check** on the operand's interval:

```python
if src == dst or (src, dst) in _EXACT_CONVERSIONS:
    return [a]
if _in_range_int_narrowing(a, src, dst):
    return [a]
if "float" in src and dst in _INT_RANGE:
    bound = _INT_RANGE[dst]
    if any(not (-bound <= x < bound) for x in (*a.los, *a.his)):
        return None
    ...
```

So the shape of the fix already exists in the same function, one direction
over. **An int→float widening is exact exactly when the operand's interval
lies inside the target float's exactly-representable integer range.**

## The narrowest rule, which is what is proposed

Not "int64→float64 when the box is inside ±2⁵³" — narrower:

> **an integer-dtyped operand whose interval is a POINT, at a value the target
> float represents exactly.**

The 41 sites are the literal `0`. A point interval needs no reasoning about
accumulated width, no interaction with the overflow guard, and no claim about
what a wide integer box does. It is the smallest change that closes the
measured blockage, and it leaves the general range rule available later as a
separate, separately-argued member.

If the wider rule is wanted, it is the same clause with the point test replaced
by an interval test, and it needs `2**53` — **a numeric constant, with a
derivation, read at its definition site.** That is the part that is not mine.

## What this does NOT claim

- **It clears 0 of the 14 internal contracts.** No MADDENING or MIME contract
  reaches it. Its entire measured demand is external.
- **It does not fix `div`.** It removes the ⊤ *upstream* of the `div` decline
  at jax-md's `safe_mask` sites. Whether the divisor's box is then narrow
  enough to discharge is **unmeasured** — the where-correlation question sits
  behind this one, and site 1 of the `div` table is where that is measured.
- **It has not been run against the external harnesses.** Re-running the
  external agents against the fixed tool is held, and it is the measurement
  that would price this properly.
