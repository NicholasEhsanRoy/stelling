<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# `int64→float64` from an integer literal — **BUILT** (`cbb1d60`)

**Status: BUILT and shipped in `cbb1d60`** ("Phase-1 transfers: is_finite,
int64->float64 point rule, solver kwarg"). The rule this page proposes at
"What the rule would be" is the rule in the tree — `_t_convert` passes an
`int64` operand whose interval is a POINT at a value `float64` represents
exactly — and the `2**53` bound the page calls "the part that is not mine" is
the constant it uses. Pinned by `tests/test_phase1_transfers.py`, whose
`test_int64_to_float64_non_point_still_declines` holds the narrow half: a
non-point `int64` interval still declines, which is what makes the rule the
narrow one this page argues for.

This page kept a `PROPOSED, NOT BUILT` header through the change that built
it, which is the defect `proposed-declaration-dtype-check.md` records and
names: a claim divergence on a DOCUMENT. Corrected here rather than quietly
retitled. The argument below is unchanged and is why the rule is the **narrow**
one; what has moved is the measurement it argued from, marked in place.

A fourth member of the fixed-width family, which the family table named as
three. **The family is parked, and `2**53` is a numeric constant, so this is a
proposal.**

## What it blocks, measured

`jax_md/util.py:86`:

<!-- doc-example: illustrative -->
```python
def safe_mask(mask, fn, operand, placeholder=0):
  masked = jnp.where(mask, operand, 0)          # <-- the Python int
  return jnp.where(mask, fn(masked), placeholder)
```

The literal `0` promotes through `convert_element_type int64→float64`, which
declines, which ⊤s the enclosing `_where` — and it is in the **body**, so all
**41** `safe_mask` call sites inherit it regardless of the placeholder the
caller passes. Isolated, **as it stood before `cbb1d60`**:

```
jnp.where(x > 1.0, x, 0)        ->  TOP          <-- one character
jnp.where(x > 1.0, x, 0.0)      ->  [0, 12]
jnp.where(x > 1.0, x, f64(0.0)) ->  [0, 12]
astype(int64 -> float64)        ->  TOP          (the transfer returns None)
```

> **The first row has since closed, and it is the row this whole page is
> about.** `cbb1d60` gave `convert_element_type` the point rule proposed
> below, so a Python-int placeholder no longer ⊤s the enclosing `_where`. The
> block that prints the current reading is at the end of this page and is
> byte-compared on every run. The paragraph above is kept as measured because
> the mechanism it describes — a placeholder that promotes through a declining
> convert — is what the rule was built to close, and the fence is what it
> looked like.
>
> **What is NOT re-measured is the consequence.** "all **41** `safe_mask` call
> sites inherit it" was a count over `jax_md`, which is not a dependency of
> this repository, and the sentence's stated mechanism ("they stop at the
> convert first") no longer fires on this page's own isolated reduction.
> Whether those 41 sites now clear is **unmeasured** — it needs the `jax_md`
> sweep `docs/state-0.1.0.md` says is not re-derivable here. The 21-of-41
> arithmetic on that page is unaffected; only the mechanism claim is.

`int64→float64` was already on the measured external-terminal list in
[state-0.1.0.md](state-0.1.0.md). This is its mechanism.

## Why the current decline is right, and narrow

`_EXACT_CONVERSIONS` contains `("int32","float64")` and not
`("int64","float64")`, which is **correct**: int64 spans values beyond
float64's exactly-representable integer range, so the conversion is not exact
in general.

But the test is **set membership on the dtype pair**, while the `float→int`
branch immediately below it does a **range check** on the operand's interval:

<!-- doc-example: illustrative -->
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

## What shipped, and this block prints it

`cbb1d60` built the narrow rule argued for above. The block below runs under
`tests/test_doc_examples.py` and its output fence is compared byte for byte,
so the four readings this page reasons from are re-derived on every run
instead of being typed once and left:

```python
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from stelling._jax_compat import trace
from stelling.harness import any_array
from stelling.propagate import _EXACT_CONVERSIONS, interval_env


def box_of(query):
    closed = trace(query)
    out = closed.jaxpr.outvars[-1]
    return interval_env(closed)[out.id]


def where_with(placeholder):
    def query():
        x = any_array((1,), jnp.float64, (0.0, 12.0))
        return jnp.where(x > 1.0, x, placeholder)
    return query


def astype_from_int64():
    return any_array((1,), jnp.int64, (0, 12)).astype(jnp.float64)


for label, ph in [("jnp.where(x > 1.0, x, 0)       ", 0),
                  ("jnp.where(x > 1.0, x, 0.0)     ", 0.0),
                  ("jnp.where(x > 1.0, x, f64(0.0))", jnp.float64(0.0))]:
    b = box_of(where_with(ph))
    print(f"{label} ->  [{b.los[0]}, {b.his[0]}]")
b = box_of(astype_from_int64)
print(f"astype(int64 -> float64)        ->  [{b.los[0]}, {b.his[0]}]")
print(f'("int64","float64") in _EXACT_CONVERSIONS: '
      f'{("int64", "float64") in _EXACT_CONVERSIONS}')
```

```
jnp.where(x > 1.0, x, 0)        ->  [0.0, 12.0]
jnp.where(x > 1.0, x, 0.0)      ->  [0.0, 12.0]
jnp.where(x > 1.0, x, f64(0.0)) ->  [0.0, 12.0]
astype(int64 -> float64)        ->  [-inf, inf]
("int64","float64") in _EXACT_CONVERSIONS: False
```

**Both halves of the argument survive, and that is the point of printing them
together.** The one-character difference is gone — all three placeholders give
`[0, 12]` — while the non-point `astype` still declines to ⊤ and
`_EXACT_CONVERSIONS` still, correctly, does not contain `("int64","float64")`.
The rule that shipped is the *point* rule, not a dtype-pair admission, which
is exactly the narrowness the section above argues for.
