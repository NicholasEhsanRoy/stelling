<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# Supported primitives

**GENERATED FILE — do not edit by hand.** This page is emitted by
`docs/gen_supported_primitives.py`, which imports stelling's live capability registries
and derives every membership, tier, count, and citation below from
them. The registries consumed, with their definition sites at
generation time:

- `stelling.propagate.TRANSFERS` (`src/stelling/propagate.py:2646`, 46 entries) — real-mode interval transfer registry (`semantics="real"`); each entry carries an assumption tier
- `stelling.propagate.IEEE_TRANSFERS` (`src/stelling/propagate.py:3816`, 46 entries) — ieee-mode interval transfer registry (`semantics="ieee"`); each entry carries an assumption tier
- `stelling.propagate._INT_COMPUTING` (`src/stelling/propagate.py:2817`, 17 entries) — transfer-side integer-semantics census: transfers that can compute a new numeric value (they carry the overflow-reachability guard)
- `stelling.propagate._INT_NON_COMPUTING` (`src/stelling/propagate.py:2868`, 29 entries) — transfer-side integer-semantics census: transfers recorded as computing no new value
- `stelling.propagate._INT_NON_COMPUTING_EXEMPT` (`src/stelling/propagate.py:2908`, 29 entries) — per-primitive written soundness reasons for the non-computing classification (reproduced in the appendix below)
- `stelling.propagate._ASSUME_CMPS` (`src/stelling/propagate.py:3971`, 5 entries) — the comparisons a point-bounded `stelling_assume` can narrow through
- `stelling.obligation._SUPPORTED` (`src/stelling/obligation.py:181`, 34 entries) — the SMT emission set: primitives an obligation slice may contain and emit
- `stelling.obligation._INT_OVERFLOW_EMITTED` (`src/stelling/obligation.py:201`, 10 entries) — emission-side integer-semantics census: emitted primitives that compute a new numeric value (integer dtypes decline)
- `stelling.obligation._INT_SAFE_EMITTED` (`src/stelling/obligation.py:225`, 24 entries) — emission-side integer-semantics census: emitted primitives recorded int-safe
- `stelling.obligation._INT_SAFE_EMITTED_REASONS` (`src/stelling/obligation.py:251`, 24 entries) — per-primitive written soundness reasons for the int-safe classification (reproduced in the appendix below)
- `stelling.obligation._REPLAY_SUPPORTED` (`src/stelling/obligation.py:2070`, 34 entries) — the exact-rational replay surface: primitives the solver-free witness replay can evaluate
- `stelling.obligation._SCALAR_STRUCT_FMT` (`src/stelling/obligation.py:143`, 12 entries) — the scalar literal decoder — keyed by numpy dtype code, not by primitive
- `stelling.coverage.DEFAULT_TRANSPARENT` (`src/stelling/coverage.py:54`, 4 entries) — call wrappers descended into (sub-jaxpr walked) instead of transferred

Regenerate with `python docs/gen_supported_primitives.py`. The drift gate
`tests/test_supported_primitives_doc.py` regenerates this page and fails
whenever the committed copy differs from the live registries.

## Per-primitive membership

One row per primitive named by any registry above (transparent call
wrappers are listed separately below). `—` means the primitive is not
a member of that registry.

| primitive | transfer (real) | transfer (ieee) | transfer int census | SMT emission | emission int census | replay | assume-narrowing |
|---|---|---|---|---|---|---|---|
| `abs` | exact | exact | computing | — | — | — | — |
| `add` | sound | exact | computing | emitted | overflow-guarded | replayed | — |
| `add_any` | sound | exact | computing | — | — | — | — |
| `and` | exact | exact | non-computing | emitted | int-safe | replayed | — |
| `broadcast_in_dim` | exact | exact | non-computing | emitted | int-safe | replayed | — |
| `concatenate` | exact | exact | non-computing | emitted | int-safe | replayed | — |
| `convert_element_type` | exact | exact | non-computing | emitted | int-safe | replayed | — |
| `copy` | exact | exact | non-computing | — | — | — | — |
| `div` | sound | exact | computing | emitted | overflow-guarded | replayed | — |
| `dot_general` | sound | exact | computing | emitted | overflow-guarded | replayed | — |
| `eq` | exact | exact | non-computing | emitted | int-safe | replayed | narrows |
| `exp` | sound-libm | sound-libm | computing | — | — | — | — |
| `gather` | exact | exact | non-computing | — | — | — | — |
| `ge` | exact | exact | non-computing | emitted | int-safe | replayed | narrows |
| `gt` | exact | exact | non-computing | emitted | int-safe | replayed | narrows |
| `integer_pow` | sound | exact | computing | emitted | overflow-guarded | replayed | — |
| `le` | exact | exact | non-computing | emitted | int-safe | replayed | narrows |
| `lt` | exact | exact | non-computing | emitted | int-safe | replayed | narrows |
| `max` | exact | exact | non-computing | emitted | int-safe | replayed | — |
| `min` | exact | exact | non-computing | emitted | int-safe | replayed | — |
| `mul` | sound | exact | computing | emitted | overflow-guarded | replayed | — |
| `ne` | exact | exact | non-computing | emitted | int-safe | replayed | — |
| `neg` | exact | exact | computing | emitted | overflow-guarded | replayed | — |
| `not` | — | — | — | emitted | int-safe | replayed | — |
| `or` | exact | exact | non-computing | emitted | int-safe | replayed | — |
| `pow` | sound-libm | sound-libm | computing | — | — | — | — |
| `reduce_or` | exact | exact | non-computing | — | — | — | — |
| `reduce_sum` | sound | exact | computing | emitted | overflow-guarded | replayed | — |
| `rem` | sound | sound | computing | — | — | — | — |
| `reshape` | exact | exact | non-computing | emitted | int-safe | replayed | — |
| `scatter` | exact | exact | non-computing | emitted | int-safe | replayed | — |
| `scatter-add` | sound | exact | computing | emitted | overflow-guarded | replayed | — |
| `select_n` | exact | exact | non-computing | emitted | int-safe | replayed | — |
| `sign` | sound | sound | computing | — | — | — | — |
| `slice` | exact | exact | non-computing | emitted | int-safe | replayed | — |
| `split` | exact | exact | non-computing | — | — | — | — |
| `sqrt` | sound | exact | computing | — | — | — | — |
| `square` | sound | exact | computing | emitted | overflow-guarded | replayed | — |
| `squeeze` | exact | exact | non-computing | emitted | int-safe | replayed | — |
| `stack` | exact | exact | non-computing | emitted | int-safe | replayed | — |
| `stelling_any` | exact | exact | non-computing | — | — | — | — |
| `stelling_assert` | exact | exact | non-computing | — | — | — | — |
| `stelling_assume` | — | — | — | emitted | int-safe | replayed | — |
| `stelling_nonvacuity` | exact | exact | non-computing | emitted | int-safe | replayed | — |
| `stop_gradient` | exact | exact | non-computing | — | — | — | — |
| `sub` | sound | exact | computing | emitted | overflow-guarded | replayed | — |
| `transpose` | exact | exact | non-computing | emitted | int-safe | replayed | — |
| `unstack` | exact | exact | non-computing | — | — | — | — |
| `xor` | — | — | — | emitted | int-safe | replayed | — |

## Transparent call primitives

`stelling.coverage.DEFAULT_TRANSPARENT` (4 members): `custom_jvp_call`, `custom_vjp_call`, `jit`, `remat2`.

Recorded role: "The wrapper primitives whose correct transfer is descend-into-sub-jaxpr, per design/transparent-primitives.md (verified on jax 0.10.2)." (src/stelling/coverage.py:52).

No member of this set appears in any of the registries above.

## Counts

All counts computed from the live registries at generation time.

- 49 primitives appear in at least one registry (transparent call wrappers counted separately).
- 46 primitives have a real-mode transfer: 31 `exact`, 13 `sound`, 2 `sound-libm`.
- 46 primitives have an ieee-mode transfer entry: 42 `exact`, 2 `sound`, 2 `sound-libm`.
- Transfer-side integer census: 17 computing, 29 non-computing, 29 written exemption reasons.
- 34 primitives are in the SMT emission set.
- Emission-side integer census: 10 overflow-guarded, 24 int-safe, 24 written int-safe reasons.
- 34 primitives are on the exact-rational replay surface.
- 5 comparisons can narrow through a constraining assume.
- 4 call wrappers are transparent.
- The scalar literal decoder covers 12 dtype codes (it is keyed by dtype, not by primitive).

## Where the sets differ

Every difference below is computed from the live registries. Reasons
are quoted verbatim from code comments/docstrings, with file:line;
where the code records no reason for a difference, the entry says
"no recorded reason".

### Real vs ieee transfer registry membership

The two transfer registries register exactly the same 46 primitives. This is enforced by an import-time
check whose recorded reason is: "the census must stay total: a registered transfer with no ieee census entry would be silent reuse, the exact thing rule 6 forbids" (src/stelling/propagate.py:3955).

### Real vs ieee assumption tiers

11 primitives carry different tiers in the two registries;
35 carry the same tier in both.

The module docstring's recorded rationale for the ieee endpoint
arithmetic is: "endpoint arithmetic for the monotone core is native binary64 with NO outward rounding (the float value itself is computable — :data:`stelling.interval.IEEE_ENDPOINT_ASSUMPTION`)" (src/stelling/propagate.py:119).

Per-primitive recorded reasons at the registry entries. Each
attached reason is checked at generation time for consistency with
the live ieee tier it is cited beside: a reason whose text asserts
a contradicting tier fails generation unless the row is presented
as a recorded discrepancy, and a discrepancy row whose recorded
text no longer contradicts the live tier fails generation too.

- `add` — real `sound`, ieee `exact`: "(ii) ieee variants: the monotone arithmetic core — native binary64 endpoints, NaN corners routed to the flag; the real-mode 0·∞ = 0 convention (iv._prod inside iv.mul) is NOT reused." (src/stelling/propagate.py:3817)
- `add_any` — real `sound`, ieee `exact`: "(ii) ieee variants: the monotone arithmetic core — native binary64 endpoints, NaN corners routed to the flag; the real-mode 0·∞ = 0 convention (iv._prod inside iv.mul) is NOT reused." (src/stelling/propagate.py:3817)
- `div` — real `sound`, ieee `exact`: "(ii) ieee variants: the monotone arithmetic core — native binary64 endpoints, NaN corners routed to the flag; the real-mode 0·∞ = 0 convention (iv._prod inside iv.mul) is NOT reused." (src/stelling/propagate.py:3817)
- `dot_general` — real `sound`, ieee `exact`: "dot_general under ieee: a whole-primitive censused REFUSAL." (src/stelling/propagate.py:3745)
- `integer_pow` — real `sound`, ieee `exact`: "(ii) censused down the same way: y in {0, 1} perform NO arithmetic and are exact (y=0 is measured 1.0 even at NaN, so it CLEARS the flag); every other exponent declines — no fixed multiply schedule" (src/stelling/propagate.py:3863)
- `mul` — real `sound`, ieee `exact`: "(ii) ieee variants: the monotone arithmetic core — native binary64 endpoints, NaN corners routed to the flag; the real-mode 0·∞ = 0 convention (iv._prod inside iv.mul) is NOT reused." (src/stelling/propagate.py:3817)
- `reduce_sum` — real `sound`, ieee `exact`: "(ii) censused DOWN to the association-free cases: <=2 contributors are exact (0 or 1 addition; IEEE add is commutative), >=3 declines — float addition is not associative and the jaxpr fixes no order" (src/stelling/propagate.py:3859)
- `scatter-add` — real `sound`, ieee `exact`: "Category (iii), whole-primitive: the censused ieee REFUSAL for scatter-add — the honest floor, chosen over an order-independent enclosure." (src/stelling/propagate.py:3731)
- `sqrt` — real `sound`, ieee `exact`: RECORDED DISCREPANCY — the registry comment reads "(ii) native binary64, CORRECTLY rounded — the float root bracketed exactly (no outward bump, tier sound not sound-libm); a negative arg is NaN routed to the flag, a maybe-NaN operand poisons the result" (src/stelling/propagate.py:3837), which asserts tier `sound`, contradicting the live tier `exact` carried by the `IEEE_TRANSFERS` entry it annotates. The comment is stale on main, flagged for correction; it is quoted here as recorded text, not as the reason for the live tier.
- `square` — real `sound`, ieee `exact`: "`square` under ieee — DECLINES, and the reason is not schedule ambiguity." (src/stelling/propagate.py:3534)
- `sub` — real `sound`, ieee `exact`: "(ii) ieee variants: the monotone arithmetic core — native binary64 endpoints, NaN corners routed to the flag; the real-mode 0·∞ = 0 convention (iv._prod inside iv.mul) is NOT reused." (src/stelling/propagate.py:3817)

### Emission set vs transfer registries

In the emission set but in neither transfer registry (3): `not`, `stelling_assume`, `xor`.

- `not` — no recorded reason for the absence of a transfer
- `stelling_assume` — "stelling_assume's *constraint* is inert (dropped, disclosed by the propagation notes) and is deliberately NOT emitted — only its data flow passes through, exactly as in propagation." (src/stelling/obligation.py:176); on the propagation side it is handled by the walk itself rather than through the transfer registry: "value semantics: the identity on the predicate — the assume's output passes its input through unchanged in BOTH modes" (src/stelling/propagate.py:5671)
- `xor` — no recorded reason for the absence of a transfer

In the transfer registries but not in the emission set (15): `abs`, `add_any`, `copy`, `exp`, `gather`, `pow`, `reduce_or`, `rem`, `sign`, `split`, `sqrt`, `stelling_any`, `stelling_assert`, `stop_gradient`, `unstack`.

An unsupported primitive in a slice declines with the message "primitive {prim!r} is outside the supported emission set" (src/stelling/obligation.py:1431). The module docstring's recorded decline classes are: "Everything else — over-budget slices, transcendentals, unknown primitives, possibly-zero divisor elements, non-float input declarations, obligations that cannot be mapped one-to-one onto top-level asserts — **declines**, with the primitive and form (and, for the budget, the count and the budget) quoted, and the obligation stays UNKNOWN." (src/stelling/obligation.py:36) Per-primitive recorded reasons:

- `abs` — no primitive-specific recorded reason
- `add_any` — no primitive-specific recorded reason
- `copy` — no primitive-specific recorded reason
- `exp` — no primitive-specific recorded reason
- `gather` — no primitive-specific recorded reason
- `pow` — no primitive-specific recorded reason
- `reduce_or` — no primitive-specific recorded reason
- `rem` — no primitive-specific recorded reason
- `sign` — no primitive-specific recorded reason
- `split` — no primitive-specific recorded reason
- `sqrt` — no primitive-specific recorded reason
- `stelling_any` — a slice endpoint, not an emitted equation: "extracts the *expression slice* — the ir equations from the ``stelling_any`` declarations and constants to the ``stelling_assert`` operand" (src/stelling/obligation.py:7); its elements become the SMT variables ("flattened topological order, sans stelling_any" (src/stelling/obligation.py:373))
- `stelling_assert` — a slice endpoint, not an emitted equation: "extracts the *expression slice* — the ir equations from the ``stelling_any`` declarations and constants to the ``stelling_assert`` operand" (src/stelling/obligation.py:7)
- `stop_gradient` — no primitive-specific recorded reason
- `unstack` — no primitive-specific recorded reason

### Emission set vs replay surface

The emission set and the replay surface are equal (34 primitives, both directions). The code records
this as an invariant: "Replay is what makes REFUTED self-certifying: a solver model is only ever promoted to a Witness after this module re-derives the violation in exact rational arithmetic, independently of the solver." (src/stelling/obligation.py:2052) And: "Measured 2026-07-26: the two sets are currently EQUAL, in both directions. That equality is an invariant to preserve, not a coincidence to note" (src/stelling/obligation.py:2059). It is asserted at import with the message "the exact-rational replay must cover exactly the emission set, or a witness can be produced that replay cannot independently confirm" (src/stelling/obligation.py:2078).

The replay path's scalar literal decoder is keyed by numpy dtype code
(`stelling.obligation._SCALAR_STRUCT_FMT`, 12 codes), not by primitive: "scalar decoders for size-1 ir.Array literals/consts (numpy dtype .str)" (src/stelling/obligation.py:142).

### The two integer-semantics censuses

Of the 31 primitives in both a transfer registry and the
emission set, 10 are classified computing AND
overflow-guarded, 21 non-computing AND int-safe, and
none is classified differently by the two censuses.

Where both apply, the recorded relationship between the two guards
is: "SMT-LIB2 Reals are unbounded; jax integers wrap. Emitting a computed integer as a Real would let the solver prove a claim the program falsifies, so integer dtypes decline here — the emission is stricter than the transfer on purpose." (src/stelling/obligation.py:1531) The transfer-side census's recorded charter is: "So the classification is mechanised instead of remembered. Every registered transfer is either COMPUTING — it can produce a numeric value its operands did not contain, so it carries the overflow-reachability guard — or NON-COMPUTING, with the reason recorded here." (src/stelling/propagate.py:2811) The emission-side census's totality rule is: "Every emittable primitive is classified, and the union must be total over `_SUPPORTED`." (src/stelling/obligation.py:222).

Censused on the transfer side only (not emitted, 15): `abs`, `add_any`, `copy`, `exp`, `gather`, `pow`, `reduce_or`, `rem`, `sign`, `split`, `sqrt`, `stelling_any`, `stelling_assert`, `stop_gradient`, `unstack`. Censused on the emission side only (no transfer, 3): `not`, `stelling_assume`, `xor`.

### Assume-narrowing comparisons vs the comparison set

Of the 6 emitted comparisons, 5 can narrow through a constraining assume; not narrowing: `ne`.

For `ne` the recorded reason is: "`ne` is a comparison but NOT here: excluding a single point from an interval does not narrow it (the hull of [lo, hi] \ {k} is [lo, hi]) — it stays inert with the reason quoted." (src/stelling/propagate.py:3967).

## Recorded classification reasons

The two reason registries, reproduced verbatim from the live dicts.

### Emission int-safe reasons (`stelling.obligation._INT_SAFE_EMITTED_REASONS`, `src/stelling/obligation.py:251`)

| primitive | recorded reason |
|---|---|
| `and` | boolean connective on Bool terms (non-bool operands decline in _validate) |
| `broadcast_in_dim` | emits NO terms: output elements alias source terms (index routing only) |
| `concatenate` | emits NO terms: output elements alias source terms (index routing only) |
| `convert_element_type` | whitelist-guarded in _validate: only value-preserving conversions emit (identity or the bool->{0,1} ite) |
| `eq` | emits a comparison; exact over Reals for in-range integers, result sort Bool |
| `ge` | emits a comparison; exact over Reals for in-range integers, result sort Bool |
| `gt` | emits a comparison; exact over Reals for in-range integers, result sort Bool |
| `le` | emits a comparison; exact over Reals for in-range integers, result sort Bool |
| `lt` | emits a comparison; exact over Reals for in-range integers, result sort Bool |
| `max` | emits an ite SELECTING an operand term; no arithmetic term is created |
| `min` | emits an ite SELECTING an operand term; no arithmetic term is created |
| `ne` | emits a comparison; exact over Reals for in-range integers, result sort Bool |
| `not` | boolean connective on Bool terms (non-bool operands decline in _validate) |
| `or` | boolean connective on Bool terms (non-bool operands decline in _validate) |
| `reshape` | emits NO terms: output elements alias source terms (index routing only) |
| `scatter` | the static-index SET form emits NO term: element k's term IS the update's and every other element's IS the operand's, so the equation performs pure data movement. Nothing is computed, so nothing can overflow an integer dtype — unlike scatter-add, whose accumulate is Real addition and therefore carries the guard. The operand/updates/output dtypes are required to agree, so the aliasing never equates values of different sorts |
| `select_n` | emits an ite selecting a case term; no arithmetic term is created |
| `slice` | emits NO terms: output elements alias source terms (index routing only) |
| `squeeze` | emits NO terms: output elements alias source terms (index routing only) |
| `stack` | emits NO terms: output elements alias source terms (index routing only) |
| `stelling_assume` | identity data flow; the constraint is deliberately never emitted |
| `stelling_nonvacuity` | identity data flow |
| `transpose` | emits NO terms: output elements alias source terms (index routing only) |
| `xor` | boolean connective on Bool terms (non-bool operands decline in _validate) |

### Transfer non-computing exemptions (`stelling.propagate._INT_NON_COMPUTING_EXEMPT`, `src/stelling/propagate.py:2908`)

| primitive | recorded reason |
|---|---|
| `and` | bool-only by its own dtype guard (the bitwise integer form declines inside the transfer); Kleene logic on {0, 1} |
| `broadcast_in_dim` | pure element routing (replication) — copies of in-range values, no arithmetic performed on them |
| `concatenate` | pure element routing (adjacency) — copies of in-range values, no arithmetic performed on them |
| `convert_element_type` | carries its own exact-conversions whitelist plus the float->int range guard and the interval-based int64->int32 index-narrowing range check; every value-changing conversion declines |
| `copy` | the identity primitive: its output IS its input, so no arithmetic occurs and an in-range integer cannot leave the range by being copied |
| `eq` | produces booleans; comparison of in-range integers is exact and its result dtype cannot wrap |
| `gather` | pure element routing (static row take) — copies of in-range values, no arithmetic performed on them |
| `ge` | produces booleans; comparison of in-range integers is exact and its result dtype cannot wrap |
| `gt` | produces booleans; comparison of in-range integers is exact and its result dtype cannot wrap |
| `le` | produces booleans; comparison of in-range integers is exact and its result dtype cannot wrap |
| `lt` | produces booleans; comparison of in-range integers is exact and its result dtype cannot wrap |
| `max` | selects one of its operands' values elementwise; no arithmetic creates a value its in-range operands did not already contain |
| `min` | selects one of its operands' values elementwise; no arithmetic creates a value its in-range operands did not already contain |
| `ne` | produces booleans; comparison of in-range integers is exact and its result dtype cannot wrap |
| `or` | bool-only by its own dtype guard (the bitwise integer form declines inside the transfer); Kleene logic on {0, 1} |
| `reduce_or` | bool-only by its own dtype guard; a three-valued OR-fold whose outputs are booleans |
| `reshape` | pure element routing (flat C-order identity) — misclassification would require the routing kernel itself to compute, which it cannot: it only copies in-range values |
| `scatter` | element REPLACEMENT (the set form): the output holds only values its operand and update already contained; the accumulate sibling 'scatter-add' computes and is probed |
| `select_n` | selects/joins operand values elementwise (measured clamp semantics); it computes no new numeric value |
| `slice` | pure element routing (static selection) — copies of in-range values, no arithmetic performed on them |
| `split` | cuts one operand along one axis at statically known offsets; every output element IS an input element, so no arithmetic occurs and an in-range integer cannot be moved out of range by being copied |
| `squeeze` | pure element routing (axis removal) — copies of in-range values, no arithmetic performed on them |
| `stack` | pure element routing (new-axis join) — copies of in-range values, no arithmetic performed on them |
| `stelling_any` | a declaration: its output box IS the declared bounds; nothing is computed from operand values |
| `stelling_assert` | the identity on its predicate operand |
| `stelling_nonvacuity` | the identity on its membership operand |
| `stop_gradient` | the identity on its operand |
| `transpose` | pure element routing (axis permutation) — copies of in-range values, no arithmetic performed on them |
| `unstack` | routes each index along one axis to its own output; every output element IS an input element, so no arithmetic occurs |

