# Maintenance treadmill — how fast the ground moves under a jaxpr-level tool

**Status:** evidence artifact, first measurement 2026-07-17. Method: on
each new jax series, re-run the census (`corpus/run_census.py`) and the
full suite, and diff the primitive profile and transcription behaviour
against the previous series. Append one row per bump — a rate needs two
points, and every future bump extends this for free. Nobody has published
this number for JAX; the value model currently prices this cost at zero.

## Bump 1: jax 0.10.2 → 0.11.0 (measured 2026-07-17, within a day of 0.11.0 landing)

Same corpus, same harnesses; both runs complete (8/8 public harnesses and
the held-out arm, on both series).

**Primitive-name level** (what the census table sees):

- appeared: `unstack` ×1 (blackjax HMC path)
- disappeared: none
- counts moved: `slice` 48→46, `squeeze` 12→10 — the `slice`+`squeeze`
  pattern became `unstack`: a *lowering* change, not a code change
  (blackjax 97→94 eqns)
- totals: 1823 → 1820 equations; 73 → 74 primitives

**Param level** (invisible to the count table; where the maintenance
actually lives):

- `scan` gained `ft_in`/`ft_out` — a new `flattree.FTTuple` structural
  type (the flat-tree redesign arriving in jax) — and **lost**
  `num_consts`/`num_carry`/`linear`. Any consumer relying on scan's carry
  layout breaks; our interrogation classifier now reports
  `layout-unknown` on 0.11 rather than guessing.
- `Jaxpr` and `ClosedJaxpr` **merged into one class**: isinstance
  distinctions across that boundary silently changed meaning (caught by a
  test that pinned 0.10 behaviour; made series-tolerant).
- `convert_element_type.new_dtype` is now a `numpy.dtypes.*` instance
  (benign — still an `np.dtype` subclass).

**Fence performance — second and third live tests.** The unknown-param
raise fired on `ft_in` exactly as designed: loud, naming the primitive and
the param. (Its first live test was the accidental 0.5.3 exposure, which
surfaced the old `custom_vjp_call_jaxpr` primitive name the same way —
renames across series gaps are real.) And the bump found a hole in a fence
itself: the untested-series warning is once-per-version, so the
series-friction test passed whenever any earlier test had already consumed
the warning — a test-order-dependent fence, fixed with a cache clear.
Fences need fences.

**Ecosystem lag: zero.** All seven corpus libraries installed and traced
on 0.11.0 on day one.

**Cost of this bump:** one 4-line transcription rule (FTTuple → structural
text), two test adjustments, one interrogation guard; under an hour
wall-clock including re-verification of both series. `TESTED_JAX_SERIES`
is now `("0.10", "0.11")`.

## The rate

| bump | new prims | removed | param changes needing rules | class-identity changes | compat LOC | ecosystem lag |
|---|---|---|---|---|---|---|
| 0.10 → 0.11 | 1 | 0 | 1 (`FTTuple`) | 1 (Jaxpr merge) | ~4 | none observed |

One point is a weak slope; it replaces a guess. Carry it into the value
model's cost side, and extend it at every bump.
