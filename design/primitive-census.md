# Primitive census — scientific JAX corpus

**Status:** evidence artifact, run 2026-07-17.
jax 0.10.2, stelling 0.1.0, harness method:
hand-written minimal harnesses per library, each targeting the core
computational path (solver loop / sampler step / stencil kernel), traced
whole with `jax.make_jaxpr` and counted at every depth by
`stelling.census` (registry-independent). Re-verify trigger: any jax
series bump, or any corpus addition. **An undated census is a rumour;
this one will go stale and says when it was taken.**

> **This is an inventory, not a value signal.** It makes no claim about
> whether stelling finds bugs, and it is not pre-registered because it
> claims nothing (`design/value-model.md`). The moment a number below is
> cited as evidence that stelling is useful, the census has become the
> thing it isn't. Its two jobs: set the transfer-registry priority
> order, and date-stamp what the ecosystem's code is made of.

## Corpus

| target | version | harness (core path) | eqns |
|---|---|---|---|
| diffrax | 0.7.2 | adaptive Tsit5 solve of Lotka-Volterra via diffeqsolve (PID controller) | 920 |
| optimistix | 0.1.0 | BFGS minimisation of 4-D Rosenbrock via optx.minimise | 240 |
| lineax | 0.1.1 | dense LU linear solve via lx.linear_solve | 10 |
| numpyro | 0.21.0 | gradient of Bayesian-regression log density (the HMC core computation) | 61 |
| blackjax | 1.6.2 | one HMC step (init + leapfrog + accept) on a Gaussian target | 97 |
| jax-md | 0.2.29 | soft-sphere pair energy over periodic space (pairwise kernel) | 69 |
| jax-md | 0.2.29 | soft-sphere energy via neighbor list + neighbor update (indexing path) | 279 |
| jax-cfd | 0.2.1 | van Leer advection of a scalar on a 16x16 staggered grid | 147 |

Corpus scope: mature maintained libraries only. The research-code arm
is **not represented** — and growth is re-aimed accordingly: since the
schedule below is ordered by cost rather than count, saturation no
longer drives corpus growth. The reason to grow is that **the research
arm is the unprimed arm**: every current target is a mature library
written by people who know exactly where the gather clamps, and every
harness was written by someone who knew what he hoped to find.
Interception on research code is the only sample with neither
property — a value-model instrument, not a ranking refinement.

## Saturation

Criterion: add targets until the top-10 ranking stops reordering. Status: **NOT saturated — the ranking was still reordering as targets were added; the per-primitive ordering is provisional** (which constrains the *inventory*, not the schedule — see below).

- after `diffrax`: first entry
- after `optimistix`: reordered
- after `lineax`: reordered
- after `numpyro`: reordered
- after `blackjax`: reordered
- after `jax-md`: reordered
- after `jax-cfd`: reordered

## Wedge-relevant primitives (gather / scatter\* / dynamic\_slice / dynamic\_update\_slice)

> **Read these rows with the harnesses in mind.** Wedge counts are a
> property of the paths the harnesses exercised, not of the libraries:
> paths not traced contribute nothing, and indexing-heavy paths (e.g.
> jax-md neighbor lists) appear only insofar as a harness reaches them.
> A low count here is a fact about this corpus and these harnesses. It
> cannot support "the bug class is rare" — the same way no number in
> this file may support "stelling is useful."
>
> **The symmetry cuts both ways.** The high counts are harness-shaped
> too, and in a worse way: `gather` ×7 exists because we went looking
> for it, knowing the neighbor list was the value model's canonical
> suspect. A census can launder a prior about where bugs live into
> apparent evidence; these rows are where that would happen. Read them
> as *where we looked*, never as *where bugs are*. Interrogation of
> these rows: `design/census-interrogation.md`.

- `scatter`: 5 eqns across 2/7 targets
- `gather`: 7 eqns across 1/7 targets
- `dynamic_slice`: 4 eqns across 1/7 targets
- `dynamic_update_slice`: 2 eqns across 1/7 targets
- `scatter-add`: 1 eqns across 1/7 targets
## Schedule — the inventory is not the build order

Ranked by **cost**, not count. Tiers 0–1 — transparency (already
done), structural/identity ops, elementwise one-liners, and
fixed-shape folds — cover **1590/1823 equations
(87%)** with a registry of
near-one-liners. They do not depend on saturation and will not
reorder out of the top: build them first. The hard decisions live in
tiers 3–4; the census informs *which* of them matter, not *when*.

| tier | eqns | % | primitives present in this corpus |
|---|---|---|---|
| 0 — transparent (already done) | 333 | 18% | `custom_jvp_call`, `custom_vjp_call`, `jit` |
| 1a — free: structural/identity | 377 | 21% | `broadcast_in_dim`, `concatenate`, `copy`, `iota`, `pad`, `reshape`, `slice`, `split`, `squeeze`, `stack`, `stop_gradient` |
| 1b — free: elementwise/compare/join one-liners | 855 | 47% | `abs`, `add`, `add_any`, `and`, `convert_element_type`, `div`, `eq`, `gt`, `integer_pow`, `le`, `lt`, `max`, `min`, `mul`, `ne`, `neg`, `not`, `or`, `rem`, `select_n`, `sign`, `sqrt`, `sub` |
| 1c — free: fixed-shape reduction folds | 25 | 1% | `cumsum`, `reduce_and`, `reduce_max`, `reduce_or`, `reduce_sum` |
| 2 — medium: bilinear/permutation | 22 | 1% | `dot_general`, `pow`, `sort` |
| 3 — wedge targets (the Stage-1 work itself) | 19 | 1% | `dynamic_slice`, `dynamic_update_slice`, `gather`, `scatter`, `scatter-add` |
| 4 — control flow (few eqns gate hundreds nested) | 12 | 1% | `cond`, `scan`, `while` |
| 5 — float-boundary (see the nextafter note) | 109 | 6% | `bitcast_convert_type`, `erf_inv`, `exp`, `is_finite`, `log`, `nextafter` |
| 6 — PRNG / bit-level (bounded-adversary story) | 9 | 0% | `random_bits`, `random_split`, `random_unwrap`, `random_wrap`, `shift_right_logical` |
| 7 — library-defined (open primitive set — design/open-primitive-set.md) | 59 | 3% | `linear_solve`, `maybe_set`, `nonbatchable`, `nondifferentiable_backward`, `select_if_vmap`, `unvmap_any`, `unvmap_max` |
| 8 — escape hatches (⊤ forever) | 2 | 0% | `pure_callback` |
| 9 — dense linear algebra (contract-level treatment later) | 1 | 0% | `lu` |

## Notes

**`nextafter` ×100 — the float boundary is on page one.** In ℝ,
`nextafter(x, y)` *is* `x`: the primitive has no real-arithmetic
content, and the first census of the ecosystem's best ODE library put
it at rank six by count (diffrax's step-size controller works in
ulps). Its transfer function must be float-aware even under the
ℝ-with-margin semantics — bound it as x ± ulp(x), monotone, easy —
and robust invariants with slack ≫ ulp absorb it, so this vindicates
design commitment 2 rather than threatening it. But the founding doc
framed ℝ-vs-IEEE as a Stage-2 decision, and the swamp showed up in
the first inventory. Noticed deliberately, here.

## Held-out arm — censused, compared, never voting

The priority order above is set by the public corpus **only**. The
arms below are the maintainer's own code: included in the ranking
they would train the registry on its author's test set, so they
are held out and asked a different question — *is their profile
covered by the order the public corpus produced?* Covered →
evidence the registry generalises past its sources. Not covered →
evidence about the mature-vs-research gap, from the only sample of
it available. Do not fold these into the corpus later as "one
more target."

| arm | version | harness | eqns | distinct | covered by public set | novel primitives |
|---|---|---|---|---|---|---|
| maddening | 0.3.1 | one coupled multi-physics graph step (table→ball contact + 32-cell heat stencil) via the compiled scheduler | 54 | 17 | 54/54 (100%) | none |

MADDENING pins `jax<0.6`; this harness ran it on jax 0.10.2
without issue, so the cap is over-tight for this path (and, during
collection, installing it silently downgraded a shared environment
— the exact resolver behaviour stelling's own packaging refuses to
inflict).

## Full table — 1823 equations, 73 distinct primitives, 7 targets

| primitive | breadth | count | top | transparent | nested | wedge |
|---|---|---|---|---|---|---|
| `jit` | 6/7 | 214 | 41 | 123 | 50 |  |
| `mul` | 6/7 | 107 | 50 | 12 | 45 |  |
| `add` | 6/7 | 85 | 25 | 11 | 49 |  |
| `convert_element_type` | 6/7 | 70 | 9 | 30 | 31 |  |
| `broadcast_in_dim` | 6/7 | 65 | 8 | 23 | 34 |  |
| `sub` | 6/7 | 50 | 22 | 6 | 22 |  |
| `select_n` | 5/7 | 188 | 2 | 130 | 56 |  |
| `slice` | 5/7 | 48 | 27 | 14 | 7 |  |
| `div` | 5/7 | 24 | 13 | 2 | 9 |  |
| `neg` | 5/7 | 16 | 7 | 2 | 7 |  |
| `gt` | 5/7 | 14 | 10 | 0 | 4 |  |
| `lt` | 4/7 | 54 | 5 | 13 | 36 |  |
| `dot_general` | 4/7 | 18 | 4 | 0 | 14 |  |
| `reduce_sum` | 4/7 | 17 | 12 | 0 | 5 |  |
| `integer_pow` | 4/7 | 13 | 7 | 0 | 6 |  |
| `or` | 4/7 | 9 | 0 | 3 | 6 |  |
| `cond` | 4/7 | 8 | 2 | 3 | 3 |  |
| `copy` | 3/7 | 195 | 2 | 41 | 152 |  |
| `eq` | 3/7 | 126 | 1 | 110 | 15 |  |
| `custom_jvp_call` | 3/7 | 117 | 0 | 109 | 8 |  |
| `and` | 3/7 | 27 | 0 | 8 | 19 |  |
| `ne` | 3/7 | 18 | 1 | 7 | 10 |  |
| `reshape` | 3/7 | 18 | 2 | 1 | 15 |  |
| `abs` | 3/7 | 17 | 8 | 0 | 9 |  |
| `squeeze` | 3/7 | 12 | 5 | 1 | 6 |  |
| `sqrt` | 3/7 | 6 | 3 | 2 | 1 |  |
| `add_any` | 3/7 | 5 | 2 | 0 | 3 |  |
| `select_if_vmap` | 2/7 | 34 | 0 | 0 | 34 |  |
| `iota` | 2/7 | 11 | 2 | 2 | 7 |  |
| `stop_gradient` | 2/7 | 7 | 0 | 5 | 2 |  |
| `unvmap_any` | 2/7 | 6 | 0 | 2 | 4 |  |
| `not` | 2/7 | 5 | 0 | 3 | 2 |  |
| `scatter` | 2/7 | 5 | 0 | 0 | 5 | ✓ |
| `max` | 2/7 | 4 | 0 | 2 | 2 |  |
| `min` | 2/7 | 4 | 0 | 2 | 2 |  |
| `nonbatchable` | 2/7 | 3 | 0 | 0 | 3 |  |
| `pow` | 2/7 | 3 | 0 | 2 | 1 |  |
| `reduce_max` | 2/7 | 3 | 0 | 0 | 3 |  |
| `stack` | 2/7 | 3 | 0 | 0 | 3 |  |
| `while` | 2/7 | 3 | 0 | 2 | 1 |  |
| `reduce_or` | 2/7 | 2 | 1 | 0 | 1 |  |
| `nextafter` | 1/7 | 100 | 0 | 100 | 0 |  |
| `concatenate` | 1/7 | 14 | 0 | 14 | 0 |  |
| `nondifferentiable_backward` | 1/7 | 13 | 0 | 13 | 0 |  |
| `gather` | 1/7 | 7 | 2 | 0 | 5 | ✓ |
| `rem` | 1/7 | 7 | 0 | 3 | 4 |  |
| `dynamic_slice` | 1/7 | 4 | 0 | 1 | 3 | ✓ |
| `sign` | 1/7 | 4 | 4 | 0 | 0 |  |
| `log` | 1/7 | 3 | 3 | 0 | 0 |  |
| `pad` | 1/7 | 3 | 0 | 0 | 3 |  |
| `random_wrap` | 1/7 | 3 | 3 | 0 | 0 |  |
| `bitcast_convert_type` | 1/7 | 2 | 0 | 2 | 0 |  |
| `custom_vjp_call` | 1/7 | 2 | 0 | 1 | 1 |  |
| `dynamic_update_slice` | 1/7 | 2 | 0 | 2 | 0 | ✓ |
| `is_finite` | 1/7 | 2 | 0 | 1 | 1 |  |
| `le` | 1/7 | 2 | 0 | 0 | 2 |  |
| `pure_callback` | 1/7 | 2 | 0 | 0 | 2 |  |
| `random_bits` | 1/7 | 2 | 0 | 2 | 0 |  |
| `reduce_and` | 1/7 | 2 | 0 | 1 | 1 |  |
| `shift_right_logical` | 1/7 | 2 | 0 | 2 | 0 |  |
| `cumsum` | 1/7 | 1 | 0 | 0 | 1 |  |
| `erf_inv` | 1/7 | 1 | 0 | 1 | 0 |  |
| `exp` | 1/7 | 1 | 1 | 0 | 0 |  |
| `linear_solve` | 1/7 | 1 | 0 | 1 | 0 |  |
| `lu` | 1/7 | 1 | 0 | 1 | 0 |  |
| `maybe_set` | 1/7 | 1 | 0 | 0 | 1 |  |
| `random_split` | 1/7 | 1 | 1 | 0 | 0 |  |
| `random_unwrap` | 1/7 | 1 | 1 | 0 | 0 |  |
| `scan` | 1/7 | 1 | 1 | 0 | 0 |  |
| `scatter-add` | 1/7 | 1 | 0 | 0 | 1 | ✓ |
| `sort` | 1/7 | 1 | 0 | 0 | 1 |  |
| `split` | 1/7 | 1 | 1 | 0 | 0 |  |
| `unvmap_max` | 1/7 | 1 | 0 | 1 | 0 |  |
