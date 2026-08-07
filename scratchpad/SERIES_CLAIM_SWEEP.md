<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# Measurement log — series-scoped claims in `src/`, `docs/`, `design/`

**Branch:** `fix/series-scoped-doc-claims` off `9efea6f`. **Method:** every
claim below was DRIVEN on both interpreters, never reasoned about. A claim
reasoned but not run is marked SUSPECTED and says so.

Interpreters (no installs into either):

- `/home/nick/venvs/stelling-jax/bin/python` — jax 0.11.0
- `/home/nick/venvs/stelling-jax010/bin/python` — jax 0.10.2

Every run under `export JAX_PLATFORMS=cpu PYTHONPATH=<worktree>/src`, with
`stelling.__file__` verified to resolve into the worktree before each
comparison (both venvs have a `stelling` installed — the trap is real).

---

## A. The load-bearing container fact, measured

```
$PY -c "from jax.extend.core import Jaxpr, ClosedJaxpr; print(ClosedJaxpr is Jaxpr)"
```

| | 0.10.2 | 0.11.0 |
|---|---|---|
| `ClosedJaxpr is Jaxpr` | **False** | **True** |
| `remat2.jaxpr` transcribes to | `ir.Jaxpr` | `ir.ClosedJaxpr` |
| `jit.jaxpr` transcribes to | `ir.ClosedJaxpr` | `ir.ClosedJaxpr` |
| `scatter-add.update_jaxpr` transcribes to | `ir.Jaxpr` | `ir.ClosedJaxpr` |

`remat2` and `scatter-add` are the two cells that move. CONFIRMED.

## B. The predecessor walk was LIVE on 0.10, not latent

`probe_walk.py` / `probe_bar.py` reimplement the pre-migration
`getattr(v, "jaxpr", None)` descent and diff it against
`coverage.sub_jaxprs` on the same transcribed equation.

| equation | walk | 0.10.2 | 0.11.0 |
|---|---|---|---|
| `scatter-add` (`jax.ops.segment_sum`) | predecessor | **0** | 1 |
| | `sub_jaxprs` | 1 | 1 |
| `remat2` (`jax.checkpoint`) | predecessor | **0** | 1 |
| | `sub_jaxprs` | 1 | 1 |

End to end, `scatter` inside `jax.checkpoint`, against
`verdict.VERIFIED_BARRED_PRIMITIVES`:

| | 0.10.2 | 0.11.0 |
|---|---|---|
| `verdict._barred_in_eqns` (current) | `('scatter',)` | `('scatter',)` |
| predecessor walk | **`()`** | `('scatter',)` |
| same, inside `jax.jit` (control) | `('scatter',)` both | `('scatter',)` |

The bar UNDER-FIRED on 0.10.2 and only on 0.10.2. CONFIRMED.

## C. `scan` lost two params in the bump, not three

Six traced scan forms (plain, with consts, `length=`-only, tuple carry,
under `grad`, `fori_loop`) plus jax's own bind sites:

```
$PY -c "import inspect,re;from jax._src.lax import control_flow as cf;
        s=inspect.getsource(cf.loops);print('linear=' in s)"
```

| | 0.10.2 | 0.11.0 |
|---|---|---|
| scan param keys (union over 6 forms) | `jaxpr length num_carry num_consts reverse unroll` | `ft_in ft_out jaxpr length reverse unroll` |
| `linear` present | **False** | False |
| `linear=` in jax's `loops` source | **False** | False |

`num_consts`/`num_carry` were lost in the bump. `linear` was already gone
on 0.10.2, so it cannot have been lost in it. CONFIRMED.

## D. Claims driven on both and found SOUND

| claim | measurement | 0.10.2 | 0.11.0 |
|---|---|---|---|
| scatter-add supported forms (`design/scatter-rows.md:19`) | `dimension_numbers`, shapes, index dtype, `_scatter_add_row_form` for `segment_sum` 1-D, `segment_sum` trailing-dims, `at[array].add`, `at[1].add` | identical | identical |
| `_is_add_combiner` on all four | same probe | True ×4 | True ×4 |
| scatter-add carries a sub-jaxpr (`scatter-rows.md:32-34`) | `sub_jaxprs` on the equation | 1 | 1 |
| coverage dial-invariance (`scatter-rows.md:62-67`) | traced `segment_sum` harness, `measure()` vs `propagate(semantics=…)` | static 8, real 8/8/0, ieee 8/6/1 | identical |
| "100% coverage, VERIFIED" payoff (`scatter-rows.md:41`) | `check(harness, vacuity_mode="inputs-only")` | VERIFIED, `8 eqns: 8 known (100%)` | identical |
| QR lowers to `geqrf`, composite `qr` never appears (`design/la-and-stack-probes.md:138`) | `lineax.linear_solve(..., solver=lx.QR())` surface trace and `lx.QR().init` deep trace | `geqrf`, no `qr` | `geqrf`, no `qr` |
| `jnp.stack` is its own primitive (`contracts.py:572`, `design/e2a-registration.md:134`) | `jax.make_jaxpr(jnp.stack)` | `stack` | `stack` |
| `int4`/`uint4` exist | `hasattr(jnp, …)` | True | True |
| `jnp.roll` traces to a `jit` equation (`transparent-primitives.md:14`) | `jax.make_jaxpr` | `jit`+`concatenate`+`slice` | same |

## E. Series facts that are genuinely one-sided (correctly scoped)

| | 0.10.2 | 0.11.0 |
|---|---|---|
| `jit`'s `inline` param | `False` | `Inline.AUTO` |
| `scan` flat-tree metadata `ft_in`/`ft_out` | absent | present |

These are why a recorded verdict transcript is series-bound; the docs that
say so are right.
