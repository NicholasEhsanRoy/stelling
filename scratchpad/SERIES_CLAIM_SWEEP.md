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

## D2. Numeric / IEEE claims — one probe, diffed byte for byte

`probe_numeric.py` and `probe_ftz.py` drive every concrete value claim in
`interval.py`, `propagate.py` and `obligation.py` and print a normalised
table. `diff` of the two series' output, banner line excluded, is **empty**:

```
diff n_0.11.0.txt n_0.10.2.txt   -> no output
```

Covered, all identical on both: the three association orders of `a+b+c+d`
at ±1e308 (`nan` / `0.0` / `+inf`, each matching its own jaxpr order);
`lax.max`/`lax.min` propagating NaN in both operand orders; `pow(NaN,0)=1`
and `pow(1,NaN)=1`; `pow(base,0)=1` for base ∈ {0, −3, 2.5, ±inf};
`float32(1e-45) > 0` False; `5e-324 > 0` False and `5e-324 == 1e-320`
True; f64/f32 subnormal flush in arithmetic, comparisons and libm, eager
**and** jit; f32→f64 convert exactness; `int4`/`uint4` ranges and wrap,
eager and jit; `jnp.square` as its own primitive; `select_n(-1)` → case 0.

The int8 index claim (`obligation.py:1138`, `propagate.py:1438`) is the
sharpest of these and is sound on both: `lax.scatter_add` with an int8
index column writes at operand length **128** (`5.0`) and drops at **129**
(`0.0`), identically on 0.10.2 and 0.11.0.

One probe artifact worth recording so nobody re-finds it as a bug: under
jit, `x * 1.0` returns the subnormal unflushed, because XLA simplifies the
identity multiply away and no arithmetic is emitted. `x * 2.0`, `x + x`,
`sqrt(x)` and a genuine underflow all flush under jit. The FTZ/DAZ claim
is sound; `x * 1.0` is not a test of it.

## D3. Other claims driven and found SOUND

| claim | measurement | result (both series) |
|---|---|---|
| `ir.py:613` `_REQUIRED_PARAMS` — "keys jax supplies on EVERY equation" | 21 traced forms, 18 of 19 entries reached, key sets compared | every required key present; key set constant per primitive |
| `_jax_compat.py:443` — no dtype reaches the `finfo`-raises admit | 38 dtypes from `jnp`/`np`/`ml_dtypes` through `jnp.finfo` | 19 raise on **each** series — same set, and all are integer/bool/object dtypes the earlier branch already returns on |
| `_jax_compat.py:833` — every concrete context rejects a negative extent | `jnp.zeros((-2,-2))`, `jnp.ones((-1,))`, `jnp.empty((-3,))`, `reshape((-2,2))`, `lax.reshape` | rejected on both (0.11 raises `MLIRError` for `jnp.empty`, 0.10 `TypeError` — different type, same refusal) |
| zero-size shapes stay legal, `jnp.all` vacuously True | `jnp.zeros((0,))` | legal, `all() == True`, both |
| `docs/preconditions.md:192`, `docs/harness-api.md:73` "Measured on jax 0.11.0" | the blocks under them are output-compared by `test_doc_examples.py` | pass on both lanes |
| the `custom_vjp` lying-`fwd` hazard (`transparent-primitives.md:128`, `founding.md:77`) | lying `fwd` returning `cos` where `f` returns `sin` | `f(x)=0.29552`, `value_and_grad` value `=0.95534`, grad jaxpr holds only `cos`/`mul` — identical, no raise, both |

## E. Series facts that are genuinely one-sided (correctly scoped)

| | 0.10.2 | 0.11.0 |
|---|---|---|
| `jit`'s `inline` param | `False` | `Inline.AUTO` |
| `scan` flat-tree metadata `ft_in`/`ft_out` | absent | present |

These are why a recorded verdict transcript is series-bound; the docs that
say so are right.

## F. The doc-stamp gap — reproduced, then closed

The off-series escape in `test_doc_examples.py` neutralises the stamp and
the query hash whenever the running jax is not the jax the doc names. It
is keyed on the doc's own text, so a doc naming a series **no lane runs**
escaped on every lane at once.

Reproduced before the fix — `docs/quickstart.md` stamped `jax 0.7.3`, every
`query <sha>` overwritten with `d`×64:

| | 0.10.2 lane | 0.11.0 lane |
|---|---|---|
| forged stamp + forged hashes, before | **37 passed** | **37 passed** |
| forged stamp + forged hashes, after | 1 failed, 38 passed | 1 failed, 38 passed |
| hash forged, stamp left at `0.11.0` | 39 passed (neutralised) | **1 failed** (`quickstart.md:37`) |

The third row is the designed limit and is unchanged — it is only honest
while every stamped series has a lane, which is now enforced by
`test_every_documented_stamp_names_a_tested_series` rather than assumed.

## G. Found by this sweep but NOT a series defect

`contracts.py:572` told callers to avoid `jnp.stack` for family assembly
because `stack` had no transfer row and the family would fall to ⊤. Stale
on **both** series: `stack` carries exact real and ieee transfers, SMT
emission and a replay row. Measured on both — a two-entry `jnp.stack`
family reports `10 eqns: 10 known (100%)`, VERIFIED, zero ⊤. Corrected,
and labelled a staleness defect rather than a series one.

## H. Not touched — inside the concurrent repair's territory

`design/constraining-assume.md:174` ("verified 0.11.0 throughout. Result:
F1 VERIFIED / F2 UNKNOWN / F3 …") is a single-series claim of exactly the
audited shape, but re-driving it means exercising `assume` semantics that
another agent is mid-repair on (`propagate.py`, `test_assume_constrain.py`).
Left alone deliberately; reported rather than measured.
