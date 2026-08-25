# Transparent primitives

**Status:** design note, normative for the Stage-0/1 interpreter. Evidence
first verified against jax 0.10.2 on 2026-07-16; re-driven on **both**
tested series 2026-08-07 and the container column corrected (below).
Re-verify on every jax series bump (the `TESTED_JAX_SERIES` constant in
`stelling/_optional.py` is the forcing function) — and re-verify it *per
series*, because the first bump's finding was that a cell of this table
moves, not that the table as a whole survived.

## The problem

"Unknown primitive → ⊤" is the right default for *math* primitives, but
there is a class of wrapper primitives whose actual computation lives as a
sub-jaxpr in `params`. Treating those as unknown makes ordinary jnp code
unanalyzable. The evidence that this is a day-one requirement and not an
edge case: **on jax 0.10, `jnp.roll` traces to a `jit` equation** wrapping
the real gather/concatenate work — under ⊤, the Stage-0 wedge dies on its
own demo.

The correct transfer function for this class is: **descend into the
sub-jaxpr with the current environment**, not ⊤.

## The class, as verified on jax 0.10.2 **and** 0.11.0

The container column is **per series**, and that is the whole point of it.
jax 0.11 merged `Jaxpr` and `ClosedJaxpr` into one class —
`ClosedJaxpr is Jaxpr` is `False` on 0.10.2 and `True` on 0.11.0 — so a
container named here is a fact about the jax that produced the param, never
a fact about the callee. Both columns below were driven on both
interpreters (`scratchpad/SERIES_CLAIM_SWEEP.md` §A — a historical
measurement, tracked and not in the sdist; the table below is its
result).

| primitive | sub-jaxpr param | container: 0.10.2 → 0.11.0 | other params of note |
|---|---|---|---|
| `jit` | `jaxpr` | `ClosedJaxpr` → merged class | shardings/layouts hold `UnspecifiedValue` (zero-payload sentinel), `ctx_mesh` holds an empty `Mesh` (sentinel; non-empty raises); `inline` is a `bool` on 0.10 and an `Inline` enum on 0.11 — the one param that moves the query hash |
| `custom_jvp_call` | `call_jaxpr` | `ClosedJaxpr` → merged class | `jvp_jaxpr_fun` is a `WrappedFun` thunk → `OpaqueParam` |
| `custom_vjp_call` | `call_jaxpr` | `ClosedJaxpr` → merged class | `fwd_jaxpr_thunk`, `bwd` (`WrappedFun`), `out_trees` (function) → `OpaqueParam` |
| `remat2` | `jaxpr` | **open `Jaxpr`, not Closed** → merged class | `policy` is `None`, or a callable when user-supplied → `OpaqueParam` |

**`remat2` is the single cell that moves, and an earlier version of this
table stated its 0.10 shape as though it held on every series.** It does
not: on 0.11 the two classes are one object, so "open `Jaxpr`, not Closed"
is not a distinction that can be drawn at all, and `isinstance(v,
ClosedJaxpr)` answers `True` for `remat2`'s body there and `False` on 0.10.
Measured, same `x.at[0].add(5.0)` and same `jax.checkpoint`: the transcribed
param is `ir.Jaxpr` on 0.10.2 and `ir.ClosedJaxpr` on 0.11.0.

**So no consumer may read this table's container column as a type test.**
The canonical accessors are `stelling.coverage.call_body` (a wrapper's
callee, closed) and `stelling.coverage.sub_jaxprs` (nesting); both accept
either container by construction, and `call_body`'s docstring carries the
same measurement as executable-adjacent prose. Every hand-rolled
`isinstance(v, ir.ClosedJaxpr)` that has been written against this class so
far was a latent series bug, and three of them shipped — see
`design/maintenance-treadmill.md`.

Not observed on 0.10 but expected in other series: `closed_call` /
`core.call` (older series), and `custom_lin` (grad-of-custom_vjp in some
versions). `custom_lin`'s `bwd` thunk is deliberately **not** in the opaque
registry yet — first contact will raise loudly, which is the designed
behavior for unverified slots.

Transcription (`stelling/_jax_compat.py`) recurses into all of the above
generically: any param value that is a `ClosedJaxpr` or `Jaxpr` — direct or
tuple-nested — is transcribed. Tests pin `jnp.roll`, `custom_jvp`
(`jax.nn.relu`), `custom_vjp` (forward and under `grad`), and
`jax.checkpoint`.

## Thunks and `OpaqueParam`

`custom_jvp_call` / `custom_vjp_call` carry their derivative rules as
callables. jax keeps them for *its own* later transforms of the jaxpr;
stelling never re-runs jax transforms on transcribed IR, so their content is
unreachable by construction for every analysis stelling performs. They are
recorded as `ir.OpaqueParam` — present, named, explicitly lossy — via an
explicit per-`(primitive, param)` registry; callables in unlisted slots
still raise.

Raising instead was not an option: these eqns **survive `jax.grad`**
(verified — a grad-traced jaxpr inlines the `bwd` math as ordinary
equations while keeping the `custom_vjp_call` wrapper for the primal), so
strictness here would make any program touching `jax.nn.relu`
untranscribable.

**Hash scope:** `OpaqueParam` content is invisible to the content hash —
two programs identical except for their custom derivative rules hash alike.
For *primal* properties this is correct, not a hazard: `custom_vjp` does
not touch `f`, so such programs are the same program, and hashing them
alike is exactly the identity relation verification wants. Gradient
verdicts key on `grad(f)`'s trace, where jax's AD has already consumed the
thunks and inlined the derivative math as ordinary equations (verified on
0.10.2), so that hash distinguishes them on its own. The structural
mitigation exists; what remains is bookkeeping — a *stored* verdict should
say which trace its hash is of, which `DOCUMENTATION_ARCHITECTURE.md` §2.6
designs under the name `opaque_params`. **The bookkeeping is still what
remains**: no stamp records how many opaque params a query held, so a reader
holding two verdicts with equal hashes cannot tell from them whether the
equality is total.

`OpaqueParam` is ⊤ at the param level: the same known-unknown discipline
the transfer registry applies to unknown primitives, one layer down. That
symmetry is why the shape is right.

## Census contact (2026-07-17, jax 0.10.2)

Running the corpus census (`corpus/run_census.py`) forced six further
transcription rules, each verified against a real trace:

- **`PyTreeDef` params** (equinox stateful ops; custom linear solves) →
  `ir.TreeDefParam`, the canonical string form; detected via the public
  `jax.tree_util.PyTreeDef`.
- **avals as params** (`pure_callback.result_avals`) → mirrored as
  `ir.Aval`, like any other aval.
- **trivial `NamedSharding`** (empty mesh, nothing partitioned; numpyro
  attaches one to `convert_element_type`) → sentinel; non-trivial
  shardings still raise.
- **bare `object()` placeholders** (lineax's `linear_solve.static`) →
  sentinel: identity-only, no payload by construction.
- **PRNGImpl function fields** (`random_wrap.impl.*`; blackjax) → opaque;
  the impl's identity survives in its `name`/`tag`/`key_shape` fields.
- **host callbacks** (`pure_callback.callback`; diffrax error paths) and
  equinox's `linear_solve.flatten` plumbing → opaque. A host callback is
  ⊤ at the param level by definition: nothing stelling runs can ever look
  inside one.

Also learned: equinox-defined primitives (`select_if_vmap`, `unvmap_any`,
`nonbatchable`, `maybe_set`, `linear_solve`, …) are load-bearing across
the mature-library arm of the corpus — the transfer registry will need a
position on them, not only on `lax`.

## The hazard that is actually there (verified on jax 0.10.2)

> **Under `grad`, `f` is dead code.** Both the value and the gradient come
> from `fwd`/`bwd`; `f` is documentation — and JAX never checks that the
> two agree.

Why nobody has noticed: tests call `f`, training and optimization loops
call `fwd`. **They exercise different code, and nothing says so.** An
`fwd` that silently disagrees with `f` has no runtime symptom and no
existing detector.

The evidence. Probe: a `custom_vjp` whose `fwd` deliberately returns
`cos(x)` while `f` returns `sin(x)`. jax raises nothing. `f(x)` evaluates
`sin(0.3) = 0.29552`; `jax.value_and_grad(f)(x)` returns **`cos(0.3) =
0.95534` as the value** — the caller asked for `f`'s value and silently
got `fwd`'s — and the grad-traced jaxpr contains only `cos`: `f`'s
computation is absent entirely.

The same probe against `custom_jvp` (2026-07-17): **identical result.** A
jvp rule whose primal output deliberately returns `cos(x)` while `f`
returns `sin(x)` raises nothing; `jax.value_and_grad` and `jax.jvp` both
return the lie as the value. The class is not custom_vjp-specific.

Population, correctly counted (equation counts are not populations — one
rule called a hundred times is one opportunity for the bug): the corpus's
117 `custom_jvp_call` equations decompose into **7 distinct rules, all
library-authored** (equinox ×3, lineax ×3, optimistix ×1; equinox's
`_nextafter` rule alone is 100 of the 117 equations), plus one equinox
`custom_vjp` rule — see `design/rule-provenance.md`. Two qualifiers from
the same probe: `jax.test_util.check_grads` **does** catch lying primals,
so the live hazard is rules that are never run through it; and rule
provenance is recoverable from transcribed IR alone (the primal
`call_jaxpr`'s `debug_info`), so a future checker can attribute findings
to the rule's author, not the caller.

The commitment this entails is broader than gradients:

> **Everything inside a grad trace comes from the grad trace — primal
> properties included.** A primal verdict proved on `f`'s IR does not
> transfer to the primal computation inside `grad(f)`. If the property
> matters under `grad`, trace `grad(f)` and verify there.

This is precisely the `custom_vjp` equivalence-checking target from the
founding roadmap, now promoted there to a Stage-2 flagship — compiled as
an `|f − fwd_primal| ≤ tol` query per design commitment 1, not built as a
subsystem.

Descend-into-`call_jaxpr` remains right for analyses of `f` and wrong as a
route to `grad(f)`: gradient-context properties are obtained by tracing
`grad(f)` in jax, never by differentiating the primal IR.
