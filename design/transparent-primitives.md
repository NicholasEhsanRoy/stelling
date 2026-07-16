# Transparent primitives

**Status:** design note, normative for the Stage-0/1 interpreter. Evidence
verified against jax 0.10.2 on 2026-07-16; re-verify on every jax series
bump (the `TESTED_JAX_SERIES` constant in `stelling/_optional.py` is the
forcing function).

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

## The class, as verified on jax 0.10.2

| primitive | sub-jaxpr param | other params of note |
|---|---|---|
| `jit` | `jaxpr` (`ClosedJaxpr`) | shardings/layouts hold `UnspecifiedValue` (zero-payload sentinel), `ctx_mesh` holds an empty `Mesh` (sentinel; non-empty raises) |
| `custom_jvp_call` | `call_jaxpr` (`ClosedJaxpr`) | `jvp_jaxpr_fun` is a `WrappedFun` thunk → `OpaqueParam` |
| `custom_vjp_call` | `call_jaxpr` (`ClosedJaxpr`) | `fwd_jaxpr_thunk`, `bwd` (`WrappedFun`), `out_trees` (function) → `OpaqueParam` |
| `remat2` | `jaxpr` (**open** `Jaxpr`, not Closed) | `policy` is `None`, or a callable when user-supplied → `OpaqueParam` |

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
mitigation exists; what remains is bookkeeping — a *stored* verdict says
which trace its hash is of, which `DOCUMENTATION_ARCHITECTURE.md` §2.6
carries as the `opaque_params` field.

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
