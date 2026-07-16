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

## The hazard that is actually there (verified on jax 0.10.2)

**JAX does not check that a `custom_vjp`'s `fwd` returns the same primal as
`f`.** Probe: a `custom_vjp` whose `fwd` deliberately returns `cos(x)`
while `f` returns `sin(x)`. jax raises nothing. `f(x)` evaluates
`sin(0.3) = 0.29552`; `jax.value_and_grad(f)(x)` returns **`cos(0.3) =
0.95534` as the value**, and the grad-traced jaxpr contains only `cos` —
`f`'s computation is absent entirely. Under `grad`, the forward pass that
executes is `fwd`'s, not `f`'s.

The commitment this entails is broader than gradients:

> **Everything inside a grad trace comes from the grad trace — primal
> properties included.** A primal verdict proved on `f`'s IR does not
> transfer to the primal computation inside `grad(f)`. If the property
> matters under `grad`, trace `grad(f)` and verify there.

An `fwd` silently disagreeing with `f` is a bug class with no existing
detector: nothing in jax checks it, it has no runtime symptom, and it
survives all primal testing because plain calls execute `f` while every
training and optimization loop executes `fwd`. It is precisely the
`custom_vjp` equivalence-checking target from the founding roadmap, now
promoted there to a Stage-2 flagship — compiled as an
`|f − fwd_primal| ≤ tol` query per design commitment 1, not built as a
subsystem.

Descend-into-`call_jaxpr` remains right for analyses of `f` and wrong as a
route to `grad(f)`: gradient-context properties are obtained by tracing
`grad(f)` in jax, never by differentiating the primal IR.
