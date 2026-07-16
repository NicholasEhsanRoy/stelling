# The primitive set is open

**Status:** design note, 2026-07-17. Position only — no implementation.
Evidence: the census (`design/primitive-census.md`), jax 0.10.2. Re-verify
trigger: the first contrib/plugin transfer rule landing.

## The finding

Equinox-defined primitives are load-bearing in the mature-library arm of
the corpus: `select_if_vmap` ×34, `nondifferentiable_backward` ×13,
`unvmap_any`, `maybe_set`, and lineax's own `linear_solve` — 59 equations
across the census's tier 7. The cost of not having a position is already
demonstrated: **lineax censuses at 10 equations** — an entire library's
computation invisible behind one library-defined primitive whose params
are opaque plumbing.

So the question is not "do we support equinox." The question was implicit
in commitment 4 (unknown → ⊤) all along and is now concrete: **the
primitive set is open**, and the open part is the difference between
analysing diffrax and not.

## Position

1. **A plugin protocol.** A library (or a user) can register transfer
   semantics for its own primitives, through the same registry interface
   as core rules: same per-domain structure, same tier declarations
   (exact / sound / heuristic), same coverage accounting. A plugin rule
   that descends into a sub-jaxpr param declares itself transparent the
   same way the built-in wrapper class does.
2. **A bundled contrib registry as the bridge**, for the equinox family
   stelling has verified itself. Explicitly second-class: separately
   versioned, separately tested, pinned to the library series it was
   verified against — a `TESTED_EQUINOX_SERIES` analog, **with its fence
   built in from day one** (the can't-drift pattern in CONTRIBUTING.md:
   a hardcoded-literal assertion plus a test that fails when CI's
   equinox outruns it). The jax fence had to be named after the fact;
   this one is named before the code exists.
3. **Marked in the verdict stamp.** Every transfer in a verdict's chain
   records its provenance — core vs. contrib vs. plugin, with the
   contributing registry's own version. `SOUNDNESS.md` now requires
   this. Without it, a verdict that leaned on a contrib rule has the
   `TESTED_JAX_SERIES` problem with no fence and no test — silently.

## What this note does not decide

Which equinox primitives get rules first (the census says
`select_if_vmap` leads by count; several may be descend-style
transparent rules rather than interpretations); the protocol's API and
naming; whether contrib rules may ever be tier `exact`. All deferred to
implementation, after Stage 0. The held-out MADDENING arm's result —
100% of its profile covered by the public corpus's primitive set — says
the *jax-level* registry generalises; it says nothing about library
primitives, which is precisely the layer this note is for.
