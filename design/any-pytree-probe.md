# The any_pytree target probe — one hand-declared case, registered before it runs

**Status:** REGISTRATION, 2026-07-18. Committed before any harness is
written or any library installed. The question, and only it:

> **Once pytree structure is declarable, does a faithful imported-library
> case pose — or does something past structure block it regardless?**

`any_pytree` will be built, if ever, under the strongest build-time stake
in the project; its audit gate governs *counting*, not *whether the build
has a target*. D416's actual cause was PRNG key reuse, and a symbolic key
has no interval meaning — pytree sugar declares structure, it does not
make a `PRNGKeyArray` representable. Whether the build has a target is
answerable by hand, in one case, without building anything.

## Targets — fixed now, in order

1. **Clean case (pure structure, no keys): diffrax `PIDController` —
   the real `adapt_step_size` with its real `_PidState`**, hand-declared
   leaf by leaf. This is dfx#207's full-body path, the case whose
   fidelity row said "posable only at the property-relevant slice; the
   full body carries pytree state." The property posed is dfx#207's own,
   on the real code: next-dt ≥ dtmin (the opt-in `dtmin=…` path, as
   disclosed). If *this* doesn't pose by hand, the blocker is structure
   and `any_pytree` is exactly right.
2. **Hard case (the key wall): blackjax MCLMC kernel step**, state built
   by the real `init` from hand-declared position leaves, kernel stepped
   with a **∀-key declaration** — raw key data as a declared `uint32`
   leaf pair wrapped by the library's own `wrap_key_data`. D416's wall,
   tested directly.

**Not MADDENING** — own-code nulls are uninterpretable and the harness
author would be posing his own system; MADDENING stays the held-out
generalisation check, as in the primitive census.

## Rules — no exception for being exploratory

- **Hand-declaration is the whole method.** No `any_pytree`, no new
  transfers. A leaf needing a transfer the registry lacks is a **coverage
  finding, recorded, not built around**.
- **A key leaf gets no invented interval.** If the key has no sound
  interval meaning, it falls to ⊤ and its cone falls to ⊤ — that is the
  finding, not a problem to engineer past.
- **Fidelity holds.** The state is the library's real state at the
  installed, pinned version — quoted signatures, not a model. This probe
  is worthless on a hand-modelled body.
- **No count.** Nothing here poses toward any band. If a case happens to
  mechanize, that is evidence `any_pytree` has a target — not a
  mechanized case, which would need the full registered pipeline.

## Outcomes — fixed before the run

| outcome | reading | next |
|---|---|---|
| clean case poses by hand | `any_pytree` is sugar over a real capability | build it — with its registered audit gate, by a fresh-context builder (the stake analysis: the builder must not know the counts or which cases it needs to unblock) |
| clean case blocked past structure | even array-only pytree state has a non-structural blocker | the blocker is the finding; `any_pytree` is premature |
| clean poses, hard (keys) blocked | `any_pytree` reaches array-state libraries, **not samplers** | build it scoped honestly — **and the demand corpus splits, predictably, into an `any_pytree`-reachable half and a key-blocked half**, bounding the corpus expansion before it runs |

"Poses" means: the property is statable on the real function's outputs
and the pipeline runs end-to-end (trace → transcribe → propagate →
judged obligation, UNKNOWN allowed); "blocked" names the stage and quotes
the blocker.

**Prior, recorded:** the third row. The clean case's risks are equinox
plumbing primitives (transcription) and registry gaps (coverage — fine);
the hard case's key cone should swallow the obligation into ⊤ by
construction.

---

# Reading (2026-07-18 — `corpus/supply/pytree_probe.py`, diffrax 0.7.2 / blackjax 1.6.2 / jax 0.11.0)

## The probe caught the tool before it caught the libraries

First contact with real library traces crashed **both** cases at
propagation — not on structure, not on keys, but on the tool's own shape
guards raising on **legal jax forms** (scalar-`which` `select_n`; rank
broadcasts like `(2,) vs (1,)`). That is the second audit's FRAGILE-5
posture incompletely repaired: the guards added to prevent silent
mis-joins killed the analysis instead of degrading. Fixed within the
probe's rules (a posture repair adds no capability): transfers now
**decline** unhandled forms — ⊤, reason quoted in the notes — and the
regression test pins it. Then the probe re-ran.

## Clean case — POSED. Row 1: `any_pytree` is sugar over a real capability

`PIDController.adapt_step_size`, real `_PidState` hand-declared leaf by
leaf, dfx#207's property stated on the real code: **traced, transcribed
with zero `UnsupportedParamError` (the whitelist and opaque registry
survived diffrax 0.7.2 on first contact), propagated end-to-end.**
Obligation: unknown. Coverage: `70 eqns: 39 known (56%); 19 ⊤ across 11
primitives (abs, eq, convert_element_type[weak-type forms], or,
stop_gradient, and, ne, pow, reduce_or, reshape, select_n[scalar-which])`.

**Structure was never the wall.** Every blocker between *posed* and
*mechanized* is ordinary, censusable transfer work: ~11 registry rows
plus two array-semantics domain gaps (rank broadcasting; scalar-`which`
select). The ⊤ list **is** the target census for this case — the
census-by-census rule producing the build list, again.

## Hard case — POSED, and the key wall is exactly as predicted

MCLMC kernel, state built by the real `init` from declared position
leaves, ∀-key via declared `uint32` bits through the library's own
`wrap_key_data`: traced, transcribed (blackjax's PRNG plumbing passed
through the registered opaque impl params), propagated. Obligation:
unknown. Coverage: `393 eqns: 220 known (56%); 139 ⊤ across 25
primitives` — of which the **key cone is structural and irreducible**:
`random_wrap ×2, random_split ×2, random_bits ×5, bitcast_convert_type
×5, shift_right_logical ×5, erf_inv ×5` — the bits-to-Gaussian pipeline
has no interval meaning, fell to ⊤ without a crash and without an
invented representation, and everything downstream of the sampled
momentum inherits it. Registry rows (`reduce_sum, sqrt, log, dot_general,
…`) are ordinary; **the key cone is not** — completing every registry row
would still leave the obligation unknown, because the state itself is
key-derived.

## Outcome: row 1 + row 3 — the registered third-row reading, with one addition

- **Clean poses by hand → build is licensed** per the fixed table: with
  its registered audit gate, by a fresh-context builder that does not
  know the counts (`design/corpus-limits.md`).
- **Keys blocked → the split is confirmed**: `any_pytree` reaches
  array-state libraries (diffrax/optimistix-shape hits), **not
  samplers**. **The demand corpus splits, predictably and now, into an
  `any_pytree`-reachable half and a key-blocked half** — bounding the
  corpus expansion before it runs: sampler hits (blackjax, numpyro
  warmup) can be posed only up to their key cones, which is UNKNOWN by
  construction for state-dependent properties. *(Relabeled 2026-07-18:
  this split is a **rough proxy**. The real boundary is whether the
  property's content survives interval abstraction — keys are the
  100%-dependency form, normalizations/ratchets the partial form; sampler
  hits wall per-case, not as a bloc. `design/second-bill.md`,
  `design/jax-verification-categories.md`.)*
- **The addition the rows didn't anticipate:** between structure and
  mechanization sits a **prerequisite neither wall named** —
  array-semantics completeness of the interval domain (rank broadcasting,
  batched/scalar selectors) plus the per-target registry rows. Ordinary
  work, censusable by the existing method, and `any_pytree`'s value is
  gated on it exactly as much as on the sugar itself.

No count moved; nothing here poses toward any band. The build decision is
the maintainer's, now with its target established, its scope bounded, and
its prerequisite named.
