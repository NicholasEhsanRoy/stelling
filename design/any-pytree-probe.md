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
