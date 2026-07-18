# MIME FVM, the mesh-skewness condition — the first usefulness test, registered before it runs

**Status:** REGISTRATION, 2026-07-18. MIME ref pinned: `7ce1efb4311b`.
Heat proved the mechanism on a proof-of-concept node and claimed nothing;
**this is the usefulness test**: a validated solver the author relies on,
a condition with a real failure behind it, and the bar is *would the
check earn a place in CI*. Held out from E2a by construction — a
usefulness result, never a count.

## The two rings

**Ring 1 (scope):** the transcriber sees `src/mime/nodes/environment/fvm/`
and its direct imports. No other MIME code.

**Ring 2 (the one that matters):** the transcriber never sees FVM's
validation results, the characterisation study, `known_anomalies.yaml`,
git history, or any statement of which regime fails. **The scar is the
main agent's**; the property is stated from it blind-to-the-transcriber.
A transcriber that knows the failing regime will, with no bad intent,
transcribe toward reproducing it.

## The condition, chosen by its scar (main agent, on the record)

**The mesh-skewness stability of the non-orthogonal Laplacian
correction.** The scar, quoted:

- Commit `91e95e6` (2026-06-11): *"Consistent and convergent where
  orthogonal-only **diverges** (manufactured solution on a sheared
  simplex mesh)"* — the fix introduces the over-relaxed coefficient
  `E_f = |Sf|²/(Sf·d)` whose **denominator is the alignment itself**:
  as `cos(Sf, d) → 0` the correction blows up. The mechanism is in the
  code at `operators.py:251–253`
  (`Sf_dot_d = jnp.sum(mesh.Sf * mesh.d, axis=1);
  ortho_coeff = mesh.area ** 2 / Sf_dot_d`).
- Commit `6c064e9` (context): the cylinder-mesh instability finding
  ("exponentially unstable … max|u| grows ~10x/step") — a neighbouring
  scar (momentum diagonal), cited as context, not the target.
- **Owner-supplied characterisation** (stated in the work order; not
  found on disk, labeled as such): the cylinder mesh reaches
  `min cos(Sf, d) ≈ 0.11`; the stability boundary sits at
  **≥ 0.71 stable / ≤ 0.59 divergent**. These numbers are ring-2
  material: the main agent uses the *stable floor* (0.71) in the
  property; the transcriber receives it as a bare number.

This is a real scar (a documented divergence, a dated fix, a
characterised boundary), a state predicate over a geometric region, and
the tool's home turf. Not manufactured.

## The property — one condition, three formulations (heat's A1/A2/B shape)

**(F1) The bounded-correction lemma — the check.** On faces satisfying
the alignment floor, the correction coefficient is bounded:
`∀ a ∈ [0.5, 2.0] (face area |Sf|), dm ∈ [0.5, 2.0] (|d|),
c ∈ [0.71, 1.0] (cos(Sf,d)): a/(dm·c) ≤ 8.0`.
Hand-transcription of `area²/(Sf·d)` with two **disclosed derivations**
(polar substitution `Sf·d = |Sf||d|cos`; the exact `a²/a` cancellation,
valid for `a > 0`), pointer to `operators.py:251–253`. Geometry boxes are
clean round numbers (disclosed as choices); `B = 8.0` gives ~42% margin
over the corner value 5.63. **Predictions:** VERIFIED; under
inputs-only ⊤-widening, unknown (non-tautological — the floor is
load-bearing: without `c ≥ 0.71` the coefficient is unbounded, which is
the scar).

**(F2) The code's own raw form — the predicted wall.** `Sf` and `d` as
independent 3-vector boxes (components declared scalar-wise; the dot
transcribed as `Sf_x·d_x + Sf_y·d_y + Sf_z·d_z`), the alignment floor
stated as it really is — **a relation**:
`assume(Sf·d ≥ 0.71·|Sf|·|d|)` — then `assert(area²/(Sf·d) ≤ B)`.
**Predictions:** the assume is inert (dropped, disclosed); `Sf·d` over
independent boxes straddles 0; the division widens to ⊤; the obligation
is **unknown — blocked (inert assume)**: mesh-quality conditions are
*relations between geometric vectors*, and this is the
**constraining-assume wall's first sighting on real solver geometry** —
the relational-domain demand evidence, from a real job.

**(F3) The imported harness — the milestone grade.** The real
`laplacian_orthogonal(phi, mesh, mu_face, non_orthogonal=True)` traced:
a small **real** mesh built by an in-package constructor (concrete
geometry via the prototype-outside-trace pattern), `phi` declared as a
field box, obligation: output flux bounded. **Predictions:** the tail
census is the second real second-bill datum and the first with genuine
numerics — expected rows/⊤s include **`gather`** (`phi[mesh.owner]` — the
census's original wedge primitive, demanded by a real job for the first
time), `segment_sum` (scatter-add class), `reduce_sum`, `sqrt`,
`dot_general` (the LSQ gradient); first-contact FRAGILE escapes possible;
the obligation lands unknown (coverage and/or the stencil dependency —
heat's wall, harder).

## The vacuity procedure, fixed before this run (closing heat's instrument note)

**The ⊤-widening for this job widens region-input declarations only** —
mechanically: non-point `stelling_any` declarations widen; point
declarations (transcribed constants) and literals (thresholds `0.71`,
`8.0`) stay. Widening a threshold makes almost any comparison straddle
and defeats the tautology detector; heat's a-fortiori luck is not a
procedure. Registered here, before any FVM result exists.

## Outcomes — fixed before the run

| outcome | reading |
|---|---|
| **VERIFIED (F1), non-vacuous, ℝ-faithful** | **the first true positive on a solver the author relies on** — a real, scarred condition checked over a region; ⊤-widening re-run in the same breath; a CI candidate |
| **UNKNOWN, dependency-shaped (F3)** | **the strongest possible evidence for affine**: the second stencil sighting, on a real solver — informs the *build* decision (real, motivated affine demand) while firing **no E2a trigger** (held out). Both halves stated wherever it appears |
| **UNKNOWN, search/coverage-shaped** | the solver's first real customer / a registry row — either is a finding about what a real solver needs |
| **REFUTED** | on a validated solver: hard scrutiny of box and transcription first; a *sound* set-level refutation of a scarred stability condition is the tool finding the failure class statically — the strongest result it can produce |
| **won't pose / wall past stencil** | named (the `lu`/dense-LA tier-9 frontier is the candidate) — bounds what the tool reaches in a real solver |

## Who does what

**Fresh-context transcriber (blind):** FVM source only; the predicates
verbatim (bare numbers); build F1/F2/F3, get them to pose; census the
tail; registry rows **only by census** under the two allowed classes
(structural-exact; three-line monotone) with witnessed constructions —
everything else declines and reports; FRAGILE escapes fixed
degrade-only with regression tests. Never told the desired verdict or
the failing regime. MIME installed `--no-deps` at the pinned ref; jax
stays 0.11.0; CALMS out of scope.

**Main agent:** the condition and scar above; the widening variant; the
semantics classification (predicted ℝ-faithful — geometric real
arithmetic; finalized post-run); verdict adjudication against the scar;
whether a dependency-shaped F3 constitutes real affine demand.
