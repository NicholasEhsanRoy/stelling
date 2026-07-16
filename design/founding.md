# Jaxpr-Level Verification for Scientific Computing

**Founding document & roadmap — v0.1, July 2026.** Name: `stelling`

## Why this exists

JAX scientific code fails in ways testing structurally cannot catch: an out-of-bounds gather silently *clamps* under `jit` and produces wrong physics with no crash; a NaN first appears at step 800,000 of a 40-hour run; `sqrt`, `abs`, `norm`, and `arctan2` are clean in the forward pass and NaN in the backward one; a conserved quantity drifts over horizons no test can unroll. This project brings Kani-style, harness-driven verification to those programs, using jaxpr as the verification IR — a target arguably cleaner than the MIR Kani works on: purely functional, no aliasing or mutation, statically typed and shaped, ANF, a small primitive vocabulary, and structured control flow. The audience is general JAX scientific computing (jax-md, jax-cfd, diffrax, numpyro, and hand-rolled solvers) — deliberately *not* neural-network robustness certification, which α,β-CROWN and its lineage already serve well. What that lineage doesn't serve is assertion-based verification of ordinary array programs — the niche Kani owns for Rust — and, beyond it, properties reachable only because JAX transforms are first-class: verified gradients, custom-VJP equivalence, certified stability of hybrid classical/learned systems.

## Design commitments

Six decisions everything downstream leans on:

1. **One query.** Every task — checkify conditions, inductive invariants, equivalence within tolerance, gradient properties — lowers to a single object: a boolean-output jaxpr `B` plus a region `R` (the assumes). The question is always: *does there exist x ∈ R with B(x) = false?* The fuzzer, the SMT encoder, and the abstract interpreter are three interpreters of that one object. This keeps N backends honest, and makes features like equivalence checking (`|f − g| ≤ tol`) fall out as query construction rather than as subsystems.
2. **Real arithmetic with margin.** Prove robust invariants — strict inequalities with slack — over ℝ, and treat floating point as a bounded perturbation to be discharged later, FPTaylor-style. This is dReal's δ-decidability stance, which is the honest notion of truth for physical properties, and it neutralizes XLA reassociation: fusion shuffles ulps, robust invariants don't care.
3. **Fixed shapes per harness.** Scientific-computing shapes are config constants fixed per run. Exploit that. Shape generalization is genuinely unsolved and lives in the long tail.
4. **Unknown is a work item, not a failure.** The transfer registry defaults unknown primitives to ⊤ and never crashes; the coverage report says exactly which primitive fell to ⊤ and where precision died; the fuzzer floor guarantees the tool always returns *something*. This is what "general JAX programs" means operationally.
5. **Verdicts carry their assumptions.** Every transfer function declares a tier — exact, sound over-approximation, or heuristic — and the report surfaces it. The trust boundary is the jaxpr: we verify what traced, XLA runs what compiled, and we say so plainly (Kani extends the same trust to rustc's backend) until translation validation exists.
6. **Induction over unrolling.** Long-horizon properties are proven as one-step inductiveness, `x ∈ S ⟹ step(x) ∈ S`, applied to the step function directly — no `scan` semantics needed, no horizon dependence. The Astrée move, aimed at differentiable physics.

## Stage 0 — proof of concept (build this first)

**Goal:** one demo, end to end — trace a harness, compile the query, prove it, break it, replay the counterexample. This immediately follows the autodidax course: the interpreter core and the harness primitives below are direct applications of its Part 1–2 machinery (tracing, jaxprs, `eval_jaxpr`-style walking, defining primitives with abstract-eval rules).

**The demo:** positivity of first-order upwind advection under a CFL assumption.

```python
import jax.numpy as jnp
import stelling

N, dx, u = 8, 1.0, 1.0   # fixed shape, constant velocity > 0

def step(rho, dt):
    return rho - (dt * u / dx) * (rho - jnp.roll(rho, 1))

@stelling.harness
def positivity():
    rho = stelling.any_array((N,), assume=lambda r: (r >= 0).all())
    dt  = stelling.any_scalar(bounds=(0.0, 10.0))
    stelling.assume(dt * u / dx <= 1.0)            # CFL
    stelling.assert_((step(rho, dt) >= 0).all())
```

**Build list:**

- **Harness primitives.** `any_array`, `any_scalar`, `assume`, `assert_` defined as JAX primitives with abstract-eval rules, so they appear as ordinary equations in the traced jaxpr. Fixed shapes only.
- **The query object.** Harness jaxpr → `(B, R)`: boolean-output jaxpr plus region. The sole input to every engine, now and forever (commitment 1).
- **Z3 encoder.** Walk `eqns` with a `dict[Var, Z3Expr]`; support only what this demo emits — expect roughly: elementwise `add/sub/mul/div/neg`, comparisons, `select_n`, `reduce_and`/`reduce_min`, `broadcast_in_dim`, and whatever `jnp.roll` actually lowers to (likely `concatenate` of `slice`s, possibly `gather`). Discovering the real lowerings is part of the PoC's value. Hardcoded — no registry yet.
- **Arithmetic fragment, in two turns.** With `dt` fixed, every product involves a constant → QF_LRA. With `dt` symbolic, the `dt·ρ` terms are bilinear → QF_NRA, which Z3's nlsat dispatches instantly at N = 8. Do linear first, then flip `dt` symbolic.
- **Counterexample replay.** Z3 model → concrete arrays → call the *jitted* function → confirm the assertion actually fails. A counterexample that doesn't reproduce is an encoder bug; replay is the encoder's unit test.
- **Vacuity guard.** A sampler must exhibit at least one point satisfying `R`; otherwise the verdict is "vacuous," never "verified." (Kani's `cover`, minimal edition.)

**Definition of done:**

1. With the CFL assume: UNSAT — positivity proven for all ρ ≥ 0 and all `dt` in range.
2. Without it: a model with `dt > dx/u` whose replay under `jit` produces negative density — a physical blowup from a two-line harness.
3. Contradictory assumes → verdict "vacuous."
4. *Stretch:* bisect on the `dt` bound to recover a certified `dt_max ≈ dx/u` automatically — a preview of certified parameter ranges.

**Deliberately excluded:** registry, coverage report, interval domain, control flow, IEEE-754 semantics, everything else. The PoC is allowed to be a single file.

## Stage 1 — the wedge: index safety on real code

The first version a stranger can use. Kills the silent clamp-and-drop bug class.

- **Primitive census + corpus.** Trace the examples directories of jax-md, jax-cfd, diffrax, numpyro; let the frequency table pick registry coverage order instead of guesswork. The corpus doubles as the differential-testing bed (Stage 2) and, later, the credibility artifact. Can start in parallel with Stage 0 — it's a weekend of tracing.
- **Transfer registry.** Per-primitive × per-domain; unknown → ⊤, never a crash (commitment 4 made real).
- **Coverage report.** Which primitives fell to ⊤, where precision died. This is most of the UX: it turns "unknown" from an insult into a work item.
- **Integer fragment.** In-bounds obligations for `gather`, `scatter`, `dynamic_slice`, `dynamic_update_slice`.
- **checkify as spec.** Reuse its error conditions (OOB, div-by-zero, NaN) instead of inventing a spec language. Milestone: *every checkify error statically unreachable* on real functions.
- **Fuzzer fallback.** Region-aware sampling on unknown/timeout — strictly better than blind property testing because it knows the assumes — and the guarantee of non-empty output.
- **Field test.** A jax-md neighbor list and a jax-cfd stencil/halo kernel: prove in-bounds, or exhibit a replayable OOB that `jit` silently clamps. The field test *is* the sales pitch.

## Stage 2 — floats, induction, first physics proofs

- **Interval domain** over the registry: elementwise, `dot_general`, reductions, `select`, structural ops. Forward NaN/Inf-freedom proofs.
- **NRA backend + the inductive flagship.** Quadratic invariants in few variables via nlsat. Flagship: leapfrog oscillator, `E(x, v) ≤ E_MAX` inductive. This forces the right division of labor: intervals *cannot represent* the energy ellipse (its bounding box has a corner at 2·E_MAX — spurious failure at dt = 0), and zonotopes can't either (centrally symmetric). Quadratic invariants live SMT-side; positivity-style linear invariants are cheap in any domain.
- **custom_vjp primal consistency — the second flagship.** JAX does not check that a `custom_vjp`'s `fwd` returns the same primal as `f` (verified on jax 0.10.2: a lying `fwd` propagates its value through `value_and_grad` with no error, and the grad trace carries only `fwd`'s math — `f`'s is absent entirely). Every plain call executes `f`; every training loop executes `fwd`; no existing tool checks the two agree, and a disagreement has no runtime symptom. Compile it as the `|f − fwd_primal| ≤ tol` equivalence query (commitment 1). Corollary, stated once and inherited by everything: *everything inside a grad trace comes from the grad trace — primal properties included*; verify grad-context properties on `grad(f)`'s jaxpr, never by transferring a verdict from `f`'s. See `design/transparent-primitives.md`.
- **Induction as a harness pattern.** Verify `step` directly; no `scan` support required (commitment 6). A symbolic `dt` in the same harness yields "certified stable up to DT_MAX."
- **Transfer tiers.** Exact / sound / heuristic declared per transfer function and surfaced in the coverage report (commitment 5).
- **Differential testing.** Concrete corpus runs must land inside computed bounds. This catches transfer-function bugs that counterexample replay can't — replay only exercises paths where a violation was already found.
- **PRNG as bounded adversary.** Under the harness, rebind the sampler to `any_array` with assumed bounds — functional PRNG makes it a one-line swap. Union bounds and probabilistic certificates come much later.
- **Soundness plumbing, first pass.** Checker-path interval arithmetic with outward rounding (`nextafter` widening, or MPFR on CPU). XLA exposes no rounding-mode control, so a *jitted* analyzer is unsound by default — acceptable for now, resolved under the GPU-scale item.

## Medium — precision, structure, differentiators

- **Affine / zonotope domain.** The biggest single precision jump; intervals lose correlated terms almost immediately. Fixes correlation, not curvature — quadratic invariants stay SMT-side.
- **Control flow.** `cond` → join or ITE; `scan` → unroll small / invariant large; `while_loop` reframed for this domain: these are convergence loops, the exit guard hands you the tolerance postcondition on the normal path for free, and the target that actually matters is the max-iters bailout — does it return something finite and flagged? Bounded unwinding stays as mechanism, rarely as spec.
- **Abstract-interpretation induction at scale + invariant inference.** Forward-propagate with widening → candidate invariant → check inductiveness → fall back to user annotations. Push-button first, annotations second; k-induction later.
- **Node contracts + assume-guarantee composition.** requires/ensures per node; verify locally, assume neighbors — without this, a coupled multiphysics jaxpr is one unverifiable blob. Also the scaling trick for conservation: global energy over N cells is NRA in 2N variables (hopeless), but flux-form schemes conserve *structurally* — prove per-face flux antisymmetry locally and global conservation follows by telescoping.
- **Gradient verification.** Properties of `jaxpr(grad(f))`: the backward-NaN class (`sqrt`/`abs`/`norm`/`arctan2` at 0 — clean forward, NaN backward, survives all forward testing), Lipschitz bounds, region-wise symbolic gradcheck. Structurally out of reach for Kani-descendants.
- **Equivalence checking.** `custom_vjp` vs the autodiff'd jaxpr; surrogate vs reference solver; fused kernel vs reference — compiled as `|f − g| ≤ tol` queries (commitment 1), not built as a subsystem.
- **Linear bound propagation (CROWN-style) for learned nodes.** MLP surrogates and GNN corrections inside classical solvers; hybrid graphs need both engines behind the one query interface.
- **Certified parameter ranges.** Bisect on an assumed bound to output a *number* — stable step size, CFL constant, relaxation factor — not just a verdict.
- **dReal / δ-decidable backend.** Transcendentals over reals; δ-sat as the honest verdict for physical properties (commitment 2).
- **GPU-scale analysis.** jit/vmap the analyzer itself and run bound propagation at simulation scale (jax_verify proved the pattern). Carries the soundness plumbing: epsilon-inflation under XLA, CPU/MPFR path for soundness-critical runs.
- **Proof caching in CI.** Key on jaxpr hash; re-verify only changed nodes. What makes the tool survive contact with a real repo.

## Long — hard and strategic

- **Shape generalization.** Parametric shapes, induction over reduction trees. Genuinely unsolved; sci-comp's fixed-per-run shapes are the gift that lets us defer it.
- **IEEE-754 semantics + rounding bounds.** FPTaylor-style per-step error, plus the margin-absorption argument connecting real-arithmetic proofs (commitment 2) to float execution.
- **Translation validation to HLO.** Verify what runs, not what traced. Until then the trust boundary stays explicit.
- **Pallas / custom-kernel verification.** Alive2-for-JAX; kernel authors would actually use it.
- **k-induction, CEGAR, refinement loops.**
- **Polyhedral / SOS domains; neural Lyapunov and barrier-certificate checking.**
- **Coupled-system stability.** Every node stable ⇏ partitioned coupling stable (added-mass instability is the canonical case). The multiphysics-native question; nobody owns it.
- **Supermartingale / drift certificates.** The research half of the stochastic story, for when bounded-adversary noise stops being enough.
- **Machine-checkable proof certificates.** An artifact an auditor validates independently rather than "trust my solver run" — required the moment this touches a regulated pipeline.
- **Sharded programs.** `shard_map` / `pjit`.

## Closing

The arc of this document: index safety buys trust on day one with pure integer reasoning; inductive invariants deliver the scientific-computing headline — properties certified over horizons no test can reach; and the differentiators (verified gradients, equivalence checking, coupled stability) are reachable only from where this project stands, because JAX transforms are first-class objects and the analyzer speaks jaxpr. Every stage is scoped to ship something usable — the fuzzer floor means the tool never returns nothing, and every verdict names its assumptions. Immediate next steps: work through autodidax (Stage 0's interpreter core and harness primitives are its Part 1–2 machinery, applied), then build Stage 0 exactly as specified; the founding milestone is a README with one green proof and one red, replayable counterexample. Treat this as a living document — in particular, let the Stage 1 primitive census overrule any tiering guess in here that it disagrees with.
