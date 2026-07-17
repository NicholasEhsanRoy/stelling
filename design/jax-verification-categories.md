# The JAX verification categories

**Status:** evidence-backed taxonomy, written overnight 2026-07-17/18.
jax 0.11.0 (probes), census jax 0.10.2→0.11.0. Every live claim carries a
measurement made for this document or earlier in this repository; the
instruments are `stelling.census` (8 harnesses, 1820 equations),
`corpus/rule_provenance.py`, `corpus/category_probes.py` (six probes with
controls), ecosystem-source greps over the installed corpus libraries, and
two pre-registered tracker reads (`design/tracker-probe.md`,
`design/tracker-probe-2.md`; registrations committed before reading).
Kani's category list exists because Rust's hazards are enumerated; this is
the JAX equivalent, and **the dead categories are half the artifact** —
what JAX *cannot* get wrong is why jaxpr is a cleaner verification target
than MIR.

Nothing in this document licenses a build. Categories a verifier cannot
reach are still categories.

---

## The frame

The calibration set (four hypotheses tested in this repository):

| hypothesis | outcome | mechanism |
|---|---|---|
| index bounds | dead | 17/19 sites hand-clamped |
| NaN at zero | dead | `safe_mask` / custom_jvp norm rules |
| custom-rule primal agreement | dead | `check_grads` catches it (probed, with controls) |
| long-horizon failures | survived | 20 hits, 4/5 trackers, 487d/258d costs; detective defences only |

The frame that retrodicts all four and predicted tonight's measurements:

> **A hazard class is dead when the hazard and its defence are both
> site-local and the defence's correctness is checkable at the site — by
> inspection or by a single test. A class is alive when its property
> quantifies globally — over trajectories, input regions, seeds, or
> component compositions — or when the site-local defence's own
> correctness requires arithmetic no reviewer can eyeball.**

Local hazards get cheap local defences because the defence is writable at
the site of the hazard; ecosystems converge on them (measured below).
Global properties have no site to defend at — they can only be *detected*,
which is why every long-horizon incident wears a detective defence
(`max_steps`, NaN checks, R-hat) that reports the death without preventing
it.

Predictions this frame made that tonight's measurements then confirmed:

- *PRNG key reuse is local and lintable → should be dead by tooling.*
  Probed: reuse is silent, **and JAX ships `jax_debug_key_reuse`** — the
  defence exists upstream. Dead.
- *vmap-of-cond primal leakage is defended at the site by `select` itself
  → should be a non-hazard.* Probed: `vmap(cond)` over `[-1, 4]` with a
  `sqrt` branch yields `[0, 2]`, no NaN — **the folklore hazard does not
  exist at the value level**; the real leak is in differentiation (below).
- *Integer overflow is local → either defended or incident-bearing.*
  Probed: computed int32 overflow wraps silently (50 000² = −1 794 967 296),
  boundary literals are loud (`OverflowError`); tracker sweep across five
  projects: **zero field incidents of arithmetic overflow**. Dead.
- *Termination and interaction properties are global → should be alive.*
  The registered `stuck`/`hang` reads produced them on cue (below).

Where the frame earns its second clause: diffrax#632's bug is one
expression — `t + dt` — and it took **258 days**, because `fl(t+dt) = t`
at f32 is arithmetic no reviewer eyeballs (probed: t=1e8, dt=1.0). Local
site, uncheckable defence: alive.

And its sharpest output: **the wedge and the guard experiment are the same
sites under different quantifiers.** "Is this index in bounds" is local
and hand-defended — dead. "Does this hand-defence ever fire over the
region" is global — alive. The census's 17 clamps killed one category and
created another.

---

## Dead categories

Equal prominence, each with its evidence. "Dead" means *not a verification
target for anyone* — either impossible by construction or already defended
by something a point method verifies.

| # | category | verdict | evidence |
|---|---|---|---|
| D1 | **Memory safety** (use-after-free, buffer overrun as corruption, aliasing) | **absent by construction** | jaxpr is pure and first-order: 74 observed primitives across 1820 corpus equations, all functional — the "mutating" ones (`scatter`, `dynamic_update_slice`) return new arrays. No pointers, no aliasing, no mutation to corrupt. This is the headline difference from MIR: Kani's flagship category has no referent here |
| D2 | **Data races / concurrency UB** | **absent by construction** | same purity; parallelism is XLA's scheduling of a pure dataflow graph, not shared mutable state |
| D3 | **Shape, rank, and dtype errors** | **dead at trace time** | every jaxpr is shape-typed before it runs: 5 610 avals across the corpus, every shape a static int tuple (asserted mechanically; the transcriber raises otherwise). The analog of Rust's type system, not of a runtime verifier |
| D4 | **Null / uninitialized reads** | **absent by construction** | no such construct exists in the IR |
| D5 | **Index out-of-bounds** | **dead twice** | as a *crash*: absent by construction (gather clamps, scatter drops — HLO semantics). As *silent wrongness*: hand-clamped at 17 of 19 real sites (census interrogation); the property is true by construction wherever someone wrote the guard, and someone wrote the guard everywhere we looked. The residual live question — *do the guards fire?* — is L3, a different quantifier |
| D6 | **NaN at zero (backward-pass NaN)** | **dead — defended, two idioms** | measured defence surface: jax-md ships **11 distinct `safe_*` symbols across 18 files**; the equinox family ships zero `safe_*` because it implements the same defence as **custom_jvp rules** (lineax `_two_norm` read: a hand derivative for the norm at 0 — 7 of the corpus's 8 custom rules are this idiom). Probed: coincident-particle gradient is finite |
| D7 | **Custom-derivative primal disagreement** | **dead — the defence is a point method and it works** | probed with controls: `check_grads` passes honest rules, catches lying tangents, **catches lying primals**. Population correctly counted: 8 distinct rules in the corpus, all library-authored. Residual: rules never run through `check_grads` — unprobed, recorded, not a category until measured |
| D8 | **PRNG key reuse / correlation** | **dead — upstream tooling** | probed: reuse is silent (identical draws, no error), and `jax.config.jax_debug_key_reuse` exists. The right tool is a linter/runtime checker and JAX ships it |
| D9 | **Integer arithmetic overflow** | **dead on field evidence** | probed: silent wrap in-trace, loud `OverflowError` for boundary literals, default index dtype int32. Registered sweep over five trackers: zero arithmetic-overflow incidents. (jax-md's ten `overflow` threads are its neighbor-list *capacity-flag protocol* — see L3) |
| D10 | **vmap-of-cond value totality** | **dead — folklore** | probed: batched `cond` becomes `select`, and `select` masks the untaken branch's value correctly. The genuine hazard is the *derivative* of the untaken branch — the where/NaN-gradient surface, which lives in L3 |
| D11 | **Statistical sampler correctness** (wrong posterior, acceptance drift) | **out of scope, honestly** | the probe-1 property test excluded both wild instances (numpyro#154, #1786): the only properties are statistical equality with a reference or expectation bounds — supermartingale territory, named on the founding roadmap's far end, not reachable by region methods |
| — | **Device nondeterminism** (GPU scatter atomics; platform variance) | **unmeasured — unranked** | two field hints (numpyro#1120 same-key-different-machines; #1427 pmap changes results) and no probe possible on CPU tonight. A paragraph, not a finding; ranks below everything measured |

Performance and resource exhaustion are excluded as non-correctness.

---

## Live categories, ranked

Each ran the loop: property → incumbent → existence → failure → attack →
customer → output shape → falsifier.

### L1 — Trajectory invariants (the long-horizon core)

- **Property.** `∀ x ∈ R, ∀ step: state stays in S` — finiteness,
  positivity, support membership, bounded energy — proven inductively on
  the step map; horizon-independent (founding commitment 6).
- **Incumbent.** Detective only, measured: diffrax carries 10 `error_if`
  sites and 70 `max_steps` mentions; users hand-build NaN terminators
  (diffrax#290, numpyro#956). A NaN check reports the death at step 800k;
  it neither prevents it nor says for which inputs.
- **Existence/failure.** ~9 of the 20 probe-1 hits (diffrax#223/#368/#417,
  blackjax#D416/#973, numpyro#249/#1133/#1360, jax-md#339's invariant
  face), plus diffrax#596. Costs: blocked training pipelines (c16),
  487-day narrowings.
- **Attack.** Survives the frame (global quantifier; no site to defend).
  Sharpest counter: ~half the instances are *solver-state* invariants
  (adaptation loops, step controllers), not user physics — which narrows
  the audience but strengthens the composition story.
- **Customer.** Maintainer-first (their control loops), downstream via a
  region on their own model.
- **Output shape.** Verdict-shaped as stated; **artifact-shaped as an
  envelope** ("stable for τ < 0.0037 on T₂ ∈ [10, 300] ms") — the
  bisection rendering competes with nothing: a sampler can find an
  envelope's inside, nothing else can bound it.
- **Falsifier for a future value model.** The inductive checker, run on
  the 9 recorded instances' reconstructions, certifies none of them and
  finds no witness — machinery exists, value doesn't.

### L2 — Termination and progress of adaptive loops

- **Property.** Ranking functions: `∀ x ∈ R: accepted-step count ≤ N`;
  `dt ≥ dt_min unless error > E`; "a clipped endpoint step is never
  re-rejected indefinitely"; loop measures strictly decrease.
- **Incumbent.** `max_steps` bailouts — 70 mentions in diffrax alone — the
  canonical detective defence: it converts non-termination into a loud
  death without saying for which inputs.
- **Existence/failure.** 5 of 20 probe-1 hits (diffrax#194/#207/#386/#752/#756)
  plus tracker-2's harvest: #707 (compile-time hang), #185 (controller
  thrash on a switched system). `stuck` alone: 15 diffrax + 16 numpyro
  threads. diffrax#386's author narrowed a step-explosion themselves over
  "large sets of solves."
- **Attack.** Different machinery from invariants — ranking synthesis is
  **absent from the founding roadmap** (named gap, live finding #2). The
  while-loop reframe in the roadmap ("the exit guard hands you the
  tolerance postcondition") addresses the *normal* path; these failures
  are all on the *abnormal* one.
- **Customer.** Almost purely maintainer (controllers, adapters, event
  loops).
- **Output shape.** **Artifact-shaped**: "this controller terminates
  within N steps for all inputs in R" is a bound nobody today can produce.
- **Falsifier.** Same template as L1, over the 7 recorded instances.

### L3 — Guard and detector obligations

- **Property.** Three sub-forms over the same surface: **liveness** (does
  this clamp/detector ever fire on R — dead defences are misinformation);
  **wiring** (a status flag, once raised, reaches a consumer); and
  **completeness** (the detector fires *whenever* the condition holds —
  event detection, neighbor-list supersets).
- **Incumbent.** Nothing. Nobody checks defences; they are written and
  trusted.
- **Existence.** The census's `select_n` ×188 at 5/7 targets is the
  manual case-split surface; 17 clamp sites already classified; diffrax
  has 71 `where`/`select` source sites and jax-md's capacity protocol
  produces 5.6% of its tracker in usage burden.
- **Failure — the strongest new evidence of the night.** Three
  independent wild instances of *defences failing as defences*:
  blackjax#969 (NaN detector provably never fires — shipped), jax-md#141
  (overflow detector misbehaves when there is no overflow), blackjax#925
  (maintainer-filed: inner L-BFGS non-convergence silently ignored →
  biased marginals). Plus the completeness face: diffrax#507 (events
  intermittently missed), jax-md#339 (487 days). Plus the derivative face
  of the where-surface: probed (naive `where`-grad NaNs at 0; double-where
  defends); wild: diffrax#363, #701, #742.
- **Attack.** The frame puts this at the intersection: local defences,
  global questions. A green liveness verdict says the guard *can* fire —
  it does not say the guarded computation is right; a Dead verdict may be
  a shrug (belt-and-braces). Both attacks are recorded in the registered
  guard experiment, which is this category's already-written value model.
- **Customer.** Both: maintainers own the guards; downstream owns the
  regions.
- **Output shape.** Verdict-shaped (dead / fires-by-design /
  fires-unexplained + witness).
- **Falsifier.** Already registered — `design/guard-experiment.md`.

### L4 — Float-exact control arithmetic (the ℝ blind spot)

- **Property.** FP-exact progress and grid properties on *scalar
  controller arithmetic*: `fl(t + dt) > t` over the region; realized
  `StepTo` times equal the requested grid; cast thresholds
  (`float32(dt) ≠ 0`).
- **Incumbent.** One library's hand machinery: diffrax carries **11
  `nextafter` sites** (and the census's rank-six primitive, ×100, is this
  machinery seen from the IR). Everyone else: nothing — jax-md#343
  silently drops dt=1e-8 through an f32 cast.
- **Existence/failure.** diffrax#632 (**258 days**), #657 (silent wrong
  result), jmd#343. Probed: t=1e8, dt=1.0, f32 → `t+dt == t`.
- **The finding this absorbs (live finding #1).** Design commitment 2
  proves #632 and #657 **green**: over ℝ, `t+dt > t` is a theorem — the
  false VERIFIED this architecture exists to prevent, produced by its own
  arithmetic stance. The category is reachable *without abandoning* the ℝ
  stance: these obligations are scalar, few, and controller-owned — a
  small FP-exact fragment (QF_FP over f32/f64 scalars is decidable and
  tiny at this size) scoped to time/step arithmetic, while state stays
  ℝ-with-margin. **Consequence for SOUNDNESS.md when a verdict type
  exists: the `arithmetic` stamp must be per-obligation, not per-verdict**
  — a verdict whose chain mixes ℝ-margin state reasoning with FP-exact
  controller reasoning has to say which obligations got which.
- **Attack.** Small: three wild instances, one library already defended.
  It survives on sharpness, not size — the defence is uneyeballable
  (clause 2), the incidents are expensive, and the fragment is cheap.
- **Customer.** Maintainer (controllers), and any user with small-dt
  physics.
- **Output shape.** Verdict-shaped; artifact-shaped as "safe dt range for
  this controller at f32".
- **Falsifier.** The FP fragment, pointed at reconstructed #632/#657/#343,
  fails to distinguish the buggy from the fixed versions.

### L5 — Transform and compilation equivalence over regions

- **Property.** `jit(f) ≈ f`, `vmap(f) ≈ stack∘f`, same-device ≈
  cross-device, within tolerance over a region — link 7's family.
- **Incumbent.** JAX's own test suite, per-primitive; nothing end-to-end.
- **Existence/failure.** jax-md#92 (jit/no-jit trajectories diverge after
  ~1000 steps, similar at 1–10 — open, c9); tracker-2 adds **dfx#692
  (vmap changes the number of accepted steps — batching alters adaptive
  control flow)** and npy#1427 (pmap changes NUTS results).
- **Attack.** Honest: **out of near reach.** Region-quantified
  equivalence of compiled artifacts is translation validation; the only
  instruments this project has named are differential testing and
  fuzz-on-verified. The category is real — the incidents are silent
  wrongness with long-horizon amplification — and it belongs to the
  taxonomy precisely because the taxonomy is about JAX, not about what
  stelling can build. Ranked low for reachability, not for reality.
- **Customer.** Everyone; producible today only as differential evidence,
  not verdicts.
- **Output shape.** Artifact-shaped (an equivalence certificate or a
  divergence witness with the step at which it grew).
- **Falsifier.** n/a until an instrument exists; the honest state is
  "named, evidenced, unreachable."

### L6 — Composition contracts (assume-guarantee)

- **Property.** Maintainer publishes `requires` on the user's `f` and
  `ensures` on the loop; downstream checks their `f` against the contract;
  interaction invariants (clipping × rejection, events × adjoints) get
  stated at the interface they cross.
- **Existence/failure.** The interaction incidents are real and expensive:
  dfx#756 (two locally-correct policies compose into an infinite loop),
  #752 (nonlinear solver × PID contract violation), **#729 (open, c15:
  post-event gradients "now always wrong" — the events×adjoint seam)**.
- **Attack — hardest, as ordered, because it is the most attractive
  synthesis on the table.** (1) The failures cited are *discovered at*
  seams, but a contract must be *statable*: the biggest real assumption in
  the corpus — "f is non-stiff for this explicit solver" — is spectral,
  input-dependent, and not writable as a region property; the statable
  subset (finiteness, positivity, bounds, Lipschitz-via-intervals) is
  real but thin. (2) No maintainer has asked for it; zero demand sampled.
  (3) It inherits every burden of L1–L3 plus an adoption story. Verdict:
  **conditionally live — as the delivery mechanism for L1–L3 verdicts
  across the maintainer/downstream boundary, not as a standalone
  category.** It gets no value model until an L1/L2/L3 result exists to
  compose.

---

## Live finding #2, verified: the 20 hits by technique

My split of the probe-1 hits (per-hit, from the recorded properties):

| technique | hits | which |
|---|---|---|
| inductive trajectory/state invariant | 9 | dfx#223 #368 #417, bjx#D416 #973, npy#249 #1133 #1360, jmd#339 (invariant face) |
| termination / ranking | 5 | dfx#194 #207 #386 #752 #756 |
| float-exact | 2 | dfx#632 #657 |
| detector completeness / guard liveness | 2 | dfx#507, bjx#969 |
| equivalence (trace↔compile) | 1 | jmd#92 |
| one-shot region safety (non-inductive) | 1 | npy#552 |

Close to the by-eye split it corrects (10/1/4/2/2): the real change is
one hit moving to equivalence (jmd#92 is link-7's family, not an
invariant) and #339 sitting on the invariant/completeness border. The
structural conclusion stands and sharpens: **a quarter of the measured
demand is termination machinery that no current plan contains.**

---

## Live finding #3, answered

jax-cfd has **32 issues total** against diffrax's 489 and numpyro's 872.
At diffrax's probe-1 hit rate, 32 issues predict ~0.7 hits; observing zero
is unremarkable. **The tracker silence is base-rate-dominated** — the
corpus artifact explanation wins. What remains true from the census: its
traced stencil path has no wedge primitives and builds halos by
`jnp.pad`+static slice. The style claim about stencil code rests on one
traced path of one quiet library, and should be re-tested on a second
stencil code before anything leans on it.

---

## What the morning holds

Two large live categories (**trajectory invariants**, **termination of
adaptive loops** — the second absent from every existing plan), one
meta-category with its experiment already registered (**guard/detector
obligations**, now carrying three wild instances of defences failing as
defences), one small sharp category that the project's own arithmetic
stance currently proves falsely green (**float-exact control
arithmetic**), one real-but-distant (**transform equivalence**), and one
conditional delivery mechanism (**contracts**). Six kills by
construction, five by measurement, one honest "unmeasured."

The frame held: 4/4 retrodicted, four predictions confirmed tonight, one
refinement earned (uneyeballable local defences stay alive). Nothing here
licenses a build; the strongest claim on offer is that the next value
model, if one is written, should be aimed at L1+L2 with L3's experiment
as the cheapest first result — and that claim is an argument, ranked
below every measurement in this file.
