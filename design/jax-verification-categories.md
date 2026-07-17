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

The frame, as sharpened after the 2026-07-18 readings (the original
clause-2 wording — "arithmetic no reviewer can eyeball" — was a judgement
call that could absorb anything; this version is decided by grep, defence
kind, and trackers):

> **Two questions about one author. A hazard class is dead iff the author
> can both *notice* the obligation and *discharge* it locally — and the
> resulting preventive defence idiom is ecosystem-pervasive. It is alive
> iff the obligation out-scopes its site (quantifying over regions,
> trajectories, batches, seeds, or compositions), or only detective
> defences exist (noticed, undischargeable), or the defence exists but has
> not spread (noticed by some authors, silently absent elsewhere).**

| defence at the site | incidents | reading |
|---|---|---|
| preventive, local, pervasive | — | **dead** — noticed and discharged |
| detective or protocol only | any | **alive** — noticed, not dischargeable locally |
| absent | present | **alive** — not noticed (`t + dt` announces nothing) |
| absent | absent | **no obligation — or silent and unnoticed.** Trackers cannot tell these apart (incidents require loud failures); only a defence census or a region instrument can. The wedge's original motivation lived exactly here |

Three amendments earned by measurement: **notice aggregates per
ecosystem, not per author** (diffrax carries 11 `nextafter` sites while
jax-md#343 falls to the same class undefended — the *class* stays alive
until the idiom is pervasive, the way clamps at 17/19 sites and `safe_*`
across every probed library are); **mixed defences classify by their
strongest preventive component** (capacity's allocation heuristic is
preventive but per-configuration; the regional obligation has only the
flag — cell 2, and reading B showed *why*: the obligation out-scopes the
site on the quantifier, not the arithmetic); and **cell 4 carries its
silence caveat** or it becomes the frame's escape hatch.

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

Calibration note, recorded as ordered: three of the four were strong
predictions; the overflow one was a disjunction
(*defended-or-incident-bearing*) that could only have failed on
undefended-and-silent — not vacuous (that is the wedge's shape), but the
soft one of the four.

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

### L5 — Equivalence over regions (two reaches, split 2026-07-18)

**L5a — custom-rule region agreement (near reach, registered and read).**
The point kill stands: `check_grads` catches lying primals at points. The
region property is a new candidate, registered before reading
(`design/registration-rules-and-capacity.md`) — and the reading returned
a narrow survival with a kill inside it. All six concrete corpus rules
return their own primal verbatim: **the primal-agreement obligation has
zero surface in the wild** — every author discharges it by construction,
and the Stage-2 "primal consistency" flagship is reshaped accordingly.
What survives is precise: two lineax norm rules whose tangents are
*intended* to equal the true Jacobian away from a stated excluded set
(`∀ x, ‖x‖ ≥ ε: tangent_out = ⟨x, tx⟩/‖x‖`) — writable, non-circular,
non-Intentional on the region, with the Intentional set handled by the
guard experiment's bucket discipline. Corpus: 2 rules (+ 2 wrapper hosts
whose wrapped populations are unresolved — a measurement-granularity
gap). Small, cheap, reachable; and it settled the check_grads-coverage
question for a second better reason — point tests applied to `nextafter`
would fail its *Intentional* tangent correctly, so testing cannot even be
aimed at the defence layer without the bucket.

### L5b — Transform and compilation equivalence over regions

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
  (3) It inherits every burden of L1–L3 plus an adoption story.
- **The capacity consolidation was tested and failed (2026-07-18).** The
  ten jax-md capacity threads — the highest term concentration measured
  anywhere (5.6% of that tracker) — were read in full against a
  registered criterion: they decompose into 1 bound-wanted, 2
  protocol-burden, 1 detector-defect, 2 silent construction-correctness
  bugs, and 4 unrelated. The capacity certificate has exactly **one**
  measured demand instance (jmd#192, where even the maintainer says no
  optimal policy is known) — not a first customer. The weight of that
  cluster lands in L3's completeness sub-form instead (#165 and #191 join
  #339 and #507: four wild silent-construction instances across two
  libraries). Verdict unchanged and now measured:
  **conditionally live — the delivery mechanism for L1–L3 verdicts
  across the maintainer/downstream boundary, not a standalone category.**
  No value model until an L1/L2/L3 result exists to compose.

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

## The ranking, after the 2026-07-18 re-rank

1. **L1 — trajectory invariants.** Unmoved at the top: 9/20 measured
   incidents, solver-state-first, the envelope rendering with no
   incumbent.
2. **L2 — termination/progress of adaptive loops.** Unmoved: 5/20 + the
   `stuck`×31 surface, absent from every existing plan, artifact-shaped.
3. **L3 — guard/detector obligations.** **Strengthened by both
   readings**: the completeness sub-form now has four wild
   silent-construction instances (#165, #191, #339, #507) plus three
   defences-failing-as-defences (#969, #141, #925); its experiment is
   already registered and remains the cheapest first result.
4. **L4 — float-exact control arithmetic.** Aliveness reason sharpened by
   the frame: the defence exists and has not spread (diffrax's 11
   `nextafter` sites vs jmd#343 undefended) — and the defence's own AD
   behaviour is an Intentional lie, so even point-testing the defence
   layer needs the bucket discipline.
5. **L5a — custom-rule region agreement.** New, small, near-reach: two
   named lineax rules, property registered, primal surface zero.
6. **L5b — transform/compile equivalence.** Real, evidenced, distant.
7. **L6 — contracts.** Conditional, now *measured*: the capacity
   consolidation failed its registered criterion; one demand instance is
   not a customer.

Six kills by construction, five by measurement, one honest "unmeasured" —
plus one kill *inside a survival* (primal agreement: zero wild surface;
every author discharges it by construction).

## The byproduct policy

**Dead killed the proposition, not the feature.** L3 is the wedge's 17
sites under a different quantifier — same gather transfer function, same
index cone, same integer SMT. Building L3 builds everything the wedge
needed, so in-bounds ships as a **free byproduct**: same query, different
obligation, zero marginal code. The honest limit, recorded: shared
machinery always, shared obligation sometimes — `isfinite(x[i])` holds
whether or not `i` is in bounds (clamping still reads a finite element);
in-bounds matters where the bound is index-dependent (completeness,
geometry, #339's neighbour-list superset). Do not overclaim the
subsumption.

And the escape hatch this section exists to close: *"but it's valuable
for new projects"* is **structurally unfalsifiable by every instrument
this project has built** — trackers measure incidents, censuses measure
code, and neither can measure the bug never written because a tool made
it unthinkable. Every dead category can be resurrected by that sentence,
forever, without evidence, and it will be available on the worst possible
day. So, as policy:

> **Dead categories ship as byproducts of live ones, and never appear in
> a pitch. Changelog, not proposition.** If someone gets value from one,
> that is data arriving from outside — the only way it could be trusted.

The one legitimate exception is history, because history is measurable:
the archaeology probe (`design/archaeology-probe.md`) ran under its own
registration and returned the **scar-tissue band** — three defence
classes with receipts across two libraries (jax-md's `safe_mask`: "Fixed
NaNs due to a JAX change in the behavior of np.sqrt at 0"; diffrax's ULP
endpoint guard fixing the 258-day #632; jax-md's boundary sentinel fixing
#378) — plus the measured transfer mechanism (the `safe_*` idiom, minted
in the 2020 fix, applied proactively in 2022/2025/2026 feature work) and
one arc nobody ordered: **the #632 fix's guard is the cause of #756's
infinite rejection loop** — defences composing into new failure modes,
dated, in one repo. A fact about the kills; it reopens none of them.

**A hypothesis about this ranking — generated, not tested (2026-07-18).**
The archaeology arc (#632's ULP guard caused #756's infinite rejection
loop) is an **L3 artifact causing an L2 failure**. If that generalises,
the two top-ranked live categories are not independent — the guards
generate the termination hazards — and the natural first result is not L3
in isolation but **L2 on guarded control loops**: *does the guard
terminate?* One dated instance with a traced causal chain, in one repo,
is all the evidence this has. The 20 probe-1 hits were classified once
under a pre-registered taxonomy and are **not** re-read for this pattern
(asking a new question of old data generates, it cannot test); the banked
terms stay banked. Testing it needs new data under a new registration.
The value model aiming at this ranking should know the hypothesis exists
and know it is untested.

The frame held and got sharper: 4/4 retrodicted, four predictions
confirmed (three strong, one soft disjunction — recorded), and the
judgement-shaped clause replaced by the notice/discharge/pervasiveness
form whose cells are decided by grep, defence kind, and trackers, with
cell 4's silence caveat keeping it honest. Nothing here licenses a build.
The strongest claim on offer: the next value model, if one is written,
should be aimed at **L1+L2, with L3's registered experiment as the
cheapest first result and L5a as a two-rule add-on corpus** — and that
claim is an argument, ranked below every measurement in this file. The
value model itself is deliberately not written here; two of its inputs
arrived tonight and it deserves its own registration-first pass.
