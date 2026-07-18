# MADDENING, the CFL obligation — the first job, registered before it runs

**Status:** REGISTRATION, 2026-07-18. MADDENING ref pinned:
`849c39126724` (untracked `spikes/` present, irrelevant).

## Framing, corrected — then corrected again (2026-07-18, mid-run, before any result)

*First correction (at registration):* this is not a generalisation check —
the instrument-generalises framing was rejected.

*Second correction (owner-supplied fact, recorded while the run is in
flight and no result exists — the one window where a reframing cannot be
result-motivated):* **heat is a proof-of-concept node** — written to
demonstrate the MADDENING API, never hardened. Its CFL hole bit
*because* it is a demo, not because CFL-checking is where the real risk
lives. So this run is **not the usefulness test** either. What it is:
**the first imported-grade harness** — a capability first, not a
usefulness first. A clean VERIFIED here answers "can the tool check a
toy," which was never the question.

**The usefulness first is FVM** — the real solvers (FVM, LBM, BEM) live
in MIME; FVM's stability and mesh-skewness conditions are real,
validated, and scarred. FVM is the actual first job, a separate pass,
and it needs two inputs this run produces: the B-harness ⊤-census (the
registry-tail estimate for MADDENING-shaped code) and the FRAGILE
escapes the imported trace shakes out. It also needs what this pass must
not manufacture: **a chosen FVM condition with a real failure behind
it** — the provenance that made CFL a legitimate target.

**Reading discipline for this run's results, fixed before they exist:**
(A1)/(A2) verdicts report as *"mechanism demonstrated on a
proof-of-concept node; usefulness untested pending a real solver"* — the
⊤-widening discipline and the §10.8 control working on a live target,
**no usefulness claim attached**. (B) reports its two deliverables in
full — the census and the escapes — as the capability milestone and
FVM's inheritance. The outcome table below stays for mechanism
adjudication; its row-1 "first true positive / milestone" language is
**superseded by this section**: a VERIFIED (A1) is a mechanism check on
a toy, said exactly so wherever it appears.

**Held out, not counted:** MADDENING never enters E2a's denominator —
author's own code, a mechanized count there is uninterpretable (the
census's own rule). This is a usefulness test, reported separately,
wherever its result appears.

## The target — specific, dated, quoted

`MADD-ANO-002`, `src/maddening/nodes/heat.py` at the pinned ref:

- The obligation: *"CFL stability limit: dt < dx^2 / (2*alpha) —
  violating this produces silently incorrect results"* (heat.py:155);
  `ValidatedRegime("CFL", 0.0, 0.5, "dt * alpha / dx^2 must be < 0.5")`
  (heat.py:163).
- The hazard: *"CFL stability not enforced at runtime"* (heat.py:166) —
  and the update confirms it: `T_new = T + alpha*dt*laplacian +
  source*dt` (heat.py:443), **no check anywhere in the node**.
- The only enforcement in the repo is a hand convention in the shipped
  demo: `dt = 0.5 * dt_max_stable` (heat_diffusion_demo.py:45–47).
- Documented in March, hit in May (misdiagnosed as float32), unenforced
  today (`design/maddening-archaeology.md`).

Shipped demo config, quoted for the region and nonvacuity choices
(demo:38–47): `n_cells=20, length=1.0 → dx=0.05, α=0.01, dt=0.0625` —
CFL number `0.25`.

## The properties — stated here, by the main agent, from the artifacts

**(A1) The obligation as written, at the shipped config, with the
author's own margin** — the CI-candidate check:
`assert dt·α/dx² < 0.5`, with `dx = length/n_cells` transcribed from
heat.py:326, `(dt, α)` declared as the demo's values with a small
config-drift box around them (`dt ∈ [0.0625·(1±10%)]`, `α ∈
[0.01·(1±10%)]` — a drift envelope, not a fitted region). Expected:
**VERIFIED with ~2× margin** (CFL ≈ 0.25 vs 0.5, slack ≫ ulp).

**(A2) The §10.8-style control for (A1)** — proves the green is not
vacuous: the same predicate over `α ∈ [10⁻⁶, 1.0]` — **the node's own
ValidatedRegime for thermal_diffusivity** (heat.py:161) — with the demo
`dt`. This region contains CFL-violating configurations (α = 1.0 →
CFL = 25), i.e. exactly the May class: a config inside the validated
α-regime, outside the CFL constraint. Expected: **partially-false →
UNKNOWN, correctly** (the amended triage bucket), demonstrating the
check *fires* on regime-legal-but-unstable configs. Its role is control,
not claim.

**(B) The obligation's state-space content — the discrete maximum
principle, on the real code:** `∀ T ∈ [0,100]ⁿ, T_left/T_right ∈
[0,100], source = 0, demo dt/α: T_new ∈ [0,100]ⁿ` — posed on the
**imported** `HeatNode.update` traced via `any_pytree` (the box is the
demo's own temperature range, demo:41–42; `y0` = the node's
`initial_state`, T ≡ 0). Mathematically true at CFL = 0.25 ≤ 0.5 (the
update's T-coefficient `1 − 2c ≥ 0`). **Predictions, registered:** first
contact crashes or ⊤s on `.at[0].set` (scatter-class primitives — the
registry's wedge tier) and possibly node plumbing; and even with
coverage complete, the obligation lands **UNKNOWN, dependency-shaped** —
`T` appears in the base term and the stencil, a linear correlation
intervals discard; **affine's first real customer if confirmed**. (B) is
the depth probe; (A1) is the job.

## Pipeline — every registered discipline applies

Fidelity: (A1)/(A2) are hand-transcriptions **of the author's own written
obligation** with pointers (there is no library code to import for a
check that does not exist — a region method replacing a comment is the
entire value proposition); (B) is **imported** — the first
imported-grade harness ever attempted. The registry tail of (B) gets the
second-bill census (trivial / array-semantics / out-of-ℝ) — the first
real datum on what supporting a codebase costs on a target the tool was
not built around. The property test's five corrections apply to (A1),
including **⊤-widening** (predicted non-tautological: widening dt, α
makes CFL straddle 0.5) and **checked-against-the-incident** (the May
mechanism was a CFL violation — the misdiagnosis was float32; the
semantics classification examines whether float was ever the mechanism).
Semantics: predicted **ℝ-faithful with margin** for (A1) (real-arithmetic
condition, 2× slack; the node computes in f32 — noted, not the
mechanism). Verdicts ship with full stamps.

## Outcomes — fixed before the run

| outcome | reading |
|---|---|
| **(A1) VERIFIED, non-vacuous, ℝ-faithful, traced dx** | **The first true positive**: a real, dated, unenforced obligation checked over a region its author currently enforces with a comment — a candidate CI check. Reported as the milestone it is, **with the ⊤-widening result in the same breath** |
| **UNKNOWN** | bucketed per the triage; dependency-shaped on (B) is affine's territory and the predicted outcome; **which bucket is itself the highest-value finding** |
| **REFUTED** | on (A1): the shipped config violates its own obligation — scrutinised hard, then reported as the strongest possible result. On (A2): expected-partial, the control working |
| **won't pose** | the wall named (leaf type, closure, shape) — bounds MADDENING CI and is the honest input to whether this line continues |

No outcome fails the pass. The pass fails only if a result ships without
its ⊤-widening check, semantics class, or fidelity bucket.

## Who does what

**Fresh-context subagent** (blind to the incident history, to this
outcome table, and to which verdict would be convenient): install
maddening `--no-deps` at the pinned ref (never touching jax), transcribe
and trace the harnesses with the predicates handed verbatim, get (B) to
pose, census its ⊤ tail, add registry rows **only by census** with the
boundary-fix audit rule (hard rows decline + report; the main agent
adjudicates entry), fix FRAGILE posture escapes with witnessed
constructions. It must not read design/, docs, other corpus files, or
anything in the MADDENING repo beyond what importing the heat node
requires. CALMS is out of scope entirely.

**Main agent:** the properties above; the ⊤-widening run; the semantics
classification; verdict adjudication against the incident; what the
outcome licenses.

---

# Reading (2026-07-18 — `corpus/supply/maddening_cfl.py`, adjudicated under the mid-run reframing)

## The verdicts, with every discipline attached

**(A1) VERIFIED — and non-tautological, mechanically shown.** The
obligation `dt·α/dx² < 0.5` discharged over the ±10% drift envelope at
the shipped config; nonvacuity checked 4/4; coverage 25/25; and under
⊤-widening the obligation is **unknown** — the declared bounds are
load-bearing. Semantics, finalized: **ℝ-faithful with ~2× margin**
(CFL ≈ 0.25 vs 0.5; slack ≫ ulp; the check's arithmetic runs in declared
f64 points even though the node is f32 — the obligation is meta-level
real arithmetic). The May misdiagnosis is now closed from two sides: the
archaeology showed CFL-not-dtype was the fix, and the semantics
classification finds no float mechanism in the obligation — **the
float32 diagnosis pointed at nothing real for this property.**

**(A2) — the control refused, as designed.** Over the node's *own*
validated α-regime, the same predicate is **unknown** (a straddle:
CFL ∈ [2.5×10⁻⁵, 25] over the region). Analytic triage: **partially
false** — the property genuinely fails for α > 0.02 at the demo dt,
which is the May class exactly (regime-legal, CFL-violating). Together
A1+A2 are the mechanism demonstration in full: the check discharges on
the safe envelope, refuses on the violating regime, and is neither
vacuous nor tautological. **Labeled per the reframing: mechanism
demonstrated on a proof-of-concept node; usefulness untested pending a
real solver.**

**(B) — the capability milestone, and the cleanest dependency sighting
on record.** The real `HeatNode.update` — imported, traced,
transcribed (the transcription layer survived its first `scatter`:
`ScatterDimensionNumbers`, `GatherScatterMode`, all by existing rules),
propagated at **100% coverage** after one censused row. Both obligations
**unknown, undecided for exactly the 18 interior cells** — the two
scatter-written Dirichlet cells are definitely in-box — with the
interior at ≈[−50, 150] against [0, 100]. This is the registered
prediction confirmed at full coverage: the property is true
(CFL = 0.25 → the update's T-coefficient 1−2c = 0.5 ≥ 0, the discrete
maximum principle holds), and the UNKNOWN is **purely dependency-shaped**
— no registry residue, no key cone, nothing but the stencil correlation
intervals discard. **Affine's first real sighting: imported code, full
coverage, a true linear-correlation invariant.** Riders, in the same
sentence: MADDENING is held out, so this sighting counts toward **no
trigger** (the corpus trigger's valid count remains 0); it is evidence
about what affine would buy, recorded where the affine ordering lives.

## The second-bill datum — the first supporting-a-codebase cost number

heat.py's registry tail was **one structural row** (static-index
`scatter`, tier exact, definite-FALSE + three decline paths tested) plus
one dead `copy` avoided by the prototype-outside-trace pattern. The
FVM inheritance: the scatter row, that pattern, the
f64-point-constants technique (which keeps a check's arithmetic f64 over
an f32 node and makes `dx = L/n` a genuine traced `div`), and the
expectation — stated now — that FVM's tail is larger.

## Instrument note, recorded for the next vacuity registration

The run's ⊤-widening widened *every* declaration, including the
transcribed-constant points (n, L, 0.5) — a *stronger* widening than
region-inputs-only, so A1's non-tautology conclusion holds a fortiori
(verified analytically: widening only dt, α still straddles). But the
two variants differ in general: widening a *threshold* constant makes
almost any comparison straddle, weakening the test as a tautology
detector. A future vacuity registration should fix **which declarations
widen** (region inputs vs transcribed constants). Recorded, not
retrofitted.

## What this licenses

Per the reframed table: the mechanism works end-to-end on a live target
— declaration, tracing, transcription, propagation, stamps, controls —
and the capability milestone stands. **No usefulness claim. No count.**
The next pass is FVM (in MIME), and its prerequisites are now in hand:
the census method, the inherited rows and patterns, and the requirement
that its condition carry real provenance — a scarred, validated
constraint, not a manufactured one.
