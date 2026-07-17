# The control-flow hypothesis — registered before the build

**Status:** REGISTRATION, 2026-07-18. Committed before any control-flow
transfer is written. Exists only because `design/control-flow-census.md`'s
gate passed (4 of 6 control-flow-only ≥ 3). E2a landed Weak and **does not
license this** — it is a *new* hypothesis with its own falsifier, built
only after this registration.

## The hypothesis

> With `cond` / `select_n` (branch/case-split) transfers **and no
> discrete-step model**, the four control-flow-only properties become
> posable as loop-body invariants, and they mechanize.

(The census found `scan`/`while` full-loop transfers unnecessary for this
subset: a loop-body invariant is posed by declaring the carry as a box
and checking the *body* maps it into itself — discrete inductiveness,
straight-line once the branch primitives are handled. If a harness turns
out to need real `scan`/`while` descent, that is recorded, not assumed.)

## The band — fixed now, before the build and the run

Subset size is 4 (dfx#207, bjx#D416, npy#249, bjx#969); "half" = 2.

| finding | reading |
|---|---|
| **0 mechanized** | **Falsified.** Control flow was not the bottleneck; the re-aim is elsewhere and it is not this |
| **≥ 2 absolute, ≥ half the subset (≥ 2 of 4), ≥ 2 libraries** | Supported. The ≥ 2-libraries clause carries over — load-bearing three times now |
| anything between | Weak. Publish, don't build on it |

## Carried over verbatim from E2a's registrations (not re-derived)

Criterion (i) mechanized (nonvacuity checked); criterion (ii) as a **named
relation** to the registered one-liner (discharges-it / precondition-with-
gap-named); the **relation breakdown in the same sentence as any count**;
`blocked` handling; the denominator's full provenance chain. A count is
"N of 4" with its relation breakdown attached.

## Prior, recorded before the run — the solver may get its first real customer

`cond` with a **straddling predicate** is the canonical search-shaped
UNKNOWN: intervals join branches but cannot say *which* is taken, and
case-splitting is what a solver does. The four properties are full of
accept/reject and finiteness decisions.

> **Prediction: once branch transfers land, accept/reject predicates
> straddle and the UNKNOWNs come back search-shaped, and the solver
> trigger (`design/unknown-triage.md`, search-shaped across ≥ 2) fires on
> real evidence** — the first evidence it has ever had (`design/e2a-run.md`
> reading: the trigger had almost no data). If instead the joins are
> precise enough, the solver stays moot **for real**, on evidence rather
> than on absence.

The solver's trigger is unchanged and it is **not** built in anticipation
— the prior is a prediction, not a licence.

---

# Reading (2026-07-18 — the four run, `corpus/supply/cf_run.py`)

Machinery built and green (100 tests): `max`/`min`/`select_n` domain ops,
a `cond` branch transfer (descend + join), `convert_element_type`
broadened for the bool→int index path. Each property posed as a loop-body
invariant (declare the carry as a box, model one body step, check the box
is preserved).

| hit | outcome | relation |
|---|---|---|
| **dfx#207** | **VERIFIED** | **discharges** the registered property — the `dtmin` clamp (`max`) makes `dt ≥ dt_min` a loop-body invariant; 94% coverage. *Property-incident gap:* the `max_steps` collapse is a step-COUNT bound the property does not reach |
| **npy#249** | **VERIFIED** | **precondition** — `step_size = exp(·) > 0` is a loop-body invariant; the `isfinite` conjunct is float-specific (ℝ-partial) and out of the registered semantics |
| **bjx#969** | UNKNOWN | does not mechanize: the shrink `0.8·s_max < s_max` is **dependency-shaped** (interval loses that `s_new` came from `s_max` — the dfx#632 exhibit's shape) → an **affine-forms** case, not the search-shaped the prior predicted |
| **bjx#D416** | UNKNOWN | does not mechanize: `step_size ≥ ε` is **not preserved by the clamp-less adaptation body I modelled** (a modelling choice, *not* the incident's cause — see the cause reconciliation below); counts 0. The incident's actual cause is RNG key reuse, a killed category out of scope |

## Band: 2 of 4, across 2 libraries → **SUPPORTED**

≥ 2 absolute ✓, ≥ half the subset (2 of 4) ✓, ≥ 2 libraries (diffrax +
numpyro) ✓.

**Relation breakdown, in the same sentence (reporting rule): 2 mechanized
— 1 discharges its registered property (dfx#207, with a property-incident
gap), 1 is a precondition (npy#249, ℝ-partial). 0 bear on an incident.**
The word "Supported" is earned on the band and thin on the breakdown, as
disclosed before the run.

## What the hypothesis actually confirmed — and what it didn't

**Control flow was the posability bottleneck, and removing it worked.** All
four became posable; the census gate was right. The two that mechanized did
so *because* the branch/clamp transfers landed. Critically, **the two that
did not fail for reasons other than control flow**: bjx#969 is
dependency-shaped (a domain-precision limit), and bjx#D416's property is
simply *not true* without a clamp (the incident itself). Neither is "the
control-flow transfers were insufficient." So the hypothesis holds:
control flow makes the subset posable and half of it mechanizes; the
residual wall is elsewhere.

## Triggers — the prior was wrong, and the wrongness is informative

- **Search-shaped UNKNOWNs: 0.** The solver prior **did not fire**, and the
  mechanism it predicted did not appear: the branch predicates that could
  be posed were *definite* (the antecedent is declared), so nothing
  straddled. Case-splitting was not the barrier.
- **Dependency-shaped UNKNOWNs: 1** (bjx#969) — the second sighting of the
  `x·k < x` shape after the dfx#632 exhibit. The affine trigger (≥ 2
  dependency-shaped) still does not fire, but the shape is recurring, and
  **the reachable failures point to affine forms, not a solver.** This
  cuts *for* the registered ordering (affine likelier next) — and affine
  now carries its IEEE-first precondition (`design/unknown-triage.md`, §1
  this pass), because tightening the `0.8·s_max < s_max` bracket is exactly
  where an ℝ-vacuous-adjacent false green could appear.

The solver's registered trigger is unchanged and still has almost no data;
what the run added is evidence about *which* upgrade the corpus wants, and
it is not the solver.

## The affine trigger did NOT fire — the recount, held against a miscount

A later review proposed that the affine trigger *fired* on "two
dependency-shaped UNKNOWNs, both blackjax." **It did not, and the recount
confirms the reading above.** The trigger is *dependency-shaped UNKNOWNs
across ≥ 2 of the corpus cases* (`design/unknown-triage.md`). The count in
counting attempts is **1**:

- **bjx#969 — dependency-shaped.** `0.8·s_max < s_max` is true for every
  `s_max > 0`; the tool returns UNKNOWN only because interval arithmetic
  loses that `s_new` came from `s_max`. Affine fixes exactly this.
- **bjx#D416 — NOT dependency-shaped.** Verified: at the corner
  `log_eps = log ε, h_stat = +5`, `step_size = exp(log ε − 0.5) ≈
  6.07×10⁻⁴ < ε = 10⁻³`. The property is **genuinely false** on part of
  the box (no repeated variable in `log_eps − 0.1·h_stat`) — a straddle
  from real non-invariance. **Affine cannot fix a real violation.**
  Counting it as dependency-shaped is the miscount. (The missing floor
  clamp is *my harness's* modelling choice, not the incident's cause; the
  incident's actual cause is RNG key reuse — see
  `design/property-recheck.md`. The two were conflated in an earlier draft
  of this line.)
- **dfx#632 — out of the denominator.** Dependency-shaped, but the exhibit
  "never counts" by its own label; not a corpus case.

So: **1 dependency-shaped UNKNOWN (bjx#969, blackjax), not 2. The affine
trigger stays unfired, and affine stays out of tree.** The two
dependency-shaped *sightings* overall (bjx#969 + the dfx#632 exhibit) are
even across two libraries (blackjax + diffrax), so the "both blackjax"
concentration does not obtain either. The registered ordering still holds
— affine is the *likeliest* next build when it fires — but it has **not**
fired, and nothing here licenses building it.
