# The E2a run — the seven in-semantics survivors

**Status:** reading, 2026-07-18. The run of the E2a corpus after three
subtractions (`design/mwe-census.md`, `design/semantics-classification.md`)
under tightened criteria (i) + (ii). Each count carries its relation
breakdown (`design/e2a-registration.md` reporting rule); every attempt is
mapped to the edge-flux continuous-flow frame and its outcome recorded.

## Denominator provenance (in the same breath as any count)

**13 addressable → 9 reconstructible → 7 in-semantics**, plus hit386 the
positive control that was never a corpus member. Any count below is "N of
7" with this chain attached.

## The seven, mapped to the frame

| hit | bucket | registered property | outcome | why |
|---|---|---|---|---|
| **dfx#417** | ℝ-faithful | `|y_n| ≤ B` over horizon | **VERIFIED — counts 1** | `sigma=0` in the repro → deterministic contracting gradient flow; `[−0.5,1.5]²` edge-flux invariant, 100% coverage, no solver; mutation (y capped at 0.5) does not verify |
| **dfx#207** | ℝ-faithful | accepted `dt ≥ dt_min` | 0 — unposeable | the property is a **PID-controller output**; its accept/reject is a discrete recursion (scan/while), not a flow-box invariant. The chemistry ODE is poseable but has no stated relation to the controller's `dt` floor — cross-layer |
| **bjx#D416** | ℝ-faithful | adapted `step_size ≥ ε` | 0 — unposeable | a **NUTS adaptation-state** property (warmup recursion); the Lotka–Volterra ODE beneath it is poseable but a flow-box has no relation to the sampler's `step_size`. Incident mechanism is RNG key reuse — a killed category, out of scope |
| **npy#1133** | ℝ-faithful | sampler state stays in support | 0 — unposeable | **NUTS state** over a correlated MVN; "support" is a static constraint region, and the dynamics are the sampler, not an ODE flow |
| **npy#249** | ℝ-partial | `isfinite(step_size) ∧ step_size>0` | 0 — unposeable + gap | HMC **adaptation-state**; and `isfinite` is float-specific (ℝ-partial), so even the poseable half would carry the semantics gap |
| **jmd#339** | ℝ-partial | neighbor list ⊇ pairs within cutoff | 0 — unposeable + gap | a **construction postcondition**, not flow inductiveness; and the incident is a PBC float-wrap (ℝ-partial) |
| **bjx#969** | ℝ-partial | non-finite ⇒ `step_size_max` shrinks | 0 — **blocked** | an implication; needs `not`/`or` (not in the census set) → inert `assume` → `blocked (inert assume)`, counts 0 and is not a mechanization failure |

## Count and band

**1 of 7 mechanized.** Band: **Weak (1–3)** — publish the result,
re-aim; not grounds to build further (`design/value-model-v2.md`).

**Relation breakdown (reporting rule, in the same sentence): the one
count is a precondition, not a discharge.** dfx#417's box discharges the
**continuous-flow version** of `|y_n| ≤ B`; the incident is
ReversibleHeun's **discrete instability** at dt0 = 0.1, a discrete-step
behaviour outside the frame — the inherited fidelity demotion, named. So:
*"1 mechanized: 0 discharge the registered property; 1 is a precondition
with a discrete-step gap named."* Never "1 of 7, weak" alone.

## The finding underneath the count

The MVP's continuous-flow edge-flux frame **mechanizes ODE flow-box
invariants** — cleanly, at 100% coverage, no solver, on both the control
(hit386) and dfx#417. But **6 of 7 survivors' registered properties are a
layer away**: they are controller / sampler adaptation state (discrete
recursions the MVP does not model — no `cond`/`scan`/`while`, no
discrete-step) or constructions, not invariants of a flow. This is the
layer probe's lesson landing in the tool: the corpus's "state predicates"
are mostly solver-internal, and reaching them needs machinery the MVP
does not have. The mechanizable-today subclass is **narrow and real**;
the corpus mostly sits outside it.

## Triggers (`design/unknown-triage.md`)

- **Search-shaped UNKNOWNs across the run: 0.** The **solver trigger does
  not fire** (needs ≥ 2) — but "moot on this evidence" **overstates it**:
  6 of 7 were unposeable and never reached an UNKNOWN at all, so the
  trigger's evidence base is essentially one case. The correct statement
  is *the trigger has almost no data, because the failure was upstream of
  the triage*. What survives is the kill: the dominant failure was
  **unposeability**, which no solver, domain, or proof format
  addresses — you cannot state the property in this frame. The Z3-vs-cvc5
  architecture is not refuted; it is untested here, killed upstream by a
  bottleneck that is not decidability.
- **Dependency-shaped UNKNOWNs in a counting attempt: 0.** The **affine
  trigger does not fire** either. (A dependency-shaped UNKNOWN did appear
  in the dfx#632 *exhibit* — `t + dt > t` — but that is out of the
  denominator and never counted.)
- The dominant failure mode was neither: it was **frame-unposeability**
  (the property is not a flow-box invariant), which no solver or domain
  upgrade addresses — it needs control-flow transfers and discrete-step
  modeling, both out of this pass's scope with their own future triggers.

## What would move the count

Recorded, not built: control-flow transfers (`cond`/`scan`/`while`) plus a
discrete-step model would bring the controller/sampler properties into
frame — that is the largest lever, and it is E2b-adjacent-but-distinct
(still check mode; the harness would still state the box). Named here so
the Weak band is read as "narrow frame," not "no value": the frame works
where it applies.
