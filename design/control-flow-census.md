# The control-flow census — classify the six unposeable, gate fixed first

**Status:** REGISTRATION, 2026-07-18. Committed before the six unposeable
properties are classified. Free: no new evidence, only the E2a run's own
artifacts. It decides whether the re-aim is **classical control-flow
machinery** (Kani-style: `cond` → join, `scan` → unroll or user
invariant, `while` → bounded unwinding / invariant check) or **the wall**
(a discrete-step model — Newton's Kantorovich ball, NN-Jacobian bounds,
dt-couplings; contracts all the way down).

## Buckets — fixed before the classification

| bucket | meaning |
|---|---|
| **control-flow-only** | the property is an invariant of a **loop body**; posing it needs `cond`/`scan`/`while` transfers and nothing more. *"`step_size ∈ [lo,hi]` through an adaptation loop is a `while`-body invariant, NOT a step-map semantics."* |
| **needs a discrete-step model** | posing it requires the semantics of the discrete recursion itself — the wall |
| **neither** | not a loop invariant at all (e.g. jmd#339's neighbour-list superset — a postcondition of a *construction*) |

Judged on the property's shape from the run's artifacts, per this rubric.

## The gate — fixed now, before the count exists

> **≥ 3 of 6 control-flow-only → control flow is the bottleneck; §4's
> hypothesis is worth registering. < 3 → the wall is the bottleneck,
> control flow is not the re-aim, and §4 does not happen.**

Set before the count, not fitted to it. The six are the E2a run's
unposeable set: dfx#207, bjx#D416, npy#249, npy#1133, jmd#339, bjx#969.

---

# Reading (2026-07-18 — the six classified)

| hit | property | bucket | grounds |
|---|---|---|---|
| **dfx#207** | accepted `dt ≥ dt_min` | **control-flow-only** | a bound on the controller-loop variable `dt` maintained across iterations by the `dtmin` clamp (`max`/`select_n`, censused) — a `while`-body invariant, no error model needed to prove the clamp holds |
| **bjx#D416** | adapted `step_size ≥ ε` | **control-flow-only** | the rubric's own example verbatim: `step_size` bounded through the adaptation loop is a `while`-body invariant. (Incident is RNG key reuse — a killed category — so even posed it discharges nothing; the classification is about posability, and cond/scan/while suffices to state it) |
| **npy#249** | `isfinite(step_size) ∧ step_size>0` through warmup | **control-flow-only** | same shape — a bound maintained through the warmup loop. `> 0` is trivial (`step_size = exp(·)`); the `isfinite` half is a loop-body invariant, ℝ-partial caveat inherited |
| **bjx#969** | non-finite ⇒ `step_size_max` shrinks | **control-flow-only** | the divergence-response is one loop iteration's conditional (`where(non_finite, shrink, keep)`); posing it needs a `cond`/`where` transfer, nothing more — a branch-logic property |
| **npy#1133** | sampler state stays in support | **neither** | numpyro samples in unconstrained space and maps through a constraining bijector; "in support" is a **construction** postcondition of the transform, not a loop-body bound |
| **jmd#339** | neighbour list ⊇ pairs within cutoff | **neither** | a postcondition of the neighbour-list **construction**, not a loop invariant — as the registration names it |

## Count and gate

**control-flow-only: 4** (dfx#207, bjx#D416, npy#249, bjx#969).
**needs a discrete-step model: 0.** **neither: 2** (npy#1133, jmd#339).

**Gate: 4 of 6 ≥ 3 → PASSES.** Control flow is the bottleneck, not the
wall. This is consistent with the corpus: the discrete-step-model cases
this project *did* meet (dfx#368 Newton, dfx#386 ranking, npy#1360
supermartingale — the supply/layer probes' walls) are the **trajectory**
hits, none of which are in the reconstructible in-semantics seven. Among
the six unposeable here, **none** hit the wall; four are loop-body
invariants and two are constructions. §4's hypothesis is worth
registering.

Recorded honestly, so the gate is not read as more than it is: two of the
four control-flow-only properties (bjx#D416, npy#249) are bounds that are
**not true invariants without a clamp** — which is *why* the incidents
happened (no floor). They are *posable* with control-flow transfers (the
gate's question), but whether they *mechanize* (VERIFY) is §4's question,
and the honest prior is that clamp-less bounds will not.

## Correction (2026-07-18, from the fidelity census)

**bjx#969 is not control-flow-only.** Its row above judged the response as
`where(non_finite, shrink, keep)` — a claim about a mental model, not the
code. The real code (`mclmc_adaptation.py:438–442`, quoted in
`design/harness-fidelity.md`) sets the cap from a *different variable*
(`step_size_max ← step_size × 0.8`), so the faithful obligation is
**relational** and needs the coupling `step_size ≤ step_size_max` — an
`assume`, inert — on top of the branch transfer. The "nothing more" clause
fails. **Control-flow-only: 4 → 3 of 6. The gate passes at exactly 3 of
6** — the outcome is unchanged and the margin is now zero, said plainly.
The census's own rubric was applied to an unverified model of the code —
the same defect the fidelity census exists to catch, one document
upstream.
