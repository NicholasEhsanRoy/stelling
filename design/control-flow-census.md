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

## Gate re-adjudication (2026-07-18): the bjx#D416 row, against quoted code

The #969 correction raised the dispositive question: **D416's row was
also classified against a hand-model** (the invented DA body). If it also
over-classified, control-flow-only is 2 of 6 and the gate fails
retroactively, unlicensing the machinery. Adjudicated against the pinned
source (blackjax `e53f46b`), same standard as #969 — quoted lines, not a
mental model:

- **The real adaptation loop body has no key and is branch-shaped.**
  `window_adaptation.py:234–239`: `update(adaptation_state,
  adaptation_stage, position, acceptance_rate)` — the rng key is consumed
  by the *kernel* step, a separate function in the warmup loop; the
  adaptation update never sees it. `:261–274`: the body is
  `jax.lax.switch(stage, (fast_update, slow_update), …)` then
  `jax.lax.cond(is_middle_window_end, slow_final, identity, …)` — the
  branching the bucket is named for, in the library's own code.
- **No semantic gap of #969's kind.** The step-size slice
  (`dual_averaging.py:117–123` + `window_adaptation.py:278`
  `step_size = jnp.exp(…log_step_size_avg)`) is straight-line arithmetic;
  its inputs are the DA carry (scalars, declarable), `acceptance_rate` (a
  probability — a library-interface bound, quotable), and the stage flags.
  No relational coupling between separately-declared carries is required
  to *state* the invariant. What it needs beyond the branch transfers:
  registry rows (`sqrt`, `pow` — coverage, by census) and per-leaf
  declaration plumbing (tedious, not semantic).
- **Granularity, addressed rather than assumed** — the one place the
  rubric was ambiguous. Judged at the *full warmup body* (kernel
  included), every row in this census, including the two that mechanized,
  would fall out (kernels consume keys; solver states are pytrees) — a
  standard under which the gate was unpassable a priori, so it was never
  the census's standard. The standard the rubric's own registered example
  fixes ("`step_size ∈ [lo,hi]` through an adaptation loop is a
  `while`-body invariant") is the **property-relevant slice**, and the
  quoted code shows that for D416 the slice boundary is the *library's
  own function boundary* (`update` is a distinct function with no key) —
  not an abstraction carved by the analyst. #969 fell under the same
  standard: its *slice itself* needs the relational assume.

**Verdict: the row survives. Control-flow-only stays 3 of 6; the gate
holds — at margin zero, on an adjudicated row, with the granularity
ambiguity now resolved in writing rather than by silence.** The machinery
remains licensed.

**The composition caveat, so the gate is not over-read:** of the three
in-bucket cases, only **dfx#207**'s mechanized harness actually used the
branch machinery; **npy#249**'s used none of it (the loop-body-invariant
*framing* was its unlock — straight-line exp); **D416**'s faithful posing
would use cond/switch plus registry rows plus per-leaf declaration
plumbing, and has not been attempted. "Control flow is the bottleneck"
was true, cleanly, for one case of three.
