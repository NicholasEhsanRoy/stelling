# The corpus-expansion registration — written before any search runs

**Status:** REGISTRATION, 2026-07-18. **Registered, not run.** This
document is the unpaid prerequisite of fork (A) — expand the corpus —
written so the (A)-vs-(B) choice can be made concretely. Nothing below
executes in the pass that writes this.

## §0 — The inherited cases, re-scored under this registration's own criterion

An expansion inherits the 7 existing in-semantics cases, scored by a test
this registration supersedes. Stated as arithmetic, up front, so the run
cannot re-derive a disclosed weakness as a discovery:

| inherited case | under the registered (extended-ℝ) test | under this registration's finite-⊤ criterion |
|---|---|---|
| dfx#417 | mechanized (sound anchor, bounds-load-bearing) | **mechanized — stands** |
| npy#249 | mechanized (bounds-load-bearing via the −∞ endpoint) | **voided — tautological over finite ℝ**: `exp`'s range over finite inputs is `(0, ∞)`; the obligation `step_size > 0` is its range theorem |
| dfx#207 | voided (tautological, `max`'s definition) | voided |
| bjx#969, bjx#D416, npy#1133, jmd#339 | unposeable / blocked | unposeable / blocked, unchanged |

> **Inherited mechanized count under this registration's criterion: 1**
> (dfx#417). The expansion starts from 1, not 2. The dual number is
> recorded at `design/obligation-vacuity.md`; both travel together.

## §1 — The vacuity criterion this registration carries (the fifth property correction)

Two parts, both required:

1. **The mechanical ⊤-widening test** as registered
   (`design/obligation-vacuity.md`) — necessary, run on every counted
   case.
2. **The named-range-theorem clause** — the finite-ℝ refinement the
   extended-real test cannot see (and, verified, no test built on this
   domain can: the domain's `exp` underflows to a 0 endpoint even at
   finite bounds, so a maxfloat-widening run still credits `exp > 0`).
   An obligation whose discharge follows from a primitive's **range over
   finite inputs** (`exp(x) > 0`, `max(x, c) ≥ c`, `|x| ≥ 0`, `x² ≥ 0`,
   …) is **tautological regardless of the widening test's verdict**. The
   claim must be justified by **naming the range theorem** — pointable-at,
   not asserted, the face-expression discipline.

## §2 — Search: terms, trackers, and what gets recorded

- **Terms, declared now:** the banked `explosion`, `wrong results`,
  `stuck`, `hang`, `biased` — plus the original five (`nan`, `inf`,
  `diverge`, `unstable`, `blow up`) for cross-tracker comparability.
- **Trackers:** the three censused-never-probed libraries —
  **optimistix, lineax, equinox** — under the full term grid; and the
  **new terms only** on the original five trackers (diffrax, numpyro,
  blackjax, jax-md, jax-cfd), which the original terms already covered.
- **Recorded at read time:** search date, per-term-per-tracker raw
  pre-filter counts, repo refs pinned (SHAs). Tracker contents are
  time-varying; the searches are dated facts.
- **The scope note inherits:** these are still detection words (plus
  outcome words). **The corpus remains post-detection by construction**
  — it can never speak to detection value — and calendar durations remain
  proxies, not effort.

## §3 — Classification: everything inherited, nothing re-derived

Applied in order, each before the next phase reads anything:

1. **Taxonomy** (registered bands per tracker): long-horizon /
   point-detectable / user-error / performance / feature-request /
   unclear; closed-as-not-a-bug still counts toward the target population
   where the mechanism was real. Low-issue-count trackers get the
   base-rate check (the jax-cfd lesson): P(0 hits) computed before
   silence is read as absence.
2. **One-line property per hit**, under the **five** corrections:
   constructive; non-circular; **a state predicate**; **checked against
   the incident it came from** (reaches the mechanism); **not
   tautological** per §1. Properties failing any correction are recorded
   with the failure named — the anchor-quality annotation rides with any
   later count.
3. **Semantics classification** (ℝ-faithful / ℝ-vacuous / ℝ-partial)
   before any harness — judged on the incident's mechanism, **plus the
   third-bite flag**: an ℝ-faithful property on an **overflow-reaching
   program** is ℝ-partial in effect (`design/soundness-audit.md`, second
   pass, finding 4-A).
4. **MWE census** (full-MWE / partial / none; reconstructible = full
   only; linked-external does not count; exclusions published with
   reasons).
5. **Fidelity buckets** for any harness written later: imported /
   user-MWE / hand-transcribed (disclosed, pointered) / hand-modelled
   (**counts 0**, exhibits only) — judged against source at pinned refs,
   lines quoted.
6. **Criteria (i) and (ii)** as mechanized/anchored: nonvacuity checked
   in-trace; the relation to the registered property named
   (discharges-it / precondition-with-gap-named).
7. **Every count carries its three breakdowns in the same sentence**:
   relation, fidelity, and the full denominator provenance chain
   (searched → long-horizon → property-sound → reconstructible →
   in-semantics → non-tautological), plus blocked cases named.

## §4 — The input-declaration trigger, registered here as promised

Per `design/corpus-limits.md`'s decision (no fitted trigger, no
freeze-by-silence — the trigger registers with this document, pre-data on
fresh cases):

> **`any_pytree` (input declaration: pytree state, and key handling if
> ever separable) is built iff ≥ 2 of this expansion's new cases, across
> ≥ 2 distinct libraries, are unposeable solely for input-declaration
> reasons — with the blocking structure quoted per case** (the pytree
> carry / key parameter named from the library's own signature, not
> asserted). The ≥ 2-sources clause is the class default, applied.

If it fires, **the build passes its registered audit gate before any case
it newly enables counts** (`design/corpus-limits.md`: a fresh-context
soundness audit, because `any_pytree` will be built under the strongest
build-time motivation in the project).

## §5 — Bands, fixed now, against the inherited-1 baseline

Let **E** = new cases that survive every §3 gate (the expansion's
eligible set). All fractions are of E; the inherited 1 never counts
toward any "new" number.

- **Demand band:** ≥ 5 new long-horizon hits across ≥ 2 of the three new
  libraries → *the demand corpus broadens* (a standalone, publishable
  finding). Below that: the base-rate check decides between "silence" and
  "absence," per tracker.
- **Supply band (mechanization of new cases):** **0 new mechanized** → no
  supply movement — reported with blocked/input-declaration cases named,
  per §6's prediction, never as bare failure. **1** → an anecdote, below
  any band (the registered consequence). **≥ 2 new, across ≥ 2
  libraries, and ≥ ⅓ of E** → supported-new. Between → weak-new.
- **Arithmetic contingency (inherited from the MWE census):** any band
  found unreachable by denominator arithmetic (E too small) suspends and
  re-registers openly; falsification-by-arithmetic is not a permitted
  reading in either direction.

## §6 — What this registration says about itself, up front

- **`any_pytree` is load-bearing and unbuilt.** The 17-of-20 finding
  predicts the new hits are solver-infrastructure; their faithful
  harnesses need pytree state. **The expansion may establish that the
  demand corpus is real and broader, and still mechanize nothing — because
  the blocker is input declaration, not corpus size.** If that happens it
  is the predicted outcome, not a surprise, and it reads as: demand
  finding real, supply gated on §4's trigger-and-gate.
- **Which fork this resolves.** The expansion's phases split cleanly:
  §§2–3 (search, taxonomy, properties, semantics, MWE census) need no
  tool and **are themselves (B)-artifact work** — they extend the very
  census/taxonomy/demand artifacts fork (B) would publish. The supply
  phases (§3.5–7 harnessing) are gated behind §4. So **(A)'s demand half
  and (B) are the same work**; (A)'s supply half is conditional. If the
  supply ceiling without `any_pytree` is Weak — and the slice-posable
  subset of solver-infrastructure hits is predicted small — then (A)
  collapses into (B)-plus-a-broader-demand-artifact unless `any_pytree`
  is built first, through its gate. Written here so the run cannot
  discover it.
