# Registration: two readings for the re-rank — committed before reading

**Status:** REGISTRATION, 2026-07-18. Two independent probes, both
read-only, both inputs to re-ranking `design/jax-verification-categories.md`.
Committed before any rule source or thread is read. Neither licenses a
build; neither resurrects a kill.

---

## A. The rule-region candidate (work order §2)

**The candidate — new, with its own registration; the point kill stands.**
`check_grads` catches lying primals *at points* — true, tested, final.
The new candidate is one quantifier up:

> For a custom-derivative rule, is there a **writable, non-circular,
> non-Intentional region property** — `primal_out(x) = f(x)` and/or
> `tangent_out(x,t) = J_f(x)·t` — over a stated region, for any of the 8
> rules in `design/rule-provenance.md`?

**Corpus, fixed:** the 8 rules (7 jvp + 1 vjp), by file:line, from the
provenance artifact. Method: read each rule's source (installed corpus,
jax 0.11 env). No tracker reads planned for this probe; if one becomes
necessary it gets its own registration.

**Per-rule classification, fixed:**

- `primal_form`: verbatim (`rule returns f(x) by calling/duplicating it
  exactly`) / recomputed-differently / other.
- `tangent_form`: true-Jacobian-intended / **Intentional deviation**
  (smoothing or regularizing a useless/undefined true derivative — e.g. a
  step function's) / unclear.
- Intentional detection rule: the deviation is Intentional if the primal
  is non-differentiable or has a useless derivative at the relevant set
  and the rule's purpose is to replace it (docstring/comment or
  construction makes this plain).

**Outcome bands, fixed:**

| finding | reading |
|---|---|
| all 8 rules: primal verbatim AND tangent Intentional-or-trivial | **candidate dies properly** — the frame ate its own line; record as a good outcome |
| ≥1 rule with a non-trivial primal obligation, or a tangent obligation that is *intended* to be the true derivative (writable over a region, possibly minus an excluded set) | **candidate lives** as a registered property with named rules; still licenses only a value-model *input* |

Settled either way: the check_grads-coverage follow-up stays dead for the
better reason — testing cannot reach the region property.

---

## B. The capacity consolidation (work order §3)

**The question.** jax-md's ten `overflow` threads (5.6% of its tracker —
the highest measured concentration in the dataset) are currently filed
across three rows. Do they consolidate into one thing — a **capacity
certificate** delivered as a contract (`requires: density ≤ ρ_max ⟹
ensures: neighbour_count ≤ capacity`) — or not?

**Corpus, fixed:** the ten threads fetched under `jax-md × overflow` in
tracker-probe-2: #101, #126, #141, #161, #165, #191, #192, #255, #377,
#392. Full threads (bodies + comments) will be read. #339 is *excluded*
(correctness near the PBC boundary, a different property, already counted
in probe 1).

**Per-thread buckets, fixed:**

| bucket | meaning |
|---|---|
| **bound-wanted** | the user's problem is not knowing what capacity suffices — a bound over a region would have answered the thread |
| **protocol-burden** | the pain is the check-flag/reallocate/re-run API dance itself — a proof removes no round-trip; stelling doesn't help |
| **detector-defect** | the overflow machinery itself misbehaved — L3's row, already registered |
| **other/dup** | anything else, incl. usage questions orthogonal to capacity |

**Consolidation criterion, fixed:** holds iff ≥4 threads are bound-wanted
across ≥3 distinct authors. Fails toward the protocol counter iff ≥5 are
protocol-burden. Anything else: mixed — the 5.6% is a mix and the
consolidation is recorded as smaller than it looks.

**Also recorded:** the clause-2 instance — a physicist can estimate max
neighbours from density and cutoff in one line; ten threads exist anyway.
Whichever bucket dominates, note *why* the one-line estimate wasn't
enough (PBC interaction, non-uniform density, cell-list mechanics, or the
bound was never the problem).

**Anti-rationalisations:** the consolidation is attractive, which is the
warning — it happens only if the threads support it; counts are incidents,
not customers; a mixed result is reported as mixed.
