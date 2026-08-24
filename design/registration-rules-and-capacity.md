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

---

# Reading A (2026-07-18): the eight rules, read

| rule | primal_form | tangent_form | note |
|---|---|---|---|
| equinox `_nextafter.py:25` (×100) | verbatim (returns its own primal) | **Intentional** — tangent ≡ `tx`: the smooth extension of a float staircase is the rule's purpose | itself part of the float-boundary defence machinery (L4): a defence whose AD behaviour is a deliberate lie |
| equinox `_ad.py:825` (`_nondifferentiable`) | verbatim (identity) | **Intentional-guard** — the jvp *raises*: a detective defence implemented as a custom rule | region property meaningless by design |
| equinox `_ad.py:777` (filter_custom_jvp wrapper) | — | — | **wrapper host**: semantics belong to the wrapped rule; the provenance handle groups at the wrapper, so "8 rules" is really 6 concrete + 2 hosts with unresolved wrapped populations (granularity finding) |
| equinox `_ad.py:1001` (filter_custom_vjp wrapper) | — | — | wrapper host, same note |
| lineax `_misc.py:58` (`_asarray`) | verbatim | true-Jacobian, trivially so (tangent = the same cast) | exists as a **workaround for JAX issue #15676** — a fourth rule-purpose: bug patch |
| optimistix `_misc.py:261` (`_asarray`) | verbatim | trivially true | identical twin of the above |
| lineax `_norm.py:70` (`_two_norm`) | verbatim-equivalent (computes the norm) | **true-Jacobian-intended away from `{‖x‖=0} ∪ {inf}`, Intentional zero on that set** ("Get zero gradient, rather than NaN gradient") | **the candidate carrier** |
| lineax `_norm.py:142` (`_zero_grad_at_zero`, max_norm) | verbatim (identity) | true away from 0, Intentional at 0 | second carrier, near-trivial |

**Outcome, against the fixed bands: the candidate lives, narrowly.** Two
rules (`_two_norm`, `_zero_grad_at_zero`) carry a tangent obligation that
is *intended* to equal the true Jacobian over a region minus a stated
excluded set — writable, non-circular, non-Intentional on that region
(`∀ x, ‖x‖ ≥ ε: tangent_out = ⟨x, tx⟩/‖x‖`). The Intentional-set
structure is exactly the guard experiment's bucket discipline, reused.

**The narrowing inside the survival:** all six concrete rules return their
own primal verbatim — **the primal-agreement obligation has zero real
surface in this corpus.** The lying-primal hazard is real as an API
property (probed, twice), and **six library-authored rules are what this
project has looked at**: every one of those authors discharges it by
construction. *Read "unpopulated in the wild" until 2026-08-24 — "the
wild" is a population no instrument here reaches, and
`design/primitive-census.md` states the rule: a low count is a fact about
this corpus and these harnesses.* The Stage-2 flagship must be reshaped accordingly: the
live obligation is tangent-region agreement on two named norm rules, not
primal consistency.

**Also recorded:** the corpus's custom rules serve four purposes — NaN
defence (norms), smoothing (nextafter), differentiation guard
(`_nondifferentiable`), and JAX-bug workaround (`_asarray` ×2). None is a
physics derivative. And `check_grads` run on `nextafter` would *fail its
Intentional tangent correctly* — point-testing cannot even be applied to
the defence layer without the Intentional bucket. The check_grads-coverage
follow-up stays dead for this second better reason.

# Reading B (2026-07-18): the ten capacity threads, read

| thread | bucket | note |
|---|---|---|
| #101 | protocol-burden | vmap × data-dependent allocation; resolution = "use the allocate/update pattern". A region bound *would also* have resolved it — noted, not counted |
| #126 | other (perf/docs) | benchmarking |
| #141 | detector-defect | update misbehaves with *no* overflow; capacity capped at N in the fix — already L3 |
| #161 | protocol-burden | the answer *is* the check-flag dance |
| #165 | other — **silent construction correctness** | wrong `idx` in free space; "you've stumbled upon a bug" |
| #191 | other — **silent construction correctness** | cell-list wrong under fractional coords; fix added a *new guard* (error on non-rectangular boxes) — defence-creation observed live |
| #192 | **bound-wanted** | capacity policy over a batch; the maintainer: "I'm not sure what an optimal policy would be" — the bound does not exist even upstream |
| #255 | other (API/shape) | box_size shape |
| #377 | other (performance, c15) | throughput, not capacity |
| #392 | other | out-of-scope proposal, closed |

**Outcome, against the fixed criterion: consolidation FAILS — mixed.**
1 bound-wanted, 2 protocol-burden, 1 detector-defect, 2 silent
construction-correctness, 4 unrelated. The 5.6% concentration is a mixed
cluster, not one demand; the capacity certificate has exactly one measured
demand instance and is **not** a first customer for contracts. The
attractive synthesis died on the reading, as the registration required it
to be allowed to.

**Where reading B's weight actually lands: L3's completeness sub-form.**
#165 and #191 join #339 and #507 — four wild instances of
detectors/structures silently wrong by construction, across two libraries.

**The clause-2 instance:** the one-line density estimate fails not on
arithmetic but on the quantifier — capacity must hold over a batch or
trajectory *region* (#101, #192), and jax-md's own per-configuration
estimator got even the single-config case wrong once (#141). Locality
fails because the obligation out-scopes the site.
