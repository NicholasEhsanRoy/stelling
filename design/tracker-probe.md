# The tracker probe — long-horizon failures, registered before reading

**Status:** REGISTRATION, 2026-07-17. This section is committed **before a
single issue is opened**; the reading lands in a separate, later commit.
Nothing below may be adjusted after reading begins. Generated findings are
appended under "Reading" after registration is committed.

**The hypothesis being tested** (fourth candidate; the prior is 0-for-3):
classes with cheap *preventive* defences come back clean — confirmed twice
(index clamps, `safe_mask`; `check_grads` for custom-derivative rules).
Classes with only *detective* defences (a NaN check tells you the run died
at step 800k, not that it won't, nor for which inputs) should have a
target. Zero confirmations of that half so far. Long-horizon failures are
**loud** — they crash, get argued about, get filed — so the tracker is a
valid instrument here in a way it could not be for the silent wedge class,
and a null is readable rather than confounded.

## Corpus — fixed

| tracker | slug | discussions |
|---|---|---|
| diffrax | `patrick-kidger/diffrax` | not enabled |
| jax-md | `jax-md/jax-md` (canonical; `google/jax-md` redirects) | not enabled |
| jax-cfd | `google/jax-cfd` | not enabled |
| blackjax | `blackjax-devs/blackjax` | **enabled — searched** |
| numpyro | `pyro-ppl/numpyro` | not enabled (its forum is off-GitHub and **out of corpus**) |

Issues open **and** closed, `type:issue` (PRs excluded). Discussions where
the platform has them (blackjax only). Same libraries as the census, so
the artifacts are comparable.

## Search terms — fixed, with operationalization

| registered term | searched as |
|---|---|
| `nan` | `nan` |
| `inf` | `inf` |
| `diverge` | `diverge` |
| `unstable` | `unstable` |
| `instability` | `instability` |
| `blow up` | `"blow up"` (quoted phrase) |
| `blows up` | `"blows up"` (quoted phrase) |
| `drift` | `drift` |
| `long run` | `"long run"` (quoted phrase) |
| `after N steps` | `after steps` (word co-occurrence — N is not literal) |
| `works for small` | `"works for small"` (quoted phrase) |
| `intermittent` | `intermittent` |

Per-(repo, term) total hit counts are recorded **before any filtering**.
Retrieval: up to 50 results per (repo, term), GitHub best-match order;
the candidate set is the deduplicated union. Classification uses title,
body, labels, state, and thread metadata, with targeted comment reads
where needed. **If a term worth adding occurs mid-read, it is recorded
here and not used — it belongs to the next registration:**
`stuck`, `hang`, `explosion`/`exploded`, `wrong results` (noticed while
reading diffrax threads; not searched).

## Taxonomy — every candidate lands in exactly one bucket

| bucket | meaning |
|---|---|
| **Long-horizon** | Fails only after many steps, or at large N, or intermittently, or seed-dependently. The target class |
| **Point-detectable** | Any single well-chosen test catches it; wrong at step 1. Not the class |
| **User error** | Genuine misuse: wrong argument, wrong config, wrong units. **Reserved for actual misuse** — *not* for "the maintainer said it wasn't a library bug" |
| **Performance** | Not the class |
| **Feature request** | Not the class |
| **Unclear** | Cannot tell from the thread. Reported as coverage, claimed as nothing |

**The counter-intuitive rule, fixed now:** issues closed as *not-a-bug*,
*your model is stiff*, *that's expected* are **the target population, not
noise**. A user hit a real long-horizon failure in their own code, loud
enough to file against a library, and the maintainer correctly said it
wasn't theirs — nobody's tool caught it. That is the class, seen through a
keyhole into the research-code arm the census could not reach. The default
instinct is to discard these; the registration forbids it.

## The property test — applied before the count is looked at

A Long-horizon hit counts **only** with a one-line constructive property:
*what property, over what region, would have turned this red before the
user hit it?* Circularity disqualifies ("the thing that went wrong doesn't
go wrong" is not a property; `state.rho > 0`, `isfinite(u)`,
`energy(s) < E_MAX` are).

## Cost signals — recorded separately, never folded into the count

Per hit: thread length, participant count, time-to-close, and explicit
cost statements ("took three days to narrow down", "lost a week of
compute"). **Pre-fixed:** a class with many instances and trivial cost is
not a target; a count with no cost signal is not a finding.

## The bands — fixed

| Long-horizon hits with a writable property | reading |
|---|---|
| **0–2** | **Falsified.** The class does not reach real code at a rate that matters, or is handled. Stop; the concept is sound and has no target; close the file. A zero is a stop, not a re-aim |
| **3–9, or ≥3 with no cost signal** | **Weak.** Real, rare or cheap. Publishable observation, not a sequencing argument |
| **≥10, in ≥3 trackers, with cost signal on ≥3** | **Supported.** Real, distributed, expensive, unreached by point methods. Licenses **only** the writing of a value model with its own corpus, experiment, and falsifier — not a build |

## The product split — required report

Every Long-horizon hit is also classified by shape: **safety-shaped**
(NaN/inf/crash/negative density/out-of-domain — universal properties, the
user supplies only a region; a tool) vs **accuracy-shaped** (drift,
conservation violation, "looked wrong after a while" — bespoke invariants
the user must write; a methodology). The ratio is reported alongside the
count and matters as much.

## Anti-rationalisations — pre-registered

- *"Closed as won't-fix, so it doesn't count."* It counts. It happened.
- *"It's really user error."* Decided by the taxonomy, not by whether the
  count is liked. A reclassification argument that first occurs after the
  count is visible is the forbidden move.
- *"The verifier probably wouldn't have caught this one either."* That is
  the property test's job, applied before the count is looked at.
- **A zero is a stop, not a re-aim.** If the loud class is absent from the
  loud instrument, that is the answer, and no fifth hypothesis gets
  written in the same breath as the fourth's obituary.

---

# Reading (2026-07-17 — separate commit, after the registration)

Retrieval per registration: 60 issue searches + 12 discussion searches,
per-(repo, term) totals recorded pre-filter (largest: diffrax
`after steps` 53, numpyro `nan` 42, diffrax `drift` 39; jax-cfd is nearly
silent — 1 hit total). **299 unique candidates**: diffrax 129, numpyro
109, blackjax 40 (incl. discussions), jax-md 20, jax-cfd 1. Every
candidate classified — full per-candidate appendix in
`design/tracker-probe-classification.md`:

| bucket | count |
|---|---|
| Long-horizon, property writable (**counted**) | **20** |
| Long-horizon, property test failed (not counted) | 2 |
| Point-detectable | 89 |
| Performance | 33 |
| User error | 5 |
| Feature request / question / docs / infra | 125 |
| Unclear (coverage, claimed as nothing) | 25 |

Authorship was checked on the two blackjax MCLMC hits because their style
raised the possibility of self-filed issues (the instrument must not count
its maker): both are third-party (`ssage0520`, `junpenglao`). Neither
excluded.

## The hits — 20, each with its constructive property

| site | failure | property (one line) | shape | cost signals |
|---|---|---|---|---|
| diffrax#194 | solver stuck after training epochs | ∀ params on training path: backward-solve state finite, steps bounded | safety | open, c3 |
| diffrax#207 | step collapse on stiff chemistry, `max_steps` irrelevant | ∀ y0, params in region: accepted dt ≥ dt_min | safety | 46d, c5, Fortran comparison done by user |
| diffrax#223 | states → inf mid-solve (port of working code) | isfinite(y(t)) ∀ t, ∀ params in region | safety | open, c6 |
| diffrax#368 | integration failures inside NN-ODE training | ∀ (x,p,k) in training region: solve completes finite | safety | **c16, 23d, training pipeline blocked** |
| diffrax#386 | step explosion at specific parameter values | ∀ (a,b) in region: num_steps ≤ N | safety | **c12, "large sets of solves", user narrowed it themselves** |
| diffrax#417 | ReversibleHeun instability | ∀ dt in region: \|y_n\| ≤ B over horizon | safety | open, c1 |
| diffrax#507 | event intermittently missed under PID | ∀ trajectories in region: cond sign-change ⇒ event fires | safety | open, c3 |
| diffrax#632 | error creep at small dt | ∀ dt in region: t_{n+1} > t_n (time strictly progresses) | accuracy/float | **258d, c10** |
| diffrax#657 | StepTo at small dt: silently wrong result in f32 | realized step times = requested grid, ∀ grids in region | accuracy/float | 2d, c4, silent wrongness |
| diffrax#752 | nonlinear max-steps error instead of step rejection | nonlinear failure ⇒ step rejected (never an error), ∀ states | safety | 6d, c4 |
| diffrax#756 | infinite final-step rejection loop (clipping × rejection) | controller progress: clipped endpoint step never re-rejected indefinitely | safety | 14d, c5, **found via downstream dynamiqs investigation** |
| jax-md#92 | jit vs no-jit trajectories diverge after ~1000 steps (similar at 1–10) | \|stat_jit(t) − stat_nojit(t)\| ≤ tol over init region, N=1000 | accuracy | open, c9 |
| jax-md#339 | neighbor list silently wrong near PBC boundary → occasional explosions | ∀ positions in box: neighbor list ⊇ all pairs within cutoff | safety | **487d to close; user did the narrowing** |
| blackjax#D416 | some seeds: adaptation emits near-zero step_size kernels | ∀ seeds/inits in region: adapted step_size ≥ ε | safety | open, c6 |
| blackjax#969 | NaN detector in MCLMC tuning provably never fires | ∀ tuning states: non-finite proposal ⇒ step_size_max shrinks | safety | fixed same day; **a dead detective defence** |
| blackjax#973 | MCLMC divergence response has no equilibrium in NaN regime | adaptation map keeps step_size in [lo, hi] ∀ states in region | safety | open, filed by core maintainer |
| numpyro#249 | warmup stuck, step_size → nan | isfinite(step_size) ∧ step_size > 0 throughout warmup, ∀ inits | safety | fixed same day |
| numpyro#552 | init fails once data size exceeds threshold | isfinite(potential(init)) ∀ init region × N | safety | 2d, c9 |
| numpyro#1133 | `init_to_uniform` sometimes yields mangled chains | ∀ inits in U-region: sampler state stays in support | safety | 30d, c1 |
| numpyro#1360 | seed-dependent non-convergence (2–3 in 10 seeds) | ∀ seeds/inits in region: zero divergent transitions | safety | 1d, **c20** |

Excluded by the property test (long-horizon, no non-circular constructive
property): numpyro#154 (wrong posterior vs Stan — the only property is
statistical equality with a reference), numpyro#1786 (long-run acceptance
statistic drifts from target — an expectation property, supermartingale
territory).

## Landing (against the pre-fixed bands)

**20 hits, in 4 of 5 trackers, with strong cost signals on well over 3
(jax-md#339: 487 days; diffrax#632: 258 days; diffrax#368: 16 comments in a
blocked training pipeline; diffrax#386: explicit narrowing labor;
diffrax#756: a downstream library's stability investigation) → the
Supported band.** Robustness: dropping the three weakest hits
(diffrax#417, diffrax#194, blackjax#D416) leaves 17 in 4 trackers — the
band does not move. jax-cfd contributed zero candidates of any kind; the
distribution requirement is carried by the other four.

Per the registration: this **licenses writing a value model** with its own
corpus, experiment, and pre-registered falsifier — nothing more. It does
not validate the inductive-invariant approach; whether an invariant
catches these specific failures is a separate experiment.

## Scope note and cost-signal correction (added 2026-07-18, derived from this registration)

**This corpus is post-detection by construction.** The registered search
terms are detection words — `nan`, `diverge`, `blow up` — so no hit can be
a failure nobody noticed: detection cost ≈ 0 across all 20, *by
selection*. This corpus can never speak to detection value; whatever the
487 days and 258 days were spent on, it was not noticing.

**And the cost signals above are calendar proxies, not effort.** A thread
open 258 days may hold four hours of work across eight months of neglect.
The demand band's cost evidence is therefore **weaker than reported** —
comment counts and explicit cost statements ("took me three days to narrow
this down") are the effort-grade signals, and their *wording* (narrowing
vs implementing) is data. The phase decomposition lives in
`design/attribution-probe.md`.

## The product split

**Safety-shaped 17 : accuracy-shaped 3** (jax-md#92 trajectory
divergence; diffrax#632/#657 float creep). The trackers are full of
*NaN/inf/stuck/step-explosion/missed-event over a region*, not "energy
drifted" — the product this points at is push-button: finite-state,
bounded-steps, progress, and guard-liveness properties over a
user-supplied region, no bespoke physics spec required.

Two observations recorded without a reading:

- **Half the hits are properties of solver infrastructure, not user
  physics** — step controllers, adaptation loops, event detection,
  endpoint clipping (diffrax#507/#632/#657/#752/#756, blackjax#D416/#969/#973,
  numpyro#249/#1838-adjacent). The region method's natural first customer
  may be the library's own control loops: universal, library-owned
  invariants, the same audience the guard experiment names.
- **Detective defences themselves appear as defects and as user labor**:
  blackjax#969 is a provably-dead NaN detector (the guard experiment's
  Dead bucket, in the wild, in a tracker); diffrax#290 and numpyro#956
  are users hand-building NaN-termination machinery. The class defends
  itself badly, which is consistent with the preventive/detective split
  this probe was designed to test.
