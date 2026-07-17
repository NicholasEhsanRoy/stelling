# The attribution probe — where did the effort go, registered before reading

**Status:** REGISTRATION, 2026-07-18. A new variable on the same 20
objects — the buckets do not move, the demand band does not move (same
grounds as the state/trajectory axis). Committed before any thread is
re-read.

## The derivable half (no probe needed, recorded here)

The tracker probe's registered terms were detection words — `nan`, `inf`,
`diverge`, `unstable`, `blow up`. You cannot search for `nan` and find a
failure nobody noticed. **The corpus is post-detection by construction:
detection cost ≈ 0 across all 20 hits, by selection.** This goes into the
tracker artifact as a scope note. What does *not* follow is that the
remaining cost is attribution rather than fix — that is the hypothesis
below.

## Definitions — fixed

| phase | definition |
|---|---|
| **Detection** | report → first notice. ≈0 by selection; not measured |
| **Attribution** | first report → the first comment that **correctly names the mechanism**, verified against what the fix actually did |
| **Fix** | that comment → the closing commit/PR |

Operationalization: the issue body counts as comment 0 — if the body
already names the mechanism, in-thread attribution is 0 and the case is
recorded as **named-in-body** (attribution paid pre-filing, off-tracker;
explicit narrowing statements in the body are its cost evidence).

Effort proxies (calendar time is not effort):

- **comment count per phase** — someone wrote each one;
- **wrong hypotheses aired** — mechanisms proposed and rejected;
- **recorded misdiagnosis** — a wrong answer **adopted** (acted on or
  accepted, not merely aired) before the right one. Binary; the sharpest
  signal. Archetype: the CFL case (float32 believed before CFL found).

## The duration-confound correction — recorded regardless of outcome

GitHub timestamps measure calendar, not effort: a 258-day thread can hold
four hours of work. Therefore the tracker artifact's "487d"/"258d" cost
signals are **calendar proxies**, and the demand band's cost evidence is
**weaker than reported**. This correction goes into the tracker artifact
whichever way this probe lands — it cuts against a band the proposer
likes, which is why it goes in.

## Bands — fixed

| finding | reading |
|---|---|
| **attribution dominates** — median attribution-comments/total > 0.7, or ≥8 of 20 with a recorded misdiagnosis | the cost is attribution; the product is diagnostic; every value hypothesis so far aimed at detection while the data was about diagnosis |
| **fix dominates** | **the reframe dies** — people knew what was wrong and it was still hard to fix; faster attribution saves nothing |
| **mixed** | the distribution, not the mean — one product per mode |

## Prior and bias, recorded (the proposer's, after four dead reframes)

> **Attribution dominates.** I want it to win; the definitions are tight
> for that reason. If the misdiagnosis count is 1–2 of 20, the reframe is
> dead and is reported as dead in the same sentence as the number.

---

# Reading (2026-07-18 — all 20 threads, full comments)

## Per-hit decomposition

| hit | in-thread attribution / total comments | misdiagnosis adopted? | notes |
|---|---|---|---|
| bjx#D416 | 2/5 | **YES** | filed as an adaptation failure; the user found it mid-thread: **PRNG key reused twice** — a killed-category (D8) hazard presenting as a sampler failure |
| bjx#969 | 0/1 — **named-in-body** | no | mechanism fully diagnosed pre-filing |
| bjx#973 | 0/1 — **named-in-body** | no | design question, mechanism in body |
| jmd#339 | 0/2 — **named-in-body** | **YES** | the body *contains* the misdiagnosis trail: thermostat/physics blamed ("flying ice cube"), edited later to the neighbor list — attribution paid by the user, pre-filing |
| jmd#92 | 9/9 | no | entire thread is diagnosis (XLA reassociation + chaotic amplification); no fix ever |
| dfx#194 | 3/3 | no | unresolved; all effort was attribution |
| dfx#207 | 4/5 | no | `error_if` debugging until the user found their own bug; the max_steps error was the detector, attribution was the thread |
| dfx#223 | 6/6 | no | user wrote their own RK23 to reproduce, then found it: **NaN in the error estimate propagating through `jnp.min`/`max` in the controller factor** (nanmin fixes) — the P5/L3 family, one line |
| dfx#368 | 11/17 | **YES** | the optimise-then-discretise/backward-solve mechanism was held and argued before correction; final mechanism confirmed at comment ~11 of 17 |
| dfx#386 | 7/12 | **YES** | "does not reproduce under 0.5.0" adopted then retracted; heisenbug arc (debug.print changes outcome); localized to the nonlinear solver at ~c7 |
| dfx#417 | 1/1 | no | mechanism named in one comment (solver known-unstable) |
| dfx#507 | 1/3 | no | "classical problem" named immediately; unfixed by design |
| dfx#632 | **1/10 — fix-dominant** | no | mechanism named in ONE comment; the other nine comments and the 258 days were **fix design** (threshold vs rtol vs ULP) |
| dfx#657 | 1/4 | no | named immediately as #632's dup |
| dfx#752 | 2/4 | no | mechanism pinned by comment 2; PR by comment 4 |
| dfx#756 | 0/5 — **named-in-body** | no | the dynamiqs stability investigation was the attribution, paid off-tracker; the thread is fix design |
| npy#249 | 0/1 — named-in-body | no | core-dev repro |
| npy#552 | 3/9 | no | mechanism (real-support sigma → NaN) named at c3 |
| npy#1133 | 1/1 | no | named in one comment |
| npy#1360 | 18/20 | **YES** | twenty comments of diagnosis; the sampler was the suspect throughout; the final answers were model spec **and the user's own Gamma-for-InverseGamma swap**, self-found at c18 |

## Band landing: **MIXED — the distribution, not the mean**

Median in-thread attribution ratio: **0.45** (< 0.7). Recorded
misdiagnoses: **5 of 20** (< 8, and > the 1–2 that would have killed the
reframe outright). Neither dominance band fires; per the registration this
is reported as modes, and the distribution is **bimodal in an informative
way**:

- **Attribution-dominant mode (~10 hits)** — ratios ≥ 0.58 plus the
  named-in-body cases whose narrowing was paid off-tracker (jmd#339's
  self-corrected trail; dfx#756's cross-library investigation): threads
  that are entirely diagnosis, some with no fix at all.
- **Fix-dominant mode (~3 hits: dfx#632, #657, #752)** — the maintainer
  named the mechanism in *one comment*; the calendar was fix-design
  latency. **Expert attribution is cheap for the expert on his own layer**;
  the 258 days were not spent attributing.
- **Named-in-body (5 hits)** — in-thread ratio 0 is an artifact of *where*
  attribution happened, not *whether*: the cost is real, off-tracker, and
  visible only in body narratives ("I found that it is the neighbor
  list"; "I was investigating the stability").

## The observation the bands didn't ask for

**All five misdiagnoses are layer errors.** D416: sampler blamed, key
reuse actual. jmd#339: physics blamed, neighbor list actual. dfx#368:
adjoint semantics blamed, step-rejection backprop actual. dfx#386: version
blamed, nonlinear-solver state actual. npy#1360: sampler blamed, model
spec and a distribution swap actual. Where diagnosis went *wrong*, it went
wrong **across a layer boundary** — recorded as data adjacent to the
elimination-mode hypothesis, which remains unadopted; this is 5 points of
the same shape, not a test of it.

Per the prior: the reframe is neither confirmed nor dead — attribution is
the dominant labour in roughly half the corpus, fix-design in a sixth, and
the sharpest signal sits between the kill line and the win line. The
diagnostic-product claim survives only in the mode-specific form and
carries the named-in-body measurement gap.
