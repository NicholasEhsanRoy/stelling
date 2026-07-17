# The supply probe — can the invariants be proved, registered before reading source

**Status:** REGISTRATION, 2026-07-18. Every supported band so far measured
**demand**; nothing has measured whether an invariant **catches** anything.
This probe takes three tracker hits, writes their properties formally, and
attempts the inductive proof **by hand** — library source, pen and paper,
and Z3 driven directly on a hand-transcribed model of the step function.
**stelling is forbidden**: it has no transfer functions past tiers 0–1,
and using it would confound problem difficulty with tool maturity. The
problem is what decides the product.

## Selection — mechanical, computed here, before any source is read

Strata per the verified technique split (categories artifact): invariant
(9 hits), termination (5 hits). Rule, fixed: **rank by recorded comment
count within stratum (probe-1 hit table), tiebreak by days-to-close
(longer = costlier); take the top 2 from the invariant stratum and the
top 1 from termination.** Cost-ranked selection deliberately tests the
expensive end of demand: a tool verdict from this sample is strong; a
methodology verdict is a statement about the priced end, and is reported
as such.

**Computed selection (before proving):**

| hit | stratum | cost signal |
|---|---|---|
| **numpyro#1360** — NUTS seed-dependent divergences on a regression model | invariant | c20 |
| **diffrax#368** — integration failures inside NN-ODE training | invariant | c16 |
| **diffrax#386** — step explosion at specific parameter values | termination | c12 |

If a selected hit proves unreadable, the next by the same rule is taken
and the substitution is reported. The selection is not re-run.

## What a round is — fixed

> **A round = any change to the invariant *or the region* required to make
> the inductiveness check pass.** Narrowing the region to rescue an
> invariant is the user writing more spec, exactly as adding a conjunct
> is.

Not rounds: transcription typos, re-reading code, correcting my own
algebra. **Model simplifications are not rounds either — they are
fidelity losses, recorded per hit, and they demote the outcome to
"proved for the simplified model."**

## Budget and outcomes — fixed

Budget: the agent analog of four hours per hit — a bounded work budget
(capped strengthening rounds and bounded solver attempts); where it stops,
the stop and the remaining obstacle are reported.

| outcome | meaning |
|---|---|
| proved in N rounds | report N and the final invariant |
| not provable by induction on the step function | a **technique gap** — ranking function, supermartingale, contract on a learned component: named explicitly |
| gave up at budget | where, and what the obstacle was |
| property was wrong | the tracker probe's one-line property did not survive contact — recorded loudly as a finding about the tracker probe |

Per proved hit, the **inferability classification**: box/interval
constraints on state components (forward propagation with widening
plausibly finds it — push-button live) vs nonlinear relations between
components (inference will not find it — the user writes it;
methodology).

## Bands — fixed

| finding across the three | reading |
|---|---|
| 0–1 rounds typical, box-shaped invariants | **Tool.** Push-button survives; the value model aims at L1+L2 as ranked |
| 2–4 rounds | **Borderline.** The Dafny loop; the model prices the strengthening honestly and stops saying push-button |
| 5+ rounds, a page of algebra, or nonlinear invariants | **Methodology for experts.** Different product, different market |
| ≥1 technique gap | reported separately and prominently; bears on the ranking regardless of round counts |

**The distribution matters more than the mean** — one 0 and two 6s is two
products, not an average of 4, and will be reported as two products.

## The limit — stated now

**The prover is not a user.** An LLM with unbounded patience is probably
better at the algebra and worse at knowing which conjunct is physically
natural. The round count measures **the problem, not the user** — the
right thing for deciding the product, and the artifact must not read as a
measurement of user effort.
