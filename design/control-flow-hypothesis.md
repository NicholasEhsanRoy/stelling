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
