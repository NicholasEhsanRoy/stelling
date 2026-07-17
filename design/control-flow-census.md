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
