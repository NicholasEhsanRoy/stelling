# The layer probe — the solver's own step functions, registered before starting

**Status:** REGISTRATION, 2026-07-18. The supply probe discharged **zero of
three incidents**: it proved properties of the user's vector field while
the incidents lived in Newton iterations, controller dynamics, and
adaptation coupling. The hypothesis here: **the incidents do live in step
functions — the solver's, not the user's.** Newton is an iterated map; the
PID controller is an iterated map; the prover targeted the wrong layer.

Why this is not a rescue (checked before accepting): (1) it is a mismatch
between two pre-registered artifacts — the tracker probe registered 17/20
hits as solver infrastructure before any proving; the supply probe then
proved user-physics properties; (2) the supply probe's own outcome table
names these objects ("ranking argument over the PID accept/reject
dynamics"; "obligations over solver-internal iterates"); (3) it can lose,
cheaply, and the loss is the same answer with a different value model:
**0/3-is-real and 0/3-is-targeting are different products.**

## Targets — forced by the named gaps, no selection rule

1. **#368 / Kvaerno5's Newton iteration.** `isfinite(z_k) ⟹
   isfinite(z_{k+1})` over a region — the incident's own layer ("the
   linear solver returned non-finite output").
2. **#386 / the PID accept/reject loop.** `num_accepted_steps ≤ N`, or the
   ranking argument the supply probe named.

Rules as before: **stelling forbidden**; pen and Z3 on hand
transcriptions; four-hour-analogue budget per target; model
simplifications are fidelity demotions, not rounds; a round is any change
to the invariant **or the region**.

## Prior, recorded before running (the proposer's)

> **Newton loses.** `isfinite(z_{k+1})` needs `J` non-singular over the
> region — a determinant condition, which smells nonlinear and could
> easily be 5+ rounds or a technique gap of its own. The PID one: no
> prior.

## Bands — fixed

| finding | reading |
|---|---|
| incident-layer properties proved in 0–2 rounds, box-shaped | the 0/3 was a **targeting error**; induction on the solver's step map reaches the incidents; the model prices a tool aimed at library internals |
| 5+ rounds, nonlinear, or a determinant/ranking condition | **the 0/3 is real**; induction reaches the state-space envelope and not the incidents; the model prices an envelope product or a methodology, or says the target is wrong |
| a technique gap again (machinery no plan contains) | **the Stage-2 plan is aimed at the wrong object** — the most consequential outcome and the most likely to be resisted; reported loudest |

---

## §3 registration — state predicates vs trajectory predicates

numpyro#1360's property ("zero divergent transitions") violated the
property test's own anti-circularity clause — *the thing that went wrong
doesn't go wrong*, wearing a quantifier. The filter lacked an axis:

- **State predicate** — a predicate of a state (or a single transition):
  `isfinite(y)`, `rho > 0`, `step_size ∈ [lo,hi]`, `t_{n+1} > t_n`.
  A candidate for induction.
- **Trajectory/event predicate** — an event over a run: "zero divergent
  transitions", "the event fires", "converges", "completes". **Not an
  invariant of anything**; no strengthening reaches it.

**Do:** classify all 20 hits' properties on this axis. This does **not**
re-run the demand band (the incidents happened, expensively, regardless of
how the property was later worded); it computes the **addressable
subset** — a supply fact.

**Correction for all future registrations, paid for twice:** a one-line
property must be constructive, non-circular, **a state predicate**, and
**checked against the incident it came from**.
