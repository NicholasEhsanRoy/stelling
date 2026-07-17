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

---

# Reading (2026-07-18 — proofs in `corpus/supply/layer_*.py`, Z3-checked)

## Target 1: Kvaerno5's Newton iteration (#368's layer)

**The prior confirmed, precisely.** The determinant condition is real and
load-bearing: Z3 proves `J = I − s·A` invertible for interval-bounded `A`
under the dt-condition `s·a ≤ 0.3`, and produces a counterexample the
moment the condition is dropped. The invariant that works is not a box:
round 1 (`z` in a box) **fails** — a Newton step from inside any fixed box
can exit it — and round 2 lands on the **Newton–Kantorovich ball** with a
nonlinear *product* side condition (`β·η·ω ≤ ½`, Z3-checked monotone in
dt). Discharging it requires: **a second learned-component contract**
(`|∂f/∂x| ≤ L` — the NN *Jacobian*, one tier deeper than the supply
probe's `|f| ≤ M`), and **a dt-condition that couples the Newton layer to
the controller layer** — the two named gaps interlock. Classical
machinery, not a research gap; machinery the Stage-2 plan does not
contain, resting on contracts nobody has.

## Target 2: the PID accept/reject loop (#386's layer)

**The ranking function exists and its pillars are proved** (any
`dt ≤ (tol/C)^{1/6}` is accepted; accepted steps ≤ span/(ρ·dt\*)) —
termination-in-principle is real, under two contracts (local-error model,
controller skeleton; fidelity demotion recorded for the simplified
skeleton). Then the number: with the error constant taken from the
supply-proved box's boundary layer, **N ≈ 1.3×10⁹ against the filed
max_steps = 10⁵**. The bound is vacuous at the filed scale; tightening it
needs the error constant along the **trajectory tube**, not the box sup —
machinery no plan contains.

## Band landing

**The 0/3 is real — with structure.** Both targets hit the second band's
disjunction (a determinant condition; a nonlinear product invariant; a
ranking condition with a vacuous constant), and the third band's sentence
fires for the machinery: *the Stage-2 plan is aimed at the wrong object*
for these incidents — what discharges them is Kantorovich conditions,
controller contracts, trajectory-tube constants, and NN Jacobian bounds,
none of which any current plan contains. Induction on the solver's step
map exists but is **contract-stacked**: each layer descended minted new
obligations (`|f| ≤ M` → `|∂f/∂x| ≤ L` → dt-conditions → error models) —
contracts all the way down. What induction sells cheaply remains the
state-space envelope.

**The CROWN dependency, checked (§5):** the blocking NN contracts are
CROWN-tier, and the JAX-native option is **dead** — `google-deepmind/
jax_verify` is archived (last push 2023-08; PyPI 1.0 from 2020-10) —
while `auto_LiRPA` is active (pushed 2026-06) but PyTorch. The
learned-component contract is reachable only across a process/framework
boundary (the `STELLING_CVC5` pattern: export, bound out-of-process,
import the contract). "Learned nodes" reclassifies from medium-tier
nice-to-have to **a prerequisite for #368's flagship-shaped proof, with
no maintained JAX-native foundation**.

## §3 — the 20 properties on the state/trajectory axis

| class | count | which |
|---|---|---|
| **state / state-relation** (induction candidates) | **13** | dfx#207 #223 #417 #632 #657 #752, jmd#339, bjx#D416 #969 #973, npy#249 #552 #1133 |
| **trajectory / event** (not an invariant of anything) | **6** | dfx#368 ("completes"), dfx#386 (termination), dfx#507 (event fires), dfx#756 (liveness), jmd#92 (run equivalence), npy#1360 (zero divergences) |
| mixed | 1 | dfx#194 (state-finite ∧ steps-bounded) |

**Addressable-by-induction subset: 13 of 20 by predicate shape** — with
this probe's own caveat attached: a state predicate can still sit at the
wrong layer (the 13 include the supply probe's provable-but-wrong-layer
cases). The demand band is untouched; this is a supply fact. The two
paid-for corrections (state-predicate requirement; checked-against-its-
incident requirement) bind every future one-line-property registration.
