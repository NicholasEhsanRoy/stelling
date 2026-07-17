# E2a — check mode, registered before the MVP runs

**Status:** REGISTRATION, 2026-07-18. Committed before the MVP produces
its first verdict. E2a's bands live in `design/value-model-v2.md` (they
are v0.2's E2 bands, unchanged); this document fixes the operational
definitions those bands need — now, because "what counts" is exactly
where the result gets renegotiated at 4am.

## What E2a is

The harness states a candidate box; stelling verifies its inductiveness
(edge-flux: at each face, the vector field points inward). E2a tests
whether **checking is mechanizable**. The user writes the invariant — the
product under test is a methodology, not push-button. E2b (derive mode)
is deliberately not registered here.

## "No hand assistance beyond harness setup" — the definition

Permitted, and only this:

1. **Transcribing the system** as a jax function faithful to the
   incident's own code (the MWE's vector field, its constants inline as
   written there);
2. **Stating candidate boxes** — that is check mode's defining input;
3. **Declaring harness inputs**: shapes, dtypes, bounds via
   `any_array` / `assume` / `assert_`.

Everything else is the tool's job, and doing any of it by hand voids the
case: bracketing transcendental constants, decomposing the box into face
obligations, interval evaluation, sign reasoning, rounding direction. If
a case needs a hand-computed fact fed in as an assumption, that case
reports as **not mechanized**, whatever the final verdict says.

## What counts as re-deriving — fixed now

- The **hand-proved box is always candidate #1** (`x1 ∈ [6.8, 415.0]`,
  `c ≥ 0.019`, from `corpus/supply/hit386_termination.py`), and its
  verdict is always reported, whatever else is tried.
- A hit counts as **mechanized (1)** iff stelling returns VERIFIED on a
  stated box that (i) contains the incident's own initial data and
  (ii) discharges the hit's load-bearing qualitative obligation — for
  hit386: bounded away from the log singularity, `c ≥ c′` for some
  `c′ > 0`.
- A **substitute box** (weaker than the hand proof but still meeting (i)
  and (ii)) counts **only with the exact-box failure reported in the same
  sentence** — never silently.
- Anything less — a box that verifies but doesn't discharge the
  obligation, a near-miss, "it almost closed" — counts **0**. No partial
  credit; "the harness was wrong" is a work item, not a band adjustment.

## Verdict semantics, fixed before the first verdict exists

VERIFIED only when every face obligation's propagated interval is
strictly on the required side. Anything else is **UNKNOWN** — an interval
straddling zero is *our imprecision, not their counterexample*; E2a's
checker never emits a refutation. (Witness-checking is the wedge's
discipline and is out of scope here.)

## The positive control (§10.8, satisfied for once by accident)

hit386's box is already hand-proved with a Z3 script in the repo — E2a's
first case has a known answer, the positive control this project has
never had. Both directions ship:

- **Known-good**: the hand-proved box. VERIFIED ⇒ mechanization is real
  at the priced layer on one case; not-VERIFIED at a known-good target is
  precisely the datum nothing else can produce, and the model already
  says what it means (E2a band: 0 including this case = falsified).
- **Known-bad (the mutation that must come back red)**: a deliberately
  non-inductive box — `x1` capped below the equilibrium (e.g.
  `x1 ∈ [6.8, 300]`, whose upper face has outward flux since
  `x1* ≈ 414.5`) — must NOT verify. A checker that passes the mutation
  proves nothing and the run is void.

## Fidelity demotions, inherited and recorded

- **Continuous flow, not discrete step.** The hand proof bounded the
  continuous flow; E2a inherits that limit verbatim. Re-deriving a
  continuous-flow box is **not** re-deriving a solver's invariant, and no
  E2a artifact may blur them.
- **Transcendental constants at libm fidelity.** The tool's `exp`
  transfer assumes a faithfully-rounded libm (≤1 ulp) and rounds outward
  one ulp each side — the *same* demotion the hand proof's
  `np.nextafter` brackets carried. E2a matches the hand proof's standard;
  it does not exceed it.
- **Scope fact, recorded before the build**: if interval arithmetic with
  outward rounding discharges all face obligations, the solver layer
  defers entirely — the first verdict then involves no solver, and the
  stamp records the solver fields as absent rather than omitting them.
