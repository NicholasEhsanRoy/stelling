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

---

# Reading (2026-07-18 — `corpus/supply/e2a_hit386.py`, the first verdict)

## Case 1: **VERIFIED, on the exact hand-proved box** — count: 1, no substitute

One traced query, three face obligations, one verdict:

```
== VERIFIED
  assert #0 (x1 = 415 face):  discharged
  assert #1 (x1 = 6.8 face):  discharged
  assert #2 (c  = 0.019 face): discharged
query f694ca39…604f19cd2b | stelling 0.1.0 | jax 0.11.0
arithmetic: interval/f64/outward-1ulp | solver: none — every obligation
  judged by outward-rounded interval arithmetic alone
coverage: 143 eqns: 143 known (100%)
```

- **Candidate #1 verified as stated** — `x1 ∈ [6.8, 415]`, `c ≥ 0.019`
  with the c-side genuinely unbounded (half-infinite intervals; the
  `0·∞ = 0` closed-interval convention carries the x1-faces). No weaker
  substitute was needed; the substitute clause goes unused.
- **The solver deferral is real**: the registered scope fact fired — all
  three faces discharged by interval arithmetic; the stamp records solver
  absence explicitly. The Z3 layer of the hand proof was never needed for
  this shape, exactly as §3 of the work order suspected.
- **No hand assistance beyond the permitted list**: the parameters enter
  as point `any_array`s so `exp` is traced and bracketed by the tool
  (`sound-libm` tier, assumption in the stamp); the box is stated in its
  own `(x1, c)` coordinates; every face substitution is traced code.

## The mutation came back red — with face precision

The x1-ceiling-at-300 harness (below `x1* ≈ 414.5`) returns **UNKNOWN**
with assert #0 reported *"definitely false over the declared box"* while
the other two faces still discharge. The checker is not vacuous, and its
red names the failing face. Per the registered verdict semantics this is
never rendered as a refutation.

## The first run was UNKNOWN — and that is the instrument working

The harness's first version assembled the parameter vector with
`jnp.stack`; on jax 0.11 `stack` is its own primitive, **outside the §2
census list** (the censused field *received* the array). Coverage came
back `143 eqns: 137 known (99%); 1 ⊤ (stack ×1)` and the verdict
correctly degraded to UNKNOWN — a one-equation scope deviation, caught by
name, by the ⊤-coverage line the falsifier discipline demanded. The fix
conformed the harness to the censused form (`jnp.array` →
broadcast+concatenate) rather than growing the registry past §2's list.
Recorded because it is the first live instance of "a null at low coverage
is not a null" — here inverted: an UNKNOWN at 99% coverage, with the 1%
named, is a work item with an address.

## Fidelity demotions, inherited as registered

Continuous-flow box, not a solver invariant — this verdict re-derives the
hand proof's object, nothing more. `exp` at libm fidelity, 1-ulp outward —
the hand proof's own standard, now printed in the stamp of every verdict
that uses it.

## E2a status

**1 of 13, on the positive control.** Mechanization is real at the priced
layer on the one case with a known answer; the E2a bands
(`design/value-model-v2.md`) stay open until the other 12
reconstructions run — future passes, not this one.

---

# Amendments (2026-07-18, after the first verdict, before the twelve)

## The amendment rule — stated before it is used

Editing a registration is the move this project forbids; an amendment is
the narrow exception, and it needs a rule written when nothing is at
stake:

> **An amendment to a live registration is permissible iff all three
> hold: (a) it is additive; (b) it cannot move a count in either
> direction; (c) the defect it corrects was found by the registration's
> own control, not by a result anyone wanted.** It is recorded as an
> amendment, with its justification, never silently.

## Amendment 1: REFUTED enters the verdict vocabulary

The registered sentence *"E2a's checker never emits a refutation"* is
superseded — **its stated reason never covered the rule it wrote**. The
reason was about straddles ("an interval straddling zero is our
imprecision, not their counterexample"), and straddle → UNKNOWN remains
correct. Definite-false is not a straddle: the propagated set contains
the reachable one, so *definitely false over the declared box* is a
**sound set-level refutation of the stated box** — the box is not
inductive as stated. It is not a witness (no concrete input; the wedge's
discipline is untouched) and not a counterexample to the program; it is
the claim the mutation control earns, and it shipped under "no
information".

Verdict semantics, as amended: **VERIFIED** (all obligations definitely
true) / **REFUTED** (at least one obligation definitely false over the
declared set) / **UNKNOWN** (everything else, straddles included).

Against the rule: (a) additive — a third status, no existing meaning
changed; (b) count-neutral — REFUTED is not VERIFIED, so an E2a case
returning it still counts 0 and the band arithmetic is untouched;
(c) found by the registration's own §10.8 mutation control, whose red
went out labeled "no information". The mutation harness now renders
REFUTED, with the failing face named.

## Amendment 2: withdrawn as an amendment, re-filed as a registration

*(2026-07-18, same day, content unchanged — now
`design/assume-registration.md`.)*

The reclassification, recorded because reclassifications are never
silent: this registration was **silent** on `assume`'s propagation
semantics. The closest it comes is the permitted-list enumeration
("bounds via `any_array` / `assume` / `assert_`"), which is an API
listing, not a semantic commitment — it lists `assert_` in the same
breath, and nobody reads `assert_` as bounds-declaring; the registered
verdict semantics spoke only to `assert` obligations. **Filling a
silence before the run is a new registration, not an amendment**, and
new registrations have never been restricted.

Had it been an amendment, it would have failed clause (c): the fix was
proposed while explicitly predicting it would bite dfx#752 and bjx#969
in the direction that helps them count — motivation by an anticipated
result, the softer version of what (c) exists to catch. The rule is
**not** widened to accommodate it; a rule loosened when it binds is
decorative. The amendment mechanism remains for one thing only:
superseding registered text that exists and is wrong, as amendment 1
did.
