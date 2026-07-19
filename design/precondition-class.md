# The assumed-preconditions class — SPD + nullspace as first instances

**Status:** REGISTRATION, 2026-07-19, committed **before the transcriber
runs**. MADDENING read at `849c391` (the ref on disk; recorded, since
this pass received no pin). Magnetics is Nick's code, **held out from
E2a** — usefulness/reliability results, never a public-track count.

## The class

**Pre-solve preconditions that solvers assume and never check**: a
property the numerical method's validity rests on, living in the
input/coefficient data *before* the expensive computation, that the code
assumes (often literally asserts to a library) and never verifies, and
that **fails silently** — a wrong answer under a healthy-looking
convergence flag. Pointwise or scalar properties of declared inputs:
no solve, no mask, no dependency wall, no relational machinery. The MVP
interval core already decides these.

**The class boundary is exactly the solve.** In-class: preconditions on
the *inputs* to the solve (coefficient positivity over the envelope).
Out-of-class: properties of the *solve's behaviour* (conditioning,
residual-bounds-error) — step 2's LA contract. An obligation that needs
the solve's behaviour or crosses the active-set mask has left the class;
**a mask crossing is a stop, not a push-through.**

## The templates (built this pass) and the named roadmap (not built)

Two reusable templates in `stelling.preconditions`, parameterised over
envelope and the code's own transform — **templates, not `1+χ`-specific
checks**:

- **`field_positive`** — pointwise positivity/coercivity:
  `transform(field) > bound` over a declared envelope. Instance #1.
- **`scalar_nonzero`** — a config scalar must be nonzero for
  well-posedness. Instance #2.

Named for the template design, **not instantiated this pass** (the
class's roadmap, recorded so the templates are designed to hold them):
operator symmetry (`A = Aᵀ`); non-zero/positive diagonal (the
`where(D > 0, D, 1.0)` clamp *substitutes* rather than fails — "is D
positive over the envelope" catches the silent substitution);
special-function domain (`log`/`sqrt`/division arguments in-domain —
the universal silent-NaN origin); operating-envelope bounds (input-side
only — the behaviour-side consequence is step 2). Plus one **sibling
sub-shape surfaced by the characterization, recorded as a design
question rather than an instance**: configuration preconditions ("is
the code running in the configuration its correctness assumes" — the
float64-flag gap below). It may be static-lint-shaped rather than
interval-shaped; deciding that is not this pass.

## The two instances, as posed

**Semantics statement (§4 of the work order, held):** obligations are
posed under `jax_enable_x64=True`, `float64` declarations, stamped
`real` — matching the code's *intended* configuration. The
characterization's finding that the code hardcodes float64 at ~13 sites
but never checks the global flag (and silently degrades to float32
without it) means the intended and actual configurations can diverge;
that gap is the configuration-precondition sibling above, disclosed
here, not modelled by these obligations.

**Instance #1 — SPD / coefficient coercivity.** The magnetics operator
is `-∇·(a∇·) + mass·I` with `a = 1 + χ`; CG/Cholesky and the lineax
solve assume SPD, which needs `a > 0` everywhere. Two posings:

- **A (supported envelope):** `χ ∈ [1e-6, 1e2]` (soft tissue → the
  supported cap; bare numbers to the transcriber), the code's own path
  from χ to the coefficient, obligation `a > 0` pointwise.
- **B (sign-spanning source):** `χ ∈ [−2, 1e2]`, same code path, same
  obligation. **The χ-sign question, stated explicitly as required:** χ
  is an *inferred* field (θ, the inversion parameter). Physically
  χ ≥ −1 (χ = −1 only at a perfect diamagnet), but *inference* is not
  physics — an unconstrained optimizer step can propose any float. What
  posing B measures is whether **the code's own construction** constrains
  the coefficient (a clamp/softplus/abs between θ and `a`) or passes it
  raw. Protection found and traced → VERIFIED over the sign-spanning
  source **is the protection, verified**. No protection → REFUTED with a
  witness at `χ ≤ −1` — the SPD hazard caught: unguarded,
  asserted-to-the-solver, silently non-SPD.

**Instance #2 — nullspace / `mass ≠ 0`.** Under periodic BCs,
`-∇·(a∇·)` annihilates constants; `mass·I` is what removes the
nullspace, and `mass` is a config float (default 1.0). Two posings:
the **default** (point 1.0) → expect VERIFIED; the **admissible config
range** `[0, 1]` (0 is a legal float a config can set) → interval-only
UNKNOWN (the range contains 0 and non-0), escalated **REFUTED with
witness `mass = 0`** — the honest reading being "the config space admits
a singular operator and nothing checks it," not "the default is wrong."

## Predictions (pre-committed; the main agent's, held from the transcriber)

- A: **VERIFIED** (supported envelope is strictly positive; `1 + χ > 0`
  by margin 1).
- B: **REFUTED with a witness at `χ ≤ −1`** — my source read found
  `_coeff_field` applies no constraint on the default path (`a = θ`
  verbatim, or a user `coeff_fn`). The transcriber reports this
  independently and blind.
- mass default: VERIFIED. mass over `[0,1]`: interval UNKNOWN →
  escalated REFUTED, witness 0.
- Registry risk, named: the code-form path runs through face averaging
  (`jnp.roll` neighbours). If `roll` lowers to unregistered forms, the
  code-form posing declines — that is the "won't pose" band, reported
  with the decline quoted; the pointwise-field posing is the documented
  fallback **with the face-averaging-preserves-positivity lemma
  disclosed as a hand step and priced per L11** (re-run both, report
  the delta).

## Bands (fixed; every row surfaces)

| outcome | reading |
|---|---|
| both instances pose; A VERIFIES over the supported envelope | the class is real, the templates validated; **step 2 (LA contract) greenlit** |
| B REFUTES with a `χ ≤ −1` witness | the SPD hazard is real and caught on Nick's code — report with the witness, replayed |
| won't pose (a wall these mask-free obligations shouldn't hit) | a finding that changes step 2's scope — surface the decline verbatim |
| an obligation would cross the active-set mask | **stop** — the class boundary was violated; step-2 territory arriving early |

## Orchestration (light, per the work order)

Main agent: templates, instances, provenance (χ-sign, float64-flag,
this registration), witness replay, the §6 over-specialization sweep
(report-only), gated commit. **Fresh transcriber**: magnetics source
only (`wavelet_elliptic.py` + its wavelets imports), blind to the
characterization's conclusions and to wanted verdicts; envelopes travel
as bare numbers. **No auditor round unless a new transfer is added** —
these run on the existing interval core; any registry gap is a coverage
finding reported, not built past.

## §6 — the over-specialization audit (report-only, end of run)

Sweep the existing machinery for things coded to a specific property
where the class-level shape recurs; list candidate generalisations with
costs; **refactor nothing**. Nick decides with the list in hand.

---

# Reading (2026-07-19 — registration `b5c28b9` preceded every number below)

`corpus/supply/maddening_preconditions.py` (blind transcriber; ring
verified against the harness text — no characterization terms, no
conclusions, no history; the quoted source lines re-checked against my
own read). Every verdict below independently re-run by the main agent;
the D witness and the fallback witness replayed in my own `Fraction`
arithmetic.

## The four posings

| posing | interval | escalated | diagnosis |
|---|---|---|---|
| **A** — code path (incl. face averaging), θ ∈ [1e-6, 1e2] | **VERIFIED** (both face arrays, 4/4 elements) | — | coverage 89% + 2 transparent (`jnp.roll`'s jit sub-jaxprs; innards propagated) |
| **B** — identical, θ ∈ [−2, 1e2] | UNKNOWN (honest straddle) | **UNKNOWN — declined**: `assert operand has shape (4,) (v1 emission is scalar-only)` | the code path is irreducibly an array (roll couples neighbours) |
| **C** — mass = 1.0 (the default, quoted from `wavelet_elliptic.py:125`) | **VERIFIED** | — | 100% coverage |
| **D** — mass ∈ [0, 1] | UNKNOWN | **REFUTED, witness `mass = 0`** (cvc5, QF_LRA, replay-confirmed; z3 sat but non-rational model — by policy not relied on) | the *config range* admits the singular operator; the default (C) is safe |

**The sign-constraint answer, from the source alone: NO.** The default
`_coeff_field` path is verbatim `return theta  # a = θ (the coefficient
itself)` (`wavelet_elliptic.py:236`); `make_varcoeff_apply` applies only
reshape, the affine face average `0.5·(a + roll(a, ±1))`, and a positive
constant. **Nothing between the inferred parameter and the operator
coefficient constrains its sign or range.**

## The registered fallback for B — run by the main agent, L11-priced

The code-form decline triggered the registration's documented fallback:
the scalar posing of the code's own default path (`a = θ`, line 236),
with the face-averaging lemma **disclosed as the hand step it is**
(cell positivity ⟹ face positivity: each face coefficient is the mean
of two cell values). Result: interval UNKNOWN → **escalated REFUTED with
witness `θ = 0`** — in-region, violating, replayed in my rationals. The
supported-envelope control on the same form: VERIFIED.

**The L11 price, stated:** code-form B is UNKNOWN-blocked at scalar-only
emission; fallback B is REFUTED-with-witness. The delta is exactly the
array-emission gap plus one disclosed one-line lemma.

**Prediction scorecard (L4):** A ✓, C ✓, D ✓ exactly as pre-committed.
B: the *verdict* prediction (REFUTED with witness) held **on the
fallback, not the code form** — I predicted the refutation as primary
and the decline as the named risk; measured, the decline was primary.
And the witness coordinate is `θ = 0`, not `χ ≤ −1`: the code's default
path has no `1+χ` — `a = θ` directly — so the SPD boundary in code
coordinates is `a ≤ 0`; the χ-form applies only when a caller supplies
`coeff_fn = 1+χ`. The finding is unchanged and sharper: **the inference
space admits an indefinite operator and nothing checks it.**

## Band adjudication — row 1, with the row-2 finding alongside

**Row 1 lands: both instances pose; A VERIFIES over the supported
envelope.** The class is real, the templates are validated (with one
measured gap, below), the two silent-failure modes are now checkable —
**step 2 (the LA contract) is greenlit as this band's outcome.** Stated,
not acted on: build order remains Nick's.

The row-2 substance arrived via the fallback: **the SPD hazard is real
and caught** — unguarded (line 236 quoted), asserted-to-the-solver,
and REFUTED with a replayed witness over the sign-spanning inference
range. D's witness (`mass = 0`) is the nullspace hazard's config-space
form: the range admits it, the default avoids it, nothing checks it.

## Findings beyond the bands

1. **Scalar-only emission blocked the code-form posing on a second
   codebase** (FVM R2/R3, now magnetics B). Recorded as **demand
   sighting #2 for array-aware emission** — the payoff analysis said
   that build should wait for "a reason other than this headline";
   independent blocking instances accumulating on real code is that
   reason forming. Recorded, not built.
2. **The template gap found on first application:** `field_positive`
   takes one transform producing one value; the real code path produces
   *two* face arrays, so the transcriber wrote the A/B harnesses
   directly. §6 candidate: transforms returning a tuple, one obligation
   per value.
3. **The layered witness policy worked as designed under divergence:**
   z3 returned sat with a non-rational model for a QF_LRA query; the
   REFUTED rests on cvc5's replay-confirmed witness, z3's answer
   recorded but not relied on.
4. **First contact was clean** — no crash, no registry gap, no posture
   escape. Not a counterexample to L14: the class ran on the well-worn
   scalar core by design (the boundary "input-side, mask-free" is also
   an attention-surface boundary); L14's prediction concerns new
   *primitive* surfaces, which this pass deliberately avoided.

## §6 — the over-specialization sweep (report-only; nothing refactored)

Candidates where a class-level shape would serve, each with its cost —
**Nick decides with this list; none of it was acted on:**

- **The ⊤-widening instrument is convention-copied per harness** ("the
  `tautology_test.py` pattern", re-implemented in `maddening_cfl.py`,
  `mime_fvm.py`, …) — and the procedure has already changed once
  (inputs-only variant), so copies can drift against the registered
  instrument. Generalisation: a library `widen(closed, mode)` in src,
  with the per-pass scripts calling it. Cost: small. **Strongest item.**
- **The regional-obligation pattern recurs unabstracted** (heat's A2
  control, R1, the fallback here): declare region, apply the code's
  transform, assert a bound. Generalisation: a `region_bound` template
  beside `field_positive`. Cost: small.
- **`field_positive`'s single-value transform** (finding 2 above).
  Cost: trivial, but it is an API change — batch with the step-4
  usability pass.
- **`stack` is still unregistered** — hit386's harness conformed
  `jnp.stack` → `jnp.array` to dodge it (ledger L11's second witness).
  Generalisation: one censused registry row. Cost: one row + tests.
- **Deliberately NOT flagged:** the corpus harnesses' specificity
  (fidelity to incidents is their point — generalising them would
  violate the fidelity census); the form-guarded registry rows
  (census-by-census scope, not over-specialization); scalar-only
  emission (a documented, this-pass-priced scope decision).

## Ledger

L14 gains a scope note (recorded in the ledger): first-contact risk
tracks new *primitive/execution* surfaces, not new codebases per se — a
class that deliberately stays on the well-worn core can meet a new
codebase cleanly, and did.
