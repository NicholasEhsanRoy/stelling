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
