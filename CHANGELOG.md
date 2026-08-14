<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# Changelog

## 0.2.0 — unreleased

### New transfers and precision improvements

- **`is_finite` transfer**: returns definite-true for bounded intervals,
  definite-false for point-at-infinity (`[inf, inf]`), unknown otherwise.
  Unlocks the `jnp.where(jnp.isfinite(x), ...)` pattern that MADDENING's
  Aitken relaxation depends on — `select_n` can now prune unreachable
  branches when the selector's `isfinite` result is decidable.

- **`int64→float64` point-interval conversion rule**: when an integer
  constant is cast to float64 and is exactly representable (in [-2^53,
  2^53]), the interval passes through instead of declining to top.
  Unblocks 41 jax-md `safe_mask` sites.

- **Boundary-aware division**: when the divisor has zero at exactly one
  boundary (`[0, hi]` or `[lo, 0]`) — the case `assume(x > 0)` produces
  — compute a meaningful result instead of declining. True straddles and
  point-at-zero still decline with an actionable message.

- **Div-straddle decline**: when float division has a divisor spanning
  zero (true straddle), the transfer now declines with a message naming
  the interval and suggesting remedies, instead of silently returning
  `[-inf, inf]`.

### Float32 / float16 / bfloat16 IEEE mode

- **Format-parametric IEEE semantics**: the existing `semantics="ieee"`
  mode (previously binary64-only) now supports all four catalogued
  formats. Each operation rounds interval endpoints outward to the target
  format's ULP grid, models per-format subnormal flush, and handles
  format-specific overflow.

- **IEEE assume-bump** (`_format_nextafter`): `assume(x > k)` in IEEE
  mode narrows to `[nextafter_fmt(k, +inf), hi]` — the smallest
  representable value strictly above k in the target format. Works for
  all k, all formats. In combination with boundary-aware division, the
  `assume(b > 0); a / b` pattern produces decidable quotients.

### Verification pipeline

- **Reachability conjunct**: a backward walk from the jaxpr's outputs
  identifies variables that flow to an output. Violated obligations on
  "dead" variables (computed but never observed by the caller) are
  downgraded from REFUTED to UNKNOWN with a note. The fail-safe is
  always REFUTED: obligations that cannot be proven dead keep their
  status.

- **Solver selection API**: `check(..., solver="z3")` or `solver="cvc5"`
  restricts the SMT portfolio to one backend. The verdict explicitly
  discloses degraded redundancy.

### SMT emission extensions

- **`is_finite` emission** (guarded): emits constant `true` when the
  operand's propagated interval has finite endpoints; declines when
  infinite (sound: bounded reals are finite by construction). Unblocks
  solver escalation on every harness containing `jnp.isfinite()`.

- **`pow` emission** (integer AND non-integer exponents): integer
  exponents (`x**2`, `x**3`, `x**(-1)`) expand to explicit products.
  Non-integer exponents emit as auxiliary-variable polynomial constraints
  (`aux^q = x^p` with sign constraints) — both z3 and cvc5 handle these in
  QF_NRA. **The rational `p/q` must be the exact value of the traced
  binary64 literal**, which admits `x**0.5`, `x**0.25`, `x**0.75`,
  `x**1.5`, `x**(1.0/64.0)`, `x**(1.0/128.0)` — every dyadic — and
  declines `x**0.1`, `x**(1.0/3.0)`, `x**(1.0/80.0)` to UNKNOWN, because
  those literals are NOT the low-denominator rationals they are written
  as and emitting about a nearby rational is emitting about a different
  function. One cap (128) bounds the degree of the emitted equation on
  both sides, so a large numerator (`x**100.5` → `aux^2 = x^201`) declines
  exactly as a large denominator does. Base must be non-negative (JAX
  returns NaN for `pow(negative, fractional)`).

### Soundness fixes

- **Rational-`pow` exponent identity** (audit 0.2.0 S1; see
  [SOUNDNESS.md](SOUNDNESS.md)): the exponent was rationalised with
  `Fraction(e).limit_denominator(128)` and admitted on a *binary64*
  distance test, which measures exactly `0.0` for `0.1`. Verdicts move
  **VERIFIED → UNKNOWN** on every non-dyadic non-integer `pow` exponent;
  affects 0.2.0 development only.

- **No emitted term is a unary `(* t)`** (audit 0.2.0 S2): `q == 1` wrote
  an application SMT-LIB2's `Reals` theory does not define — cvc5 1.3.4
  segfaults on it, z3 reads it as the operand. Every repeated product now
  goes through one renderer (`smt._repeated_product`).

- **The rational-`pow` replay is exact** (audit 0.2.0 S3, M8): it computed
  `Fraction(float(base) ** exp)` while every REFUTED witness claimed
  "independent exact-rational replay". It now extracts exact integer
  `q`-th roots, or declines the witness through the existing
  "witness not independently replayable" channel. The public `check()` no
  longer raises `EmissionInfidelityError` on correct emissions, and the
  replay's `OverflowError` on large operands is gone with the float.

- **The fragment stamp follows the aux encoding** (audit 0.2.0 M9): a
  non-integer `pow` over a declaration-independent base was stamped
  `QF_LRA` while the emission wrote `(* aux aux)`, and both backends
  refused the script.

- **Relational assumes forwarded to solver**: when `assume(e1 < e2)`
  involves two variable operands (a constraint the interval domain cannot
  apply), the comparison is recorded and emitted as a positive axiom
  alongside the negated obligation. The solver sees the full constraint
  set.

- **SOUNDNESS FIX — a forwarded assume is now resolved by a scope-correct
  identity; it could previously be emitted about the wrong values.**
  See the SOUNDNESS.md log entry for the full account. In brief: a
  relational `assume` traced inside a `jit` / `custom_jvp` body was
  forwarded as its producing comparison equation, whose operand ids belong
  to that body, and `smt.emit` resolved them with a bare integer lookup
  against the slice's *renumbered* table. When the two id ranges met, the
  axiom was emitted about unrelated terms — measured as the CONVERSE of
  the user's own precondition, returning VERIFIED on an obligation false at
  every admitted point. Development-only; no released version is affected.

  What changed, user-visible:

  * `propagation.relational_assumes` now holds
    `stelling.propagate.RelationalAssume` records (the comparison equation
    plus the scope path its operand ids belong to), not bare
    `ir.JaxprEqn`s.
  * `ObligationSlice` carries `assumes` (translated into the slice's own id
    namespace) and `assumes_skipped` (one quoted reason per assume this
    obligation cannot state). The two partition the assumes the slicer was
    given, so *emitted versus requested* is derivable from the slice alone.
  * `stelling.smt.emit` no longer takes a `relational_assumes` parameter —
    the axioms come off the slice. `Script.relational_assumes_emitted` now
    counts assumes emitted **about the terms their operands denote**, and
    `Script.emitted_origins` names *which* ones, by their index in the
    propagation's forwarded tuple (`SliceAssume.origin`).
  * `slice_obligation` gained a `relational_assumes=` keyword;
    `slice_unknown_obligations` passes the propagation's.
  * **Once escalation dispatches, every assume the slice declines to state
    is disclosed** in the verdict notes, naming the assume's source line and
    the reason. Emission previously skipped silently in five places. The
    per-assume disclosure is produced *at dispatch*, so a run refused before
    dispatch — a constraining assume present, `semantics="ieee"`, no solver
    installed — or an obligation whose slice declines does not carry one; on
    those runs the propagator's own coarse `assume constraint DROPPED` note
    is still emitted, so no assume goes unmentioned, but it names no
    per-obligation reason.
  * **An assume inside a `jit` / `custom_jvp` body is now forwarded
    CORRECTLY rather than skipped**, which decides obligations that
    previously returned UNKNOWN. Measured on a **288-harness** generated
    sweep (`sweep_assume_scope.py`, the instrument's full product:
    4 carriers × 2 ndecls × 3 tails × 3 assume-sets × 2 exprs × 2 orders):
    **96 UNKNOWN→VERIFIED and 36 UNKNOWN→REFUTED**, no harness moving away
    from a decided verdict, and zero verdict changes on the **72**
    top-level-assume harnesses. Of the 96 new VERIFIEDs, **48 are vacuous**
    — an `unsat` assume set now reaches the solver from a `jit` body as it
    already did from top level; see the SOUNDNESS.md entry.
  * **A relational assume inside a `lax.cond` branch is no longer forwarded
    at all.** It is a branch-scoped precondition, not a fact about the
    query; the drop says so and keeps violations withheld.
  * `smt.emit` no longer raises `IndexError` on a shape-mismatched assume,
    and no longer emits a partial axiom over element 0 of an unrelated
    array (both arms of the same missing check).

- **SOUNDNESS FIX — a withheld violation is released only when every
  `assume` is accounted for, and that is now decided by a per-assume
  LEDGER rather than by two counts.** See the SOUNDNESS.md entry. The rule
  compared `len(propagation.relational_assumes)` against a script's emitted
  count, and that shape produced a false REFUTED twice: once because the
  denominator counted only the *relational* assumes while the flag gating
  the rule is set by any drop reason at all (audit 0.2.0 S6), and once
  because no longer forwarding branch-scoped assumes silently moved the
  denominator, so `1 == 1` released a witness whose branch precondition the
  solver had never been told. Development-only; no released version is
  affected.

  What changed, user-visible:

  * `Propagation.assume_ledger` — one
    `stelling.propagate.AssumeDisposition` per assumed conjunct the
    propagator classified, with kind `applied`, `no-op`, `forwarded` or
    `dropped`. It is written where the classification happens and is TOTAL
    over the assumes the walk sees, including inert mode.
  * `stelling.propagate.unaccounted_assumes(ledger, emitted_origins)` is
    the release test: a definite violation is released only when it returns
    empty. It joins on identity, counts nothing, and **whitelists** the
    accounted-for dispositions — a kind it has not been taught is
    unaccounted, so a drop reason added later refuses rather than defaults
    open.
  * The withholding note now NAMES the conjunct that caused it, with its
    disposition, reason and source line, instead of restating the rule.
  * `Propagation.assume_dropped` is unchanged and still gates the rule.

- **z3 tactic workaround for high-degree polynomials**: when a solver
  obligation contains a rational-pow auxiliary variable (`y^q = x^p`
  encoding), z3 uses a custom tactic chain (`simplify`, `solve-eqs`,
  `factor`, `purify-arith`, `tseitin-cnf`, `nlsat`) instead of the
  default `Solver()`. This restores the z3 cross-check on high-degree
  polynomials (measured: d=80 from 10s+ timeout to 0.35-0.6s). The tactic
  is activated automatically; cvc5 handles these natively.

- **Per-obligation withholding refinement**: when relational assumes are
  only partially emitted for a given obligation slice (some operands fall
  outside the backward cone), the solver ran over a wider domain than
  intended. A definite violation is un-withheld ONLY when every assume the
  user wrote is accounted for on **that** obligation's query — see the
  ledger entry above for the rule that decides it.

- **An assume that excludes nothing no longer withholds forever.** An
  assume whose entire content is a conjunct definitely TRUE over the boxes
  in force (`x ∈ [0,10]`, `assume(x >= -1. | x >= -2.)`) took the whole-drop
  path, which sets the withholding flag unconditionally, and the old release
  test could never fire on it. The ledger records that conjunct as `no-op`
  and the violation is released — the rule the mixed-conjunction path
  already applied to the same class of conjunct. Measured: UNKNOWN → REFUTED
  at `x = 6`, which is in the declared box, satisfies the assume, and
  falsifies the assert.

- **Emission guards resolve through inlined aliases**: guards (div, is_finite)
  now follow the slicer's alias chain to find propagated intervals for
  variables defined inside transparent calls (jit, custom_jvp_call).

### Inductive step verification

- **`stelling.inductive.check_inductive_step`**: verify that a loop body
  preserves declared bounds in one step. VERIFIED means the invariant
  holds for all iterations by induction. Constructs the harness
  automatically from the body function and declared state bounds.
  Supports scalar and array-shaped state variables (shape specified per
  variable in the bounds declaration).

### Known limitations (0.2.0)

- `assume(x > 0)` in real mode still narrows to `[0, hi]` (closed
  intervals cannot represent open bounds in exact reals). The IEEE bump
  is exact; the real-mode overapproximation is sound. In real mode,
  boundary-aware division handles the resulting `[0, hi]` gracefully.
- The dependency problem (A ∧ ¬A = unknown in intervals) is inherent to
  the non-relational domain. Solver escalation is the designed remedy.
- Rational pow requires non-negative base (JAX returns NaN for
  `pow(negative, fractional)`). One cap (128) bounds the degree of the
  emitted `aux^q = x^p` on both sides.
- **A non-integer `pow` exponent escalates only when it is a small dyadic
  rational**, because that is the only case where the emitted rational IS
  the traced binary64 literal. `x**(1.0/3.0)` and `x**0.1` decline to
  UNKNOWN. Admitting them soundly is a larger feature and was deliberately
  not built in this round: it needs the substitution *stamped as an
  assumption*, its amplified error `|x^a − x^(p/q)| ≤ x^a·(e^{|δ|·ln hi} − 1)`
  bounded against the obligation's slack over the declared box, and the
  discharge direction barred until that bound exists. Declining is the
  sound posture in the meantime.
- **A REFUTED through a non-integer `pow` needs a witness whose exact
  value is rational.** The replay extracts exact `q`-th roots; where the
  true value is irrational it reports "witness not independently
  replayable" and the obligation stays UNKNOWN rather than resting on a
  rounded float. Deciding those points needs exact algebraic (not
  rational) arithmetic in the replay, which this release does not have.
- A relational `assume` inside a `lax.cond` branch is **not** forwarded to
  the solver, and is not emitted as an implication either — the drop says
  so. Branch-scoped preconditions therefore buy no solver precision.
- An **unsatisfiable** set of relational assumes makes the emitted script
  `unsat` for a reason unrelated to the obligation, and the discharge that
  follows is vacuous. The unsatisfiable-precondition refusal consults the
  interval domain, which by construction cannot decide a relational
  assume, so it does not see this. Correct forwarding widens the reach of
  this pre-existing limitation from top-level assumes to `jit`-carried
  ones; see the SOUNDNESS.md entry of 2026-08-14.
- An obligation discharged with a **forwarded relational axiom cannot
  narrow the VERIFIED bar**: the bar's re-derivation re-slices without the
  propagation, so its script does not carry the axiom and the two do not
  match. In a query containing a barred primitive the bar therefore falls
  back to the whole query. Conservative (a wider bar, never a narrower
  one), pre-existing, and made more frequently reachable by this release;
  see the SOUNDNESS.md entry of 2026-08-14.

---

## 0.1.0 — 2026-08-12

Initial release.

### Static verification

- Forward interval propagation over the jax-free IR, outward-rounded (one
  deliberate ulp per operation), with three-valued verdicts: VERIFIED,
  REFUTED, UNKNOWN.
- SMT escalation via an optional portfolio (cvc5 for nonlinear, Z3 for
  linear, cross-checked when both are installed). REFUTED verdicts carry a
  concrete witness confirmed by exact-rational replay.
- Every verdict carries a full stamp: versions, query content hash,
  arithmetic mode and semantics, precision configuration, solver
  invocations (or their recorded absence), transfer tiers and provenance,
  assumptions, and coverage.
- Precondition obligation templates (`field_positive`, `scalar_nonzero`)
  with a one-call entry point (`check()`).
- Vacuity checking (two modes: `inputs-only`, `all`) built into the
  pipeline — a VERIFIED that does not depend on its declared envelope says
  so in itself.
- Affine (zonotope) refinement layer for interval-undecided obligations,
  opt-in via `refine="affine"`.
- IEEE-semantics mode (opt-in): judges censused binary64 behaviours and
  stamps itself separately from real-mode verdicts.

### Overflow tripwire

- `pytest -p stelling.overflow` — hooks the constant-fold site where JAX
  silently narrows out-of-range integer literals during tracing.
- Reports each narrowing with source location, arithmetic, independent
  recomputation, and a one-line reproducer.
- **Gates the verifier**: when the tripwire is armed and a narrowing fires
  during a harness trace, the verdict is UNKNOWN — the pipeline refuses to
  certify a jaxpr that does not represent the program as written.
- xdist support: workers serialise findings back; the controller reports
  the true total and flags lost workers.
- Fail-closed on every JAX version change: probes in both directions at
  arm time, disables itself cleanly if the hook site moved.

### Architecture

- Zero required dependencies. JAX and SMT solvers are opt-in extras,
  imported lazily.
- `import stelling` never imports JAX. Only `stelling/_jax_compat.py` may
  import jax; enforced by pre-commit hook and test.
- REUSE-compliant (SPDX headers on every file), DCO-signed commits, PyPI
  Trusted Publishing with PEP 740 attestations.

### Known limitations

- Control flow (`cond`, `scan`, `while`) falls to top and is counted in
  coverage — not handled.
- Default semantics is real arithmetic (ℝ); a predicate can hold in ℝ and
  fail in floats. The stamp names this.
- The tripwire does not see `jnp.full`, `jnp.where`, `jnp.clip`, eager
  execution, or anything traced before the plugin armed. Each is documented
  and printed on every run.

Tested on JAX 0.10.2 and 0.11.0, Python 3.10–3.12, Linux x86_64.
