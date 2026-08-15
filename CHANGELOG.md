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

- **Boundary-aware division, REAL MODE ONLY, and only where a strict
  `assume` excludes the zero**: when the divisor has zero at exactly one
  boundary (`[0, hi]` or `[lo, 0]`) **and** a strict `assume` certifies
  the divisor is nonzero, compute a meaningful result instead of
  declining. True straddles and point-at-zero still decline with an
  actionable message — and so, since the B5-1 fix below, does a
  zero-touching divisor with no certificate.

  **The certificate, and what carries it.** `assume(d > 0)` narrows `d`
  to the CLOSED `[0, hi]` — an interval cannot hold an open bound — so
  the box alone can never say whether its zero endpoint is a value the
  program reaches. The propagator records the exclusion separately and
  carries it through `mul`, `div`, `add`/`add_any`, `neg`, `abs`,
  `square`, `integer_pow`, `reduce_sum` and `dot_general`, which is what
  keeps the row's headline shape — `assume(x > 0); 1 / jnp.sum(x*x)` —
  decidable in all four of its spellings. **A subtraction breaks the
  chain** (two positives can differ by zero), as does every primitive not
  in that list: those decline, naming the remedy.

  **Under `semantics="ieee"` the tightening is WITHDRAWN entirely**: an
  IEEE format has two zeros and an interval endpoint has no sign bit, so
  a divisor box reaching zero divides to `[-inf, inf]` there — and the
  transfer now says so, quoting `interval.IEEE_ZERO_DIVISOR_TOP` as its
  decline reason instead of returning ⊤ as an ordinary result. That ⊤ was
  counted "known", so a reader was told "none fell to ⊤ … compatible with
  a precision near-miss" about a `[-inf, +inf]` box while the same
  verdict's `top_despite_coverage` line named `div ×1`. See the S10 and
  B5-1 entries under Soundness fixes; the two kernels disagree
  deliberately.

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
  all k, all formats. **The `assume(b > 0); a / b` pattern does NOT
  produce a decidable quotient in ieee mode** (it does in real mode):
  `nextafter_fmt(0, +inf)` is the format's smallest subnormal, which the
  DAZ haze immediately hulls back to 0, and a zero-containing divisor is
  ⊤ under ieee since the S10 fix. An assume whose bound is above the
  format's subnormal band (`assume(b > 1e-30)` in float32, say) keeps its
  quotient.

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

- **An IEEE divisor box that reaches zero divides to ⊤** (audit 0.2.0
  S10; see [SOUNDNESS.md](SOUNDNESS.md)): `ieee_div`/`ieee_div_fmt` read
  `[lo, 0]` as *"the divisor approaches 0 from below"* and returned a
  one-signed infinity. Under IEEE the divisor does not approach zero, it
  IS zero at that endpoint, and the sign of `x/0` comes from the ZERO's
  sign bit — which an interval endpoint cannot carry. `+0.0 == 0.0`, so
  `+0.0` is a value of `[lo, 0]` and the excluded `-inf` is a value of
  the program. **FALSE VERIFIED in all four formats**, a 0.2.0
  regression against `v0.1.0` (measured: `v0.1.0` returns `(-inf, inf)`
  where the pre-fix tree returned `(2.0, inf)`). Verdicts move
  **VERIFIED → UNKNOWN** wherever an ieee-mode division has a divisor box
  reaching zero. The boundary-aware branch also raised
  `IntervalError("NaN endpoint")` on `[-inf,-inf] / [-inf, 0]`; returning
  ⊤ before any endpoint arithmetic removes that too. Real-mode
  `boundary_div` is a sound kernel over `b ≠ 0` and is not wrong for this
  reason — ℝ has one zero and `a/0` is undefined there — but *reaching*
  it needs a premise the box does not carry, which is the next entry.

- **A real-mode divisor box that reaches zero declines unless a strict
  `assume` excludes the zero** (audit 0.2.0 B5-1; see
  [SOUNDNESS.md](SOUNDNESS.md)). **FALSE VERIFIED, real mode, made
  reachable by the M16 fix below.** With `mul` exact, `Σxᵢ²` floors at
  exactly `0`, so `Σxᵢ² − c` turned from a TRUE STRADDLE (which declines)
  into a ONE-SIDED BOUNDARY — and the one-sided arm was the only one of
  `div`'s four zero-containing shapes that did not decline. It called
  `boundary_div`, which drops `b = 0` from the image, and nothing in the
  verdict disclosed the drop. Measured: `x` declared `[0, 2]²`,
  `1/(jnp.sum(x*x) − 8.0)` boxed to `(-inf, -0.125]` and DISCHARGED
  `q <= -0.125`, while jax at `x = [2, 2]` — a point of the declared box
  — returns `+inf`. The three sibling shapes (`[0,0]`, a true straddle,
  a negative `sqrt` domain) all decline citing the same fact, that ℝ has
  no value there; this one minted a definite verdict from the rest of the
  box. Verdicts move **VERIFIED/REFUTED → UNKNOWN** wherever a real-mode
  division's divisor box reaches zero with no strict assume excluding it.
  See "Boundary-aware division" above for what now licenses the
  tightening and what carries the licence.

- **`boundary_div` answers `inf/inf` instead of raising** (audit 0.2.0
  B5-3). The claim recorded for the S10 fix — "returning ⊤ before any
  endpoint arithmetic removes the `NaN endpoint` raise too" — was true of
  `ieee_div` and false of the real-mode sibling, which was never touched:
  `_boundary_div_lo`/`_hi` fall to `_down(num/den)` on an infinite
  operand, and `inf/inf` is NaN. `boundary_div([inf,inf], [0,inf])`
  raised `IntervalError("NaN endpoint in interval arithmetic")` — caught
  by the dispatcher, so nothing crashed, but the domain's internal
  invariant string was printed as the user-facing reason `div` declined.
  `div`'s own `inf/inf` guard now runs first in both of `boundary_div`'s
  arms; 8 box pairs in the endpoint sweep raised before, 0 after.

- **`mul` is exact when its corner products are representable** (audit
  0.2.0 M16): it was the only arithmetic transfer with no exact-rational
  path, bumping every endpoint outward unconditionally. `[2,3]×[2,3]`
  boxed to `[3.9999999999999996, 9.000000000000002]` for an image that is
  exactly `[4, 9]`, and the exactly-zero corner of `[0,4]×[0,4]` bumped to
  `-5e-324` — below zero, which defeats `reduce_sum`'s nonnegative clamp.
  A sum of squares written `x*x` therefore became a true straddle and the
  division consuming it declined, while `x**2` and `jnp.square(x)`
  verified: one real property, three spellings, two verdicts — on exactly
  the `assume(x > 0)` sum-of-squares shape boundary-aware division was
  added for. Sound in both directions (the weak spelling only lost
  precision), so no verdict was wrong; verdicts move **UNKNOWN →
  VERIFIED/REFUTED** where the lost ulp was what prevented a decision.
  `mul` now takes the same `_exactable`/`Fraction` route `add` and `div`
  already had, confined the same way (an infinite endpoint keeps the bump,
  because `Fraction(inf)` raises and `0·±inf = 0` is an endpoint
  convention). The ieee `mul` kernels deliberately do NOT change: under
  ieee the value IS `fl(x*y)`, which the native corner products already
  compute exactly.

  **`dot_general` follows the same rule, because it now IS the same rule**
  (audit 0.2.0 B5-2). It carried an inlined COPY of `mul`'s four corners
  and M16 converted only the original, so `jnp.sum(x*x)` floored at
  exactly 0 while `jnp.dot(x, x)` floored at `-1e-323` — the M16 defect,
  one level up, in the second copy. Both call `interval._mul_corners` now.
  Measured over `x in [0,4]²`: the contraction returns `(0.0, 32.0)`,
  identical to the reduction, where it returned
  `(-1e-323, 32.00000000000001)`; a `[2,3]`-valued 2×2 matmul returns the
  exact `[8, 18]` where it returned `[7.999999999999999,
  18.000000000000004]`. Verdicts move **UNKNOWN → VERIFIED/REFUTED**,
  never the other way. Only the product corners changed: the accumulation
  already used `_add_lo`/`_add_hi`, and the association-order argument the
  contraction rests on is untouched by this and always was.

- **Relational assumes forwarded to solver**: when `assume(e1 < e2)`
  involves two variable operands (a constraint the interval domain cannot
  apply), the comparison is recorded and emitted as a positive axiom
  alongside the negated obligation. The solver sees the full constraint
  set.

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
  intended. The per-obligation withholding now un-withholds a violation
  ONLY when ALL relational assumes were actually emitted for that specific
  obligation's script — a genuine violation from the constrained domain.

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
  is exact; the real-mode overapproximation is sound. In real mode, the
  strict-sign certificate — not the box — is what lets boundary-aware
  division use the resulting `[0, hi]`.
- **The strict-sign certificate is dropped by every primitive without an
  explicit rule**, and by every `sub`. So `assume(x > 0); 1/(Σxᵢ² − c)`
  declines even where `c` makes the divisor genuinely nonzero, and
  `assume(x > 0); y = jnp.sqrt(x); 1/jnp.sum(y*y)` declines because
  `sqrt` has no rule (both measured). Sound in that direction (a dropped
  fact can only turn a
  VERIFIED into an UNKNOWN) and extending it is a rule-per-primitive job,
  each rule a soundness claim of its own. It is also whole-array
  granularity — "every element of this value is certainly positive" —
  rather than per-element, so a mixed-sign array carries nothing even
  where some elements are certified.
- **The certificate does not cross a sub-jaxpr boundary.** A transparent
  wrapper (`remat`, `custom_jvp`) or a `cond` branch runs with a fresh
  table, so a division inside one of them sees no certificate from its
  caller and the cond's outputs carry none back. Conservative, and it is
  what keeps a branch-local assume from licensing anything outside its
  branch.
- **The interval domain cannot represent the sign of an IEEE zero**, so
  under `semantics="ieee"` every divisor box that reaches zero divides to
  ⊤ — including the one-sided shapes real mode tightens, and including
  the ones the subnormal haze creates by hulling a strictly-signed
  interval with `0.0`. Closing this needs a signed-zero lattice threaded
  through every kernel that can produce or consume one, which is a larger
  feature and was deliberately not built here: a half-done version would
  put a trustworthy sign bit on values only some producers set, which is
  the defect S10 already was. Declining to tighten is the sound posture in
  the meantime.
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
