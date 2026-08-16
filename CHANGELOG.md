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

- **float16 and bfloat16 constants are readable** (audit 0.2.0 M12).
  `propagate._STRUCT_FMT` had no entry for float16's `<f2` or bfloat16's
  `<V2`, so every constant in those formats bound ⊤-maybe-NaN and *any*
  harness mentioning a scalar — including the ubiquitous
  `assert_(y > 0.0)` — answered UNKNOWN. Sound, and it made two of the
  four catalogued formats unusable for the ordinary shape of a harness.
  float16 decodes through `struct`'s `e` code (IEEE binary16, exact);
  **bfloat16 needs the aval**, because its dtype `.str` is `<V2` — an
  anonymous 2-byte VOID that every 2-byte structured dtype spells, so the
  byte string alone does not identify the format. The decoder therefore
  takes the aval's dtype NAME and reads `<V2` only under `"bfloat16"`;
  anything else stays ⊤-with-a-note rather than being read as a float.
  Verdicts move **UNKNOWN → VERIFIED/REFUTED** on float16 and bfloat16
  harnesses with constants, in both `real` and `ieee` semantics.

- **A mixed-format comparison gets the WIDEST operand band, never the
  alphabetically-first** (audit 0.2.0 M13). `_ieee_cmp_get_min_normal`
  sorted the operands' float dtypes and took `[0]`, and
  `bfloat16 < float16 < float32 < float64`, so a `{bfloat16, float16}`
  comparison was hazed with bfloat16's `2**-126` where the float16
  operand needs `2**-14` — 112 decades too narrow, and the band is what
  keeps a verdict sound for a flushing target. The rule is now a maximum
  over the operands' formats, which is sound for every one of them
  because the haze HULLS with 0 rather than replacing. Reachable only
  through hand-built or deserialized IR (jax promotes before it
  computes). The *arithmetic* face still declines a mixed equation, and
  the asymmetry is deliberate: an arithmetic result needs a grid to round
  onto, a comparison produces a bool and uses only the band.

- **The two mode-wide IEEE assumption stamps are format-parametric**
  (audit 0.2.0 M14). `IEEE_ENDPOINT_ASSUMPTION` and
  `SUBNORMAL_INDETERMINACY_ASSUMPTION` are binary64 sentences and were
  stamped verbatim on narrow-format verdicts, where both are false: the
  endpoints **were** outward-rounded to the target grid (that is the whole
  of `_ieee_round_box`), and the band applied was the format's, not
  `2**-1022`. The `semantics:` line disclosed the parametric mode
  correctly, so the two `assumes:` lines contradicted the line above them.
  Both sentences now name the formats the query contains and their own
  bands; a binary64-only run stamps the identical text it always did.
  Disclosure only — no verdict moves.

- **A binary IEEE kernel with no format-parametric row declines** (audit
  0.2.0 M15). `_ieee_arith`'s fallback used the binary64 kernel — whose
  haze band is `2**-1022` — for a narrow format, and `_ieee_round_box`
  afterwards **cannot** recover the missing haze: outward rounding onto
  the format grid does not hull with 0. Measured, float32 `x + x` at
  `x = 2**-140` came back `[1.4349e-42, 1.4349e-42]` where jax computes
  `0.0`. Dead today, and the hazard was that the fifth binary kernel
  registered without a `_FMT_BINARY_OPS` row would be a silent
  regression: `_FMT_BINARY_OPS` and `IEEE_TRANSFERS` are two hand-written
  lists that must agree, the coupling `affine.py`'s `AFFINE_SUPPORTED`
  already names as load-bearing. An import-time census now refuses the
  import when they disagree in either direction, and the runtime arm
  declines as a second guard.

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

- **The VERIFIED bar's re-derivation is given the query's forwarded
  relational assumes** (audit 0.2.0 **M10**; a verdict flip, `UNKNOWN` →
  `VERIFIED` — see [SOUNDNESS.md](SOUNDNESS.md)). `verdict._bar_scope`
  narrows the bar to the decided obligations' own slices only when a recorded
  invocation reproduces both the slice's fingerprint and the script it emits.
  It re-derived the slice without `relational_assumes`, and `smt.emit` takes
  its axioms from `sl.assumes` and from nowhere else — so on any query with a
  forwarded relational assume the re-emitted script was the recorded one minus
  its `(assert …)` axiom lines and an HONEST record failed to be recognised,
  widening the bar to the whole query. The assumes are now re-derived from
  `closed` (never read off an argument: `make_solver_verdict`'s `propagation`
  is not bound to its query by the pairing gate). Conservative before and
  after — the fix can only NARROW — and the residue it does not close, an
  `assert_position` the re-derivation still cannot reproduce, is disclosed at
  `_bar_scope` and in SOUNDNESS.md.

- **The `pow` emission row has a fidelity gauge, and stays out of the
  VERIFIED bar** (audit 0.2.0 **S4**). The row's emission now goes through
  three named seams — `smt._pow_integer_body`, `smt._pow_rational_lines`,
  `smt._pow_aux_name` — extracted behaviour-identically so that a mutation
  battery can express an emitted-`pow` wrongness at all;
  `tests/test_pow_row_gauge_jax.py` runs 21 such mutations across both
  exponent branches with zero survivors. One of them declares a single
  auxiliary constant for two elements of a vectorised `pow`, which is
  well-formed SMT-LIB2, collapses `sqrt(x0) - sqrt(x1)` to zero, and silently
  DISCHARGES an obligation that is false at `x = [4, 1]` — the
  missed-violation direction, caught by one gate and by nothing in the tree
  before. `tests/test_bar_membership_policy.py` carries the decision not to
  bar `pow` or `is_finite`, the reading of the standing rule it rests on, and
  the cost of the alternative measured on two corpora rather than estimated.
  `docs/gauge-coverage.md` states what the gauge reaches and what it does
  not, including `pow`'s integer-dtype guard, which is recorded UNCOVERED
  because no jax program can reach it.

- **That gauge was overfit to the exponents it drove, and now states its
  arity as a MEASUREMENT.** A blinded adversarial audit instrumented the two
  seams and found the shipped battery reaching integer exponents `{-2, 3}`
  and the single pair `(1, 2)` — an unstated SCOPE, since "both exponent
  branches" is true of the branches and says nothing about the exponents. It
  then wrote three wrongnesses conditioned OUTSIDE that set (wrong only above
  degree 3, wrong only for `q >= 4`, wrong only for `p != 1`); all three
  passed every one of the fourteen gates, and each turned a genuinely REFUTED
  query into VERIFIED on an exponent the admission guard admits — `2^6 = 64`
  under a bound of 40, `81^(1/4) = 3` under 2.9, `4^(3/2) = 8` under 7.9.
  Conditional wrongness is the shape a REPAIRED row regresses in, which is
  the shape a one-point-per-branch battery cannot see.

  The battery now drives integer exponents `[-4, -2, 3, 5]` (both signs at
  both magnitude classes, because `_pow_integer_body` reads sign and
  magnitude separately) and the pairs `1/2`, `1/4`, `3/2` (three of the 448
  admitted pairs, because `_pow_rational_lines` reads `p` and `q`
  separately). All three mutations are in it, so the widening is pinned and
  not merely applied once. **The arity is derived from the fixture table and
  measured at the seams** —
  `test_the_driven_arity_is_MEASURED_at_the_seams_not_asserted_in_prose`
  instruments both seams, runs every gate against the baseline and fails if
  the reach is not exactly the declared set — so the SCOPE the gauge prints
  cannot go stale the way the prose version did.

- **`docs/gauge-coverage.md`'s coverage figures are recomputed from the
  battery instead of typed.** The page said *"Sixteen of the seventeen are
  caught by more than one gate"* while the table printed directly beneath it
  already showed five single-covered entries; the measurement was **six**.
  Every other cell re-derived exactly — only the summary was wrong, and it
  was wrong in the direction that made the gauge look stronger, since the
  page's own premise is that *"a gauge with one single-covered mutation is
  one edit from a hole"*. Correcting the digit would have left the class
  alone, so the digits are now parsed out of the page and compared against a
  live gauging run by
  `test_the_documented_coverage_figures_are_the_MEASURED_ones`: the battery
  size, survivor and asymmetry counts, the multi/single partition, the
  single-covered set BY NAME, and every gate the mutation table names
  including each `ALONE` exclusivity. The bare gate counts that column
  carried are deleted rather than corrected — five of them were stale too.

- **`pow`'s rational branch no longer has an unreachable arm that reads as
  covered.** The gauge claimed the branch covered "`q` even **and** odd", and
  odd `q` was structurally unreachable: `obligation.pow_exponent_rational` is
  `Fraction` of a binary64, every finite binary64 is a dyadic rational, so in
  lowest terms `q` is a power of two and `q == 1` takes the integer branch.
  Measured: `q` over the whole 448-pair admitted set is exactly
  `{2, 4, 8, 16, 32, 64, 128}`, 0 odd in 500 000 random draws — so
  `smt._pow_rational_lines`' `if q % 2 == 0:` had a dead `else`, presented in
  its docstring as one of two live cases. An untested branch that READS as
  covered is worse than no branch, so the repair is enforcement rather than a
  corrected sentence, at all three altitudes: the DERIVATION refuses to
  return a non-dyadic rational (so a widening cannot happen quietly), the
  ADMISSION guard `rational_pow_problem` DECLINES an odd denominator (a
  decline belongs where an UNKNOWN can be returned), and the EMISSION refuses
  one with the root guard now unconditional (emission can only write a script
  or refuse). `test_the_odd_denominator_branch_is_UNREACHABLE_and_FAILS_CLOSED`
  pins the unreachability and all three refusals on the standard
  `pow`'s integer-dtype guard already gets. No behaviour changes for any
  exponent jax can produce; the emitted text is byte-identical. Three PROSE
  sites went on describing the odd arm as a live case the encoding handles —
  an admission comment in `obligation.py` on a path where odd `q` declines
  several lines earlier, `_negative_base_harness`'s docstring, and a 0.2.0
  regression docstring. All three now say what is actually true there (`q` is
  even, so the encoding has NO solution at a negative base, and the guard
  stops a trivially-unsat negation coming back VERIFIED), and
  `test_no_source_text_presents_the_ODD_q_ARM_as_a_live_case` scans the tree
  for the claim SHAPE rather than for those three sentences.

- **That gauge measured two of its row's three seams, and the SHAPE axis was
  the one left as prose.** The round above instrumented `_pow_integer_body`
  and `_pow_rational_lines` and left out `smt._pow_aux_name` — the only seam
  handed the array shape — then wrote "any other array shape" into the
  not-reached list. A blinded audit conditioned a mutation on the ELEMENT
  INDEX (correct at elements 0 and 1, the whole of what the two-element
  vector fixture drives; wrong from element 2 on, where it emits
  `aux^6 = x^1` for `x^(1/4)`). It passed all twenty-one gates and minted a
  false VERIFIED: on `x[2]**0.25 - x[1]**0.25 <= 1.9` over `[1, 81]` the truth
  is `81^(1/4) = 3` less `1^(1/4) = 1`, which is 2. The measured/asserted line
  ran through the middle of the instrument.

  `_measured_seam_reach()` now instruments all three seams and the
  `(element, n_out)` pairs are compared against a `DRIVEN_AUX_ELEMENTS` read
  off the fixture table, exactly as the exponents are. **What closes the axis
  rather than sampling it is a new gate**, added here as
  `emission-is-invariant-to-the-array-shape` and renamed further down this
  section to `emission-is-invariant-to-every-seam-argument-but-the-EXPONENT` when it
  took on the base-term axis too: `pow` is elementwise, so an
  element's emitted lines cannot legitimately depend on which element it is
  or on how many there are, and the gate asserts that every element's seam
  output at every count in `[1, 2, 3, 4, 5, 6]`, on BOTH branches, is the
  SCALAR output with the symbol names substituted. Emission is text, so a
  shape-conditioned emission wrongness IS a per-element difference in that
  text — which makes the gate evidence about every conditioning function
  inside the range instead of about one more sampled point. It needs no
  solver. The audit's mutation is in the battery and is caught by that gate
  ALONE; delete the gate and `fidelity.gauge` refuses the run with an
  unexplained survivor. The claim is now split honestly in `SCOPE` and on the
  page: element counts `[1, 2, 3, 4, 5, 6]` reach the seam, `[1, 2]` are
  driven end to end to a VERDICT, and every NON-emission stage past two
  elements — transfer, slice, dispatch, replay, verdict — is a sample, said
  to be one.

  The same page and the gauge file now also record what the mechanism cannot
  do: the equality between declared and measured reach is DRIFT protection,
  the anti-vacuity floors are the only part carrying new coverage, and every
  floor is typed at the radius of a mutation someone already wrote. The shape
  invariance is the one argument in this row's gauge that is not
  radius-shaped.

- **And then the radius moved onto the seam's OTHER ARGUMENT: the BASE TERM.**
  `smt._pow_integer_body(term, exp_val)` takes two arguments and the gauge
  measured the joint reach of one of them. Every fixture in the battery raised
  a DECLARED PROGRAM INPUT to a power — measured across all 22 gates, the
  base-term prefix reaching either exponent seam was `x` and nothing else — so
  a wrongness conditioned on "the base is not an input" was correct on every
  fixture the instrument had. A blinded audit conditioned two mutations that
  way, at a DRIVEN exponent, element 0 and `n_out` 1, and both survived all 22
  gates and minted verdict-level false VERIFIEDs on ordinary jax:
  `(x+1)**3 - (x+1) <= 23` over `[1, 2]` (truth 24) and `(x+1)**0.5 <= 8.9`
  over `[0, 80]` (truth 9), REFUTED becoming VERIFIED on both branches.
  Enumerating the axis properly found three more: the base can also be a
  rational `pow`'s own AUXILIARY (`(x**0.5)**3 <= 7.9` over `[1, 4]`, truth
  `4^(3/2) = 8`, likewise REFUTED → VERIFIED) or an inlined NUMERAL. Five
  survivors in total, none of them the disclosed exponent radius. **Real
  programs almost always `pow` a computed quantity.**

  The base's SPELLING is a coordinate of the joint reach now —
  `(element, n_out, base_kind, exp)` and `(element, n_out, base_kind, p, q)` —
  and the invariance gate sweeps the three spellings `smt.emit` writes for a
  symbol (`input`, `intermediate`, `auxiliary`) against a reference shared
  with the shape sweep, so what is closed is the PRODUCT and not a fourth
  marginal. The gate is renamed
  `emission-is-invariant-to-every-seam-argument-but-the-EXPONENT` to say so. The joint
  reach went from 84 and 63 tuples to **252 and 189, still the full product**;
  four battery mutations conditioned on the base kind at a driven exponent pin
  it, each caught by that gate ALONE. `_element_index_of` no longer guesses
  the element index off a trailing `_<digits>` — it parses the emitter's
  naming grammar, which the old regex got wrong for the auxiliary branch's
  single-element spelling (`aux_2` is element 0 of output 2, not element 2).
  A base spelling the grammar does not know is reported as an INCONSISTENCY
  rather than counted, so the NUMERAL case is disclosed in both not-reached
  lists instead of being absorbed into a driven set. No source behaviour
  changes: the seams were already correct, and the whole repair is in the
  instrument.

  **What is meant to end the pattern is not the fourth axis but that the axis
  LIST is no longer written from memory.** A seam is a pure function of its
  arguments, so its arguments are the complete set of things a wrongness in it
  can be conditioned on.
  `test_every_SEAM_ARGUMENT_is_a_gauged_COORDINATE_or_a_named_exemption` reads
  all three seams' signatures with `inspect` and requires every parameter to
  be classified. A future round that adds
  an argument to a seam fails that test instead of shipping an axis nobody
  enumerated. It says nothing about whether a coordinate is swept WIDELY
  enough — the exponent radius is still a finite set of points and the NUMERAL
  base spelling is outside the driven range of `base_kind` — which is the
  half the entry below had to repair.

  **Running the enumeration corrected its own first draft, which is the
  strongest thing that can be said for it.** `_pow_aux_name`'s `out_id` — the
  auxiliary's freshness across two `pow` OUTPUTS rather than across two
  elements — was written down as an undriven residual, on the reasoning that
  no fixture holds two rational `pow`s. Measured, that was already false: this
  round's `auxiliary` probe arm is `(x**0.5)**e`, two rational `pow`s and so
  two `out_id`s. Two distinct values reach the seam across the battery and a
  mutation dropping `out_id` from the name — colliding two outputs'
  auxiliaries while keeping them fresh per element — is CAUGHT. The
  disclosure would have shipped false on the day it was written; `out_id` is
  classified DRIVEN and pinned by
  `emit-rational-aux-collides-across-two-pow-OUTPUTS`.

- **And then the LIST was closed and every RANGE was still open, which is the
  defect the enumeration itself shipped.** A wrongness is conditioned on an
  argument's VALUE, not on its name, so enumerating the parameters closed the
  list and left every parameter's range exactly as open as before. Two entries
  said otherwise. `out_id` was classified DRIVEN under a definition written as
  a UNIVERSAL — *"reaches the seam at more than one value AND a wrongness
  conditioned on it is CAUGHT"* — on a measured reach of `{2, 3}`, of which
  only `2` reaches any verdict-producing gate; what held was the EXISTENTIAL,
  and mutations conditioned on `out_id >= 4`, `>= 5` and `>= 6` survived all 22
  gates. One mints a verdict-level false VERIFIED on three ordinary jax
  operations: `y = (x+1)-1 ; r = y**0.5 ; r[0]-r[1] <= 7.9` over `[1, 81]^2`,
  where the exact truth is `9 - 1 = 8` and the mutated encoding cannot exceed
  `81^(1/4) = 3`. And `aux_name` was classified DERIVED because it is *"a pure
  function of parameters already classified, SO it carries no axis of its
  own"* — the same `so` this batch had already corrected in the monotonicity
  sentence. Derived-ness closes the LIST; the RANGES compose, and a product
  over `element` and `n_out` says nothing about `out_id`.

  The same defect ran one seam over on an argument nobody had called an id:
  both exponent seams read a BASE TERM, and `x0`, `t2`, `aux_2` spell an id
  exactly as `aux_2_0` does. `_base_spelling` keeps the KIND and throws the ID
  away, so three spellings were swept while the ids reaching the seams were
  `{0, 2}` — and `emit-integer-wrong-only-at-a-LATER-BASE-ID` and its rational
  twin survived every gate too, the second minting the same false VERIFIED.

  **The repair is an INVARIANCE over a printed range, because a range is what
  a sample cannot close.** `emission-is-invariant-to-every-seam-argument-but-the-EXPONENT`
  now asserts that the block a seam returns is the REFERENCE block with its
  symbol names substituted, at every id in `[0, 15]`, for every symbol
  spelling, at every driven shape and every driven exponent — one reference
  shared with the emitted half, so it is one claim and not two marginals. The
  FRESHNESS claim needs its own gate, and that is the finding rather than an
  economy: the invariance comparison canonicalises the auxiliary's name to
  `AUX`, so a wrongness that changes only the NAME is invariant there by
  construction. `every-auxiliary-is-declared-ONCE-and-named-FRESHLY` asserts
  INJECTIVITY of the naming seam over the same range — two elements of one
  output never share a name, two outputs never share one — and, end to end,
  that every `declare-const` in an emitted probe script is declared once.
  Writing it corrected a sentence in the process: `_pow_aux_name` is NOT
  injective over its own signature, since `aux_{out_id}_{element}` does not
  spell `n_out` and `(0, 0, 2)` and `(0, 0, 3)` both mint `aux_0_0`. Those
  cannot co-occur — one output has one element count — so the claim asserted
  is the one the row actually makes.

  **And the vocabulary now forces the choice, which is what stops this
  happening a fifth time.** Every seam parameter carries a CLOSURE as well as
  an axis — SWEPT (every value in its range is driven; only `element` is one),
  INVARIANT (a named gate asserts the text is independent of it over a
  declared range) or DISCLOSED (a finite set of driven points, with the rest
  named in `SCOPE`) — plus a RESIDUE naming the part of its range that is not
  closed, whose phrase has to appear in `SCOPE`. DISCLOSED was EMPTY for a
  round and its emptiness was the tell: a vocabulary with nowhere to say
  *"listed, but not closed"* had nowhere to put the exponent, which is the most
  disclosed thing in the file. `exp_val`, `p` and `q` are in it now, and a
  DERIVED argument is required to carry its sources' residues — the `so` above,
  made mechanical. Four battery mutations pin the id range, a fifth test drives
  the whole `>= 3/4/5/6` family through `fidelity.gauge`, and the measured id
  reach is asserted to be a PROPER subset of the closed range so that a sweep
  which began recording itself would fail rather than print its own range as
  the reach. What is NOT closed is stated: an id is unbounded, `[0, 15]` is
  finite, and a wrongness conditioned past the top is disclosed in both
  not-reached lists. No source behaviour changes; the whole repair is in the
  instrument.

  **Carried as a disclosed follow-up, measured harmless today**: the integer
  branch's `n_out` RECOVERY can silently OVERSTATE. Two separate scalar `pow`
  eqns emitted consecutively present as elements 0 and 1 of one two-element
  eqn — not out of order, not gappy — so nothing fires and the recorder books
  an `n_out` of 2 where the truth is two eqns of 1. Checked against jaxpr
  ground truth on this tree: nothing overstated, nothing understated, because
  no gate holds two consecutive scalar `pow`s. The docstring stated that
  guarantee as an INVARIANT and it is a coincidence of the gate set; the honest
  fix keys the run on the eqn, and it is recorded in both the recorder's
  docstring and `docs/gauge-coverage.md` rather than taken here.

- **Two documented claims about this gauge were out of date in its own file.**
  `test_the_documented_coverage_figures_are_the_MEASURED_ones` read
  `reach.shapes`, `.integer` and `.rational` and asserted "the FULL PRODUCT"
  without the `assert not reach.inconsistent` guard its sibling has —
  measured: perturbing `smt.emit` so the recorder's LIFO mis-pairs left this
  test green, printing 63 of 63 off 54 inconsistent entries. The guard is
  there now — **and it is the WHOLE of the defence, which this entry first
  described as one omission among several figures.** Under that same
  perturbation the joint reach still counts its tuples and still equals the
  declared FULL PRODUCT, so the equality the page argues from does not notice a
  broken measurement at all; `assert not reach.inconsistent` is the only thing
  in either test that does. A product equality is evidence about the reach only
  once the reach is known to be a reading. And the bar-independence paragraph
  counted only the nine
  pre-existing tests elsewhere in the suite: measured with `pow` injected into
  `verdict.VERIFIED_BARRED_PRIMITIVES`, the BATTERY does read identically (32
  mutations, 0 survivors either way, every catch set unchanged, baseline
  passes every gate) but **10 demonstration assertions in the gauge file
  itself go red**, because their per-item verdict lines read `check().status`
  — which is the point of them. 21 red in total: 9 pre-existing elsewhere, 2
  detectors in `tests/test_bar_membership_policy.py`, 10 here. **That battery
  size was written `27` against a tree holding 28**: the run was taken one
  mutation early, the substance re-measured true at 28 — 0 survivors and
  identical catch sets both ways — and only the digit was stale, in the two
  files whose whole argument is that a documented count should be written by
  the tree and not by an author. Both copies are machine-checked now, by
  `test_the_DOCSTRING_and_CHANGELOG_battery_SIZE_is_the_one_that_RAN`.
  Separately, the
  shape probe's docstring said `pow` is monotone on a strictly positive box
  "so" the difference of two elements is symmetric about zero; the `so` does
  not carry. Every element of a declared array carries the SAME interval, so
  `r[n-1] - r[0]` is a SELF-subtraction and symmetric whether or not the row
  is monotone (demonstrated with a non-monotone `x*x` on a straddling box,
  equally undecidable). Strict positivity is load-bearing for DOMAIN
  ADMISSIBILITY instead — a negative integer exponent needs a base excluding
  0, a fractional exponent declines on a negative base.

- **The document test accepted wrong documents and rejected a right one.**
  `test_the_documented_coverage_figures_are_the_MEASURED_ones` required only
  that each gate the table NAMED be a real catcher, so it never required the
  naming to be complete: deleting one gate name from the
  `emit-integer-loses-the-reciprocal` row left the suite green while deleting
  exactly the fact the page's narrative argues from. The bare gate counts
  removed last round had been replaced by hedges (`and most others`, `and
  others`) with measured totals behind them from 4 of 21 to 18 of 21, all
  unchecked; the third column was checked in no respect at all and a
  falsified cell passed; and the single-covered comparison was ordered, so a
  correct alphabetised table was REJECTED. The catch column is now a parsed
  SET compared for equality — complete, prose-free, order-insensitive, with
  `ALONE` derived from the measurement rather than trusted — the
  single-covered rows are compared as a set, and every `A^B = C` anywhere in
  the section is decided exactly in `Fraction` arithmetic. Because a complete
  list runs to nineteen gate names on one row, the column is GENERATED:
  `python tests/test_pow_row_gauge_jax.py --doc-blocks` prints both blocks
  from a live run.

- **A fixture docstring described an emission this tree had just outlawed.**
  `_rat_denominator_false_harness` said the battery's mutation emits
  `aux^5 = x` with a cap of `81^(1/5)`; the mutation adds TWO to the
  denominator (`q + 1` would be an odd `q`, which the same round made
  `_pow_rational_lines` REFUSE, so the item would have been caught by
  malformedness instead of by the denominator it exists to measure), and
  `q = 5` is not emittable at all. `docs/gauge-coverage.md` had it right,
  because the page is machine-checked and a docstring is not. Both are now:
  `test_the_conditional_mutations_CAP_the_value_below_the_bound_RECOMPUTED`
  reads the emitted `(p, q)` off the LIVE mutation, decides
  `cap <= bound < truth` exactly (raising both sides to `q` so no irrational
  root is ever taken as a float), and fails unless the fixture docstring and
  the doc row both quote what it computed.

- **`exp` and `pow` under `semantics="ieee"` now require a DECLARED libm
  accuracy budget** (audit 0.2.0 **S9** and **S11**; S11 reaches the
  released **0.1.0** — see [SOUNDNESS.md](SOUNDNESS.md)). Under `ieee` a
  verdict is a claim about the float value the program computes, and
  stelling's bracket was built around CPython's `math.exp` — the libm of
  the machine running the analysis. The program runs whatever XLA
  compiled. Measured on jax 0.11.0 / jaxlib 0.11.0, CPU, x86_64,
  exhaustively over every `float32` argument whose result is normal and
  finite (2,237,668,967 of them), XLA's `exp` is out by up to **5.51
  float32 ulps** — not faithfully rounded at all, so no fixed widening is
  sound; in binary64 by up to **1.67 ulps** over 3,000,000 samples, which
  is what leaks past a ±1-ulp bracket. On the *same* backend `bfloat16`
  `exp` is exhaustively **correctly rounded** over every normal finite
  result, while `float16` misses correct rounding on 2 of its 63,487
  arguments (0.500028 ulps) — a factor of eleven between two formats of
  one op, so one number cannot be right for all four.

  Both transfers therefore **fail closed** and are re-enabled by a
  declaration:

      check(harness, vacuity_mode="inputs-only", semantics="ieee",
            libm_budget="xla-cpu-2026-08")

  `"xla-cpu-2026-08"` is a shipped, **named and dated** profile of
  per-`(op, format)` budgets; `stelling.propagate.LibmBudget` states your
  own. Both `check()` and `propagate()` take the keyword. The decline
  carries the measurement that justifies it and a line that **runs as
  written**. The budget widens the bracket by the declared ulps before
  the format rounding, and is stamped as **declared, not verified** —
  because a budget smaller than the backend's real error mints a VERIFIED
  stelling cannot catch. A budget of `0.5` ulps (correctly rounded) widens
  by nothing at all, which is `interval.sqrt`'s own argument generalised;
  `sqrt` is a correctly-rounded basic operation, carries no libm demotion,
  and needs no budget. `semantics="real"` is untouched and refuses the
  argument. Verdicts move **VERIFIED → UNKNOWN** and **REFUTED → UNKNOWN**
  on ieee-mode queries containing `exp` or `pow`; the coverage cost,
  measured at a point argument, is **12 float32 / 6 binary64 / 2 float16
  extra grid steps** of bracket width, and zero for `bfloat16`, the one
  format this backend's `exp` is exhaustively correctly rounded in.

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

- **SOUNDNESS FIX — a discharge is no longer accepted when an EMPTY assumed
  region alone explains it.** See the SOUNDNESS.md entry. A relational
  `assume` is inert in the interval domain, so the empty-declared-set oracle
  (`UnsatisfiableAssumptionError`) never saw it — that oracle meets a box
  with a half-space. Since 0.2.0 the same assume is emitted to the solver as
  a positive axiom, and an unsatisfiable axiom set makes
  `boxes ∧ axioms ∧ ¬P` unsat for every `P`: every obligation discharged and
  the verdict was VERIFIED. Measured: `dt ∈ [5, 10]`, `dt_max ∈ [0, 1]`,
  `assume(dt < dt_max)`, `assert_(dt + dt_max <= 1.0)` — VERIFIED, and
  REFUTED with the assume deleted (audit 0.2.0 S7). The non-relational form
  of the identical mistake has always been refused; this closes the route
  around that refusal. Development-only; no released version is affected —
  at `v0.1.0` no assume reaches the solver at all.

  What changed, user-visible:

  * **`check()` and `check_inductive_step()` now raise
    `stelling.propagate.UnsatisfiableAssumptionError` when a forwarded
    relational assume set admits no point of the declared set** *and one
    obligation's script states the whole contradiction*. Same class, same
    closing sentence ("harness defect; nothing was verified"), as the
    non-relational refusal. `check()` already documents that class among the
    two it does not convert to a status. A contradiction spread across
    obligation cones — `assume(x<y); assume(y<z); assume(z<x)` with an
    assert depending on two of the three — cannot be refused, because no
    script ever holds more than one link of it; it is DISCLOSED instead (see
    two bullets down, and audit B3 in SOUNDNESS.md).
  * Before crediting an `unsat`, the backend that produced it is asked one
    more question — the same script with the negated obligation removed
    (`stelling.smt.emit(..., states_obligation=False)`) — and only on an
    obligation that discharged with at least one forwarded axiom on its
    script. **Zero extra solver calls on a query with no relational
    assume**, and none when the propagation's own non-emptiness certificate
    (`Propagation.region_inhabited`) already settled the question. Measured
    on the 288-harness sweep, where every harness carries a relational
    assume: 324 admitted-region invocations out of 1044 total, +11% wall.
  * An admitted-region check that does not settle the question does not
    withdraw the discharge; it stamps it. The obligation detail gains
    `[MAY BE VACUOUS: …]` and the stamp gains an `assumes:` line beginning
    `precondition satisfiability uncertified` — the may-be-vacuous line
    SOUNDNESS.md's constraining-assume policy already required and this path
    did not emit. **Two ways not to settle it, both stamped, each naming its
    mechanism on the obligation**: nobody answered, or the answer was `sat`
    over an axiom set that is not the whole query's (audit B3 — a model of a
    relaxation of your precondition is not a point of your precondition).
  * **A forwarded relational axiom now stamps its conditionality.** New
    `assumes:` line `forwarded relational assume(s) on obligation(s) …`,
    carrying the same `the verdict holds where the precondition holds`
    phrase an interval narrowing has always carried. It names the
    obligations it reaches, and the two readers of that phrase —
    `Verdict.render`'s conditional REFUTED wording and the inductive-step
    note — read the SCOPE (audit B3): a whole-query narrowing line qualifies
    every obligation, a forwarded line only the ones it names. Before that,
    a forwarded axiom on one obligation made an unrelated interval
    refutation render as "conditional … judged over the propagated superset
    of the precondition-narrowed set" and an unconditional inductive step
    render as "CONDITIONAL — NOT the inductive step".
  * The `vacuity checked …` line appends `WHAT THIS MEASUREMENT DOES NOT
    SAY: …` whenever the stamp carries any `precondition satisfiability
    uncertified` line: widening a bound can make an unsatisfiable
    precondition satisfiable again, so a re-check that fails to re-derive an
    obligation is not, there, evidence that the VERIFIED is substantive.
  * `stelling.solvers.Escalation` gained `region_uncertified` and
    `conditional_on_assumes` (obligation indices). Neither decides a
    verdict; both feed the stamp.
  * **`check_inductive_step`: an `assume` in the body no longer gets the
    unconditional note.** An assume is a precondition on the whole query, so
    a VERIFIED means "every state in the ASSUMED SUB-REGION stays in bounds
    after one step" — not the inductive step, because the successor need not
    re-enter that sub-region. The note now begins `inductive step
    CONDITIONAL — NOT the inductive step` and names the fix (put the
    restriction in `state_bounds`); the module docstring and
    `docs/inductive-step.md` say the same (audit 0.2.0 M5). Measured:
    `x -> 1.5x` on `[-1, 1]` under `|x| <= 0.5` is VERIFIED and iterating
    from the admitted `x = 0.4` leaves the invariant at step 3.
  * **`check_inductive_step`'s REFUTED note no longer names the wrong
    variable** when `body` declares its own `assert_` (audit 0.2.0 M4). The
    obligation-to-state-variable map was positional against an index that
    every body obligation shifts; the offset is now derived. A REFUTED whose
    violated obligations are all the body's own says so instead of blaming
    the invariant.

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

- **An `assume` inside a `scan` or `while_loop` body is recorded instead of
  ignored** (audit 0.2.0 S13; see [SOUNDNESS.md](SOUNDNESS.md) — **this one
  reaches the released 0.1.0**). The propagation descends the transparent
  wrappers and `cond`; it does not enter a loop body, so a `stelling_assume`
  written in one was never classified — and, the part that made it a
  soundness defect rather than a precision limit, left no record that
  anything had been ignored. Nothing withheld, and a REFUTED came back
  naming a point the user's own precondition excludes. Measured on the
  `v0.1.0` tag and on `main`: `assume(x <= y)` inside a `lax.scan` body with
  `assert_(x - y <= 0.0)` returned REFUTED at `x = 0, y = -1`.

  `propagate._record_undescended_assumes` now reconciles the assume ledger
  against the STATIC set of assume equations the query contains, before
  anything reads the run's assume state, and writes a `dropped` disposition,
  a note naming the construct and the source line, and a stamped
  `precondition satisfiability uncertified` assumption. The same missing
  record reached three rules and all three now see it: the withholding rule
  (**REFUTED → UNKNOWN**, the violation withheld and the reason quoted), the
  admitted-region gate, and `REGION_NOT_ASKED` — which used to skip the
  region question outright whenever no relational axiom was forwarded, on a
  ground that is untrue for an assume that never narrowed anything.

  **The loop is still not descended**, deliberately: a loop body's `assume`
  is a per-iteration statement about a carry this analysis does not model.
  Write the precondition at the top level of the harness to have it
  honoured — see
  [docs/harness-api.md](docs/harness-api.md#an-assume-inside-a-scan-or-while_loop-body-is-not-descended).

  Verdicts move **REFUTED → UNKNOWN** on harnesses of that shape, and a
  discharge there gains a may-be-vacuous line. **This costs correct
  refutations, and the number is not zero.** Measured over a 240-harness
  loop-carrier corpus (`scan`/`while_loop`/`fori_loop`/nested `scan`/
  `scan`-in-`cond`/top control, comparison set `lt`/`le`/`gt`/`ge`, four
  asserts in both directions), scoring every moved row against the pre-fix
  run's own witness in exact `Fraction`: **200 rows move, and 80 of them —
  40 % — were correct refutations carrying correct witnesses**, spread
  evenly over all five loop carriers; 40 more were vacuous, 40 had no
  correct refutation at all, 40 had one with a different witness. The 40
  top-level control rows move 0. (A narrower 144-harness corpus scored 96
  moved rows all false; that partition is a property of ITS `lt`/`le`,
  one-assert-direction pairing — see
  [SOUNDNESS.md](SOUNDNESS.md).) Over the 288-harness
  `jit`/`cond`/`custom_jvp` corpus: 0 verdicts and 0 caveat states move —
  though the tightening is not gated on loops, and two non-loop shapes
  outside that corpus do gain a correct caveat (an assume inside a
  `lax.cond` branch, and `assume(jnp.all(...))` with no control flow).

### Inductive step verification

- **`stelling.inductive.check_inductive_step`**: verify that a loop body
  preserves declared bounds in one step. VERIFIED means the invariant
  holds for all iterations by induction. Constructs the harness
  automatically from the body function and declared state bounds.
  Supports scalar and array-shaped state variables (shape specified per
  variable in the bounds declaration).

### Known limitations (0.2.0)

- **An `assume` inside a `scan` or `while_loop` body is not honoured.** The
  propagation does not enter those bodies, so such an assume narrows
  nothing and is not forwarded to the solver. It is now RECORDED as a
  dropped assumption rather than ignored — the note names the construct and
  the source line, the stamp carries `precondition satisfiability
  uncertified`, and every definite violation is withheld to UNKNOWN — but
  the precondition still does not constrain the analysis. Write it at the
  top level of the harness. Descending the loop is a separate feature: a
  loop body's assume is a per-iteration statement about a carry that
  changes, and this release models neither.
- **The libm accuracy budget is DECLARED, never verified.** stelling
  widens the `exp`/`pow` bracket by the ulps you declare and stamps the
  declaration; it has no way to measure the function your backend
  executes, so a budget smaller than that function's real error mints a
  VERIFIED nothing here can catch. The shipped profile
  `"xla-cpu-2026-08"` is a measurement of **one** jaxlib on **one** device
  class on **one** day, and its name says so; on any other target it is a
  guess with a date on it. There is also no *residual* budget: an
  `(op, format)` pair a budget does not name declines, and stelling never
  extrapolates from one format to another (measured, the same backend
  ranges over 0.50 to 5.51 ulps across the four formats for the same op).
- **`sqrt` under `ieee` still brackets binary64 with a POINT** — no
  outward bump at all — which is sound only because IEEE-754 *requires*
  `sqrt` to be correctly rounded, so `math.sqrt` and the compiled `sqrt`
  must agree bit for bit. That is a standard's guarantee rather than a
  measurement, and it is a genuinely different footing from `exp`/`pow`,
  which IEEE-754 does not constrain at all. A backend that violates it
  (a fast-math build, an approximate reciprocal-sqrt path) is outside
  what this mode can catch, and `sqrt` carries no budget dial to say so.
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
  A nonzero finite CONSTANT does **not** drop it, whether it reaches the
  rules as a literal (a scalar) or as a constvar (an array): `0.5*Σxᵢ²`,
  `2.0*x`, `x/2.0`, the `/n` inside `jnp.mean`, and
  `jnp.sum(jnp.array([1.,2.,3.,4.]) * x*x)` all keep the chain (measured
  VERIFIED). A constant array must be strictly one-signed THROUGHOUT — a
  mixed-sign weight vector really can sum a positive quadratic to zero —
  and a zero element, a non-finite element, or a dtype with no decoder
  still drops it.
- **The certificate does not cross a sub-jaxpr boundary — `jit`
  included.** Any transparent call wrapper, or a `cond` branch, runs with
  a fresh table, so a division inside one of them sees no certificate
  from its caller and the cond's outputs carry none back. The wrappers
  are `stelling.coverage.DEFAULT_TRANSPARENT` = `jit`, `remat2`,
  `custom_jvp_call`, `custom_vjp_call` — and **`jit` is the one that
  matters in practice**: `assume(x > 0); 1/jax.jit(lambda v: jnp.sum(v*v))(x)`
  is UNKNOWN, and so is the same query with the `assume` moved inside the
  `jit` (both measured, 0.2.0). Earlier text here named only `remat` and
  `custom_jvp`, which understated the cost: almost no jax user writes
  those, and almost every jax user writes `jit`. Conservative in the
  sound direction, and it is what keeps a branch-local assume from
  licensing anything outside its branch.
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
