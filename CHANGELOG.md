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

- **An `assert_` nested in a `jit` no longer declines solver escalation for
  every OTHER obligation in the query** (audit 0.2.0 **M17**). Escalation
  slices top-level `stelling_assert` equations, and it used to decide
  whether it could map obligations onto them by COUNTING: unequal totals
  meant nothing could be mapped, so *every* unknown obligation declined.
  One `assert_` written inside a `@jax.jit` helper — or a `cond` branch, or
  a `scan` body — therefore cost escalation for the whole query. This is
  the mechanism behind reports that "several asserts that each pass
  individually come back UNKNOWN together"; it was widely attributed to the
  per-obligation element budget, which was never involved.

  The count check was *sound* (equal totals really did mean index `k` is
  assert `k` — nothing was ever mis-sliced); it was simply a whole-query
  answer to a per-obligation question. The walk now records, per
  obligation, the position of the `stelling_assert` equation it came from
  (`ObligationReport.top_level_eqn_pos`, `None` for anything inside a
  sub-jaxpr), and `slice_unknown_obligations` VERIFIES that record against
  the query — the position must name a `stelling_assert`, carry the same
  `source_info`, and be claimed by exactly one obligation — before slicing
  by it. An obligation failing any of those declines individually with the
  reason quoted. The result is FINER than the count check — it answers per
  obligation what the count answered per query — and on the wrong-query
  attack it catches strictly more than the count did, but not all of it;
  see the narrowing under **M17′** below.

  **Measured** on a 246-harness / 684-obligation corpus of multi-assert
  queries with jit-nested asserts (jax 0.11.0, `JAX_ENABLE_X64=1`, z3 +
  cvc5 wheels), run before and after: **244 of 584 previously-undecided
  obligations became decided (41.8%)** — 123 discharged and 121
  violated-with-witness — with **0** regressions and **0** disagreements
  against an exact-`Fraction` oracle computed independently of stelling.
  109 of the 208 nested-containing harnesses moved UNKNOWN → REFUTED. The
  38 all-top-level control harnesses were byte-identical.

  **A nested `assert_` is still not sliceable**, and its own obligation
  still declines — with a reason that now names the actual cause instead of
  an arithmetic mismatch. Every obligation still undecided after this fix,
  on the corpus above, is a nested one.

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

- **`dot_general` shape well-formedness is now ONE definition, shared by
  the interval transfer and the SMT emission** (audit 0.2.0 **S12**;
  reaches the released **0.1.0** through `ir.ClosedJaxpr.from_dict` — see
  [SOUNDNESS.md](SOUNDNESS.md)). `interval.dot_general` checked contracted-
  and batch-extent agreement and raised; `obligation._dot_general_plan`
  re-derived the same geometry from the **LHS alone**. On `lhs=(2,) @
  rhs=(4,)` the transfer refused the equation while the emission returned a
  two-term linear combination over a four-element constant operand —
  **dropped addends, with no decline** — and because a refused transfer
  binds ⊤, and ⊤ leaves the obligation `unknown`, the truncating plan is
  exactly what solver escalation then ran. Measured: the four-term sum lies
  in `[4, 8]` and `<= 4.5` does not hold; the truncated two-term sum lies in
  `[2, 4]` and it does — a **false VERIFIED**. On `lhs=(4,) @ rhs=(2,)` the
  same loop indexed off the end of the constant operand and raised a raw
  `IndexError` out of `slice_obligation`, whose caller catches only
  declines.

  Neither face owns a shape predicate now: both call
  `interval.dot_general_geometry`, which is the single definition of dim
  ranges, duplicate dims, list pairing, extent agreement, and the derived
  output shape and contraction ranges.
  `tests/test_dot_general_both_faces.py` asserts the two faces AGREE over
  a well-formed and a malformed half — agreement, not "the emission
  declines these forms", because two copies of a predicate that happen to
  match is the arrangement that produced the defect.

  **No traced query is affected**: jax refuses to trace the equation
  (`dot_general requires contracting dimensions to have the same shape`).
  **No well-formed query changes verdict**: the oracle refuses exactly what
  the transfer already refused. `from_dict` still accepts the document, by
  decision — `ir.py` scopes per-primitive shape inference out of the door in
  writing, and a rule there would leave the two faces free to disagree on
  any hand-built query.

  Also in this fix: `slice_obligation` can no longer raise. An unexpected
  exception becomes a quoted `internal error` decline (UNKNOWN), the same
  posture `solvers.escalate` already takes around `_dispatch_obligation`,
  and its range test is two-sided, so an index past the start of the assert
  list declines instead of raising `IndexError`.

  **The claim "the two faces cannot hold different opinions about whether
  an equation is admissible" was too strong, and the residue was a live
  soundness defect — see the next entry.**

- **The emission may not model a DIFFERENT ARRAY than the propagation did:
  one shape per value, checked for every primitive at once** (audit 0.2.0
  **S12′**; reaches the released **0.1.0** through
  `ir.ClosedJaxpr.from_dict` — see [SOUNDNESS.md](SOUNDNESS.md)). The S12
  fix above gave `dot_general` a shape oracle both faces call. **The oracle
  is shared; its ARGUMENTS are not**: `interval.dot_general` asks it about
  the shapes of the propagated BOXES, `obligation._dot_general_plan` asks
  the same function about the shapes recorded on the equation's INVAR
  AVALS. Leave the declaration and the constant operand alone, edit only
  those avals — which `from_dict` accepts — and the two faces disagree
  again, in the asserting direction.

  Worse than S12's own presentation, and this is what makes it hard to
  recognise: there the transfer REFUSED and left a ⊤ in the coverage
  record. Here it does not refuse. It agrees the contraction has four
  terms, prints the box `[4, 8]`, and the verdict comes back **VERIFIED at
  `4 eqns: 4 known (100%)`** on the claim `Σ <= 4.5`, whose truth in exact
  rationals is `8 <= 9/2` — false. The same lie also mints a **false
  REFUTED**, at a point where the predicate is true, carrying the sentence
  *"confirmed by independent exact-rational replay"* — honest about the
  arithmetic and false about the plan, because replay re-derives the same
  truncated plan. A witness is independent of the SOLVER, never of the
  plan, and that distinction is now stated where the claim is made.

  **It is a class, not a row**: `reduce_sum` truncates identically through
  `_group_reduce_sum`, and two further shapes reach it from inside a
  `jax.jit` body. So the fix is not a third `dot_general` shape rule but
  one cross-check in the slicer — `_Slicer._one_shape_per_value`, over
  every equation of every slice before any plan is built: **no equation may
  be modelled at a shape that disagrees with the shape the value actually
  has.** Two witnesses to "actually has", complementary because each is
  blind where the other sees: the value's BINDING SITE (needs no
  propagation, so it reaches inside transparent call bodies, where no
  interval environment holds a box at all) and the PROPAGATED BOX (the one
  witness a consistently-applied lie cannot forge, blind outside the top
  level). An operand the slicer cannot bind at all declines rather than
  passing.

  **And "the shape the value actually has" means the shape the EMISSION
  MINTS TERMS FROM.** The first spelling of the binding witness read the
  producing equation's outvar aval, which is the record of the binding for
  every producer but one: `slice` mints one SMT constant per element of a
  `stelling_any`'s **`shape` param**, never per element of that outvar's
  aval. A declaration saying four elements in its param and two in its aval
  therefore minted four symbols, summed the two the reference asked for,
  and came back `discharged` on `8 <= 4.5` — inside a `jit` body, where the
  box witness is blind by construction. The three sites in the emission
  path that need a declaration's element count — the budget, the
  input-term construction and the check — all call
  `_Slicer._declared_shape`, so none can implement a different rule from
  the others. **That is not sole readership and it is not a single read**:
  `propagate._declared_element_count` reads the outvar aval for the
  certificate search's cap (sound — the cap only gates whether the search
  runs, and the search re-derives its witness honestly), and each call
  re-reads the param, an object that answers differently between calls
  being caught by `ClosedJaxpr.content_hash()` rather than here. Both
  claims were made in the first spelling and both are struck.
  `ir._validate_decl_eqn` was closed alongside it: it compared a
  declaration's two self-descriptions only `if isinstance(shape, tuple)`,
  so a `list` skipped it entirely, and it now compares the extents whatever
  holds them and refuses a `shape` param it cannot read at all rather than
  passing it. **The door is not the containment** — a declaration with no
  `shape` param at all stays legal, as hand-built IR requires, and the
  slicer closes that form on its own.

  **Cost, measured** over every obligation slice the test suite builds, by
  a stated method: wrap the check, mirror its short-circuits, attribute
  every count to the test file that produced it, run the whole suite, and
  partition on `declines > 0`. The partition lands on exactly two files,
  and both hand the check malformed IR on purpose. Over the
  well-formed remainder (10,503 equations; 13,286 operand references, all
  13,286 with a binding found; 23,789 atoms, 23,112 of them with a
  propagated box): **zero** disagreements on either witness and **zero**
  declines.

- **`interval.dot_general_geometry` keeps its documented contract on
  non-integer `dimension_numbers`** (audit 0.2.0 **S12″**). A float or
  string dim passed the range test (`0 <= 0.0 < 1` is True) and then
  indexed a tuple with it, raising a raw `TypeError` — out of the public
  `propagate()`, since both consumers catch `IntervalError` and nothing
  else, and, on the emission side, as an *"internal error"* decline. The
  dims now go through `operator.index` first, exactly as `check_shape`
  already does for extents, **and are BOUND to what it returns** — the
  first spelling called it and discarded the result, so the dims were
  validated and never normalised, and an object that is indexable but
  UNHASHABLE (a 0-d `numpy` array) passed the guard and then raised a raw
  `TypeError: unhashable type` inside `len(set(dims))`, out of the public
  `propagate()`, while the emission declined: the same two-faces split one
  type level up. The returned geometry now holds plain `int`s. The crash
  was pre-existing; the docstring asserting it could not happen was not.

- **A guard that refuses a malformed extent can no longer be stopped by
  the extent** (audit 0.2.0 B6 **audit 3**, F1/F2/F3 — three shapes of one
  mistake, at the five `operator.index` sites the batch had touched). Each
  is in the safe direction and none moves a verdict; they are listed
  because "the guard is closed" was said about all five.

  - **A guard must PRODUCE the value it validated, not merely test it.**
    `_Slicer._declared_shape` called `_shape_problem(shape)` — which bound
    `operator.index(d)`, tested it and discarded it — and then RETURNED a
    second read, `tuple(_op_index(d) for d in shape)`. An extent answering
    `4` and then `-1` was validated at 4 and emitted as `(-1,)`, where the
    element budget takes a negative contribution and `range(-1)` mints no
    symbols at all. This is verbatim the defect the entry above fixed in
    `dot_general_geometry`, one module over, in the same batch.
    `ir._load_extent_problem` carried it too, in both its callers: the
    declaration door compared the RAW param objects with `==` after
    validating them through `__index__`, and the array length check re-read
    them with `int(d)`. All three now read once and hand back what they
    read, so every comparison downstream is `int`-to-`int`.
  - **`operator.index` raises whatever `__index__` raises.** Four guards
    caught `TypeError` alone, so a `ValueError` or `OverflowError` from a
    hostile extent left `ir.JaxprEqn(...)`, `ir.Aval(...)`,
    `interval.check_shape`, `interval.dot_general` and the public
    `propagate()` **raw**, while the emission face declined on the same
    object — the S12″ two-faces split, from a guard written as an
    enumeration of the exception types its author expected.
  - **A refusal message may not itself raise.** Two composers interpolated
    an unguarded `{!r}` of the object being refused, so a hostile
    `__repr__` turned a decided decline into *"internal error:
    RuntimeError: repr refuses"*. In `ir._validate_decl_eqn` it was worse
    than that: `_load_check`'s message is an ARGUMENT and is composed on
    the passing path too, so a **well-formed** declaration whose extent
    merely had a refusing `__repr__` raw-crashed the public constructor.
    Every such quote now goes through a placeholder-substituting read
    (`obligation._safely`, `interval._safe_repr`, `ir._safe_repr`).

  Re-measured over the malformed-`dimension_numbers` corpus that entry
  publishes, extended by the family it did not contain and driven through
  the public `interval.dot_general` on all three trees: **31 of 34 raised
  raw on `dee8bc2`, 6 on `d6b6d0b`, 0 on this tree.**

- **A declaration's `shape` param is accepted by a POSITIVE rule** (audit
  0.2.0 B6 audit 3). Both faces refused `str`/`bytes`/`bytearray` by name,
  because `tuple(b"34")` is `(51, 52)` — a pair of plausible extents the
  declaration never said. `memoryview` and `array.array` read the same way
  and were not on the list: the door ACCEPTED a `memoryview` shape param
  and the slicer sliced a four-element declaration off it. Adding two more
  names is "the container type I happened to enumerate", which
  `ir._validate_param_value` is annotated in this same batch as
  condemning, so the rule is stated the other way round: **a declaration
  records its extents in a `tuple` or a `list`** — the only forms
  `ir._decode` builds and the only forms jax's own params carry — and
  anything else declines. The character sequences fall out of it instead
  of being named by it, and so does whichever sequence type is noticed
  next.

- **`slice_unknown_obligations` can no longer raise** (audit 0.2.0
  **M17′**; a regression of the M17 fix above, caught and fixed before
  release). Its association check called `tuple(...)` on a `source_info` it
  had not established was iterable, and both callers (`solvers.escalate`,
  `affine.refine_propagation`) iterate this function **in the `for`
  header**, outside their own per-obligation nets — so on hand-built IR
  carrying a non-tuple there, `escalate` raised `TypeError: 'int' object is
  not iterable` and every obligation's verdict went with it. The comparison
  is now total (a non-frame-list means the association cannot be CHECKED,
  which gets its own decline sentence rather than the useless *"traced at 7
  but records 7"*) and the per-obligation body is netted **per obligation**,
  so a sibling still gets its own answer.

  Four totality claims in that repair were **not total**, and none of them
  moves a verdict — each is a raise where a decline belongs. `_frames`
  tested `isinstance(v, list)`, which a `list` SUBCLASS whose `__iter__`
  raises satisfies; the association net's own handler could raise while
  composing its message (`str(e)` runs the exception's `__str__`; `getattr`
  with a default swallows only `AttributeError`); a decline sentence read
  its claimant count a second time and raised `KeyError` printing it; and
  the docstring's *"the preamble cannot raise on any object"* is replaced
  with the true argument, which is that both callers read the same objects
  first and the residual is named.
  Also narrowed: the per-obligation association is **finer** than the count
  check it replaced, not *strictly stronger*. Two queries traced from the
  same factory carry identical `source_info` at the same position, so all
  three guards pass and the wrong-query slice comes out — as it did under
  the count. The containment is `make_solver_verdict`'s query-hash
  pairing, which is the same defence the count check had — **and that
  pairing binds the ESCALATION to the query and does not bind the
  PROPAGATION**, so it is not containment for a mispaired propagation.
  Measured on this tree, on `main`, and on the released **0.1.0**:
  `make_solver_verdict(query_B, propagation_of_A, escalate(B, p_A))`
  returns **VERIFIED** where `B`'s honest verdict is **REFUTED**, with no
  exception anywhere — `escalate` hashes the `closed` it was handed, so
  the gate sees a matching pair while `B`'s obligations are reported with
  `A`'s statuses. `carries_work=False` — an escalation with no records,
  no notes, no spawns and no stamps — exempts the gate entirely and
  reaches the same false VERIFIED with no solver record at all. The
  identity belongs on the `Propagation`, checked wherever a propagation is
  consumed against a query; that is cross-module work and is scheduled as
  its own change. Until then this is a **disclosed residue, not a closed
  one** — see [SOUNDNESS.md](SOUNDNESS.md) and
  `tests/test_verified_bar.py::test_the_pairing_gate_binds_the_ESCALATION_and_not_the_propagation`.

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

- **ATTRIBUTION FOR THIS BATCH, PUBLISHED — with the census method, so the
  numbers can be re-derived rather than trusted** (audit 0.2.0 B6 audit 3,
  F5). The batch's commit message said *"every code change was reverted
  ALONE and the claiming tests go red"* and shipped **no table**, so the
  claim rested on the author. Re-deriving it moved two of the numbers.

  **CENSUS METHOD.** A raw hunk count is a property of the DIFF, not of the
  change: adjacent edits merge at wider context. So the width is stated.

  1. `git diff -U<W> 96ab47a d6b6d0b -- src/`, counting `@@` markers.
  2. Split that diff into one patch per hunk, each applicable alone with
     `git apply -R`.
  3. Classify each hunk **SEMANTIC** or **PROSE**: revert it alone, parse
     the file with `ast`, strip every docstring, compare `ast.dump`. A
     hunk whose lone revert leaves the docstring-stripped AST identical
     cannot change behaviour — nothing can red on it except a test that
     reads source line numbers.
  4. Run the whole suite once per SEMANTIC hunk, reverted alone.

  ```
  raw hunk census         -U0   -U3 (git's default)
    obligation.py          18    10
    interval.py             4     2
    ir.py                   5     2
    solvers.py              1     1
    TOTAL                  28    15

  at -U3:  SEMANTIC 12   PROSE 3
    PROSE: interval.h1 (the R4 comment), obligation.h8 (the preamble
    docstring), solvers.h1 (the Escalation docstring)
  ```

  So the batch is **15 hunks, 12 of them semantic** — not the 8 an earlier
  summary gave, and not the 10 a later one did.

  **TWO CONFOUNDS, AND BOTH ARE ELIMINABLE BY CONSTRUCTION rather than
  subtractable.** A revert experiment needs a clean tree, and two obvious
  ways to make one are not clean. `cp -a` preserves mtimes, so the copied
  `__pycache__` validates and its `co_filename` still names the ORIGINAL
  tree — `test_undescended_assume.py` compares a traced frame's filename
  against the test module's `__file__`, and reds in the UNREVERTED base.
  `git archive` carries no `.git`, so `test_reuse_pins.py`'s scratchpad
  floor skips ("not a git repository") and
  `test_skip_inventory.py::test_no_session_skip_is_undisclosed` reds on
  the undisclosed skip — again in the unreverted base. `git clone` has
  neither, and is the method.

  One confound genuinely does have to be subtracted:
  `test_supported_primitives_doc.py::test_committed_page_matches_live_registries`
  reds on ANY line-count change in `src/stelling/obligation.py`, because
  `docs/supported-primitives.md` embeds source line numbers, and
  regenerating the page per revert would make the experiment circular.

  **RESULT** — full suite per revert, `JAX_ENABLE_X64=1`, jax 0.11.0,
  `pytest -q -p no:randomly`; NET = raw failures minus that row's base
  confounds:

  ```
  revert (hunks, -U3)                      raw  conf  NET  the tests that red
  R1  _declared_shape family h1+h2+h3+h4     8     1    7  the four below, plus
                                                            ..._DECLINES[bytes]
  R1a _binding_shape dispatch     h2         5     1    4  aval_lie: NO_shape_param
                                                            _binds_at_the_scalar;
                                                            slicer_closes_..._ON_ITS_OWN;
                                                            lie_no_longer_reaches_a_discharge;
                                                            BINDING_witness_alone_closes
  R1b the element-budget reader   h3         3     1    2  ..._DECLINES[str],
                                                            ..._DECLINES[not-iterable]
  R1c the slice-input reader      h4         1     1    0  NOTHING  <-- see below
  R2  TranscriptionError decline  h5         2     1    1  lie_is_refused_when_the_
                                                            descent_re_transcribes_it
  R3  the handler's _safely   h6+h10         3     1    2  net_around_the_association_
                                                            cannot_itself_raise;
                                                            ..._DECLINES[not-iterable] (*)
  R4  _frames list arm            h7         2     1    1  frames_is_total_on_a_list_
                                                            that_will_not_iterate
  R5  claimants read once         h9         2     1    1  the_claimants_count_is_read_ONCE
  R6  ir list recursion       ir.h1          1     0    1  load_walk_recurses_into_LIST_params
  R7  ir _validate_decl_eqn   ir.h2          1     0    1  declaration_check_reads_the_
                                                            EXTENTS_not_the_param_type
  R8  interval _indices    interval.h2       4     0    4  oracle_NORMALISES_its_dims (+3
                                                            0-d-array rows of ..._AND_NOTHING_ELSE)

  PROSE controls (the anti-vacuity half: a prose revert must red nothing)
  P1  interval R4 comment  interval.h1       0     0    0
  P2  preamble docstring          h8         1     1    0
  P3  Escalation docstring solvers.h1        0     0    0

  the unreverted base (a clone at d6b6d0b):  3557 passed, 10 skipped, 0 failed
  ```

  **(*) R3's GROUP IS ONE HUNK SHORT, and the row says so rather than
  banking the extra red.** `_safely` has a THIRD call site, installed by
  `obligation.h2` inside `_binding_shape`, so reverting h6+h10 leaves it
  live: `..._DECLINES[not-iterable]` reds with a bare `NameError: name
  '_safely' is not defined` leaking into the decline reason, which
  measures an inconsistent tree and not the handler's degraded
  composition. R3's one genuine behavioural red is
  `test_the_net_around_the_association_cannot_itself_raise`, where the
  handler raises `RuntimeError` out of `getattr(o, "index", -1)`;
  reverting h6 alone reds that test and
  `test_slice_unknown_obligations_CANNOT_RAISE_from_its_OWN_body`. A
  revert group defined by "which hunks mention this symbol" is not the
  same as "which hunks the symbol needs", and this is what the difference
  costs.

  **And `obligation.h1`'s only OWN attributable red is
  `..._DECLINES[bytes]`** — it appears in the family row and in none of
  h2/h3/h4 individually. Since h1 cannot be reverted alone without
  breaking the tree, that single test is the whole behavioural evidence
  for the `_declared_shape` extraction, visible only through the group.
  Recorded because "seven tests red on the family" reads as seven tests
  red on the extraction, and it is one.

  Two hunks cannot be reverted alone at all and are reported as such
  rather than as measurements: `obligation.h1` removes `_declared_shape`
  while h2/h3/h4 still call it (**607 failed, 2949 passed** — an
  inconsistent tree, not a difference), and `obligation.h6` removes
  `_safely` while h1 and h10 still call it. Both are grouped above for
  that reason and neither is a row.

  **`obligation.h4` — the slice-input reader — REDS NOTHING, and is
  recorded as UNREACHABLE AS A GUARD rather than claimed.** The element
  budget calls `_declared_shape` over the same vids first, so no document
  can reach this call in a state the budget did not already decline: it is
  unreachable *as a difference*. `docs/norms.md` forbids exactly the move
  of asserting coverage by construction, and the batch's blanket "each
  change has a test that reds when reverted alone" was false here. It is
  KEPT and not deleted, because it is not a guard: it is a VALUE read, and
  the value it must produce is the one the budget counted and the one
  `_binding_shape` compared every reference against. An independent read
  there is UNSOUND-1 itself. That no test can tell the two apart today is
  a fact about today's readers agreeing, not a licence to let them
  diverge.

  Note also what P2 shows: a PROSE revert of `obligation.py` reds the
  supported-primitives page and nothing else, which is what makes that
  subtraction a line-count effect rather than a behavioural one.

  **AND AUDIT 3'S OWN FIXES, ATTRIBUTED THE SAME WAY** — by MUTATION,
  which is what `docs/norms.md` prescribes for a one-line guard and what a
  hunk revert degenerates into at this size. Each mutation asserts its own
  anchor before running, so a mutation that lands on nothing is an error
  rather than a green run; each is driven over the whole suite in its own
  clone. The base confound here is
  `test_sdist_contents.py::test_no_untracked_file_anywhere_would_ship`,
  which reds in every row including the control because `git apply` leaves
  a new test file untracked in a clone; with the file `git add`ed it is
  green.

  ```
  mutation                                  raw  conf  NET  the tests that red
  (control: no mutation)                      1     1    0  --
  F1 return the SECOND read in
     _declared_shape                          2     1    1  declared_shape_RETURNS_
                                                              the_extents_it_validated
  F1 compare RAW objects in the ir door,
     and re-read with int(d) for the
     byte-length product                      4     1    3  door_compares_the_extents_
                                                              it_VALIDATED_not_a_second_read;
                                                              byte_length_product_uses_the_
                                                              extents_the_guard_validated;
                                                              a_hostile___repr___cannot_raise_
                                                              out_of_the_public_constructor
  F2 narrow all four handlers back to
     `except TypeError`                      13     1   12  door_refuses_whatever___index__
                                                              _raises x3;
                                                              declaration_reader_... x3;
                                                              check_shape_refuses_... x3;
                                                              oracle_refuses_... x2;
                                                              declaration_refusal_cannot_be_
                                                              stopped_by_a_hostile___repr__
  F3 unguard every quoted repr             4     1    3  the three ..._hostile___repr__
                                                              tests, one per module
  F4 restore "THE ONE READER" and drop
     the named second reader                  2     1    1  declared_shape_is_NOT_the_
                                                              librarys_only_reader_of_an_
                                                              element_count
  F6 remove clause 4's convention from
     SOUNDNESS.md                             2     1    1  the_entrys_clause_4_states_
                                                              the_convention
  F7 remove the blindness paragraph           2     1    1  the_entry_names_the_screens_
                                                              blind_classes
  OPT restore the (str, bytes, bytearray)
     enumeration on both faces                4     1    3  declaration_check_reads_the_
                                                              EXTENTS_not_the_param_type;
                                                              ..._DECLINES[memoryview];
                                                              ..._DECLINES[array.array]
  ```

  F5 is the table above and is pinned by
  `test_ir_screen.py::test_the_batch_ships_an_attribution_table_that_adds_up`,
  driven three ways rather than by a whole-suite mutation: with no table at
  all (the state the finding reports), with one row's arithmetic broken,
  and with the `R1c` row claiming a red it does not have. All three red.

### Inductive step verification

- **`stelling.inductive.check_inductive_step`**: verify that a loop body
  preserves declared bounds in one step. VERIFIED means the invariant
  holds for all iterations by induction. Constructs the harness
  automatically from the body function and declared state bounds.
  Supports scalar and array-shaped state variables (shape specified per
  variable in the bounds declaration).

### Known limitations (0.2.0)

- **An `assert_` inside a sub-jaxpr does not reach the solver.** Solver
  escalation slices top-level `stelling_assert` equations; an `assert_`
  written inside a `jax.jit` helper, a `cond` branch or a `scan`/
  `while_loop` body is judged by interval propagation and then declines
  escalation, with the reason quoted per obligation. Since the M17 fix it
  costs only ITS OWN escalation — its siblings are decided normally — but
  it is still undecided, so a query containing one cannot reach VERIFIED on
  the strength of the solver. Write the `assert_` at the top level of the
  harness. Lifting this is a capability change rather than a repair, and
  the `cond` case is not merely mechanical: a branch assert is
  CONDITIONAL, so slicing it as an unconditional obligation would be
  unsound.
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
