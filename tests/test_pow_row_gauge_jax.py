# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The `pow` row's fidelity gauge — measured discriminating power over BOTH
of its emission branches.

`pow` is the one emission row whose output is not a body but ASSERTED LINES.
An integer exponent expands to a product; a non-integer one declares a fresh
`aux` and asserts `aux^q = x^p`, because SMT-LIB2's real theory has no power
operator to name. That second branch states three separable claims — a fresh
constant per element, the even-`q` root guard, and which side of the equation
carries which exponent — and audit 0.2.0 found three soundness defects in the
row (S1, S2, S3) without any of them being expressible as a mutation, because
the row had no named seam at all. This file is the instrument that would have
expressed them.

**IT IS DELIBERATELY BAR-INDEPENDENT.** Every discharge gate reads the
per-obligation OUTCOME off `stelling.solvers.escalate` rather than the
`check()` STATUS, for the reason `tests/test_scatter_gauge_jax.py`'s
`_set_row_records` gives: if `pow` were ever added to
`verdict.VERIFIED_BARRED_PRIMITIVES`, `check` would report UNKNOWN for a
correctly discharged obligation and every such gate would silently start
measuring the bar instead of the row. Measured: with `pow` in that set the
whole battery below reads identically, and nine PRE-EXISTING tests elsewhere
in the suite go red (eleven counting the two detectors in
`tests/test_bar_membership_policy.py`, which owns that measurement). A bar is not a gauge, and this gauge does not depend on one.

SCOPE — what these gates REACH (CONTRIBUTING.md, "an instrument must declare
its SCOPE"):

* **BOTH exponent branches under `real` semantics, AT A MEASURED AND FINITE
  ARITY.** The INTEGER branch (product expansion, positive and negative
  exponents) and the RATIONAL branch (`aux^q = x^p`, `q` always EVEN — see
  below), each driven through the slice, the fragment stamp, the SMT text,
  the portfolio dispatch, the exact-rational replay, and the end-to-end
  `preconditions.check` verdict.

  **The exponents this battery drives are the exponents its FIXTURES name,
  and that set is neither the admitted set nor open-ended.** It is
  `DRIVEN_INTEGER_EXPONENTS` and `DRIVEN_RATIONAL_PQ` below — derived from
  the fixture table rather than typed into this paragraph, and MEASURED at
  the two seams by
  `test_the_driven_arity_is_MEASURED_at_the_seams_not_asserted_in_prose`,
  which instruments `smt._pow_integer_body` and `smt._pow_rational_lines`,
  runs every gate against the baseline, and fails if the reach is not
  exactly that set. It is written that way because the prose version of it
  went stale immediately: the shipped battery drove **two** integer
  exponents and **one** `(p, q)` pair while claiming both branches, and
  three wrongnesses conditioned outside that set — wrong only above degree
  3, wrong only for `q >= 4`, wrong only for `p != 1` — survived all
  fourteen gates and each minted a false VERIFIED on an admitted exponent.
  All three are in the battery now (`emit-integer-wrong-only-above-degree-
  three`, `emit-rational-wrong-only-at-a-larger-denominator`,
  `emit-rational-wrong-only-at-a-numerator-past-one`) so the widening is
  PINNED and not merely done once. What the widened battery may honestly
  claim is stated at `DRIVEN_INTEGER_EXPONENTS`: four integer exponents of
  both signs and both magnitude classes, three of the 448 admitted `(p, q)`
  pairs. A wrongness conditioned outside that set is still not gauged here,
  and the set being finite is why it is printed rather than described.
* **BOTH faces.** The emission face as above; the transfer face
  (`interval.pow_`'s corner rule) by containment against the values jax
  computes on this target, sampled eagerly AND under `jax.jit` — the oracle
  is the TARGET, never numpy and never stelling.
* **The independent leg**: every witness executed back through the real
  traced jnp program, eagerly and under `jit`.
* **Two admission guards**: the non-dyadic exponent refusal (audit S1 — a
  nearby rational is a DIFFERENT function) and the negative-base refusal
  under a fractional exponent (jax returns NaN there and the Real encoding
  does not model it).
* **An ARRAY-shaped rational `pow`**, two elements, for the per-element
  freshness of the auxiliary constant — and, separately, **the SHAPE axis as a
  MEASUREMENT rather than a sentence.** `_pow_aux_name` is the one seam handed
  the element count, so the `(element, n_out)` pairs reaching it are measured
  across every gate the same way the exponents are, declared at
  `DRIVEN_AUX_ELEMENTS`, and asserted by the same test. What CLOSES that axis
  rather than sampling it is `gate_emission_is_invariant_to_the_array_shape`:
  emission is text, so a shape-conditioned emission wrongness is exactly a
  per-element difference in the emitted text, and that gate asserts every
  element's seam output at every count in `_INVARIANCE_ELEMENT_COUNTS` is the
  SCALAR output with the symbol names substituted. The predecessor drove one
  two-element fixture and said "any array shape past the one item named above"
  is out of scope; a mutation correct at elements 0 and 1 and wrong from
  element 2 on passed all twenty-one gates and minted a false VERIFIED
  (`81^(1/4) - 1 = 2` under a bound of 1.9).
* **THAT INVARIANCE AT EVERY DRIVEN EXPONENT, AND THE AXES MEASURED JOINTLY.**
  The gate above ran ONE hardwired exponent per branch for a round, so what it
  closed was the shape axis AT `(1, 4)` and AT integer exponent 3 while the
  paragraph above said it closed the shape axis. Every element index `>= 2` was
  driven at one `(p, q)`; the other two printed pairs and three of the four
  printed integer exponents reached elements 0 and 1 only. The gate sweeps the
  driven exponent set now — one probe per exponent, shaped so that no bound
  needs tuning per exponent — and `_measured_seam_reach` records the
  `(element, n_out, exponent)` reach JOINTLY, which the arity test asserts is
  the FULL PRODUCT: `DRIVEN_INTEGER_JOINT` and `DRIVEN_RATIONAL_JOINT`. The
  marginals this file prints are projections of that product, not a second
  measurement.

**WHAT THIS MECHANISM CAN AND CANNOT DO, said here because "MEASURED" reads as
"sufficient".** The equality assertions in
`test_the_driven_arity_is_MEASURED_at_the_seams_not_asserted_in_prose` are
DRIFT protection — real, and exactly what failed last round, when a paragraph
outlived the fixtures it described. They are not discovery: an equality between
what the fixtures declare and what the seams see cannot tell you that the
declared set is the right one. The genuinely new coverage in that test lives in
its ANTI-VACUITY floors, and a floor is typed at the radius of the mutations
the last audit happened to write — `|exp| >= 4`, `q >= 4`, `p != 1`, element
index `>= 2`. Mutations conditioned OUTSIDE the floors still survive, are still
disclosed below, and are still not gauged. The one part of this file that is
not radius-shaped ON THE SHAPE AXIS is the INVARIANCE gate, which holds for
every conditioning function on `(element, n_out)` up to a printed element count
rather than at a sampled point — and holds it at every DRIVEN exponent, because
the gate sweeps that set. **That is a claim about the product and it was
previously overstated as a claim about the exponent axis too.** It is not: the
emitted text is legitimately different at a different exponent, so no
invariance is available there, and the exponent axis remains a finite set of
driven points. A gate that closes one axis at one point of another has moved
the radius rather than removed it, which is exactly what the predecessor of
this gate did — see the block at `SHAPE_PROBE_BOX`.

**AND THE THREE AXES ARE MEASURED JOINTLY, because three marginals are
consistent with almost any joint.** `_measured_seam_reach` records
`(element, n_out, exp)` and `(element, n_out, p, q)` as TUPLES and the arity
test asserts each is the FULL PRODUCT of the driven shapes with the driven
exponents. The predecessor asserted the three axes separately, every equality
passed, and every element index `>= 2` was nevertheless driven at `(1, 4)`
alone — so `element >= 2 AND q == 2`, with both coordinates inside the printed
sets, was ungauged and undisclosed.

SCOPE — what these gates DO NOT reach, and are therefore no evidence about:

* **`integer_pow`'s row.** It is a different primitive with its own
  equation, and this battery's seams are `pow`'s own
  (`smt._pow_integer_body`, `smt._pow_rational_lines`,
  `smt._pow_aux_name`) precisely so that a mutation here does not move it.
  The renderer they share, `smt._repeated_product`, is NOT in this battery
  for the same reason — mutating it moves both rows, so a catch would not
  be attributable to either.
* **`ieee` semantics.** `escalate` refuses every ieee propagation whatever
  it contains, and the ieee `pow` transfer rides a declared libm accuracy
  budget whose gauge is `tests/test_libm_budget.py`. Driven here only as a
  decline probe.
* **The two emission CAPS** (`INTEGER_POW_EXPANSION_CAP`,
  `RATIONAL_POW_DEGREE_CAP`). `tests/test_pow_audit_findings.py` pins both
  boundaries and the 448-pair admitted set; re-measuring them here would be
  a second instrument on one claim, and mutating the degree cap upward
  cannot be done safely — a non-dyadic exponent admitted past it renders a
  product with 3.6e16 factors.
* **`pow`'s INTEGER-DTYPE guard**, and it is not gauged because it is
  UNREACHABLE rather than because nobody wrote a gate. `pow` is in
  `obligation._INT_OVERFLOW_EMITTED`, but no jax program can put an integer
  dtype on it: measured on jax 0.11.0, `lax.pow` raises *"pow does not
  accept dtype int32 at position 0"* and `jnp.power(int32, 2.0)` inserts a
  `convert_element_type` to float64 before the `pow`, while
  `jnp.power(int32, 2)` binds `integer_pow` instead. So the entry is
  defence-in-depth, the mutation that removes it changes nothing any
  jax-traced gate can see, and the guard is recorded UNCOVERED rather than
  gauged (docs/norms.md, "Guard coverage is proven by mutation"). The
  unreachability itself is pinned —
  `test_the_integer_dtype_guard_is_UNREACHABLE_through_jax` — so that a jax
  release which admits it fails loudly instead of leaving this paragraph
  stale.
* **The rational branch at an ODD `q`, which no longer exists to reach.**
  This paragraph used to say the branch covered "`q` even and odd" and the
  odd half was never reachable: `obligation.pow_exponent_rational` is
  `Fraction` of a binary64, every finite binary64 is a dyadic rational, so
  in lowest terms `q` is a power of two and `q == 1` takes the integer
  branch. Measured: `q` over the whole 448-pair admitted set is exactly
  `{2, 4, 8, 16, 32, 64, 128}`, 0 odd in 500 000 random draws. An
  untested branch that READS as covered is worse than no branch, so the
  arm is gone: the root guard is now unconditional, admission DECLINES an
  odd denominator and the emission REFUSES one, and
  `test_the_odd_denominator_branch_is_UNREACHABLE_and_FAILS_CLOSED` pins
  all three on the same standard the integer-dtype guard gets.
* **The affine refinement domain**, which does not admit `pow`.
* **Any other primitive's row**; **any rank past 1**; **any element count past
  the top of `_INVARIANCE_ELEMENT_COUNTS`**; and, for every stage that is NOT
  the emission — the transfer, the slice, the solver dispatch, the replay and
  the verdict — **any element count past the top of
  `_VERDICT_ELEMENT_COUNTS`**. The emission's shape dependence is closed by
  invariance over the first range; the other stages are driven at two element
  counts and that is a sample, stated as one.

Positive control: a property genuinely FALSE inside the declared box must come
back REFUTED with a witness that replays exactly and that violates the
predicate when executed through jax. Negative control: the same programs with
the bound moved to the exact extremum — genuinely TRUE, and still
interval-undecidable — must come back `discharged` from `escalate` with no
witness. Without both, "everything caught" and "the instrument is blind" print
the same page.
"""

from __future__ import annotations

import contextlib
import pathlib
import re
from dataclasses import dataclass
from fractions import Fraction
from unittest import mock

import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("z3")
pytest.importorskip("cvc5")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import stelling.obligation as OB  # noqa: E402
import stelling.propagate as P  # noqa: E402
import stelling.smt as SM  # noqa: E402
from stelling import affine as AF  # noqa: E402
from stelling.fidelity import gauge  # noqa: E402
from stelling.harness import any_array, assert_, trace  # noqa: E402
from stelling.preconditions import check  # noqa: E402
from stelling.propagate import interval_env, propagate  # noqa: E402
from stelling.solvers import (  # noqa: E402
    EmissionInfidelityError, SolverConfig, escalate,
)

TIMEOUT_MS = 30_000

# --- THE DRIVEN ARITY, READ OFF THE FIXTURES ---------------------------------
#
# The exponents below are the ones the fixtures pass to `jnp.power`, and every
# fixture takes its exponent from HERE rather than writing a literal, so this
# is the fixture table and not a description of one. The reach they produce at
# the two seams is MEASURED — see
# `test_the_driven_arity_is_MEASURED_at_the_seams_not_asserted_in_prose` — so a
# fixture that drifts, or a gate deleted, moves these sets and fails that test
# rather than leaving a prose claim standing.
#
# WHY FOUR AND THREE RATHER THAN TWO AND ONE. The shipped battery drove
# integer exponents {-2, 3} and the single pair (1, 2), which is a SCOPE it
# never stated. Three mutations conditioned outside it — wrong only for
# |exp| >= 4, wrong only for q >= 4, wrong only for p != 1 — passed all
# fourteen gates and each turned a genuinely REFUTED query into VERIFIED on an
# exponent the shipped guard admits (2^6 = 64 for a bound of 40; 81^(1/4) = 3
# for a bound of 2.9; 4^(3/2) = 8 for a bound of 7.9). The repaired-row
# regression shape is CONDITIONAL wrongness, so a battery at one point per
# branch cannot see it. Each seam now gets both of the parameters it reads:
#
#   `_pow_integer_body` reads the exponent's SIGN (which decides the
#   reciprocal) and its MAGNITUDE (which is the arity). Both signs are now
#   driven at both magnitude classes — {3, -2} were the cheap ones the
#   expansion special-cases nothing for, {5, -4} are past the degree-3 line
#   the surviving mutation was keyed on.
#
#   `_pow_rational_lines` reads p and q separately. q = 2 and q = 4 are two
#   of the seven reachable denominators; p = 1 and p = 3 are the p == 1
#   family and one member outside it. Three of the 448 admitted pairs.
#
# This is FOUR and THREE, not the admitted space, and the SCOPE string says
# so. Finite and printed beats open-ended and asserted.
_INTEGER_FIXTURE_EXPONENTS = (3.0, -2.0, 5.0, -4.0)
_RATIONAL_FIXTURE_EXPONENTS = (0.5, 1.5, 0.25)

INT_EXP, NEG_EXP, INT_HIGH_EXP, NEG_HIGH_EXP = _INTEGER_FIXTURE_EXPONENTS
RAT_EXP, RAT_NUMERATOR_EXP, RAT_DENOMINATOR_EXP = _RATIONAL_FIXTURE_EXPONENTS

DRIVEN_INTEGER_EXPONENTS = tuple(
    sorted(int(e) for e in _INTEGER_FIXTURE_EXPONENTS)
)
DRIVEN_RATIONAL_PQ = tuple(
    sorted(
        (Fraction(e).numerator, Fraction(e).denominator)
        for e in _RATIONAL_FIXTURE_EXPONENTS
    )
)

# --- THE DRIVEN SHAPE, ALSO READ OFF THE FIXTURES ----------------------------
#
# THE EXPONENT AXIS WAS MEASURED AND THE SHAPE AXIS WAS NOT, WHICH IS THE SAME
# DEFECT ONE ROUND LATER. The round that wrote the block above instrumented
# `_pow_integer_body` and `_pow_rational_lines` and left `_pow_aux_name` — the
# ONLY one of the three seams handed the array shape — out of both `DRIVEN_*`
# and `_measured_seam_reach()`. The SCOPE paragraph then said "any other array
# shape" is not driven, which is prose of exactly the kind the rest of this
# file argues must be a measurement; and a mutation conditioned on the ELEMENT
# INDEX (correct at elements 0 and 1, wrong from element 2 on) passed all 21
# gates and minted a false VERIFIED — `81^(1/4) - 1 = 2` under a bound of 1.9.
# The measured/asserted line ran through the middle of the instrument.
#
# TWO COUNTS, BECAUSE THE TWO ARE DIFFERENT EVIDENCE.
#
#   `_VERDICT_ELEMENT_COUNTS` are the element counts driven END TO END —
#   through the transfer, the slice, the solver, the replay and the verdict.
#   They are the scalar fixtures (1) and the vector fixture (2).
#
#   `_INVARIANCE_ELEMENT_COUNTS` are the counts driven through the EMISSION
#   alone, by `gate_emission_is_invariant_to_the_array_shape`. That gate does
#   not sample the shape axis, it CLOSES it: emission is text, so a
#   shape-conditioned emission wrongness IS a per-element difference in the
#   text, and the gate asserts there is none — every element's seam output at
#   every count in the range is the SCALAR output with the symbol names
#   substituted. One more sampled point would have been a floor typed at the
#   radius of the mutation the last audit happened to write; an invariance
#   holds for every conditioning function on the SHAPE inside the printed
#   range, which is the difference between widening and closing.
#
#   AND IT HOLDS AT EVERY DRIVEN EXPONENT, WHICH IS NOT THE SAME SENTENCE. The
#   round that wrote the paragraph above drove the gate at one hardwired
#   exponent per branch, so the invariance was over the shape AT A POINT of the
#   exponent axis while the paragraph read as though the exponent did not enter
#   into it. It does: `element >= 2 AND q == 2` names an element index and a
#   denominator that are both printed as driven and survived all 22 gates. The
#   gate sweeps `_RATIONAL_FIXTURE_EXPONENTS` and `_INTEGER_FIXTURE_EXPONENTS`
#   now, so what it closes is the PRODUCT of this range with those sets — and
#   `DRIVEN_INTEGER_JOINT` / `DRIVEN_RATIONAL_JOINT` below are that product,
#   asserted as an equality against the measured joint reach.
#
# The shape reach is MEASURED at `_pow_aux_name` by `_measured_seam_reach()`
# and compared against `DRIVEN_AUX_ELEMENTS`, so deleting the invariance gate
# collapses the measurement and fails the arity test rather than leaving this
# paragraph standing.
_VERDICT_ELEMENT_COUNTS = (1, 2)
_INVARIANCE_ELEMENT_COUNTS = tuple(range(1, 7))

DRIVEN_ELEMENT_COUNTS = tuple(
    sorted(set(_VERDICT_ELEMENT_COUNTS) | set(_INVARIANCE_ELEMENT_COUNTS))
)
DRIVEN_AUX_ELEMENTS = tuple(
    sorted((i, n) for n in DRIVEN_ELEMENT_COUNTS for i in range(n))
)

# --- THE JOINT REACH: THE PLANE, NOT TWO MARGINALS ---------------------------
#
# THREE MEASURED AXES AND A CLAIM ABOUT THEIR PRODUCT IS THE SAME DEFECT AGAIN,
# ONE DIMENSION UP. The round that added the shape axis measured the exponents
# reaching one seam, the `(p, q)` pairs reaching another and the
# `(element, n_out)` pairs reaching a third, asserted each against its declared
# set — and every one of those assertions passed on a tree where element index
# `>= 2` was driven at `(1, 4)` and NOWHERE ELSE. Three marginals are consistent
# with almost any joint distribution; the SCOPE paragraph then described the
# plane. A wrongness conditioned on `element >= 2 AND q == 2` — both coordinates
# inside the printed sets, so not covered by the disclosed exponent gap — passed
# all 22 gates and minted a false VERIFIED. NO MUTATION WAS NEEDED TO SEE IT:
# recording `(element, n_out, p, q)` jointly shows the hole on its own.
#
# So the reach is recorded as tuples and compared against the FULL PRODUCT
# below. That the product is achievable at all is what the exponent sweep in
# `gate_emission_is_invariant_to_the_array_shape` buys: one probe shape that
# reaches the emission at every driven exponent means every driven exponent can
# be driven at every element count, and the claim is then an equality rather
# than a disclosure. Where it were not achievable the honest form would be to
# print the joint reach and claim exactly it — but it IS achievable here, so
# the assertion is equality with the product and the SCOPE says the product.


def _driven_product(exponent_keys):
    """The full ``(element, n_out) x exponent`` product, as flat tuples."""
    return frozenset(
        (element, n_out) + key
        for element, n_out in DRIVEN_AUX_ELEMENTS
        for key in exponent_keys
    )


DRIVEN_INTEGER_JOINT = _driven_product(
    tuple((e,) for e in DRIVEN_INTEGER_EXPONENTS)
)
DRIVEN_RATIONAL_JOINT = _driven_product(DRIVEN_RATIONAL_PQ)

_DRIVEN_INT_TEXT = ", ".join(str(e) for e in DRIVEN_INTEGER_EXPONENTS)
_DRIVEN_PQ_TEXT = ", ".join(f"{p}/{q}" for p, q in DRIVEN_RATIONAL_PQ)
_DRIVEN_N_TEXT = ", ".join(str(n) for n in DRIVEN_ELEMENT_COUNTS)
_VERDICT_N_TEXT = ", ".join(str(n) for n in sorted(_VERDICT_ELEMENT_COUNTS))

SCOPE = (
    "BOTH exponent branches of the `pow` row under real semantics, at a "
    "MEASURED and finite arity — the integer product expansion at exponents "
    f"[{_DRIVEN_INT_TEXT}] (both signs, both magnitude classes) and the "
    f"rational aux encoding at exactly the pairs [{_DRIVEN_PQ_TEXT}], which "
    "is 3 of the 448 admitted (p, q) pairs and not the admitted space; the "
    "reach is instrumented at the seams rather than claimed here. Through the "
    "slice, the fragment stamp, the SMT text, the portfolio dispatch, the "
    "exact-rational replay and the end-to-end verdict, eager AND under jit; "
    "the interval transfer's corner rule by containment against jax on this "
    "target; the witness executed back through jax; and the non-dyadic-"
    "exponent and negative-base admission guards. THE SHAPE AXIS IS MEASURED "
    f"TOO, at `_pow_aux_name`: element counts [{_DRIVEN_N_TEXT}] reach that "
    f"seam, of which [{_VERDICT_N_TEXT}] are driven end to end to a VERDICT "
    "while the whole range is driven through the EMISSION, where the row's "
    "per-element output is asserted INVARIANT to the shape — so an emission "
    "wrongness conditioned on the element index or the element count is "
    "caught for every conditioning function inside that range and not only at "
    "a sampled point. THE TWO AXES ARE MEASURED JOINTLY AND CLAIMED AS THEIR "
    f"PRODUCT: all {len(DRIVEN_AUX_ELEMENTS)} (element, n_out) shapes are "
    f"driven at every one of the {len(DRIVEN_INTEGER_EXPONENTS)} integer "
    f"exponents ({len(DRIVEN_INTEGER_JOINT)} tuples) and at every one of the "
    f"{len(DRIVEN_RATIONAL_PQ)} (p, q) pairs ({len(DRIVEN_RATIONAL_JOINT)} "
    "tuples), so an emission wrongness conditioned on ANY COMBINATION of an "
    "element index, an element count and a DRIVEN exponent is caught — and "
    "the invariance argument is about the shape axis only, never about the "
    "exponent axis, which stays a finite set of points. Does NOT drive: any "
    "exponent outside the two sets "
    f"above, any element count past {max(DRIVEN_ELEMENT_COUNTS)} or any rank "
    f"past 1, any NON-emission stage past {max(_VERDICT_ELEMENT_COUNTS)} "
    "elements (and the interval-containment probe is SCALAR-shaped, so the "
    "transfer face is driven at one element by that gate and at two only "
    "through the vector fixture's verdict), `integer_pow`'s row, the shared "
    "`_repeated_product` renderer, ieee semantics past a decline probe, "
    "either emission CAP, the INTEGER-DTYPE guard or the ODD-q rational arm "
    "(both unreachable — see the module docstring), or the affine domain."
)


@pytest.fixture(autouse=True, scope="module")
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


# --- the controls, as programs ----------------------------------------------
#
# EVERY ONE OF THESE IS INTERVAL-UNDECIDABLE, and that is asserted below
# rather than assumed. A control the cheap layer settles never runs the stage
# under test and would report "did not fire" about an instrument that never
# reached the row.
#
# The integer-branch pair uses a box that STRADDLES ZERO, which is what makes
# it interval-undecidable for free: `interval.pow_` has a sound corner rule
# only for strictly positive bases, so the transfer declines to ⊤ there. The
# rational-branch and negative-exponent fixtures need a strictly positive box
# (a fractional exponent on a negative base is NaN in jax and the emission
# refuses it), so their undecidability comes from a CORRELATION the interval
# domain cannot see instead.

INT_BOX = (-2.0, 3.0)
INT_TRUE_BOUND = 24.0   # x^3 - x attains exactly 24 at x = 3: TRUE inside
INT_FALSE_BOUND = 23.0  # ... so 23 is violated there: FALSE inside

INT_HIGH_TRUE_BOUND = 240.0   # x^5 - x attains exactly 240 at x = 3
INT_HIGH_FALSE_BOUND = 239.0  # ... so 239 is violated there

NEG_EXP_BOX = (1.0, 2.0)   # strictly positive: a reciprocal needs it
RAT_BOX = (1.0, 4.0)       # sqrt and x^(3/2) both reach a round value here
RAT_FALSE_BOX = (1.0, 16.0)
RAT_DENOMINATOR_BOX = (1.0, 81.0)  # 81^(1/4) = 3 exactly


def _int_value(x):
    return jnp.power(x, INT_EXP) - x


def _int_harness(bound, *, jit=False):
    fn = jax.jit(_int_value) if jit else _int_value

    def h():
        x = any_array((), "float64", INT_BOX)
        return (assert_(fn(x) <= bound),)

    return h


def _int_high_harness(bound):
    """THE SECOND INTEGER MAGNITUDE, and the fixture the surviving
    `|exp| >= 4` mutation dies on. Same straddling box and same shape as
    `_int_harness` — deliberately, so the only difference between them is
    the exponent, and a catch here that is absent there is attributable to
    the MAGNITUDE and to nothing else. `x^5 - x` attains exactly 240 at
    x = 3, so 239 is violated inside the box and 240 is not."""

    def h():
        x = any_array((), "float64", INT_BOX)
        return (assert_(jnp.power(x, INT_HIGH_EXP) - x <= bound),)

    return h


def _neg_exponent_harness(bound):
    """`x**-2 * x * x` is exactly 1 in ℝ, so `<= 1.0` is TRUE and `<= 0.999`
    is FALSE — and the interval domain sees [0.25, 1] x [1, 4] = [0.25, 4],
    which decides neither. The reciprocal is the whole content: drop it and
    the emitted expression is `x^2 * x * x = x^4`, which reaches 16."""

    def h():
        x = any_array((), "float64", NEG_EXP_BOX)
        return (assert_(jnp.power(x, NEG_EXP) * x * x <= bound),)

    return h


def _neg_high_exponent_harness(bound):
    """The RECIPROCAL at the second magnitude: `x**-4 * x * x * x * x` is
    exactly 1 in ℝ. It is not a duplicate of the pair above, and the reason
    is what `_pow_integer_body` reads. That seam reads the exponent's SIGN
    and its MAGNITUDE, and the shipped battery drove sign-negative only at
    magnitude 2 and magnitude past 2 only at sign-positive — so a wrongness
    keyed on the COMBINATION (a reciprocal beyond -3, say) sat in the gap
    between two gates that each looked like they covered it. Interval:
    [1/16, 1] x [1, 16] = [1/16, 16], which decides neither bound."""

    def h():
        x = any_array((), "float64", NEG_EXP_BOX)
        y = jnp.power(x, NEG_HIGH_EXP)
        return (assert_(y * x * x * x * x <= bound),)

    return h


def _rat_upper_harness(bound):
    """`sqrt(x) <= 2` over [1, 4] is TRUE at the bound and FALSE below it.
    Interval-undecidable because `pow_`'s corner rule gives [1, 2] and the
    obligation asks about the endpoint itself... which it would DISCHARGE. So
    the correlation is what makes it escalate: `sqrt(x) * 1 - (x - x)`."""

    def h():
        x = any_array((), "float64", RAT_BOX)
        r = jnp.power(x, RAT_EXP)
        return (assert_(r + (x - x) <= bound),)

    return h


def _rat_lower_harness(bound):
    """The LOWER-bound direction, and it is not a duplicate: it is the only
    shape in which dropping the even-`q` guard changes an answer. Without
    `aux >= 0` the negation of `sqrt(x) >= 1` is satisfied by the NEGATIVE
    root, which the guard exists to exclude and which jax never computes."""

    def h():
        x = any_array((), "float64", RAT_BOX)
        r = jnp.power(x, RAT_EXP)
        return (assert_(r + (x - x) >= bound),)

    return h


def _rat_false_harness():
    """FALSE, and the shape a WRONG EXPONENT hides in: `sqrt(x) <= 3` over
    [1, 16] is violated at x = 16. An emitted `aux^3 = x` instead of
    `aux^2 = x` makes the largest reachable value 16^(1/3) < 3, so the
    violation disappears — a missed violation, which is the direction with
    nothing downstream to catch it."""

    def h():
        x = any_array((), "float64", RAT_FALSE_BOX)
        return (assert_(jnp.power(x, RAT_EXP) <= 3.0),)

    return h


# The two exponents past (1, 2). Each is a SEPARATE parameter of
# `_pow_rational_lines`, and the shipped battery pinned neither: it drove
# p == 1 and q == 2 only, so a repair correct there and wrong anywhere else
# was invisible. Both fixtures come in the two directions the (1, 2) pair
# already had, so the new arity is gauged for missed violations AND for
# false refutations rather than only the first.


def _rat_numerator_false_harness():
    """p = 3. `x^(3/2) <= 7.9` over [1, 4] is violated at x = 4, where the
    exact value is `4^(3/2) = 8`. A mutation that emits `aux^2 = x^1` — the
    p == 1 the shipped fixtures were the whole of — caps the reachable value
    at `4^(1/2)`, so the violation disappears and a genuinely REFUTED query
    comes back VERIFIED. Interval-undecidable because `pow_`'s corner rule
    gives [1, 8+ulp], which straddles the bound.

    The emitted form and the cap are read off the LIVE mutation and the
    inequality decided in exact `Fraction` arithmetic by
    `test_the_conditional_mutations_CAP_the_value_below_the_bound_RECOMPUTED`
    — see `_rat_denominator_false_harness` for why this paragraph is not
    allowed to carry a typed number."""

    def h():
        x = any_array((), "float64", RAT_BOX)
        return (assert_(jnp.power(x, RAT_NUMERATOR_EXP) <= 7.9),)

    return h


def _rat_numerator_harness(bound):
    """p = 3, the TRUE direction, at the exact extremum: `x^(3/2) <= 8` holds
    over [1, 4] and holds AT x = 4. The `+ (x - x)` is the same correlation
    trick `_rat_upper_harness` uses and for the same reason — without it the
    transfer's own box can settle the endpoint and the gate would never
    reach the row."""

    def h():
        x = any_array((), "float64", RAT_BOX)
        r = jnp.power(x, RAT_NUMERATOR_EXP)
        return (assert_(r + (x - x) <= bound),)

    return h


def _rat_denominator_false_harness():
    """q = 4. `x^(1/4) <= 2.9` over [1, 81] is violated at x = 81, where the
    exact value is `81^(1/4) = 3`. The mutation this gate exists for emits
    `aux^6 = x^1` — correct-looking, and wrong only for a denominator the
    shipped fixtures never reached — which caps the reachable value at
    `81^(1/6)`, below the bound, so the violation disappears.

    THE EMITTED FORM AND THE CAP ARE NOT TYPED HERE, THEY ARE CHECKED. This
    docstring said `aux^5` and an `81^(1/5)` cap for a round: the battery's
    mutation adds TWO to the denominator (`q + 1` would be an ODD `q`, which
    the same round made `_pow_rational_lines` refuse, so the item would have
    been caught by malformedness instead of by the denominator), and `q = 5`
    is no longer emittable at all. The doc page had it right and this
    docstring did not, because the page is machine-checked and a docstring is
    not. It is now:
    `test_the_conditional_mutations_CAP_the_value_below_the_bound_RECOMPUTED`
    reads the emitted `(p, q)` off the LIVE mutation, decides `cap <= bound
    < truth` in exact `Fraction` arithmetic, and fails unless this paragraph
    and the doc row both quote what it computed. No decimal appears here for
    the same reason — `81^(1/6)` is irrational, and an approximation of it is
    one more digit nothing recomputes."""

    def h():
        x = any_array((), "float64", RAT_DENOMINATOR_BOX)
        return (assert_(jnp.power(x, RAT_DENOMINATOR_EXP) <= 2.9),)

    return h


def _rat_denominator_harness(bound):
    """q = 4, the TRUE direction, at the exact extremum 3."""

    def h():
        x = any_array((), "float64", RAT_DENOMINATOR_BOX)
        r = jnp.power(x, RAT_DENOMINATOR_EXP)
        return (assert_(r + (x - x) <= bound),)

    return h


def _rat_vector_harness():
    """The ARRAY item: two elements of one rational `pow`, and a claim that
    is only false because they are INDEPENDENT. `sqrt(x0) - sqrt(x1) <= 0.9`
    is violated at x = [4, 1] (2 - 1 = 1). One auxiliary constant shared
    between the elements makes the difference identically 0 and the violation
    vanishes."""

    def h():
        x = any_array((2,), "float64", RAT_BOX)
        r = jnp.power(x, RAT_EXP)
        return (assert_(r[0] - r[1] <= 0.9),)

    return h


# --- the shape probe, ONE shape for EVERY driven exponent --------------------
#
# THE RADIUS MOVED FROM THE ELEMENT AXIS TO THE EXPONENT AXIS, WHERE IT WAS
# INVISIBLE. The predecessor of this probe was two functions, one per branch,
# each hardwired to a SINGLE exponent — `0.25` on the rational side and `3.0`
# on the integer one — because each carried a bound tuned by hand to straddle
# at that exponent (2.9 for `x^(1/4)` over [1, 81]; 0.0 for `x^3 - x` over a
# box straddling zero). So the invariance gate closed the shape axis AT THOSE
# TWO EXPONENTS and the SCOPE string said it closed the shape axis, full stop.
# Measured on that tree: every element index >= 2 was driven at `(1, 4)` only,
# `(1, 2)` and `(3, 2)` — both printed in `DRIVEN_RATIONAL_PQ` — reached
# elements 0 and 1, and the integer exponents -4, -2 and 5 reached element 0.
# A wrongness conditioned on `element >= 2 AND q == 2`, with BOTH values inside
# the printed driven sets, survived all 22 gates and minted a false VERIFIED:
# `81^(1/2) = 9` against `1^(1/2) = 1` is a difference of 8, the mutated
# encoding caps element 2 at `81^(1/4) = 3` so the reachable difference is 2,
# and a bound of 7.9 sits between them. Those numbers are RECOMPUTED by
# `test_the_shape_conditioned_mutations_are_CAUGHT_by_the_INVARIANCE_gate`
# rather than typed here as results. That is worse than the
# exponent radius it replaced, because the exponent radius is DISCLOSED and
# this one was not: a reader who sees element counts and `(p, q)` pairs both
# declared driven concludes their COMBINATION is driven.
#
# The repair is a probe SHAPE that does not need a bound per exponent, so the
# gate can quantify over the whole driven set instead of one point in it.
SHAPE_PROBE_BOX = (1.0, 81.0)  # STRICTLY POSITIVE: a negative exponent needs it
SHAPE_PROBE_BOUND = 0.0


def _shape_probe_harness(exponent, n):
    """ONE driven exponent at ONE element count, for the shape INVARIANCE gate.

    Not a control: it is never asked for a verdict, only for emitted text, so
    what it has to be is EMITTED — the obligation must survive interval
    propagation or there is no slice and nothing to compare.

    THE PREDICATE IS A DIFFERENCE OF TWO ELEMENTS AGAINST ZERO, WHICH IS WHAT
    LETS ONE PROBE SERVE EVERY EXPONENT. `pow` is monotone on a strictly
    positive box, so whatever interval the transfer gives the row's output —
    `[lo^e, hi^e]` (or its reverse, at a negative exponent) under the REAL
    transfer, `[lo, hi]` under `transfer-is-the-base` — the difference
    `r[n-1] - r[0]` lands in an interval SYMMETRIC about zero, and zero is the
    bound. Zero is therefore strictly interior under both transfers, at every
    exponent, at every element count, so the probe reaches the emission either
    way and the invariance gate cannot end up reporting a TRANSFER wrongness as
    a shape one. Nothing here is tuned per exponent, which is the whole point:
    a hand-tuned bound is what limited the predecessor to one exponent per
    branch, and one exponent per branch is where `element >= 2 AND q == 2`
    lived. That undecidability is asserted, not argued, by
    `test_the_shape_probe_reaches_the_EMISSION_at_every_DRIVEN_exponent`.

    The difference is also the correlation the interval domain cannot see, so
    the `+ (x[0] - x[0])` the predecessor carried is gone: the difference IS
    the correlation. At `n == 1` it is `r[0] - r[0]`, which interval arithmetic
    still cannot collapse, so the scalar arm is driven on the same footing.
    """

    def h():
        x = any_array((n,), "float64", SHAPE_PROBE_BOX)
        r = jnp.power(x, exponent)
        return (assert_(r[n - 1] - r[0] <= SHAPE_PROBE_BOUND),)

    return h


def _non_dyadic_harness():
    """AUDIT S1's headline. `0.1` is not `1/10`: the binary64 literal denotes
    `3602879701896397/36028797018963968` exactly, whose auxiliary encoding is
    far over the degree cap, so this must DECLINE. The predecessor rationalised
    to `1/10` and emitted a script about a different function, which verified
    a claim false at the declared upper bound."""

    def h():
        x = any_array((), "float64", (1.0, 1e300))
        return (assert_(jnp.power(x, 0.1) <= 1e30),)

    return h


def _negative_base_harness():
    """A fractional exponent over a box reaching below zero: jax returns NaN
    and the Real encoding does not model that. `q` is EVEN on every path that
    reaches the encoding — the odd arm is refused at admission, see
    `test_the_odd_denominator_branch_is_UNREACHABLE_and_FAILS_CLOSED` — so
    `aux^q = x^p` at a negative base has no real solution, the negated
    obligation is trivially unsat, and a false VERIFIED is what would come
    back. Must decline."""

    def h():
        x = any_array((), "float64", (-1.0, 4.0))
        return (assert_(jnp.power(x, RAT_EXP) <= 2.0),)

    return h


# --- subjects and patching --------------------------------------------------


def _patched(subject):
    """Enter a subject: ``transfers`` entries patch the live transfer registry
    and ``__patches__`` entries (``(module, attr, value)``) patch module
    attributes, so EMISSION- and REPLAY-layer wrongness is expressible and not
    only transfer-layer wrongness."""
    stack = contextlib.ExitStack()
    transfers = subject.get("transfers", {})
    if transfers:
        stack.enter_context(
            mock.patch.dict(
                P.TRANSFERS,
                {p: (fn, P.TRANSFERS[p][1]) for p, fn in transfers.items()},
            )
        )
    for mod, attr, val in subject.get("__patches__", ()):
        stack.enter_context(mock.patch.object(mod, attr, val))
    return stack


@contextlib.contextmanager
def _maybe_linear_fragment(subject):
    """The fragment mutation, as a METHOD patch (``_fragment`` is a
    ``_Slicer`` method, so ``__patches__`` cannot reach it)."""
    if not subject.get("__linear_fragment__"):
        yield
        return
    with mock.patch.object(
        OB._Slicer, "_fragment", lambda self, inputs, eqns: OB.QF_LRA
    ):
        yield


BASELINE: dict = {}


# -- emission mutations, all against `pow`'s OWN seams ------------------------


def _emit_int_off_by_one(term, exp_val):
    return SM._repeated_product(term, abs(exp_val) + 1)


def _emit_int_exponent_ignored(term, exp_val):
    return SM._repeated_product(term, 2)


def _emit_int_no_reciprocal(term, exp_val):
    return SM._repeated_product(term, abs(exp_val))


def _emit_rat_sides_swapped(aux_name, base, p, q):
    lines = [f"(declare-const {aux_name} Real)"]
    if q % 2 == 0:
        lines.append(f"(assert (>= {aux_name} 0.0))")
    lines.append(
        f"(assert (= {SM._repeated_product(aux_name, p)} "
        f"{SM._repeated_product(base, q)}))"
    )
    return lines


def _emit_rat_guard_dropped(aux_name, base, p, q):
    return [
        f"(declare-const {aux_name} Real)",
        f"(assert (= {SM._repeated_product(aux_name, q)} "
        f"{SM._repeated_product(base, p)}))",
    ]


def _emit_rat_never_asserted(aux_name, base, p, q):
    """The constraint EMITTED but never asserted: the line is written as a
    comment, so the script declares `aux` and says nothing about it."""
    lines = [f"(declare-const {aux_name} Real)"]
    if q % 2 == 0:
        lines.append(f"(assert (>= {aux_name} 0.0))")
    lines.append(
        f"; (assert (= {SM._repeated_product(aux_name, q)} "
        f"{SM._repeated_product(base, p)}))"
    )
    return lines


def _emit_rat_denominator_off_by_one(aux_name, base, p, q):
    lines = [f"(declare-const {aux_name} Real)"]
    if q % 2 == 0:
        lines.append(f"(assert (>= {aux_name} 0.0))")
    lines.append(
        f"(assert (= {SM._repeated_product(aux_name, q + 1)} "
        f"{SM._repeated_product(base, p)}))"
    )
    return lines


# -- CONDITIONAL emission mutations -------------------------------------------
#
# The three above are UNIFORM wrongnesses: they are wrong at every exponent, so
# one fixture per branch finds them. These three are wrong only OUTSIDE the
# arity the shipped battery drove, which is the shape a repaired row regresses
# in — a fix applied to the exercised exponent and not to the general one. All
# three passed all fourteen shipped gates and each minted a real false VERIFIED
# on an exponent the admission guard admits, so they are here to PIN the
# widening rather than to leave it a one-time correction.
#
# Each is built as a CLOSURE over the live seam rather than looking it up by
# name inside the patch: `mock.patch.object` has already replaced the attribute
# by the time the mutation runs, so `SM._pow_integer_body(...)` from inside one
# would recurse forever. `_mutations()` is called before any patch is entered,
# which is what makes the captured reference the real one.


def _int_wrong_only_above_degree_three():
    """Wrong only for `|exp| >= 4`, correct at every exponent the shipped
    fixtures drove. Emits `x^5` for `x^6`; on `x**6 <= 40` over [1, 2] that
    turns a REFUTED into a VERIFIED, and the truth is `2^6 = 64`."""
    real = SM._pow_integer_body

    def mutated(term, exp_val):
        if abs(exp_val) >= 4:
            return real(term, exp_val - 1 if exp_val > 0 else exp_val + 1)
        return real(term, exp_val)

    return mutated


def _rat_wrong_only_at_a_larger_denominator():
    """Wrong only for `q >= 4`, correct at the `q == 2` the shipped fixtures
    were the whole of. Emits `aux^6 = x` for `x^(1/4)`; the truth at the
    declared upper bound is `81^(1/4) = 3`, and a bound of 2.9 went VERIFIED.

    ``q + 2`` RATHER THAN ``q + 1``, WHICH IS NOT A DETAIL. The auditor's
    version of this mutation added one, and one is now REFUSED outright: the
    same round made `_pow_rational_lines` raise on an odd denominator, so
    ``q = 5`` would be caught by malformedness and this item would stop
    measuring the thing it is for. A catch by refusal is a catch, but it is
    not evidence that a WELL-FORMED wrongness at a larger denominator would
    be seen — and that is the whole claim. Adding two keeps the emitted
    script legal and the missed violation silent, which is the direction
    with nothing downstream to catch it.
    """
    real = SM._pow_rational_lines

    def mutated(aux_name, base, p, q):
        return real(aux_name, base, p, q + 2 if q >= 4 else q)

    return mutated


def _rat_wrong_only_at_a_numerator_past_one():
    """Wrong only for `p != 1`, correct across the whole `p == 1` family the
    shipped fixtures drove. Emits `aux^2 = x^1` for `x^(3/2)`; the truth at
    the declared upper bound is `4^(3/2) = 8`, and a bound of 7.9 went
    VERIFIED."""
    real = SM._pow_rational_lines

    def mutated(aux_name, base, p, q):
        return real(aux_name, base, 1 if p != 1 else p, q)

    return mutated


_ELEMENT_SUFFIX = re.compile(r"_(\d+)$")


def _element_index_of(base):
    """The element index a base term's name carries, or 0 for the unsuffixed
    single-element spelling. `_pow_rational_lines` is not handed the index —
    that is the seam gap `docs/gauge-coverage.md` names — so an
    element-conditioned mutation has to read it off the term, exactly as
    `_emit_rat_one_declaration_for_two_elements` does."""
    m = _ELEMENT_SUFFIX.search(base)
    return int(m.group(1)) if m else 0


def _rat_wrong_only_past_the_second_element():
    """THE SHAPE AXIS'S CONDITIONAL WRONGNESS, and the fourth mutation to
    survive a battery that had just been widened on the exponent axis.

    Correct at elements 0 and 1 — the whole of what the two-element vector
    fixture drives — and wrong from element 2 on, where it emits `aux^6 = x`
    for `x^(1/4)`. It passed all twenty-one gates of the predecessor and minted
    a real false VERIFIED: on `x[2]**0.25 - x[1]**0.25 <= 1.9` over [1, 81]^3
    the truth is `81^(1/4) - 1^(1/4) = 3 - 1 = 2`, and the mutated encoding
    caps element 2 at `81^(1/6) < 2.09`, so the violation disappears.

    It is in the battery to PIN the shape measurement, the way the three
    exponent-conditioned mutations pin the exponent one. What catches it is
    `emission-is-invariant-to-the-array-shape`, and that gate catches it for
    every index inside its range rather than at the one index this mutation
    happens to name."""
    real = SM._pow_rational_lines

    def mutated(aux_name, base, p, q):
        if _element_index_of(base) >= 2:
            return real(aux_name, base, 1, q + 2)
        return real(aux_name, base, p, q)

    return mutated


def _rat_wrong_only_past_the_second_element_at_q_two():
    """THE PRODUCT OF THE TWO AXES, which is where the round that closed each
    of them separately left a hole.

    Conditioned on `element >= 2` AND `q == 2` — and BOTH of those values are
    inside the sets this file prints as driven, `DRIVEN_AUX_ELEMENTS` and
    `DRIVEN_RATIONAL_PQ`. It is therefore NOT covered by the exponent radius
    disclosed in `docs/gauge-coverage.md`: a reader who sees element counts
    `[1..6]` and pairs `1/2, 1/4, 3/2` both declared driven concludes their
    combination is driven, and on the predecessor tree it was not. Every
    element index `>= 2` was reached at `(1, 4)` alone, because the shape
    invariance gate ran ONE hardwired exponent per branch while the arity
    measurement recorded the two axes as marginals. This survived all 22 gates.

    Emits `aux^(q+2) = x^p` past element 1 at `q == 2`, so on
    `x[2]**0.5 - x[1]**0.5 <= 7.9` over [1, 81] the reachable value at element
    2 is capped at `81^(1/4)` while the truth is `81^(1/2)` — a REFUTED query
    comes back VERIFIED, which is the missed-violation direction with nothing
    downstream to catch it. `q + 2` rather than `q + 1` for the reason
    `_rat_wrong_only_at_a_larger_denominator` gives: an odd `q` is refused
    outright and would be caught by malformedness instead of by the axis this
    item is for."""
    real = SM._pow_rational_lines

    def mutated(aux_name, base, p, q):
        if q == 2 and _element_index_of(base) >= 2:
            return real(aux_name, base, p, q + 2)
        return real(aux_name, base, p, q)

    return mutated


def _int_wrong_only_past_the_second_element_at_degree_five():
    """The SAME product on the INTEGER branch, and it is a separate item
    because the two branches are two seams with two shape recoveries.

    Conditioned on `element >= 2` AND `exp == 5`, both printed as driven. The
    predecessor's integer shape probe was hardwired to exponent 3, so every
    integer exponent other than 3 was reached at element 0 only and this
    survived exactly as its rational twin did. Emits `x^(exp-3)` past element
    1 at exponent 5, so on `x[2]**5 - x[1]**5 <= 10` over [1, 2] the reachable
    value at element 2 is capped at `2^2` while the truth is `2^5 - 1^5 = 31`
    — a REFUTED query comes back VERIFIED.

    `_pow_integer_body` is handed neither the element index nor the count, so
    the condition reads the index off the base term exactly as the rational
    element mutations do; that seam gap is named in `docs/gauge-coverage.md`
    and this item is one more consequence of it."""
    real = SM._pow_integer_body

    def mutated(term, exp_val):
        if exp_val == 5 and _element_index_of(term) >= 2:
            return real(term, exp_val - 3)
        return real(term, exp_val)

    return mutated


def _emit_rat_aux_is_the_base(aux_name, base, p, q):
    return [
        f"(declare-const {aux_name} Real)",
        f"(assert (= {aux_name} {base}))",
    ]


def _emit_shared_aux(out_id, element, n_out):
    return f"aux_{out_id}"


def _emit_rat_one_declaration_for_two_elements(aux_name, base, p, q):
    """The WELL-FORMED form of aux-sharing, and the reason it is a separate
    battery item from the name collision above.

    Sharing the NAME alone emits two ``declare-const`` of one symbol, which
    is illegal SMT-LIB2: both backends refuse the script and the obligation
    comes back `unknown`. That is a catch, but it is a catch by
    MALFORMEDNESS, and a reader could take it for evidence that the gauge
    would notice a well-formed sharing. It would not, unless something
    expresses one — so this does.

    Paired with :func:`_emit_shared_aux` it declares the symbol once and
    then constrains it TWICE, which is well-formed and says
    ``x0_0 == x0_1``: the difference of the two roots collapses to 0, and a
    claim that is FALSE at ``[4, 1]`` is silently DISCHARGED. That is the
    missed-violation direction, with nothing downstream to catch it.

    Keyed on the base term's element suffix rather than on state, so the
    scalar fixtures (whose base is ``x0``) are untouched and the catch is
    attributable to the vector gate."""
    lines = []
    if not base.endswith("_1"):
        lines.append(f"(declare-const {aux_name} Real)")
        if q % 2 == 0:
            lines.append(f"(assert (>= {aux_name} 0.0))")
    lines.append(
        f"(assert (= {SM._repeated_product(aux_name, q)} "
        f"{SM._repeated_product(base, p)}))"
    )
    return lines


# -- replay mutations ---------------------------------------------------------


def _replay_exponent_inverted(v, p, q):
    return OB._exact_rational_power(v, q, p)


def _replay_identity(v, p, q):
    return v


# -- transfer mutation --------------------------------------------------------


def _transfer_pow_is_the_base(eqn, params, ins):
    """A `pow` transfer that returns the BASE's box unchanged — right at
    exponent 1 and wrong everywhere else, so it EXCLUDES values jax computes.
    Transfer-face wrongness the emission face does not share, which is the
    mutation that makes a face asymmetry visible."""
    import stelling.interval as iv

    a = ins[0]
    return [iv.IntervalArray(shape=a.shape, los=a.los, his=a.his)]


def _always_emittable(exp_float):
    return None


def _rationalise_the_exponent(exp_float):
    """AUDIT S1's own defect, as a mutation: the exponent replaced by a nearby
    low-denominator rational. `float(Fraction(1, 10))` rounds back to `0.1`, so
    no binary64 distance can see the substitution."""
    return Fraction(exp_float).limit_denominator(128)


def _mutations():
    """The battery, built lazily so the row's seams are looked up against
    whichever tree is loaded."""
    muts = {
        "row-absent-from-the-emission-set": {
            "__patches__": ((OB, "_SUPPORTED", OB._SUPPORTED - {"pow"}),),
        },
        "fragment-claims-linear": {"__linear_fragment__": True},
        "rational-admission-always-yes": {
            "__patches__": ((OB, "rational_pow_problem", _always_emittable),),
        },
        "exponent-rationalised-to-a-nearby-fraction": {
            "__patches__": ((OB, "pow_exponent_rational", _rationalise_the_exponent),),
        },
        "transfer-is-the-base": {"transfers": {"pow": _transfer_pow_is_the_base}},
        "emit-integer-off-by-one": {
            "__patches__": ((SM, "_pow_integer_body", _emit_int_off_by_one),),
        },
        "emit-integer-exponent-ignored": {
            "__patches__": ((SM, "_pow_integer_body", _emit_int_exponent_ignored),),
        },
        "emit-integer-loses-the-reciprocal": {
            "__patches__": ((SM, "_pow_integer_body", _emit_int_no_reciprocal),),
        },
        "emit-integer-wrong-only-above-degree-three": {
            "__patches__": (
                (SM, "_pow_integer_body", _int_wrong_only_above_degree_three()),
            ),
        },
        "emit-integer-wrong-only-past-the-second-element-at-degree-five": {
            "__patches__": (
                (SM, "_pow_integer_body",
                 _int_wrong_only_past_the_second_element_at_degree_five()),
            ),
        },
        "emit-rational-wrong-only-at-a-larger-denominator": {
            "__patches__": (
                (SM, "_pow_rational_lines",
                 _rat_wrong_only_at_a_larger_denominator()),
            ),
        },
        "emit-rational-wrong-only-at-a-numerator-past-one": {
            "__patches__": (
                (SM, "_pow_rational_lines",
                 _rat_wrong_only_at_a_numerator_past_one()),
            ),
        },
        "emit-rational-wrong-only-past-the-second-element": {
            "__patches__": (
                (SM, "_pow_rational_lines",
                 _rat_wrong_only_past_the_second_element()),
            ),
        },
        "emit-rational-wrong-only-past-the-second-element-at-q-two": {
            "__patches__": (
                (SM, "_pow_rational_lines",
                 _rat_wrong_only_past_the_second_element_at_q_two()),
            ),
        },
        "emit-rational-sides-swapped": {
            "__patches__": ((SM, "_pow_rational_lines", _emit_rat_sides_swapped),),
        },
        "emit-rational-root-guard-dropped": {
            "__patches__": ((SM, "_pow_rational_lines", _emit_rat_guard_dropped),),
        },
        "emit-rational-constraint-never-asserted": {
            "__patches__": ((SM, "_pow_rational_lines", _emit_rat_never_asserted),),
        },
        "emit-rational-denominator-off-by-one": {
            "__patches__": (
                (SM, "_pow_rational_lines", _emit_rat_denominator_off_by_one),
            ),
        },
        "emit-rational-aux-is-the-base": {
            "__patches__": ((SM, "_pow_rational_lines", _emit_rat_aux_is_the_base),),
        },
        "emit-rational-aux-shared-across-elements": {
            "__patches__": ((SM, "_pow_aux_name", _emit_shared_aux),),
        },
        "emit-rational-one-aux-for-two-elements": {
            "__patches__": (
                (SM, "_pow_aux_name", _emit_shared_aux),
                (SM, "_pow_rational_lines",
                 _emit_rat_one_declaration_for_two_elements),
            ),
        },
        "replay-exponent-inverted": {
            "__patches__": ((OB, "_exact_rational_power", _replay_exponent_inverted),),
        },
        "replay-as-the-identity": {
            "__patches__": ((OB, "_exact_rational_power", _replay_identity),),
        },
    }
    return muts


# --- the gates ---------------------------------------------------------------
#
# Every gate returns True = the subject PASSED it, False = the gate CAUGHT the
# subject. Exceptions are converted to a catch DELIBERATELY and only here:
# `EmissionInfidelityError` and `SolverDisagreement` are the machinery WORKING
# — a mutation that trips one has been caught, not crashed — and
# `fidelity.gauge` would otherwise propagate them as broken gates. That
# conversion is why a gate must never be the only thing asserting a behaviour:
# the row's own tests assert the loud errors loudly.


def _run(harness, subject, *, timeout=TIMEOUT_MS):
    with _patched(subject), _maybe_linear_fragment(subject):
        return check(harness, vacuity_mode="inputs-only",
                     solver_timeout_ms=timeout)


def _outcomes(harness, subject, *, timeout=TIMEOUT_MS):
    """The per-obligation ESCALATION outcomes, not the verdict status. See
    the module docstring: this is what keeps the discharge gates independent
    of `verdict.VERIFIED_BARRED_PRIMITIVES`."""
    with _patched(subject), _maybe_linear_fragment(subject):
        closed = trace(harness)
        p = propagate(closed)
        esc = escalate(closed, p, SolverConfig(timeout_ms=timeout))
        return tuple(r.outcome for r in esc.records)


def _sole_witness(v):
    """The single witness's rational value, or None when there is none."""
    if v.status != "REFUTED" or len(v.witnesses) != 1:
        return None
    (w,) = v.witnesses
    if len(w.values) != 1:
        return None
    ((_, text),) = w.values
    return Fraction(text)


def gate_refutes_the_false_integer_property(subject):
    """POSITIVE CONTROL, integer branch. `x^3 - x <= 23` over [-2, 3] is
    violated at x = 3 (24 > 23) and the interval transfer declines the
    straddling box, so the product expansion is what answers it."""
    try:
        return _sole_witness(_run(_int_harness(INT_FALSE_BOUND), subject)) is not None
    except Exception:  # EmissionInfidelityError included: a CATCH, not a crash
        return False


def gate_refutes_under_jit(subject):
    """The same property with the `pow` fused inside a ``jit`` call, where the
    equation sits one transparent descent below the top level. A row reached
    only by the eager shape passes the gate above and is bypassed here."""
    try:
        return (
            _sole_witness(_run(_int_harness(INT_FALSE_BOUND, jit=True), subject))
            is not None
        )
    except Exception:
        return False


def gate_discharges_the_true_integer_property(subject):
    """NEGATIVE CONTROL, integer branch. Bound moved to the exact maximum, so
    the property HOLDS everywhere in the box and nothing may refute it."""
    try:
        return _outcomes(_int_harness(INT_TRUE_BOUND), subject) == ("discharged",)
    except Exception:
        return False


def gate_refutes_the_false_integer_property_at_degree_five(subject):
    """POSITIVE CONTROL at the SECOND integer magnitude. `x^5 - x <= 239` over
    [-2, 3] is violated at x = 3 (240 > 239). Identical in every respect to
    the degree-3 gate except the exponent, so a mutation this catches and that
    one does not is attributable to the MAGNITUDE — which is what
    `emit-integer-wrong-only-above-degree-three` is, and what nothing in the
    shipped battery could see."""
    try:
        return _sole_witness(
            _run(_int_high_harness(INT_HIGH_FALSE_BOUND), subject)
        ) is not None
    except Exception:  # EmissionInfidelityError included: a CATCH, not a crash
        return False


def gate_discharges_the_true_integer_property_at_degree_five(subject):
    """NEGATIVE CONTROL at the second integer magnitude: bound moved to the
    exact maximum 240, so the property HOLDS and nothing may refute it."""
    try:
        return _outcomes(
            _int_high_harness(INT_HIGH_TRUE_BOUND), subject
        ) == ("discharged",)
    except Exception:
        return False


def gate_discharges_the_negative_exponent_identity(subject):
    """`x**-2 * x * x <= 1.0` is exactly 1 in ℝ. The RECIPROCAL is the whole
    content: without it the emitted value is `x^4`, which reaches 16."""
    try:
        return _outcomes(_neg_exponent_harness(1.0), subject) == ("discharged",)
    except Exception:
        return False


def gate_discharges_the_fourth_power_reciprocal_identity(subject):
    """The reciprocal at the SECOND magnitude: `x**-4 * x*x*x*x <= 1.0` is
    exactly 1 in ℝ. The pair (sign, magnitude) is what `_pow_integer_body`
    reads and this is the corner of it the shipped battery left empty."""
    try:
        return _outcomes(
            _neg_high_exponent_harness(1.0), subject
        ) == ("discharged",)
    except Exception:
        return False


def gate_discharges_the_true_rational_upper_bound(subject):
    """NEGATIVE CONTROL, rational branch, UPPER direction: `sqrt(x) <= 2` over
    [1, 4] holds, at the endpoint."""
    try:
        return _outcomes(_rat_upper_harness(2.0), subject) == ("discharged",)
    except Exception:
        return False


def gate_discharges_the_true_rational_lower_bound(subject):
    """NEGATIVE CONTROL, rational branch, LOWER direction — the only shape in
    which the even-`q` root guard changes an answer."""
    try:
        return _outcomes(_rat_lower_harness(1.0), subject) == ("discharged",)
    except Exception:
        return False


def gate_refutes_the_false_rational_property(subject):
    """POSITIVE CONTROL, rational branch: `sqrt(x) <= 3` over [1, 16] is
    violated above 9, and a wrong DENOMINATOR makes the violation vanish."""
    try:
        return _sole_witness(_run(_rat_false_harness(), subject)) is not None
    except Exception:
        return False


def gate_refutes_the_false_rational_property_at_numerator_three(subject):
    """POSITIVE CONTROL at the SECOND numerator. `x^(3/2) <= 7.9` over [1, 4]
    is violated at x = 4, where the exact value is 8. A repair that reads `p`
    only for the `p == 1` family — the whole of what shipped — hides that
    violation, and this is the only gate that sees it."""
    try:
        return _sole_witness(
            _run(_rat_numerator_false_harness(), subject)
        ) is not None
    except Exception:
        return False


def gate_discharges_the_true_rational_bound_at_numerator_three(subject):
    """NEGATIVE CONTROL at the second numerator: `x^(3/2) <= 8` over [1, 4]
    holds, at the endpoint."""
    try:
        return _outcomes(
            _rat_numerator_harness(8.0), subject
        ) == ("discharged",)
    except Exception:
        return False


def gate_refutes_the_false_rational_property_at_denominator_four(subject):
    """POSITIVE CONTROL at the SECOND denominator. `x^(1/4) <= 2.9` over
    [1, 81] is violated at x = 81, where the exact value is 3."""
    try:
        return _sole_witness(
            _run(_rat_denominator_false_harness(), subject)
        ) is not None
    except Exception:
        return False


def gate_discharges_the_true_rational_bound_at_denominator_four(subject):
    """NEGATIVE CONTROL at the second denominator: `x^(1/4) <= 3` over
    [1, 81] holds, at the endpoint."""
    try:
        return _outcomes(
            _rat_denominator_harness(3.0), subject
        ) == ("discharged",)
    except Exception:
        return False


def gate_refutes_the_false_vector_property(subject):
    """The ARRAY item: `sqrt(x0) - sqrt(x1) <= 0.9` is violated at [4, 1], and
    only because the two auxiliaries are INDEPENDENT."""
    try:
        v = _run(_rat_vector_harness(), subject)
        return v.status == "REFUTED" and bool(v.witnesses)
    except Exception:
        return False


def gate_witness_executes_through_jax(subject):
    """The independent leg: the witness's rational value, executed through the
    SAME jnp program EAGERLY and under ``jit``, must actually violate the
    predicate. Shares no code with the emission or the replay."""
    try:
        x = _sole_witness(_run(_int_harness(INT_FALSE_BOUND), subject))
        if x is None or not (INT_BOX[0] <= x <= INT_BOX[1]):
            return False
        xs = jnp.asarray(float(x), jnp.float64)
        eager = float(np.asarray(_int_value(xs)))
        jitted = float(np.asarray(jax.jit(_int_value)(xs)))
        return (eager > INT_FALSE_BOUND and jitted > INT_FALSE_BOUND
                and eager == jitted)
    except Exception:
        return False


def _pow_box(box, exponent):
    """The propagated box of the `pow` EQUATION's output — located by
    primitive name so the gate cannot end up measuring something downstream."""

    def h():
        x = any_array((), "float64", box)
        return (assert_(jnp.power(x, exponent) <= 1e308),)

    closed = trace(h)
    env = interval_env(closed)
    eqns = [e for e in closed.jaxpr.eqns if str(e.primitive) == "pow"]
    if not eqns:
        raise AssertionError(
            "the fixture produced no 'pow' equation — the gauge would be "
            "measuring a different program"
        )
    return env[eqns[0].outvars[0].id]


def gate_interval_containment_eager_and_jit(subject):
    """TRANSFER face. Every value jax computes on this target over the
    declared box lies inside the propagated box — sampled eagerly and under
    ``jit``, because a rule keyed on a name a fused program never binds passes
    eagerly and is bypassed under ``jit``. The oracle is the TARGET, not numpy
    and not stelling.

    STRICTLY POSITIVE BOXES ONLY, because that is the whole domain of
    `interval.pow_`'s corner rule: a base reaching 0 or below has no sound
    rule there and the transfer declines to ⊤, which contains everything and
    would make this gate vacuous."""
    try:
        with _patched(subject), _maybe_linear_fragment(subject):
            for box, exponent in (
                ((1.0, 4.0), RAT_EXP),
                ((1.0, 16.0), RAT_DENOMINATOR_EXP),
                ((0.5, 2.0), INT_EXP),
                ((1.0, 2.0), NEG_EXP),
                ((2.0, 8.0), RAT_NUMERATOR_EXP),
                # the widened arity, on the transfer face too: the emission
                # gates now drive |exp| >= 4 and the containment gate would
                # otherwise be measuring a strictly smaller exponent set than
                # the row's other face.
                ((0.5, 2.0), INT_HIGH_EXP),
                ((1.0, 2.0), NEG_HIGH_EXP),
            ):
                b = _pow_box(box, exponent)
                lo, hi = box
                pts = [lo, hi, (lo + hi) / 2.0, lo + (hi - lo) / 3.0]
                for pt in pts:
                    xs = jnp.asarray(pt, jnp.float64)
                    fn = (lambda z, e=exponent: jnp.power(z, e))
                    eager = float(np.asarray(fn(xs)))
                    jitted = float(np.asarray(jax.jit(fn)(xs)))
                    if eager != jitted:
                        return False
                    for val in (eager, jitted):
                        if not (b.los[0] <= val <= b.his[0]):
                            return False
        return True
    except Exception:
        return False


def _slice_of(harness):
    closed = trace(harness)
    p = propagate(closed)
    items = OB.slice_unknown_obligations(closed, p, interval_env(closed))
    if len(items) != 1 or isinstance(items[0], OB.DeclinedObligation):
        return None
    return items[0]


def gate_fragment_is_nonlinear(subject):
    """A `pow` slice is QF_NRA on BOTH branches, and for two different
    reasons: the integer expansion is a product of a declared input with
    itself, and the rational encoding's `aux^q` is a product of a fresh symbol
    with itself WHATEVER the base is (audit 0.2.0 M9). Emitting either under a
    linear logic claims a fragment the problem is not in — and both backends
    then refuse the script."""
    try:
        with _patched(subject), _maybe_linear_fragment(subject):
            for harness in (_int_harness(INT_FALSE_BOUND), _rat_false_harness()):
                sl = _slice_of(harness)
                if sl is None or sl.fragment != OB.QF_NRA:
                    return False
        return True
    except Exception:
        return False


def _declines_emission(harness):
    closed = trace(harness)
    p = propagate(closed)
    if not [o for o in p.obligations if o.status == "unknown"]:
        return None  # the transfer settled it; nothing was ever emitted
    items = OB.slice_unknown_obligations(closed, p, interval_env(closed))
    return all(isinstance(i, OB.DeclinedObligation) for i in items)


def gate_non_dyadic_exponent_declines(subject):
    """AUDIT S1. `0.1` denotes a dyadic rational far over the degree cap, so
    the obligation must decline — never be emitted about a nearby fraction."""
    try:
        with _patched(subject), _maybe_linear_fragment(subject):
            if _declines_emission(_non_dyadic_harness()) is not True:
                return False
        return _run(_non_dyadic_harness(), subject).status != "VERIFIED"
    except Exception:
        return False


def gate_negative_base_declines(subject):
    """A fractional exponent over a box reaching below zero is NaN in jax and
    has no faithful Real encoding, so the slice must decline."""
    try:
        with _patched(subject), _maybe_linear_fragment(subject):
            return _declines_emission(_negative_base_harness()) is not False
    except Exception:
        return False


def _canonicalise(lines, names):
    """`lines` with each symbol in `names` replaced by its canonical token.

    One simultaneous alternation rather than chained `str.replace`, so a
    substitution cannot rewrite the output of an earlier one, and `\\b`-anchored
    so `aux_2_1` is never matched inside `aux_2_12`. Longest name first, because
    Python's alternation takes the first branch that matches."""
    if not names:
        return tuple(lines)
    pattern = re.compile(
        "|".join(
            rf"\b{re.escape(n)}\b"
            for n in sorted(names, key=len, reverse=True)
        )
    )
    return tuple(pattern.sub(lambda m: names[m.group(0)], ln) for ln in lines)


def _seam_blocks(harness, attr, record):
    """Emit `harness` with `smt.<attr>` spied, and return the per-element
    CANONICAL seam output — or None if there was no slice to emit.

    The spy wraps whatever is installed at call time, which is the MUTATED
    function: `_patched` has already entered by the time a gate calls this, so
    what gets recorded is the subject's emission and not the row's."""
    sl = _slice_of(harness)
    if sl is None:
        return None
    calls = []
    real = getattr(SM, attr)

    def spy(*args):
        out = real(*args)
        calls.append((args, out))
        return out

    with mock.patch.object(SM, attr, spy):
        SM.emit(sl, "z3", TIMEOUT_MS)
    return [record(args, out) for args, out in calls]


def _rational_block(args, out):
    aux_name, base, _p, _q = args
    return _canonicalise(out, {aux_name: "AUX", base: "BASE"})


def _integer_block(args, out):
    term, _exp = args
    return _canonicalise((out,), {term: "BASE"})


_INVARIANCE_SWEEP = (
    ("_pow_rational_lines", _RATIONAL_FIXTURE_EXPONENTS, _rational_block),
    ("_pow_integer_body", _INTEGER_FIXTURE_EXPONENTS, _integer_block),
)


def gate_emission_is_invariant_to_the_array_shape(subject):
    """THE SHAPE AXIS, CLOSED RATHER THAN SAMPLED — AT EVERY DRIVEN EXPONENT.

    Every other gate in this battery drives one or two element counts and is
    therefore evidence about those counts only — which is how a mutation
    correct at elements 0 and 1 and wrong from element 2 on passed all
    twenty-one of them and minted a false VERIFIED. Adding a three-element
    fixture would have answered that mutation and nothing else; the next index
    would sit in the next gap.

    So this gate asserts an INVARIANT instead. `pow` is elementwise, so the
    lines one element emits cannot legitimately depend on which element it is
    or on how many there are: for every element count in
    `_INVARIANCE_ELEMENT_COUNTS`, every element's seam output must equal the
    SCALAR output with the auxiliary and base symbols substituted. Emission is
    text, so a shape-conditioned emission wrongness IS a per-element difference
    in that text — which makes this gate evidence about every conditioning
    function on the SHAPE and not about a sampled point.

    AND IT SWEEPS THE EXPONENT, WHICH IS THE HALF THE PREDECESSOR LEFT OUT.
    A shape invariance driven at ONE exponent per branch is an invariance in
    the shape and a SAMPLE in the exponent, and the product of the two is where
    the wrongness lives: `element >= 2 AND q == 2` names an element index and a
    denominator that are BOTH printed as driven, and it survived every gate
    because their combination was not. So the loop is over the driven exponent
    set — the same `_*_FIXTURE_EXPONENTS` tuples `DRIVEN_INTEGER_EXPONENTS` and
    `DRIVEN_RATIONAL_PQ` are derived from, so the sweep cannot drift from the
    printed scope — with one probe per exponent and a fresh scalar reference
    per exponent, because the emitted text legitimately differs BETWEEN
    exponents and must not differ WITHIN one.

    What this closes is therefore the full (element, n_out) x (driven exponent)
    product for the EMISSION, and `_measured_seam_reach` records that product
    jointly rather than as three marginals. What it is NOT: evidence about the
    transfer, the slice, the replay or the verdict at those counts, nor about
    any exponent outside the driven set. Those stages are driven at
    `_VERDICT_ELEMENT_COUNTS` and the SCOPE string says so.

    No slice means nothing was emitted and the invariance is vacuous, so that
    is a CATCH — the same conservative direction every other gate takes with an
    exception. The probe is built to survive `transfer-is-the-base` at every
    exponent for that reason, or this gate would report a transfer wrongness as
    a shape one."""
    try:
        with _patched(subject), _maybe_linear_fragment(subject):
            for attr, exponents, record in _INVARIANCE_SWEEP:
                for exponent in exponents:
                    reference = None
                    for n in _INVARIANCE_ELEMENT_COUNTS:
                        blocks = _seam_blocks(
                            _shape_probe_harness(exponent, n), attr, record
                        )
                        # one seam call per element, or the probe is not
                        # measuring the shape it names
                        if blocks is None or len(blocks) != n:
                            return False
                        if reference is None:
                            reference = blocks[0]
                        if any(b != reference for b in blocks):
                            return False
        return True
    except Exception:
        return False


def gate_replay_agrees_with_jax(subject):
    """EMISSION == REPLAY, measured rather than declared: at a grid of
    rational points inside the declared box, the exact-rational replay's
    verdict on the predicate must equal the one jax's own execution gives.
    Driven on the RATIONAL branch, whose replay is the one audit S3 found
    doing float arithmetic behind an exactness claim."""
    try:
        with _patched(subject), _maybe_linear_fragment(subject):
            harness = _rat_false_harness()
            sl = _slice_of(harness)
            if sl is None:
                return False
            (name,) = [i.name for i in sl.inputs]
            for num in (1, 4, 9, 16):
                q = Fraction(num)
                replayed = OB.evaluate_predicate(sl, {name: q})
                executed = bool(
                    np.asarray(
                        jax.jit(lambda z: jnp.power(z, 0.5) <= 3.0)(
                            jnp.asarray(float(q), jnp.float64)
                        )
                    )
                )
                if replayed != executed:
                    return False
        return True
    except Exception:
        return False


GATES = {
    "refutes-the-false-integer-property": gate_refutes_the_false_integer_property,
    "refutes-under-jit": gate_refutes_under_jit,
    "discharges-the-true-integer-property": gate_discharges_the_true_integer_property,
    "refutes-the-false-integer-property-at-degree-five":
        gate_refutes_the_false_integer_property_at_degree_five,
    "discharges-the-true-integer-property-at-degree-five":
        gate_discharges_the_true_integer_property_at_degree_five,
    "discharges-the-negative-exponent-identity":
        gate_discharges_the_negative_exponent_identity,
    "discharges-the-fourth-power-reciprocal-identity":
        gate_discharges_the_fourth_power_reciprocal_identity,
    "discharges-the-true-rational-upper-bound":
        gate_discharges_the_true_rational_upper_bound,
    "discharges-the-true-rational-lower-bound":
        gate_discharges_the_true_rational_lower_bound,
    "refutes-the-false-rational-property": gate_refutes_the_false_rational_property,
    "refutes-the-false-rational-property-at-numerator-three":
        gate_refutes_the_false_rational_property_at_numerator_three,
    "discharges-the-true-rational-bound-at-numerator-three":
        gate_discharges_the_true_rational_bound_at_numerator_three,
    "refutes-the-false-rational-property-at-denominator-four":
        gate_refutes_the_false_rational_property_at_denominator_four,
    "discharges-the-true-rational-bound-at-denominator-four":
        gate_discharges_the_true_rational_bound_at_denominator_four,
    "refutes-the-false-vector-property": gate_refutes_the_false_vector_property,
    "witness-executes-through-jax": gate_witness_executes_through_jax,
    "interval-containment-eager-and-jit": gate_interval_containment_eager_and_jit,
    "fragment-is-nonlinear": gate_fragment_is_nonlinear,
    "emission-is-invariant-to-the-array-shape":
        gate_emission_is_invariant_to_the_array_shape,
    "non-dyadic-exponent-declines": gate_non_dyadic_exponent_declines,
    "negative-base-declines": gate_negative_base_declines,
    "replay-agrees-with-jax": gate_replay_agrees_with_jax,
}

RESIDUAL: dict[str, str] = {}


# --- one measurement, read by several assertions -----------------------------

_REPORT_CACHE: list = []


def _report():
    """ONE gauging run, shared by every assertion that reads the same
    measurement.

    :func:`stelling.fidelity.gauge` is a pure function of (baseline, gates,
    battery), and this one runs every gate in `GATES` against the baseline and
    against every mutation in `_mutations()` — a product that has grown every
    round and carries no digit here for that reason. Running it once per
    assertion measured the identical thing four times and spent four times the
    solver budget on it; the widening below would have made that five.

    The one assertion that must NOT share this is
    :func:`test_in_process_stubs_are_destroyed_and_the_registries_restored`,
    whose subject IS a fresh run's effect on the live registries — it builds
    its own, and it is also what would catch a cached report papering over
    cross-run contamination.
    """
    if not _REPORT_CACHE:
        _REPORT_CACHE.append(
            gauge(BASELINE, GATES, _mutations(), residual=RESIDUAL, scope=SCOPE)
        )
    return _REPORT_CACHE[0]


def single_covered(report):
    """The mutations exactly ONE gate catches, as ``(mutation, gate)`` pairs.

    COMPUTED, and that is the point. ``docs/gauge-coverage.md`` stated this
    figure in prose as "sixteen of the seventeen are caught by more than one
    gate" while its own table two lines below printed five single-covered
    entries and the measured count was six — the file contradicting itself in
    the direction that made the gauge look stronger. The premise the page
    argues from ("a gauge with one single-covered mutation is one edit from a
    hole") was true six times over while the prose said once. A digit in prose
    that nothing recomputes is the defect class, so the figure is derived from
    the battery here and the page is compared against it by
    :func:`test_the_documented_coverage_figures_are_the_MEASURED_ones`.
    """
    return tuple(
        (name, gates[0]) for name, gates in report.caught_by if len(gates) == 1
    )


@dataclass(frozen=True)
class _SeamReach:
    """What the gates drive through the `pow` row's three seams, recorded as
    the JOINT tuples rather than as one set per axis.

    ``integer`` is ``(element, n_out, exp)`` at :func:`smt._pow_integer_body`
    and ``rational`` is ``(element, n_out, p, q)`` at
    :func:`smt._pow_rational_lines`; ``aux`` is the ``(element, n_out)`` pairs
    :func:`smt._pow_aux_name` was handed, kept separately because it is its own
    seam and the two must AGREE. ``inconsistent`` carries anything the
    recorder could not reconcile, so a broken measurement fails loudly rather
    than reporting a narrow reach as a clean one.

    The marginals below are PROJECTIONS of the joint sets and not a second
    measurement, which is the property the predecessor lacked."""

    integer: frozenset
    rational: frozenset
    aux: tuple
    inconsistent: tuple

    @property
    def integer_exponents(self):
        return tuple(sorted({e for _, _, e in self.integer}))

    @property
    def rational_pq(self):
        return tuple(sorted({(p, q) for _, _, p, q in self.rational}))

    @property
    def shapes(self):
        """Every ``(element, n_out)`` either exponent seam was driven at."""
        return tuple(sorted(
            {(i, n) for i, n, _ in self.integer}
            | {(i, n) for i, n, _, _ in self.rational}
        ))


def _measured_seam_reach() -> _SeamReach:
    """What the gates ACTUALLY drive through all THREE `pow` seams, measured by
    instrumenting each and running every gate against the BASELINE — and
    measured JOINTLY, which is the whole of the difference from its
    predecessor.

    This is the instrument that turns the SCOPE paragraph's arity claim from
    prose into a measurement. Prose went stale the moment it was written: the
    shipped file claimed "BOTH exponent branches" and drove `{-2, 3}` and
    `{(1, 2)}`, which is true of the branches and says nothing about the
    exponents — and the difference is exactly where three surviving mutations
    lived.

    THE THIRD SEAM WAS MISSING FROM HERE FOR A ROUND, which is the same defect
    at one remove: `_pow_aux_name` is the only seam handed the array shape, the
    predecessor instrumented the other two, and a mutation conditioned on the
    ELEMENT INDEX survived the widened battery. An instrument that measures two
    of the three parameters its row reads is measuring the ones it thought of.

    AND THEN IT MEASURED THREE AXES AS MARGINALS AND WROTE A CLAIM ABOUT THE
    PLANE. All three equalities passed on a tree where element index `>= 2` was
    driven at `(1, 4)` alone, so `element >= 2 AND q == 2` — both coordinates
    printed as driven — was ungauged and undisclosed. This function therefore
    records TUPLES.

    HOW THE SHAPE IS RECOVERED AT SEAMS THAT ARE NOT HANDED IT, because that is
    the one thing here that is not a plain spy:

    * on the RATIONAL branch `smt.emit` calls `_pow_aux_name(out.id, i, n_out)`
      and then `_pow_rational_lines` for the SAME element, so the shape the aux
      seam was just handed is the shape the lines seam is emitting at. The two
      are cross-checked against the element index carried by the base term's
      own name, and a disagreement is recorded rather than assumed away;
    * on the INTEGER branch there is no aux call at all, and
      `_pow_integer_body` is handed neither index nor count. But `smt.emit`
      renders one eqn's elements 0..n_out-1 CONSECUTIVELY, so a RUN starts at
      element 0 and its length is `n_out`. The element index comes off the base
      term the way :func:`_element_index_of` documents, and the run is required
      to be exactly `range(n_out)` — an out-of-order or gappy run means the
      recovery is unsound for this tree and says so instead of inventing a
      count.

    That the integer seam cannot see its own shape is the seam gap
    `docs/gauge-coverage.md` names; recovering it here measures the gap rather
    than closing it.
    """
    integer: set[tuple[int, int, int]] = set()
    rational: set[tuple[int, int, int, int]] = set()
    aux: set[tuple[int, int]] = set()
    inconsistent: list[str] = []
    real_int, real_rat = SM._pow_integer_body, SM._pow_rational_lines
    real_aux = SM._pow_aux_name
    # the shape `_pow_aux_name` was handed for the element being emitted now
    pending: list[tuple[int, int]] = []
    # the integer branch's current eqn run, as (element, exp) in emission order
    run: list[tuple[int, int]] = []

    def close_run():
        if not run:
            return
        n_out = len(run)
        if [e for e, _ in run] != list(range(n_out)):
            inconsistent.append(
                f"`_pow_integer_body` was driven at elements "
                f"{[e for e, _ in run]}, which is not one eqn's 0..n-1 run, so "
                f"the element COUNT cannot be recovered from the call order"
            )
        else:
            for element, exp_val in run:
                integer.add((element, n_out, exp_val))
        run.clear()

    def spy_int(term, exp_val):
        element = _element_index_of(term)
        if element == 0:
            close_run()
        run.append((element, exp_val))
        return real_int(term, exp_val)

    def spy_rat(aux_name, base, p, q):
        if not pending:
            inconsistent.append(
                f"`_pow_rational_lines` emitted `{aux_name}` with no preceding "
                f"`_pow_aux_name` call, so its shape is unknown"
            )
        else:
            element, n_out = pending.pop()
            if element != _element_index_of(base):
                inconsistent.append(
                    f"`_pow_aux_name` says element {element} while the base "
                    f"term `{base}` says {_element_index_of(base)}"
                )
            rational.add((element, n_out, p, q))
        return real_rat(aux_name, base, p, q)

    def spy_aux(out_id, element, n_out):
        aux.add((element, n_out))
        pending.append((element, n_out))
        return real_aux(out_id, element, n_out)

    with mock.patch.object(SM, "_pow_integer_body", spy_int), \
            mock.patch.object(SM, "_pow_rational_lines", spy_rat), \
            mock.patch.object(SM, "_pow_aux_name", spy_aux):
        for gate in GATES.values():
            gate(BASELINE)
            close_run()
            if pending:
                inconsistent.append(
                    f"{len(pending)} `_pow_aux_name` call(s) named an auxiliary "
                    f"no `_pow_rational_lines` call then emitted"
                )
                pending.clear()
    return _SeamReach(
        integer=frozenset(integer),
        rational=frozenset(rational),
        aux=tuple(sorted(aux)),
        inconsistent=tuple(inconsistent),
    )


def _render_joint(label, measured, declared, n_exponents):
    """The joint reach, PRINTED — as the product it is claimed to be when it is
    one, and as exactly what is missing from it when it is not. A set of this
    many tuples is not readable as a list, and a count alone is not a set; the
    factored form plus the equality against the product is both. When it is NOT
    the product the tuples ARE listed, because that is the case where a reader
    needs the names and not the shape of the claim."""
    missing = sorted(declared - measured)
    extra = sorted(measured - declared)
    if not missing and not extra:
        return (
            f"{label}: the FULL product of {len(DRIVEN_AUX_ELEMENTS)} "
            f"(element, n_out) shapes x {n_exponents} exponent(s) = "
            f"{len(measured)} tuples"
        )
    return (
        f"{label}: {len(measured)} tuples, NOT the declared product — "
        f"MISSING {missing}; EXTRA {extra}"
    )


# --- the stage table ---------------------------------------------------------


@dataclass(frozen=True)
class Reading:
    """One run of the instrument against whatever tree is loaded."""

    tree: str
    registries: tuple[tuple[str, str], ...]
    stages: tuple[tuple[str, str, str], ...]
    battery: str

    def render(self) -> str:
        lines = [
            "== pow-row gauge — reading",
            f"tree: {self.tree}",
            f"SCOPE — what these gates reach: {SCOPE}",
            "-- registries, printed from the definitions --",
        ]
        for name, val in self.registries:
            lines.append(f"  {name}: {val}")
        lines.append("-- stages in scope --")
        width = max((len(s) for s, _, _ in self.stages), default=0)
        for stage, outcome, detail in self.stages:
            lines.append(f"  {stage:<{width}s}  {outcome:<8s}  {detail}")
        lines.append("-- battery --")
        lines.append(self.battery)
        lines.append(
            f"== every line above is scoped to: {SCOPE} — a surface these "
            f"gates do not drive is NOT gauged by this reading =="
        )
        return "\n".join(lines)


def _outcome(fn):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 — a stage table records, never raises
        return ("ERROR", f"{type(e).__name__}: {e}")


def _stage_emit(harness, label):
    def run():
        sl = _slice_of(harness)
        if sl is None:
            return ("DECLINE", f"{label}: no slice to emit")
        text = SM.emit(sl, "z3", TIMEOUT_MS).text.splitlines()
        body = [ln for ln in text
                if "define-fun" in ln or "declare-const aux" in ln
                or ln.startswith("(assert (=") or ln.startswith("(assert (>=")]
        return ("REACH", f"{label}: {len(body)} row line(s); "
                         f"last: {body[-1] if body else '-'}")

    return run


def _stage_verdict(harness, label):
    def run():
        v = check(harness, vacuity_mode="inputs-only",
                  solver_timeout_ms=TIMEOUT_MS)
        if v.status == "REFUTED":
            (w,) = v.witnesses
            return ("REFUTED", f"{label}: witness {dict(w.values)}")
        return (
            "DECLINE" if v.status == "UNKNOWN" else v.status,
            f"{label}: {'; '.join(v.notes)[:200] or 'no notes'}",
        )

    return run


def _stage_escalation(harness, label):
    def run():
        return ("REACH", f"{label}: outcomes {_outcomes(harness, {})}")

    return run


def _stage_driven_arity():
    """The arity, INTO the rendered reading — including the JOINT reach, which
    is the part a reader cannot reconstruct from the marginals. A reader who
    quotes this page gets the measurement and not only the SCOPE sentence the
    page quotes."""
    reach = _measured_seam_reach()
    detail = (
        f"integer exponents {list(reach.integer_exponents)} ; (p,q) pairs "
        f"{[list(pq) for pq in reach.rational_pq]} ; (element, n_out) pairs "
        f"{[list(a) for a in reach.aux]} — measured at all three seams across "
        f"every gate, and NOT the admitted set. JOINTLY: "
        + _render_joint(
            "(element, n_out, exp)", reach.integer, DRIVEN_INTEGER_JOINT,
            len(DRIVEN_INTEGER_EXPONENTS),
        )
        + " ; "
        + _render_joint(
            "(element, n_out, p, q)", reach.rational, DRIVEN_RATIONAL_JOINT,
            len(DRIVEN_RATIONAL_PQ),
        )
    )
    if reach.inconsistent:
        return ("ERROR", detail + " ; INCONSISTENT: "
                + "; ".join(reach.inconsistent))
    return ("REACH", detail)


def _stage_ieee():
    closed = trace(_int_harness(INT_FALSE_BOUND))
    p = propagate(closed, semantics="ieee")
    e = escalate(closed, p, SolverConfig(timeout_ms=TIMEOUT_MS))
    st = p.obligations[0].status
    return (
        "DECLINE" if st == "unknown" else st.upper(),
        f"transfer -> {st}; escalation -> {'; '.join(e.notes)[:120]}",
    )


def measure() -> Reading:
    """Run the instrument and return the reading. Never raises: a stage that
    blows up is recorded as ``ERROR`` with the exception quoted."""
    import stelling

    registries = (
        ("obligation._SUPPORTED has 'pow'", repr("pow" in OB._SUPPORTED)),
        ("obligation._REPLAY_SUPPORTED has 'pow'",
         repr("pow" in OB._REPLAY_SUPPORTED)),
        ("obligation._INT_OVERFLOW_EMITTED has 'pow'",
         repr("pow" in OB._INT_OVERFLOW_EMITTED)),
        ("obligation.INTEGER_POW_EXPANSION_CAP",
         repr(OB.INTEGER_POW_EXPANSION_CAP)),
        ("obligation.RATIONAL_POW_DEGREE_CAP",
         repr(OB.RATIONAL_POW_DEGREE_CAP)),
        ("propagate.TRANSFERS['pow'] tier",
         repr(P.TRANSFERS.get("pow", (None, "-"))[1])),
        ("propagate.IEEE_TRANSFERS['pow'] tier",
         repr(P.IEEE_TRANSFERS.get("pow", (None, "-"))[1])),
        ("smt._FOLDABLE has 'pow'", repr("pow" in SM._FOLDABLE)),
        ("affine.AFFINE_SUPPORTED has 'pow'", repr("pow" in AF.AFFINE_SUPPORTED)),
        ("NEGATIVE registry control — 'exp' in _SUPPORTED",
         repr("exp" in OB._SUPPORTED)),
        ("emission set size", repr(len(OB._SUPPORTED))),
        ("battery size", repr(len(_mutations()))),
    )
    stages = tuple(
        (name,) + _outcome(fn)
        for name, fn in (
            ("bind-eager", lambda: (
                "REACH",
                repr([str(e.primitive)
                      for e in trace(_int_harness(INT_FALSE_BOUND)).jaxpr.eqns]),
            )),
            ("bind-jit", lambda: (
                "REACH",
                repr([str(e.primitive) for e in
                      trace(_int_harness(INT_FALSE_BOUND, jit=True)).jaxpr.eqns]),
            )),
            ("transfer-real-positive-box", lambda: (
                "REACH",
                f"pow([1,4], 0.5) = [{_pow_box((1.0, 4.0), 0.5).los[0]}, "
                f"{_pow_box((1.0, 4.0), 0.5).his[0]}]",
            )),
            ("emit-integer", _stage_emit(_int_harness(INT_FALSE_BOUND), "x^3 - x")),
            ("emit-rational", _stage_emit(_rat_false_harness(), "sqrt(x)")),
            ("emit-rational-vector", _stage_emit(_rat_vector_harness(), "sqrt(x[:])")),
            ("verdict-positive-integer",
             _stage_verdict(_int_harness(INT_FALSE_BOUND), "x^3 - x <= 23 FALSE")),
            ("verdict-positive-rational",
             _stage_verdict(_rat_false_harness(), "sqrt(x) <= 3 over [1,16] FALSE")),
            ("verdict-positive-vector",
             _stage_verdict(_rat_vector_harness(), "sqrt(x0)-sqrt(x1) <= 0.9 FALSE")),
            ("escalation-negative-integer",
             _stage_escalation(_int_harness(INT_TRUE_BOUND), "x^3 - x <= 24 TRUE")),
            ("escalation-negative-rational-upper",
             _stage_escalation(_rat_upper_harness(2.0), "sqrt(x) <= 2 TRUE")),
            ("escalation-negative-rational-lower",
             _stage_escalation(_rat_lower_harness(1.0), "sqrt(x) >= 1 TRUE")),
            ("escalation-negative-exponent",
             _stage_escalation(_neg_exponent_harness(1.0), "x^-2 * x * x <= 1 TRUE")),
            ("admission-non-dyadic", lambda: (
                "DECLINE" if _declines_emission(_non_dyadic_harness()) else "REACH",
                "x ** 0.1 over [1, 1e300]",
            )),
            ("admission-negative-base", lambda: (
                "DECLINE" if _declines_emission(_negative_base_harness()) else "REACH",
                "x ** 0.5 over [-1, 4]",
            )),
            ("ieee-semantics", _stage_ieee),
            # the arity, MEASURED into the rendered reading rather than left
            # only in the SCOPE prose the reading quotes
            ("driven-arity", _stage_driven_arity),
        )
    )
    try:
        battery = _report().render()
    except Exception as e:  # noqa: BLE001
        battery = f"fidelity.gauge REFUSED: {type(e).__name__}: {e}"
    return Reading(
        tree=stelling.__file__,
        registries=registries,
        stages=stages,
        battery=battery,
    )


# --- the assertions ----------------------------------------------------------


def test_the_battery_is_not_empty_and_the_baseline_reaches_the_solver():
    """Anti-vacuity, first. A battery of zero mutations and a pipeline in
    which every point declines print the same perfect score, so: the battery
    must be non-empty, and the positive control must record an ACTUAL solver
    invocation — not a decline that happens to be green."""
    muts = _mutations()
    assert len(muts) >= 20, sorted(muts)
    v = check(_int_harness(INT_FALSE_BOUND), vacuity_mode="inputs-only",
              solver_timeout_ms=TIMEOUT_MS)
    solver = v.stamp.solver
    assert isinstance(solver, tuple) and solver, (
        f"the positive control recorded no solver invocation ({solver}) — the "
        f"instrument did not reach the stage it claims to gauge"
    )
    assert all(s.invoked for s in solver)


def test_every_control_is_interval_undecidable_so_the_row_is_what_decides():
    """The controls must all survive interval propagation, or the gates
    measure the cheap layer instead of the row."""
    for label, harness in (
        ("int-false", _int_harness(INT_FALSE_BOUND)),
        ("int-true", _int_harness(INT_TRUE_BOUND)),
        ("int-jit", _int_harness(INT_FALSE_BOUND, jit=True)),
        ("int-high-false", _int_high_harness(INT_HIGH_FALSE_BOUND)),
        ("int-high-true", _int_high_harness(INT_HIGH_TRUE_BOUND)),
        ("neg-exponent", _neg_exponent_harness(1.0)),
        ("neg-high-exponent", _neg_high_exponent_harness(1.0)),
        ("rat-upper", _rat_upper_harness(2.0)),
        ("rat-lower", _rat_lower_harness(1.0)),
        ("rat-false", _rat_false_harness()),
        ("rat-numerator-false", _rat_numerator_false_harness()),
        ("rat-numerator-true", _rat_numerator_harness(8.0)),
        ("rat-denominator-false", _rat_denominator_false_harness()),
        ("rat-denominator-true", _rat_denominator_harness(3.0)),
        ("rat-vector", _rat_vector_harness()),
    ):
        p = propagate(trace(harness))
        assert all(o.status == "unknown" for o in p.obligations), (
            f"{label}: {[o.status for o in p.obligations]} — the transfer "
            f"settled it, so the gate driving it never reaches the row"
        )


def test_every_control_really_binds_pow_and_not_integer_pow():
    """ANTI-VACUITY. `x ** 2` binds `integer_pow` and `x ** 2.0` binds `pow`;
    a fixture that drifted to the former would gauge a different row while
    looking green."""
    for label, harness, at_top in (
        ("int-false", _int_harness(INT_FALSE_BOUND), True),
        ("int-jit", _int_harness(INT_FALSE_BOUND, jit=True), False),
        ("int-high-false", _int_high_harness(INT_HIGH_FALSE_BOUND), True),
        ("neg-exponent", _neg_exponent_harness(1.0), True),
        ("neg-high-exponent", _neg_high_exponent_harness(1.0), True),
        ("rat-false", _rat_false_harness(), True),
        ("rat-numerator-false", _rat_numerator_false_harness(), True),
        ("rat-denominator-false", _rat_denominator_false_harness(), True),
        ("rat-vector", _rat_vector_harness(), True),
    ):
        closed = trace(harness)
        prims = {str(e.primitive) for e in closed.jaxpr.eqns}
        assert ("pow" in prims) is at_top, (
            f"{label}: top-level primitives {sorted(prims)} — the jit fixture "
            f"must NOT have `pow` at top level (it is testing the descent) "
            f"and every other one must"
        )
        assert "integer_pow" not in prims, (
            f"{label}: bound `integer_pow`, not `pow` — this gauge would be "
            f"measuring the wrong row"
        )


def test_positive_control_refutes_with_a_witness_that_replays():
    v = check(_int_harness(INT_FALSE_BOUND), vacuity_mode="inputs-only",
              solver_timeout_ms=TIMEOUT_MS)
    assert v.status == "REFUTED", v.render()
    (w,) = v.witnesses
    assert "exact-rational replay" in w.replay
    ((_, text),) = w.values
    x = Fraction(text)
    assert INT_BOX[0] <= x <= INT_BOX[1]
    assert x ** 3 - x > INT_FALSE_BOUND


def test_positive_control_refutes_under_jit_too():
    closed = trace(_int_harness(INT_FALSE_BOUND, jit=True))
    assert "pow" not in {str(e.primitive) for e in closed.jaxpr.eqns}, (
        "the jit fixture put `pow` at top level — it is not testing the "
        "descent it claims to"
    )
    v = check(_int_harness(INT_FALSE_BOUND, jit=True),
              vacuity_mode="inputs-only", solver_timeout_ms=TIMEOUT_MS)
    assert v.status == "REFUTED", v.render()
    ((_, text),) = v.witnesses[0].values
    x = Fraction(text)
    assert x ** 3 - x > INT_FALSE_BOUND


def test_negative_controls_discharge_and_produce_no_witness():
    """Read at `escalate`, so this measures the ROW and not the bar."""
    for label, harness in (
        ("int-true", _int_harness(INT_TRUE_BOUND)),
        ("int-high-true", _int_high_harness(INT_HIGH_TRUE_BOUND)),
        ("neg-exponent", _neg_exponent_harness(1.0)),
        ("neg-high-exponent", _neg_high_exponent_harness(1.0)),
        ("rat-upper", _rat_upper_harness(2.0)),
        ("rat-lower", _rat_lower_harness(1.0)),
        ("rat-numerator-true", _rat_numerator_harness(8.0)),
        ("rat-denominator-true", _rat_denominator_harness(3.0)),
    ):
        assert _outcomes(harness, {}) == ("discharged",), (
            f"{label}: {_outcomes(harness, {})}"
        )


def test_the_driven_arity_is_MEASURED_at_the_seams_not_asserted_in_prose():
    """WHAT THIS BATTERY ACTUALLY DRIVES, instrumented rather than described.

    The shipped file said "BOTH exponent branches" and drove integer exponents
    ``{-2, 3}`` and the single pair ``(1, 2)``. Both statements were true; the
    first is about branches and the second is the SCOPE, and only the first was
    written down. Three mutations conditioned outside the second passed all
    fourteen gates.

    So the arity is a MEASUREMENT here. ``DRIVEN_*`` are read off the fixture
    table, this test instruments the two seams and runs every gate against the
    baseline, and the SCOPE string the gauge report prints is built from the
    same constants — so a fixture that drifts, an exponent literal that creeps
    back in, or a gate deleted all fail HERE rather than leaving a paragraph
    standing that nothing measures.
    """
    reach = _measured_seam_reach()
    assert not reach.inconsistent, (
        "the seam recorder could not reconcile what it saw, so the reach "
        "below is not a measurement:\n  " + "\n  ".join(reach.inconsistent)
    )
    ints, pqs, aux = reach.integer_exponents, reach.rational_pq, reach.aux
    assert ints == DRIVEN_INTEGER_EXPONENTS, (
        f"the gates drive integer exponents {list(ints)}, but the fixture "
        f"table declares {list(DRIVEN_INTEGER_EXPONENTS)} — the SCOPE this "
        f"file prints is now wrong about what it reaches"
    )
    assert pqs == DRIVEN_RATIONAL_PQ, (
        f"the gates drive (p, q) pairs {list(pqs)}, but the fixture table "
        f"declares {list(DRIVEN_RATIONAL_PQ)}"
    )
    assert aux == DRIVEN_AUX_ELEMENTS, (
        f"the gates drive (element, n_out) pairs {list(aux)}, but the fixture "
        f"table declares {list(DRIVEN_AUX_ELEMENTS)} — the SHAPE axis is the "
        f"one this file left as prose for a round, and a mutation "
        f"conditioned on the element index lived in that gap"
    )
    # ...and the two seams that see the shape must AGREE about it: the aux
    # names handed out and the shapes the two exponent seams were driven at are
    # the same set, or one of the two recordings is measuring something else
    assert reach.shapes == DRIVEN_AUX_ELEMENTS, (
        f"the exponent seams are driven at (element, n_out) pairs "
        f"{list(reach.shapes)} while `_pow_aux_name` hands out "
        f"{list(aux)} — the two shape recordings disagree"
    )

    # THE JOINT REACH, WHICH IS THE CLAIM THE THREE MARGINALS ABOVE DO NOT
    # MAKE. Every equality above passed on a tree where element index >= 2 was
    # driven at (1, 4) and nowhere else, so `element >= 2 AND q == 2` — both
    # coordinates inside the printed sets — was ungauged AND undisclosed. The
    # marginals are a projection of what follows and are kept because they are
    # what the SCOPE string prints; what makes the SCOPE string TRUE is this:
    assert reach.integer == DRIVEN_INTEGER_JOINT, _render_joint(
        "the integer branch's (element, n_out, exp) reach",
        reach.integer, DRIVEN_INTEGER_JOINT, len(DRIVEN_INTEGER_EXPONENTS),
    )
    assert reach.rational == DRIVEN_RATIONAL_JOINT, _render_joint(
        "the rational branch's (element, n_out, p, q) reach",
        reach.rational, DRIVEN_RATIONAL_JOINT, len(DRIVEN_RATIONAL_PQ),
    )
    # ANTI-VACUITY on the product itself: an empty exponent set or an empty
    # shape set would make both equalities hold over nothing
    assert len(reach.integer) == len(DRIVEN_AUX_ELEMENTS) * len(ints) > 0
    assert len(reach.rational) == len(DRIVEN_AUX_ELEMENTS) * len(pqs) > 0
    # the SCOPE the report and the reading both quote is BUILT from these,
    # so the printed claim cannot drift from the measured one
    assert f"[{_DRIVEN_INT_TEXT}]" in SCOPE and f"[{_DRIVEN_PQ_TEXT}]" in SCOPE
    assert f"[{_DRIVEN_N_TEXT}]" in SCOPE and f"[{_VERDICT_N_TEXT}]" in SCOPE

    # ANTI-VACUITY, and READ THE MODULE DOCSTRING ON WHAT THIS IS. The three
    # equalities above are DRIFT protection: they fail when the fixtures and
    # the printed SCOPE part company, which is exactly what happened last
    # round, and they would pass just as well at the arity that let four
    # mutations through. Everything genuinely new is below, and every floor
    # below is typed at the radius of a mutation someone already wrote.
    assert max(abs(e) for e in ints) >= 4, (
        "no fixture drives |exponent| >= 4, so a wrongness conditioned above "
        "degree 3 is invisible again — that mutation minted a false VERIFIED"
    )
    assert {e for e in ints if e < 0} and {e for e in ints if e > 0}
    assert min(e for e in ints) <= -4 and max(e for e in ints) >= 4, (
        "both SIGNS must be driven at both magnitude classes: the seam reads "
        "sign and magnitude separately and a repair can be wrong in the corner"
    )
    assert len({q for _, q in pqs}) >= 2, "one denominator is one q, not a branch"
    assert len({p for p, _ in pqs}) >= 2, "p == 1 is a family, not the space"
    assert max(q for _, q in pqs) >= 4 and max(p for p, _ in pqs) >= 3

    # the SHAPE floors. The element-index one is the mutation that survived;
    # the scalar one is the OTHER arm of `_pow_aux_name`'s naming rule, which
    # the byte-level emission tests pin and which a shape-only battery could
    # lose without noticing.
    assert (0, 1) in aux, (
        "the single-element arm of `_pow_aux_name` is never driven, so the "
        "unsuffixed spelling is emitted by nothing this gauge measures"
    )
    assert max(i for i, _ in aux) >= 2, (
        "no fixture reaches an element past index 1, so a wrongness "
        "conditioned there is invisible — that mutation minted a false "
        "VERIFIED (81^(1/4) - 1 = 2 under a bound of 1.9)"
    )
    assert max(n for _, n in aux) >= 3, (
        "no fixture drives an array past two elements, so a wrongness "
        "conditioned on the element COUNT is invisible"
    )
    # ...and the shape reach is not merely wide, it is CONTIGUOUS from the
    # scalar up, because the invariance gate's claim is about a range and a
    # gap in the range is a gap in the claim
    assert aux == tuple(
        sorted((i, n) for n in range(1, max(n for _, n in aux) + 1)
               for i in range(n))
    ), list(aux)

    # and every driven pair is one the shipped guard actually ADMITS, or the
    # widening bought coverage of exponents no program can reach
    for p, q in pqs:
        assert OB.rational_pow_problem(p / q) is None, (p, q)
    for e in ints:
        assert abs(e) <= OB.INTEGER_POW_EXPANSION_CAP, e


def test_the_three_conditional_mutations_are_CAUGHT_and_by_the_new_arity():
    """The three wrongnesses that survived the shipped battery, each pinned to
    the gate that kills it.

    Named individually rather than left to the aggregate "every mutation is
    caught", because what matters is not that something caught them — it is
    that the thing which caught them is the exponent the battery did not have.
    Delete the widening and these attributions fail before the survivor count
    does.
    """
    caught = dict(_report().caught_by)

    assert "refutes-the-false-integer-property-at-degree-five" in caught[
        "emit-integer-wrong-only-above-degree-three"]
    assert "refutes-the-false-integer-property" not in caught[
        "emit-integer-wrong-only-above-degree-three"], (
        "the degree-3 gate caught the above-degree-3 mutation, so the two "
        "fixtures are no longer distinguishing the magnitude and this "
        "attribution measures nothing"
    )

    assert caught["emit-rational-wrong-only-at-a-larger-denominator"] == (
        "refutes-the-false-rational-property-at-denominator-four",
    ), (
        "the q >= 4 mutation must be caught by the q = 4 gate and by that "
        "gate ALONE — the exclusivity is what says every other fixture in "
        "this battery is blind to a denominator past 2"
    )
    assert caught["emit-rational-wrong-only-at-a-numerator-past-one"] == (
        "refutes-the-false-rational-property-at-numerator-three",
    ), (
        "the p != 1 mutation must be caught by the p = 3 gate ALONE — every "
        "other fixture is inside the p == 1 family"
    )


def test_the_shape_probe_reaches_the_EMISSION_at_every_DRIVEN_exponent():
    """THE SWEEP'S LOAD-BEARING PROPERTY, ASSERTED RATHER THAN ARGUED.

    `gate_emission_is_invariant_to_the_array_shape` treats "no slice" as a
    CATCH, which is the right conservative direction and also means a probe
    that quietly stops being emitted at some exponent would turn the gate into
    a detector for that instead of for the shape. So the probe must be
    interval-UNDECIDABLE at every exponent it is driven at, and at every
    element count — and under TWO transfers, not one:

    * under the REAL transfer, or the gate never reaches the row and the
      baseline fails;
    * under `transfer-is-the-base`, or the gate CATCHES that mutation — which
      would be reporting a transfer wrongness as a shape one, and would move a
      documented catch set.

    The predecessor could only satisfy this by hand-tuning a bound per branch,
    which is why it drove one exponent per branch and why `element >= 2 AND
    q == 2` survived. The difference-against-zero shape satisfies it for free
    at every exponent (see `_shape_probe_harness`), and this test is what says
    so about the exponents actually driven rather than about the argument.

    AN EXPONENT THAT CANNOT CARRY A PROBE WOULD BE NAMED HERE. The auditor's
    own sweep stopped at the integer branch because a box straddling zero
    cannot carry a negative exponent; the failure below prints every
    (transfer, branch, exponent, element count) that did not reach the
    emission, so an undriven combination is disclosed by name and never left
    inside a set the file prints as driven.
    """
    unreached = []
    for label, subject in (
        ("the real transfer", BASELINE),
        ("transfer-is-the-base", _mutations()["transfer-is-the-base"]),
    ):
        with _patched(subject):
            for attr, exponents, record in _INVARIANCE_SWEEP:
                for exponent in exponents:
                    for n in _INVARIANCE_ELEMENT_COUNTS:
                        blocks = _seam_blocks(
                            _shape_probe_harness(exponent, n), attr, record
                        )
                        got = None if blocks is None else len(blocks)
                        if got != n:
                            unreached.append(
                                f"{label}: {attr} at exponent {exponent} with "
                                f"{n} element(s) produced "
                                f"{'no slice' if got is None else f'{got} seam call(s)'}"
                            )
    assert not unreached, (
        "the shape probe does not reach the emission everywhere the invariance "
        "gate drives it, so that gate is measuring emittability and not "
        "shape-invariance at:\n  " + "\n  ".join(unreached)
    )
    # ANTI-VACUITY: the sweep must actually be the DRIVEN sets, or the loop
    # above is a small green nothing
    swept = {attr: tuple(exps) for attr, exps, _ in _INVARIANCE_SWEEP}
    assert swept == {
        "_pow_rational_lines": _RATIONAL_FIXTURE_EXPONENTS,
        "_pow_integer_body": _INTEGER_FIXTURE_EXPONENTS,
    }, swept
    assert len(_INVARIANCE_ELEMENT_COUNTS) > 1 and min(
        _INVARIANCE_ELEMENT_COUNTS) == 1


# --- the conditional mutations' arithmetic, in ONE place that computes it ----
#
# Every one of these mutations is a MISSED violation: it caps the value the
# emitted script can reach BELOW the fixture's bound, so a genuinely REFUTED
# query comes back VERIFIED and nothing downstream sees it. That is three
# numbers per item — the truth at the box extremum, the bound, and the cap —
# and they were typed into three places each (the fixture docstring, the
# mutation docstring, `docs/gauge-coverage.md`). One of them had been wrong
# since the round that wrote it: `_rat_denominator_false_harness` described an
# `aux^5` emission with a cap of `81^(1/5)`, when the battery adds TWO to the
# denominator and `q = 5` is not emittable at all. The page said `aux^6` and
# was right, because the page is machine-checked and a docstring is not.
#
# So the emitted exponents are READ OFF THE LIVE MUTATION rather than declared,
# the inequalities are decided in exact `Fraction` arithmetic, and the rendered
# forms are required to appear in both the fixture docstring and the doc row.


@dataclass(frozen=True)
class _ConditionalCatch:
    """One conditional mutation's arithmetic. Nothing here is a result: the
    fixture's own driven exponent, box extremum and bound, and the mutation to
    interrogate. Everything derived is derived."""

    mutation: str
    harness: object
    driven: tuple[int, int]
    worst: Fraction
    bound: Fraction


_CONDITIONAL_CATCHES = (
    _ConditionalCatch(
        mutation="emit-rational-wrong-only-at-a-larger-denominator",
        harness=_rat_denominator_false_harness,
        driven=(1, 4),
        worst=Fraction(int(RAT_DENOMINATOR_BOX[1])),
        bound=Fraction("2.9"),
    ),
    _ConditionalCatch(
        mutation="emit-rational-wrong-only-at-a-numerator-past-one",
        harness=_rat_numerator_false_harness,
        driven=(3, 2),
        worst=Fraction(int(RAT_BOX[1])),
        bound=Fraction("7.9"),
    ),
)


def _emitted_pq(aux_name, base, lines):
    """The ``(p, q)`` a rational emission ACTUALLY wrote, counted off the
    defining equation rather than taken from the arguments — which is the whole
    point, since the mutation's job is to write something other than what it
    was passed."""
    eq = [ln for ln in lines if ln.startswith("(assert (= ")]
    if len(eq) != 1:
        raise AssertionError(f"expected one defining equation, got {lines}")
    q = len(re.findall(rf"\b{re.escape(aux_name)}\b", eq[0]))
    p = len(re.findall(rf"\b{re.escape(base)}\b", eq[0]))
    return p, q


def _emitted_form(p, q):
    """How an emitted ``aux^q = x^p`` is written in prose on this page and in
    `docs/gauge-coverage.md`. Always both exponents, so a forced ``p = 1`` is
    visible rather than spelled as a bare `x`."""
    return f"aux^{q} = x^{p}"


def _power_form(base, p, q):
    return f"{base}^({p}/{q})"


def _exceeds(worst: Fraction, p: int, q: int, bound: Fraction) -> bool:
    """``worst ** (p/q) > bound``, decided EXACTLY. Both sides are positive, so
    raising to ``q`` is order-preserving and no root is ever taken — which
    matters because `81^(1/6)` is irrational and a float comparison of it
    against a bound is the kind of evidence this file exists to refuse."""
    return worst**p > bound**q


def test_the_conditional_mutations_CAP_the_value_below_the_bound_RECOMPUTED():
    """WHY EACH CONDITIONAL MUTATION IS A MISSED VIOLATION, recomputed.

    For each: the truth at the box extremum must EXCEED the fixture's bound
    (so the query is genuinely REFUTED and the gate is a real positive
    control), and the value the MUTATED script can reach must not (so the
    violation disappears silently). Both decided in exact `Fraction`
    arithmetic, with the mutated exponents read off the live mutation.

    Then the rendered forms are required to appear in the fixture's docstring
    and in the doc row, which is what stops either from rotting the way
    `_rat_denominator_false_harness`'s did.
    """
    muts = _mutations()
    doc = _DOC.read_text(encoding="utf-8")
    for item in _CONDITIONAL_CATCHES:
        p, q = item.driven
        ((_mod, _attr, mutation),) = muts[item.mutation]["__patches__"]
        emitted_p, emitted_q = _emitted_pq(
            "aux_0", "x0", mutation("aux_0", "x0", p, q)
        )
        assert (emitted_p, emitted_q) != (p, q), (
            f"{item.mutation} emitted the CORRECT exponent at the fixture's "
            f"own (p, q) = {(p, q)} — it is not wrong where its gate looks, "
            f"so that gate is not measuring it"
        )
        # the baseline emission at the same call must be the driven pair, or
        # the comparison above is against a moving reference
        assert _emitted_pq(
            "aux_0", "x0", SM._pow_rational_lines("aux_0", "x0", p, q)
        ) == (p, q)

        assert _exceeds(item.worst, p, q, item.bound), (
            f"{item.mutation}'s fixture is not violated at the extremum: "
            f"{_power_form(item.worst, p, q)} <= {item.bound}, so the gate is "
            f"not a positive control and catches nothing"
        )
        assert not _exceeds(item.worst, emitted_p, emitted_q, item.bound), (
            f"{item.mutation} does NOT hide the violation: "
            f"{_power_form(item.worst, emitted_p, emitted_q)} still exceeds "
            f"{item.bound}, so it would be refuted anyway and this item is "
            f"measuring nothing"
        )

        form = _emitted_form(emitted_p, emitted_q)
        cap = _power_form(item.worst, emitted_p, emitted_q)
        truth = _power_form(item.worst, p, q)
        for where, text in (
            (f"{item.harness.__name__}'s docstring", item.harness.__doc__),
            (f"{_DOC.name}'s `{item.mutation}` row", _doc_row(doc, item.mutation)),
        ):
            for phrase in (form, cap, truth):
                assert phrase in text, (
                    f"{where} does not quote `{phrase}`, which is what this "
                    f"test just recomputed — the number and the prose have "
                    f"parted company again"
                )


@dataclass(frozen=True)
class _ShapeConditionedCatch:
    """One element-conditioned mutation and the fixture whose violation it
    hides. Nothing here is a result: the exponent it is conditioned at, the
    box, and the bound. The truth and both outcomes are computed.

    ``verdict_obstruction`` is None when the missed violation rides all the way
    out as a false VERIFIED, and otherwise names WHY it does not — and the test
    then measures the obstruction instead of taking the sentence's word for it.
    """

    mutation: str
    exponent: float
    box: tuple[float, float]
    bound: float
    verdict_obstruction: str | None = None


_SHAPE_CONDITIONED_CATCHES = (
    _ShapeConditionedCatch(
        mutation="emit-rational-wrong-only-past-the-second-element",
        exponent=RAT_DENOMINATOR_EXP, box=RAT_DENOMINATOR_BOX, bound=1.9,
    ),
    _ShapeConditionedCatch(
        mutation="emit-rational-wrong-only-past-the-second-element-at-q-two",
        exponent=RAT_EXP, box=RAT_DENOMINATOR_BOX, bound=7.9,
    ),
    _ShapeConditionedCatch(
        mutation="emit-integer-wrong-only-past-the-second-element-at-degree-five",
        exponent=INT_HIGH_EXP, box=NEG_EXP_BOX, bound=10.0,
        verdict_obstruction=(
            "the VACUITY WIDEN RE-CHECK, and it is a property of the branch "
            "rather than of this fixture. `inputs-only` widens every declared "
            "box to (-inf, inf) and re-runs the pipeline; there the RATIONAL "
            "branch declines (a fractional exponent on a negative base has no "
            "faithful Real encoding) so its re-check is recorded UNCHECKED and "
            "the false VERIFIED stands, while the INTEGER branch still emits — "
            "and an emission mutation on an UNBOUNDED box is satisfiable at "
            "points the real program does not violate, which is exactly what "
            "`solvers._require_valid_refutation` exists to refuse. `check` "
            "therefore raises `EmissionInfidelityError`. That is a CATCH, but "
            "by stelling's own emission-fidelity guard rather than by a false "
            "VERIFIED, so this branch's demonstration stops at the escalation "
            "outcome — which is the layer this whole file measures at anyway, "
            "for the reason the module docstring gives"
        ),
    ),
)


def _exact_power(value: Fraction, exponent: float) -> Fraction:
    """``value ** exponent`` in EXACT rational arithmetic, or an error.

    The ``q``-th root is GUESSED in float and then CERTIFIED by raising it
    back — ``root**q == value`` or this raises — so no approximation ever
    reaches a comparison. `81^(1/4)` is 3 because `3**4 == 81`, and a fixture
    whose extremum has no exact rational root cannot silently be compared
    against its bound in binary64 the way this file exists to refuse."""
    frac = Fraction(exponent)
    p, q = frac.numerator, frac.denominator
    root = Fraction(round(float(value) ** (1.0 / q)))
    if root**q != value:
        raise AssertionError(
            f"{value}^(1/{q}) is not rational, so this fixture's truth cannot "
            f"be decided exactly"
        )
    return root**p


def _shape_conditioned_harness(item):
    """Three elements, and a claim about the difference of the last two —
    false only because element 2 is a DIFFERENT value from element 1, which is
    what an element-conditioned emission wrongness destroys."""
    lo, hi = item.box

    def h():
        x = any_array((3,), "float64", (lo, hi))
        r = jnp.power(x, item.exponent)
        return (assert_(r[2] - r[1] <= item.bound),)

    return h


@pytest.mark.parametrize(
    "item", _SHAPE_CONDITIONED_CATCHES, ids=lambda i: i.mutation
)
def test_the_shape_conditioned_mutations_are_CAUGHT_by_the_INVARIANCE_gate(item):
    """The element-conditioned mutations, each pinned to the gate that kills it
    and to the missed violation it produces without one.

    The attribution matters more than the catch. What catches these must be the
    gate that closes the shape axis — not a three-element fixture that would
    answer this index and leave the next one open — so the exclusivity is
    asserted: no other gate in the battery sees them, which is exactly the
    statement that every solver-driven fixture here is blind past element 1.

    THERE ARE THREE OF THEM AND THAT IS THE POINT OF THIS ROUND. The first is
    conditioned on the element index alone. The other two are conditioned on
    the element index AND an exponent — `q == 2` on one branch, `exp == 5` on
    the other — with both coordinates inside the sets this file prints as
    driven, and both survived every gate on the tree that had just closed the
    shape axis, because that closure ran ONE hardwired exponent per branch. A
    gate that closes one axis at a point on the other has moved the radius, not
    removed it. The invariance gate now sweeps the driven exponent set, so what
    it closes is the PRODUCT, and `_measured_seam_reach` records that product
    jointly rather than as marginals that cannot see it.

    The arithmetic is RECOMPUTED, never typed: the truth at the box extremum is
    decided in exact `Fraction` arithmetic with the root certified by raising
    it back, and it is checked against the binary64 bound the traced program
    actually carries rather than against a decimal that only resembles it.

    THE MISSED VIOLATION IS READ OFF `escalate`, NOT OFF `check().status`, for
    the reason the module docstring gives at length: a `check` status carries
    the bar and the vacuity machinery on top of the row, and a gate that reads
    it measures those instead. `violated-witness` becoming `discharged` IS the
    missed violation, at the row's own layer. What the verdict then does with
    it is asserted separately and per item, because on the integer branch the
    vacuity re-check intercepts it — which is disclosed in the table above and
    MEASURED here rather than left as a sentence."""
    caught = dict(_report().caught_by)
    assert caught[item.mutation] == ("emission-is-invariant-to-the-array-shape",), (
        f"{item.mutation} must be caught by the invariance gate ALONE — every "
        f"other fixture in this battery drives one or two elements, and that "
        f"blindness is the finding. Measured: {list(caught[item.mutation])}"
    )

    # the violation is REAL: at x = [_, lo, hi] the exact difference exceeds
    # the bound the traced program carries (the binary64, not the decimal)
    lo, hi = item.box
    truth = _exact_power(Fraction(hi), item.exponent) - _exact_power(
        Fraction(lo), item.exponent
    )
    assert truth > Fraction(item.bound), (
        f"{item.mutation}'s fixture is not violated at the extremum: "
        f"{hi}^{item.exponent} - {lo}^{item.exponent} = {truth} <= "
        f"{item.bound}, so there is no missed violation to hide"
    )
    # ...and jax agrees on this target, which is the oracle that matters
    witness = jnp.asarray([lo, lo, hi], jnp.float64)
    executed = float(
        np.asarray(jnp.power(witness, item.exponent)[2]
                   - jnp.power(witness, item.exponent)[1])
    )
    assert executed > item.bound, (item.mutation, executed)

    # the MISSED VIOLATION itself, at the row's own layer
    h = _shape_conditioned_harness(item)
    assert _outcomes(h, BASELINE) == ("violated-witness",)
    assert _outcomes(h, _mutations()[item.mutation]) == ("discharged",), (
        f"{item.mutation} no longer turns a violated obligation into a "
        f"discharged one; it is in the battery to pin the shape measurement "
        f"and has stopped being the thing it pins"
    )

    # ...and what the END-TO-END verdict does with it
    assert _run(h, BASELINE).status == "REFUTED"
    if item.verdict_obstruction is None:
        assert _run(h, _mutations()[item.mutation]).status == "VERIFIED", (
            f"{item.mutation} no longer mints a false VERIFIED, and this item "
            f"declares no obstruction — either the pipeline changed or the "
            f"table above is now wrong about this branch"
        )
    else:
        # the declared obstruction, MEASURED. A named residual that nothing
        # re-derives is the defect class this whole file is about.
        with pytest.raises(EmissionInfidelityError):
            _run(h, _mutations()[item.mutation])


def test_the_odd_denominator_branch_is_UNREACHABLE_and_FAILS_CLOSED():
    """The rational branch's `q`-odd arm, pinned on the standard
    :func:`test_the_integer_dtype_guard_is_UNREACHABLE_through_jax` sets.

    This file used to claim the rational branch covered "`q` even and odd".
    The even half is driven at two denominators; the odd half was never
    reachable and never tested, and it READ as covered, which is the worse of
    the two states — `smt._pow_rational_lines`' docstring presented it as one
    of two live cases and nothing in the tree said otherwise.

    Three measurements, and then the three places the invariant is now
    enforced. UNREACHABILITY first:

    * `obligation.pow_exponent_rational` is ``Fraction`` of the binary64 the
      literal denotes, and every finite binary64 is ``m * 2**e``, so in lowest
      terms ``q`` is a power of two;
    * over the whole 448-pair admitted set that is exactly
      ``{2, 4, 8, 16, 32, 64, 128}``, and ``q == 1`` takes the integer branch;
    * a random sweep finds no odd denominator at all.

    Then FAIL-CLOSED, because unreachable-today is not a guarantee and the
    principled form of "nothing drives this" is not a comment:

    * the DERIVATION refuses to return a non-dyadic rational, so a widening
      cannot happen quietly;
    * ADMISSION declines an odd denominator — that is where a decline belongs,
      and it is measured here with the derivation replaced, which is exactly
      what a widening would be;
    * EMISSION refuses one, because emission cannot return UNKNOWN; it can
      only write a script or refuse, and writing one down an ungauged path is
      the outcome this is for.
    """
    import random

    # -- unreachable ---------------------------------------------------------
    reachable = set()
    for k in range(1, 8):
        q = 1 << k
        for p in range(1, OB.RATIONAL_POW_DEGREE_CAP + 1):
            if Fraction(p, q).denominator != q:
                continue
            frac = OB.pow_exponent_rational(p / q)
            assert frac == Fraction(p, q)
            reachable.add(frac.denominator)
    assert reachable == {2, 4, 8, 16, 32, 64, 128}, sorted(reachable)

    rng = random.Random(0)
    odd = [
        x for x in (rng.uniform(-8.0, 8.0) for _ in range(20_000))
        if Fraction(x).denominator % 2 and Fraction(x).denominator > 1
    ]
    assert odd == [], odd[:3]

    # -- fails closed at the DERIVATION --------------------------------------
    with pytest.raises(ValueError, match="not a power of two"):
        OB.pow_exponent_rational(Fraction(1, 3))

    # -- fails closed at ADMISSION, with the derivation replaced -------------
    # ...which is precisely the shape a widening past dyadics would take, and
    # the only way to reach the arm at all.
    with mock.patch.object(OB, "pow_exponent_rational",
                           lambda _e: Fraction(1, 3)):
        problem = OB.rational_pow_problem(0.5)
        assert problem is not None and "ODD" in problem, problem
        assert _declines_emission(_rat_false_harness()) is True, (
            "an odd denominator reached emission instead of declining — the "
            "row would walk a path no test in this tree drives"
        )

    # -- fails closed at EMISSION --------------------------------------------
    with pytest.raises(ValueError, match="ODD denominator"):
        SM._pow_rational_lines("aux_0", "x0", 1, 3)

    # -- and the guard the arm used to gate is now UNCONDITIONAL -------------
    for q in sorted(reachable):
        lines = SM._pow_rational_lines("aux_0", "x0", 1, q)
        assert lines[1] == "(assert (>= aux_0 0.0))", (q, lines)


# The odd-`q` arm was removed from the CODE last round and left standing in
# three PROSE sites, each of which describes it as a live case the encoding
# handles: an admission comment in `obligation.py` on a path where odd `q`
# declines several lines earlier, `_negative_base_harness`'s docstring, and a
# 0.2.0 regression docstring. This file's own standard is that an untested
# branch which READS as covered is worse than no branch, and a reader who
# trusts any of the three concludes the negative-base guard exists to stop a
# root the row can compute — when in fact `q` is always even there, the
# encoding has no solution at all, and what the guard stops is a trivially
# UNSAT negation coming back as a false VERIFIED.
#
# The two shapes are described rather than quoted, and that is not squeamish:
# the rule applies to THIS file too, so a verbatim example here would be a
# self-inflicted hit. The originals are at `2d0339d`.
#
# WHAT THE PATTERNS MATCHED BEFORE, AND WHY THAT WAS A CLAIM THIS FILE HAD NO
# RIGHT TO MAKE. They required the literal `for odd` or a PARENTHESISED
# `(odd q)`, which is two sentences, not a class — and the docstring below said
# they were written against the class. Five natural rephrasings walked through,
# the first of them the same sentence reworded once. The two shapes are now
# matched in BOTH orders and without the parenthesis, and the corpus that
# forced each widening is asserted positively by the test rather than left to
# the reader's confidence: a pattern set with no positive corpus is a pattern
# set nobody has measured.
_ODD_Q = (
    r"(?:odd[\s-]*(?:`?q`?|denominators?|exponents?)"
    r"|`?q`?\s+(?:is\s+)?odd"
    r"|denominators?\s+(?:is|are)\s+odd)"
)
_A_ROOT = r"(?:real\s+)?(?:(?:cube|square|nth)[\s-]*)?(?:roots?|solutions?)"

_ODD_Q_AS_A_LIVE_CASE = (
    # a real root or a solution attributed to a denominator that is not even,
    # with the root named first and the denominator after it
    re.compile(rf"(?i)\b{_A_ROOT}\b[^.\n]{{0,100}}?\bfor\s+odd\b"),
    # ...and the same attribution written the other way round, which is how
    # four of the five surviving rephrasings were spelled
    re.compile(rf"(?i)\b{_ODD_Q}\b[^.\n]{{0,100}}?\b{_A_ROOT}\b"),
    # a denominator that is not even, named as a case the encoding handles.
    # The parentheses are gone — the unparenthesised spelling was one of the
    # five that walked through — and the verb is a CLASS rather than the
    # single word it was, `computes` included: that is the verb the class is
    # named after, and it is also what found the one live sentence left in the
    # tree when this widened. `smt._pow_rational_lines`' own refusal message
    # attributed a real root to such a denominator; true, and still an
    # attribution, and it now states the guard's contract instead. An
    # exemption for the one site that fails a rule is how the rule stops
    # meaning anything.
    re.compile(
        rf"(?i)\b(?:models?|handles?|supports?|admits?|computes?|encodes?"
        rf"|covers?)\b[^.\n]{{0,80}}?\b{_ODD_Q}\b"
    ),
)

# The corpus is held as TEMPLATES with the trigger word substituted at runtime,
# because the scan below reads this file too and a verbatim example here would
# be a self-inflicted hit — the same reason the patterns above carry
# descriptions rather than quotations. The test asserts BOTH halves: every
# template fires once filled in, and no template fires as it is written here.
_ODD = "odd"
_ODD_Q_CORPUS = (
    "when `q` is {odd} the row returns the real root",
    "the {odd}-`q` branch returns the real cube root",
    "`q` {odd} admits one real solution",
    "for an {odd} denominator, a real solution exists",
    "models {odd} q",
    "the real root for {odd} q",
    "the encoding models the two cases ({odd} q)",
    "an {odd} q has one real root and no guard",
)


def test_no_source_text_presents_the_ODD_q_ARM_as_a_live_case():
    """The odd-`q` arm is refused at three altitudes and described as live in
    none, checked across the whole tree rather than at the three sites that
    were wrong.

    WHAT THIS CAN AND CANNOT DO, SAID AT THE ALTITUDE THE PATTERNS ACTUALLY
    WORK AT. It is a text check over TEMPLATES: two claim shapes — a real root
    or solution attributed to an odd denominator, in either order, and an odd
    denominator named as what the encoding models, handles or computes. That is
    a wider net than the two sentences it started as, and the corpus below
    pins each widening, but it is NOT a decision procedure for the class. A
    determined rephrasing evades it, and the previous version of this docstring
    claimed otherwise — it said the patterns were "written against that class
    rather than against the sentences" while requiring the literal `for odd` or
    a parenthesised `(odd q)`, which five natural rewordings walked straight
    through. Templates catch drift and typo-level recurrence; they do not
    certify absence.

    WHAT DOES HOLD IS THE ENFORCEMENT, and it is not this test: the arm
    DECLINES at admission, RAISES at the derivation and RAISES at emission, and
    `test_the_odd_denominator_branch_is_UNREACHABLE_and_FAILS_CLOSED` pins all
    three. This test is the prose half of that, and the soundness stake sits in
    the other half.

    The rule covers this file too, which is why the patterns above carry
    descriptions instead of examples and the corpus is held as templates with
    the trigger word substituted at runtime. HISTORICAL QUOTATION IS ALLOWED —
    this module and `smt._pow_rational_lines` both quote the old "`q` even and
    odd" claim to say it was wrong, and a rule that forbade that would delete
    the record of the defect along with the defect.
    """
    # POSITIVE CORPUS FIRST: a pattern set that matches nothing reports a clean
    # tree, and that is indistinguishable from a clean tree until someone
    # checks. Every entry here is a rephrasing that walked through the shipped
    # patterns; the last is the one live sentence the widening found in the
    # tree, in `smt._pow_rational_lines`' refusal message, now reworded.
    for template in _ODD_Q_CORPUS:
        filled = template.format(odd=_ODD)
        assert any(p.search(filled) for p in _ODD_Q_AS_A_LIVE_CASE), (
            f"the odd-`q` patterns do not catch {filled!r} — this corpus is "
            f"what stops them from narrowing back to the two sentences they "
            f"were written from"
        )
        assert not any(p.search(template) for p in _ODD_Q_AS_A_LIVE_CASE), (
            f"the corpus TEMPLATE {template!r} fires as written, so this "
            f"file's own text is a hit and the scan below is about to fail on "
            f"its own corpus — substitute the trigger word at runtime"
        )

    root = pathlib.Path(__file__).resolve().parent.parent
    scanned, hits = 0, []
    for path in sorted(
        list(root.glob("src/stelling/*.py"))
        + list(root.glob("tests/*.py"))
        + list(root.glob("docs/*.md"))
        + list(root.glob("*.md"))
    ):
        text = path.read_text(encoding="utf-8")
        scanned += 1
        for pattern in _ODD_Q_AS_A_LIVE_CASE:
            for m in pattern.finditer(text):
                line = text[: m.start()].count("\n") + 1
                hits.append(f"{path.relative_to(root)}:{line}: {m.group(0)}")
    # ANTI-VACUITY: a glob that matched nothing would report a clean tree
    assert scanned > 20, f"only {scanned} files scanned — the globs are wrong"
    assert any(
        "pow_exponent_rational" in p.read_text(encoding="utf-8")
        for p in root.glob("src/stelling/*.py")
    ), "the scan did not reach the module that owns the arm"
    assert not hits, (
        "the odd-`q` arm is described as a live case, on a path where it "
        "DECLINES at admission, RAISES at derivation and RAISES at emission:\n  "
        + "\n  ".join(hits)
    )


def test_the_integer_dtype_guard_is_UNREACHABLE_through_jax():
    """`pow` is in ``obligation._INT_OVERFLOW_EMITTED`` and NOTHING this
    gauge can trace reaches that guard — so it is recorded UNCOVERED, and the
    unreachability is pinned rather than asserted in prose.

    Three measurements on jax 0.11.0, all three of which must hold for the
    disclosure to stay true:

    * ``lax.pow`` REFUSES an integer operand outright, so there is no direct
      route at all;
    * ``jnp.power(int32, 2.0)`` converts to float64 FIRST, so the `pow`
      equation's operand dtype is never the integer one;
    * ``jnp.power(int32, 2)`` binds ``integer_pow``, a different row with its
      own guard.

    If a jax release ever admits an integer `pow`, this test fails and the
    module docstring's UNCOVERED paragraph has to be replaced by a gate — which
    is the point of pinning it. The guard itself is NOT removed on the strength
    of this: an unreachable guard costs nothing and the emission set is not the
    only door into `smt.emit`."""
    import jax.lax as lax

    with pytest.raises(TypeError, match="pow does not accept dtype"):
        jax.make_jaxpr(lambda b: lax.pow(b, 2.0))(jnp.int32(3))

    converted = jax.make_jaxpr(lambda b: jnp.power(b, 2.0))(jnp.int32(3))
    rows = [(str(e.primitive), tuple(str(v.aval.dtype) for v in e.invars))
            for e in converted.eqns]
    assert rows[0][0] == "convert_element_type", rows
    assert [r for r in rows if r[0] == "pow"] == [("pow", ("float64",) * 2)], rows

    integral = jax.make_jaxpr(lambda b: jnp.power(b, 2))(jnp.int32(3))
    assert [str(e.primitive) for e in integral.eqns] == ["integer_pow"]


def test_the_row_has_the_named_seams_a_gauge_needs():
    """The three seams this battery mutates, asserted to exist and to be the
    ones the emission actually calls. A seam nothing routes through is a
    mutation that changes no bytes and a battery that measures nothing."""
    for name in ("_pow_integer_body", "_pow_rational_lines", "_pow_aux_name"):
        assert hasattr(SM, name), name
    import inspect

    src = inspect.getsource(SM.emit)
    for name in ("_pow_integer_body", "_pow_rational_lines", "_pow_aux_name"):
        assert name in src, f"{name} is not called by smt.emit"


def test_the_pow_seams_do_not_move_the_integer_pow_row():
    """The property `_square_body` was extracted to preserve, asserted for
    `pow`: a mutation of this row's seam must leave `integer_pow`'s emitted
    text byte-identical, or a catch here is not attributable to `pow`."""
    def h():
        x = any_array((), "float64", (-2.0, 3.0))
        return (assert_(x ** 3 - x <= INT_FALSE_BOUND),)

    closed = trace(h)
    assert "integer_pow" in {str(e.primitive) for e in closed.jaxpr.eqns}, (
        "this fixture no longer binds `integer_pow`; it cannot show the "
        "independence it claims"
    )
    sl = _slice_of(h)
    assert sl is not None
    before = SM.emit(sl, "z3", TIMEOUT_MS).text
    for mutation in ("emit-integer-off-by-one", "emit-rational-sides-swapped",
                     "emit-rational-aux-shared-across-elements"):
        with _patched(_mutations()[mutation]):
            assert SM.emit(sl, "z3", TIMEOUT_MS).text == before, (
                f"{mutation} moved the `integer_pow` row's emitted text — the "
                f"two rows share a seam and this battery is measuring both"
            )


def test_every_gate_passes_the_baseline_and_catches_its_battery():
    report = _report()
    print("\n" + report.render())
    caught = dict(report.caught_by)
    for name in _mutations():
        assert caught[name], f"{name} survived every gate — a measured hole"
    # the attributions the battery was built to measure, named rather than
    # left to the aggregate
    assert "refutes-the-false-integer-property" in caught[
        "row-absent-from-the-emission-set"]
    assert "fragment-is-nonlinear" in caught["fragment-claims-linear"]
    assert "non-dyadic-exponent-declines" in caught[
        "exponent-rationalised-to-a-nearby-fraction"]
    assert "interval-containment-eager-and-jit" in caught["transfer-is-the-base"]
    assert "discharges-the-true-rational-lower-bound" in caught[
        "emit-rational-root-guard-dropped"]
    assert "refutes-the-false-rational-property" in caught[
        "emit-rational-denominator-off-by-one"]
    assert "refutes-the-false-vector-property" in caught[
        "emit-rational-aux-shared-across-elements"]
    # the well-formed sharing, whose catch is a MISSED VIOLATION rather than a
    # refused script. It used to be caught by the vector gate ALONE and the
    # exclusivity carried the claim "the scalar fixtures are blind to it"; the
    # shape-INVARIANCE gate now sees it too, because declaring the auxiliary
    # for element 0 and not for element 1 is a per-element difference in the
    # emitted text and that is exactly what the invariance gate is. So the
    # claim is asserted DIRECTLY instead of through the exclusivity, which is
    # what the exclusivity was standing in for and is stronger than it:
    assert set(caught["emit-rational-one-aux-for-two-elements"]) == {
        "refutes-the-false-vector-property",
        "emission-is-invariant-to-the-array-shape",
    }, (
        f"the well-formed aux sharing is caught by "
        f"{list(caught['emit-rational-one-aux-for-two-elements'])}; it must be "
        f"caught by the two ARRAY-driving gates and by nothing else, because "
        f"those two are the only gates in this battery that emit more than "
        f"one element and 'the scalar fixtures are blind to it' is the claim"
    )
    assert "discharges-the-negative-exponent-identity" in caught[
        "emit-integer-loses-the-reciprocal"]


_DOC = pathlib.Path(__file__).resolve().parent.parent / "docs" / "gauge-coverage.md"

# whitespace-tolerant: the sentence is prose in a markdown file and gets
# re-wrapped, and a pin that a reflow can break is a pin people delete
_MEASURED_RE = re.compile(
    r"MEASURED:\s+\*\*(?P<muts>\d+)\s+mutations,\s+(?P<surv>\d+)\s+survivors,"
    r"\s+(?P<asym>\d+)\s+face\s+asymmetries,\s+(?P<multi>\d+)\s+caught\s+by"
    r"\s+more\s+than\s+one\s+gate,\s+(?P<single>\d+)\s+caught\s+by\s+exactly"
    r"\s+one\.\*\*"
)
# THE JOINT REACH, AS A MACHINE-CHECKED SENTENCE. The page carried these five
# digits as prose for exactly as long as it took to write them, which is the
# defect the MEASURED line above already exists to prevent — and the joint reach
# is the figure this round is about, so it gets the same treatment rather than a
# weaker one. The shape count is a BACKREFERENCE, so the two halves of the
# sentence cannot disagree about it.
_JOINT_RE = re.compile(
    r"JOINT\s+REACH:\s+\*\*(?P<shapes>\d+)\s+\(element,\s*n_out\)\s+shapes"
    r"\s+x\s+(?P<ints>\d+)\s+integer\s+exponents\s+=\s+(?P<int_tuples>\d+)"
    r"\s+tuples;\s+(?P=shapes)\s+x\s+(?P<pqs>\d+)\s+\(p,\s*q\)\s+pairs"
    r"\s+=\s+(?P<rat_tuples>\d+)\s+tuples\s+.\s+the\s+FULL\s+PRODUCT\s+in"
    r"\s+both\s+cases\.\*\*"
)
_SINGLE_ROW_RE = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|\s*$")
_BACKTICKED = re.compile(r"`([^`]+)`")

# THE CATCH COLUMN IS A PARSED SET, NOT PROSE, and this regex is the whole of
# what it may contain: backticked gate names, comma-separated, optionally
# followed by the word ALONE. No connective, no hedge, no count.
#
# The predecessor accepted any cell that NAMED a real catcher, which made
# `and most others`, `and others` and `and nearly every other gate` all
# unchecked — measured totals behind those ranged from 4 of 21 to 18 of 21 —
# and, worse, made the naming optional. Deleting one gate name from the
# `emit-integer-loses-the-reciprocal` row left the test green while deleting
# exactly the fact the page's own narrative argues from (that the widening
# took it off the single-covered list BECAUSE the second-magnitude reciprocal
# gate now sees it). The bare counts that column used to carry were removed
# last round for going stale; what replaced them was a hedge, which is the
# same claim with the digit taken out. The rule now has no policy parameter in
# it: name every catcher.
_CATCH_CELL_RE = re.compile(
    r"^\s*(?P<names>`[^`]+`(?:\s*,\s*`[^`]+`)*)\s*(?P<alone>ALONE)?\s*$"
)


def _doc_block(text, marker):
    return text.split(f"<!-- {marker}: BEGIN -->")[1].split(
        f"<!-- {marker}: END -->")[0]


def _doc_rows(text):
    """The mutation table's rows, as ``name -> (catch cell, prose cell)``.

    A REPEATED MUTATION NAME IS REFUSED HERE, because a dict silently keeps the
    last one. The single-covered table has been explicitly protected against
    exactly this since the round that made its comparison order-insensitive
    (``len(set(documented)) == len(documented)``) and the mutation table was
    not, so a document carrying TWO rows for one mutation — the first denying
    the measurement, the second agreeing with it — read as a clean table to
    every assertion downstream, including the one that compares the row order
    against the battery, because the dict was the right LENGTH. A reader sees
    both rows."""
    rows = {}
    for ln in _doc_block(text, "mutation-table").splitlines():
        if not ln.startswith("| `"):
            continue
        cells = ln.split("|")
        name = cells[1].strip().strip("`")
        if name in rows:
            raise AssertionError(
                f"{_DOC.name}'s mutation table has more than one row for "
                f"`{name}` — a reader sees both and only the last is checked"
            )
        rows[name] = (cells[2], cells[3])
    return rows


def _doc_row(text, mutation):
    """One mutation's whole row, for a test that needs to read its prose."""
    catch, prose = _doc_rows(text)[mutation]
    return catch + "|" + prose


# Any `A^B = C` written anywhere on the page is an arithmetic CLAIM and is
# checkable, so it is checked. This is the general form of the defect
# `_rat_denominator_false_harness`'s docstring had: a number in prose that
# nothing recomputes. It is deliberately not tied to the mutation table — the
# same rule covers the narrative paragraphs, which is where `2^6 = 64` and
# `4^(3/2) = 8` live.
_ARITH_CLAIM_RE = re.compile(
    r"(?<![\w.])(?P<base>\d+)\s*(?:\^|\*\*)\s*\(?"
    r"(?P<p>-?\d+)(?:\s*/\s*(?P<q>\d+))?\)?"
    r"\s*=\s*(?P<val>-?\d+(?:\.\d+)?)(?![\w.])"
)


def _arithmetic_claims(text):
    """Every ``A^B = C`` in `text`, with a verdict decided EXACTLY.

    ``C == A**(p/q)`` is decided as ``C**q == A**p`` so no root is taken: the
    claims on this page include irrational neighbours and a float comparison
    is not the standard the rest of the file holds itself to.

    RAISING TO ``q`` LOSES THE SIGN WHEN ``q`` IS EVEN, which made the one
    arithmetic error this page most needs to reject the one it accepted:
    ``81^(1/4) = -3`` satisfies ``(-3)**4 == 81**1`` and was recorded TRUE. The
    row's even-``q`` guard exists precisely to pick the NON-NEGATIVE root — it
    is the branch jax computes and one of the three claims
    `smt._pow_rational_lines` makes — so a page that spelled the negative one
    would be contradicting the encoding it documents. An even ``q`` therefore
    requires a non-negative value as well as the exact power. Odd ``q``
    (integer exponents, ``q == 1``) needs no such condition: the base group is
    ``\\d+``, so the base is non-negative and an odd power already fixes the
    sign."""
    out = []
    for m in _ARITH_CLAIM_RE.finditer(text):
        base = Fraction(m.group("base"))
        p = int(m.group("p"))
        q = int(m.group("q") or 1)
        val = Fraction(m.group("val"))
        exact = val**q == base**p and (val >= 0 or q % 2 == 1)
        out.append((m.group(0), exact))
    return out


def test_the_documented_coverage_figures_are_the_MEASURED_ones():
    """`docs/gauge-coverage.md`'s figures for this row, RECOMPUTED.

    The page said "Sixteen of the seventeen are caught by more than one gate"
    while the table printed directly beneath it listed five single-covered
    entries and the measurement said six. Every other cell on the page
    re-derived exactly; only the summary sentence was wrong, and it was wrong
    in the direction that made the gauge look stronger — the page's own
    premise, *"a gauge with one single-covered mutation is one edit from a
    hole"*, was true six times over while the prose said once.

    Correcting the digit would have left the class untouched, and the widening
    in this round changes the counts again. So the page's figures are parsed
    out of it and compared against a live gauging run, and the single-covered
    SET is compared by name — a count can be right about a set that has
    drifted, and the names are what a reader acts on.

    WHAT THE PREDECESSOR OF THIS TEST STILL ACCEPTED, replayed offline by the
    audit and now each a case below:

    * a `caught by` cell that named ONE of a row's two catchers, deleting
      exactly the fact the page's narrative argues from — the cell must now
      name the COMPLETE catch set;
    * `and most others` / `and others` / `and nearly every other gate`, with
      measured totals behind them from 4 of 21 to 18 of 21 — no cell may
      contain prose at all;
    * a `what the catch looks like` cell made factually false — every `A^B =
      C` in the section is now decided exactly;
    * and it REJECTED a correct document whose single-covered rows had been
      alphabetised, because it compared an ordered tuple against an ordered
      tuple. Order is not a claim; the pairs are compared as a set.
    """
    report = _report()
    text = _DOC.read_text(encoding="utf-8")

    measured = _MEASURED_RE.search(text)
    assert measured is not None, (
        f"{_DOC.name} no longer carries the machine-checked MEASURED line "
        f"(look for the '<!-- gauge-figures: pow -->' marker) — the figures "
        f"went back to being prose nothing recomputes"
    )
    single = single_covered(report)
    survivors = [n for n, gates in report.caught_by if not gates]
    multi = [n for n, gates in report.caught_by if len(gates) > 1]
    for field, value in (
        ("muts", len(report.caught_by)),
        ("surv", len(survivors)),
        ("asym", len(report.asymmetries)),
        ("multi", len(multi)),
        ("single", len(single)),
    ):
        assert int(measured.group(field)) == value, (
            f"{_DOC.name} says {field}={measured.group(field)}; the battery "
            f"measures {value}. Regenerate the sentence from a run "
            f"(python tests/test_pow_row_gauge_jax.py), do not retype it"
        )
    # the arithmetic, confirmed rather than assumed: these three partition the
    # battery, which is the property that made the old sentence checkable at
    # all and the one nobody checked
    assert len(single) + len(multi) + len(survivors) == len(report.caught_by)

    # -- and the JOINT REACH sentence, on the same footing as the one above ---
    #
    # It is the figure this round is about, so it is recomputed and not typed.
    # A page that prints "the FULL PRODUCT" while the measurement is 24 of 84
    # is the exact shape of the claim the three marginals used to license.
    reach = _measured_seam_reach()
    joint = _JOINT_RE.search(text)
    assert joint is not None, (
        f"{_DOC.name} no longer carries the machine-checked JOINT REACH line "
        f"(look for the '<!-- gauge-joint-reach: pow -->' marker) — the joint "
        f"figures went back to being prose nothing recomputes"
    )
    for field, value in (
        ("shapes", len(reach.shapes)),
        ("ints", len(reach.integer_exponents)),
        ("int_tuples", len(reach.integer)),
        ("pqs", len(reach.rational_pq)),
        ("rat_tuples", len(reach.rational)),
    ):
        assert int(joint.group(field)) == value, (
            f"{_DOC.name}'s JOINT REACH line says {field}="
            f"{joint.group(field)}; the battery measures {value}"
        )
    # ...and the words "the FULL PRODUCT" are themselves a claim
    assert reach.integer == DRIVEN_INTEGER_JOINT, _render_joint(
        f"{_DOC.name} claims the FULL PRODUCT; the integer branch's reach",
        reach.integer, DRIVEN_INTEGER_JOINT, len(DRIVEN_INTEGER_EXPONENTS),
    )
    assert reach.rational == DRIVEN_RATIONAL_JOINT, _render_joint(
        f"{_DOC.name} claims the FULL PRODUCT; the rational branch's reach",
        reach.rational, DRIVEN_RATIONAL_JOINT, len(DRIVEN_RATIONAL_PQ),
    )

    documented = tuple(
        m.groups()
        for m in (
            _SINGLE_ROW_RE.match(ln)
            for ln in _doc_block(text, "single-covered").splitlines()
        )
        if m
    )
    # ORDER IS NOT A CLAIM. The predecessor compared two ordered tuples, so a
    # document listing the SAME six pairs alphabetised was rejected — a test
    # that fails on a correct document is a test people stop believing. The
    # pairs are the claim, and duplicates are excluded separately so that
    # comparing as a set cannot hide a repeated row.
    assert len(set(documented)) == len(documented), (
        f"{_DOC.name}'s single-covered table repeats a row: {documented}"
    )
    assert set(documented) == set(single), (
        f"{_DOC.name}'s single-covered table is\n  "
        + "\n  ".join(f"{n} -> {g}" for n, g in documented)
        + "\nbut the battery measures\n  "
        + "\n  ".join(f"{n} -> {g}" for n, g in single)
    )
    # ANTI-VACUITY: an empty table would compare equal to an empty measurement
    assert single, (
        "no mutation is single-covered, which would be a real improvement — "
        "and this test would then be comparing two empty sets, so say so "
        "here and delete the table rather than leaving it as a green nothing"
    )

    # -- and the WHOLE mutation table, not only the summary above ------------
    #
    # The summary sentence was the cell that was wrong, but every "11 gates"
    # and "(+5)" beside it was the same kind of claim and five of them had
    # gone stale too. Deleting the counts left a HEDGE in their place, which
    # is the same claim with the digit taken out and nothing recomputing it.
    # So the catch column is now a parsed SET compared for equality: complete,
    # prose-free, order-insensitive, with ALONE derived rather than trusted.
    caught = dict(report.caught_by)
    rows = _doc_rows(text)
    assert list(rows) == [name for name, _ in report.caught_by], (
        f"{_DOC.name}'s mutation table lists a different battery, or lists it "
        f"in a different order, than the one that ran"
    )
    for name, (cell, prose) in rows.items():
        shape = _CATCH_CELL_RE.match(cell)
        assert shape is not None, (
            f"{_DOC.name}'s `caught by` cell for {name} is\n    {cell.strip()}\n"
            f"which is not a catch SET. That column may contain backticked "
            f"gate names, commas and the word ALONE, and nothing else — no "
            f"'and others', no count, no connective. Every hedge that column "
            f"has ever carried was unchecked, and the totals behind them ran "
            f"from 4 of 21 to 18 of 21"
        )
        named = _BACKTICKED.findall(shape.group("names"))
        assert len(set(named)) == len(named), f"{name}: a gate named twice"
        assert set(named) == set(caught[name]), (
            f"{_DOC.name} says {name} is caught by {named}; the battery "
            f"measures {list(caught[name])}. The cell must name the COMPLETE "
            f"catch set — naming a subset is how a row lost the second gate "
            f"the page's own narrative argues from, with this test green"
        )
        assert bool(shape.group("alone")) == (len(caught[name]) == 1), (
            f"{_DOC.name}'s ALONE marking on {name} disagrees with the "
            f"measurement ({len(caught[name])} catchers) — the exclusivity is "
            f"the claim, and it is what says the other fixtures are blind"
        )
        assert prose.strip(), f"{name}: no 'what the catch looks like' cell"

    # -- the THIRD column, which nothing checked at all ----------------------
    #
    # It cannot be checked as prose, but the arithmetic in it can be, and the
    # arithmetic is what a reader acts on: the audit falsified a whole cell
    # (`2^2 = 5`, `x^(7/9)`, `aux^999`) and this test passed. Every `A^B = C`
    # is now decided exactly, WHEREVER IT APPEARS — and that used to be a
    # sentence rather than a scan. The scan was `text.split("## The pow row's
    # gauge")[1]`, i.e. one section, while the docstring above claimed the
    # page; latent only because no arithmetic lived past that section. A claim
    # this file makes about its own reach is exactly the class it exists to
    # refuse, so the scan is the whole document now: the same rule covers the
    # narrative paragraphs, and the CONDITIONAL rows get their emitted form and
    # cap recomputed from the live mutation by
    # `test_the_conditional_mutations_CAP_the_value_below_the_bound_RECOMPUTED`.
    claims = _arithmetic_claims(text)
    bad = [c for c, ok in claims if not ok]
    assert not bad, (
        f"{_DOC.name} states arithmetic that is FALSE: {bad}"
    )
    # ANTI-VACUITY, and it is the POW SECTION that has to carry it: widening
    # the scan to the page must not let this row's own third column go empty
    # while some other row's arithmetic keeps the check looking alive.
    section = text.split("## The `pow` row's gauge")[1].split("\n## ")[0]
    assert _arithmetic_claims(section), (
        f"{_DOC.name}'s pow section states no arithmetic at all, so this "
        f"check is comparing nothing — it is the third column's only pin"
    )


def test_the_two_faces_are_gauged_and_their_asymmetry_is_visible():
    """A survivor count cannot see a one-face regression ("caught" is
    disjunctive). The transfer-face mutation must be caught by the transfer
    gate and ADMITTED by at least one emission gate — that disagreement is the
    signal, and its absence would mean this gauge has one face."""
    report = _report()
    asym = {name: (c, a) for name, c, a in report.asymmetries}
    assert "transfer-is-the-base" in asym, report.render()
    caught, admitted = asym["transfer-is-the-base"]
    assert "interval-containment-eager-and-jit" in caught
    assert admitted, (
        "every gate caught the transfer mutation, so the two faces are not "
        "visibly independent here"
    )


def test_ieee_mode_is_exercised_as_a_decline_and_says_so():
    """Both semantics modes are driven. `ieee` has nothing else to drive here:
    `escalate` refuses every ieee propagation whatever it contains, so no ieee
    `pow` EMISSION exists to gauge. The ieee TRANSFER rides a declared libm
    accuracy budget and is `tests/test_libm_budget.py`'s subject, not this
    file's."""
    closed = trace(_int_harness(INT_FALSE_BOUND))
    p = propagate(closed, semantics="ieee")
    assert p.semantics == "ieee"
    e = escalate(closed, p, SolverConfig(timeout_ms=TIMEOUT_MS))
    assert not any(r.invocations for r in e.records)


def test_out_of_scope_affine_is_recorded_as_a_decline_not_gauged():
    """OUT OF SCOPE, stated rather than left implicit: `pow` is not in
    AFFINE_SUPPORTED, so the refinement declines. Sound (it decides nothing)
    and named here so nobody reads this gauge as covering it."""
    assert "pow" not in AF.AFFINE_SUPPORTED


def test_in_process_stubs_are_destroyed_and_the_registries_restored():
    """No stubbed verdict is recorded as a verdict. Every mutation above is a
    context-managed in-process patch; this asserts the live registries survive
    the whole battery unchanged.

    DELIBERATELY NOT :func:`_report`. Its subject is a fresh run's effect on
    the live registries, so a cached report would make it assert that nothing
    happened — and this is also the one assertion that would catch a shared
    report papering over cross-run contamination, which is why the sharing is
    safe everywhere else."""

    def snapshot():
        return (
            frozenset(OB._SUPPORTED),
            frozenset(OB._INT_OVERFLOW_EMITTED),
            P.TRANSFERS["pow"],
            SM._pow_integer_body,
            SM._pow_rational_lines,
            SM._pow_aux_name,
            OB._exact_rational_power,
            OB.pow_exponent_rational,
            OB.rational_pow_problem,
            OB._Slicer._fragment,
        )

    before = snapshot()
    gauge(BASELINE, GATES, _mutations(), residual=RESIDUAL, scope=SCOPE)
    assert snapshot() == before


def test_the_reading_renders_and_names_its_scope():
    reading = measure()
    text = reading.render()
    print("\n" + text)
    assert "SCOPE" in text and "Does NOT drive" in text
    assert reading.tree.endswith("stelling/__init__.py")
    outcomes = dict((s, o) for s, o, _ in reading.stages)
    assert outcomes["emit-integer"] == "REACH"
    assert outcomes["emit-rational"] == "REACH"
    assert outcomes["admission-non-dyadic"] == "DECLINE"
    # the arity is IN the reading, not only in the SCOPE prose the reading
    # quotes: a reader who takes this page as the record gets the measurement
    assert outcomes["driven-arity"] == "REACH"
    (arity,) = [d for s, _, d in reading.stages if s == "driven-arity"]
    assert str(list(DRIVEN_INTEGER_EXPONENTS)) in arity, arity
    assert str([list(pq) for pq in DRIVEN_RATIONAL_PQ]) in arity, arity
    assert str([list(a) for a in DRIVEN_AUX_ELEMENTS]) in arity, arity
    # ...and the JOINT reach, which is the half the three marginals above do
    # NOT determine — a reading that printed only them would be the same
    # incomplete record the SCOPE paragraph was
    for declared, n in (
        (DRIVEN_INTEGER_JOINT, len(DRIVEN_INTEGER_EXPONENTS)),
        (DRIVEN_RATIONAL_JOINT, len(DRIVEN_RATIONAL_PQ)),
    ):
        assert f"x {n} exponent(s) = {len(declared)} tuples" in arity, arity
    assert "FULL product" in arity, arity


def doc_blocks() -> str:
    """The two machine-checked blocks of `docs/gauge-coverage.md`, RENDERED.

    The catch column has to name every catching gate, and a complete list of
    up to twenty names per row is not something anyone should be retyping —
    the hedge that stood there existed because the alternative was manual. So
    the column is generated: run this file, paste the blocks. The third column
    is prose and stays in the page, which is why only the first two cells of
    each row are emitted here and the paste is a MERGE rather than an
    overwrite; the test is what confirms the merge.
    """
    report = _report()
    caught = dict(report.caught_by)
    single = single_covered(report)
    survivors = [n for n, gates in report.caught_by if not gates]
    multi = [n for n, gates in report.caught_by if len(gates) > 1]
    out = [
        "<!-- gauge-figures: pow -->",
        f"MEASURED: **{len(report.caught_by)} mutations, {len(survivors)} "
        f"survivors, {len(report.asymmetries)} face asymmetries, "
        f"{len(multi)} caught by more",
        f"than one gate, {len(single)} caught by exactly one.**",
        "",
        "<!-- single-covered: BEGIN -->",
        "| single-covered mutation | its only gate |",
        "|---|---|",
    ]
    out += [f"| `{n}` | `{g}` |" for n, g in single]
    out += [
        "<!-- single-covered: END -->",
        "",
        "<!-- mutation-table: BEGIN -->",
        "| mutation | caught by | what the catch looks like |",
        "|---|---|---|",
    ]
    for name, gates in report.caught_by:
        cell = ", ".join(f"`{g}`" for g in gates)
        if len(gates) == 1:
            cell += " ALONE"
        out.append(f"| `{name}` | {cell} | <<< KEEP THE PAGE'S PROSE >>> |")
    out.append("<!-- mutation-table: END -->")
    return "\n".join(out)


if __name__ == "__main__":  # the runnable-reading entry point
    import sys

    jax.config.update("jax_enable_x64", True)
    if "--doc-blocks" in sys.argv:
        print(doc_blocks())
    else:
        print(measure().render())
