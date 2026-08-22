# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Outward-rounded interval arithmetic over IEEE-754 doubles. Zero-dep.

This is the one module where paranoia is the design: **a rounding bug here
is a false VERIFIED**, which is the project's own thesis defect. The rules:

* Every endpoint of an **ℝ-mode** operation is rounded **OUTWARD**, and
  that — not a width — is this module's one universal claim: under ℝ the
  exact real result of the operation lies inside the returned bracket,
  always. It is what the stamp ``interval/f64/outward-1ulp`` names, a
  DIRECTION and a worst case, and it is the invariant a new ℝ-mode
  operation added to this module has to preserve.

  **ℝ-mode is the scoping word and it is load-bearing.** The ``ieee_*``
  kernels, ``ieee_fma_hull`` and ``subnormal_haze``/``subnormal_haze_fmt``
  serve ``semantics="ieee"``, where the thing being bracketed is the FLOAT
  the compiled program computes and not a real, so they do not round
  outward and this claim is not about them — driven:
  ``ieee_add([0.1], [0.2])`` returns a bracket that EXCLUDES the exact real
  ``0.1 + 0.2``, and ``ieee_sqrt([2])`` one that excludes √2. ``meet`` is
  outside it for a different reason — it is an intersection, so it drops
  points on purpose, and its own docstring opens *"No outward rounding,
  deliberately."* The carve-out used to appear only in the ``ieee_*``
  paragraph at the end of this docstring, which is a long way after the
  word *always*. ``tests/test_interval.py``'s discipline table drives one
  excluded value for every one of these.

  **How TIGHT the bracket is is per-operation and is NOT universal**, so
  the scope below travels with any claim about tightness.

  **Correctly directed-rounded** — the endpoint is the nearest double on
  the outward side of the EXACT real result, computed against ``Fraction``
  and bumped with ``math.nextafter`` **only when the double is on the
  wrong side** (:func:`_exact_down`, :func:`_exact_up`), so a
  representable exact result is returned unwidened. Measured, in float64:
  ``[0.25, 0.5]`` op ``[0.25, 0.5]`` gives exactly ``[0.0625, 0.25]`` for
  ``mul``, ``[0.5, 1.0]`` for ``add``, ``[-0.25, 0.25]`` for ``sub`` and
  ``[0.5, 2.0]`` for ``div``.

  **Not correctly directed-rounded, and here is what that costs.** These
  kinds of operation are outside it — the list is not counted, because a
  count is one more thing to get wrong — and every one of them is still
  sound and every one is wider:

  - the **unconditional one-ulp bump, where the ulp buys something** —
    ``exp``, ``pow``, ``sqrt``, where it is paying for a libm assumption or
    a sqrt-fidelity margin rather than for endpoint representation.
    Measured, on arguments whose exact result is a small integer:
    ``sqrt([4, 4])`` is ``(1.9999999999999998, 2.0000000000000004)``,
    ``exp([0, 0])`` is ``(0.9999999999999999, 1.0000000000000002)`` and
    ``pow_([2, 2], [3, 3])`` is ``(7.999999999999999, 8.000000000000002)``.
    Neither endpoint is the exact value, and nothing here collapses to it.
  - **multi-step reductions**, where each STEP is correctly directed-rounded
    and the total is not — ``reduce_sum`` and ``dot_general``.
    ``reduce_sum`` of ``[1, 2**-53, 2**-53]`` returns
    ``(1.0, 1.0000000000000004)`` while the exact total,
    ``1.0000000000000002``, is representable and is *neither* endpoint —
    the single-op property that one endpoint equals ``fl(R)`` does not
    extend. :func:`reduce_sum`'s own docstring says the same thing.
  - the **⊤ escapes and endpoint conventions** that are not real arithmetic
    at all: a divisor interval containing zero, ``inf/inf``, an infinite
    operand under ``mul``'s ``0·±inf = 0`` rule. Those widen deliberately
    and are named where they happen.
  - **A DEBT, and the only entry here that nobody is defending:**
    ``scatter_add_rows`` folds, like ``reduce_sum`` — and unlike
    ``reduce_sum`` its steps are bumped **unconditionally**, so the ulp
    buys nothing but endpoint representation, which is the reason the
    first entry above exists to exclude. Measured, on operands whose exact
    sum is representable: ``add([1], [1])`` is ``(2.0, 2.0)`` while
    ``scatter_add_rows`` of the same two, accumulating row 0 into row 0,
    is ``(1.9999999999999998, 2.0000000000000004)``; and accumulating one
    ``[0, 16]`` update into a ``[0, 0]`` operand gives
    ``(-5e-324, 16.000000000000004)`` where ``reduce_sum`` of the same
    contributions floors at ``0.0``. **So the M16 divergence the next
    paragraph records as fixed reproduces one operation over**:
    ``jnp.sum(x*x)`` keeps its nonnegative floor and
    ``zeros.at[i].add(x*x)`` does not, so a ``boundary_div`` fed the
    scatter form DECLINES on a straddle where the ``reduce_sum`` form
    verifies. The fix is the exact-``Fraction`` route ``mul`` and ``add``
    already take and it is dispatched on its own, because it is a numeric
    change in the soundness-critical module and this is a documentation
    change. **When it lands, delete this entry**; the bullets around it are
    true either way.

  So: *outward* is a claim about the module; *tight* is a claim about one
  operation, and the operation's own docstring is the authority for it.
  **Do not read a tightness claim off this bullet for an operation it does
  not name** — that is the direction the old wording failed in, one level
  up.

  *This bullet has now been wrong in both directions.* It read "computed
  in double precision and then **bumped one ulp outward** … we do not
  attempt tight rounding; we buy soundness with one deliberate ulp of
  slack per operation" — the rule once, and not the rule now, which was
  NARROWER than the code in the safe direction. It was then replaced by
  "**Every** arithmetic endpoint is correctly directed-rounded … when the
  exact result is representable, both endpoints ARE it and nothing is
  bumped", which is BROADER than the code: the measurements above are
  counter-examples to it, and ``reduce_sum`` is a counter-example whose own
  docstring said so all along — :func:`reduce_sum`, which is where a reader
  should be sent. *That pointer read "sixty lines below". It was wrong, and
  a hand-typed POSITION inside the bullet whose whole subject is hand-typed
  numerals is that defect one level further down, so no distance is written
  here at all: the cross-reference resolves and cannot rot.* Nothing
  unsound rested on either wording — what rests on it is every reader's
  model of how much slack a chain of operations accumulates, and a reader
  told the bracket is tighter than it is is the worse of the two errors.

  *And then a THIRD failure, of a different kind: the corrected bullet was
  itself wider than the code.* It said the fold rule covered "any other
  operation that folds" and that nothing merely-representational was left
  in the bump group. ``scatter_add_rows`` — a registered, ``TIER_SOUND``
  transfer — falsified both, in the tree the bullet shipped in. The pin
  that was added with the correction could not have caught it: it drove
  four operations named in this text and asked only that their endpoint
  reprs appear here, so it measured the digits and not the scope.

  The stamp still reads ``interval/f64/outward-1ulp``. That string is a
  published surface and is not changed here; it names the guarantee's
  DIRECTION and its worst case, both of which still hold.
* ``integer_pow`` endpoints take a **tighter** route to the same
  guarantee: a double is a dyadic rational, so ``Fraction(x) ** n`` is the
  EXACT power, and it is bumped one ulp outward only on the side the
  round-to-nearest double actually fell — not at all when the result is
  representable. No libm is involved, so the tier stays ``sound``.
* ``exp`` endpoints assume a **faithfully-rounded libm** (error ≤ 1 ulp)
  and are bumped one ulp outward — the *same* fidelity demotion the
  supply probe's hand brackets carried (``np.nextafter`` around
  ``np.exp``); it is recorded in every verdict that uses this transfer
  (:data:`EXP_LIBM_ASSUMPTION`). ``pow`` (strictly positive base only)
  makes the same demotion around ``math.pow`` at the monotone corners
  (:data:`POW_LIBM_ASSUMPTION`).

  **THAT ASSUMPTION IS ABOUT THIS PROCESS'S libm, WHICH IS THE RIGHT ONE
  HERE AND THE WRONG ONE UNDER ieee.** These brackets are about the TRUE
  REAL value, and CPython's ``math`` module is what computes them, so
  assuming *it* is faithful is exactly the assumption the bracket needs —
  in real mode, where the verdict is about ℝ, nothing more is required.
  Under ``semantics="ieee"`` the verdict is about the float the compiled
  program computes, and that program runs a DIFFERENT ``exp``: measured,
  XLA's float32 ``exp`` is up to 5.5 float32 ulps from the true value and
  its binary64 ``exp`` up to 1.65 (audit 0.2.0 S9, S11). So the ieee
  transfers widen these brackets by a DECLARED per-(op, format) budget
  and decline without one — see
  :class:`stelling.propagate.LibmBudget`. Nothing in this module changes
  for that; the widening happens above it.
* ``sqrt`` is a **correctly-rounded IEEE-754 basic operation** (error ≤
  0.5 ulp, like +, -, *, /), so it carries no libm-fidelity demotion:
  ``math.sqrt`` is bumped one ulp outward, which contains the true real
  root under correct rounding with a full ulp to spare and stays sound
  even on a platform whose sqrt is merely faithfully rounded (≤ 1 ulp).
  It is monotone increasing on its domain ``[0, ∞)``; a below-0 lower
  endpoint is the OBLIGATION ``arg ≥ 0`` failing and raises (the ``pow``
  out-of-domain posture), and the lower endpoint is floored at 0 so the
  ``sqrt(x) ≥ 0`` fact is produced rather than left to an outward bump.
**SEVERAL ENDPOINT DISCIPLINES LIVE HERE — they are listed, not counted.
They are not interchangeable and the call site does not say which one
applies, so the rule is:**

**This is not a census, and must not be read as one.** It names the
operations whose discipline a reader is most likely to guess wrong; the
authority for any single operation is that operation's own docstring, and an
operation added to this module states its discipline THERE. A list here that
someone forgets to extend is exactly how the universal claim above went
wrong, so nothing is inferred from an absence from it.

* **exact-when-representable** — ``add``, ``sub``, ``mul``,
  ``integer_pow``, and ``reduce_sum`` PER ACCUMULATION STEP (see the
  *exact per step, not per result* bullet below, and :func:`reduce_sum`,
  for why the step rule is not a rule about the total). The endpoint is
  computed exactly (``Fraction``; a double is a dyadic rational) and then
  rounded outward ONLY if the exact result
  is not representable. Sound because these are correctly-rounded ops whose
  slack was pure *endpoint-representation* conservatism. (``div`` belongs
  here for its finite, non-zero-straddling case; its two ⊤ escapes — a
  divisor interval containing 0, and ``inf/inf`` — are untouched, which is
  what keeps ``integer_pow``'s negative-exponent use of the zero discipline
  exactly as it was. ``mul`` keeps the bump on the same confinement: an
  infinite endpoint, where ``Fraction`` cannot represent the operand and the
  ``0·±inf = 0`` convention is an endpoint rule rather than real
  arithmetic.)
* **unconditional one-ulp bump, where the ulp buys something** — ``exp``,
  ``pow``, ``sqrt``. **Do not "optimise" these to the rule above**: in
  every one of them the ulp is doing a second job. ``exp``/``pow`` carry
  the faithfully-rounded-libm assumption (error ≤ 1 ulp, stamped into every
  verdict that uses them), and ``sqrt``'s ulp is what keeps it sound on a
  platform whose ``sqrt`` is merely faithfully rounded rather than
  correctly rounded. ``mul`` was converted OUT of this group and its bump
  had not been free: it put the exactly-zero corner of a squared quantity
  BELOW zero, which defeated ``reduce_sum``'s nonnegative clamp and made
  ``x*x`` and ``x**2`` reach different verdicts on the same property (audit
  0.2.0 M16).
* **unconditional one-ulp bump paying for nothing but endpoint
  representation** — ``scatter_add_rows``, and it is the M16 defect
  surviving one operation over rather than a discipline anyone chose. See
  the DEBT entry in the first bullet of this docstring for the
  measurements and for who is fixing it. **When that lands, delete this
  bullet** and read ``scatter_add_rows`` off the one below.
* **exact per step, not per result** — ``reduce_sum`` and ``dot_general``.
  Every step is exact-when-representable and rounds outward only where THAT
  step is inexact, which is sound and is NOT the first discipline: for an
  ``n``-element sum with ``n >= 3`` neither endpoint need be a neighbour of
  the exact total. Measured, in float64: ``reduce_sum`` of
  ``[1, 2**-53, 2**-53]`` is ``(1.0, 1.0000000000000004)`` around an exact
  total of ``1.0000000000000002`` that a double represents exactly.
  *This bullet read "``reduce_sum``, and any other operation that folds".
  ``scatter_add_rows`` folds and its steps are not exact, so the
  generalisation was false where the named operation was true — the same
  shape of error, one level down, as the universal claim this docstring
  opens by retracting.*

* Endpoints may be ``±inf`` (overflow saturates outward; half-infinite
  sets are representable). Interval multiplication uses the ``0·±inf = 0``
  endpoint convention (sound for closed real intervals, as in IEEE 1788).
* Any ``NaN`` endpoint raises: it means the domain was asked something it
  has no sound answer for, and continuing silently is exactly the failure
  mode this project exists to catch. Division by an interval containing
  zero widens to ⊤ (``[-inf, inf]``) rather than raising — sound, and it
  degrades the verdict to UNKNOWN instead of crashing the walk.

Comparisons return **three-valued** boolean intervals encoded on {0.0, 1.0}
endpoints: ``[1,1]`` definitely true, ``[0,0]`` definitely false, ``[0,1]``
unknown. A definite comparison of outward-rounded operands is sound; an
unknown one is reported as such, never guessed.

One addition sits outside the outward-rounded ℝ rules above: the
``ieee_*`` kernels (their own section below) serve ``semantics="ieee"``
propagation with NATIVE binary64 endpoints — no outward rounding, no
``0·∞ = 0`` convention — and route NaN-producing corners into a
``made_nan`` flag instead of ever raising or leaking a NaN endpoint
(:data:`IEEE_ENDPOINT_ASSUMPTION`).
"""

from __future__ import annotations

import itertools
import math
import sys
from fractions import Fraction  # stdlib; the exact-endpoint route's arithmetic
import operator
from dataclasses import dataclass
from typing import NamedTuple

_INF = math.inf
_FMAX = sys.float_info.max  # largest finite double: the outward-saturation endpoint

# The REAL-MODE stamp for the exp/pow brackets. It is a claim about the
# libm of the process computing the bracket (CPython's `math`), which is
# the claim a bracket of the TRUE REAL value needs. Under
# `semantics="ieee"` it is NOT the whole claim — the verdict is about the
# float a different, compiled libm produces — and stamping it alone there
# was audit 0.2.0 S9/S11. The ieee stamp is
# `stelling.propagate.LibmBudget.render`, which names both halves.
EXP_LIBM_ASSUMPTION = (
    "exp endpoints assume a faithfully-rounded libm exp (error <= 1 ulp), "
    "bumped 1 ulp outward — the same demotion as the hand proofs' "
    "np.nextafter brackets"
)

POW_LIBM_ASSUMPTION = (
    "pow endpoints assume a faithfully-rounded libm pow (error <= 1 ulp), "
    "evaluated at the four monotone (base, exponent) corners and bumped "
    "1 ulp outward — the same fidelity demotion as exp's"
)

# The ieee-mode counterpart of the real mode's 0·∞ = 0 convention line:
# under semantics="ieee" the semantic value of an op IS the float result,
# so endpoints are computed with the very float operations the program
# executes and NOT bumped outward (outward rounding brackets the real
# value; the float value is computable). Soundness rests on the
# monotonicity of the fl-rounded basic ops. The claim is qualified inside
# the open subnormal band, where flush-to-zero targets diverge from
# gradual underflow — see SUBNORMAL_INDETERMINACY_ASSUMPTION.
IEEE_ENDPOINT_ASSUMPTION = (
    "ieee endpoint arithmetic is native binary64 round-to-nearest: interval "
    "endpoints are the same float results the traced program computes, with "
    "NO outward rounding (the float value itself is bracketed exactly); "
    "relied on: monotonicity of the fl-rounded basic ops (add, sub, mul, "
    "div, max, min are monotone in each argument after rounding), so box "
    "images are bracketed by endpoint/corner evaluation — qualified inside "
    "the open subnormal band, where results are additionally hulled with 0 "
    "(see the subnormal-indeterminacy assumption)"
)

# Whether the execution target flushes subnormals is device/compiler-
# dependent: measured jax 0.11.0 CPU binary64 flushes (FTZ on results,
# DAZ on operands) in arithmetic, comparisons, and libm — eager and jit
# alike — while strict IEEE-754 keeps gradual underflow. Neither pure
# semantics is right for every target, so ieee mode is sound for BOTH:
# any interval touching the open subnormal band (-MIN_NORMAL, MIN_NORMAL)
# excluding {0} is hulled with 0 (the flushed image joins the gradual
# values already present), and subnormal-band outcomes are therefore
# indeterminate, never definite.
# The reliance ieee mode has always had, disclosed once the three-row
# round made it load-bearing (audit COSMETIC 4). ieee mode models each
# jaxpr EQUATION as the float operation it names, which is a claim about
# the compiler: that it does not reassociate ACROSS equations. The new
# reduce_sum decline draws exactly this contrast — it refuses a reduction
# because the order is free INSIDE one equation, while continuing to model
# an `add` chain whose order the dataflow records — so the contrast is now
# doing work and belongs in the stamp rather than in a comment.
# Measured (jax 0.11.0 CPU, under jit) at a = b = 1e308, c = d = -1e308:
# `(a+b)+(c+d)`, `(a+c)+(b+d)` and `((a+b)+c)+d` return nan, 0.0 and +inf
# respectively — each matching its own jaxpr order, so the reliance holds
# on the measured target. It is an assumption about the compiler either
# way, and an undisclosed one is exactly what ledger L10 forbids.
IEEE_EQUATION_ORDER_ASSUMPTION = (
    "ieee equation-order reliance: ieee mode judges each jaxpr equation as "
    "the binary64 operation it names, which assumes the compiler does not "
    "REASSOCIATE ACROSS equations — a float `add` chain is modelled in the "
    "order the jaxpr's dataflow records it. Verified on the measured target "
    "(jax 0.11.0 CPU, jit: the three association orders of a+b+c+d at "
    "±1e308 return nan / 0.0 / +inf, each matching its own jaxpr order), "
    "but it is a compiler assumption, not a language guarantee. It covers "
    "ORDER only, and order is not the compiler's only freedom over these "
    "equations: CONTRACTION (fusing a multiply and an add into one "
    "fused-multiply-add, rounding once instead of twice) leaves the order "
    "untouched and IS exercised on this target, so it is not assumed away "
    "here — it is MODELLED, by hulling both roundings (see the ieee "
    "contraction assumption). Order freedom INSIDE a single equation is "
    "covered by neither and is declined instead (reduce_sum over >2 "
    "elements, integer_pow beyond y=1)"
)

SUBNORMAL_INDETERMINACY_ASSUMPTION = (
    "subnormal indeterminacy: whether the target flushes subnormals "
    "(FTZ/DAZ) is device/compiler-dependent — measured jax 0.11.0 CPU "
    "binary64 flushes subnormals in arithmetic, comparisons, and libm, "
    "while strict IEEE-754 keeps gradual underflow. ieee-mode intervals "
    "touching the open subnormal band (0 < |x| < 2**-1022) are hulled "
    "with 0, making verdicts sound for both semantics; subnormal-band "
    "outcomes are treated as indeterminate, never definite"
)


# -- the two stamps above, said truthfully about a NARROW-format run ----------
#
# BOTH constants above are binary64 SENTENCES, and both were stamped
# verbatim on float16/bfloat16/float32 verdicts once ieee mode became
# format-parametric (audit 0.2.0 M14). Both are FALSE of such a run:
#
#   * the endpoints of a narrow-format run ARE outward-rounded — that is
#     the whole of `propagate._ieee_round_box` — so they are not "the same
#     float results the traced program computes";
#   * the band applied was the FORMAT's (2**-14 / 2**-126), not 2**-1022.
#
# The `semantics:` stamp line discloses the parametric mode correctly, so
# the two `assumes:` lines contradicted the line above them. These two
# builders say the same things about the formats a run actually contains.
# The binary64-only run keeps its exact original text — it is the case
# those sentences were written for, it is by far the common one, and a
# reworded stamp on an unchanged run would be its own disclosure noise.

_FORMAT_MIN_NORMAL_TEXT = {
    "float16": "2**-14",
    "bfloat16": "2**-126",
    "float32": "2**-126",
    "float64": "2**-1022",
}

# -- WHICH FORMATS THE MEASURED TARGET FLUSHES: the one table ----------------
#
# THE FLUSH IS PER-FORMAT AND IT IS MEASURED, NOT ASSUMED. Driven on this
# target in both x64 cells, eager AND under `jit`, by asking `x > 0` at
# magnitudes strictly inside each format's OWN subnormal band — each stored
# and read back nonzero first, so the question really is the comparison's:
#
#     format    magnitudes asked          `x > 0` reads True   verdict
#     float16   2**-24 … 3.05e-05 (5)     5 of 5, both ways    KEEPS GRADUAL
#                                                              UNDERFLOW
#     bfloat16  9.18e-41 … 1e-39  (4)     0 of 4, both ways    flushes
#     float32   1.4e-45 … 1e-39   (4)     0 of 4, both ways    flushes
#     float64   5e-324 … 1e-310   (3)     0 of 3, both ways    flushes
#
# The control on the float64 row: 1e-300, a NORMAL float64, reads True in the
# same run, so the row is measuring the band and not the dtype.
#
# EVERY PARAMETRIC SENTENCE THIS PROJECT WRITES ABOUT THE FLUSH IS DERIVED
# FROM HERE. The one exception is deliberate and is the common path:
# `SUBNORMAL_INDETERMINACY_ASSUMPTION`, the binary64-only stamp that predates
# the parametric builders, spells the fact out verbatim and is returned
# unchanged for `()` and `("float64",)`. It is held to this table by
# `test_the_flush_SENTENCE_is_the_ieee_stamps_own_sentence`, not by
# derivation, and that is the whole of what "no second spelling" means here.
#
# An earlier draft of this block said "no second spelling of it ANYWHERE",
# which is false of that common path — a false universal written into the
# same commit that withdrew one. The word has now been wrong three times in
# this batch's lineage, which is its own argument for naming the exception
# rather than reaching for the absolute. The reason is
# a defect this table was added to close (0.2.0 B18 fixup): the real-mode
# subnormal tell carried its OWN hard-coded float64 sentence — *"the
# measured one does (jax 0.11.0 CPU reads 5e-324 > 0 as False)"* — and
# stamped it on float16 runs, where it is FALSE and where the `ieee` face of
# the very same run said so correctly. Two faces of one run contradicting
# each other on a measured fact is exactly what one source of truth is for.
#
# A format ABSENT from this table has not been measured. Callers that must
# assert something about the target read :func:`target_flushes_subnormals`
# and get ``None``, which is not ``False``: ieee mode's haze covers both
# behaviours regardless (it HULLS), so an unmeasured format costs the haze
# nothing, while an assertion about the target must simply stay silent.
_TARGET_MEASURED = "jax 0.11.0 CPU"

_FORMAT_TARGET_FLUSHES = {
    "float16": False,
    "bfloat16": True,
    "float32": True,
    "float64": True,
}

if set(_FORMAT_TARGET_FLUSHES) != set(_FORMAT_MIN_NORMAL_TEXT):  # pragma: no cover
    # `raise`, not `assert`: a module-level assert is stripped under -O, and
    # a census that stops being enforced in an optimised deployment is not a
    # census (tests/test_optimize_mode_guards.py).
    raise RuntimeError(
        "the measured-flush table and the band table must describe the "
        "same formats: a format with a band and no measurement would be "
        "hazed with no evidence, and a format with a measurement and no "
        "band could not be hazed at all — "
        f"{sorted(set(_FORMAT_TARGET_FLUSHES) ^ set(_FORMAT_MIN_NORMAL_TEXT))}"
    )

_COUNT_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


def target_flushes_subnormals(dtype: str):
    """Does the MEASURED target flush subnormals of ``dtype``?

    ``True``/``False`` for a measured format, ``None`` for one that is not
    in :data:`_FORMAT_TARGET_FLUSHES`. ``None`` is NOT ``False`` — it is
    "no measurement on file", and a caller asserting something about the
    hardware must treat it as a reason to say nothing.
    """
    return _FORMAT_TARGET_FLUSHES.get(dtype)


def _measured_format_names(flushes: bool) -> str:
    return "/".join(
        sorted(f for f, v in _FORMAT_TARGET_FLUSHES.items() if v is flushes)
    )


def measured_flush_clause() -> str:
    """*"measured jax 0.11.0 CPU, bfloat16/float32/float64 flush subnormals
    in arithmetic, comparisons and libm while float16 keeps gradual
    underflow"* — the one sentence, DERIVED from
    :data:`_FORMAT_TARGET_FLUSHES` rather than written out beside it.

    Used by :func:`subnormal_indeterminacy_assumption` (the `ieee` stamp)
    and by the real-mode subnormal tell in :mod:`stelling.propagate`, so
    the two faces of a run cannot disagree about which formats flush.
    """
    return (
        f"measured {_TARGET_MEASURED}, {_measured_format_names(True)} flush "
        f"subnormals in arithmetic, comparisons and libm while "
        f"{_measured_format_names(False)} keeps gradual underflow"
    )


def _format_list(formats) -> str:
    fs = tuple(formats)
    return ", ".join(fs) if fs else "float64"


def ieee_endpoint_assumption(formats=()) -> str:
    """The endpoint-arithmetic stamp for a run over ``formats``.

    ``formats`` is the set of float format NAMES the query contains. Empty
    or ``("float64",)`` returns :data:`IEEE_ENDPOINT_ASSUMPTION` unchanged.
    """
    fs = tuple(sorted(set(formats) or {"float64"}))
    if fs == ("float64",):
        return IEEE_ENDPOINT_ASSUMPTION
    bands = ", ".join(
        f"{f}: {_FORMAT_MIN_NORMAL_TEXT.get(f, '?')}" for f in fs
    )
    return (
        f"ieee endpoint arithmetic over {_format_list(fs)}: endpoints are "
        f"computed in native binary64 round-to-nearest and then rounded "
        f"OUTWARD onto the target format's own ulp grid (lo down, hi up), "
        f"so an endpoint is a value of that format BRACKETING what the "
        f"traced program computes rather than the float result itself — "
        f"the no-outward-rounding claim holds for binary64 alone, where "
        f"that rounding is the identity; relied on: monotonicity of the "
        f"fl-rounded basic ops (add, sub, mul, div, max, min are monotone "
        f"in each argument after rounding), so box images are bracketed by "
        f"endpoint/corner evaluation — qualified inside each format's OWN "
        f"open subnormal band ({bands}), where results are additionally "
        f"hulled with 0 (see the subnormal-indeterminacy assumption)"
    )


def subnormal_indeterminacy_assumption(formats=()) -> str:
    """The subnormal-band stamp for a run over ``formats``.

    Empty or ``("float64",)`` returns
    :data:`SUBNORMAL_INDETERMINACY_ASSUMPTION` unchanged.
    """
    fs = tuple(sorted(set(formats) or {"float64"}))
    if fs == ("float64",):
        return SUBNORMAL_INDETERMINACY_ASSUMPTION
    bands = ", ".join(
        f"{f}: 0 < |x| < {_FORMAT_MIN_NORMAL_TEXT.get(f, '?')}" for f in fs
    )
    n_measured = _COUNT_WORDS.get(
        len(_FORMAT_TARGET_FLUSHES), str(len(_FORMAT_TARGET_FLUSHES))
    )
    return (
        f"subnormal indeterminacy: whether the target flushes subnormals "
        f"(FTZ/DAZ) is device/compiler-dependent — {measured_flush_clause()}"
        f", and strict IEEE-754 keeps it for all {n_measured}. ieee-mode "
        f"intervals touching the open subnormal band OF THEIR OWN FORMAT "
        f"({bands}) are hulled with 0, making verdicts sound for both "
        f"semantics; subnormal-band outcomes are treated as indeterminate, "
        f"never definite. {_measured_format_names(False)}'s band is wider "
        f"than this target needs, which costs precision and never soundness "
        f"(the haze HULLS, it does not replace)"
    )

# A precision boundary of the mode, disclosed because it is real and
# because a non-green under ieee must be readable against it (the same
# graceful-degradation property the dependency-wall notes carry).
# Measured 2026-07-19 (design/ieee-reexamination.md): the maybe-NaN flag
# is unioned across an array's elements at construction, so a DISCARDED
# component's NaN possibility spreads to a co-located, asserted,
# float-clean one. It compounds with the interval domain's dependency
# loss: a half-infinite declaration recomputed through its own
# coordinates (`c` declared [k, inf), recomputed as `exp(a) - x0` where
# `x0 = exp(a) - c`) loses its lower bound, which makes `0 * inf`
# reachable in a component that no obligation reads. The union is SOUND —
# an over-approximated flag only ever blocks a discharge, never mints
# one — so this costs precision, never correctness: ieee mode may return
# UNKNOWN exactly where the real-mode verdict discharges. The fix
# (per-element flag tracking through array construction) would make such
# obligations discharge and therefore needs the full build-and-audit
# treatment; it is deliberately not a silent precision bump.
IEEE_NAN_HYGIENE_SCOPE = (
    "ieee precision boundary (not a soundness limit): the maybe-NaN flag "
    "is unioned across an array's elements at construction, so a "
    "discarded component's NaN possibility can block a co-located, "
    "asserted, float-clean obligation — compounding with interval "
    "dependency loss on half-infinite declarations recomputed through "
    "their own coordinates. Half-infinite declarations with recomputed "
    "coordinates are therefore OUTSIDE what ieee mode can reproduce: it "
    "may return UNKNOWN where the real-mode verdict discharges. The "
    "union is sound (an over-approximated flag only blocks discharges, "
    "never mints them); per-element tracking is a known fix path, "
    "deliberately unbuilt"
)


# The ieee declines this module owns. All are the SAME defect class,
# and naming it once is the point: the jaxpr records WHICH values an
# equation combines but not in WHAT ORDER the compiler combines them, and
# neither float addition nor float multiplication is associative. Where an
# equation contracts more than one such operation into itself, no
# single-order model is sound for the compiled program, so ieee mode
# declines instead of picking one (the guard rule: an unmodelled behaviour
# disclosed is sound, a mismodelled one is not). Verified by construction,
# not asserted — tests/test_three_rows.py::test_reassociation_* builds the
# counterexamples these reasons cite, so a quoted reason cannot rot.
REDUCE_SUM_IEEE_ORDER_DECLINE = (
    "reduce_sum over {n} elements has no ieee transfer: float addition is "
    "NOT associative and the jaxpr fixes no summation order, so XLA may "
    "evaluate the reduction in any association order and a transfer "
    "modelling one order would be unsound for the compiled program. The "
    "orders are not near-equal either — with a = b = 1e308 and c = d = "
    "-1e308, ((a+b)+(c+d)) is NaN, ((a+c)+(b+d)) is 0.0 and (((a+b)+c)+d) "
    "is +inf: one reduction over finite operands, three association "
    "orders, three qualitatively different results. No all-orders bound "
    "is offered here, so this declines with the gap quoted. Reductions of "
    "at most 2 elements ARE modelled: they perform 0 or 1 additions, and "
    "IEEE addition is commutative, so no association freedom exists"
)

INTEGER_POW_IEEE_SCHEDULE_DECLINE = (
    "integer_pow with exponent y={y} has no ieee transfer: the jaxpr fixes "
    "no evaluation schedule for the power, and the candidate lowerings "
    "disagree in the last ulps — float multiplication is NOT associative "
    "(measured: ((x*x)*x)*x differs from (x*x)*(x*x) for 34% of x in "
    "[0.5, 2.0]), and for negative exponents 1/(x*x) differs from "
    "(1/x)*(1/x) for 53% of the same sample; a correctly-rounded libm pow "
    "is a third distinct answer. A transfer modelling one schedule would "
    "be unsound for the compiled program, so this declines with the gap "
    "quoted. y=0 and y=1 ARE modelled: they perform NO arithmetic (the "
    "empty product and the identity), so no schedule freedom exists"
)

DOT_GENERAL_IEEE_DECLINE = (
    "dot_general has no ieee transfer: the real transfer's soundness rests "
    "on R-associativity of the per-output-element accumulation, and float "
    "addition is NOT associative — the jaxpr fixes no summation order for a "
    "contraction, so a transfer modelling one order would be unsound for the "
    "compiled program (the reduce_sum construction transfers verbatim). "
    "Contraction freedom is the second gap and it is sharper here than "
    "anywhere else in the registry: a dot product is exactly what a backend "
    "fuses into FMAs, so the products need not be rounded before they are "
    "summed, and no taint-hull for that is built. Declines on EVERY form, "
    "including the one-term contraction, because the refusal is about the "
    "argument rather than about how many terms happen to be present."
)

SCATTER_ADD_IEEE_DECLINE = (
    "scatter-add has no ieee transfer: duplicate scatter indices ACCUMULATE "
    "(out[i] = operand[i] + Σ updates[j] over every index row j mapping to "
    "i — the defining semantic; jax.ops.segment_sum exists because of it), "
    "the jaxpr fixes no order for that per-element accumulation, and float "
    "addition is NOT associative — a transfer modelling one association "
    "order would be unsound for the compiled program (the reduce_sum "
    "construction transfers verbatim: one sum over finite operands, three "
    "association orders, NaN vs 0.0 vs +inf). Contraction is the second "
    "compiler freedom over the same equation: an update arriving as a "
    "product could be fused into the accumulate's add, rounding once "
    "instead of twice, and no taint-hull is built for the scatter path. "
    "No all-orders bound is offered here, so EVERY scatter-add form "
    "declines under ieee semantics — including the duplicate-free and "
    "empty-update forms, deliberately: the refusal is the censused floor, "
    "not a per-case judgement"
)


IEEE_ZERO_DIVISOR_TOP = (
    "Under ieee semantics a divisor box that CONTAINS zero divides to ⊤, "
    "including the one-sided-boundary shapes [0, hi] and [lo, 0]. The "
    "0.2.0 boundary-aware tightening is WITHDRAWN here and must not be "
    "re-added: it read [lo, 0] as 'the divisor approaches 0 from below' and "
    "returned a one-signed infinity, but under IEEE the divisor does not "
    "approach zero, it IS zero at that endpoint, and the sign of x/0 comes "
    "from sign(x) XOR signbit(0) — the SIGN BIT OF THE ZERO, which an "
    "interval endpoint cannot carry. +0.0 == 0.0, so +0.0 is a value of "
    "[lo, 0] and -0.0 is a value of [0, hi]; both infinities are attainable "
    "and the box excluded one of them (audit 0.2.0 S10, FALSE VERIFIED in "
    "all four formats). WHICH BOUNDARY IS ZERO IS NOT ENOUGH INFORMATION "
    "TO DECIDE THIS, so no test on the endpoints' positions can repair it — "
    "only a domain that carries the zero's sign could, and this one does "
    "not. ⊤ is not merely the conservative answer here, it is the EXACT "
    "hull: whenever the dividend can be nonzero both ±inf are attained, and "
    "in the only remaining case (dividend exactly [0, 0]) every quotient is "
    "0/0 = NaN or ±0, which already raises the NaN flag. The real-mode "
    ":func:`boundary_div` keeps the tightening and is NOT wrong for this "
    "reason: ℝ has ONE zero and a/0 is undefined there, so the box need "
    "only cover b ≠ 0. The two kernels disagree deliberately."
)


class IntervalError(ArithmeticError):
    """The domain met a value it has no sound treatment for (e.g. NaN)."""


class IndexOutOfBoundsError(IntervalError):
    """An index is out of bounds for **every input the user declared**.

    A subclass of :class:`IntervalError` so it keeps that class' sound
    accounting exactly — the walk still degrades the equation to a noted ⊤,
    still counts it unknown, still refuses to decide anything downstream of
    it. What the subclass adds is the *reason*, which is categorically
    different from every other decline in the tree: a plain
    :class:`IntervalError` says **stelling** has no rule for this form; this
    one says **the program** indexes outside its array on the whole declared
    set, and stelling proved it.

    It is a finding, not an imprecision, and the walk gives it its own note
    so a reader can tell those apart. It never changes a verdict's status:
    an out-of-bounds index does not make an asserted predicate false, and
    manufacturing a REFUTED from it would be claiming something the
    obligations do not say. It can only ever *withhold* — which it does, via
    the ⊤ — so the direction stays safe."""


def _check(x: float) -> float:
    if x != x:  # NaN
        raise IntervalError("NaN endpoint in interval arithmetic")
    return x


def _down(x: float) -> float:
    """Round a computed lower endpoint downward by one ulp (sound bracket)."""
    _check(x)
    return x if x == -_INF else math.nextafter(x, -_INF)


def _up(x: float) -> float:
    """Round a computed upper endpoint upward by one ulp (sound bracket)."""
    _check(x)
    return x if x == _INF else math.nextafter(x, _INF)


def _exactable(*xs: float) -> bool:
    """True when every endpoint is finite, so ``Fraction`` can represent it.

    ``Fraction(inf)`` raises, and the closed-interval ``0 * ±inf = 0``
    convention is an endpoint rule rather than a real-arithmetic one, so the
    exact route is confined to the finite case and the unconditional bump
    stays in force everywhere else.
    """
    return all(x == x and x != _INF and x != -_INF for x in xs)


def _exact_down(exact: Fraction) -> float:
    """The largest double ``<= exact`` — correct directed rounding, not a bump.

    ``float()`` rounds to nearest, which may land above; one step down fixes
    it. When ``exact`` is representable this returns it UNCHANGED, which is
    the whole point: an endpoint computed without rounding needs no slack.
    """
    try:
        f = float(exact)
    except OverflowError:
        # The exact sum is outside the double range. Saturating outward is
        # the sound answer and matches the overflow posture elsewhere.
        return -_INF if exact < 0 else _FMAX
    if f == -_INF:
        return f
    return f if Fraction(f) <= exact else math.nextafter(f, -_INF)


def _exact_up(exact: Fraction) -> float:
    """The smallest double ``>= exact`` — correct directed rounding."""
    try:
        f = float(exact)
    except OverflowError:
        return _INF if exact > 0 else -_FMAX
    if f == _INF:
        return f
    return f if Fraction(f) >= exact else math.nextafter(f, _INF)


def _prod(a: float, b: float) -> float:
    """Endpoint product with the closed-interval convention 0 * ±inf = 0."""
    if (a == 0.0 and (b == _INF or b == -_INF)) or (
        b == 0.0 and (a == _INF or a == -_INF)
    ):
        return 0.0
    return a * b


@dataclass(frozen=True)
class IntervalArray:
    """A box: per-element closed intervals over a fixed shape, flat C-order."""

    shape: tuple[int, ...]
    los: tuple[float, ...]
    his: tuple[float, ...]

    def __post_init__(self) -> None:
        # THE VALIDATED EXTENTS ARE INSTALLED, not merely checked (audit
        # 0.2.0 B8a, item 1). `check_shape` reads every extent ONCE through
        # `__index__` and hands back what it read; writing that back onto
        # the field is what makes `self.shape` — and therefore `.size`,
        # `_size_of`, `_strides` and `_coords`, each of which re-reads it —
        # a plain tuple of plain ints no protocol can move between reads.
        # Same repair, same reason, as `ir.Aval.__post_init__`'s
        # `object.__setattr__(self, "shape", _validate_aval(...))`. Before
        # it, the count below was `n *= d` over the RAW objects: an extent
        # answering 2 to `__index__` and 1 to `__mul__` made this
        # constructor refuse the correct two-element payload for its own
        # shape.
        object.__setattr__(self, "shape", check_shape(self.shape))
        n = 1
        for d in self.shape:
            n *= d
        if not (len(self.los) == len(self.his) == n):
            raise IntervalError(
                f"shape {self.shape} needs {n} elements, got "
                f"{len(self.los)}/{len(self.his)}"
            )
        for lo, hi in zip(self.los, self.his):
            _check(lo)
            _check(hi)
            if lo > hi:
                raise IntervalError(f"empty interval [{lo}, {hi}]")

    @property
    def size(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    def is_scalar(self) -> bool:
        return self.shape == ()


# `ir.ClosedJaxpr.consts` may hold one of these IN PLACE OF A VALUE — "a
# const already handed over as a box, provenance unknown", the form
# `propagate._Propagator.run` binds directly and `SOUNDNESS.md` records
# with its own test. `ir`'s canonicalization door stores only types it is
# closed over (audit 0.2.0 B6 audit 6) and cannot name this one, because
# it may not import anything outside the standard library — so the
# declaration is made from this side, by the module that owns the type.
#
# WHAT THIS CLASS OWES IN RETURN, since `ir` CARRIES it rather than
# canonicalizing it: single-valuedness. It has it — and NOT for the reason
# this comment used to give, which was that `__post_init__` "reads
# `shape`, `los` and `his` at construction, so a later read of any of the
# three is the same read" (audit 0.2.0 B6 audit 8). READING A FIELD
# INSTALLS NOTHING. `__post_init__` above validates and returns; every
# later reader goes back to the attribute, and were that attribute a
# descriptor it would be free to answer differently — which is the whole
# of the finding `ir._canonicalise` exists for, and it is why the sibling
# guards in `ir` INSTALL what they read instead of merely checking it.
#
# What actually makes a later read the same read here is narrower and
# checkable: the three fields are ORDINARY INSTANCE ATTRIBUTES of a frozen
# dataclass — no `property`, no descriptor, no `__getattr__`, so a read is
# a `__dict__` lookup — and frozen-ness stops any later rebinding of one.
# `ir` carries the EXACT registered type and nothing else: a subclass that
# made `los` a property is refused by `ir._canonical` as `_NotCanonical`
# (measured, audit 0.2.0 B6 audit 8), so this claim cannot be inherited
# out from under itself.
#
# `ir._register_stored_type` checks the frozen half and says in its own
# docstring that it does NOT check the descriptor half; that gap is a
# measurement in `tests/test_canonicalization_routes.py`, not a promise.
# If a field of this class ever becomes a property or a cached descriptor,
# this paragraph is what stops being true, and the registration below is
# what has to go with it.
#
# The edge is one-way and adds no dependency: `stelling.ir` is the only
# package module this file imports, and `ir` itself imports nothing
# outside the standard library, so the zero-dependency posture is
# unchanged and there is no cycle.
from stelling import ir as _ir  # noqa: E402  (below the class it registers)

_ir._register_stored_type(IntervalArray)


def _safe_repr(obj) -> str:
    """``repr(obj)``, or a visible placeholder when the object refuses.

    For quoting an object a guard has ALREADY DECIDED TO REFUSE, and for
    nothing else. A guard that turns a malformed value into a quotable
    :class:`IntervalError` must not raise while quoting it: the object it
    is describing is by construction misbehaving, and a ``__repr__`` that
    refuses converts the refusal into exactly the raw escape the guard
    exists to prevent (audit 0.2.0 B6 audit 3, F3 — measured, a
    ``RuntimeError`` out of :func:`check_shape` and out of
    :func:`dot_general_geometry`). The placeholder is visible so a reader
    sees that something could not be read rather than a plausible value.
    (``stelling.obligation._safely`` is the same helper for the same
    reason; this module deliberately imports nothing from the rest of the
    library, so it carries its own four lines rather than acquiring a
    dependency for them.)"""
    try:
        return repr(obj)
    except Exception:  # noqa: BLE001 — the message's own totality
        return "<unreadable>"


def check_shape(shape) -> tuple[int, ...]:
    """Refuse shapes no jax program can carry, through the decline channel,
    and RETURN the extents that survived, normalised to plain ``int``.

    Two predicates, both measured on jax 0.11.0 (fix re-attacks R1/N2):
    every extent must be INTEGRAL (``from_dict`` does not coerce shape
    entry types, so a JSON shape can hold a string — comparing it raised
    a raw TypeError before this guard) and NONNEGATIVE (every concrete
    jax context rejects a negative extent — "shape must have every
    element be nonnegative"; the type is uninhabited, and a box for it
    would be internally incoherent: on (-2,-2) the element-count product
    and the coordinate enumeration disagreed, 4 vs 1 — the
    dropped-addends UNSOUND). Zero extents are LEGAL (jax constructs
    zero-size arrays; do not over-guard). Every box construction routes
    through here (:class:`IntervalArray.__post_init__`), plus the
    pre-construction sites that compute element products first.

    **THE REFUSAL IS TOTAL IN BOTH DIRECTIONS** — audit 0.2.0 B6 audit 3,
    F2 and F3. ``operator.index`` raises whatever ``__index__`` raises,
    and this handler caught only ``TypeError``, so a ``ValueError`` or an
    ``OverflowError`` from an extent left this function raw and left
    ``propagate()`` raw with it — the S12″ two-faces split again, since
    the emission face declined on the same object. An extent that will not
    answer ``__index__`` is a non-integer extent whatever it raises saying
    so. And the message that says so quotes both the shape and the extent
    through :func:`_safe_repr`, because an object that refuses one dunder
    may refuse ``__repr__`` too.

    **AND IT RETURNS WHAT IT VALIDATED, BECAUSE A PREDICATE IS NOT A
    VALUE** — audit 0.2.0 B8a, item 1. This function used to answer only
    "well-formed or not", and every caller that needed the ELEMENT COUNT
    then re-read the raw objects with ``n *= d`` — ``__mul__``, a THIRD
    protocol beside the ``__index__`` validated here and the ``__eq__``
    the shape comparisons use. Measured on ``aabb58d`` with an extent
    whose ``__index__`` answers ``2`` and whose ``__mul__`` is the
    identity: ``point(1.0, (d,))`` returned a ONE-element box for a
    two-element shape, and ``IntervalArray(shape=(d,), los=(1.0, 1.0),
    his=(1.0, 1.0))`` — the CORRECT payload for that shape — was refused
    as *"shape (…) needs 1 elements, got 2/2"*. Two faces, one shape,
    from one unvalidated read.

    This is :func:`stelling.obligation._extents`' and
    :func:`stelling.ir._load_extents`' repair, one module over and for the
    identical reason: read once, hand back what was read, and let no
    caller obtain a count from a protocol no guard validated.
    :meth:`IntervalArray.__post_init__` goes one step further and
    INSTALLS the returned extents, the way :meth:`stelling.ir.Aval.
    __post_init__` does — so ``box.shape`` is a plain ``tuple`` of plain
    ``int`` from construction onward, and every later reader of it
    (:attr:`IntervalArray.size`, :func:`_size_of`, :func:`_coords`,
    :func:`_strides`) is reading a single-valued object rather than
    trusting whatever the caller passed."""
    out: list[int] = []
    for d in shape:
        try:
            k = operator.index(d)
        except Exception:  # noqa: BLE001 — unreadable IS the finding
            raise IntervalError(
                f"shape {_safe_repr(shape)} has a non-integer extent "
                f"{_safe_repr(d)} (malformed IR: from_dict does not coerce "
                f"shape entries)"
            ) from None
        if k < 0:
            raise IntervalError(
                f"shape {_safe_repr(shape)} has a negative extent: no jax "
                f"program constructs such a value (measured: jax rejects "
                f"negative dims in every concrete context)"
            )
        out.append(k)
    return tuple(out)


def point(value: float, shape: tuple[int, ...] = ()) -> IntervalArray:
    """A degenerate (exact) interval; no outward bump — the value itself is
    the set."""
    shape = check_shape(shape)
    n = 1
    for d in shape:
        n *= d
    v = float(value)
    return IntervalArray(shape=shape, los=(v,) * n, his=(v,) * n)


def from_bounds(shape: tuple[int, ...], lo: float, hi: float) -> IntervalArray:
    shape = check_shape(shape)
    n = 1
    for d in shape:
        n *= d
    return IntervalArray(shape=shape, los=(float(lo),) * n, his=(float(hi),) * n)


def from_values(shape: tuple[int, ...], values: list[float]) -> IntervalArray:
    shape = check_shape(shape)
    vals = tuple(float(v) for v in values)
    return IntervalArray(shape=shape, los=vals, his=vals)


def top(shape: tuple[int, ...]) -> IntervalArray:
    """⊤: the unbounded interval — sound for any real value."""
    return from_bounds(shape, -_INF, _INF)


def straddles_zero(a: IntervalArray) -> bool:
    """True when ANY element of ``a`` has an interval containing zero.

    An interval ``[lo, hi]`` contains zero iff ``lo <= 0 <= hi``.
    Used by the ``div`` transfer to detect the decline condition."""
    return any(lo <= 0.0 <= hi for lo, hi in zip(a.los, a.his))


BOOL_TRUE = (1.0, 1.0)
BOOL_FALSE = (0.0, 0.0)
BOOL_UNKNOWN = (0.0, 1.0)


# -- elementwise plumbing -----------------------------------------------------


def _broadcast_shape(sa: tuple[int, ...], sb: tuple[int, ...]) -> tuple[int, ...]:
    """numpy-style broadcast of two shapes: trailing axes aligned, size-1
    axes replicate, missing leading axes replicate. Incompatible shapes
    raise :class:`IntervalError` (the transfer declines; never a crash)."""
    out: list[int] = []
    for da, db in itertools.zip_longest(reversed(sa), reversed(sb), fillvalue=1):
        if da == db or db == 1:
            out.append(da)
        elif da == 1:
            out.append(db)
        else:
            raise IntervalError(
                f"shapes {sa} and {sb} do not broadcast "
                f"(axis sizes {da} vs {db})"
            )
    return tuple(reversed(out))


def _bcast_elements(x: IntervalArray, out_shape: tuple[int, ...]):
    """Element (lo, hi) pairs of ``x`` replicated to ``out_shape`` (which
    must be a broadcast target of ``x.shape``), flat C-order."""
    n = 1
    for d in out_shape:
        n *= d
    if n == 0:
        return []
    k, r = len(x.shape), len(out_shape)
    elems = []
    for coord in _coords(out_shape):
        src = tuple(
            0 if x.shape[j] == 1 else coord[r - k + j] for j in range(k)
        )
        i = _flat_index(src, x.shape)
        elems.append((x.los[i], x.his[i]))
    return elems


def _pair_elements(a: IntervalArray, b: IntervalArray):
    """Zip two operands elementwise: equal shapes, the scalar-vs-any fast
    path (jaxprs carry scalar literals as rank-0 operands of elementwise
    eqns), and general numpy-style shape broadcasting (size-1 axes and
    missing leading axes replicate). Incompatible shapes raise
    :class:`IntervalError`, which the propagator converts to a noted
    ⊤-decline."""
    if a.shape == b.shape:
        return a.shape, list(zip(a.los, a.his)), list(zip(b.los, b.his))
    if a.is_scalar():
        n = b.size
        return b.shape, [(a.los[0], a.his[0])] * n, list(zip(b.los, b.his))
    if b.is_scalar():
        n = a.size
        return a.shape, list(zip(a.los, a.his)), [(b.los[0], b.his[0])] * n
    out_shape = _broadcast_shape(a.shape, b.shape)
    return out_shape, _bcast_elements(a, out_shape), _bcast_elements(b, out_shape)


def _binary(a: IntervalArray, b: IntervalArray, f) -> IntervalArray:
    shape, xs, ys = _pair_elements(a, b)
    los, his = [], []
    for (alo, ahi), (blo, bhi) in zip(xs, ys):
        lo, hi = f(alo, ahi, blo, bhi)
        los.append(lo)
        his.append(hi)
    return IntervalArray(shape=shape, los=tuple(los), his=tuple(his))


# -- arithmetic ---------------------------------------------------------------


def _add_lo(alo: float, blo: float) -> float:
    if not _exactable(alo, blo):
        return _down(alo + blo)
    return _exact_down(Fraction(alo) + Fraction(blo))


def _add_hi(ahi: float, bhi: float) -> float:
    if not _exactable(ahi, bhi):
        return _up(ahi + bhi)
    return _exact_up(Fraction(ahi) + Fraction(bhi))


def add(a: IntervalArray, b: IntervalArray) -> IntervalArray:
    return _binary(
        a, b, lambda alo, ahi, blo, bhi: (_add_lo(alo, blo), _add_hi(ahi, bhi))
    )


def sub(a: IntervalArray, b: IntervalArray) -> IntervalArray:
    return _binary(
        a, b, lambda alo, ahi, blo, bhi: (_add_lo(alo, -bhi), _add_hi(ahi, -blo))
    )


def neg(a: IntervalArray) -> IntervalArray:
    return IntervalArray(
        shape=a.shape,
        los=tuple(-h for h in a.his),
        his=tuple(-l for l in a.los),
    )


def abs_(a: IntervalArray) -> IntervalArray:
    """Piecewise-exact |·|: negation and max of doubles are exact, so the
    endpoints are the true image endpoints — no rounding bump."""
    los, his = [], []
    for lo, hi in zip(a.los, a.his):
        if lo >= 0.0:
            los.append(lo)
            his.append(hi)
        elif hi <= 0.0:
            los.append(-hi)
            his.append(-lo)
        else:  # straddles zero: image is [0, max(|lo|, hi)]
            los.append(0.0)
            his.append(max(-lo, hi))
    return IntervalArray(shape=a.shape, los=tuple(los), his=tuple(his))


def _mul_corners(
    alo: float, ahi: float, blo: float, bhi: float
) -> tuple[float, float]:
    """THE interval product of two elements — one implementation, two
    callers (:func:`mul` and :func:`dot_general`'s per-term product).

    They were two copies of the same four-corner rule, and M16 converted
    only one of them: `dot_general` kept the unconditional bump, so
    `jnp.sum(x*x)` floored at exactly 0 while `jnp.dot(x, x)` floored at
    `-1e-323` and lost the same nonnegative clamp M16 was about. The
    divergence was disclosed and attributed to "a contraction's numerics
    need their own measurement of the association order" — which was the
    wrong reason: the ACCUMULATION already used `_add_lo`/`_add_hi`, the
    exact-when-representable route, and association order has nothing to
    do with a product's corners. Only the corners bumped. Sharing the code
    is what stops the next reader from having to notice that again.

    Finite endpoints: the four corner products are exact rationals, so the
    extremum is exact and the single directed rounding leaves it UNCHANGED
    whenever it is representable. An infinite endpoint keeps the
    unconditional bump — ``Fraction(inf)`` raises, and :func:`_prod`'s
    closed-interval ``0 * ±inf = 0`` is an endpoint convention rather than
    real arithmetic.
    """
    if _exactable(alo, ahi, blo, bhi):
        ex = [Fraction(x) * Fraction(y)
              for x in (alo, ahi) for y in (blo, bhi)]
        return _exact_down(min(ex)), _exact_up(max(ex))
    products = (
        _prod(alo, blo), _prod(alo, bhi), _prod(ahi, blo), _prod(ahi, bhi)
    )
    return _down(min(products)), _up(max(products))


def mul(a: IntervalArray, b: IntervalArray) -> IntervalArray:
    """Real multiplication, exact on finite endpoints.

    The four corner products of FINITE endpoints are exact rationals and
    ``min``/``max`` over them is exact, so there is one rounding in the
    whole transfer and ``_exact_down``/``_exact_up`` do it *directed* —
    returning the endpoint UNCHANGED when the extremum is representable.
    This is the same route `add` (`_add_lo`/`_add_hi`) and `div` have
    always taken; `mul` was the only arithmetic transfer that bumped
    unconditionally (audit 0.2.0 M16).

    The bump it used to apply was not free precision. `mul([2,3],[2,3])`
    returned `[3.9999999999999996, 9.000000000000002]` for an image that
    is exactly `[4, 9]`, and — the consequence that mattered — the
    exactly-zero corner of `mul([0,4],[0,4])` bumped to `-5e-324`, BELOW
    ZERO. That defeats `reduce_sum`'s nonnegative clamp, so a
    sum-of-squares written `x*x` became a true straddle and the divisor it
    fed declined, while the same sum written `x**2` verified: one real
    property, two verdicts, decided by the spelling. Sound in both
    directions — a wider box only loses precision — but the loss landed
    exactly on the `assume(x > 0)` sum-of-squares shape boundary-aware
    division was added for.

    Clamping the sign of a zero corner would have been a bandaid: it
    leaves `[2,3]×[2,3]` inexact, and the inexactness is the defect.

    An infinite endpoint keeps the unconditional bump: ``Fraction(inf)``
    raises, and the closed-interval ``0 * ±inf = 0`` convention in
    :func:`_prod` is an endpoint rule rather than real arithmetic, so it
    stays where it already lived. Same confinement as `add` and `div`.

    The rule itself lives in :func:`_mul_corners`, which
    :func:`dot_general` also calls — the sharing is the fix, not tidiness
    (a second copy is what let M16 survive its own fix one level up).
    """
    return _binary(a, b, _mul_corners)


def div(a: IntervalArray, b: IntervalArray) -> IntervalArray:
    def f(alo, ahi, blo, bhi):
        if blo <= 0.0 <= bhi:
            # denominator may vanish: ⊤ is the only sound closed-interval
            # answer here; the verdict degrades to UNKNOWN, never crashes.
            return -_INF, _INF
        for x in (alo, ahi):
            for y in (blo, bhi):
                if (x == _INF or x == -_INF) and (y == _INF or y == -_INF):
                    # inf/inf is indeterminate; widen fully outward.
                    return -_INF, _INF
        if _exactable(alo, ahi, blo, bhi):
            # The zero-straddle test above has already established that
            # neither divisor endpoint is 0, so these rational divisions are
            # total. Fraction division is EXACT, so no corner is inexact and
            # there is nothing to taint per-corner: take the extrema of four
            # exact rationals and round outward ONCE, and not at all when the
            # extremum is representable. `0.0 / x` is exactly 0 and survives,
            # which is what lets a sum-of-squares residual keep its floor
            # through the division that follows it.
            ex = [Fraction(x) / Fraction(y)
                  for x in (alo, ahi) for y in (blo, bhi)]
            return _exact_down(min(ex)), _exact_up(max(ex))
        # An infinite endpoint is in play: `Fraction(inf)` raises, and
        # `finite/±inf -> ±0.0` is an endpoint convention rather than real
        # arithmetic, so the unconditional bump stays in force here.
        quotients = [x / y for x in (alo, ahi) for y in (blo, bhi)]
        return _down(min(quotients)), _up(max(quotients))

    return _binary(a, b, f)


def _boundary_div_lo(num: float, den: float) -> float:
    """Sound lower endpoint for a single finite boundary-div quotient."""
    if num == 0.0:
        return 0.0
    if not _exactable(num, den):
        return _down(num / den)
    return _exact_down(Fraction(num) / Fraction(den))


def _boundary_div_hi(num: float, den: float) -> float:
    """Sound upper endpoint for a single finite boundary-div quotient."""
    if num == 0.0:
        return 0.0
    if not _exactable(num, den):
        return _up(num / den)
    return _exact_up(Fraction(num) / Fraction(den))


def boundary_div(a: IntervalArray, b: IntervalArray) -> IntervalArray:
    """Division when the divisor has zero at exactly ONE boundary.

    Precondition: every element of ``b`` that straddles zero has zero at
    exactly one boundary (i.e., lo == 0 with hi > 0, or hi == 0 with lo < 0).
    No element may be [0, 0] or a true straddle (lo < 0 < hi).

    For elements where the divisor does NOT contain zero, normal division
    is used. For one-sided boundary elements, the result is computed with
    the appropriate infinite endpoint.

    ``inf/inf`` IS ⊤ HERE TOO, and it is checked once for both arms. It
    used to be checked only on the non-zero-containing arm, so
    ``boundary_div([inf, inf], [0, inf])`` fell into
    :func:`_boundary_div_lo`, computed ``inf/inf = NaN`` and raised
    ``IntervalError("NaN endpoint")`` — an internal invariant string
    surfacing as a user-facing decline reason out of a kernel whose
    contract is to degrade rather than crash (audit 0.2.0 B5-3; 8 such box
    pairs in the sweep). The guard is :func:`div`'s, verbatim, so the two
    kernels answer the indeterminate form the same way; it is deliberately
    the same conservative four-corner test rather than a
    which-arm-uses-which-corner refinement, because ⊤ is always sound and
    a narrower test here is a second thing to keep right.
    """
    def f(alo, ahi, blo, bhi):
        # Precondition validation: reject divisors that are not valid
        # one-sided-boundary cases. A function must be correct regardless of
        # who calls it — wrong results are worse than crashes.
        if blo < 0.0 < bhi:
            raise IntervalError(
                f"boundary_div requires a one-sided-boundary divisor, but got "
                f"a true straddle [{blo}, {bhi}] (lo < 0 < hi); use div() for "
                f"divisors that span both signs"
            )
        if blo == 0.0 and bhi == 0.0:
            raise IntervalError(
                f"boundary_div requires a one-sided-boundary divisor, but got "
                f"point-at-zero [0, 0]; division by zero has no finite result"
            )
        for x in (alo, ahi):
            for y in (blo, bhi):
                if (x == _INF or x == -_INF) and (y == _INF or y == -_INF):
                    # inf/inf is indeterminate; widen fully outward, before
                    # any endpoint arithmetic can manufacture a NaN
                    return -_INF, _INF
        b_contains_zero = blo <= 0.0 <= bhi
        if not b_contains_zero:
            # Normal division (no zero in divisor)
            if _exactable(alo, ahi, blo, bhi):
                ex = [Fraction(x) / Fraction(y)
                      for x in (alo, ahi) for y in (blo, bhi)]
                return _exact_down(min(ex)), _exact_up(max(ex))
            quotients = [x / y for x in (alo, ahi) for y in (blo, bhi)]
            return _down(min(quotients)), _up(max(quotients))
        # One-sided boundary: zero at exactly one end
        if blo == 0.0:
            # b = [0, hi], hi > 0: divisor approaches 0 from above
            if alo >= 0.0:
                # Non-negative / positive-approaching-0: [alo/hi, +inf]
                return _boundary_div_lo(alo, bhi), _INF
            elif ahi <= 0.0:
                # Non-positive / positive-approaching-0: [-inf, ahi/hi]
                return -_INF, _boundary_div_hi(ahi, bhi)
            else:
                # Dividend straddles zero: both ±inf reachable
                return -_INF, _INF
        else:
            # b = [lo, 0], lo < 0: divisor approaches 0 from below
            if alo >= 0.0:
                # Non-negative / negative-approaching-0: [-inf, alo/lo]
                return -_INF, _boundary_div_hi(alo, blo)
            elif ahi <= 0.0:
                # Non-positive / negative-approaching-0: [ahi/lo, +inf]
                return _boundary_div_lo(ahi, blo), _INF
            else:
                # Dividend straddles zero: both ±inf reachable
                return -_INF, _INF

    return _binary(a, b, f)


def maximum(a: IntervalArray, b: IntervalArray) -> IntervalArray:
    # max is monotone in both args: no rounding, endpoints are real values
    return _binary(a, b, lambda alo, ahi, blo, bhi: (max(alo, blo), max(ahi, bhi)))


def minimum(a: IntervalArray, b: IntervalArray) -> IntervalArray:
    return _binary(a, b, lambda alo, ahi, blo, bhi: (min(alo, blo), min(ahi, bhi)))


def join(cases: list[IntervalArray]) -> IntervalArray:
    """Interval hull (union) of same-shape boxes — the sound over-approximation
    of a branch whose taken case is not determined."""
    shape = cases[0].shape
    if any(c.shape != shape for c in cases):
        # a larger case would be silently truncated to case 0's element
        # count — refuse rather than mis-join (audit finding 7)
        raise IntervalError(f"join over mismatched shapes {[c.shape for c in cases]}")
    los = tuple(min(c.los[i] for c in cases) for i in range(cases[0].size))
    his = tuple(max(c.his[i] for c in cases) for i in range(cases[0].size))
    return IntervalArray(shape=shape, los=los, his=his)


def meet(a: IntervalArray, b: IntervalArray) -> IntervalArray:
    """Exact intersection (meet) of two same-shape boxes: elementwise
    ``[max(lo_a, lo_b), min(hi_a, hi_b)]``.

    **No outward rounding, deliberately.** ``max``/``min`` perform no
    arithmetic — each result endpoint *is* one of the operands' own
    endpoints — so the intersection of exact endpoints is exact. An
    outward bump would readmit values both operands exclude (a needless
    precision loss, though sound); an inward bump would shrink the set,
    which for the constraining-assume path is the false-VERIFIED
    direction. The soundness algebra this serves: the meet of two
    supersets of a set S is itself a superset of S.

    An elementwise-empty intersection raises :class:`IntervalError` — the
    caller decides what emptiness means (for an assume it is an
    unsatisfiable precondition, a harness defect). Mismatched shapes also
    raise :class:`IntervalError`: broadcasting is the caller's business,
    never guessed here.
    """
    if a.shape != b.shape:
        raise IntervalError(
            f"meet over mismatched shapes {a.shape} vs {b.shape}"
        )
    los = tuple(max(x, y) for x, y in zip(a.los, b.los))
    his = tuple(min(x, y) for x, y in zip(a.his, b.his))
    for lo, hi in zip(los, his):
        if lo > hi:
            raise IntervalError(
                f"empty meet: intersection element [{lo}, {hi}] contains "
                f"no real"
            )
    return IntervalArray(shape=a.shape, los=los, his=his)


def select_n(which: IntervalArray, cases: list[IntervalArray]) -> IntervalArray:
    """`select_n(which, *cases)`: elementwise pick of ``cases[which]``.

    ``which`` is a predicate/index interval on {0, 1, …}. Where it is
    **definite** (a single integer at that element) the exact case is
    taken; where it **straddles** (the branch is undetermined) the possible
    cases are joined — sound, and the source of branch imprecision that a
    solver would resolve. An infinite (⊤) selector element joins every
    case rather than crashing on the int conversion (audit finding 5 —
    reachable from any trace whose predicate involves an unregistered
    primitive).

    Out-of-range selectors **clamp** — jax's measured ``lax.select_n``
    semantics (0.11, eager and jit agree: index −1 → case 0), which is NOT
    ``cond``'s convention (measured: index −1 → last branch). Second
    audit, finding 3: the earlier last-case fallback here selected the
    wrong end of that asymmetry.

    Shapes: all cases must agree; ``which`` is either case-shaped
    (elementwise selection) or a **scalar** broadcast across the cases'
    elements (jax permits exactly these two forms). Anything else raises
    :class:`IntervalError` — a decline the propagator notes, not a
    crash."""
    if not cases:
        raise IntervalError("select_n with no cases")
    if any(c.shape != cases[0].shape for c in cases[1:]):
        raise IntervalError(
            f"select_n cases disagree on shape: {[c.shape for c in cases]}"
        )
    scalar_which = which.is_scalar() and not cases[0].is_scalar()
    if which.shape != cases[0].shape and not scalar_which:
        raise IntervalError(
            f"select_n case shapes {[c.shape for c in cases]} != which "
            f"{which.shape} (equal shapes or a scalar selector are the only "
            f"supported forms)"
        )
    n = cases[0].size
    last = len(cases) - 1
    los, his = [], []
    for i in range(n):
        wi = 0 if scalar_which else i
        w_lo, w_hi = which.los[wi], which.his[wi]
        if w_lo == -_INF or w_hi == _INF:
            picks = cases  # ⊤ selector: any case possible
        else:
            lo_idx, hi_idx = int(math.floor(w_lo)), int(math.floor(w_hi))
            possible = set(range(max(0, lo_idx), min(last, hi_idx) + 1))
            if lo_idx < 0:
                possible.add(0)  # below-range mass clamps to the first case
            if hi_idx > last:
                possible.add(last)  # above-range mass clamps to the last case
            picks = [cases[k] for k in sorted(possible)]
        los.append(min(c.los[i] for c in picks))
        his.append(max(c.his[i] for c in picks))
    return IntervalArray(shape=cases[0].shape, los=tuple(los), his=tuple(his))


def exp(a: IntervalArray) -> IntervalArray:
    def e(x: float, up_side: bool) -> float:
        if x == -_INF:
            return 0.0
        if x == _INF:
            return _INF
        try:
            v = math.exp(x)
        except OverflowError:
            return _INF if up_side else math.nextafter(_INF, 0.0)
        return _up(v) if up_side else max(0.0, _down(v))

    return IntervalArray(
        shape=a.shape,
        los=tuple(e(l, False) for l in a.los),
        his=tuple(e(h, True) for h in a.his),
    )


# Deliberately provenance-neutral: this layer sees only the argument
# INTERVAL, not where it came from — the interval may be the user's
# declaration or an artifact ⊤ propagated from an upstream decline, and
# the propagator (which knows which) appends the provenance. The earlier
# wording said "over the declared box", which misattributed upstream ⊤s
# to the user's declaration (docs/proposed-decline-messages.md #2).
SQRT_DOMAIN_REASON = (
    "sqrt is real only for a nonnegative argument, and the argument "
    "interval's lower bound {lo} is negative — the interval reaches "
    "out-of-domain points (jnp.sqrt of a negative is NaN), so arg >= 0 is "
    "not established over it; declined rather than silently narrowed"
)


def sqrt(a: IntervalArray) -> IntervalArray:
    """Real square root over ``[lo, hi]`` — monotone increasing on the
    domain ``[0, inf)``, outward-rounded.

    sqrt is a CORRECTLY-ROUNDED IEEE-754 basic operation (error <= 0.5 ulp,
    like +, -, *, /), so — unlike :func:`exp` and :func:`pow_`, which ride a
    faithfully-rounded libm demotion — its endpoints need no libm-fidelity
    assumption: ``math.sqrt`` is bumped one ulp outward, which contains the
    true real root under correct rounding with a full ulp to spare and stays
    sound even on a platform whose sqrt is merely faithfully rounded (error
    <= 1 ulp). Tier ``sound``.

    The lower endpoint is floored at 0: ``sqrt(x) >= 0`` for every ``x`` in
    the domain, so non-negativity is PRODUCED here rather than left to an
    outward bump to (wrongly) admit a negative value — the :func:`exp`
    lower-floor discipline. ``sqrt(0) = 0`` exactly and ``sqrt(inf) = inf``.

    The argument's domain is ``arg >= 0`` — the OBLIGATION a sqrt call
    carries. Where a base interval reaches below 0 (``lo < 0``) the box
    includes out-of-domain points, so the transfer RAISES
    :class:`IntervalError` (:data:`SQRT_DOMAIN_REASON`) — the :func:`pow_`
    domain posture, which the propagator turns into a noted top-decline,
    never a crash. sqrt's domain is CLOSED at 0 (``sqrt(0) = 0`` is defined),
    so only a strictly-negative lower bound declines; ``lo == 0`` is
    in-domain.
    """
    if any(lo < 0.0 for lo in a.los):
        raise IntervalError(SQRT_DOMAIN_REASON.format(lo=min(a.los)))

    def s(x: float, up_side: bool) -> float:
        if x == _INF:
            return _INF
        if x == 0.0:
            return 0.0  # sqrt(0) = 0 exactly (covers -0.0 too)
        v = math.sqrt(x)
        return _up(v) if up_side else max(0.0, _down(v))

    return IntervalArray(
        shape=a.shape,
        los=tuple(s(l, False) for l in a.los),
        his=tuple(s(h, True) for h in a.his),
    )


def _frac_bracket(fr) -> tuple[float, float]:
    """The tightest SOUND double bracket of an exact :class:`Fraction`.

    Round-to-nearest gives one endpoint; comparing the rounded double back
    against the exact rational says which side it fell on, so the bracket
    bumps one ulp on that side only — and not at all when the rational is
    exactly representable. Overflow saturates outward, keeping maxfloat as
    a finite witness (the :func:`exp` overflow treatment).
    """
    from fractions import Fraction  # stdlib; kept local to the one user

    try:
        x = float(fr)
    except OverflowError:
        maxf = math.nextafter(_INF, 0.0)
        return (maxf, _INF) if fr > 0 else (-_INF, -maxf)
    if x == _INF:
        return math.nextafter(_INF, 0.0), _INF
    if x == -_INF:
        return -_INF, math.nextafter(-_INF, 0.0)
    xf = Fraction(x)
    if xf == fr:
        return x, x  # exactly representable: no bump, the value IS the set
    if xf < fr:
        return x, math.nextafter(x, _INF)
    return math.nextafter(x, -_INF), x


def _int_pow_bracket(x: float, n: int) -> tuple[float, float]:
    """Sound double bracket of ``x ** n`` for an integer ``n >= 0``.

    Finite endpoints go through the EXACT rational power (a double is a
    dyadic rational, so ``Fraction(x) ** n`` is exact) and are bracketed
    outward only where the double cannot represent the result — tighter
    than repeated outward-rounded multiplication and, unlike
    :func:`pow_`, riding on no libm fidelity claim (tier ``sound``, not
    ``sound-libm``).
    """
    from fractions import Fraction

    if n == 0:
        return 1.0, 1.0  # the empty product, for EVERY base (see integer_pow)
    if x == _INF:
        return _INF, _INF
    if x == -_INF:
        return (-_INF, -_INF) if n % 2 else (_INF, _INF)
    return _frac_bracket(Fraction(x) ** n)


def _recip_bracket(p: float) -> tuple[float, float]:
    """Sound double bracket of ``1 / p`` for a NONZERO ``p``. An infinite
    endpoint reciprocates to exactly 0 (the extended-real limit), which
    keeps the sign fact an outward bump would destroy."""
    from fractions import Fraction

    if p == _INF or p == -_INF:
        return 0.0, 0.0
    return _frac_bracket(Fraction(1, 1) / Fraction(p))


def _integer_pow_elt(lo: float, hi: float, y: int) -> tuple[float, float]:
    """One element of :func:`integer_pow` — see that function's contract."""
    if y == 0:
        return 1.0, 1.0
    if y == 1:
        return lo, hi
    if y > 0:
        if y % 2 == 0:
            # even: decreasing on (-inf, 0], increasing on [0, inf)
            if lo >= 0.0:
                return max(0.0, _int_pow_bracket(lo, y)[0]), _int_pow_bracket(hi, y)[1]
            if hi <= 0.0:
                return max(0.0, _int_pow_bracket(hi, y)[0]), _int_pow_bracket(lo, y)[1]
            # straddles 0: the image is [0, max(lo**y, hi**y)]. The lower
            # endpoint is EXACTLY 0 (attained at x = 0) — non-negativity is
            # PRODUCED here, never left to an outward bump to permit.
            return 0.0, max(
                _int_pow_bracket(lo, y)[1], _int_pow_bracket(hi, y)[1]
            )
        # odd: strictly increasing over the whole line
        return _int_pow_bracket(lo, y)[0], _int_pow_bracket(hi, y)[1]
    # y < 0 — a reciprocal, and therefore the SAME zero-in-divisor
    # discipline div uses: a base interval reaching 0 has a genuine pole
    # inside it (the image is unbounded on at least one side), and ⊤ is the
    # only sound closed-interval answer. Never a silently-inverted interval.
    if lo <= 0.0 <= hi:
        return -_INF, _INF
    p_lo, p_hi = _integer_pow_elt(lo, hi, -y)  # base excludes 0
    if p_lo <= 0.0 <= p_hi:
        # the power's own bracket reached 0 (underflow of a tiny base): the
        # reciprocal is unbounded over it, so the same discipline applies
        return -_INF, _INF
    cands = (_recip_bracket(p_lo), _recip_bracket(p_hi))
    return min(c[0] for c in cands), max(c[1] for c in cands)


# The exact-rational endpoint path costs time proportional to the size of
# `Fraction(x) ** n`, whose numerator and denominator grow LINEARLY in n —
# and jax puts no bound on the exponent (`x ** 10_000_000` is one legal
# equation). Degrade-don't-crash extends to degrade-don't-hang, so the
# path is capped (audit FRAGILE 2).
#
# 1024 chosen on measured single-element cost of `integer_pow` at the cap,
# worst base class first:
#     base ~1e300      9.2 ms      base 5e-324   3.2 ms
#     base 0.1         0.55 ms     base 2.0      0.009 ms
# and on the scaling above it: y=4096 -> 80 ms, y=16384 -> 705 ms,
# y=65536 -> 6.5 s (all base ~1e300). 1024 is 16x the SMT emission's
# INTEGER_POW_EXPANSION_CAP, so the transfer is never the binding
# constraint on anything that could be escalated, while still covering
# every exponent hand-written numerical code plausibly contains.
INTEGER_POW_EXACT_CAP = 1024

# The cap above bounds the per-element cost f(y); the transfer is
# ELEMENTWISE, so the bill is size x f(y) and the exponent cap alone does
# not bound it (audit FRAGILE 3: a 200 000-element array at the capped
# exponent cost 112 s). Measured at the cap, cost per unit of `size * |y|`
# is ~0.54 us for a full-mantissa base and ~9 us for a base near the
# overflow boundary — the worst class. 100 000 units therefore bounds the
# worst case near 0.9 s while still admitting y=2 over 50 000 elements,
# y=64 over 1 562, and the capped y=1024 over 97.
INTEGER_POW_WORK_CAP = 100_000

INTEGER_POW_WORK_DECLINE = (
    "integer_pow over {size} elements at |y| = {n} would cost {work} units "
    "of exact-rational work, beyond the budget ({cap}): the endpoints are "
    "computed per element and the exponent cap alone bounds only the "
    "per-element cost, not size x f(y) — declined with the shape and "
    "exponent quoted"
)

INTEGER_POW_CAP_DECLINE = (
    "integer_pow exponent |y| = {n} exceeds the exact-rational endpoint cap "
    "({cap}): the endpoints are computed as the EXACT rational power, whose "
    "size grows linearly in the exponent, and jax bounds the exponent "
    "nowhere (`x ** 10_000_000` is a single legal equation costing minutes). "
    "Unbounded compute is neither a verdict nor a decline, so this declines "
    "with the exponent quoted"
)


def integer_pow(a: IntervalArray, y: int) -> IntervalArray:
    """``x ** y`` for a fixed integer ``y`` (jax's ``integer_pow`` primitive).

    * ``y = 0`` → the constant 1 for **every** base. This is the empty
      product, and it is what jax computes: measured on jax 0.11.0 CPU
      (binary64), ``integer_pow(x, 0) == 1.0`` at ``x`` = 0.0, -0.0, ±inf
      **and NaN**.
    * ``y > 0`` even → non-negative by construction: ``[0, max(lo**y,
      hi**y)]`` when the base straddles 0, else the ordered endpoints.
    * ``y > 0`` odd → monotone increasing: the ordered endpoints.
    * ``y < 0`` → a reciprocal, routed through the same zero-in-divisor
      discipline as :func:`div`: a base interval containing 0 yields ⊤
      (the pole is real — the image really is unbounded there), never an
      inverted interval.

    Endpoints come from the EXACT rational power bracketed outward
    (:func:`_int_pow_bracket`), so no libm fidelity is assumed: tier
    ``sound``.

    Exponents beyond :data:`INTEGER_POW_EXACT_CAP` return ⊤ rather than
    paying an unbounded exact-rational cost — sound, and the same posture
    :func:`div` already takes for a vanishing denominator (a kernel-level
    ⊤, never a raise). The **registered transfer** checks the cap first
    and declines with :data:`INTEGER_POW_CAP_DECLINE` quoted, so the
    refusal reaches the verdict notes rather than passing silently; this
    ⊤ is the belt to that braces, for any caller reaching the kernel
    directly.
    """
    if abs(y) > INTEGER_POW_EXACT_CAP or a.size * abs(y) > INTEGER_POW_WORK_CAP:
        return top(a.shape)
    los, his = [], []
    for lo, hi in zip(a.los, a.his):
        l, h = _integer_pow_elt(lo, hi, y)
        los.append(l)
        his.append(h)
    return IntervalArray(shape=a.shape, los=tuple(los), his=tuple(his))


def _check_axes(a: IntervalArray, axes: tuple[int, ...], name: str) -> set[int]:
    ax = set(axes)
    if any(not (isinstance(d, int) and 0 <= d < len(a.shape)) for d in ax):
        raise IntervalError(
            f"{name} axes {axes} out of range for shape {a.shape}"
        )
    return ax


def _reduced_shape(
    a: IntervalArray, ax: set[int]
) -> tuple[tuple[int, ...], int, int]:
    """``(out_shape, out_n, n_contrib)`` for a reduction over ``ax``."""
    out_shape = tuple(d for i, d in enumerate(a.shape) if i not in ax)
    out_n = 1
    for d in out_shape:
        out_n *= d
    n_contrib = 1
    for i, d in enumerate(a.shape):
        if i in ax:
            n_contrib *= d
    return out_shape, out_n, n_contrib


def reduce_sum(a: IntervalArray, axes: tuple[int, ...]) -> IntervalArray:
    """Sum over ``axes``: output shape is the input shape with those axes
    removed.

    Under the declared **ℝ** semantics addition is exactly associative and
    commutative, so the true sum of the reduced elements lies in
    ``[Σ lo_i, Σ hi_i]`` whatever order the compiler picks — the bracket
    bounds every association order at once, because in ℝ they all denote
    the same number. The accumulator is SEEDED with the first contributor
    rather than with 0, so an ``n``-element reduction runs the ``n - 1``
    additions its ``n - 1`` real additions earn and no more, and a
    one-element reduction is exact.

    *This paragraph read "each accumulation step is bumped one ulp
    outward, so an n-element sum spends exactly the n − 1 bumps its n − 1
    real additions earn". That was the rule before the exact-endpoint
    route landed, and the very next paragraph has contradicted it since:
    a step that is exact is not bumped at all, so the count of bumps is a
    property of the DATA and not of ``n``. The seeding claim was the true
    half and is kept.*

    The fold identity is 0: an empty reduction range yields exactly
    ``[0, 0]``, matching jax (measured: ``jnp.sum`` of a size-0 array is
    ``0.0``).

    **What this actually does, which is weaker than "sum exactly, round
    once":** each accumulation step is exact-when-representable
    (:func:`_add_lo`/:func:`_add_hi`) and rounds outward only where THAT
    STEP is inexact. So for ``n >= 3`` the result is NOT the correctly
    directed rounding of the exact total, and the endpoints are NOT the two
    neighbours of it — the single-op property that one endpoint equals
    ``fl(R)`` does not extend here. Sound (every step is directed outward),
    tighter than the old unconditional bump, and deliberately not maximally
    tight: exact ``Fraction`` accumulation over field-sized arrays is a cost
    nobody has measured. Revisit only if a measured obligation needs it.

    One consequence worth relying on: **nonnegative in, nonnegative out.**
    Exact addition of nonnegative endpoints is nonnegative, and rounding a
    nonnegative real downward cannot cross zero (``0.0`` is a double), so a
    sum of ``lo >= 0`` contributors cannot produce ``lo < 0`` — measured
    over 20k randomised reductions, worst observed ``lo`` is ``0.0``. That
    is the "sum of squares >= 0" class discharging without a sign clamp.

    **This reasoning is available only under ℝ.** Float addition is not
    associative and the jaxpr fixes no summation order, so the ieee
    counterpart (:func:`ieee_reduce_sum`) cannot reuse it — see that
    function and :data:`REDUCE_SUM_IEEE_ORDER_DECLINE`.
    """
    ax = _check_axes(a, axes, "reduce_sum")
    out_shape, out_n, _ = _reduced_shape(a, ax)
    los = [0.0] * out_n  # additive identity: the empty sum is exactly 0
    his = [0.0] * out_n
    seen = [False] * out_n
    nonneg = [True] * out_n  # every contributor so far had lo >= 0
    if a.size:
        for coord in _coords(a.shape):
            i = _flat_index(coord, a.shape)
            j = _flat_index(
                tuple(c for k, c in enumerate(coord) if k not in ax), out_shape
            )
            if not seen[j]:
                los[j], his[j] = a.los[i], a.his[i]
                seen[j] = True
                nonneg[j] = a.los[i] >= 0.0
            else:
                los[j] = _add_lo(los[j], a.los[i])
                his[j] = _add_hi(his[j], a.his[i])
                nonneg[j] = nonneg[j] and a.los[i] >= 0.0
    # Sign-awareness, deliberately NOT gated on semantics: rounding a
    # nonnegative real yields a nonnegative float under every rounding mode
    # and every association order, overflow included. So a sum whose every
    # contributor has ``lo >= 0`` cannot have a negative lower endpoint, and
    # a lower endpoint that went below zero is slack, never information.
    # ``-0.0`` is normalised to ``+0.0`` so no downstream row can branch on
    # the sign bit of a bound this rule produced.
    for j in range(out_n):
        if nonneg[j] and los[j] <= 0.0:
            los[j] = 0.0
    return IntervalArray(shape=out_shape, los=tuple(los), his=tuple(his))


class DotGeneralGeometry(NamedTuple):
    """The index geometry of one well-formed ``dot_general``."""

    lc: tuple[int, ...]          # lhs contracting dims, in jax's order
    rc: tuple[int, ...]          # rhs contracting dims, paired with ``lc``
    lb: tuple[int, ...]          # lhs batch dims
    rb: tuple[int, ...]          # rhs batch dims, paired with ``lb``
    # lhs / rhs dims that are neither batch nor contracted
    lfree: tuple[int, ...]
    rfree: tuple[int, ...]
    out_shape: tuple[int, ...]   # batch dims, then lhs free, then rhs free
    contracted_extents: tuple[int, ...]  # the AGREED extent of each contraction


def dot_general_geometry(
    lhs_shape, rhs_shape, dimension_numbers
) -> DotGeneralGeometry:
    """THE definition of a well-formed ``dot_general``, and the index
    geometry that follows from it — **one definition, driven by both
    faces**.

    Called by :func:`dot_general` (the interval transfer, hence the whole
    propagation leg) and by
    :func:`stelling.obligation._dot_general_plan` (the SMT emission and
    the exact-rational replay). Every well-formedness predicate this row
    has lives here and nowhere else.

    **WHAT THAT DOES AND DOES NOT BUY — corrected, audit 0.2.0 S12′.**
    The sentence above used to continue "…so the two faces cannot hold
    different opinions about whether an equation is admissible", and that
    was FALSE AS WRITTEN. **The oracle is shared; its ARGUMENTS are not.**
    :func:`dot_general` asks about the shapes of the PROPAGATED BOXES
    (``a.shape``, ``b.shape``); ``_dot_general_plan`` asks this same
    function about the shapes recorded on the equation's INVAR AVALS.
    Where those disagree — which ``ir.ClosedJaxpr.from_dict`` accepts,
    ``ir.py`` having scoped per-primitive shape inference out of
    ``_validate_loaded`` in writing — the two faces reach different
    answers again, and still in the asserting direction. Measured on
    ``4d793cf``: the transfer AGREED the contraction had four terms and
    printed the box ``[4, 8]``, the emission planned two, and the verdict
    read VERIFIED at 100% coverage on a claim whose truth is ``8 <= 4.5``.

    What this function guarantees is the narrower and true thing:
    **given the same shapes, the two faces reach the same admissibility
    answer, for the same stated reason.** That they are GIVEN the same
    shapes is a separate property enforced separately, by
    :meth:`stelling.obligation._Slicer._one_shape_per_value` — whose
    docstring carries the two witnesses it uses and where each is blind.

    **This function exists because they did.** Audit 0.2.0 S12: the
    contracted-extent agreement check was written inline in
    :func:`dot_general` and the emission re-derived the same geometry
    independently, iterating the LHS contraction extents alone. On
    ``lhs=(2,) @ rhs=(4,)`` the transfer RAISED and the emission returned a
    two-term combination — silently DROPPING two addends, the class this
    codebase's own comments name as the unsound one — and because the
    transfer's refusal lands the obligation at ⊤ → ``unknown``, the
    truncating plan is exactly what solver escalation then runs. Over the
    truncated reading ``Σ`` lay in ``[2, 4]`` and ``<= 4.5`` was
    unsat-on-negation (VERIFIED); over the true four-term reading it lay in
    ``[4, 8]`` and the threshold is violated. On ``lhs=(4,) @ rhs=(2,)`` the
    same loop indexed off the end of the constant operand and raised a raw
    ``IndexError`` out of a slicer that catches only ``_Decline``.

    A check in one face that the other does not consult is the SHAPE of
    that defect, not its repair — which is why this is a shared oracle and
    not a second copy of the predicate in the emission. It is the same
    discipline :func:`stelling.obligation._route_structural` already
    follows for the structural rows (the interval function *is* the
    routing) and that :func:`stelling.propagate._dot_general_row_form`
    follows for this row's *params and dtypes*; the two are complementary
    and disjoint — that oracle never sees a shape, this one never sees a
    param.

    Raises :class:`IntervalError` on any malformation — and that promise
    was false in exactly one line until audit 0.2.0 B6 (a non-integer dim
    passed the range test and then indexed a tuple with a float, raising a
    raw ``TypeError`` past both consumers' ``except IntervalError``), then
    false again in the same line for a different reason until B6's
    re-audit (the guard CALLED ``operator.index`` and discarded the
    result, so an object that is indexable but unhashable still reached
    ``set(dims)`` raw). The dims are now put through ``operator.index``
    and **bound to what it returns**, the same way :func:`check_shape`
    handles extents, so everything downstream — including the returned
    geometry — holds plain ``int``s. The emission face quotes the
    ``IntervalError`` as a decline.
    """
    # BOUND, like the dimension numbers below: `check_shape` normalises
    # every extent through `__index__` and returns what it read, so the
    # geometry this builds — `out_shape`, `contracted_extents`, both of
    # which the emission and the replay read — holds plain `int`s (audit
    # 0.2.0 B8a, item 1). `tuple(lhs_shape)` preserved the RAW objects.
    lhs_shape = check_shape(lhs_shape)
    rhs_shape = check_shape(rhs_shape)
    try:
        (lc, rc), (lb, rb) = dimension_numbers
        lc, rc, lb, rb = tuple(lc), tuple(rc), tuple(lb), tuple(rb)
    except (TypeError, ValueError):
        raise IntervalError(
            "dot_general dimension_numbers not in jax's "
            f"((lhs_contract, rhs_contract), (lhs_batch, rhs_batch)) form: "
            f"{dimension_numbers!r}"
        ) from None
    # THE DIMS ARE INDICES, so they must be INTEGRAL before anything indexes
    # with them — the same predicate, spelled the same way, that
    # :func:`check_shape` above already applies to extents.
    #
    # This line is audit 0.2.0 B6/S12″ and the finding was the DOCSTRING, not
    # only the code. A float dim passes the range test below (`0 <= 0.0 < 1`
    # is True) and then reaches `lhs_shape[i]`, where python raises a raw
    # `TypeError: tuple indices must be integers or slices, not float`. Both
    # consumers catch `IntervalError` and nothing else — `propagate._t_dot_
    # general` and `propagate.eqn` on the transfer side, `obligation.
    # _dot_general_plan` on the emission side — so the promise three
    # paragraphs up ("Raises IntervalError on any malformation") was false in
    # exactly one line, and it was false in the direction that matters: a raw
    # crash out of the public `propagate()` on a document `from_dict`
    # accepts, and, on the emission side, a decline quoted as an "internal
    # error" while the transfer face crashed instead. Two faces, two
    # behaviours, from one malformation — which is the S12 shape again.
    #
    # AND THE RESULT IS BOUND, not discarded — audit 0.2.0 B6 re-audit R4.
    # The first spelling of this guard CALLED `operator.index(d)` and threw
    # the answer away, so the dims were validated and never NORMALISED and
    # everything below still ran on the raw objects: `set(dims)` hashes
    # them, `0 <= d < len(shape)` orders them, `shape[i]` indexes with them.
    # Three protocols, one unvalidated object — and a 0-d `np.array(0)`
    # satisfies `__index__` while being UNHASHABLE, so it passed the guard
    # and then raised a raw `TypeError: unhashable type` out of the public
    # `propagate()` while the emission face declined: the same two-faces
    # split S12″ was, recurring one type level up because the fix had
    # checked a predicate instead of producing a value. `check_shape` above
    # is the model and always was: it binds `k = operator.index(d)` and
    # tests `k`. Binding here means every consumer below — and the
    # `DotGeneralGeometry` this returns, which the emission and the replay
    # both read — sees plain `int`s that no protocol can surprise.
    #
    # AND IT CATCHES WHATEVER `__index__` RAISES, quoting the dimension
    # through `_safe_repr` — audit 0.2.0 B6 audit 3, F2 and F3. Catching
    # only `TypeError` left a `ValueError` from a hostile `__index__`, and
    # a `RuntimeError` from a hostile `__repr__` in the handler itself,
    # raw out of the public `propagate()`: the same split one more level
    # in, from the same cause, that a guard was written as an enumeration
    # of the exception types the author happened to expect.
    def _indices(name: str, dims) -> tuple[int, ...]:
        out = []
        for d in dims:
            try:
                out.append(operator.index(d))
            except Exception:  # noqa: BLE001 — unreadable IS the finding
                raise IntervalError(
                    f"dot_general {name} dimension {_safe_repr(d)} is not "
                    f"an integer (malformed IR: from_dict does not coerce "
                    f"dimension_numbers entries)"
                ) from None
        return tuple(out)

    lc = _indices("lhs", lc)
    lb = _indices("lhs", lb)
    rc = _indices("rhs", rc)
    rb = _indices("rhs", rb)
    for name, dims, shape in (
        ("lhs", lc + lb, lhs_shape), ("rhs", rc + rb, rhs_shape)
    ):
        if len(set(dims)) != len(dims):
            raise IntervalError(
                f"dot_general {name} names a dimension twice: {dims}"
            )
        for d in dims:
            if not 0 <= d < len(shape):
                raise IntervalError(
                    f"dot_general {name} dimension {d} out of range for "
                    f"shape {shape}"
                )
    if len(lc) != len(rc) or len(lb) != len(rb):
        raise IntervalError(
            "dot_general contracting/batch dimension lists must pair up: "
            f"{dimension_numbers}"
        )
    for i, j in zip(lc, rc):
        if lhs_shape[i] != rhs_shape[j]:
            raise IntervalError(
                f"dot_general contracted dims disagree: lhs[{i}]="
                f"{lhs_shape[i]} vs rhs[{j}]={rhs_shape[j]}"
            )
    for i, j in zip(lb, rb):
        if lhs_shape[i] != rhs_shape[j]:
            raise IntervalError(
                f"dot_general batch dims disagree: lhs[{i}]={lhs_shape[i]} vs "
                f"rhs[{j}]={rhs_shape[j]}"
            )
    lfree = tuple(
        i for i in range(len(lhs_shape)) if i not in lb and i not in lc
    )
    rfree = tuple(
        j for j in range(len(rhs_shape)) if j not in rb and j not in rc
    )
    out_shape = (
        tuple(lhs_shape[i] for i in lb)
        + tuple(lhs_shape[i] for i in lfree)
        + tuple(rhs_shape[j] for j in rfree)
    )
    out_shape = check_shape(out_shape)
    return DotGeneralGeometry(
        lc=lc,
        rc=rc,
        lb=lb,
        rb=rb,
        lfree=lfree,
        rfree=rfree,
        out_shape=out_shape,
        # the extents are read from the LHS only because they have just been
        # checked EQUAL to the RHS's; that check is why reading one side is
        # not the S12 truncation
        contracted_extents=tuple(lhs_shape[i] for i in lc),
    )


def dot_general(
    a: IntervalArray,
    b: IntervalArray,
    dimension_numbers,
) -> IntervalArray:
    """General contraction: the interval meaning of jax's ``dot_general``.

    ``dimension_numbers`` is jax's own ``((lhs_contract, rhs_contract),
    (lhs_batch, rhs_batch))``. The output is ordered as jax orders it:
    batch dims first, then the lhs free dims ascending, then the rhs free
    dims ascending.

    **Why this is exact per output element, modulo rounding — and it is the
    property the whole row rests on.** Fix one output element. The lhs
    elements contributing to it are exactly ``{(batch, lfree, c)}`` as ``c``
    ranges over the contracted index tuples, and those multi-indices are
    pairwise DISTINCT — enumerated and duplicate-checked across matvec,
    matmul, a (64,32,32,19)x(19,3) contraction, a two-axis contraction and a
    batched contraction. So **no operand element appears twice in any one
    output element's sum**, and the dependency problem that makes interval
    arithmetic pessimistic does not arise within an element. Correlations
    BETWEEN output elements are still lost, which is the ordinary image gap
    and not this function's concern.

    Each term is a four-corner interval product — :func:`_mul_corners`,
    literally the function :func:`mul` calls, so the two cannot drift — and
    the terms are accumulated with the same exact-when-representable steps
    :func:`reduce_sum` uses, seeded with the first contributor, so a
    one-term contraction of representable operands spends no rounding at
    all.

    **The two used to be separate copies of one rule, and that is how M16
    survived its own fix.** Audit 0.2.0 M16 gave `mul` the
    exact-when-representable route and left the copy inlined here bumping
    unconditionally, so `jnp.sum(x*x)` floored at exactly 0 while
    `jnp.dot(x, x)` floored at `-1e-323` — a sum-of-squares residual losing
    its nonnegative clamp by being spelled as a contraction. The
    divergence was recorded but its stated reason was wrong: the
    ACCUMULATION here already used `_add_lo`/`_add_hi`, only the product
    corners bumped, and a product's corners have nothing to do with
    association order. The association-order argument below is untouched by
    this and always was.

    Under **R** semantics addition is associative, so the bracket bounds
    every association order XLA might pick at once. This reasoning is not
    available under ieee, exactly as for :func:`reduce_sum`, which is why
    the ieee census entry for this primitive declines rather than reusing
    this function.

    An EMPTY contraction (no contracted axes) is the outer product: the
    contracted index range is the single empty tuple, so each output element
    is one product and no accumulation happens.

    Well-formedness and the index geometry come from
    :func:`dot_general_geometry`, the oracle the SMT emission drives too —
    read its docstring for why this function no longer owns those
    predicates (audit 0.2.0 S12).
    """
    geom = dot_general_geometry(a.shape, b.shape, dimension_numbers)
    lc, rc, lb, rb = geom.lc, geom.rc, geom.lb, geom.rb
    lfree, rfree, out_shape = geom.lfree, geom.rfree, geom.out_shape
    contracted_ranges = [range(n) for n in geom.contracted_extents]

    los: list[float] = []
    his: list[float] = []
    for out_coord in _coords(out_shape):
        nb, nl = len(lb), len(lfree)
        bcoord = out_coord[:nb]
        lcoord_free = out_coord[nb:nb + nl]
        rcoord_free = out_coord[nb + nl:]
        acc_lo = 0.0
        acc_hi = 0.0
        seen = False
        nonneg = True
        for c in itertools.product(*contracted_ranges):
            ac = [0] * len(a.shape)
            bc = [0] * len(b.shape)
            for d, v in zip(lb, bcoord):
                ac[d] = v
            for d, v in zip(rb, bcoord):
                bc[d] = v
            for d, v in zip(lfree, lcoord_free):
                ac[d] = v
            for d, v in zip(rfree, rcoord_free):
                bc[d] = v
            for d, v in zip(lc, c):
                ac[d] = v
            for d, v in zip(rc, c):
                bc[d] = v
            ia = _flat_index(tuple(ac), a.shape)
            ib = _flat_index(tuple(bc), b.shape)
            alo, ahi = a.los[ia], a.his[ia]
            blo, bhi = b.los[ib], b.his[ib]
            plo, phi = _mul_corners(alo, ahi, blo, bhi)
            if not seen:
                acc_lo, acc_hi, seen = plo, phi, True
            else:
                acc_lo = _add_lo(acc_lo, plo)
                acc_hi = _add_hi(acc_hi, phi)
            nonneg = nonneg and plo >= 0.0
        # Same sign rule as reduce_sum, and sound for the same reason:
        # rounding a nonnegative real cannot cross zero under any mode, so a
        # sum of nonnegative-lo terms cannot have a negative lower endpoint,
        # and one that does is slack rather than information.
        if seen and nonneg and acc_lo <= 0.0:
            acc_lo = 0.0
        los.append(acc_lo)
        his.append(acc_hi)
    return IntervalArray(shape=out_shape, los=tuple(los), his=tuple(his))


def scatter_add_rows(
    a: IntervalArray, updates: IntervalArray, ks: list[int]
) -> IntervalArray:
    """Leading-axis row ACCUMULATE: the operand ``a`` with updates row
    ``j`` **added into** row ``ks[j]`` — the interval meaning of a
    static-index ``scatter-add``.

    The defining semantic, and the one that distinguishes this from
    :func:`take_rows`-style data movement and from the set/replace
    scatter: **duplicate rows in ``ks`` accumulate**. Each output element
    is ``operand[i] + Σ updates[j]`` over every ``j`` whose row maps to
    ``i`` (measured on jax 0.11.0:
    ``zeros(3).at[[0,2,0,0]].add([1,10,100,1000])`` is
    ``[1101, 0, 10]``, where the set form's last-wins answer would be
    ``[1000, 0, 10]``). Replacing instead of adding here would be
    unsound; the fidelity battery's first mutation is exactly that
    variant.

    Sound under ℝ for EVERY accumulation order at once: real addition is
    associative and commutative, so the true value of each output element
    is one number whatever order the contributions combine in, and the
    interval sum brackets it. Each accumulation step is bumped one ulp
    outward — ``n`` contributions spend the ``n`` bumps their ``n`` real
    additions earn — and untouched elements are copies of the operand's
    (no arithmetic, no bump). This ℝ-associativity reasoning is exactly
    what float addition does not offer, so the ieee mode declines the
    primitive instead of reusing this kernel
    (:data:`SCATTER_ADD_IEEE_DECLINE`).

    ``updates`` must have shape ``(len(ks), *a.shape[1:])``; each ``k``
    must be an in-range row. Violations raise :class:`IntervalError` (the
    transfer's decline channel; the registered transfer checks ranges
    before calling here). A NaN-producing accumulation (``inf + -inf``)
    raises through the endpoint check, exactly as :func:`add` would.
    """
    if not a.shape:
        raise IntervalError(
            "scatter_add_rows needs a leading axis; got rank-0 operand"
        )
    rowsz = 1
    for d in a.shape[1:]:
        rowsz *= d
    expect = (len(ks),) + a.shape[1:]
    if updates.shape != expect:
        raise IntervalError(
            f"scatter_add_rows updates shape {updates.shape} does not match "
            f"{expect} (one row of {a.shape[1:]} per index)"
        )
    los, his = list(a.los), list(a.his)
    for j, k in enumerate(ks):
        if not 0 <= k < a.shape[0]:
            raise IntervalError(
                f"scatter_add_rows row {k} out of range for leading axis "
                f"{a.shape[0]}"
            )
        for t in range(rowsz):
            oi = k * rowsz + t
            ui = j * rowsz + t
            los[oi] = _down(los[oi] + updates.los[ui])
            his[oi] = _up(his[oi] + updates.his[ui])
    return IntervalArray(shape=a.shape, los=tuple(los), his=tuple(his))


def pow_(a: IntervalArray, b: IntervalArray) -> IntervalArray:
    """``base ** exponent`` for a **strictly positive** base interval.

    For base > 0, ``x**y = exp(y·ln x)`` is monotone in ``x`` for every
    fixed ``y`` and monotone in ``y`` for every fixed ``x``, so the
    extremum over the (base, exponent) box lies at one of the four
    corners. Corners are evaluated with ``math.pow`` under the
    faithfully-rounded-libm assumption (:data:`POW_LIBM_ASSUMPTION`) and
    bumped one ulp outward; ``x > 0`` also gives ``x**y > 0``, so lower
    endpoints are floored at 0.

    A base interval reaching 0 or below has no sound rule here
    (``0**negative`` diverges, negative bases alternate sign with the
    exponent's parity): :class:`IntervalError` — the propagator turns it
    into a noted ⊤-decline, never a crash.
    """
    if any(lo <= 0.0 for lo in a.los):
        raise IntervalError(
            f"pow has a sound corner rule only for strictly positive bases; "
            f"base lower bound {min(a.los)} <= 0"
        )

    def f(alo, ahi, blo, bhi):
        lo_bounds, hi_bounds = [], []
        for x in (alo, ahi):
            for y in (blo, bhi):
                try:
                    v = math.pow(x, y)
                except OverflowError:
                    # the true corner value exceeds the double range:
                    # finite but > maxfloat — saturate outward, keeping
                    # maxfloat as a sound finite lower witness (the exp
                    # overflow treatment).
                    lo_bounds.append(math.nextafter(_INF, 0.0))
                    hi_bounds.append(_INF)
                    continue
                except ValueError as e:  # unreachable for x > 0; degrade anyway
                    raise IntervalError(f"math.pow({x}, {y}): {e}") from None
                # v == inf without OverflowError only happens at a corner
                # with an infinite operand endpoint (IEEE pow limits, e.g.
                # pow(inf, 2), pow(0.5, -inf)): the corner's true value is
                # inf itself, not a rounded finite — keep it exact.
                # (CPython math.pow raises OverflowError for finite
                # operands that overflow; it never returns inf silently.)
                lo_bounds.append(v if v == _INF else max(0.0, _down(v)))
                hi_bounds.append(v if v == _INF else _up(v))
        return min(lo_bounds), max(hi_bounds)

    return _binary(a, b, f)


# -- ieee (binary64) endpoint arithmetic --------------------------------------
#
# The semantics="ieee" kernels for the monotone arithmetic core. Under ieee
# semantics the semantic value of an op IS the float result, so endpoints
# are computed with native binary64 round-to-nearest arithmetic and NOT
# bumped outward: fl-rounded add/sub/mul/div are monotone in each argument,
# and the real extremum of each op over a box sits at a corner, so the
# corner evaluations bracket the float image exactly
# (:data:`IEEE_ENDPOINT_ASSUMPTION`). Each kernel returns
# ``(IntervalArray, made_nan)``: NaN-producing corner classes (inf − inf,
# 0·±inf, 0/0, ±inf/±inf) are detected from the operand endpoints, routed
# into the ``made_nan`` flag, and the interval is the hull of the non-NaN
# corners — a NaN endpoint never leaks into an interval. When every corner
# is NaN the non-NaN value set is empty and the kernels return ⊤ (any
# interval is a sound superset of the empty set) with ``made_nan=True``.
# Operand maybe-NaN flags are the CALLER's business (NaN poisons all four
# ops, so OR-ing operand flags into the result is sound there).
#
# Subnormal haze (the flush-fidelity fix): the kernels model BOTH gradual
# underflow and flush-to-zero targets. Operand endpoint pairs are hazed
# before the corner evaluation (DAZ: a subnormal operand may read as 0 —
# which also routes DAZ-created NaN classes like subnormal/subnormal =
# 0/0 into the flag) and result endpoints are hazed after it (FTZ: a
# subnormal result may flush to 0). See :func:`subnormal_haze` and
# :data:`SUBNORMAL_INDETERMINACY_ASSUMPTION`.
#
# ZERO IS WHERE THIS DOMAIN IS WEAKER THAN THE REAL ONE, not stronger. An
# IEEE format has TWO zeros and an interval endpoint has no sign bit, so a
# box that reaches zero cannot say which zero the program will meet — and
# the haze above MANUFACTURES such boxes, hulling with the positive literal
# 0.0 whatever the sign of the value it flushed. Division is where that
# bites: see :data:`IEEE_ZERO_DIVISOR_TOP`, which is also why the real-mode
# :func:`boundary_div` and :func:`ieee_div` deliberately disagree.

MIN_NORMAL = 2.0**-1022  # smallest positive normal binary64

# Format-parametric smallest normals: 2**emin for each supported format.
# Used by the parametric subnormal haze (_elt_haze_fmt) when the caller
# specifies a format narrower than binary64.
_MIN_NORMAL_FOR_EMIN: dict[int, float] = {
    -14: 2.0**-14,      # float16
    -126: 2.0**-126,    # float32, bfloat16
    -1022: 2.0**-1022,  # float64
}


def _band_touching(lo: float, hi: float) -> bool:
    """Does [lo, hi] contain a point of the OPEN subnormal band
    (-MIN_NORMAL, MIN_NORMAL) excluding {0}?"""
    return (hi > 0.0 and lo < MIN_NORMAL) or (lo < 0.0 and hi > -MIN_NORMAL)


def _band_touching_fmt(lo: float, hi: float, min_normal: float) -> bool:
    """Format-parametric band touching: does [lo, hi] contain a point of
    the open subnormal band (-min_normal, min_normal) excluding {0}?"""
    return (hi > 0.0 and lo < min_normal) or (lo < 0.0 and hi > -min_normal)


def _elt_haze(lo: float, hi: float) -> tuple[float, float]:
    """One element of the subnormal haze: hull a band-touching interval
    with 0 (identity when the interval already contains 0 or stays clear
    of the band)."""
    if _band_touching(lo, hi):
        return min(lo, 0.0), max(hi, 0.0)
    return lo, hi


def _elt_haze_fmt(lo: float, hi: float, min_normal: float) -> tuple[float, float]:
    """Format-parametric element haze: hull a band-touching interval with 0,
    using the format's own min_normal threshold."""
    if _band_touching_fmt(lo, hi, min_normal):
        return min(lo, 0.0), max(hi, 0.0)
    return lo, hi


def subnormal_haze(a: IntervalArray) -> tuple[IntervalArray, bool]:
    """The subnormal haze: every element whose interval touches the open
    subnormal band ``(-MIN_NORMAL, MIN_NORMAL)`` excluding {0} is hulled
    with 0.

    Whether a target flushes subnormals (FTZ/DAZ) is device/compiler-
    dependent (measured jax 0.11.0 CPU binary64 flushes in arithmetic,
    comparisons, and libm; strict IEEE-754 keeps gradual underflow), so
    ieee mode covers BOTH: the flushed image (0) joins the gradual values
    already present, and band-located claims become indeterminate rather
    than definite (:data:`SUBNORMAL_INDETERMINACY_ASSUMPTION`). Returns
    ``(hazed, changed)``; ``changed`` is False when the haze was the
    identity (no band contact, or the interval already contained 0 so no
    endpoint moved) — the exactness machinery keys off it.
    """
    changed = False
    los, his = list(a.los), list(a.his)
    for i, (lo, hi) in enumerate(zip(a.los, a.his)):
        nlo, nhi = _elt_haze(lo, hi)
        if nlo != lo or nhi != hi:
            los[i], his[i] = nlo, nhi
            changed = True
    if not changed:
        return a, False
    return IntervalArray(shape=a.shape, los=tuple(los), his=tuple(his)), True


def subnormal_haze_fmt(
    a: IntervalArray, min_normal: float
) -> tuple[IntervalArray, bool]:
    """Format-parametric subnormal haze: uses the format's own min_normal
    threshold instead of the binary64 constant. Semantics identical to
    :func:`subnormal_haze` but with a caller-specified band width.

    For float64 (min_normal = 2**-1022), this is byte-identical to
    :func:`subnormal_haze`. For float32 (min_normal = 2**-126), the band
    is ~270 orders of magnitude wider — correctly covering the float32
    subnormal region that the binary64 haze cannot see.
    """
    changed = False
    los, his = list(a.los), list(a.his)
    for i, (lo, hi) in enumerate(zip(a.los, a.his)):
        nlo, nhi = _elt_haze_fmt(lo, hi, min_normal)
        if nlo != lo or nhi != hi:
            los[i], his[i] = nlo, nhi
            changed = True
    if not changed:
        return a, False
    return IntervalArray(shape=a.shape, los=tuple(los), his=tuple(his)), True


def _ieee_binary(a: IntervalArray, b: IntervalArray, f):
    shape, xs, ys = _pair_elements(a, b)
    los, his = [], []
    made_nan = False
    for (alo, ahi), (blo, bhi) in zip(xs, ys):
        # DAZ face: a subnormal operand may read as 0 at runtime — hazing
        # the operand pairs widens the corner hull to cover the flushed
        # products/quotients AND lets the NaN-class detection see the
        # flushed 0 (a DAZ-created 0/0 or 0·±inf is a real NaN)
        alo, ahi = _elt_haze(alo, ahi)
        blo, bhi = _elt_haze(blo, bhi)
        lo, hi, nan_here = f(alo, ahi, blo, bhi)
        # FTZ face: a subnormal result may flush to 0
        lo, hi = _elt_haze(lo, hi)
        los.append(lo)
        his.append(hi)
        made_nan = made_nan or nan_here
    return (
        IntervalArray(shape=shape, los=tuple(los), his=tuple(his)),
        made_nan,
    )


def _ieee_binary_fmt(a: IntervalArray, b: IntervalArray, f, min_normal: float):
    """Format-parametric version of _ieee_binary: uses the format's own
    min_normal for the subnormal haze (both DAZ on operands and FTZ on
    results). The corner hull arithmetic is still in native float64, which
    is exact for all narrower formats' endpoints."""
    shape, xs, ys = _pair_elements(a, b)
    los, his = [], []
    made_nan = False
    for (alo, ahi), (blo, bhi) in zip(xs, ys):
        alo, ahi = _elt_haze_fmt(alo, ahi, min_normal)
        blo, bhi = _elt_haze_fmt(blo, bhi, min_normal)
        lo, hi, nan_here = f(alo, ahi, blo, bhi)
        lo, hi = _elt_haze_fmt(lo, hi, min_normal)
        los.append(lo)
        his.append(hi)
        made_nan = made_nan or nan_here
    return (
        IntervalArray(shape=shape, los=tuple(los), his=tuple(his)),
        made_nan,
    )


def _corner_hull(corners, made_nan):
    """Hull of the non-NaN corner values; ⊤ when every corner is NaN."""
    finite = [c for c in corners if c == c]  # drops NaN, keeps ±inf
    if not finite:
        return -_INF, _INF, True
    return min(finite), max(finite), made_nan or len(finite) < len(corners)


def ieee_add(a: IntervalArray, b: IntervalArray):
    def f(alo, ahi, blo, bhi):
        # NaN class: (+inf) + (−inf). inf is attainable only at endpoints,
        # so every NaN-producing pair is one of the four corners.
        corners = (alo + blo, alo + bhi, ahi + blo, ahi + bhi)
        return _corner_hull(corners, False)

    return _ieee_binary(a, b, f)


def ieee_sub(a: IntervalArray, b: IntervalArray):
    def f(alo, ahi, blo, bhi):
        corners = (alo - blo, alo - bhi, ahi - blo, ahi - bhi)
        return _corner_hull(corners, False)

    return _ieee_binary(a, b, f)


def ieee_mul(a: IntervalArray, b: IntervalArray):
    """IEEE multiplication.

    **This kernel deliberately does NOT take the exact-rational route the
    real-mode :func:`mul` takes** (audit 0.2.0 M16 asked the question).
    Under ieee semantics the value the program has IS ``fl(x*y)``, the
    correctly-rounded float product, and the native binary64 corner
    products already ARE that value — exactly, for binary64 operands, and
    exactly for every narrower format too, since a product of two float32
    (or bfloat16, or float16) values needs at most 48 significand bits and
    lands well inside binary64's exponent range, after which
    ``_ieee_round_box`` rounds outward onto the target's grid.

    Routing through ``Fraction`` would compute the REAL product and then
    round it outward, which is up to an ulp wider on each side than the
    set of values the target can produce — slack where there is none, and
    a claim about ℝ under a dial that speaks floats. That is the whole
    argument, and it is deliberately no longer accompanied by an overflow
    one. The overflow sentence this docstring used to add — "two operands
    near ``FMAX`` multiply to ``inf``, so the exact route's ``[FMAX, inf]``
    names a value the program cannot compute" — is true of binary64 and
    PROVES TOO MUCH: the format-parametric path already returns exactly
    that box, since the corner products are computed in binary64 and
    ``_ieee_round_box`` rounds them outward onto the narrow grid. Measured,
    float32: ``FMAX×FMAX`` boxes to ``(3.4028234663852886e+38, inf)`` while
    ``np.float32`` computes ``inf``. Sound — the box holds the value — but
    an argument that condemns the sibling row cannot be this row's reason.
    """

    def f(alo, ahi, blo, bhi):
        # NaN class 0·±inf: 0 may sit in the interior, inf only at
        # endpoints — detected from containment, not only from corners.
        a0, b0 = alo <= 0.0 <= ahi, blo <= 0.0 <= bhi
        ainf = alo == -_INF or ahi == _INF
        binf = blo == -_INF or bhi == _INF
        made_nan = (a0 and binf) or (b0 and ainf)
        corners = (alo * blo, alo * bhi, ahi * blo, ahi * bhi)
        return _corner_hull(corners, made_nan)

    return _ieee_binary(a, b, f)


def ieee_div(a: IntervalArray, b: IntervalArray):
    """IEEE division. A divisor box containing zero divides to ⊤ —
    :data:`IEEE_ZERO_DIVISOR_TOP` states why, and why the real-mode
    :func:`boundary_div` deliberately does not agree."""

    def f(alo, ahi, blo, bhi):
        a0, b0 = alo <= 0.0 <= ahi, blo <= 0.0 <= bhi
        ainf = alo == -_INF or ahi == _INF
        binf = blo == -_INF or bhi == _INF
        # NaN classes: 0/0 and ±inf/±inf. x/0 for x ≠ 0 is ±inf — a
        # VALUE under ieee, not NaN.
        made_nan = (a0 and b0) or (ainf and binf)
        if b0:
            # A zero-containing divisor divides to ⊤ under ieee, with no
            # case split on WHERE the zero sits — see IEEE_ZERO_DIVISOR_TOP
            # for why the boundary-aware branch that used to be here was
            # unsound and why no endpoint test can replace it.
            return -_INF, _INF, made_nan
        corners = (alo / blo, alo / bhi, ahi / blo, ahi / bhi)
        return _corner_hull(corners, made_nan)

    return _ieee_binary(a, b, f)


def ieee_add_fmt(a: IntervalArray, b: IntervalArray, min_normal: float):
    """Format-parametric ieee_add: subnormal haze uses format's band."""
    def f(alo, ahi, blo, bhi):
        corners = (alo + blo, alo + bhi, ahi + blo, ahi + bhi)
        return _corner_hull(corners, False)

    return _ieee_binary_fmt(a, b, f, min_normal)


def ieee_sub_fmt(a: IntervalArray, b: IntervalArray, min_normal: float):
    """Format-parametric ieee_sub: subnormal haze uses format's band."""
    def f(alo, ahi, blo, bhi):
        corners = (alo - blo, alo - bhi, ahi - blo, ahi - bhi)
        return _corner_hull(corners, False)

    return _ieee_binary_fmt(a, b, f, min_normal)


def ieee_mul_fmt(a: IntervalArray, b: IntervalArray, min_normal: float):
    """Format-parametric ieee_mul: subnormal haze uses format's band."""
    def f(alo, ahi, blo, bhi):
        a0, b0 = alo <= 0.0 <= ahi, blo <= 0.0 <= bhi
        ainf = alo == -_INF or ahi == _INF
        binf = blo == -_INF or bhi == _INF
        made_nan = (a0 and binf) or (b0 and ainf)
        corners = (alo * blo, alo * bhi, ahi * blo, ahi * bhi)
        return _corner_hull(corners, made_nan)

    return _ieee_binary_fmt(a, b, f, min_normal)


def ieee_div_fmt(a: IntervalArray, b: IntervalArray, min_normal: float):
    """Format-parametric ieee_div: subnormal haze uses format's band."""
    def f(alo, ahi, blo, bhi):
        a0, b0 = alo <= 0.0 <= ahi, blo <= 0.0 <= bhi
        ainf = alo == -_INF or ahi == _INF
        binf = blo == -_INF or bhi == _INF
        made_nan = (a0 and b0) or (ainf and binf)
        if b0:
            # Same rule as :func:`ieee_div`, and it is format-independent:
            # every IEEE format has two zeros. See IEEE_ZERO_DIVISOR_TOP.
            return -_INF, _INF, made_nan
        corners = (alo / blo, alo / bhi, ahi / blo, ahi / bhi)
        return _corner_hull(corners, made_nan)

    return _ieee_binary_fmt(a, b, f, min_normal)


def ieee_sqrt_fmt(a: IntervalArray, min_normal: float):
    """Format-parametric ieee_sqrt: subnormal haze uses format's band."""
    los, his = [], []
    made_nan = False
    for lo, hi in zip(a.los, a.his):
        lo, hi = _elt_haze_fmt(lo, hi, min_normal)
        nan_here = lo < 0.0
        if hi < 0.0:
            rlo, rhi = -_INF, _INF
        else:
            d_lo = lo if lo > 0.0 else 0.0
            rlo = math.sqrt(d_lo)
            rhi = _INF if hi == _INF else math.sqrt(hi)
        rlo, rhi = _elt_haze_fmt(rlo, rhi, min_normal)
        los.append(rlo)
        his.append(rhi)
        made_nan = made_nan or nan_here
    return (
        IntervalArray(shape=a.shape, los=tuple(los), his=tuple(his)),
        made_nan,
    )


def ieee_reduce_sum_fmt(
    a: IntervalArray, axes: tuple[int, ...], min_normal: float
):
    """Format-parametric ieee_reduce_sum: subnormal haze uses format's band."""
    ax = _check_axes(a, axes, "reduce_sum")
    out_shape, out_n, n_contrib = _reduced_shape(a, ax)
    if n_contrib > 2:
        raise IntervalError(REDUCE_SUM_IEEE_ORDER_DECLINE.format(n=n_contrib))
    buckets: list[list[tuple[float, float]]] = [[] for _ in range(out_n)]
    if a.size:
        for coord in _coords(a.shape):
            i = _flat_index(coord, a.shape)
            j = _flat_index(
                tuple(c for k, c in enumerate(coord) if k not in ax), out_shape
            )
            buckets[j].append((a.los[i], a.his[i]))
    los, his = [], []
    made_nan = False
    for parts in buckets:
        if not parts:
            los.append(0.0)
            his.append(0.0)
            continue
        if len(parts) == 1:
            lo, hi = _elt_haze_fmt(*parts[0], min_normal)
            los.append(lo)
            his.append(hi)
            continue
        (alo, ahi), (blo, bhi) = parts
        alo, ahi = _elt_haze_fmt(alo, ahi, min_normal)
        blo, bhi = _elt_haze_fmt(blo, bhi, min_normal)
        lo, hi, nan_here = _corner_hull(
            (alo + blo, alo + bhi, ahi + blo, ahi + bhi), False
        )
        lo, hi = _elt_haze_fmt(lo, hi, min_normal)
        los.append(lo)
        his.append(hi)
        made_nan = made_nan or nan_here
    return (
        IntervalArray(shape=out_shape, los=tuple(los), his=tuple(his)),
        made_nan,
    )


def ieee_sqrt(a: IntervalArray):
    """Native binary64 square root — the ieee counterpart of :func:`sqrt`.

    sqrt is correctly-rounded under IEEE-754, so the interval endpoints are
    the very float roots the program computes (NO outward rounding), and
    sqrt is monotone increasing on ``[0, inf)`` so the image of a box sits at
    its endpoints (:data:`IEEE_ENDPOINT_ASSUMPTION`). A NEGATIVE argument
    produces NaN under ieee (``jnp.sqrt(-1.0)`` is NaN — a VALUE here, routed
    into ``made_nan``, never leaked as an endpoint), so a box whose lower
    bound is below 0 sets the flag while its non-negative part still brackets
    the real roots. A wholly-negative box has an empty non-NaN image and
    returns top with ``made_nan=True`` (the all-NaN-corner convention of the
    other ieee kernels).

    Subnormal haze, DAZ-before / FTZ-after, as in every ieee kernel: a
    subnormal argument may flush to 0 (``sqrt`` of it reads as
    ``sqrt(0) = 0``) and a subnormal result may flush to 0. Returns
    ``(IntervalArray, made_nan)`` like the other ieee kernels.
    """
    los, his = [], []
    made_nan = False
    for lo, hi in zip(a.los, a.his):
        # DAZ face: a subnormal operand may read as 0 at runtime
        lo, hi = _elt_haze(lo, hi)
        nan_here = lo < 0.0  # a negative argument yields NaN under ieee
        if hi < 0.0:
            # wholly out of domain: every value is NaN, the non-NaN image is
            # empty -> top (any interval is a sound superset of the empty set)
            rlo, rhi = -_INF, _INF
        else:
            # the in-domain part is [max(0, lo), hi]; sqrt is monotone there
            d_lo = lo if lo > 0.0 else 0.0
            rlo = math.sqrt(d_lo)
            rhi = _INF if hi == _INF else math.sqrt(hi)
        # FTZ face: a subnormal result may flush to 0
        rlo, rhi = _elt_haze(rlo, rhi)
        los.append(rlo)
        his.append(rhi)
        made_nan = made_nan or nan_here
    return (
        IntervalArray(shape=a.shape, los=tuple(los), his=tuple(his)),
        made_nan,
    )


def ieee_reduce_sum(a: IntervalArray, axes: tuple[int, ...]):
    """The ieee counterpart of :func:`reduce_sum` — **bounded to the
    association-free cases, declining the rest**.

    The real transfer's bracket rests on ℝ addition being associative, and
    that is exactly what float addition is not: XLA is free to reassociate
    a reduction, the jaxpr records no order, and the orders do not agree
    (:data:`REDUCE_SUM_IEEE_ORDER_DECLINE` carries the finite-operands
    NaN-vs-0.0-vs-inf construction). So this kernel models only the
    reductions that contain **no association freedom at all**:

    * ``n = 0`` contributors — the empty sum is exactly ``0.0`` (measured
      on jax 0.11.0), no addition performed;
    * ``n = 1`` — the identity on the single element, no addition
      performed;
    * ``n = 2`` — exactly one addition, and IEEE addition is *commutative*
      (verified in-tree: 0 counterexamples in 5 000 random pairs plus the
      ±inf / ±0 / subnormal / ±maxfloat corners; independently re-measured
      at 211 396 pairs, also 0), so the single binary tree is the only one
      there is.

    ``n >= 3`` admits genuinely distinct association trees and raises
    :class:`IntervalError` with the gap quoted — the propagator turns that
    into a noted ⊤ (maybe-NaN) decline, never a crash.

    Returns ``(IntervalArray, made_nan)`` like the other ieee kernels, with
    the same DAZ-before / FTZ-after subnormal haze and NaN-corner routing.
    """
    ax = _check_axes(a, axes, "reduce_sum")
    out_shape, out_n, n_contrib = _reduced_shape(a, ax)
    if n_contrib > 2:
        raise IntervalError(REDUCE_SUM_IEEE_ORDER_DECLINE.format(n=n_contrib))
    buckets: list[list[tuple[float, float]]] = [[] for _ in range(out_n)]
    if a.size:
        for coord in _coords(a.shape):
            i = _flat_index(coord, a.shape)
            j = _flat_index(
                tuple(c for k, c in enumerate(coord) if k not in ax), out_shape
            )
            buckets[j].append((a.los[i], a.his[i]))
    los, his = [], []
    made_nan = False
    for parts in buckets:
        if not parts:  # the empty sum: exactly 0.0, nothing read, never NaN
            los.append(0.0)
            his.append(0.0)
            continue
        if len(parts) == 1:  # identity, but DAZ still reads the operand
            lo, hi = _elt_haze(*parts[0])
            los.append(lo)
            his.append(hi)
            continue
        (alo, ahi), (blo, bhi) = parts
        alo, ahi = _elt_haze(alo, ahi)  # DAZ face
        blo, bhi = _elt_haze(blo, bhi)
        lo, hi, nan_here = _corner_hull(
            (alo + blo, alo + bhi, ahi + blo, ahi + bhi), False
        )
        lo, hi = _elt_haze(lo, hi)  # FTZ face
        los.append(lo)
        his.append(hi)
        made_nan = made_nan or nan_here
    return (
        IntervalArray(shape=out_shape, los=tuple(los), his=tuple(his)),
        made_nan,
    )


IEEE_CONTRACTION_ASSUMPTION = (
    "ieee contraction hull: XLA may CONTRACT a multiply feeding an add/sub "
    "into a single fused multiply-add, rounding once instead of twice — "
    "measured on jax 0.11.0 CPU, where the compiled HLO contains "
    "multiply_add_fusion and (a*b)-1 at a = 1+2**-27, b = 1-2**-27 is 0.0 "
    "eager but -2**-54 under jit. Contraction is a compiler freedom over "
    "the SAME equation order, so it is not covered by the equation-order "
    "reliance. Rather than assume it away, every add/sub whose operand is a "
    "product carries the HULL of both roundings: the contracted value is "
    "computed exactly (the exact rational a*b+c rounded once) and joined "
    "with the uncontracted one, so results agree wherever the two "
    "roundings agree and are indeterminate only where they differ"
)

IEEE_CONTRACTION_DECLINE = (
    "add/sub over a product operand has no ieee transfer for this form: XLA "
    "may contract it into a fused multiply-add (measured on this target), "
    "and the contracted value cannot be bracketed here because {why} — "
    "declined rather than modelled with one of the two roundings"
)


def _fma_corner(a: float, b: float, c: float) -> float:
    """The CONTRACTED value ``fl(a*b + c)`` — one rounding, exactly.

    ``a*b + c`` over exact rationals is exact, and ``float(Fraction)`` is
    round-to-nearest, so this is precisely what a fused multiply-add
    computes. Operands are finite (the caller declines infinities).
    """
    from fractions import Fraction

    try:
        return float(Fraction(a) * Fraction(b) + Fraction(c))
    except OverflowError:
        # the exact value exceeds the double range: an fma saturates to the
        # signed infinity, exactly as the hardware does
        return _INF if (Fraction(a) * Fraction(b) + Fraction(c)) > 0 else -_INF


def ieee_fma_hull(
    a: IntervalArray,
    b: IntervalArray,
    c: IntervalArray,
    *,
    negate_product: bool = False,
    negate_addend: bool = False,
):
    """Native-precision interval of the CONTRACTED ``±(a*b) ± c`` over the
    operand boxes — the value an FMA-contracting compiler computes.

    ``a*b + c`` is monotone in each argument separately (the product is
    monotone in ``a`` for fixed ``b`` and vice versa; the sum is monotone
    in ``c``) and rounding is monotone, so the extremes over the box sit at
    the eight corners. Each corner is evaluated exactly and rounded once
    (:func:`_fma_corner`).

    Infinite operand endpoints raise :class:`IntervalError`: an fma over
    infinities has delicate corner semantics this does not model, and the
    caller turns that into a quoted decline. Returns
    ``(IntervalArray, made_nan)`` like the other ieee kernels.
    """
    shape = a.shape if a.size >= b.size else b.shape
    if c.size > _size_of(shape):
        shape = c.shape
    xs, ys, zs = (
        _bcast_elements(a, shape), _bcast_elements(b, shape),
        _bcast_elements(c, shape),
    )
    los, his = [], []
    made_nan = False
    for (alo, ahi), (blo, bhi), (clo, chi) in zip(xs, ys, zs):
        for v in (alo, ahi, blo, bhi, clo, chi):
            if v == _INF or v == -_INF:
                raise IntervalError(
                    IEEE_CONTRACTION_DECLINE.format(
                        why="an operand endpoint is infinite and fused "
                            "multiply-add corner semantics over infinities "
                            "are not modelled here"
                    )
                )
        alo, ahi = _elt_haze(alo, ahi)  # DAZ face, as in the other kernels
        blo, bhi = _elt_haze(blo, bhi)
        clo, chi = _elt_haze(clo, chi)
        if negate_product:
            alo, ahi = -ahi, -alo
        if negate_addend:
            clo, chi = -chi, -clo
        corners = [
            _fma_corner(x, y, z)
            for x in (alo, ahi) for y in (blo, bhi) for z in (clo, chi)
        ]
        lo, hi, nan_here = _corner_hull(tuple(corners), False)
        lo, hi = _elt_haze(lo, hi)  # FTZ face
        los.append(lo)
        his.append(hi)
        made_nan = made_nan or nan_here
    return (
        IntervalArray(shape=shape, los=tuple(los), his=tuple(his)),
        made_nan,
    )


def _size_of(shape: tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def hull(a: IntervalArray, b: IntervalArray) -> IntervalArray:
    """Elementwise interval hull of two same-shape boxes — the join used to
    cover two legal compiler behaviours at once."""
    if a.shape != b.shape:
        raise IntervalError(
            f"hull over mismatched shapes {a.shape} vs {b.shape}"
        )
    return IntervalArray(
        shape=a.shape,
        los=tuple(min(x, y) for x, y in zip(a.los, b.los)),
        his=tuple(max(x, y) for x, y in zip(a.his, b.his)),
    )


# -- integer semantics --------------------------------------------------------


INT_DIV_DECLINE = (
    "'div' on dtype {dtype!r}: jax integer division TRUNCATES toward zero "
    "(measured: lax.div(-7, 2) = -3, not -3.5), which real division does "
    "not model. {why} — declined"
)


def _trunc_div(a: int, b: int) -> int:
    """Integer division truncating toward zero — jax/XLA's measured
    semantics, and NOT python's ``//`` (which floors: ``-7 // 2 == -4``
    while ``lax.div(-7, 2) == -3``)."""
    q = abs(a) // abs(b)
    return -q if (a < 0) != (b < 0) else q


_EXACT_INT_FLOAT = 2**53


def _exact_int(v: float) -> int | None:
    """``v`` as an exact python int, or None if it is not an integer this
    double represents exactly."""
    if v != v or v == _INF or v == -_INF:
        return None
    if v != math.floor(v) or abs(v) > _EXACT_INT_FLOAT:
        return None
    return int(v)


def int_div(a: IntervalArray, b: IntervalArray) -> IntervalArray:
    """Integer ``div``: truncation toward zero, computed EXACTLY.

    For a denominator of fixed sign, ``a/b`` is monotone in each argument,
    and truncation toward zero is monotone non-decreasing, so the image
    extremes sit at the four corners — evaluated in exact python integer
    arithmetic, so no rounding enters at all.

    A denominator whose interval contains 0 yields ⊤ (integer division by
    zero is not a value this domain models), matching :func:`div`'s
    discipline. Endpoints that are not exactly-representable integers
    raise :class:`IntervalError` — the exact corner rule needs exact
    integers, and truncation is not something to approximate.
    """
    def f(alo, ahi, blo, bhi):
        if blo <= 0.0 <= bhi:
            return -_INF, _INF  # division by zero reachable: ⊤, as for div
        ints = [_exact_int(v) for v in (alo, ahi, blo, bhi)]
        if any(i is None for i in ints):
            raise IntervalError(
                INT_DIV_DECLINE.format(
                    dtype="integer",
                    why=(
                        f"an operand endpoint ({alo}, {ahi}) / ({blo}, {bhi}) "
                        f"is not an exactly-representable integer, so the "
                        f"exact truncating-corner rule does not apply"
                    ),
                )
            )
        ai_lo, ai_hi, bi_lo, bi_hi = ints
        qs = [
            _trunc_div(x, y)
            for x in (ai_lo, ai_hi) for y in (bi_lo, bi_hi)
        ]
        return float(min(qs)), float(max(qs))

    return _binary(a, b, f)


# -- comparisons (three-valued) ----------------------------------------------


def _compare(a: IntervalArray, b: IntervalArray, definite_true, definite_false):
    def f(alo, ahi, blo, bhi):
        if definite_true(alo, ahi, blo, bhi):
            return BOOL_TRUE
        if definite_false(alo, ahi, blo, bhi):
            return BOOL_FALSE
        return BOOL_UNKNOWN

    return _binary(a, b, f)


def lt(a: IntervalArray, b: IntervalArray) -> IntervalArray:
    return _compare(
        a, b,
        definite_true=lambda alo, ahi, blo, bhi: ahi < blo,
        definite_false=lambda alo, ahi, blo, bhi: alo >= bhi,
    )


def gt(a: IntervalArray, b: IntervalArray) -> IntervalArray:
    return lt(b, a)


def le(a: IntervalArray, b: IntervalArray) -> IntervalArray:
    return _compare(
        a, b,
        definite_true=lambda alo, ahi, blo, bhi: ahi <= blo,
        definite_false=lambda alo, ahi, blo, bhi: alo > bhi,
    )


def ge(a: IntervalArray, b: IntervalArray) -> IntervalArray:
    return le(b, a)


def eq(a: IntervalArray, b: IntervalArray) -> IntervalArray:
    """Three-valued equality. Definitely true **only** when both operands
    are the same single point (a point interval guarantees the true value
    exactly, so the two true values coincide); definitely false when the
    intervals are disjoint (supersets that never meet contain no equal
    pair); everything else — including identical non-point intervals — is
    unknown, never guessed."""
    return _compare(
        a, b,
        definite_true=lambda alo, ahi, blo, bhi: alo == ahi == blo == bhi,
        definite_false=lambda alo, ahi, blo, bhi: ahi < blo or bhi < alo,
    )


def ne(a: IntervalArray, b: IntervalArray) -> IntervalArray:
    """Three-valued inequality: the negation of :func:`eq`'s logic —
    definitely true where eq is definitely false (disjoint), definitely
    false where eq is definitely true (same single point), else unknown."""
    return _compare(
        a, b,
        definite_true=lambda alo, ahi, blo, bhi: ahi < blo or bhi < alo,
        definite_false=lambda alo, ahi, blo, bhi: alo == ahi == blo == bhi,
    )


# -- three-valued logic on {0,1}-encoded bool intervals -----------------------


def _bool3(lo: float, hi: float) -> tuple[float, float]:
    """Canonicalize one {0,1}-encoded three-valued element. Anything that
    is not exactly the definite-true or definite-false encoding (including
    a ⊤ interval flowing in from an unregistered producer) reads as
    unknown — sound whenever the true values are booleans, which the
    transfers' bool-dtype guard establishes."""
    if lo == 1.0 and hi == 1.0:
        return BOOL_TRUE
    if lo == 0.0 and hi == 0.0:
        return BOOL_FALSE
    return BOOL_UNKNOWN


def logical_and(a: IntervalArray, b: IntervalArray) -> IntervalArray:
    """Kleene AND: false ∧ anything = false; true ∧ true = true; else
    unknown. On the {0,1} encoding that is endpoint-wise min of the
    canonicalized operands."""

    def f(alo, ahi, blo, bhi):
        (alo, ahi), (blo, bhi) = _bool3(alo, ahi), _bool3(blo, bhi)
        return min(alo, blo), min(ahi, bhi)

    return _binary(a, b, f)


def logical_or(a: IntervalArray, b: IntervalArray) -> IntervalArray:
    """Kleene OR: true ∨ anything = true; false ∨ false = false; else
    unknown — endpoint-wise max of the canonicalized operands."""

    def f(alo, ahi, blo, bhi):
        (alo, ahi), (blo, bhi) = _bool3(alo, ahi), _bool3(blo, bhi)
        return max(alo, blo), max(ahi, bhi)

    return _binary(a, b, f)


def logical_not(a: IntervalArray) -> IntervalArray:
    """Kleene NOT: ¬true = false, ¬false = true, ¬unknown = unknown.
    On the {0,1} encoding: flip endpoints — ``[1 - hi, 1 - lo]``."""
    los, his = [], []
    for lo, hi in zip(a.los, a.his):
        lo, hi = _bool3(lo, hi)
        los.append(1.0 - hi)
        his.append(1.0 - lo)
    return IntervalArray(shape=a.shape, los=tuple(los), his=tuple(his))


def reduce_or(a: IntervalArray, axes: tuple[int, ...]) -> IntervalArray:
    """Three-valued OR-fold over ``axes``: output shape is the input shape
    with those axes removed. The fold identity is definite-false (an OR
    over an empty reduction range is false), so empty-range axes reduce to
    ``[0, 0]`` exactly as jax's ``reduce_or`` does."""
    ax = set(axes)
    if any(not (isinstance(d, int) and 0 <= d < len(a.shape)) for d in ax):
        raise IntervalError(
            f"reduce_or axes {axes} out of range for shape {a.shape}"
        )
    out_shape = tuple(d for i, d in enumerate(a.shape) if i not in ax)
    out_n = 1
    for d in out_shape:
        out_n *= d
    los = [0.0] * out_n  # OR identity: definitely false
    his = [0.0] * out_n
    if a.size:
        for coord in _coords(a.shape):
            i = _flat_index(coord, a.shape)
            j = _flat_index(
                tuple(c for k, c in enumerate(coord) if k not in ax), out_shape
            )
            lo, hi = _bool3(a.los[i], a.his[i])
            los[j] = max(los[j], lo)
            his[j] = max(his[j], hi)
    return IntervalArray(shape=out_shape, los=tuple(los), his=tuple(his))


def is_finite(a: IntervalArray) -> IntervalArray:
    """Three-valued ``isfinite``: definitely true when both endpoints are
    finite (every value in a bounded interval is finite); definitely false
    when both endpoints are the same infinity (the interval is a point at
    ±inf — every value IS infinite); unknown otherwise (the interval spans
    finite and infinite values)."""
    los, his = [], []
    for lo, hi in zip(a.los, a.his):
        if math.isfinite(lo) and math.isfinite(hi):
            los.append(1.0)
            his.append(1.0)
        elif not math.isfinite(lo) and not math.isfinite(hi) and lo == hi:
            # point at ±inf: definitely NOT finite
            los.append(0.0)
            his.append(0.0)
        else:
            # spans finite and infinite values: unknown
            los.append(0.0)
            his.append(1.0)
    return IntervalArray(shape=a.shape, los=tuple(los), his=tuple(his))


# -- structural ops (exact: no arithmetic, no bump) ---------------------------


def _strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides, acc = [], 1
    for d in reversed(shape):
        strides.append(acc)
        acc *= d
    return tuple(reversed(strides))


def _flat_index(coord: tuple[int, ...], shape: tuple[int, ...]) -> int:
    return sum(c * s for c, s in zip(coord, _strides(shape)))


def _coords(shape: tuple[int, ...]):
    if shape == ():
        yield ()
        return
    shape = check_shape(shape)  # non-integer or negative extents must never
    # enumerate: on (-2,-2) the loop below would yield exactly ONE
    # coordinate while the element product says 4 — the silent
    # inconsistency behind the fix-re-attack's dropped-addends UNSOUND
    # (R1); a string extent would raise a raw TypeError (N2)
    if 0 in shape:  # a zero-size array has no elements — no coordinates
        return  # (audit-gate finding 2: the phantom first coordinate
        # produced an IndexError that bypassed the decline channel)
    idx = [0] * len(shape)
    while True:
        yield tuple(idx)
        for axis in range(len(shape) - 1, -1, -1):
            idx[axis] += 1
            if idx[axis] < shape[axis]:
                break
            idx[axis] = 0
        else:
            return


def slice_(
    a: IntervalArray,
    start_indices: tuple[int, ...],
    limit_indices: tuple[int, ...],
    strides: tuple[int, ...] | None,
) -> IntervalArray:
    """Static slice. Params inconsistent with the operand's shape raise
    :class:`IntervalError` — the normal decline channel — rather than
    escaping as an ``IndexError`` from the element read below.

    A legal jax trace cannot produce such params (jax clamps its own), but
    :meth:`stelling.ir.ClosedJaxpr.from_dict` is a public deserialisation
    entry point, and a query arriving through it must degrade like any
    other unsupported form instead of killing the walk (audit FRAGILE 1;
    the registered degrade-don't-crash posture).
    """
    rank = len(a.shape)
    steps = tuple(strides) if strides else (1,) * rank
    if not (len(start_indices) == len(limit_indices) == len(steps) == rank):
        raise IntervalError(
            f"slice params do not match operand rank {rank}: "
            f"start_indices={tuple(start_indices)}, "
            f"limit_indices={tuple(limit_indices)}, strides={steps}"
        )
    for ax, (lo, hi, st) in enumerate(zip(start_indices, limit_indices, steps)):
        if st < 1 or not (0 <= lo <= hi <= a.shape[ax]):
            raise IntervalError(
                f"slice axis {ax} selects [{lo}, {hi}) step {st}, which is "
                f"outside the operand's extent {a.shape[ax]} (shape "
                f"{a.shape}) — no element rule for an out-of-range selection"
            )
    out_shape = tuple(
        -(-(hi - lo) // st) for lo, hi, st in zip(start_indices, limit_indices, steps)
    )
    los, his = [], []
    for coord in _coords(out_shape):
        src = tuple(lo + c * st for c, lo, st in zip(coord, start_indices, steps))
        i = _flat_index(src, a.shape)
        los.append(a.los[i])
        his.append(a.his[i])
    return IntervalArray(shape=out_shape, los=tuple(los), his=tuple(his))


def reshape(a: IntervalArray, new_sizes: tuple[int, ...]) -> IntervalArray:
    """Data-preserving shape change: element storage is already flat
    C-order, so a C-order reshape is the identity on the element tuples.
    (Reshapes with a ``dimensions`` permutation are not this function —
    the transfer declines them before calling here.)

    ``new_sizes`` must be nonnegative — measured jax 0.11.0 rejects
    negative entries ("reshape new_sizes must all be positive", while a
    zero extent is accepted: ``lax.reshape(zeros((0,)), (0, 3))`` works),
    and the element-count check alone would admit sign-coincidences like
    ``(-1, -4)`` on 4 elements (fix-re-attack R2): checked explicitly,
    before the count."""
    n = 1
    for d in new_sizes:
        if d < 0:
            raise IntervalError(
                f"reshape new_sizes {tuple(new_sizes)} contain a negative "
                f"extent — jax rejects this form"
            )
        n *= d
    if n != a.size:
        raise IntervalError(
            f"reshape {a.shape} -> {tuple(new_sizes)} changes element count"
        )
    return IntervalArray(shape=tuple(new_sizes), los=a.los, his=a.his)


def squeeze(a: IntervalArray, dimensions: tuple[int, ...]) -> IntervalArray:
    """Remove size-1 axes: flat C-order elements are untouched. The
    contract is jax's, measured on 0.11.0: each named axis must be
    in-range, distinct, and of size 1 (lax rejects all three violations;
    negative axes are normalized away before the traced equation, so an
    IR carrying one is ``from_dict``-only and refused here). Violations
    raise :class:`IntervalError` — the decline channel — where an
    out-of-range axis was previously IGNORED silently."""
    dims = tuple(int(d) for d in dimensions)
    rank = len(a.shape)
    if len(set(dims)) != len(dims):
        raise IntervalError(f"squeeze dimensions {dims} are not distinct")
    for d in dims:
        if not 0 <= d < rank:
            raise IntervalError(
                f"squeeze dimension {d} is out of range for rank {rank} "
                f"(shape {a.shape})"
            )
        if a.shape[d] != 1:
            raise IntervalError(
                f"squeeze dimension {d} has size {a.shape[d]}, not 1 "
                f"(shape {a.shape}) — jax rejects this form"
            )
    out_shape = tuple(d for i, d in enumerate(a.shape) if i not in set(dims))
    return IntervalArray(shape=out_shape, los=a.los, his=a.his)


def broadcast_in_dim(
    a: IntervalArray,
    out_shape: tuple[int, ...],
    broadcast_dimensions: tuple[int, ...],
) -> IntervalArray:
    """Static broadcast: pure data movement per jax's own contract.

    Params outside that contract raise :class:`IntervalError` — the
    normal decline channel — instead of silently mis-routing. This
    function is the ONE routing oracle for broadcast (the propagation
    transfer, the SMT emission, and the witness replay all drive it), so
    an unvalidated precondition here is wrong in all three at once: a
    ``broadcast_dimensions`` shorter than the operand rank used to
    zip-truncate the source coordinate and alias every output element to
    element 0 — a silently dropped dependence, the worst routing shape.

    The enforced contract is jax 0.11.0's, measured (each rejected by
    ``jax.lax.broadcast_in_dim``; every trigger is ``from_dict``-only —
    the one form lax ACCEPTS with a short bd, the equal-shape identity,
    short-circuits and records no equation):

    * ``len(broadcast_dimensions) == operand rank``;
    * entries distinct, nonnegative, and ``< len(out_shape)``;
    * each operand extent is 1 or equals its output dimension's extent.

    NON-MONOTONIC ``broadcast_dimensions`` are LEGAL (measured: lax
    accepts and traces ``bd=(1, 0)`` — transpose-broadcast semantics)
    and are routed exactly as jax routes them; the validation must never
    refuse them.
    """
    bd = tuple(int(d) for d in broadcast_dimensions)
    out_shape = tuple(int(d) for d in out_shape)
    rank = len(a.shape)
    if len(bd) != rank:
        raise IntervalError(
            f"broadcast_in_dim broadcast_dimensions {bd} must have length "
            f"equal to the operand rank {rank} (shape {a.shape}) — jax "
            f"rejects this form; routing it would alias elements"
        )
    if len(set(bd)) != len(bd):
        raise IntervalError(
            f"broadcast_in_dim broadcast_dimensions {bd} contain duplicates "
            f"— jax rejects this form; routing it would alias elements"
        )
    for in_axis, out_axis in enumerate(bd):
        if not 0 <= out_axis < len(out_shape):
            raise IntervalError(
                f"broadcast_in_dim broadcast_dimensions {bd} name axis "
                f"{out_axis}, outside the output rank {len(out_shape)} "
                f"(shape {out_shape})"
            )
        if a.shape[in_axis] != 1 and a.shape[in_axis] != out_shape[out_axis]:
            raise IntervalError(
                f"broadcast_in_dim operand extent {a.shape[in_axis]} of axis "
                f"{in_axis} is neither 1 nor the output extent "
                f"{out_shape[out_axis]} of axis {out_axis} (operand "
                f"{a.shape} -> output {out_shape}, broadcast_dimensions {bd})"
            )
    los, his = [], []
    for coord in _coords(out_shape):
        src = tuple(
            coord[out_axis] if a.shape[in_axis] != 1 else 0
            for in_axis, out_axis in enumerate(bd)
        )
        i = _flat_index(src, a.shape)
        los.append(a.los[i])
        his.append(a.his[i])
    return IntervalArray(shape=out_shape, los=tuple(los), his=tuple(his))


def transpose(a: IntervalArray, permutation: tuple[int, ...]) -> IntervalArray:
    """Axis permutation: ``out.shape[j] = a.shape[permutation[j]]`` and
    ``out[coord]`` reads ``a`` at ``src`` with ``src[permutation[j]] =
    coord[j]`` — numpy/XLA transpose semantics, pure data movement, no
    arithmetic. A malformed ``permutation`` (not a permutation of the
    axes) raises :class:`IntervalError` — a decline the propagator notes,
    not a crash."""
    perm = tuple(permutation)
    if sorted(perm) != list(range(len(a.shape))):
        raise IntervalError(
            f"transpose permutation {perm} is not a permutation of the "
            f"{len(a.shape)} axes of shape {a.shape}"
        )
    out_shape = tuple(a.shape[p] for p in perm)
    los, his = [], []
    for coord in _coords(out_shape):
        src = [0] * len(perm)
        for j, p in enumerate(perm):
            src[p] = coord[j]
        i = _flat_index(tuple(src), a.shape)
        los.append(a.los[i])
        his.append(a.his[i])
    return IntervalArray(shape=out_shape, los=tuple(los), his=tuple(his))


def take_rows(a: IntervalArray, ks: list[int]) -> IntervalArray:
    """Leading-axis row take: ``out[i] = a[ks[i]]`` with each row's
    trailing block copied whole — the interval meaning of a static-index
    gather along axis 0. Pure data movement (rows are contiguous in the
    flat C-order layout), no arithmetic. Rank-0 input or an out-of-range
    row raises :class:`IntervalError` (the transfer's decline channel;
    the registered gather transfer checks ranges before calling here)."""
    if not a.shape:
        raise IntervalError("take_rows needs a leading axis; got rank-0 input")
    rowsz = 1
    for d in a.shape[1:]:
        rowsz *= d
    los: list[float] = []
    his: list[float] = []
    for k in ks:
        if not 0 <= k < a.shape[0]:
            raise IntervalError(
                f"take_rows row {k} out of range for leading axis "
                f"{a.shape[0]}"
            )
        los.extend(a.los[k * rowsz:(k + 1) * rowsz])
        his.extend(a.his[k * rowsz:(k + 1) * rowsz])
    return IntervalArray(
        shape=(len(ks),) + a.shape[1:], los=tuple(los), his=tuple(his)
    )


def take_row_ranges(
    a: IntervalArray, ranges: list[tuple[int, int]]
) -> IntervalArray:
    """Leading-axis row take where each taken row is known only to a RANGE:
    ``out[i] = hull(a[lo_i], …, a[hi_i])``, trailing block by trailing
    block. The generalisation of :func:`take_rows` to a gather whose index
    is not a single point but a declared integer interval.

    Soundness of the hull is the whole point and it is one sentence: for
    any one input the program takes exactly ONE row ``k`` with ``lo_i <= k
    <= hi_i``, and every element of that row lies inside the elementwise
    hull of rows ``lo_i … hi_i`` — so the returned box contains the taken
    row for **every** index the declared set admits, not merely for the
    ones an enumeration happened to visit. When ``lo_i == hi_i`` the hull
    is that single row and this agrees with :func:`take_rows` element for
    element (pinned as a control).

    Both endpoints must name real rows: ``0 <= lo_i <= hi_i <
    a.shape[0]``. Out-of-range endpoints raise :class:`IntervalError` (the
    decline channel) rather than being clamped here — the registered
    gather transfer classifies in-range / straddling / disjoint BEFORE
    calling, because clamping is jax's runtime behaviour and modelling it
    would answer about the executed program instead of the written one."""
    if not a.shape:
        raise IntervalError(
            "take_row_ranges needs a leading axis; got rank-0 input"
        )
    rowsz = 1
    for d in a.shape[1:]:
        rowsz *= d
    los: list[float] = []
    his: list[float] = []
    for lo_k, hi_k in ranges:
        if not 0 <= lo_k <= hi_k < a.shape[0]:
            raise IntervalError(
                f"take_row_ranges rows [{lo_k}, {hi_k}] out of range for "
                f"leading axis {a.shape[0]}"
            )
        base = lo_k * rowsz
        rlo = list(a.los[base:base + rowsz])
        rhi = list(a.his[base:base + rowsz])
        for k in range(lo_k + 1, hi_k + 1):
            base = k * rowsz
            for j in range(rowsz):
                v = a.los[base + j]
                if v < rlo[j]:
                    rlo[j] = v
                v = a.his[base + j]
                if v > rhi[j]:
                    rhi[j] = v
        los.extend(rlo)
        his.extend(rhi)
    return IntervalArray(
        shape=(len(ranges),) + a.shape[1:], los=tuple(los), his=tuple(his)
    )


def stack(parts: list[IntervalArray], axis: int) -> IntervalArray:
    """Join ``k`` same-shape boxes along a NEW axis at position ``axis``:
    ``out[..., i, ...] = parts[i]`` — pure element routing, no arithmetic,
    the interval meaning of jax's ``stack`` primitive (jnp.stack traces to
    it on the measured jax 0.11.0; the traced ``axis`` param arrives
    already normalized — ``jnp.stack(..., axis=-1)`` on rank-2 operands
    records ``axis=2`` — so only canonical positions ``0 <= axis <=
    rank`` exist in traced IR, and anything else is refused as
    ``from_dict``-only). A non-empty operand list and agreeing shapes are
    required, as for :func:`concatenate`; violations raise
    :class:`IntervalError` (the decline channel), never a crash."""
    if not parts:
        raise IntervalError("stack with no operands")
    axis = int(axis)
    base = parts[0].shape
    if any(p.shape != base for p in parts[1:]):
        raise IntervalError(
            f"stack operand shapes {[p.shape for p in parts]} disagree "
            f"(stack joins equal shapes along a new axis)"
        )
    if not 0 <= axis <= len(base):
        raise IntervalError(
            f"stack axis {axis} out of bounds for operand rank {len(base)} "
            f"(the new axis can sit at positions 0..{len(base)})"
        )
    out_shape = base[:axis] + (len(parts),) + base[axis:]
    los, his = [], []
    for coord in _coords(out_shape):
        p = parts[coord[axis]]
        src = coord[:axis] + coord[axis + 1:]
        i = _flat_index(src, base)
        los.append(p.los[i])
        his.append(p.his[i])
    return IntervalArray(shape=out_shape, los=tuple(los), his=tuple(his))


def concatenate(parts: list[IntervalArray], dimension: int) -> IntervalArray:
    """Concatenate along a static axis: pure data movement per jax's
    contract, measured on 0.11.0 — a non-empty operand list, equal ranks,
    ``0 <= dimension < rank`` (lax rejects negatives), and equal extents
    off the concatenation axis. Violations raise :class:`IntervalError`
    (the decline channel): the off-axis-extent case previously read
    elements through the WRONG shape silently — a mis-join no legal jax
    trace can produce but ``from_dict`` can."""
    if not parts:
        raise IntervalError("concatenate with no operands")
    dimension = int(dimension)
    base = parts[0].shape
    rank = len(base)
    if not 0 <= dimension < rank:
        raise IntervalError(
            f"concatenate dimension {dimension} out of bounds for rank "
            f"{rank} (shapes {[p.shape for p in parts]})"
        )
    for p in parts[1:]:
        if len(p.shape) != rank or any(
            ax != dimension and p.shape[ax] != base[ax] for ax in range(rank)
        ):
            raise IntervalError(
                f"concatenate operand shapes {[q.shape for q in parts]} "
                f"disagree off the concatenation dimension {dimension}"
            )
    out_shape = tuple(
        sum(p.shape[dimension] for p in parts) if ax == dimension else d
        for ax, d in enumerate(base)
    )
    los, his = [], []
    for coord in _coords(out_shape):
        offset = coord[dimension]
        for p in parts:
            if offset < p.shape[dimension]:
                src = tuple(
                    offset if ax == dimension else c for ax, c in enumerate(coord)
                )
                i = _flat_index(src, p.shape)
                los.append(p.los[i])
                his.append(p.his[i])
                break
            offset -= p.shape[dimension]
    return IntervalArray(shape=out_shape, los=tuple(los), his=tuple(his))


# Element-visit budget for the two dynamic-index hulls below. Their cost is
# the product of the output size and the widths of the declared start
# ranges, which a wide declaration over a large operand can make quadratic
# in the operand size — an unbounded loop in a verifier is a hang, and a
# hang is a worse failure than a ⊤. Exceeding the budget DECLINES (sound:
# the caller turns it into a noted ⊤), it never truncates the hull, so the
# budget can only cost precision and can never cost soundness.
# degrade-don't-hang, the posture `_integer_pow_budget` and the
# membership-hint walk cap already take.
DYNAMIC_INDEX_WORK_CAP = 1 << 22


def _range_widths(start_ranges: tuple[tuple[int, int], ...]) -> list[int]:
    return [hi - lo + 1 for lo, hi in start_ranges]


def dynamic_slice_hull(
    a: IntervalArray,
    start_ranges: tuple[tuple[int, int], ...],
    slice_sizes: tuple[int, ...],
) -> IntervalArray:
    """``lax.dynamic_slice`` where each axis' start index is known only to a
    declared integer RANGE: the elementwise hull, over every admitted start,
    of the window that start selects.

        ``out[c] = hull{ a[s + c] : s_d ∈ [lo_d, hi_d] for every axis d }``

    **Every start range must already lie inside the axis' LEGAL start
    window** ``0 <= lo_d <= hi_d <= a.shape[d] - slice_sizes[d]``; a range
    that leaves it raises :class:`IntervalError`. That is deliberate and it
    is the soundness argument of the whole transfer: jax CLAMPS an
    out-of-window start, and the clamped read is a fact about the executed
    program, not about the program the user wrote. This function is only
    ever the rule for starts where the clamp is the **identity**, so no
    value it returns depends on clamping having happened.

    Soundness of the hull, stated so it can be checked: for one input the
    program performs exactly one read, at one start ``s`` in the declared
    box, and ``out[c]`` is a hull taken over a source set that CONTAINS
    ``a[s + c]`` for every such ``s`` — the enumeration is over the
    declared range's endpoints-inclusive integer span, never over a sample
    of it. Axes are hulled ONE AT A TIME, which computes the same box as
    one pass over the whole product: a minimum over an axis-aligned box is
    the minimum over one axis of the minima over the rest.

    Interval propagation has already forgotten any coupling BETWEEN the
    axes' start indices (each arrived as its own box), so the product of
    the per-axis ranges over-approximates the reachable set of starts —
    which is the sound direction: a superset of starts gives a superset
    hull.

    Precision, for the record: on a single axis the result is TIGHT — every
    endpoint of every output element is attained by some admitted start —
    so nothing here is thrown away that intervals could have kept."""
    rank = len(a.shape)
    slice_sizes = tuple(int(s) for s in slice_sizes)
    if not (len(start_ranges) == len(slice_sizes) == rank):
        raise IntervalError(
            f"dynamic_slice_hull needs one start range and one slice size "
            f"per axis of a rank-{rank} operand (shape {a.shape}); got "
            f"{len(start_ranges)} range(s) and {len(slice_sizes)} size(s)"
        )
    for d, ((lo_d, hi_d), s_d) in enumerate(zip(start_ranges, slice_sizes)):
        if s_d < 0 or s_d > a.shape[d]:
            raise IntervalError(
                f"dynamic_slice_hull axis {d} takes {s_d} element(s) of an "
                f"extent-{a.shape[d]} axis (operand shape {a.shape})"
            )
        if not 0 <= lo_d <= hi_d <= a.shape[d] - s_d:
            raise IntervalError(
                f"dynamic_slice_hull axis {d} start range [{lo_d}, {hi_d}] "
                f"leaves the legal start window [0, {a.shape[d] - s_d}] — "
                f"the hull rule is defined only where jax's clamp is the "
                f"identity"
            )
    widths = _range_widths(start_ranges)
    work, running = 0, 1
    for d in range(rank):
        running *= slice_sizes[d]
        tail = 1
        for e in range(d + 1, rank):
            tail *= a.shape[e]
        work += running * widths[d] * tail
    if work > DYNAMIC_INDEX_WORK_CAP:
        raise IntervalError(
            f"dynamic_slice_hull over operand shape {a.shape}, slice sizes "
            f"{slice_sizes} and start ranges {tuple(start_ranges)} would "
            f"visit {work} elements, past the {DYNAMIC_INDEX_WORK_CAP} "
            f"budget — declined rather than run unbounded"
        )
    cur_shape = a.shape
    los = list(a.los)
    his = list(a.his)
    for d in range(rank):
        lo_d, _ = start_ranges[d]
        w = widths[d]
        new_shape = cur_shape[:d] + (slice_sizes[d],) + cur_shape[d + 1:]
        n_new = 1
        for e in new_shape:
            n_new *= e
        nlos = [_INF] * n_new
        nhis = [-_INF] * n_new
        for coord in _coords(new_shape):
            best_lo, best_hi = _INF, -_INF
            head, tail = coord[:d], coord[d + 1:]
            for t in range(w):
                i = _flat_index(head + (lo_d + coord[d] + t,) + tail, cur_shape)
                if los[i] < best_lo:
                    best_lo = los[i]
                if his[i] > best_hi:
                    best_hi = his[i]
            o = _flat_index(coord, new_shape)
            nlos[o] = best_lo
            nhis[o] = best_hi
        cur_shape, los, his = new_shape, nlos, nhis
    return IntervalArray(shape=cur_shape, los=tuple(los), his=tuple(his))


def dynamic_update_slice_hull(
    operand: IntervalArray,
    update: IntervalArray,
    start_ranges: tuple[tuple[int, int], ...],
) -> IntervalArray:
    """``lax.dynamic_update_slice`` where each axis' start index is known
    only to a declared integer RANGE: the operand with ``update`` written at
    some admitted start, hulled over every admitted start.

    Two things can be true of an output position ``c``, and the rule keeps
    both:

    * some admitted start writes ``update[j]`` there, for every ``j`` with
      ``j_d ∈ [max(0, c_d - hi_d), min(s_d - 1, c_d - lo_d)]``; and
    * some admitted start does NOT write there, leaving ``operand[c]`` —
      which happens unless ``c`` falls in the window of **every** admitted
      start, i.e. unless ``c_d ∈ [hi_d, lo_d + s_d - 1]`` on every axis.

    ``out[c]`` is the hull of whichever of those are reachable. When the
    start is a single point this is exact: the written region takes the
    update's values and nothing else, the rest keeps the operand's.

    As for :func:`dynamic_slice_hull`, every start range must already sit
    inside the legal window ``[0, operand.shape[d] - update.shape[d]]``;
    outside it jax clamps, and the clamp is not modelled."""
    rank = len(operand.shape)
    if len(update.shape) != rank or len(start_ranges) != rank:
        raise IntervalError(
            f"dynamic_update_slice_hull needs an update and one start range "
            f"of the operand's rank {rank} (operand shape {operand.shape}); "
            f"got update shape {update.shape} and {len(start_ranges)} range(s)"
        )
    for d, ((lo_d, hi_d), s_d) in enumerate(zip(start_ranges, update.shape)):
        if s_d > operand.shape[d]:
            raise IntervalError(
                f"dynamic_update_slice_hull axis {d} writes {s_d} element(s) "
                f"into an extent-{operand.shape[d]} axis (operand shape "
                f"{operand.shape}, update shape {update.shape})"
            )
        if not 0 <= lo_d <= hi_d <= operand.shape[d] - s_d:
            raise IntervalError(
                f"dynamic_update_slice_hull axis {d} start range [{lo_d}, "
                f"{hi_d}] leaves the legal start window "
                f"[0, {operand.shape[d] - s_d}] — the hull rule is defined "
                f"only where jax's clamp is the identity"
            )
    work = operand.size
    for s_d, w in zip(update.shape, _range_widths(start_ranges)):
        work *= min(s_d, w)
    if work > DYNAMIC_INDEX_WORK_CAP:
        raise IntervalError(
            f"dynamic_update_slice_hull over operand shape {operand.shape}, "
            f"update shape {update.shape} and start ranges "
            f"{tuple(start_ranges)} would visit {work} elements, past the "
            f"{DYNAMIC_INDEX_WORK_CAP} budget — declined rather than run "
            f"unbounded"
        )
    los: list[float] = []
    his: list[float] = []
    for c in _coords(operand.shape):
        j_ranges = []
        reachable = True
        covered_always = True
        for d, (lo_d, hi_d) in enumerate(start_ranges):
            s_d = update.shape[d]
            j_lo = max(0, c[d] - hi_d)
            j_hi = min(s_d - 1, c[d] - lo_d)
            if j_lo > j_hi:
                reachable = False
                break
            j_ranges.append((j_lo, j_hi))
            if not hi_d <= c[d] <= lo_d + s_d - 1:
                covered_always = False
        i = _flat_index(c, operand.shape)
        if covered_always and reachable:
            best_lo, best_hi = _INF, -_INF
        else:
            best_lo, best_hi = operand.los[i], operand.his[i]
        if reachable:
            for j in _coords(tuple(hi - lo + 1 for lo, hi in j_ranges)):
                src = tuple(lo + o for (lo, _), o in zip(j_ranges, j))
                k = _flat_index(src, update.shape)
                if update.los[k] < best_lo:
                    best_lo = update.los[k]
                if update.his[k] > best_hi:
                    best_hi = update.his[k]
        los.append(best_lo)
        his.append(best_hi)
    return IntervalArray(
        shape=operand.shape, los=tuple(los), his=tuple(his)
    )
