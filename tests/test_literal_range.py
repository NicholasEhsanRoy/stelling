# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""A LITERAL'S SCALAR VALUE IS A VALUE OF ITS AVAL'S DTYPE — WHERE THIS
MODULE NAMES THE DTYPE — OR IT IS REFUSED.

**BOTH QUALIFICATIONS IN THAT HEADLINE ARE LOAD-BEARING, AND THE
UNQUALIFIED SENTENCE WAS WHAT STOOD HERE.** An `Array` value is never
decoded (:func:`test_an_Array_literal_is_never_decoded`), and on the
real trace path `_jax_compat` writes even a shape-``()`` scalar as an
`ir.Array`, so most of the literals this suite constructs are outside
this check by design — the recorder census that measured the ratio, and
its derivation, are written at the check in `ir.py` rather than re-typed
here. A dtype string `ir`'s own tables do not name gets NO CLAIM at all,
so ``Literal(2**200, Aval(dtype="int128"))`` and ``Literal(1e300,
Aval(dtype="complex32"))`` construct
(:func:`test_an_unrecognised_dtype_string_gets_no_claim`). Both are
deliberate; neither is visible from the headline it used to have.

`ir._validate_value_against_aval` cross-checks SHAPE and never asked
whether the value was *in* the dtype, so::

    ir.Literal(256, ir.Aval(kind="ShapedArray", shape=(), dtype="int8"))

CONSTRUCTED, and every downstream reader that assumes literal-fits-aval
was reading a lie. Nothing in-tree noticed because jax narrows a python
scalar to the dtype before it writes a literal, supplying the invariant
for free; a frontend that transcribes a foreign graph cannot, since
writing ``256`` makes the aval a lie and writing ``0`` makes the source a
lie.

**ONE TEST PER ROW OF THE RULE TABLE, DRIVEN.** A rule with no positive
control is a rule nobody has seen fire, and the rows that must NOT fire
(NaN, both infinities, ordinary rounding) are driven just as hard,
because a range check whose failure mode is over-refusal breaks correct
programs at construction and cannot be caught downstream — the object
never exists.

**AND ONE PER VALUE TYPE ON EVERY ROW THAT DISPATCHES ON ONE.** The
first pass stated the standard above and did not meet it: the no-claim
row was driven by two ints, and the arm that refuses a complex value ran
BEFORE the dtype had been recognised, so every unrecognised dtype
string — `complex32`, `complex256`, `key<fry>` — was told *"no value of
that dtype is complex"*, a claim about a dtype this module has just said
it knows nothing about. Two int drives never enter that arm and so could
not see it. The rule table dispatches on the VALUE's type as hard as on
the dtype's name, so a row driven by one value type is a row driven by
one arm; see
:func:`test_an_unrecognised_dtype_string_gets_no_claim`, now a cross
product.

**AND THE TABLES ARE CHECKED AGAINST SOMETHING THAT IS NOT THEM.**
`ir` may import nothing outside the standard library, so its float and
integer bounds are a second copy of `propagate`'s; the copies are
compared here rather than trusted. The bfloat16 image, which has no
`struct` code and could only be written by hand, is compared against
`jax.numpy.bfloat16` — the bfloat16 this tree's dtype NAMES come from —
and the comparison found the "obviously more correct" direct rounder to
be the wrong one. See
:func:`test_bfloat16_agrees_with_jaxs_own_bfloat16_and_direct_rounding_would_not`.

**AND A MEASURED CLAIM THAT LIVES ONLY IN A DOCSTRING IS A CLAIM THAT
CAN ROT, WHICH ONE DID.** `ir._float_image`'s paragraph on the float32
subnormal band said a value landing there "records INEXACT"; it records
NOTHING at all 24 of the magnitudes that paragraph itself names, every
one of them an exact float32 — 23 subnormal and ``2**-126`` the smallest
NORMAL, which an earlier wording of this sentence called a subnormal too.
Three routes' worth of measurement sat above the sentence and the
sentence said the opposite, because nothing re-ran it. It is re-run now,
admissions and note and route split alike, in
:func:`test_the_subnormal_band_is_admitted_and_the_note_is_the_IEEE_route`.
"""
from __future__ import annotations

import ast
import math
import random
import struct
import sys
import warnings

import pytest

from stelling import ir
from stelling.propagate import _FLOAT_FORMATS, _INT_DTYPE_BOUNDS

FLOATS = ("float16", "bfloat16", "float32", "float64")
# the three whose values a python `float` can be outside of; float64's
# absence from the overflow and underflow rows is a fact about binary64
# and is asserted by its own test rather than skipped
NARROW = ("float16", "bfloat16", "float32")
INTS = tuple(ir._LIT_INT_BOUNDS)
HUGE = 10 ** 400  # the value whose `float()` raises; see the value-first rule


def aval(dtype, shape=()):
    return ir.Aval(kind="ShapedArray", shape=shape, dtype=dtype)


def refused(val, dtype, shape=()):
    """The refusal text for a literal the door must not admit."""
    with pytest.raises(ir.TranscriptionError) as exc:
        ir.Literal(val=val, aval=aval(dtype, shape))
    return str(exc.value)


def admitted(val, dtype, shape=()):
    """The literal, for a value the door must admit."""
    return ir.Literal(val=val, aval=aval(dtype, shape))


# -- the hole, closed ---------------------------------------------------------


def test_the_literal_that_used_to_construct_is_refused_and_says_what_it_stores():
    """SPEC-LIT's headline subject. The message quotes the WRAP — what the
    value would actually store as — because "out of range" alone tells a
    frontend author nothing about which of the two lies they are writing."""
    msg = refused(256, "int8")
    assert "int8" in msg and "[-128, 127]" in msg
    assert "would store as 0" in msg, msg
    # and it is the module's existing refusal class, not a new one
    with pytest.raises(TypeError):  # TranscriptionError subclasses TypeError
        ir.Literal(val=256, aval=aval("int8"))


# -- row: a value that is not a python number --------------------------------


def test_a_non_numeric_value_gets_no_claim():
    """`str` carries no dtype claim this pass can judge, and an `Array`
    is out STRUCTURALLY — see the next test."""
    assert ir.Literal(val="a token", aval=aval("int8")).val == "a token"


def test_an_Array_literal_is_never_decoded():
    """An `Array`'s bytes ARE a dtype: `_validate_array_value` already
    holds ``len(data) == product(shape) x itemsize``, so a fixed-width
    buffer cannot encode a value outside THE WIDTH IT IS MEASURED
    AGAINST — which is the `Array`'s OWN ``.str``, not the aval's dtype.
    See the next test for what that qualification costs.

    Driven rather than asserted: this payload holds 100000 under a `<i4`
    Array whose aval says `int8`. Decoding it under the AVAL's dtype
    would refuse; the aval-vs-Array dtype disagreement is a different and
    still-open class, and this pass makes no claim about it. The literal
    must therefore construct — and the check must reach that verdict off
    ``type(val)`` alone, having read no byte of the payload."""
    arr = ir.Array(dtype="<i4", shape=(1,), data=struct.pack("<i", 100000))
    lit = ir.Literal(val=arr, aval=aval("int8", (1,)))
    assert lit.val is arr
    assert ir._literal_range_problem(arr, aval("int8", (1,))) == (None, None)
    assert ir.literal_inexact(lit) is None


def test_an_Array_whose_own_dtype_disagrees_with_the_aval_still_constructs():
    """**"ONLY A SCALAR CAN LIE" IS FALSE AS AN UNQUALIFIED SENTENCE, and
    this is the witness.** It stood at the check and in the commit
    message that armed it, stated as a structural fact; the structural
    fact is narrower. An `Array`'s buffer is honest about the `Array`'s
    OWN dtype, and nothing in this pass compares that dtype to the
    aval's, so the same lie the scalar row refuses can be written through
    an `Array` — at shape ``()``, where the two spellings denote exactly
    the same value.

    The class is PRE-EXISTING and this test does not close it: decoding
    the payload would need numpy, which `ir`'s docstring forbids, and the
    aval-vs-`Array`-dtype disagreement is a separate open question. What
    is fixed here is the CLAIM, not the code.

    **AND THE ESCAPE IS WHERE THE POPULATION IS.** `stelling._jax_compat`
    transcribes a scalar as a shape-``()`` `ir.Array`, so the recorder
    census counted 38,133 `Array` values against 4,907 scalars out of
    43,047 literals — a shade under 89 % of them outside this check by
    construction. A torch frontend that transcribed the same way would
    land in the same place. A door that reads as though it covers its
    whole population and covers about a ninth of it is worth saying out
    loud."""
    scalar_aval = aval("int8")
    # the scalar spelling of 256-under-int8, refused
    assert "[-128, 127]" in refused(256, "int8")
    # ...and the Array spelling of the SAME value under the SAME aval,
    # admitted, because the buffer is measured against its own `<q`
    arr = ir.Array(dtype="<q", shape=(), data=struct.pack("<q", 256))
    lit = ir.Literal(val=arr, aval=scalar_aval)
    assert lit.val is arr
    assert ir._literal_range_problem(arr, scalar_aval) == (None, None)
    # the disagreement is between two NAMES, and both are readable; this
    # pass simply never compares them
    assert arr.dtype == "<q" and scalar_aval.dtype == "int8"


# -- row: a dtype-less aval ---------------------------------------------------


@pytest.mark.parametrize("dtype", [None, ""])
def test_a_dtypeless_aval_gets_no_claim(dtype):
    """Tokens carry an aval with no dtype, and `""` is what a document can
    spell that with. There is nothing to be in or out of."""
    assert ir.Literal(val=HUGE, aval=aval(dtype)).val == HUGE


# -- row: bool ----------------------------------------------------------------


@pytest.mark.parametrize("val", [True, False, 0, 1])
def test_bool_admits_the_two_values_and_their_two_ints(val):
    assert admitted(val, "bool").val == val


@pytest.mark.parametrize("val", [2, -1, 0.0, 1.0, HUGE])
def test_bool_refuses_everything_else(val):
    """`2` is not `True`; it is a value `bool` cannot hold. `0.0` and
    `1.0` are refused too: a float is not how a bool is spelled, and
    admitting them would make the rule "anything that is truthy", which
    is a coercion and not a range check."""
    assert "not a bool value" in refused(val, "bool")


# -- row: the integer dtypes --------------------------------------------------


@pytest.mark.parametrize("dtype", INTS)
def test_every_integer_dtype_admits_its_own_endpoints_and_refuses_one_past(dtype):
    """A positive control per integer dtype, from the dtype's own bounds
    rather than from a table typed beside it — and the wrap the message
    quotes is checked against two's complement, computed here."""
    lo, hi = ir._LIT_INT_BOUNDS[dtype]
    admitted(lo, dtype)
    admitted(hi, dtype)
    bits = ir._LIT_INT_BITS[dtype]
    assert hi - lo + 1 == 1 << bits
    for past, wrap in ((hi + 1, lo), (lo - 1, hi)):
        msg = refused(past, dtype)
        assert f"[{lo}, {hi}]" in msg, msg
        assert f"would store as {wrap}" in msg, msg


@pytest.mark.parametrize("dtype", INTS)
def test_an_integer_dtype_refuses_a_non_integral_float(dtype):
    """`0.5` is not an `int8`; neither is a NaN and neither is an
    infinity, and `float.is_integer()` answers False for all three, which
    is why one predicate covers them."""
    for val in (0.5, math.nan, math.inf, -math.inf):
        assert "not an integral value" in refused(val, dtype)


@pytest.mark.parametrize("dtype", INTS)
def test_an_integer_dtype_admits_an_integral_float_in_range(dtype):
    """The float SPELLING of an in-range integer is admitted: the rule is
    about the value, not about how it was typed. `test_three_rows.py`
    writes its declared boxes this way and `harness.any_array` records
    every accepted bound spelling as a python `float`."""
    lo, hi = ir._LIT_INT_BOUNDS[dtype]
    admitted(float(lo), dtype)
    admitted(0.0, dtype)
    assert ir.literal_inexact(admitted(float(lo), dtype)) is None


def test_the_integer_row_refuses_the_slack_bound_test_three_rows_used_to_write():
    """The repair in the same commit, pinned from the other side.
    `wrapping_int_query` asserted against ``lit(-1e30, a)`` under an
    INTEGER aval as a slack bound so that only the overflow guard could
    decide — a legitimate intent with an impossible spelling, since an
    `int32` value can never be -1e30."""
    msg = refused(-1e30, "int32")
    assert "[-2147483648, 2147483647]" in msg
    # ...and it predicts NO stored value, because -1e30 is a float
    assert "would store as" not in msg, msg
    assert "PREDICTS NO STORED VALUE" in msg, msg


@pytest.mark.parametrize("dtype", INTS)
def test_an_out_of_range_FLOAT_literal_is_refused_WITHOUT_a_prediction(dtype):
    """**THE MESSAGE SPLITS ON THE SOURCE TYPE, AND SPEC-LIT §3 DID NOT.**

    That row prescribes ``((val - lo) mod 2**bits) + lo`` — "what the
    value would store as" — for every out-of-range integer refusal
    without distinguishing an `int` source from a `float` one. The first
    commit implemented it faithfully. It is correct for an `int`, whose
    narrowing is defined bit-for-bit, and FALSE for a `float`, whose
    out-of-range conversion IEEE 754 leaves unspecified. The two
    spellings of the same magnitude therefore get two different messages,
    and this drives both at every integer dtype: the int one predicts,
    the float one does not.

    What settles it is not that the formula picks the wrong answer but
    that there is no right one to pick — see
    :func:`test_the_backends_DISAGREE_so_a_float_refusal_predicts_nothing`."""
    lo, hi = ir._LIT_INT_BOUNDS[dtype]
    over = hi + 1
    int_msg = refused(over, dtype)
    float_msg = refused(float(over), dtype)
    # both name the range, and both name the same one
    for msg in (int_msg, float_msg):
        assert f"[{lo}, {hi}]" in msg, msg
    assert "would store as" in int_msg, int_msg
    assert "would store as" not in float_msg, float_msg
    assert "PREDICTS NO STORED VALUE" in float_msg, float_msg
    assert "unspecified" in float_msg, float_msg


def test_the_int_wrap_the_message_quotes_is_what_the_backends_store():
    """THE INT HALF, VERIFIED AGAINST TWO IMPLEMENTATIONS THAT ARE NOT
    THIS ONE. The message predicts, so the prediction has to be checked
    against something that actually stores the value.

    `ir` may not import numpy or jax, so the arithmetic it quotes is its
    own; here it is compared cell by cell against `numpy`'s narrowing and
    `jax`'s. Neither backend is asked a question it cannot answer: the
    SOURCE has to exist as an array before it can be narrowed, so a value
    outside every 64-bit container (`int64`'s ``2**63`` and `uint64`'s
    ``2**64``) is skipped rather than fudged, and the jax half is
    restricted to the dtypes whose out-of-range neighbours fit an `int32`
    — because widening jax's default container means flipping
    `jax_enable_x64`, which is exactly what `tests/_state_guard.py`
    exists to catch a test doing.

    The two lists are not typed: they are computed from
    `ir._LIT_INT_BOUNDS` and from what each backend actually exposes, and
    the test asserts it checked something rather than trusting that it
    did."""
    np = pytest.importorskip("numpy")
    # `jax`, not `jax.numpy`: `tests/test_skip_inventory.py` declares the
    # optional dependencies this suite may gate on by their TOP-LEVEL
    # name, and an undeclared gate fails that file
    jnp = pytest.importorskip("jax").numpy
    by_numpy, by_jax = set(), set()
    for dtype, (lo, hi) in ir._LIT_INT_BOUNDS.items():
        bits = ir._LIT_INT_BITS[dtype]
        for v in (hi + 1, lo - 1):
            wrap = ((v - lo) % (1 << bits)) + lo
            # ...and the message really does quote that number
            assert f"would store as {wrap}" in refused(v, dtype)
            container = np.uint64 if v >= 0 else np.int64
            fits = (v < 2 ** 64) if v >= 0 else (v >= -(2 ** 63))
            if fits and hasattr(np, dtype):
                stored = np.array(v, dtype=container).astype(
                    getattr(np, dtype)).item()
                assert int(stored) == wrap, (dtype, v, stored, wrap)
                by_numpy.add(dtype)
            if (-(2 ** 31) <= v < 2 ** 31 and bits <= 32
                    and hasattr(jnp, dtype)):
                stored = jnp.array(v, dtype=jnp.int32).astype(
                    getattr(jnp, dtype)).item()
                assert int(stored) == wrap, (dtype, v, stored, wrap)
                by_jax.add(dtype)
    # COVERAGE IS ASSERTED AS A SET, NOT AS A TOTAL, because the two
    # backends cover different rows and a typed sum would be a third
    # number to keep in step: numpy has no `int4`/`uint4` and cannot hold
    # `2**63`/`2**64` as a source, jax has both widths but is held to
    # `int32` sources here. What matters is that no row of the table was
    # checked by NEITHER.
    assert by_numpy and by_jax
    assert by_numpy | by_jax == set(ir._LIT_INT_BOUNDS), (
        sorted(set(ir._LIT_INT_BOUNDS) - (by_numpy | by_jax)))


def test_the_backends_DISAGREE_so_a_float_refusal_predicts_nothing():
    """**THE MEASUREMENT THAT SETTLES FINDING 2, RE-DRIVEN RATHER THAN
    QUOTED.**

    The float half of the integer row predicts nothing, and the reason is
    not that the wrap formula picks the wrong answer — it is that THERE
    IS NO SINGLE ANSWER TO PICK. IEEE 754 leaves a float-to-integer
    conversion whose result is outside the integer format unspecified,
    and the two backends this tree can reach resolve it differently:
    numpy's answer follows the platform's C cast and jax CLAMPS to the
    dtype's endpoints. Every figure below is re-measured by this test
    rather than typed, including the one that shows numpy is not
    uniformly "the wrap" either — at 1e30 it raises *"invalid value
    encountered in cast"* and returns something that is neither the wrap
    nor an endpoint. There is no second formula to swap in.

    ``300.0`` under `int8` is the cleanest witness and the one this test
    pins from both sides: numpy answers the wrap this module would have
    quoted, and jax answers the maximum instead. Neither is wrong; a
    message claiming either as "what it would store as" is.

    **AND THE CELLS WHERE THEY AGREE ARE PINNED HERE TOO, BECAUSE THE
    REFUSAL'S CLAUSE IS ABOUT THE CLASS AND NOT ABOUT THE CONVERSION IN
    FRONT OF THE READER.** It used to end *"and the backends disagree
    about it"*, which is false at every cell where the two happen to
    agree — a set this test requires to be nonempty and
    :func:`test_the_census_behind_that_comment_is_BUILT_AND_COUNTED_HERE`
    sizes, so no fraction is typed in this docstring either: an earlier
    wording said *"about a fifth"* and nothing recomputed it. jax
    clamps, so it always answers an endpoint, and numpy
    coincides with that wherever the platform's cast happens to land on
    the same one — ``-1e30`` under `int64` is such a cell and BOTH store
    INT64_MIN there, which is the cell the previous commit message cites
    precisely BECAUSE they agree. A reader who checked it against the
    old clause was pushed toward "so it should have predicted
    INT64_MIN", the repair the class-level argument rules out. So this
    test counts BOTH outcomes and requires both to be nonempty: if the
    backends ever stopped agreeing anywhere, the "in general" the
    message now says would be understating, and if they stopped
    disagreeing the refusal should be revisited.

    **AND THE AGREEMENT IS NOT A SIGN ASYMMETRY.** The comment at the
    check used to add that the two coincide *"never once on a positive
    overflow"*, which is true of the sweep it was measured on and NOT of
    the mechanism: while the platform's cast still wraps, numpy answers
    the low ``bits`` of the truncated value, so it lands on ``hi`` — where
    jax's clamp already is — for every ``v`` with
    ``int(v) ≡ hi (mod 2**bits)``. That sweep held no such value for a
    reason that is arithmetic and not judgement, and the reason is
    asserted below: every ``hi`` is ``2**k - 1`` and so ODD, every ``lo``
    is even, and the sweep's positive values were all even — so its
    positive cells could not land on ``hi`` and its negative ones could
    land on ``lo``. ``383.0`` under `int8` is one step past the sweep's
    own ``300.0``, is odd, and both backends answer 127. It is driven
    below alongside the negative agreements: a POSITIVE agreeing cell,
    which the old sentence said could not exist.

    **AND PARITY IS ONE OF TWO MECHANISMS, NOT A FRACTION OF THAT
    ASYMMETRY** — an earlier wording of the block below, and of the
    comment at the check, first said it was the whole and then said it
    was *"half"*, a fraction of a denominator neither of them named.
    Parity is an argument about a WRAP, and the platform's cast does not
    always wrap: where it stops it returns a constant that depends on the
    dtype and the SIGN but not on the value, and that constant is never
    ``hi``. THE SECOND HALF OF THAT SENTENCE IS THE HALF THAT SURVIVES,
    and it is derivable rather than enumerated — every ``hi`` is
    ``2**k - 1`` and so ODD and every such constant is EVEN — where the
    earlier *"``lo`` or ``0``"* was an enumeration of one sample, and
    `uint64` answers neither of those at cells of that very sample. How
    many is asserted over there rather than typed here; an earlier
    wording of this paragraph typed it. So the saturating answer can
    coincide with jax's clamp BELOW the range and never above it — a
    second, separate reason for the same one-sided agreement, with its
    own witness driven below and the whole split counted in
    :func:`test_the_census_behind_that_comment_is_BUILT_AND_COUNTED_HERE`.

    Nothing here is typed. The disagreement is COUNTED over the cells the
    default jax configuration can reach, and the test asserts the count
    is nonzero — so if the backends ever converge, this fails and the
    refusal's wording should be revisited rather than silently left
    stale."""
    np = pytest.importorskip("numpy")
    # `jax`, not `jax.numpy`: `tests/test_skip_inventory.py` declares the
    # optional dependencies this suite may gate on by their TOP-LEVEL
    # name, and an undeclared gate fails that file
    jnp = pytest.importorskip("jax").numpy
    disagreed = agreed = cells = 0
    # -65536.0 is in the list so the AGREEING outcome is reachable
    # without `jax_enable_x64` — it is out of range for all four dtypes,
    # and under the two unsigned ones both backends answer the lower
    # endpoint. `tests/_state_guard.py` exists to catch a test flipping
    # x64, so the agreeing witness has to be one a 32-bit container can
    # be asked about.
    for dtype in ("int8", "int16", "uint8", "uint16"):
        lo, hi = ir._LIT_INT_BOUNDS[dtype]
        for v in (float(hi + 1), float(lo - 1), 300.0, -300.0, -65536.0):
            if lo <= v <= hi:
                # in range, so nothing is refused and there is no message
                # whose clause could be right or wrong about it — the
                # counts below are about the CELLS THAT EMIT IT, which is
                # what the clause is a claim over
                continue
            assert "PREDICTS NO STORED VALUE" in refused(v, dtype)
            cells += 1
            npv = np.array(v).astype(getattr(np, dtype)).item()
            jv = jnp.array(v).astype(getattr(jnp, dtype)).item()
            disagreed += int(npv) != int(jv)
            agreed += int(npv) == int(jv)
    assert disagreed, f"the two backends agreed on all {cells} cells"
    # ...and the other half of the same fact: the clause the message
    # carries has to survive the cells where they AGREE, which is why it
    # says "in general" rather than claiming this conversion
    assert agreed, f"the two backends disagreed on all {cells} cells"

    lo, _ = ir._LIT_INT_BOUNDS["uint16"]
    assert int(np.array(-65536.0).astype(np.uint16).item()) == lo
    assert int(jnp.array(-65536.0).astype(jnp.uint16).item()) == lo
    agree_msg = refused(-65536.0, "uint16")
    assert "PREDICTS NO STORED VALUE" in agree_msg, agree_msg
    assert "do not agree on the answer in general" in agree_msg, agree_msg
    assert "the backends disagree about it" not in agree_msg, agree_msg
    # the witness, both halves of it derived rather than typed
    lo, hi = ir._LIT_INT_BOUNDS["int8"]
    wrap = ((300 - lo) % (1 << ir._LIT_INT_BITS["int8"])) + lo
    assert int(np.array(300.0).astype(np.int8).item()) == wrap
    assert int(jnp.array(300.0).astype(jnp.int8).item()) == hi != wrap
    # ...and numpy is not even uniformly "the wrap", which is what rules
    # out simply quoting ITS answer instead. At 1e30 it says so itself:
    # the cast RAISES A WARNING NAMING THE RESULT INVALID, and returns a
    # number that is neither the wrap nor an endpoint, while jax still
    # clamps. numpy's own diagnostic is the strongest available evidence
    # that there is nothing here for a message to predict.
    with pytest.warns(RuntimeWarning, match="invalid value encountered"):
        huge_np = int(np.array(1e30).astype(np.int8).item())
    assert huge_np != wrap and huge_np != hi and huge_np != lo
    assert int(jnp.array(1e30).astype(jnp.int8).item()) == hi
    # ...and this is what the refusal says instead of choosing one
    msg = refused(300.0, "int8")
    assert "PREDICTS NO STORED VALUE" in msg, msg
    assert "would store as" not in msg, msg
    # while the INT spelling of the same magnitude still predicts, and
    # numpy still agrees with it
    assert f"would store as {wrap}" in refused(300, "int8")

    # THE POSITIVE AGREEING CELLS, which the comment at the check used to
    # say did not exist. The witnesses are DERIVED, not typed: the first
    # positive value past `hi` whose truncation is congruent to `hi` is
    # `hi + 2**bits`, and that is where numpy's wrap lands back on the
    # endpoint jax clamps to. All three fit a float32 exactly, so no
    # `jax_enable_x64` is needed and `tests/_state_guard.py` stays happy.
    positives = 0
    for dtype in ("int8", "uint8", "int16"):
        lo, hi = ir._LIT_INT_BOUNDS[dtype]
        v = float(hi + (1 << ir._LIT_INT_BITS[dtype]))
        assert v > 0 and float(v).is_integer() and v == int(v)
        assert "PREDICTS NO STORED VALUE" in refused(v, dtype)
        npv = int(np.array(v).astype(getattr(np, dtype)).item())
        jv = int(jnp.array(v).astype(getattr(jnp, dtype)).item())
        assert npv == jv == hi, (dtype, v, npv, jv, hi)
        positives += 1
    assert positives == 3
    # and the one the comment names by value, so a reader who pastes it
    # into an interpreter meets the same number
    assert float(ir._LIT_INT_BOUNDS["int8"][1] + 256) == 383.0

    # THE PARITY THAT MADE THE OLD SWEEP LOOK ASYMMETRIC, asserted off
    # the table rather than typed: every `hi` is ``2**k - 1`` and so ODD,
    # every `lo` is ``-2**(bits-1)`` or ``0`` and so EVEN. A sweep whose
    # positive values are all even therefore cannot produce a positive
    # agreement BY WRAPPING — numpy's wrap has the wrong parity to land
    # on `hi` — while its negative values can land on `lo`.
    for dtype, (lo, hi) in ir._LIT_INT_BOUNDS.items():
        assert hi % 2 == 1, (dtype, hi)
        assert lo % 2 == 0, (dtype, lo)

    # ...AND PARITY IS ONE OF TWO MECHANISMS AND NOT A FRACTION OF THE
    # ASYMMETRY — an earlier wording of this block typed the fraction,
    # and typed it wrong twice running. It is an argument about a WRAP,
    # so it reaches the wrapping cells and no others; the split by
    # mechanism is COUNTED, over the census the comment at the check
    # argues from, in
    # `test_the_census_behind_that_comment_is_BUILT_AND_COUNTED_HERE`,
    # and no number for it is typed here. THE
    # SECOND MECHANISM IS SATURATION, and it has a witness a 32-bit
    # container can be asked about, so it is pinned here rather than
    # left in prose: at `-1e10` under `uint16` numpy does NOT wrap — the
    # wrap is derived below and is not what it answers — it returns the
    # value-independent 0 it returns for any negative too far out to
    # cast, which is exactly where jax clamps. The two AGREE for a
    # reason parity says nothing about. jax sees the float32 image of
    # -1e10 rather than -1e10 itself; both are far below the range, so
    # the clamp is the same either way and no `jax_enable_x64` is
    # needed. numpy names its own answer invalid while returning it,
    # which is why the cast is bracketed.
    lo, _ = ir._LIT_INT_BOUNDS["uint16"]
    saturating = -1e10
    assert "PREDICTS NO STORED VALUE" in refused(saturating, "uint16")
    wrap = ((int(saturating) - lo) % (1 << ir._LIT_INT_BITS["uint16"])) + lo
    with pytest.warns(RuntimeWarning, match="invalid value encountered"):
        npv = int(np.array(saturating).astype(np.uint16).item())
    jv = int(jnp.array(saturating).astype(jnp.uint16).item())
    assert npv == jv == lo, (npv, jv, lo)
    assert npv != wrap, (npv, wrap)


def test_the_census_behind_that_comment_is_BUILT_AND_COUNTED_HERE():
    """**THE CENSUS THE COMMENT AT THE CHECK RESTS ON, MOVED OUT OF ITS
    PROSE AND INTO CODE. THE COMMENT NOW TYPES NO FIGURE AT ALL.**

    Three consecutive audits of that block found `ir.py`'s executable
    half untouched and something wrong with the same paragraph of prose,
    and every time it was a hand-maintained count or proportion: *"false
    at about a fifth of the cells"*, *"parity is about half of the
    asymmetry"*, *"it is ``hi`` at NONE of the 155"*, *"a constant that
    is ``lo`` or ``0``"*. Two of those were repairs FOR an earlier
    miscount. A number typed beside the thing it counts cannot redden
    when the thing moves, and these count the answers of TWO FOREIGN
    LIBRARIES pinned at one version each — so the failure mode is not
    carelessness, it is that nothing was ever going to look again.

    So the census lives here. This test BUILDS the value set from
    :data:`ir._LIT_INT_BOUNDS` rather than listing it — a described set
    was twice rebuilt wrong from the description — crosses it with the
    dtype table, asks both backends, and asserts every figure the
    comment used to carry. If numpy or jax moves an answer, THIS fails
    and names the cell.

    **THIS DOCSTRING TYPES NO FIGURE EITHER.** Every count lives in an
    ``assert`` below, where a wrong one is a red test rather than a
    sentence nobody re-reads. What it pins, in the order the comment
    argues it:

    1. **The set and the cells.** The integral values, crossed with the
       whole dtype table; how many cells emit this refusal, and how many
       of those can be asked — the remainder is every emitting
       ``int4``/``uint4`` cell, which neither backend spells.
    2. **jax CLAMPS.** Its answer is an endpoint at every cell, and it
       is ``hi`` above the range and ``lo`` below it. numpy's is not,
       which is the whole disagreement.
    3. **The two mechanisms.** numpy's cast WRAPS at most of the cells
       and returns a constant per dtype and SIGN at the rest — `uint64`
       returns a different one on each side, which is why it is not a
       constant per dtype and not one independent of the value. Parity is
       an argument about the first only, which is why it is ONE OF TWO
       mechanisms and not a fraction of the asymmetry — the earlier
       *"about half"* was the fraction of a denominator nothing named.
       The ``(dtype, sign)`` groups this sample leaves EMPTY are named,
       so the one-constant-per-group check is a control that can fire
       rather than a row of vacuous truths.
    4. **The constant is never ``hi``, and the reason is arithmetic.**
       Every ``hi`` in the table is ``2**k - 1`` and so ODD; every constant
       this box returns is EVEN. The old sentence enumerated the
       constants instead — *"``lo`` or ``0``"* — and ``uint64`` answers
       neither of those at cells of this very census.
    5. **``hi`` at none of the SATURATING cells is not ``hi`` at none of
       the census.** numpy's answer IS ``hi`` at cells of it, all of
       them negative, all reached by WRAPPING, and every one a
       DISAGREEMENT because jax clamps a negative overflow to ``lo``.
       Fusing the two denominators is what made the old sentence false.
    6. **The indefinite is ``lo`` only where the dtype is as wide as the
       temporary.** ``int32`` and ``int64`` answer their own ``lo``;
       ``int8`` and ``int16`` answer ``0``, which is that same constant
       NARROWED to their width by the same truncation the wrap uses, and
       ``uint64`` answers it REINTERPRETED unsigned below the range while
       answering ``0`` above it — the sign in item 3.
    7. **The ladder, and that it is not a half-line.** ``hi + 2**bits``
       agrees for the dtypes whose wrap regime reaches it, not for
       ``int32`` — whose regime is over below its own first rung — and
       not for ``int64``/``uint64``, which have no exact float64 for it
       to ask. It is NOT the narrow dtypes that agree, which is what one
       round's wording of the comment said: ``uint32`` is ``int32``'s
       width and is one of them, and the width comes off the table here.
       The two ``int8`` values BETWEEN the first and second
       rungs disagree while the second rung agrees again; both rungs are
       computed from the table here rather than copied from the comment.

    **EVERY NUMPY FIGURE BELOW IS PLATFORM-SCOPED, AND THAT IS AN OPEN
    QUESTION RATHER THAN A MEASUREMENT** — the same one the comment at
    the check discloses, in the same terms, and it belongs here too now
    that the figures do. They are what THIS x86-64 box's ``cvttsd2si``
    returns when the result does not fit; a SATURATING target such as
    aarch64's ``fcvtzs`` answers ``hi`` for a positive overflow instead,
    which would put a positive agreement at an even value and redden the
    positive counts and the saturating-constant asserts below. That is
    the tree PREDICTING a red run on ARM, not disclaiming one: nothing
    here has been run on such a box, because there is no ARM box here.

    x64 is needed and is set and PUT BACK in a ``finally``: without it
    `jnp.asarray(v, dtype=jnp.float64)` hands back an f32, ``1e300``
    becomes ``inf``, and the census measures a different question. That
    is the same set-and-restore
    :func:`test_the_subnormal_band_is_admitted_and_the_note_is_the_IEEE_route`
    does further down this file, for the same reason:
    `tests/_state_guard.py` brackets each test, so a restored set is
    silent to it and a leak is named.
    """
    np = pytest.importorskip("numpy")
    # `jax`, not `jax.numpy`: `tests/test_skip_inventory.py` declares the
    # optional dependencies this suite may gate on by their TOP-LEVEL
    # name, and an undeclared gate fails that file
    jax = pytest.importorskip("jax")
    jnp = jax.numpy

    def emits(v, dtype):
        """Does this cell emit the no-prediction refusal? Never raises."""
        try:
            ir.Literal(val=v, aval=aval(dtype))
        except ir.TranscriptionError as exc:
            return "PREDICTS NO STORED VALUE" in str(exc)
        return False

    # (1) THE VALUE SET, BUILT AND NOT LISTED. `float(hi) + 1` and
    # `float(lo) - 1` per dtype, plus a fixed spread of larger
    # magnitudes, minus the non-integral and the duplicates.
    values = set()
    for lo, hi in ir._LIT_INT_BOUNDS.values():
        values.add(float(hi) + 1.0)
        values.add(float(lo) - 1.0)
    for magnitude in (16.0, 300.0, 65536.0, 1e10, 1e30, 3e38, 1e300):
        values.add(magnitude)
        values.add(-magnitude)
    values.update(
        {2.0 ** 31, -(2.0 ** 31) - 1.0, 2.0 ** 63, -(2.0 ** 63) * 2.0, 2.0 ** 64}
    )
    values = sorted(v for v in values if float(v).is_integer())
    assert len(values) == 29, len(values)
    # every positive one is EVEN, which is the parity bar itself and not
    # an observation about it — see (4) below
    assert [v for v in values if v > 0 and v % 2] == []

    emitting = [
        (d, v) for d in ir._LIT_INT_BOUNDS for v in values if emits(v, d)
    ]
    assert len(ir._LIT_INT_BOUNDS) * len(values) == 290
    assert len(emitting) == 211, len(emitting)

    askable = [
        d
        for d in ir._LIT_INT_BOUNDS
        if getattr(np, d, None) is not None and getattr(jnp, d, None) is not None
    ]
    assert set(ir._LIT_INT_BOUNDS) - set(askable) == {"int4", "uint4"}
    asked = [(d, v) for d, v in emitting if d in askable]
    assert len(askable) == 8 and len(asked) == 155, (len(askable), len(asked))
    assert len(emitting) - len(asked) == 56

    # (2) ASK BOTH BACKENDS. x64 is not optional: `1e300` and `2**64`
    # have no float32, so without it the wide array is an f32 `inf` and
    # every cell of this census answers a different question.
    old_x64 = jax.config.jax_enable_x64
    rows = []
    try:
        jax.config.update("jax_enable_x64", True)
        for d, v in asked:
            lo, hi = ir._LIT_INT_BOUNDS[d]
            wide = jnp.asarray(v, dtype=jnp.float64)
            assert wide.dtype == jnp.float64  # the flag really took
            with warnings.catch_warnings():
                # numpy names its own answer invalid while returning it;
                # `test_the_backends_DISAGREE_...` above brackets one
                # such cast with `pytest.warns` to pin that it does
                warnings.simplefilter("ignore", RuntimeWarning)
                npv = int(np.asarray(v).astype(getattr(np, d)).item())
            rows.append(
                dict(
                    d=d,
                    v=v,
                    np=npv,
                    jax=int(wide.astype(getattr(jnp, d)).item()),
                    lo=lo,
                    hi=hi,
                    wrap=((int(v) - lo) % (1 << ir._LIT_INT_BITS[d])) + lo,
                )
            )
        # the ladder's first rung, per dtype, asked in the same window:
        # `hi + 2**bits` is where numpy's wrap lands back on `hi`
        rung = {}
        for d in askable:
            hi = ir._LIT_INT_BOUNDS[d][1]
            want = hi + (1 << ir._LIT_INT_BITS[d])
            if int(float(want)) != want:
                # no float64 spells it exactly, so it cannot be asked
                rung[d] = None
                continue
            wide = jnp.asarray(float(want), dtype=jnp.float64)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                rung[d] = int(
                    np.asarray(float(want)).astype(getattr(np, d)).item()
                ) == int(wide.astype(getattr(jnp, d)).item())
    finally:
        jax.config.update("jax_enable_x64", old_x64)

    pos = [r for r in rows if r["v"] > 0]
    neg = [r for r in rows if r["v"] < 0]
    flat = [r for r in rows if r["np"] != r["wrap"]]
    assert len(pos) == 68 and len(neg) == 87

    # (3) jax CLAMPS — an endpoint at every cell — and numpy does not
    assert all(r["jax"] in (r["lo"], r["hi"]) for r in rows)
    assert not all(r["np"] in (r["lo"], r["hi"]) for r in rows)
    assert all(r["jax"] == (r["hi"] if r["v"] > 0 else r["lo"]) for r in rows)

    agreed = sum(r["np"] == r["jax"] for r in rows)
    assert agreed == 32 and len(rows) - agreed == 123
    assert sum(r["np"] == r["jax"] for r in pos) == 0
    assert sum(r["np"] == r["jax"] for r in neg) == 32

    # (4) THE TWO MECHANISMS, and parity reaches only the first
    assert len(rows) - len(flat) == 120 and len(flat) == 35
    assert sum(r["np"] == r["wrap"] for r in pos) == 54
    assert sum(r["np"] != r["wrap"] for r in pos) == 14
    neg_agree = [r for r in neg if r["np"] == r["jax"]]
    assert sum(r["np"] == r["wrap"] for r in neg_agree) == 18
    assert sum(r["np"] != r["wrap"] for r in neg_agree) == 14
    # the parity bar, off the table rather than off the sample: numpy's
    # wrap lands on jax's clamp only at `int(v) ≡ hi (mod 2**bits)`, and
    # every `hi` is `2**k - 1` and so ODD while every value above is EVEN
    assert all(hi % 2 == 1 for _, hi in ir._LIT_INT_BOUNDS.values())
    assert all(lo % 2 == 0 for lo, _ in ir._LIT_INT_BOUNDS.values())
    # ...and the SECOND mechanism is why that bar is not the whole story:
    # where the cast does not wrap, numpy returns a constant, and the
    # constant is EVEN too, so it cannot be `hi` either. The constant is
    # per dtype AND SIGN — `uint64` is why the sign is in there, see (6).
    #
    # THE (dtype, sign) GROUPS THAT ARE EMPTY IN THIS SAMPLE ARE NAMED,
    # so the per-group check below asks for EXACTLY ONE constant at every
    # group this sample reaches, rather than the `<= 1` an empty group
    # satisfies without being looked at. A control that cannot fire is
    # the defect this census exists to replace, and that applies to the
    # census itself: `uint32` reaches NEITHER side of this sample's
    # non-wrapping regime, and `int8`/`uint8` reach only the negative.
    empty = {
        (d, positive)
        for d in askable
        for positive in (True, False)
        if not [r for r in flat if r["d"] == d and (r["v"] > 0) is positive]
    }
    assert empty == {
        ("int8", True),
        ("uint8", True),
        ("uint32", True),
        ("uint32", False),
    }, sorted(empty)
    for d in askable:
        for positive in (True, False):
            group = {
                r["np"] for r in flat if r["d"] == d and (r["v"] > 0) is positive
            }
            assert len(group) == (0 if (d, positive) in empty else 1), (
                d,
                positive,
                group,
            )
    assert sum(r["np"] == r["lo"] for r in flat) == 27
    assert sum(r["np"] == r["hi"] for r in flat) == 0
    assert all(r["np"] % 2 == 0 for r in flat), sorted({r["np"] for r in flat})

    # (5) AND `hi` AT NONE OF THE SATURATING CELLS IS NOT `hi` AT NONE
    # OF THE CENSUS. A wrap FROM BELOW can land on `hi`; the saturating
    # constant cannot, and fusing the two denominators is the whole of
    # what made the comment's old sentence false.
    at_hi = [r for r in rows if r["np"] == r["hi"]]
    assert len(at_hi) == 7
    assert all(r["v"] < 0 for r in at_hi)
    assert all(r["np"] == r["wrap"] for r in at_hi)
    assert all(r["jax"] == r["lo"] != r["np"] for r in at_hi)

    # (6) THE INDEFINITE IS `lo` ONLY WHERE THE DTYPE IS AS WIDE AS THE
    # TEMPORARY. `int8` and `int16` answer `0`, which is `int32`'s own
    # constant narrowed by the same truncation the wrap uses, and those
    # are cells the comment used to lump under "something else".
    def constant(d):
        return sorted({r["np"] for r in flat if r["d"] == d})

    assert constant("int32") == [ir._LIT_INT_BOUNDS["int32"][0]]
    assert constant("int64") == [ir._LIT_INT_BOUNDS["int64"][0]]
    for d in ("int8", "int16"):
        lo = ir._LIT_INT_BOUNDS[d][0]
        narrowed = ((constant("int32")[0] - lo) % (1 << ir._LIT_INT_BITS[d])) + lo
        assert constant(d) == [narrowed] == [0], (d, constant(d), narrowed)
    # and `uint64` is why the constant is per-SIGN and not per-dtype, and
    # why the old *"``lo`` or ``0``"* enumeration could not name it: its
    # two constants are ``0`` above the range and, below it, `int64`'s
    # own indefinite REINTERPRETED UNSIGNED — the same temporary as in
    # the narrowing above, read through a different dtype instead of
    # truncated to a shorter one. For `uint64` ``lo`` IS ``0``, so a
    # constant that is neither is the whole of what the enumeration
    # missed. The assert this replaces asked whether ``lo`` was outside
    # ``constant("uint64")[1:]``: the answers are unsigned and ``lo`` is
    # ``0``, so it is element 0 of that sorted set whenever it is present
    # at all and the check could not fail for any data.
    u_lo = ir._LIT_INT_BOUNDS["uint64"][0]
    u_pos = [r["np"] for r in flat if r["d"] == "uint64" and r["v"] > 0]
    u_neg = [r["np"] for r in flat if r["d"] == "uint64" and r["v"] < 0]
    indefinite = ir._LIT_INT_BOUNDS["int64"][0] % (1 << ir._LIT_INT_BITS["uint64"])
    assert len(constant("uint64")) == 2, constant("uint64")
    assert u_lo == 0 and set(u_pos) == {u_lo} and len(u_pos) == 1
    assert set(u_neg) == {indefinite} and len(u_neg) == 4, u_neg
    assert u_lo not in u_neg and 0 not in u_neg

    # (7) THE LADDER'S FIRST RUNG. `int32`'s wrap regime is over below
    # it, so it disagrees there; `int64` and `uint64` cannot spell
    # `hi + 2**bits` as an exact float64 to ask at all.
    assert sum(v is True for v in rung.values()) == 5, rung
    assert rung["int32"] is False, rung
    assert rung["int64"] is None and rung["uint64"] is None, rung
    # ...AND IT IS NOT THE NARROW DTYPES THAT AGREE. One round's wording
    # of the comment at the check said "`int32` disagrees there while the
    # NARROWER dtypes agree", which sorts eight dtypes into three buckets
    # and leaves `uint32` in none of them: it is `int32`'s WIDTH and it
    # is one of the five. Width is read off :data:`ir._LIT_INT_BITS`, so
    # this CONTRADICTS that sentence rather than restating it, and it is
    # the assert the comment's "which dtypes those are" now points at.
    assert ir._LIT_INT_BITS["uint32"] == ir._LIT_INT_BITS["int32"] == 32
    assert rung["uint32"] is True, rung
    assert {d for d, v in rung.items() if v is True} == {
        "int8",
        "int16",
        "uint8",
        "uint16",
        "uint32",
    }, rung

    # ...AND IT IS A LADDER AND NOT A HALF-LINE. The comment names four
    # `int8` values for that; all four are exact float32s, so this needs
    # no x64, and the two rungs are DERIVED off the table rather than
    # read off the comment.
    lo8, hi8 = ir._LIT_INT_BOUNDS["int8"]
    step = 1 << ir._LIT_INT_BITS["int8"]
    assert (float(hi8 + step), float(hi8 + 2 * step)) == (383.0, 639.0)
    rungs = ((383.0, True), (384.0, False), (385.0, False), (639.0, True))
    for v, agrees in rungs:
        assert "PREDICTS NO STORED VALUE" in refused(v, "int8")
        npv = int(np.asarray(v).astype(np.int8).item())
        assert npv == ((int(v) - lo8) % step) + lo8  # still inside the wrap regime
        assert (npv == int(jnp.asarray(v).astype(jnp.int8).item())) is agrees, v

    # (8) THE CELL THE COMMENT NAMES BY VALUE: `-1e30` under `int64`
    # AGREES, and by the saturating mechanism rather than by a wrap that
    # landed — which is the half `e66fef3`'s reader got backwards.
    (cell,) = [r for r in rows if r["d"] == "int64" and r["v"] == -1e30]
    assert cell["np"] == cell["jax"] == cell["lo"]
    assert cell["np"] != cell["wrap"]


# -- row: the float dtypes ----------------------------------------------------


@pytest.mark.parametrize("dtype", FLOATS)
def test_a_float_dtype_refuses_an_int_past_its_range_without_calling_float(dtype):
    """THE VALUE-FIRST RULE, and it is not defensive styling: ``float(10
    ** 400)`` raises ``OverflowError: int too large to convert to
    float``, so a check that converts before it compares crashes on the
    value it exists to refuse. The plan's prototype did exactly that, and
    `tests/test_exact_recording.py` documents the same escape one layer
    up — "the phase1 escape, closed value-first". This asserts the door's
    refusal AND that the failure it replaces is real."""
    with pytest.raises(OverflowError):
        float(HUGE)
    for val in (HUGE, -HUGE):
        assert "overflows" in refused(val, dtype)


@pytest.mark.parametrize("dtype", NARROW)
def test_a_float_dtype_refuses_a_float_past_its_range(dtype):
    """The float spelling of the same class, which only the NARROW
    formats have — and float64's absence from this row is a fact about
    binary64 rather than a gap, asserted below rather than skipped."""
    tie = ir._LIT_FLOAT_OVERFLOW_AT[dtype]
    assert "overflows" in refused(float(tie), dtype)
    assert "overflows" in refused(-float(tie), dtype)


def test_no_python_float_lies_past_float64_so_that_row_has_no_float_spelling():
    """Why the row above is parametrized over three formats and not four.
    A python `float` IS a binary64, so the only spelling that can be past
    float64's tie point is an `int` — which is the value-first row, and
    which `float()` refuses to convert at exactly the same threshold."""
    tie = ir._LIT_FLOAT_OVERFLOW_AT["float64"]
    assert float(tie - 1) == ir._LIT_FLOAT_MAX["float64"]
    with pytest.raises(OverflowError):
        float(tie)
    assert "overflows" in refused(tie, "float64")


@pytest.mark.parametrize("dtype", NARROW)
def test_a_float_dtype_refuses_a_value_that_underflows_to_zero(dtype):
    """A nonzero value stored as 0.0 is DESTROYED, not rounded: every
    fact about its sign and its nonzero-ness is gone, and
    `propagate._literal_strict_sign` is a live reader of exactly those.

    float64 is absent for the same reason as the row above — the image is
    the identity, so nothing can underflow it — and that is asserted in
    `test_nothing_underflows_float64_because_its_image_is_the_identity`
    rather than skipped."""
    tiny = ir._LIT_FLOAT_MIN_SUBNORMAL[dtype] / 4
    assert tiny != 0.0
    msg = refused(tiny, dtype)
    assert "underflows" in msg and "0.0" in msg
    assert "underflows" in refused(-tiny, dtype)
    # ...and the smallest thing the format DOES hold is admitted exactly
    assert ir.literal_inexact(
        admitted(ir._LIT_FLOAT_MIN_SUBNORMAL[dtype], dtype)
    ) is None


def test_nothing_underflows_float64_because_its_image_is_the_identity():
    """binary64's smallest subnormal has no representable quarter — it
    rounds to 0.0 as a python float before any check sees it — so the
    underflow row has no float64 member to drive. The int side has none
    either: an integer that is not zero is at least 1."""
    assert ir._LIT_FLOAT_MIN_SUBNORMAL["float64"] / 4 == 0.0
    assert admitted(5e-324, "float64").val == 5e-324
    assert ir._literal_range_problem(5e-324, aval("float64")) == (None, None)


@pytest.mark.parametrize("dtype", FLOATS)
def test_nan_and_both_infinities_are_values_of_every_float_dtype(dtype):
    """The rows that must NOT fire. NaN and +-inf are values of every
    binary float format; a literal carrying one is not lying about its
    dtype, and `test_literal_strict_sign_drops_zero_and_nonfinite` builds
    them deliberately."""
    for val in (math.nan, math.inf, -math.inf):
        lit = admitted(val, dtype)
        assert ir.literal_inexact(lit) is None


@pytest.mark.parametrize("dtype", FLOATS)
def test_a_float_dtypes_own_extremes_are_admitted_exactly(dtype):
    for val in (0.0, -0.0, ir._LIT_FLOAT_MAX[dtype], -ir._LIT_FLOAT_MAX[dtype],
                ir._LIT_FLOAT_MIN_SUBNORMAL[dtype]):
        assert ir.literal_inexact(admitted(val, dtype)) is None, (dtype, val)


# -- row: ordinary rounding is INEXACT and never a refusal --------------------


def test_ordinary_rounding_is_recorded_and_never_refused():
    """0.1 has no float32. 0.10000000149011612 is what a float32 holds.
    That is what every float32 program does, and refusing it would have
    failed correct tests: the recorder census that preceded this patch
    found every INEXACT in the suite to be exactly this."""
    lit = admitted(0.1, "float32")
    note = ir.literal_inexact(lit)
    assert note is not None and "0.10000000149011612" in note, note
    assert ir.literal_inexact(admitted(0.1, "float64")) is None


def test_an_int_too_wide_for_the_mantissa_is_INEXACT_not_refused():
    """2**53+1 has no float64 — it stores as 2**53. The value is not
    DESTROYED (it is finite, nonzero and signed), so it is recorded, not
    refused. This is the same value `test_exact_recording.py` built the
    longdouble false-VERIFIED reproducer out of, one layer down."""
    lit = admitted(2 ** 53 + 1, "float64")
    note = ir.literal_inexact(lit)
    assert note is not None and repr(float(2 ** 53)) in note, note


def test_a_value_between_the_largest_finite_and_the_tie_is_INEXACT_not_overflow():
    """SPEC-LIT's float row says to compare an int value against the
    largest FINITE; its next row says a value that round-trips to a
    different finite number is ordinary rounding and must NEVER be
    refused. On the band between the largest finite and the
    round-to-nearest TIE point those two rows contradict each other:
    every value in that band round-trips to the largest finite, which is
    a different finite number.

    The tie is the reading that satisfies both, and here is the witness
    that distinguishes them — measured against `struct`, which is an
    independent implementation of the same rounding."""
    maxf = ir._LIT_FLOAT_MAX["float32"]
    tie = ir._LIT_FLOAT_OVERFLOW_AT["float32"]
    inside = math.nextafter(float(tie), 0.0)
    assert maxf < inside < tie
    # struct agrees that this value has a finite float32 image...
    assert struct.unpack("<f", struct.pack("<f", inside))[0] == maxf
    # ...and refuses the tie itself, which is where infinity begins
    with pytest.raises(OverflowError):
        struct.pack("<f", float(tie))
    # so the band is INEXACT and the tie is DESTROYED
    assert ir.literal_inexact(admitted(inside, "float32")) is not None
    assert "overflows" in refused(float(tie), "float32")
    # ...and the same on the int side, which is the side the spec's rule
    # is worded for
    assert ir.literal_inexact(admitted(int(inside), "float32")) is not None
    assert "overflows" in refused(tie, "float32")


# -- row: the complex dtypes --------------------------------------------------


def test_a_complex_dtype_judges_each_part_and_names_THE_FAILING_ONE():
    """The message names WHICH half failed: "out of range" about a pair
    tells a frontend author nothing about which of the two to fix.

    That the part dtype is READ and not guessed is driven from both
    sides — the same complex value that a `complex64` cannot hold is
    admitted under `complex128`, whose parts are float64."""
    over32 = float(ir._LIT_FLOAT_OVERFLOW_AT["float32"])
    assert "the real part" in refused(complex(over32, 0.0), "complex64")
    assert "the imag part" in refused(complex(0.0, over32), "complex64")
    assert "the imag part" in refused(complex(1.0, 1e-50), "complex64")
    admitted(complex(over32, over32), "complex128")
    admitted(complex(1.0, -2.0), "complex64")
    # ...and an INEXACT part is recorded, naming the part, never refused
    note = ir.literal_inexact(admitted(complex(0.1, 0.0), "complex64"))
    assert note is not None and "the real part" in note, note


def test_a_complex_dtype_records_BOTH_parts_when_BOTH_of_them_round():
    """**THE CONTROL THE ONE-PART CASE DID NOT HAVE, and its absence is
    why half of a two-half record was being dropped.**

    The loop over the parts ASSIGNED the note instead of accumulating it,
    so the last part to round was the only one recorded and
    ``complex(0.1, 0.2)`` under `complex64` reported its imag half alone.
    The test above drives one rounding part at a time and passes either
    way; only a value whose BOTH parts round can tell the two
    implementations apart, which is what this is.

    SPEC-LIT §5's reason for recording rather than warning is that a
    verdict has to be able to QUOTE the record. Half a record is the
    failure that channel was chosen to avoid."""
    note = ir.literal_inexact(admitted(complex(0.1, 0.2), "complex64"))
    assert note is not None
    assert "the real part" in note and "the imag part" in note, note
    # each half names its own stored value, and they are different
    # numbers — so this is two facts joined, not one fact repeated
    assert repr(struct.unpack("<f", struct.pack("<f", 0.1))[0]) in note
    assert repr(struct.unpack("<f", struct.pack("<f", 0.2))[0]) in note
    # one-part values still record exactly one clause, so the join did
    # not turn every note into a pair
    assert " — and " not in ir.literal_inexact(
        admitted(complex(0.1, 0.0), "complex64"))
    assert " — and " in note


def test_a_real_value_under_a_complex_dtype_is_its_own_real_part():
    """`complex(10**400)` raises the same OverflowError `float()` does, so
    the parts of a REAL value are taken WITHOUT constructing a complex at
    all: the value IS the real part and the imaginary part is exactly
    zero. This is also the only way to overflow a `complex128`, whose
    parts no python `complex` can exceed."""
    with pytest.raises(OverflowError):
        complex(HUGE)
    assert "the real part" in refused(HUGE, "complex128")
    assert "the real part" in refused(HUGE, "complex64")
    admitted(3, "complex128")
    admitted(True, "complex64")


def test_a_complex_value_under_a_non_complex_dtype_is_a_category_error():
    """Named as a category error rather than as an out-of-range endpoint,
    because 1+2j is not near float64's range — it is not on its line."""
    msg = refused(complex(1.0, 2.0), "float64")
    assert "non-complex dtype" in msg and "float64" in msg
    assert "non-complex dtype" in refused(complex(1.0, 2.0), "int32")


# -- row: an unrecognised dtype gets no claim ---------------------------------


# every one of these is real, and none is invented. `key<fry>` is a jax
# extended dtype (`stelling._tripwire.prop_guard` names it); `float0` and
# `float8_e4m3fn` are jax dtypes with no row here; `void16`, `xx` and
# `int128` are strings the suite itself builds literals under, measured
# in the recorder census; `complex256` and `float128` are
# ``str(np.dtype(np.clongdouble))`` and ``str(np.dtype(np.longdouble))``
# on this box; and `complex32` is how torch's half-precision complex
# spells itself, which is the frontend this whole door exists for.
UNRECOGNISED = ["complex32", "complex256", "float128", "key<fry>", "float0",
                "float8_e4m3fn", "void16", "xx", "int128", "not a dtype"]
# ONE VALUE OF EVERY PYTHON TYPE THE RULE TABLE NAMES. The rule table's
# other rows dispatch on the VALUE's type as hard as on the dtype's name,
# so a no-claim row driven by one value type is a row driven by one arm.
# The id is written out rather than left to `repr`, because `repr(HUGE)`
# is 401 digits and a 493-character node id is not a name.
NO_CLAIM_VALUES = [
    ("bool", True),
    ("int-in-range", 1),
    ("int-out-of-int8", 256),
    ("int-past-every-float", HUGE),
    ("float-non-integral", 0.5),
    ("float-out-of-int32", -1e30),
    ("complex", complex(1.0, 2.0)),
]


@pytest.mark.parametrize("dtype", UNRECOGNISED)
@pytest.mark.parametrize("kind,val", NO_CLAIM_VALUES,
                         ids=[k for k, _v in NO_CLAIM_VALUES])
def test_an_unrecognised_dtype_string_gets_no_claim(dtype, kind, val):
    """The same posture `_load_itemsize` takes when a dtype code does not
    name a size: return None rather than guess. A guessed range is worse
    than no range, because it refuses documents this module has no
    standing to judge.

    **THIS ROW USED TO BE DRIVEN BY TWO INTS, AND THAT IS HOW IT SHIPPED
    BROKEN.** The complex-category arm ran BEFORE the dtype was
    recognised, so every one of these dtypes refused a complex value with
    *"no value of that dtype is complex"* — a claim about a dtype the
    module has just said it knows nothing about, and a FALSE one for
    `complex32` and `complex256`, whose values are all complex. Two int
    drives could not see it: they never entered that arm.

    The asymmetry is the proof and it is in this parametrization rather
    than in a sentence. `float128` and `complex256` are equally
    unrecognised; before the fix the first constructed and the second did
    not. The file's own standard is ONE TEST PER ROW OF THE RULE TABLE,
    DRIVEN, and for this row that means one per value type, because the
    rows above it dispatch on the value's type as hard as on the dtype's
    name."""
    lit = ir.Literal(val=val, aval=aval(dtype))
    assert lit.val == val, (kind, dtype)
    assert ir.literal_inexact(lit) is None, (kind, dtype)
    assert ir._literal_range_problem(val, aval(dtype)) == (None, None), (
        kind, dtype)


def test_the_unrecognised_row_and_the_recognised_one_disagree_about_NOTHING_but_the_dtype():
    """The finding stated as the one-line control it needed. Same value,
    two dtype strings the module is equally ignorant of, one of which
    merely LOOKS like a float and the other like a complex: the answer
    must be the same, and it must be no claim.

    Kept beside the parametrization above because the parametrization
    proves each cell separately and this proves they AGREE, which is the
    property the shipped arm broke."""
    v = complex(1.0, 2.0)
    assert (ir._literal_range_problem(v, aval("float128"))
            == ir._literal_range_problem(v, aval("complex256"))
            == ir._literal_range_problem(v, aval("key<fry>"))
            == (None, None))
    # ...while a dtype it DOES recognise still gets the category error,
    # which is what makes the arm above a narrowing rather than a removal
    assert "non-complex dtype" in refused(v, "float64")


# -- the tables, against something that is not them ---------------------------


def test_the_format_tables_agree_with_propagates():
    """`ir` may import nothing outside the standard library, and
    `propagate` imports `ir`, so the two tables cannot be one table. Two
    copies of one fact drift; this is what stops them."""
    assert ir._LIT_FLOAT_FORMATS == _FLOAT_FORMATS
    assert ir._LIT_INT_BOUNDS == {
        d: _INT_DTYPE_BOUNDS[d] for d in ir._LIT_INT_BOUNDS
    }
    # ...and `ir` carries every integer width `propagate` does, including
    # the 4-bit rows SPEC-LIT's own table leaves out. The only row of
    # `propagate`'s that is not here is `bool`, which this pass judges by
    # a rule of its own rather than as the integer range [0, 1] — and the
    # two rules DIFFER, on the float spelling: 1.0 is integral and inside
    # [0, 1], so the integer rule would admit it, and it is still not how
    # a bool is written.
    assert set(_INT_DTYPE_BOUNDS) - set(ir._LIT_INT_BOUNDS) == {"bool"}
    assert _INT_DTYPE_BOUNDS["bool"] == (0, 1) and "bool" in ir._LIT_DTYPE_NAME
    assert ir._literal_range_problem(1.0, aval("bool"))[0] is not None
    assert ir._literal_range_problem(1.0, aval("int8")) == (None, None)


def test_the_derived_float_bounds_are_the_ones_struct_encodes():
    """Every float bound is DERIVED from `(p, emin, emax)`; these are the
    same numbers read off the IEEE bit patterns, which is a different
    computation reaching the same place."""
    assert ir._LIT_FLOAT_MAX["float64"] == sys.float_info.max
    assert ir._LIT_FLOAT_MAX["float32"] == struct.unpack(
        "<f", struct.pack("<I", 0x7F7FFFFF))[0]
    assert ir._LIT_FLOAT_MAX["float16"] == struct.unpack(
        "<e", struct.pack("<H", 0x7BFF))[0]
    assert ir._LIT_FLOAT_MAX["bfloat16"] == struct.unpack(
        "<f", struct.pack("<I", 0x7F7F0000))[0]
    assert ir._LIT_FLOAT_MIN_SUBNORMAL["float64"] == 5e-324
    for dtype, (p, emin, emax) in ir._LIT_FLOAT_FORMATS.items():
        assert ir._LIT_FLOAT_OVERFLOW_AT[dtype] == 2 ** (emax + 1) - 2 ** (emax - p)
        assert isinstance(ir._LIT_FLOAT_OVERFLOW_AT[dtype], int)


@pytest.mark.parametrize("dtype,code", [("float32", "<f"), ("float16", "<e")])
def test_the_float_image_reads_structs_REFUSAL_as_the_infinity_it_stands_for(
    dtype, code
):
    """`_float_image` DELEGATES to `struct` for the two narrow formats
    `struct` encodes, so agreeing with it there is a tautology and is not
    what this asserts. What it asserts is the one place the delegation is
    not transparent: `struct` answers an out-of-range finite double with
    `OverflowError` rather than with a value, and reading that as `±inf`
    is only correct because it fires at the round-to-nearest TIE POINT
    and not at the largest finite. Both halves are driven — the value one
    ulp below the tie comes back finite, the tie itself raises."""
    tie = ir._LIT_FLOAT_OVERFLOW_AT[dtype]
    below = math.nextafter(float(tie), 0.0)
    assert struct.unpack(code, struct.pack(code, below))[0] == \
        ir._LIT_FLOAT_MAX[dtype]
    with pytest.raises(OverflowError):
        struct.pack(code, float(tie))
    assert ir._float_image(below, dtype) == ir._LIT_FLOAT_MAX[dtype]
    assert ir._float_image(float(tie), dtype) == math.inf
    assert ir._float_image(-float(tie), dtype) == -math.inf
    # ...and infinities and NaN go straight through rather than through
    # that arm, which is why the arm may assume a finite input
    assert ir._float_image(math.inf, dtype) == math.inf
    assert math.isnan(ir._float_image(math.nan, dtype))


def test_the_float64_image_is_the_identity():
    """A python `float` IS a binary64; there is nothing to round it to,
    and a round-trip through `struct`'s ``'d'`` would only be a slower
    way of saying so. The INEXACT of an INT under float64 is therefore
    carried by `float(val)` itself and is caught by comparing the image
    against the ORIGINAL value rather than against the converted one."""
    for v in (0.0, -0.0, 1.0, 0.1, 5e-324, sys.float_info.max, math.inf,
              -math.inf):
        assert ir._float_image(v, "float64") is v
    assert math.isnan(ir._float_image(math.nan, "float64"))
    assert ir._float_image(float(2 ** 53 + 1), "float64") != 2 ** 53 + 1


def test_bfloat16_agrees_with_jaxs_own_bfloat16_and_direct_rounding_would_not():
    """THE ONE FORMAT WITH NO INDEPENDENT IMPLEMENTATION IN THE STANDARD
    LIBRARY, checked against one that is not the standard library.

    The authority is `jax.numpy.bfloat16` — deliberately jax's and not
    some other project's, because the dtype NAME this pass keys on is
    ``str(np.dtype(...))`` of exactly that type (`propagate` says so at
    :data:`~stelling.propagate._BFLOAT16_DTYPE_STR`), so the bfloat16 this
    tree means is the bfloat16 jax has.

    SPEC-LIT says to verify the masking route rather than assume it, and
    verifying it FALSIFIED THE OBVIOUS IMPROVEMENT. Rounding binary64 to
    bfloat16 THROUGH binary32 rounds twice, and double rounding is not the
    identity: ``1 + 2**-8 + 2**-30`` lies above bfloat16's midpoint and a
    direct binary64->bfloat16 rounder answers 1.0078125, while binary32
    first rounds it down ONTO the midpoint exactly and ties-to-even then
    answers 1.0. I wrote that direct rounder, checked it against `struct`
    on the three formats `struct` knows (no disagreements, so the
    machinery was right), and then checked both against the authority:
    the masking route agreed everywhere and the direct rounder disagreed
    on the crafted subject below. The authority itself converts through
    binary32, so the "more correct" rounder was the one that disagreed
    with every bfloat16 that exists, and it was deleted from `ir.py`.

    **THE SWEEP IS RE-RUN HERE RATHER THAN REPORTED**, so neither this
    docstring nor `ir._bfloat16_image`'s states a count that nothing
    recomputes. The subject and the crafted counterexample are the same;
    the seed is fixed, so a disagreement names a value the next reader
    can paste into an interpreter.
    """
    jax = pytest.importorskip("jax")
    np = pytest.importorskip("numpy")
    subject = 1.0 + 2.0 ** -8 + 2.0 ** -30

    def authority_many(xs):
        a = np.array(xs, dtype=np.float64)
        # `over="ignore"`: a double past bfloat16's range casting to `inf`
        # is the ANSWER here, not an accident, and numpy warns about it
        with np.errstate(over="ignore"):
            b = a.astype(jax.numpy.bfloat16).astype(np.float64)
        return [float(v) for v in b]

    # THE FALSIFIED ALTERNATIVE, kept here because it is the measurement:
    # a direct binary64 -> bfloat16 rounder, exact integer arithmetic, no
    # intermediate format. It is *more* faithful to the definition of the
    # format and *less* faithful to the format that exists.
    def direct(x):
        p, emin = ir._LIT_FLOAT_FORMATS["bfloat16"][:2]
        if x == 0.0 or not math.isfinite(x):
            return x
        if abs(x) >= ir._LIT_FLOAT_OVERFLOW_AT["bfloat16"]:
            return math.inf if x > 0 else -math.inf
        _, e = math.frexp(abs(x))
        scale = max(e - 1, emin) - p + 1
        out = math.ldexp(float(round(math.ldexp(abs(x), -scale))), scale)
        return -out if x < 0 else out

    assert authority_many([subject]) == [1.0]
    assert direct(subject) == 1.0078125
    assert ir._float_image(subject, "bfloat16") == 1.0

    named = [0.0, -0.0, 1.0, -1.0, 0.1, -0.1, 1e-40, 1e-45, 257.0, 256.0,
             1.5, 3.0, 2.0 ** -133, 2.0 ** -134, 2.0 ** -135, math.pi,
             subject, ir._LIT_FLOAT_MAX["bfloat16"],
             -ir._LIT_FLOAT_MAX["bfloat16"], 1e-300, 1e300]
    rng = random.Random(11)
    sweep = [
        v for v in (
            struct.unpack("<d", struct.pack("<Q", rng.getrandbits(64)))[0]
            for _ in range(50_000)
        )
        if math.isfinite(v)
    ]
    values = named + sweep
    refs = authority_many(values)
    mine = [ir._float_image(v, "bfloat16") for v in values]
    disagree = [
        (v, r, m) for v, r, m in zip(values, refs, mine) if repr(r) != repr(m)
    ]
    assert not disagree[:5], (len(disagree), len(values), disagree[:5])
    # ...and the deleted rounder really would have disagreed: not on the
    # random sweep, where doubles are never that close to a bfloat16
    # midpoint, but on the one crafted double that is
    assert direct(subject) != authority_many([subject])[0]

    for v in (math.inf, -math.inf):
        assert ir._float_image(v, "bfloat16") == v
    assert math.isnan(ir._float_image(math.nan, "bfloat16"))


def test_the_subnormal_band_is_admitted_and_the_note_is_the_IEEE_route():
    """**`ir._float_image`'S ROUTE-DEPENDENCE LIVED ONLY IN A DOCSTRING,
    AND A MEASURED CLAIM WITH NO TEST ROTS SILENTLY.** It also rotted:
    that paragraph said a value landing in the float32 subnormal band
    "therefore records INEXACT", and it records NOTHING at all 24 of the
    magnitudes the same paragraph names, because each of them is an
    exact float32. This test is the paragraph, driven — every figure
    recomputed here so neither it nor this docstring carries a number
    that nothing checks.

    Three claims, in the order the paragraph makes them:

    1. **What the module answers.** ``2**e`` for ``e`` in ``-149 … -126``
       is exactly a float32, so :func:`ir._float_image` is the identity
       on it and the literal is admitted with NO note. That is the
       correction; the old sentence predicted the opposite at every one
       of these cells.
    2. **When a note does appear.** A value that ROUNDS inside the band
       gets one — ``1.5 * 2**-149`` has no float32 and stores as the
       next subnormal up. A value can also round UP out of the band
       entirely, onto ``2**-126``; it gets a note and it is the one
       image here the XLA convert does NOT destroy, which is why
       `ir._float_image`'s docstring no longer says the convert would
       destroy the band's literals *either way*. The band is admitted
       either way: the underflow REFUSAL lives strictly BELOW it, which
       is checked here too, since "admitted" is the half of the claim a
       reader relies on.
    3. **That the answer is a ROUTE.** `numpy.float32` and
       `jnp.asarray` agree with `struct` on all 24; an XLA
       ``convert_element_type`` on an f64 array flushes every subnormal
       to ``0.0`` and agrees only on the one magnitude that is NORMAL.
       Counted, not typed — if a future jaxlib stopped flushing, the
       docstring's "route-dependent" would be stale and this fails.

    The third claim needs an f64 jax array and therefore needs
    ``jax_enable_x64``, which is set and PUT BACK inside the test — the
    same set-and-restore `tests/test_ieee_narrow_formats.py` does one
    scope out, and `tests/_state_guard.py` brackets each test, so it
    names a LEAK and not a bracketed set. Measured, the flag is not
    optional here: without it ``jnp.asarray(x, dtype=jnp.float64)``
    hands back an f32, the convert is a no-op, and the cell goes green
    having tested nothing.
    """
    band = [2.0 ** e for e in range(-149, -125)]
    assert len(band) == 24
    assert band[-1] == 2.0 ** -126  # the one NORMAL magnitude in the list

    # (1) each of the 24 IS an exact float32 — 23 subnormals and the
    # smallest NORMAL — so the image is the identity, the literal is
    # admitted, and there is no note
    for x in band:
        assert ir._float_image(x, "float32") == x
        assert ir.literal_inexact(admitted(x, "float32")) is None

    # (2) a value that rounds INSIDE the band does get a note...
    rounder = 1.5 * 2.0 ** -149
    note = ir.literal_inexact(admitted(rounder, "float32"))
    assert note is not None and "stores as" in note, note
    image = ir._float_image(rounder, "float32")
    assert image == 2.0 ** -148 and repr(image) in note, (image, note)
    assert 0.0 < abs(image) < 2.0 ** -126  # and the note's value IS subnormal
    # ...AND A VALUE CAN ROUND UP OUT OF THE BAND ENTIRELY, which is the
    # cell that makes "the literal is admitted and the XLA convert would
    # destroy it" false in the branch it was written for. Just under the
    # midpoint below `2**-126` there is no float32, so this stores as
    # `2**-126` — a note, and the one image in this band the convert
    # PRESERVES; the assertion that it does is in the x64 block below.
    up = (2 ** 23 - 0.25) * 2.0 ** -149
    assert 0.0 < up < 2.0 ** -126
    assert ir._float_image(up, "float32") == 2.0 ** -126 != up
    up_note = ir.literal_inexact(admitted(up, "float32"))
    assert up_note is not None and repr(2.0 ** -126) in up_note, up_note
    # ...and the whole band is ADMITTED: the underflow refusal is below it
    assert "underflows float32" in refused(2.0 ** -151, "float32")
    assert "underflows float32" in refused(2.0 ** -150, "float32")  # exact tie
    for x in band + [rounder]:
        ir.Literal(val=x, aval=aval("float32"))  # constructs, does not raise

    # (3) the route disagreement, counted
    np = pytest.importorskip("numpy")
    # `jax`, not `jax.numpy`: `tests/test_skip_inventory.py` declares the
    # optional dependencies this suite may gate on by their TOP-LEVEL
    # name, and an undeclared gate fails that file
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    like_numpy = like_asarray = 0
    for x in band:
        mine = ir._float_image(x, "float32")
        like_numpy += float(np.float32(x)) == mine
        like_asarray += float(jnp.asarray(x, dtype=jnp.float32)) == mine
    assert like_numpy == 24, like_numpy
    assert like_asarray == 24, like_asarray

    # THE CONVERT LEG NEEDS AN f64 SOURCE, WHICH NEEDS x64, so it is set
    # and PUT BACK in a `finally`. x64 is process-global in jax and an
    # unrestored set leaks into every later test; `tests/_state_guard.py`
    # brackets each test and names the one that changed something, so a
    # set-and-restore inside one function is silent to it and a leak is
    # not. `tests/test_ieee_narrow_formats.py` does the same at module
    # scope for the same reason. Without the flag there is nothing to
    # measure rather than something weaker: `jnp.asarray(x,
    # dtype=jnp.float64)` silently hands back an f32, the convert becomes
    # a no-op, and all 24 "agree" — a green cell that has tested nothing.
    old_x64 = jax.config.jax_enable_x64
    try:
        jax.config.update("jax_enable_x64", True)
        convert = jax.jit(lambda a: a.astype(jnp.float32))
        like_convert = flushed = eager_flushed = 0
        for x in band:
            mine = ir._float_image(x, "float32")
            wide = jnp.asarray(x, dtype=jnp.float64)
            assert wide.dtype == jnp.float64  # the flag really took
            back = float(convert(wide))
            like_convert += back == mine
            flushed += back == 0.0
            # JIT IS NOT THE DISCRIMINATOR — the word the docstring at
            # `ir._float_image` used to use. Eager `.astype` lowers to the
            # same `convert_element_type` and flushes the same cells.
            eager_flushed += float(wide.astype(jnp.float32)) == 0.0
        # and the one it agrees on is the NORMAL magnitude, not an
        # arbitrary cell
        normal = jnp.asarray(band[-1], dtype=jnp.float64)
        assert float(convert(normal)) == band[-1] == 2.0 ** -126
        # ...and so is the value that ROUNDS onto it, which therefore
        # gets a note AND survives the convert — both halves of the
        # sentence that used to say the convert destroys it "either way"
        assert float(convert(jnp.asarray(up, dtype=jnp.float64))) == 2.0 ** -126
    finally:
        jax.config.update("jax_enable_x64", old_x64)
    assert flushed == 23, flushed
    assert eager_flushed == 23, eager_flushed
    assert like_convert == 1, like_convert


# -- the INEXACT recording mechanism ------------------------------------------


def test_inexact_is_not_a_warning_and_not_process_global():
    """SPEC-LIT §5 rules out three channels for INEXACT: raising it,
    warning it, and keeping it in process-global mutable state. Raising
    is driven by the INEXACT rows above, which admit the literal; the
    other two are driven here. `warnings` are process-global,
    order-dependent and off by default; a module-level list is state two
    concurrent `check()` calls interleave into. The recording is
    per-object, so two literals recorded in any order carry their own."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        a = admitted(0.1, "float32")
        b = admitted(0.2, "float32")
    assert ir.literal_inexact(a) != ir.literal_inexact(b)
    assert ir.literal_inexact(admitted(0.5, "float32")) is None
    # nothing at module scope accumulates them
    assert not [
        n for n, v in vars(ir).items()
        if isinstance(v, list) and "inexact" in n.lower()
    ]


def test_inexact_is_invisible_to_equality_hash_and_the_document():
    """It is an observation ABOUT the transcription, not part of the
    document. A dataclass FIELD would enter `_encode`, `__eq__`,
    `__hash__` and `content_hash`, so two literals denoting the same
    number would stop being equal because one was written in a lossier
    spelling — and `content_hash` is a cross-process stability
    commitment."""
    rounded = admitted(0.1, "float32")
    exact = admitted(ir._float_image(0.1, "float32"), "float32")
    assert ir.literal_inexact(rounded) is not None
    assert ir.literal_inexact(exact) is None
    assert rounded.val != exact.val  # they are different numbers...
    same = admitted(0.1, "float32")
    assert rounded == same and hash(rounded) == hash(same)
    doc = ir.Literal(val=0.1, aval=aval("float32"))
    assert "inexact" not in repr(ir._encode(doc, True)).lower()


def test_the_inexact_note_survives_serialization_by_being_RECOMPUTED():
    """The note is not in the document and does not need to be: it is a
    pure function of ``(val, dtype)``, so `from_dict` produces it again
    from the same two things the document does carry. That is the
    property that makes the storage choice free — the document stays
    exactly what it was (`content_hash` is a cross-process stability
    commitment) and a reader on the far side of a serialization still
    has the fact."""
    out = ir.Var(id=0, aval=aval("bool"))
    q = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(), invars=(), outvars=(out,),
            eqns=(ir.JaxprEqn(
                primitive="gt",
                invars=(ir.Literal(val=0.1, aval=aval("float32")),
                        ir.Literal(val=0.0, aval=aval("float32"))),
                outvars=(out,), params=()),),
        )
    )
    note = ir.literal_inexact(q.jaxpr.eqns[0].invars[0])
    assert note is not None
    back = ir.ClosedJaxpr.from_dict(q.to_dict())
    assert back == q and back.content_hash() == q.content_hash()
    assert ir.literal_inexact(back.jaxpr.eqns[0].invars[0]) == note


def test_a_CONST_is_not_range_checked_and_the_boundary_is_deliberate():
    """SPEC-LIT scopes this pass to `Literal`, and a `ClosedJaxpr` const
    is a value beside an aval in the same way. The census measured what
    the boundary costs: of the 3,622 consts the suite constructs, 556 are
    scalars and none of them violates, so the rule would refuse nothing
    there today. Driven so that the boundary is visible — if a later pass
    extends the rule to consts, this test names the decision it changes
    rather than going quietly green."""
    cv = ir.Var(id=0, aval=aval("int8"))
    q = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(constvars=(cv,), invars=(), outvars=(cv,), eqns=()),
        consts=(256,),
    )
    assert q.consts == (256,)
    assert ir.ClosedJaxpr.from_dict(q.to_dict()).consts == (256,)
    # ...while the identical value in a LITERAL is refused
    assert "would store as 0" in refused(256, "int8")


# -- the module's import rule -------------------------------------------------


def test_ir_imports_nothing_outside_the_standard_library_at_module_scope():
    """`ir.py`'s docstring is normative: it may import nothing outside the
    standard library, and `stelling._jax_compat` is the only module
    allowed to touch jax. This pass added `math` and `struct` and nothing
    else. `Array.to_numpy` imports numpy INSIDE the method, deliberately
    and with the reason written there, which is why the scan is of
    module-scope imports rather than of the token."""
    with open(ir.__file__, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    roots = set()
    for node in tree.body:  # MODULE SCOPE, which is the whole claim
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    assert roots <= sys.stdlib_module_names, sorted(roots - sys.stdlib_module_names)
    assert {"math", "struct"} <= roots
