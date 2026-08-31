# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""A LITERAL'S VALUE IS A VALUE OF ITS AVAL'S DTYPE, OR IT IS REFUSED.

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

**AND THE TABLES ARE CHECKED AGAINST SOMETHING THAT IS NOT THEM.**
`ir` may import nothing outside the standard library, so its float and
integer bounds are a second copy of `propagate`'s; the copies are
compared here rather than trusted. The bfloat16 image, which has no
`struct` code and could only be written by hand, is compared against
`jax.numpy.bfloat16` — the bfloat16 this tree's dtype NAMES come from —
and the comparison found the "obviously more correct" direct rounder to
be the wrong one. See
:func:`test_bfloat16_agrees_with_jaxs_own_bfloat16_and_direct_rounding_would_not`.
"""
from __future__ import annotations

import ast
import math
import random
import struct
import sys

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
    """An `Array`'s bytes ARE the dtype: `_validate_array_value` already
    holds ``len(data) == product(shape) x itemsize``, so a fixed-width
    buffer cannot encode a value outside the width it is measured
    against. ONLY A SCALAR CAN LIE.

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


@pytest.mark.parametrize("dtype", ["key<fry>", "float0", "float8_e4m3fn",
                                   "void16", "xx", "int128", "not a dtype"])
def test_an_unrecognised_dtype_string_gets_no_claim(dtype):
    """The same posture `_load_itemsize` takes when a dtype code does not
    name a size: return None rather than guess. A guessed range is worse
    than no range, because it refuses documents this module has no
    standing to judge.

    None of these is invented. `key<fry>` is a real jax extended dtype
    (`stelling._tripwire.prop_guard` names it); `float0` and
    `float8_e4m3fn` are real jax dtypes with no row here; and `void16`,
    `xx` and `int128` are dtype strings the suite itself builds literals
    under — measured in the recorder census, which found literals under
    19 distinct dtype strings including those three."""
    assert ir.Literal(val=HUGE, aval=aval(dtype)).val == HUGE
    assert ir._literal_range_problem(256, aval(dtype)) == (None, None)


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
