# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""What a declared bound's VALUE is, decided exactly or not at all.

The declaration layer records bounds as binary64 in the IR, and this
layer's defect history (three "a case the author did not enumerate"
escapes, ending in a measured false VERIFIED) is one pattern repeated:
storability was judged by the bound's TYPE while its VALUE went through
``float()``, a conversion that silently rounds exactly the values the
judgment exists to protect.

This module replaces the type question with a value question, asked in a
single exact domain: every accepted spelling is mapped to the exact
number it denotes — a :class:`fractions.Fraction` when finite, the
binary64 ``±inf``/``nan`` when not — and every downstream judgment (is
it storable? does rounding narrow or widen? does the declared interval
hold a dtype value?) is made by exact rational comparison. Spelling
independence is then structural rather than enumerated: two spellings of
one value map to one rational, so no route and no guard can tell them
apart.

THE FAMILY IS CLOSED, AND EVERY SPELLING OUTSIDE IT IS COVERED BY
REFUSAL. Accepted: python ``int``/``bool``/``float``; numpy integer,
floating (every width, ``longdouble`` included), and bool scalars;
``decimal.Decimal``; ``fractions.Fraction``; 0-d numpy arrays of those.
Anything else — ``str``, ``complex``, a jax array, an ``ml_dtypes``
scalar, any third-party number, any object with a ``__float__`` — makes
:func:`declared_bound_value` return None, and the declaration layer
refuses it loudly, naming this family. That is the whole coverage story
for spellings nobody thought of: they are refused BY DEFAULT, before any
conversion, so a new numeric type cannot ride in through ``float()`` and
round on the way in. (A ``str`` like ``'0.1'`` denotes a decidable
value, so its refusal is POLICY rather than necessity: accepting text
would reopen the family this module exists to close. Respell it
``Decimal('0.1')`` for the exact decimal, or ``0.1`` for the binary64.)

jax-free and numpy-lazy, so :mod:`stelling.contracts` can validate at
authoring time in a bare environment: numpy is recognized when it is
importable and irrelevant when it is not — a numpy-spelled bound cannot
exist in a process that has no numpy.
"""

from __future__ import annotations

import math
from decimal import Decimal
from fractions import Fraction

__all__ = ["ACCEPTED_SPELLINGS", "binary64_image", "declared_bound_value"]

# The one sentence every unknown-spelling refusal quotes, so the family
# is named identically at every route into the layer.
ACCEPTED_SPELLINGS = (
    "a python int, float, or bool; a numpy integer, floating, or bool "
    "scalar; a decimal.Decimal; a fractions.Fraction; or a 0-d numpy "
    "array of one of those (np.asarray(x) converts a concrete jax scalar)"
)


def _np():
    try:
        import numpy
    except ImportError:  # pragma: no cover — every tested venv has numpy
        return None
    return numpy


def declared_bound_value(raw):
    """The exact value ``raw`` declares: a :class:`Fraction` for a
    finite nonzero value; a FLOAT for the values binary64 holds exactly
    but Fraction cannot represent faithfully — ``±inf``, ``nan``, and
    the signed zeros (a Fraction has no ``-0``, and the recorded param
    must keep the zero's sign the parent recorded); or **None** for a
    spelling outside the accepted family — the caller refuses a None;
    nothing here or downstream ever converts one."""
    # bool before int (bool is an int subclass); both are exact already
    if isinstance(raw, bool):
        return Fraction(int(raw))
    if isinstance(raw, int):
        return Fraction(raw)
    if isinstance(raw, float):
        # np.float64 subclasses float and lands here; it IS a binary64,
        # so this branch and the np.floating one below agree on it. Two
        # value classes go through float() instead of Fraction: the
        # non-finite ones (nothing to round about ±inf/nan, and float()
        # NORMALIZES np.float64('inf') so the recorded param keeps the
        # python-float type the parent recorded) and ZERO, whose IEEE
        # sign a Fraction cannot carry — classifying -0.0 through
        # Fraction recorded +0.0 where the parent recorded -0.0, moving
        # recorded params and the query content hash for a spelling
        # class disclosed as unmoved (measured; blinded lens, repair
        # round 1). float() of a zero is exact and keeps the sign.
        if math.isfinite(raw) and raw != 0.0:
            return Fraction(raw)
        return float(raw)
    if isinstance(raw, Fraction):
        return raw
    if isinstance(raw, Decimal):
        if raw.is_nan():
            return math.nan
        if raw.is_infinite():
            return math.inf if raw > 0 else -math.inf
        if raw.is_zero():
            # Decimal('-0') is a signed zero too, and the parent's
            # float() image recorded -0.0 for it (measured); keep that
            return float(raw)
        return Fraction(raw)
    np = _np()
    if np is not None:
        if isinstance(raw, np.bool_):
            return Fraction(int(raw))
        if isinstance(raw, np.integer) and raw.dtype.kind in "iu":
            # numpy's scalar lattice is not a number lattice:
            # np.timedelta64 IS an np.integer subclass (measured; blinded
            # lens, repair rounds 1-2), so the isinstance test alone let
            # datetime arithmetic in as bounds: a tick count was silently
            # ADMITTED with its unit discarded — recorded as a plain
            # integer — for the generic form and every unit whose value
            # int() cannot turn into a datetime.timedelta (as/fs/ps/ns
            # and the calendar units M/Y; measured per unit), while the
            # timedelta-representable units (us through W) and NaT
            # crashed in int() with a bare TypeError, on every route. The
            # dtype-kind conjunct asks
            # numpy's VALUE system (kind 'i'/'u' means integer) rather
            # than its class tree, so timedelta64 (kind 'm'),
            # datetime64 (kind 'M') and anything else the lattice grafts
            # onto an integer parent falls through to the default
            # refusal. np.floating below needs no such conjunct: every
            # np.floating subclass this numpy has is kind 'f', and an
            # unexercisable guard would be an unpinnable half.
            return Fraction(int(raw))
        if isinstance(raw, np.floating):
            # as_integer_ratio is exact at every numpy width, the
            # longdouble 64-bit significand included; float(raw) is the
            # rounding this module exists to avoid, and it is also
            # WRONG here — float(np.longdouble('1e400')) silently
            # returns inf for a finite declared value
            if np.isnan(raw):
                return math.nan
            if np.isinf(raw):
                return math.inf if raw > 0 else -math.inf
            if raw == 0:
                # ±0.0 at any width: float() is exact and keeps the
                # sign the parent recorded; as_integer_ratio drops it
                return float(raw)
            return Fraction(*raw.as_integer_ratio())
        if isinstance(raw, np.ndarray) and raw.ndim == 0:
            inner = raw[()]
            # a 0-d array unwraps to its numpy scalar; a dtype the
            # family does not hold (complex, datetime, object) falls
            # through to None exactly as the bare scalar would
            if not isinstance(inner, np.ndarray):
                return declared_bound_value(inner)
    return None


def binary64_image(v):
    """The binary64 the IR would record for exact value ``v`` (an output
    of :func:`declared_bound_value`): round-to-nearest, ties-to-even,
    overflowing to the infinity of ``v``'s sign.

    Agrees with ``float(raw)`` everywhere ``float(raw)`` returns a
    finite answer (both are correct nearest-even rounding; measured over
    the spelling grid in tests), but is TOTAL where ``float`` is not:
    ``float(10**400)`` raises OverflowError while the longdouble
    spelling of the same magnitude silently returned inf — one value,
    two behaviours. Here the overflow that makes ``float`` raise is
    caught and mapped to the signed infinity the rounding denotes, and
    the caller judges that image (it is non-finite for a finite declared
    value exactly when the value cannot be recorded at all).

    CPython's int/int true division is correctly rounded, which is what
    makes ``numerator / denominator`` the exact rounding rather than an
    approximation of it.
    """
    if isinstance(v, float):
        # ±inf, nan, and the signed zeros: binary64 holds each exactly,
        # nothing to round (the zero's sign survives precisely because
        # it never went through Fraction)
        return v
    try:
        return v.numerator / v.denominator
    except OverflowError:
        return math.inf if v > 0 else -math.inf
