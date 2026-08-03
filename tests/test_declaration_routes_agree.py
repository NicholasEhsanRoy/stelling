# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Every route into a declaration must reach the SAME decision, because the
storability guard keys on the operand's own type and a route that pre-converts
its bounds hands the guard a value already rounded.

Four routes have now been found, each after the previous was closed:
`np.integer` into `any_array`, `np.longdouble` into `any_array`, `float()`
through `any_pytree`'s `canon()`, and `float()` through
`contracts._closed_range`. The first two were fixable inside the guard. **The
last two were not, because the loss happened upstream of it** — which is why
this file tests ROUTE AGREEMENT rather than any single route's behaviour.

The measured defect: the hand form refused `int64 (0, 2**53 + 1)` while the
sugar admitted it with `hi` stored as `9007199254740992.0`, so the declared
integer was not in the box the tool then reasoned about.
"""
from __future__ import annotations

import math

import pytest

jax = pytest.importorskip("jax")
import numpy as np

from stelling.harness import any_array, any_pytree

# Bound pairs spanning the decision: exact, widening (admitted), narrowing
# (refused), and ordinary. `dtype` matters — the guard is about integers that
# binary64 cannot hold.
PAIRS = [
    ("int64", 0, 2**53),                 # exact
    ("int64", 0, 2**53 + 1),             # hi rounds DOWN -> narrows -> refuse
    ("int64", 2**53 + 1, 2**60),         # lo rounds DOWN -> widens -> admit
    ("int64", -(2**53) - 1, 0),          # lo rounds UP  -> narrows -> refuse
    # (the three annotations above were inverted — float() rounds 2**53+1
    # DOWN to 2**53, ties-to-even, so the hi narrows and both los shown
    # here widen or narrow opposite to what the old comments said —
    # measured, published in the investigation brief, and on the landing
    # session's known-unfixed list; the decisions never moved)
    ("int64", -(2**63), 2**63 - 1),      # the natural "any int64"
    ("uint64", 0, 2**64 - 1),
    ("uint64", 2**64 - 1, 2**64 - 1),
    ("int64", 0, 10),
    ("float64", 0.0, 100.0),
    ("float32", 0.0, 1.0),
    # the dtype-aware gate's class: an int bound past 2**53 on a dtype wholly
    # inside binary64 — the narrowing image drops no dtype value, both routes
    # must admit it (and must keep agreeing if the gate is ever re-litigated)
    ("int32", 0, 10**23),
    ("float32", 0, 10**23),
    # the gate's F1 boundary: a narrowing endpoint rounding onto a gap edge
    # around an EMPTY declared set — both routes must refuse it, with the
    # empty-set cause, and must keep agreeing
    ("float64", 2**53 + 1, 2**53 + 1),
    # the WIDENING half of that boundary and its one-step neighbours: the
    # declared interval sits strictly inside a representation gap (refused)
    # or reaches one of the gap's edges (admitted), and the two routes must
    # reach the same answer either way
    ("float64", 2**54 + 1, 2**54 + 3),
    ("float64", 2**54 + 1, 2**54 + 4),
    ("int64", -(2**64), -(2**63) - 1),
    ("int64", -(2**64), -(2**63)),
]


def _hand(dtype, lo, hi):
    try:
        jax.make_jaxpr(lambda: (any_array((1,), dtype, (lo, hi)),))()
        return "admit"
    except ValueError as e:
        return f"refuse:{str(e)[:40]}"


def _sugar(dtype, lo, hi):
    proto = np.zeros((1,), np.dtype(dtype))
    try:
        jax.make_jaxpr(lambda: any_pytree(proto, (lo, hi)))()
        return "admit"
    except ValueError as e:
        return f"refuse:{str(e)[:40]}"


@pytest.mark.parametrize("dtype,lo,hi", PAIRS)
def test_the_hand_and_sugar_routes_reach_the_same_decision(dtype, lo, hi):
    """The property, stated as agreement rather than as either route's answer,
    so it holds whichever way the guard's decision goes and fails the moment a
    route starts deciding for itself."""
    h, s = _hand(dtype, lo, hi), _sugar(dtype, lo, hi)
    assert h == s, (
        f"{dtype} ({lo!r}, {hi!r}): the hand form says {h!r} and the sugar "
        f"says {s!r}. A route that pre-processes its bounds hands the "
        f"storability guard a value already rounded, and the guard keys on the "
        f"operand's own type — so it never sees what it exists to judge."
    )


def _declared_dtype_extremes(dtype, lo, hi):
    """The declared interval clamped to the dtype's own value set: (smallest
    dtype value >= lo, largest dtype value <= hi). Computed with numpy alone —
    an oracle independent of the helpers in `_jax_compat` — and exactly:
    python compares int against float without rounding, and the nextafter walk
    moves one representable step at a time from the nearest cast."""
    d = np.dtype(dtype)
    if d.kind in "iu":
        info = np.iinfo(d)
        return (max(math.ceil(lo), int(info.min)),
                min(math.floor(hi), int(info.max)))
    up, down = np.array(np.inf, d), np.array(-np.inf, d)
    c_lo = np.array(lo, d)
    while float(c_lo) < lo:
        c_lo = np.nextafter(c_lo, up)
    c_hi = np.array(hi, d)
    while float(c_hi) > hi:
        c_hi = np.nextafter(c_hi, down)
    return float(c_lo), float(c_hi)


@pytest.mark.parametrize("dtype,lo,hi", PAIRS)
def test_an_admitted_declaration_stores_a_box_containing_what_was_declared(dtype, lo, hi):
        # ANTI-VACUITY on the thing that actually went wrong: not "does it
        # refuse" but "if it admits, is the declared value IN the stored box".
        # The defect admitted `hi = 2**53 + 1` and stored `2**53`.
        #
        # "The declared value" is the declared DTYPE-SET, not the raw real
        # endpoint: the storability guard is dtype-aware, and it admits a
        # narrowing endpoint exactly when the shaved slice holds no value of
        # the dtype — `int32 (0, 10**23)` stores `hi=1e+23`, below the raw
        # 10**23 and above every int32. So the containment asserted here is of
        # the endpoints CLAMPED to the dtype (an independent numpy oracle).
        # For int64/uint64 the clamped extremes coincide with every in-range
        # declared endpoint, so the original defect — sugar storing `2**53`
        # for a declared `2**53 + 1` on int64 — still fails this assert
        # (mutation-verified: re-adding `float()` to `canon` turns it red).
    proto = np.zeros((1,), np.dtype(dtype))
    try:
        cj = jax.make_jaxpr(lambda: any_pytree(proto, (lo, hi)))()
    except ValueError:
        return  # refused: nothing stored, nothing to check
    params = dict(cj.eqns[0].params)
    d_lo, d_hi = _declared_dtype_extremes(dtype, lo, hi)
    assert params["lo"] <= d_lo, (
        f"stored lo={params['lo']!r} is ABOVE {d_lo!r}, the smallest {dtype} "
        f"value of the declared [{lo!r}, {hi!r}]: the box excludes values "
        f"that were declared"
    )
    assert params["hi"] >= d_hi, (
        f"stored hi={params['hi']!r} is BELOW {d_hi!r}, the largest {dtype} "
        f"value of the declared [{lo!r}, {hi!r}]: the box excludes values "
        f"that were declared"
    )


def test_the_contract_template_route_does_not_pre_convert_either():
    """`contracts._closed_range` returned `float(rng[0]), float(rng[1])`, which
    both mis-declared the envelope and printed the rounded number into the
    contract's own requires-description."""
    from stelling.contracts import _closed_range
    v = 2**53 + 1
    lo, hi = _closed_range("t", "n", (0, v))
    assert hi == v and type(hi) is int, (
        f"_closed_range must return the caller's own value, got {hi!r}"
    )
    # and it still validates
    with pytest.raises(ValueError, match="empty envelope"):
        _closed_range("t", "n", (1, 0))
    with pytest.raises(ValueError, match="non-finite endpoint"):
        _closed_range("t", "n", (0, float("inf")))


def test_the_alias_comparison_still_ignores_int_versus_float_spelling():
    """`canon` no longer normalises, so this is worth pinning: python compares
    `0 == 0.0` as equal, so the same object given `(0, 1)` at one tree position
    and `(0.0, 1.0)` at another is still one declaration."""
    a = np.zeros((1,), np.float64)
    assert jax.make_jaxpr(
        lambda: any_pytree({"x": a, "y": a}, {"x": (0, 1), "y": (0.0, 1.0)})
    )() is not None
    # and genuinely different bounds on one object still refuse
    with pytest.raises(ValueError, match="aliases an earlier leaf"):
        jax.make_jaxpr(
            lambda: any_pytree({"x": a, "y": a}, {"x": (0, 1), "y": (0, 2)})
        )()


# -- the spelling floor -------------------------------------------------------
#
# One VALUE, every spelling of it the layer accepts, and both routes: the
# decision must be a function of (value, dtype), never of the spelling. The
# live defect this pins closed: `np.longdouble(2**53+1)` was ADMITTED
# recording `[2**53, 2**53]` — disjoint from the declared point — while the
# python-int spelling of the identical declaration was refused, and a claim
# false at the only declared point came back VERIFIED through
# preconditions.check. Spellings are compared against the python-int/float
# REFERENCE spelling of the same value, and hand-vs-sugar messages are
# compared in FULL (both routes delegate to any_array, so the bytes match).

import decimal
import fractions


def _spellings_of_int(v):
    """Every floor spelling of integer value ``v`` (np.longdouble included
    only where it holds ``v`` exactly — 2**53+1 fits its 64-bit
    significand)."""
    return [
        ("int", v),
        ("np.int64", np.int64(v)),
        ("np.longdouble", np.longdouble(2) ** 53 + 1
         if v == 2**53 + 1 else np.longdouble(v)),
        ("Decimal", decimal.Decimal(v)),
        ("Fraction", fractions.Fraction(v)),
        ("0d-int64-array", np.array(v, dtype=np.int64)),
        ("0d-longdouble-array", np.array(np.longdouble(2) ** 53 + 1)
         if v == 2**53 + 1 else np.array(np.longdouble(v))),
    ]


_FLOOR_CASES = [
    # (dtype, value-as-int, expected decision measured for the int spelling)
    ("int64", 2**53 + 1, "refuse"),   # narrowing hi on a dtype that loses
    ("float64", 2**53 + 1, "refuse"),  # declared point holds no float64
    ("int64", 2**53, "admit"),        # exact everywhere
    ("float64", 2**53, "admit"),
]


@pytest.mark.parametrize("dtype,value,expected", _FLOOR_CASES)
def test_every_spelling_of_one_value_reaches_the_int_spellings_decision(
    dtype, value, expected
):
    for tag, spelled in _spellings_of_int(value):
        h, s = _hand(dtype, spelled, spelled), _sugar(dtype, spelled, spelled)
        assert h.split(":")[0] == expected, (
            f"{dtype} point {value} spelled as {tag}: hand route decided "
            f"{h!r}, the python-int spelling decides {expected!r} — the "
            f"decision depended on the spelling"
        )
        assert h.split(":")[0] == s.split(":")[0], (
            f"{dtype} point {value} spelled as {tag}: hand={h!r} sugar={s!r}"
        )


# inexact-VALUED spellings (no python-int reference exists; the reference is
# the decision measured for the exact rational the spelling denotes)
_INEXACT_PAIRS = [
    # Decimal('0.1') is 1/10 exactly — not a binary64, lo narrows on int64
    ("int64", decimal.Decimal("0.1"), decimal.Decimal("0.2"), "refuse"),
    # same VALUE spelled as a Fraction must get the same refusal
    ("int64", fractions.Fraction(1, 10), fractions.Fraction(1, 5), "refuse"),
    # on float64 the narrowing drops no dtype value: admitted
    ("float64", decimal.Decimal("0.1"), decimal.Decimal("0.2"), "admit"),
    ("float64", fractions.Fraction(1, 10), fractions.Fraction(1, 5), "admit"),
    ("float64", fractions.Fraction(1, 3), fractions.Fraction(2, 3), "admit"),
    # a longdouble strictly between two binary64s, hi position: narrows,
    # drops no float64, admitted — and both routes must agree
    ("float64", np.longdouble("0.5"),
     np.longdouble("2.00000000000000000003"), "admit"),
]


@pytest.mark.parametrize("dtype,lo,hi,expected", _INEXACT_PAIRS)
def test_inexact_valued_spellings_agree_across_routes(dtype, lo, hi, expected):
    h, s = _hand(dtype, lo, hi), _sugar(dtype, lo, hi)
    assert h.split(":")[0] == expected, (h, expected)
    assert h == s, f"{dtype} ({lo!r}, {hi!r}): hand={h!r} sugar={s!r}"


def _full(route, dtype, lo, hi):
    """Decision plus FULL message — the routes delegate to any_array, so
    even the refusal bytes must match."""
    fn = {"hand": lambda: (any_array((1,), dtype, (lo, hi)),),
          "sugar": lambda: any_pytree(np.zeros((1,), np.dtype(dtype)), (lo, hi))}[route]
    try:
        jax.make_jaxpr(fn)()
        return "admit"
    except (ValueError, TypeError) as e:
        return f"refuse:{e}"


@pytest.mark.parametrize("dtype,lo,hi", [
    ("int64", np.longdouble(2) ** 53 + 1, np.longdouble(2) ** 53 + 1),
    ("float64", decimal.Decimal(2**53 + 1), decimal.Decimal(2**53 + 1)),
    ("float64", "0.25", "0.5"),                # str: refused by policy
    ("float64", decimal.Decimal("1e400"), decimal.Decimal("1e400")),
    ("int64", fractions.Fraction(1, 10), fractions.Fraction(1, 5)),
    # 0-d arrays, spelled to REFUSE: an admitted pair here compared
    # "admit" == "admit" and could never fail on refusal bytes (repair
    # round 1) — admit-agreement is pinned by the spelling-floor tests
    ("int64", np.array(2**53 + 1), np.array(2**53 + 1)),
])
def test_hand_and_sugar_refusal_messages_are_byte_identical(dtype, lo, hi):
    h, s = _full("hand", dtype, lo, hi), _full("sugar", dtype, lo, hi)
    assert h.startswith("refuse:"), "param must refuse for the bytes to matter"
    assert h == s


# -- the TEMPLATE routes, measured rather than argued -------------------------
#
# `_hand` and `_sugar` cover two ways to declare an input; the other public
# ways go through a template — `contracts.conditioning_2x2` (whose ranges
# pass `_closed_range`) and the two `preconditions` templates. Every one of
# them ends in `any_array`, but that is exactly the argument the call graph
# also supported for `any_pytree` and `_closed_range` before each was
# measured pre-converting its bounds, so the routes are MEASURED here.


def _template_routes(dtype, lo, hi):
    """The decision each template route reaches, as {route: decision}."""
    from stelling import contracts, preconditions

    def go(fn):
        try:
            jax.make_jaxpr(fn)()
            return "admit"
        except ValueError as e:
            return f"refuse:{e}"

    return {
        "contract": go(lambda: contracts.conditioning_2x2(
            dtype, (lo, hi), (1, 2), (0, 1), 10.0).harness()),
        "field_positive": go(
            lambda: preconditions.field_positive((1,), dtype, (lo, hi))),
        "scalar_nonzero": go(
            lambda: preconditions.scalar_nonzero(dtype, (lo, hi))),
    }


@pytest.mark.parametrize("dtype,lo,hi,expected", [
    # the widening-empty class, and its one-step neighbours which must stay
    # admitted on every route (a route that refuses these has stopped
    # delegating and started deciding)
    ("float64", 2**54 + 1, 2**54 + 3, "refuse"),
    ("float64", 2**54 + 1, 2**54 + 4, "admit"),
    ("int64", -(2**64), -(2**63) - 1, "refuse"),
    ("int64", -(2**64), -(2**63), "admit"),
])
def test_the_template_routes_reach_the_hand_routes_decision(
        dtype, lo, hi, expected):
    """Every public way to declare an input inherits `any_array`'s judgment,
    refusal bytes included — measured on the class where the judgment most
    recently changed."""
    ref = _full("hand", dtype, lo, hi)
    assert ref.split(":")[0] == expected, (ref, expected)
    assert _full("sugar", dtype, lo, hi) == ref
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)   # the templates build arrays
    try:
        routes = _template_routes(dtype, lo, hi)
    finally:
        jax.config.update("jax_enable_x64", old)
    for route, got in routes.items():
        assert got == ref, (
            f"{dtype} ({lo!r}, {hi!r}): the {route} route says {got!r} and "
            f"the hand route says {ref!r} — a template that pre-processes its "
            f"bounds hands the declaration layer a value already rounded, and "
            f"then decides for itself"
        )


def test_no_module_pre_converts_a_bound_before_any_array_sees_it():
    """AN INVARIANT WITH A TEST IT CANNOT DRIFT PAST, rather than a grep.

    Four routes have been found, each after the previous was closed, so the
    next one is the expected outcome rather than a surprise. This makes the
    property structural: no function in `src/stelling` may hand `any_array` a
    bound it has already pushed through `float()`, and no helper whose result
    feeds `any_array` may return one.

    It is deliberately coarse — it flags the SHAPE (a `float()` on a bound
    expression) rather than proving reachability — because a false alarm here
    costs a comment and the failure it prevents cost a soundness defect three
    times.
    """
    import ast
    import pathlib

    src = pathlib.Path(__file__).resolve().parent.parent / "src" / "stelling"
    offenders = []
    for f in sorted(src.glob("*.py")):
        text = f.read_text()
        # DERIVED, not hand-listed: only a module that mentions `any_array` can
        # be on a declaration route. Without this the check flags
        # `interval.py`'s truncating-division corners, which return
        # `float(min(qs)), float(max(qs))` as interval arithmetic and have
        # nothing to do with a declared bound — measured, and it is the reason
        # this filter exists rather than a module allowlist.
        #
        # THE LIMITATION IS REAL AND STATED: a NEW declaration route in a module
        # that reaches `any_array` indirectly would escape this. Four routes have
        # been found, each after the previous was closed, so a fifth is the
        # expected outcome — and this check narrows where to look rather than
        # proving there is nowhere left.
        if "any_array" not in text:
            continue
        tree = ast.parse(text)
        for node in ast.walk(tree):
            # (a) a bounds argument that is itself a float() call or a tuple of them
            if isinstance(node, ast.Call):
                fn = node.func
                name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
                if name == "any_array" and len(node.args) >= 3:
                    b = node.args[2]
                    elts = b.elts if isinstance(b, (ast.Tuple, ast.List)) else [b]
                    for e in elts:
                        if (isinstance(e, ast.Call)
                                and isinstance(e.func, ast.Name)
                                and e.func.id == "float"):
                            offenders.append(f"{f.name}:{node.lineno} any_array bound is float(...)")
            # (b) `return float(a), float(b)` — the canon()/_closed_range shape
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple):
                floats = [e for e in node.value.elts
                          if isinstance(e, ast.Call)
                          and isinstance(e.func, ast.Name) and e.func.id == "float"]
                if len(floats) == len(node.value.elts) == 2:
                    offenders.append(
                        f"{f.name}:{node.lineno} returns (float(...), float(...)) — "
                        f"the canon()/_closed_range shape"
                    )
    assert not offenders, (
        "a declaration route pre-converts its bounds:\n  "
        + "\n  ".join(offenders)
        + "\n`any_array`'s storability guard keys on the operand's own type, so "
          "a pre-converted bound arrives already rounded and the guard never "
          "sees the value it exists to judge. Pass the caller's own values "
          "through and let the guard decide — it already made this decision, "
          "and re-deciding it is how the layers came to disagree."
    )
