# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""A declared box the dtype cannot hold is EMPTY, and empty declarations are
refused at declaration time — the fourth instance of a posture the harness
already held for a negative extent, `lo > hi`, and the infinite point.

Found because `sign` returned `[-1, -1]` on a `uint8` box of `(-3, -1)` at
100% coverage and minted a REFUTED. Routing `sign` through the overflow guard
fixed one row; a MAJORITY of transfers admitted that box, including all six
comparisons, which return a definite boolean straight into an assert. (The
exact count depends on an operand convention that was never stated: 13 with a
valid second operand, 20 of 26 traceable rows with the impossible box on every
operand. Neither number is robust; the qualitative claim is.) The hole is at
the declaration, so the check is too.

The design rule is NOT "a bound outside the dtype's range" — it is "no
representable value inside the interval". A box wider than the dtype is an
over-approximation and stays sound; a box disjoint from it describes nothing.
"""
from __future__ import annotations

import functools
import math
import re
import warnings
from fractions import Fraction

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
import numpy as np
from jax import lax

from stelling import interval as iv
from stelling import propagate as P
from stelling import _jax_compat as JC
from stelling._bound_spelling import binary64_image, declared_bound_value
from stelling.harness import any_array, assert_, trace


@pytest.fixture(autouse=True)
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def _declare(dtype, lo, hi):
    return jax.make_jaxpr(lambda: (any_array((1,), dtype, (lo, hi)),))()


_NEIGHBOUR_RE = re.compile(
    r"The nearest (?:representable integers|\S+ values) are ([^;]+);"
)


def _parse_neighbours(msg):
    """The neighbour clause, parsed into `{below, above}` numbers.

    A PARSER RATHER THAN A SUBSTRING TEST, because the substring test could not
    fail: `"0 above"` is contained in `"0 below and 0 above"`, so the collapsed
    message satisfied every expectation the correct one did. And the previous
    anti-collapse control split on the FIRST `" and "`, which sits inside
    *"the integers [-128, 127] and the interval contains none of them"* — so it
    compared a sentence to a number and `b != a` could never be False.
    """
    m = _NEIGHBOUR_RE.search(msg)
    assert m, f"no neighbour clause found in:\n  {msg}"
    out = {"below": None, "above": None}
    for part in m.group(1).split(" and "):
        part = part.strip()
        for side in ("below", "above"):
            if part.endswith(" " + side):
                out[side] = float(part[: -len(side) - 1])
    assert any(v is not None for v in out.values()), f"unparsed: {m.group(1)!r}"
    return out


def _fmt_sides(sides):
    """Canonical rendering of a parsed clause, for exact comparison."""
    bits = []
    if sides["below"] is not None:
        bits.append(f"{sides['below']:g} below")
    if sides["above"] is not None:
        bits.append(f"{sides['above']:g} above")
    return " and ".join(bits)


# --------------------------------------------------------------------------
# what it rejects
# --------------------------------------------------------------------------
@pytest.mark.parametrize("dtype,lo,hi,why", [
    ("uint8", -3.0, -1.0, "the audit's box — no uint8 in [-3, -1]"),
    ("uint8", 256.0, 300.0, "entirely above uint8"),
    ("int8", 200.0, 300.0, "entirely above int8"),
    ("int8", 0.2, 0.8, "no INTEGER lies in the interval"),
    ("bool", 2.0, 3.0, "no bool in [2, 3]"),
    ("float32", 1e39, 1e40, "above float32's finite max"),
    ("float32", -1e40, -1e39, "below float32's finite min"),
])
def test_rejects_a_box_the_dtype_cannot_hold(dtype, lo, hi, why):
    with pytest.raises(ValueError, match="EMPTY under dtype"):
        _declare(dtype, lo, hi)


# --------------------------------------------------------------------------
# what it admits — the half that matters more
# --------------------------------------------------------------------------
@pytest.mark.parametrize("dtype,lo,hi,why", [
    ("uint8", -3.0, 10.0, "partial overlap: an over-approximation, still sound"),
    ("uint8", 0.0, 255.0, "the ordinary full-range box"),
    ("int8", -200.0, 200.0, "wider than int8 — over-approximation"),
    ("int8", 0.0, 0.5, "contains the integer 0"),
    ("int32", -(2.0**31), 2.0**31 - 1, "the exact int32 range"),
    ("float32", 0.0, 1e39, "upper bound unrepresentable, box still non-empty"),
    ("float32", 1e-45, 1e-40, "the subnormal band"),
    ("float64", 0.0, 100.0, "the corpus's ordinary case"),
    ("float64", 0.0, float("inf"), "half-infinite: unbounded above"),
    ("complex64", -3.0, 3.0, "complex admitted unconditionally, by policy"),
    # A WIDENING BOUND ON A DTYPE THE EXACT RE-CHECK CANNOT JUDGE. Since the
    # re-check covers the widening direction, a widening bound reaches it for
    # EVERY dtype — a narrowing one could not, because each dtype below
    # answers "can lose" and is refused at the storability gate first. The
    # re-check's walk steps through the DTYPE and compares in BINARY64, so it
    # is screened to the formats binary64 contains, and the screen is an
    # ADMIT. Measured with it absent, on these very params:
    #   complex  -> TypeError. jnp.finfo answers for the float64 COMPONENT,
    #               and float() of a complex array raises — a crash where
    #               the policy says "no claim either way".
    #   object   -> TypeError inside np.nextafter.
    #   float128 -> A FALSE REFUSAL, the one that matters: x86 longdouble
    #               HOLDS 2**54+1, but the step up from 2**54 is smaller
    #               than a binary64 ulp and float() rounds it straight back,
    #               so the walk cannot advance and reports an empty interval
    #               for one with three inhabitants.
    ("complex128", 2**54 + 1, 2**54 + 3, "widening bound, complex by policy"),
    ("complex64", 2**54 + 1, 2**54 + 3, "widening bound, complex by policy"),
    ("object", 2**54 + 1, 2**54 + 3, "a dtype neither lookup knows"),
    (str(np.dtype(np.longdouble)), 2**54 + 1, 2**54 + 3,
     "a format WIDER than binary64 — it holds every value declared"),
])
def test_admits_every_legitimate_envelope(dtype, lo, hi, why):
    """A declaration check that refuses a legitimate envelope is worse than
    the hole it closes, so this half is the load-bearing one."""
    assert _declare(dtype, lo, hi) is not None


def test_float_bounds_ARE_tested_for_exact_representability():
    """REVERSED from the first design, deliberately and after measurement.

    The original rule tested RANGE only, justified by "refusing
    `float32 (0.1, 0.1)` would reject the most ordinary envelope there is."
    A blinded audit showed the same under-reach admits an interval lying
    wholly inside a representation gap — `float32 (1e-50, 1e-49)` — which
    reached a REFUTED at 100% coverage. No rule admits one and rejects the
    other: both hold no value of the dtype.

    The trade resolves asymmetrically. Admitting `(0.1, 0.1)` costs nothing —
    it IS an empty set, and a verdict over it is meaningless whether or not
    the check fires. Admitting the gap interval mints a false counterexample.
    And refusing a point declaration at a decimal literal is not over-strict:
    it tells the user their declaration does not mean what they think, which
    is this check's whole purpose.
    """
    for v in (0.1, 1.0 / 3.0, float(np.pi)):
        assert float(np.float32(v)) != v, f"{v} must not be a float32 exactly"
        with pytest.raises(ValueError, match="EMPTY under dtype"):
            _declare("float32", v, v)
    # float64 is unaffected: every python float IS a float64, so a point
    # declaration there is always exactly representable
    for v in (0.1, 1.0 / 3.0, float(np.pi)):
        assert _declare("float64", v, v) is not None
    # and a RANGE spanning the gap is still admitted
    assert _declare("float32", 0.1, 0.2) is not None


def test_the_refusal_names_the_nearest_representable_values():
    """The message carries the weight — the `div`-guard pattern: name the
    primitive, give the reason, PRINT THE NUMBERS. A user who writes
    `float32 (0.1, 0.1)` must learn what to write instead."""
    with pytest.raises(ValueError) as exc:
        _declare("float32", 0.1, 0.1)
    msg = str(exc.value)
    assert "nearest float32 values" in msg
    sides = _parse_neighbours(msg)
    # WHICH SIDE, not merely which numbers. Swapping the two words left the
    # suite green before this, because the assertion only checked that both
    # reprs appeared somewhere.
    assert sides["below"] == float(np.nextafter(np.float32(0.1), np.float32(0)))
    assert sides["above"] == float(np.float32(0.1))
    assert sides["below"] < 0.1 < sides["above"], (
        f"the float path's direction words do not bracket the interval: {msg}"
    )
    # the integer branch names its neighbours too, parsed the same way
    with pytest.raises(ValueError) as exc2:
        _declare("int8", 0.2, 0.8)
    isides = _parse_neighbours(str(exc2.value))
    assert (isides["below"], isides["above"]) == (0.0, 1.0)
    assert isides["below"] < 0.2 and isides["above"] > 0.8


# --------------------------------------------------------------------------
# the defect it closes, and the anti-vacuity control
# --------------------------------------------------------------------------
def _sign_query():
    def h():
        x = any_array((1,), "uint8", (-3.0, -1.0))
        return assert_(lax.sign(x)[0] >= 0)
    return h


def test_the_refuted_is_gone():
    with pytest.raises(ValueError, match="EMPTY under dtype"):
        trace(_sign_query())


def test_the_check_is_load_bearing(monkeypatch):
    """ANTI-VACUITY. Neuter the check and the REFUTED must come back —
    otherwise the test above passes for a reason unrelated to it.

    `sign` itself now also declines this box via the overflow guard, so the
    control reads a *neighbouring* transfer: `neg`, which passes the guard
    (its box [1, 3] IS inside uint8) and whose answer is definite. The
    obligation is kept inside the uint8 domain deliberately — routing through
    `convert_element_type` would decline on uint8→float64 and give an
    `unknown` for a reason unrelated to what this control tests.
    """
    monkeypatch.setattr(JC, "_dtype_holds_a_value_in", lambda dt, lo, hi: (True, ""))

    def h():
        x = any_array((1,), "uint8", (-3.0, -1.0))
        return assert_(lax.neg(x)[0] <= 0)   # box [1, 3]: definitely violated

    p = P.propagate(trace(h))
    assert p.obligations[0].status == "violated-over-set", (
        f"with the check neutered the dtype-impossible box must reach the "
        f"transfers and mint a REFUTED; got {p.obligations[0].status}. If it "
        f"does not, this file is not testing what it claims to"
    )
    assert p.coverage.unknown == 0, "and it must do so at FULL coverage"


def test_the_refusal_happens_AT_TRACE_before_a_mode_exists():
    """The refusal cannot be mode-dependent, and this says so by testing where
    it happens rather than by running it twice.

    An earlier version parametrized over `real`/`ieee` and wrapped
    `P.propagate(trace(q), semantics=mode)` in `pytest.raises` — but `trace`
    raises first, so `semantics` was never evaluated and the two
    parametrizations were byte-identical. A blinded audit caught it. The
    docstring claimed the modes were "exercised rather than argued"; for that
    half they were argued.
    """
    with pytest.raises(ValueError, match="EMPTY under dtype"):
        trace(_sign_query())          # raises HERE — no propagate, no mode


@pytest.mark.parametrize("mode", ["real", "ieee"])
def test_an_admitted_declaration_still_propagates_in_both_modes(mode):
    """The half that CAN differ by mode: it must trace and propagate."""
    def ok():
        x = any_array((1,), "uint8", (0.0, 255.0))
        return assert_(jnp.sum(jnp.asarray(lax.sign(x), jnp.float64)) >= 0.0)
    assert P.propagate(trace(ok), semantics=mode).obligations[0].status in (
        "discharged", "unknown"
    )


# --------------------------------------------------------------------------
# the surface, re-measured — the measurement that found it is the criterion
# --------------------------------------------------------------------------
SURFACE = {
    "neg": lambda x: lax.neg(x),
    "copy": lambda x: jnp.array(x, copy=True),
    "stop_gradient": lax.stop_gradient,
    "square": jnp.square,
    "integer_pow": lambda x: x ** 2,
    "min": lambda x: jnp.minimum(x, x),
    "max": lambda x: jnp.maximum(x, x),
    "lt": lambda x: x < x, "gt": lambda x: x > x, "ge": lambda x: x >= x,
    "le": lambda x: x <= x, "eq": lambda x: x == x, "ne": lambda x: x != x,
}


def test_the_declaration_refuses_before_any_transfer_runs():
    """One refusal, not thirteen — and the distinction is the point.

    The previous version parametrized this over all 13 transfers, but
    `any_array` raises before the traced function body is ever evaluated, so
    `fn` was never called for any of them: measured, 0 invocations. All 13
    cases were byte-identical and the test would have passed with a single
    bogus entry. A re-audit caught it, the third vacuity finding in this file.
    """
    called = []

    def h():
        x = any_array((1,), "uint8", (-3.0, -1.0))
        called.append(1)                      # never reached
        return assert_(lax.neg(x)[0] <= 0)

    with pytest.raises(ValueError, match="EMPTY under dtype"):
        trace(h)
    assert not called, "the refusal must precede the body, not follow it"


@pytest.mark.parametrize("prim", sorted(SURFACE))
def test_each_entry_point_really_was_open(prim, monkeypatch):
    """THE PARAMETRIZATION THAT MEANS SOMETHING: with the check neutered, each
    of these transfers ADMITS the dtype-impossible box. That is what makes
    "thirteen entry points" a count rather than a phrase — the refusal closes
    them all at once, so only this direction can distinguish them."""
    monkeypatch.setattr(JC, "_dtype_holds_a_value_in", lambda dt, lo, hi: (True, ""))
    from stelling._jax_compat import transcribe
    cj = transcribe(jax.make_jaxpr(SURFACE[prim])(jnp.zeros((1,), "uint8")))
    eqn = [e for e in cj.jaxpr.eqns if str(e.primitive) == prim][0]
    bad = iv.IntervalArray(shape=(1,), los=(-3.0,), his=(-1.0,))
    tf, _tier = P.TRANSFERS[prim]
    out = tf(eqn, dict(eqn.params_dict()), [bad] * len(eqn.invars))
    assert out is not None, (
        f"{prim} was listed as an entry point but declines the box on its own; "
        f"the surface count is wrong"
    )


def _uint8_outcome(prim):
    """(outcome, box) for `prim` driven on an in-range uint8 box."""
    from stelling._jax_compat import transcribe
    cj = transcribe(jax.make_jaxpr(SURFACE[prim])(jnp.zeros((1,), "uint8")))
    eqn = [e for e in cj.jaxpr.eqns if str(e.primitive) == prim][0]
    box = iv.IntervalArray(shape=(1,), los=(0.0,), his=(255.0,))
    tf, _tier = P.TRANSFERS[prim]
    try:
        out = tf(eqn, dict(eqn.params_dict()), [box] * len(eqn.invars))
    except iv.IntervalError:
        return "refused", None
    if out is None:
        return "top", None
    if eqn.outvars[0].aval.dtype != "uint8":
        return "not-uint8", out
    return "uint8-box", out


def test_a_legitimate_box_never_leaves_the_dtype():
    """The OTHER entry point: a box arriving by propagation, not declaration.

    REWRITTEN — the parametrized version could not fail. A blinded audit
    measured that 9 of 13 rows exited before the assertion (6 produce bools,
    3 raise) and the 4 that reached it are identity/selection rows fed the
    same box on every operand, so it re-asserted its own input. No input
    could have made it fail.

    So the outcome DISTRIBUTION is asserted, not just the surviving cases:
    at least one row must actually refuse (proving the overflow guard is on
    this path at all) and at least one must return a uint8 box (proving the
    range assertion is reached), and every returned uint8 box must be in
    range.
    """
    outcomes = {p: _uint8_outcome(p) for p in SURFACE}
    kinds = [k for k, _ in outcomes.values()]
    assert "refused" in kinds, (
        "no row refused, so the overflow guard is not exercised here and the "
        "in-range assertion below is decorative"
    )
    assert "uint8-box" in kinds, "no row reached the range assertion at all"
    for prim, (kind, out) in outcomes.items():
        if kind != "uint8-box":
            continue
        assert 0 <= min(out[0].los) and max(out[0].his) <= 255, (
            f"{prim} took an in-range uint8 box to [{min(out[0].los)}, "
            f"{max(out[0].his)}], which leaves the dtype"
        )


def test_that_range_assertion_can_actually_fail(monkeypatch):
    """ANTI-VACUITY for the test above: doctor a row to leave the dtype and
    the assertion must catch it. Registry restored by monkeypatch."""
    orig = P.TRANSFERS["copy"]
    monkeypatch.setitem(
        P.TRANSFERS, "copy",
        (lambda eqn, params, ins: [iv.IntervalArray(
            shape=ins[0].shape, los=(-1.0,), his=(300.0,))], orig[1]),
    )
    kind, out = _uint8_outcome("copy")
    assert kind == "uint8-box"
    assert not (0 <= min(out[0].los) and max(out[0].his) <= 255), (
        "the doctored row must violate the assertion; if it does not, the "
        "assertion is not checking the range"
    )


# --------------------------------------------------------------------------
# regressions from the blinded audits of THIS check — each was live in a
# green tree, and two of them minted a definite verdict over an empty set
# --------------------------------------------------------------------------
@pytest.mark.parametrize("dtype,lo,hi", [
    # the integer half: widths the bounds registry has never heard of
    ("int2", 100.0, 200.0),
    ("uint2", -3.0, -1.0),
    # the float half: every ml_dtypes extended type, for which numpy's finfo
    # raises and the old code fell through to "admit"
    ("bfloat16", 1e39, 1e40),
    ("float8_e4m3fn", 1e39, 1e40),
    ("float8_e5m2", 1e39, 1e40),
    ("float4_e2m1fn", 10.0, 20.0),
    ("float6_e3m2fn", 100.0, 200.0),
    # the asymmetric range: exponent-only, all finite values strictly positive,
    # so negating the max would have declared this inhabited
    ("float8_e8m0fnu", -100.0, -50.0),
])
def test_no_dtype_is_silently_exempt(dtype, lo, hi):
    """The check dispatched on `_INT_DTYPE_BOUNDS` membership and `numpy.finfo`,
    and a blinded audit measured that combination exempting SIXTEEN of the thirty
    dtypes jax builds arrays in — so the box this check exists to reject was
    accepted verbatim one width away: `uint8 (-3,-1)` refused, `uint2 (-3,-1)`
    admitted and DISCHARGED at 100% coverage, a VERIFIED over an empty set.
    """
    import ml_dtypes
    t = getattr(ml_dtypes, dtype, None)
    name = np.dtype(t).name if t is not None else dtype
    with pytest.raises(ValueError, match="EMPTY under dtype"):
        _declare(name, lo, hi)


def test_the_exempt_dtypes_really_did_mint_a_verdict(monkeypatch):
    """ANTI-VACUITY for the parametrization above — REWRITTEN, because the
    first version asserted `monkey is P.TRANSFERS` after binding
    `monkey = P.TRANSFERS`, which is `x is x`, and computed no verdict at all
    while claiming to show one. A re-audit caught it.

    This neuters the check and shows the uint2 box reaching a DEFINITE verdict
    at full coverage, which is the thing the refusal prevents.
    """
    import ml_dtypes
    name = np.dtype(ml_dtypes.uint2).name
    assert (int(jnp.iinfo(np.dtype(ml_dtypes.uint2)).min),
            int(jnp.iinfo(np.dtype(ml_dtypes.uint2)).max)) == (0, 3)

    monkeypatch.setattr(JC, "_dtype_holds_a_value_in", lambda dt, lo, hi: (True, ""))

    def h():
        x = any_array((1,), name, (-3.0, -1.0))     # no uint2 is negative
        return assert_(jnp.sum(jnp.asarray(x == x, jnp.float64)) >= -1e30)

    p = P.propagate(trace(h))
    assert p.obligations[0].status == "discharged" and p.coverage.unknown == 0, (
        f"with the check neutered the empty uint2 box must reach a DEFINITE "
        f"verdict at full coverage — that is what makes the refusal load-"
        f"bearing; got {p.obligations[0].status} at unknown={p.coverage.unknown}"
    )


@pytest.mark.parametrize("dtype,lo,hi", [
    ("int64", float(2**63), float(2**64)),
    ("uint64", float(2**64), float(2**64 + 9)),
])
def test_the_integer_comparison_is_exact_at_the_64_bit_boundary(dtype, lo, hi):
    """`float(2**63 - 1)` rounds UP to `2**63`, so comparing against a
    float-converted dtype bound made int64/uint64 wrong at exactly the boundary
    the bounds registry's own comment says it exists to be exact at. A box
    wholly above int64 was ADMITTED while the same shape on uint8 was refused.
    """
    with pytest.raises(ValueError, match="EMPTY under dtype"):
        _declare(dtype, lo, hi)


def test_a_bound_that_the_ir_cannot_store_exactly_is_refused_by_name():
    """The other half of the 64-bit finding, and it is a REFUSAL of a
    legitimate envelope — so the message must name the representability loss
    rather than claim the user's set was empty.

    `int64 (2**63 - 1, 2**63 - 1)` is inhabited (by int64 max), but the IR
    stores bounds as binary64 and `float(2**63 - 1)` is `2**63`, which no
    int64 holds. Silently propagating that box was measured minting both a
    vacuous VERIFIED and a vacuous REFUTED at 100% coverage.
    """
    with pytest.raises(ValueError, match="not representable as the binary64"):
        _declare("int64", 2**63 - 1, 2**63 - 1)
    # and an exactly-storable integer bound is unaffected
    assert _declare("int64", 2**53, 2**53) is not None


def test_a_zero_size_declaration_is_never_refused():
    """The one FALSE-REJECTION the audits found, and the failure this check
    must not have. A zero-size array satisfies ANY element-wise bounds
    vacuously and jax constructs one, so no (bounds, dtype) pair is empty for
    it — but the dtype check is per-element and never sees the shape.
    """
    assert jax.make_jaxpr(
        lambda: (any_array((0,), "uint8", (-3.0, -1.0)),))() is not None
    empty = jnp.zeros((0,), "uint8")
    assert bool(jnp.all(empty >= -3)), "the declaration is vacuously inhabited"
    # and the same bounds at a non-zero extent still refuse
    with pytest.raises(ValueError, match="EMPTY under dtype"):
        _declare("uint8", -3.0, -1.0)


def test_the_representation_gap_is_CLOSED_and_pinned_both_directions():
    """A5, decided and taken: the exact test.

    `float32 (1e-50, 1e-49)` sits below the smallest subnormal and holds no
    float32. Under the range-only rule it was admitted and reached a REFUTED
    at 100% coverage. It is refused now — and the test pins BOTH directions,
    so a regression to range-only fails here and so does an over-reach that
    starts refusing inhabited intervals.
    """
    smallest = float(np.nextafter(np.float32(0), np.float32(1)))
    assert smallest > 1e-49, "the interval genuinely holds no float32"
    with pytest.raises(ValueError, match="gap between representable values"):
        _declare("float32", 1e-50, 1e-49)
    with pytest.raises(ValueError, match="gap between representable values"):
        _declare("float32", -1e-49, -1e-50)      # the negative twin
    # THE OTHER DIRECTION: intervals that DO hold a value must still admit
    for lo, hi in [(0.0, 100.0), (1e-45, 1e-40), (0.1, 0.2), (-3.0, 3.0),
                   (smallest, smallest), (0.0, 0.0)]:
        assert _declare("float32", lo, hi) is not None, (lo, hi)


@pytest.mark.parametrize("dtype,lo,hi", [
    ("bfloat16", 1e-45, 1e-44),
    ("float8_e4m3fn", 1e-4, 1e-3),
    ("float4_e2m1fn", 0.1, 0.4),
])
def test_the_exact_test_reaches_the_narrow_dtypes_too(dtype, lo, hi):
    """Enumeration rather than nextafter for these — `float8_e8m0fnu` has no
    zero to step from, so nextafter returns NaN there and a step-based test
    would have been silently wrong for it."""
    import ml_dtypes
    name = np.dtype(getattr(ml_dtypes, dtype)).name
    with pytest.raises(ValueError, match="EMPTY under dtype"):
        _declare(name, lo, hi)


def test_the_no_zero_dtype_needs_enumeration_and_here_is_the_case():
    """`float8_e8m0fnu` is exponent-only: no zero, no negatives, NO INFINITY.

    REWRITTEN — the first version's two assertions gave the same answer under
    either mechanism (one admits via the clamp landing on a representable
    bound, the other refuses on the range branch), so it never exercised the
    enumeration it was named for. A re-audit caught it, and also corrected the
    stated reason: it is not that stepping FROM zero fails, it is that
    `np.array(inf, e8m0fnu)` is **NaN**, so the nextafter TARGET is NaN and
    every step returns NaN.
    """
    import ml_dtypes
    d = np.dtype(ml_dtypes.float8_e8m0fnu)
    with np.errstate(over="ignore", invalid="ignore"):
        assert bool(np.isnan(np.array(np.inf, d))), "no infinity in this dtype"
        assert bool(np.isnan(np.nextafter(np.array(9.5e16, d), np.array(np.inf, d))))
    # 2**57 is a value of this dtype and lies in the interval, so it IS
    # inhabited — enumeration says so, a nextafter implementation would not
    assert _declare(d.name, 9.5e16, 4.1e28) is not None, (
        "an inhabited interval must be admitted; a step-based implementation "
        "returns NaN here and would refuse it"
    )
    assert float(np.array(2.0**57, d)) == 2.0**57, "and the witness is real"
    with pytest.raises(ValueError, match="EMPTY under dtype"):
        _declare(d.name, -100.0, -50.0)     # no negative values exist


def test_a_refusal_never_names_an_infinity_as_a_neighbour():
    """`inf` is not a value of any float dtype, and advising the caller to
    declare it is advice `any_array` itself refuses. The wide-dtype path
    returned it while the narrow path returned None for the same question —
    the two mechanisms disagreeing in a message. Re-audit."""
    from stelling._jax_compat import _smallest_at_or_above, _largest_at_or_below
    assert _smallest_at_or_above("float32", 1e40) is None
    assert _largest_at_or_below("float32", -1e40) is None
    with pytest.raises(ValueError) as exc:
        _declare("float32", 1e39, 1e40)
    assert "inf" not in str(exc.value).replace("infinite point", ""), str(exc.value)


@pytest.mark.parametrize("dtype,lo,hi", [
    ("uint8", float("-inf"), -1.0),
    ("uint8", 256.0, float("inf")),
    ("int8", float("-inf"), -200.0),
    ("int64", float(2**64), float("inf")),
])
def test_an_infinite_endpoint_refuses_rather_than_crashing(dtype, lo, hi):
    """A REGRESSION THIS FILE'S OWN CHANGE INTRODUCED, caught by re-audit.

    The predicate clamps before ceil/floor — `math.floor(-inf)` raises — and a
    comment twelve lines from the call site says so. The message helper added
    to name the nearest neighbours did NOT clamp, so every integer refusal
    with an infinite endpoint escaped as an uncaught `OverflowError` instead
    of the `ValueError` the declaration layer promises.
    """
    with pytest.raises(ValueError, match="EMPTY under dtype"):
        _declare(dtype, lo, hi)


def test_a_numpy_integer_bound_gets_the_representability_reason():
    """`np.integer` is not a python `int`, and `_is_bounds_pair` accepts it —
    so a numpy-typed bound skipped the binary64 guard and got the EMPTY-set
    message, naming the wrong cause on exactly the case the guard exists for.

    The comparison also has to happen in PYTHON ints: numpy compares a float64
    against an int64 by converting the int, which discards the inexactness
    under test.
    """
    assert (float(np.int64(2**63 - 1)) != np.int64(2**63 - 1)) is np.False_ or (
        float(np.int64(2**63 - 1)) == np.int64(2**63 - 1)
    ), "numpy's own comparison cannot see the loss"
    for bound in (2**63 - 1, np.int64(2**63 - 1), np.uint64(2**64 - 1)):
        dt = "uint64" if isinstance(bound, np.unsignedinteger) else "int64"
        with pytest.raises(ValueError, match="not representable as the binary64"):
            _declare(dt, bound, bound)
    # an exactly-storable numpy bound is unaffected
    assert _declare("int64", np.int64(2**53), np.int64(2**53)) is not None


# --------------------------------------------------------------------------
# regressions from the RE-AUDIT — two of these are defects the fixes for the
# first audit introduced, in the same helper, on consecutive attempts
# --------------------------------------------------------------------------
@pytest.mark.parametrize("dtype,lo,hi,expect", [
    ("uint8", -3.0, -1.0, "0 above"),                 # nothing below: uint8 starts at 0
    ("uint8", 256.0, 300.0, "255 below"),             # nothing above
    ("int8", 200.0, 300.0, "127 below"),
    ("int8", float("-inf"), -129.0, "-128 above"),
    ("int8", 0.2, 0.8, "0 below and 1 above"),        # the only two-sided case
    ("bool", 2.0, 3.0, "1 below"),
])
def test_the_direction_words_in_a_refusal_are_true(dtype, lo, hi, expect):
    """THE SECOND DEFECT IN THIS HELPER, introduced by the fix for the first.

    Clamping both ends into `[d_lo, d_hi]` before choosing the direction word
    collapsed them onto the same value while keeping their labels, so
    `uint8 (-3, -1)` read *"0 below and 0 above"* — 0 is not below `[-3, -1]`.
    A re-audit measured the words false in a MAJORITY of refused integer boxes
    while the numbers stayed right (the original "77 of 113" population was
    not retained; a scripted census re-derives 69 of 94 —
    stelling-sweeps/verify_9b555_replacements.py), and the suite could not
    catch it because the only pinned case was one where the collapse happened
    to be correct.
    """
    with pytest.raises(ValueError) as exc:
        _declare(dtype, lo, hi)
    msg = str(exc.value)
    sides = _parse_neighbours(msg)
    # (1) EXACT side presence, not a substring. "0 above" is a substring of
    #     "0 below and 0 above", which is why the previous version could not
    #     tell the collapsed message from the correct one.
    assert _fmt_sides(sides) == expect, (
        f"expected neighbour clause {expect!r}, parsed {_fmt_sides(sides)!r} "
        f"from:\n  {msg}"
    )
    # (2) THE SEMANTIC PROPERTY, which is what "below" and "above" MEAN. This
    #     is what the collapse violated: `uint8 (-3, -1)` read "0 below" and 0
    #     is not below -3.
    if sides["below"] is not None:
        assert sides["below"] < lo, (
            f"{sides['below']} is labelled BELOW an interval starting at {lo}"
        )
    if sides["above"] is not None:
        assert sides["above"] > hi, (
            f"{sides['above']} is labelled ABOVE an interval ending at {hi}"
        )


@pytest.mark.parametrize("dtype,lo,hi", [
    ("int64", -(2**63), 2**63 - 1),      # the natural way to say "any int64"
    ("uint64", 0, 2**64 - 1),
    ("int64", 0, 2**63 - 1),
    ("int64", -(2**63), 0),
])
def test_a_widening_conversion_is_admitted(dtype, lo, hi):
    """A FALSE REJECTION the re-audit found — the failure this layer must not
    have, and one the first version of the representability guard caused.

    `float(2**63 - 1)` is `2**63`, so as an UPPER bound it widens the stored
    box: an over-approximation, and every transfer's answer over it still
    contains the executed value. Refusing it made the ordinary way to declare
    "any int64" an error. Only a NARROWING conversion is refused now.
    """
    assert _declare(dtype, lo, hi) is not None


@pytest.mark.parametrize("dtype,bound", [
    ("int64", 2**63 - 1), ("uint64", 2**64 - 1),
    ("int64", np.int64(2**63 - 1)), ("uint64", np.uint64(2**64 - 1)),
])
def test_a_narrowing_conversion_is_still_refused(dtype, bound):
    """The other direction: as a LOWER bound the same value shifts up, so the
    tool would reason about fewer values than the caller declared."""
    with pytest.raises(ValueError, match="NARROWS the declared set"):
        _declare(dtype, bound, bound)


def test_the_advice_a_refusal_gives_is_followable():
    """A refusal that names a value the check itself refuses is worse than one
    that names nothing. Where a named neighbour is not exactly storable, the
    message says which position it can be used in."""
    with pytest.raises(ValueError) as exc:
        _declare("int64", 1e19, 2e19)
    msg = str(exc.value)
    assert "9223372036854775807" in msg
    assert "not exactly representable as the binary64" in msg, msg
    assert "UPPER bound" in msg, msg
    # and the advice works: as an upper bound it is admitted
    assert _declare("int64", 0, 2**63 - 1) is not None


# --------------------------------------------------------------------------
# the storability guard is DTYPE-AWARE — the dtype-blind version refused a
# narrowing integer bound for every dtype, a false-rejection class on every
# dtype but int64/uint64
# --------------------------------------------------------------------------

# the guard's own message, pinned as a LITERAL template so int64/uint64
# refusals cannot drift by a byte while the dtype-aware gate is edited
_STORABILITY_REFUSAL = (
    "any_array bound {name}={raw!r} is not representable as the "
    "binary64 the IR stores; it would be recorded as {stored!r}, "
    "which NARROWS the declared set — the tool would reason about "
    "fewer values than you declared. (Rounding the other way is "
    "admitted: it widens the box, and an over-approximation still "
    "contains every executed value.) Integer bounds that must be "
    "exact as doubles satisfy |bound| <= 2**53"
)


@pytest.mark.parametrize("dtype,lo,hi,side", [
    ("float64", 0, 10**23, "hi"),
    ("float32", 0, 10**23, "hi"),
    ("int32", 0, 10**23, "hi"),
    # THE SAME CLASS IN THE LO POSITION (audit F2). Before these params
    # existed, a mutant that kept the dtype-blind refusal for `lo` alone —
    # `... and (name == "lo" or _narrowing_can_lose_values(dt))` — survived
    # the whole suite: every admission pin sat in the hi position.
    ("float64", -(10**23), 0, "lo"),
    ("float32", -(10**23), 0, "lo"),
    ("int32", -(10**23), 0, "lo"),
])
def test_a_narrowing_integer_bound_is_admitted_where_the_dtype_cannot_lose(
        dtype, lo, hi, side):
    """THE FALSE-REFUSAL CLASS, measured on the dtype-blind guard: the three
    hi-position declarations here were all refused, and nothing would have
    been lost.

    `float(10**23)` rounds toward zero, so the recorded endpoint genuinely
    narrows the declared interval AS REALS — but `float(v)` is the nearest
    binary64 to `v`, so no binary64 value lies strictly between them, and
    every value of these dtypes IS a binary64 value (float32/float64 by
    format, int32 because |v| < 2**53). The shaved slice holds no value of
    the dtype, the recorded box still contains every declared dtype value,
    and refusing it was the failure this layer's own docstring says it must
    not have.
    """
    cj = _declare(dtype, lo, hi)
    assert cj is not None
    params = dict(cj.eqns[0].params)
    lo_rec, hi_rec = params["lo"], params["hi"]
    # the recorded box: the exact endpoint survives exactly, the other is the
    # rounded image — and the rounding genuinely narrowed as reals (python
    # compares int/float exactly)
    if side == "hi":
        assert lo_rec == float(lo) == lo, "the exact endpoint may not move"
        assert hi_rec == float(hi) and hi_rec < hi, (
            "this case exists BECAUSE the stored hi sits below the declared "
            "integer; if these are equal the test is testing nothing"
        )
    else:
        assert hi_rec == float(hi) == hi, "the exact endpoint may not move"
        assert lo_rec == float(lo) and lo_rec > lo, (
            "this case exists BECAUSE the stored lo sits above the declared "
            "integer; if these are equal the test is testing nothing"
        )
    # ...and yet no declared dtype value fell out: nothing of the dtype lies
    # in the shaved slice between the raw endpoint and its recording.
    # Computed with numpy alone, independently of the module's own helpers.
    d = np.dtype(dtype)
    if d.kind in "iu":
        if side == "hi":
            top = int(np.iinfo(d).max)
            assert top <= hi_rec, (
                f"the largest {dtype} value {top} fell out of the recorded "
                f"box [{lo_rec}, {hi_rec}]"
            )
        else:
            bot = int(np.iinfo(d).min)
            assert bot >= lo_rec, (
                f"the smallest {dtype} value {bot} fell out of the recorded "
                f"box [{lo_rec}, {hi_rec}]"
            )
    elif side == "hi":
        c = np.array(hi_rec, d)          # nearest dtype value to the stored hi
        if float(c) <= hi_rec:
            c = np.nextafter(c, np.array(np.inf, d))
        assert float(c) > hi, (
            f"{float(c)} is a {dtype} value inside the shaved slice "
            f"({hi_rec}, {hi}] — the recorded box dropped a declared value"
        )
    else:
        c = np.array(lo_rec, d)          # nearest dtype value to the stored lo
        if float(c) >= lo_rec:
            c = np.nextafter(c, np.array(-np.inf, d))
        assert float(c) < lo, (
            f"{float(c)} is a {dtype} value inside the shaved slice "
            f"[{lo}, {lo_rec}) — the recorded box dropped a declared value"
        )


@pytest.mark.parametrize("dtype,lo,hi,name,raw", [
    ("int64", 0, 2**53 + 1, "hi", 2**53 + 1),
    ("uint64", 2**64 - 1, 2**64 - 1, "lo", 2**64 - 1),
])
def test_the_lossy_dtypes_keep_the_refusal_byte_for_byte(dtype, lo, hi, name, raw):
    """int64 and uint64 are the dtypes the guard EXISTS for — of the REAL
    dtypes jax builds arrays in, the only ones that hold values binary64
    does not (complex also keeps the refusal, by policy rather than by that
    property — no claim is made either way about complex boxes) — and the
    dtype-aware gate must not move them: decision AND message, byte for
    byte (the template above is a literal, not an import from the source)."""
    with pytest.raises(ValueError) as exc:
        _declare(dtype, lo, hi)
    assert str(exc.value) == _STORABILITY_REFUSAL.format(
        name=name, raw=raw, stored=float(raw)
    )


@pytest.mark.parametrize("dtype,lo,hi", [
    ("int64", -(2**63), 2**63 - 1),   # the natural "any int64": hi rounds UP
    ("uint64", 0, 2**64 - 1),         # hi rounds UP
    ("int32", 0, 2**53 + 3),          # hi rounds UP on a loss-free dtype
    # INEXACT WIDENING IN THE LO POSITION ON THE LOSSY DTYPES (audit F3a).
    # Before these params existed, a mutant refusing ANY inexact lo —
    # `narrows = (stored != v) if name == "lo" else (stored < v)` — survived
    # the whole suite: every widening pin's lo was exactly representable.
    ("int64", -(2**53) - 3, 0),       # lo rounds DOWN (away): widens
    ("uint64", 2**53 + 1, 2**60),     # lo rounds DOWN (toward 0): widens
    # WIDENING INT BOUNDS ON FLOAT DTYPES (audit F3b). A float-specific
    # refusing mutant — `or ((not narrows) and dt.startswith("float"))`
    # inside the inexact branch — survived before these: no widening pin
    # had a float dtype with an integer bound.
    ("float64", 0, 2**53 + 3),        # hi rounds UP: widens
    ("float32", 0, 2**53 + 3),
])
def test_the_widening_direction_stays_admitted_everywhere(dtype, lo, hi):
    """The other half of the guard's contract, restated across the gate: a
    widening conversion is an over-approximation and is admitted for every
    dtype — lossy, loss-free, float or integer, in either bound position,
    at any magnitude."""
    assert _declare(dtype, lo, hi) is not None


@pytest.mark.parametrize("dtype,lo,hi", [
    ("float64", 2**53 + 1, 2**53 + 1),   # point ON the first gap
    ("float64", 2**55 + 5, 2**55 + 7),   # interval strictly inside a gap
    ("float64", 2**63 - 1, 2**63 - 1),   # int64 max, as a FLOAT declaration
    ("float32", 2**53 + 1, 2**53 + 1),
    ("bfloat16", 2**54 + 1, 2**54 + 1),
    ("float8_e8m0fnu", 2**54 + 1, 2**54 + 1),
])
def test_an_empty_set_behind_a_narrowing_gap_edge_refuses_with_the_true_cause(
        dtype, lo, hi):
    """AUDIT F1 — the hole the dtype-aware gate UNMASKED, measured on the
    gated-but-unrepaired build: each of these ADMITTED, recording a box
    whose only inhabitant the caller had excluded (`float64
    (2**53+1, 2**53+1)` recorded `[2**53, 2**53]`; the `(2**55+5, 2**55+7)`
    interval recorded a point ABOVE its own declared hi). The parent refused
    them all — with the false NARROWS cause.

    The emptiness check receives already-rounded endpoints, so a narrowing
    endpoint that rounds ONTO a dtype value at the edge of the gap makes the
    RECORDED box inhabited while the DECLARED interval holds nothing. These
    must refuse with the true cause: the EMPTY-set class, judged against the
    raw integer endpoints in exact arithmetic — and the neighbours the
    message names must be outside the DECLARED interval, not the recorded
    one (the rounded helpers would call `2**63` "below" an interval ending
    at `2**63 - 1`).
    """
    with pytest.raises(ValueError) as exc:
        _declare(dtype, lo, hi)
    msg = str(exc.value)
    assert "EMPTY under dtype" in msg, msg
    assert "NARROWS" not in msg, (
        f"the old false cause resurfaced (nothing about this set is a "
        f"narrowing problem — it is empty):\n  {msg}"
    )
    # the caller's own numbers, not their roundings
    assert repr(lo) in msg and repr(hi) in msg, msg
    # direction words true AGAINST THE DECLARED ENDPOINTS, exactly (python
    # compares int against float without rounding)
    sides = _parse_neighbours(msg)
    if sides["below"] is not None:
        assert sides["below"] < lo, (sides, lo)
    if sides["above"] is not None:
        assert sides["above"] > hi, (sides, hi)


# Every row here must be a control on the WIDENING half specifically —
# neither endpoint narrowing, and the rounded check admitting the recorded
# box — or it passes on the parent's logic and pins nothing. The first
# version of this list got that wrong in four of fourteen rows (measured on
# 69ea5d3: `uint64 (-(2**64), -1/2)`, `float16 (2**54+1, 2**54+3)` and both
# `(-3/2**1200, -1/2**1200)` rows already refused there, the first two
# because the ROUNDED range check catches them and the last two because
# their `lo` NARROWS onto -0.0). They are replaced below, and the property
# is now asserted per row by
# `test_every_widening_empty_row_is_a_control_on_the_widening_direction`,
# which needs no parent build to check it.
_WIDENING_EMPTY = [
    # -- the reproducers ---------------------------------------------------
    # a float64 interval strictly between two consecutive float64s (ulp 4 at
    # 2**54); the recorded box [2**54, 2**54+4] is exactly the pair it sits
    # between. Measured on the narrowing-only build: admitted, and
    # `assert_(x < 0.0)` over it returned REFUTED at 100% coverage.
    ("float64", 2**54 + 1, 2**54 + 3),
    ("float64", -(2**54) - 3, -(2**54) - 1),        # the negative twin
    # its integer-dtype twin: an interval wholly BELOW int64's minimum whose
    # hi widens UP onto that minimum, the recorded box's only inhabitant.
    # Measured the same way: admitted, `assert_(x >= 0)` returned REFUTED.
    ("int64", -(2**64), -(2**63) - 1),
    # -- the same shape at every float width -------------------------------
    ("float32", 2**53 + 1, 2**53 + 3),
    ("bfloat16", 2**53 + 1, 2**53 + 3),             # the flipped parity pin
    ("float8_e8m0fnu", 2**53 + 1, 2**53 + 3),
    # -- ABOVE THE DTYPE'S LARGEST FINITE VALUE, by one -------------------
    # the widening lo rounds back DOWN onto float32's max, so the recorded
    # box is inhabited while the declared interval holds no finite float32
    # (and `inf` is not a member of it as a real either)
    ("float32", int(np.finfo(np.float32).max) + 1, float("inf")),
    ("float32", float("-inf"), -int(np.finfo(np.float32).max) - 1),
    # -- SUBNORMAL: strictly inside the gap between 0 and 2**-1074 --------
    # lo = 1/4 ulp rounds DOWN to +0.0, hi = 3/4 ulp rounds UP to the
    # smallest subnormal, so the recorded box is [0.0, 5e-324] — two float64
    # values, both excluded by the declaration. Both endpoints widen.
    ("float64", Fraction(1, 2**1076), Fraction(3, 2**1076)),
    ("float8_e4m3", Fraction(1, 2**1076), Fraction(3, 2**1076)),
    # -- UNDERFLOW TO -0.0, with the lo recorded EXACTLY ------------------
    # the hi is too small to record, rounds UP to -0.0, and the recorded box
    # therefore touches zero while the declared interval is strictly
    # negative — so an unsigned dtype's smallest value, or a float dtype's
    # zero, enters a box that excluded it. The class was found by a
    # randomized sweep rather than by hand; the ROWS are hand-built, because
    # the sweep's own instances had both endpoints underflowing, which makes
    # the `lo` narrow and hands the case to the parent's half of the check.
    ("uint64", -1.0, Fraction(-1, 2**1200)),
    ("uint32", -1.0, Fraction(-1, 2**1200)),
    ("bool", -1.0, Fraction(-1, 2**1200)),
    ("float16", -1e-300, Fraction(-1, 2**1200)),
    # -- and a FRACTIONAL spelling of the float64 reproducer ---------------
    ("float64", Fraction(2**55 + 1, 2), Fraction(2**55 + 7, 2)),
]


@pytest.mark.parametrize("dtype,lo,hi", _WIDENING_EMPTY)
def test_an_empty_set_behind_a_widening_gap_edge_refuses_too(dtype, lo, hi):
    """The other half of the gap-edge class, and the last known way a
    definite verdict could be minted over a set no execution can inhabit.

    A NARROWING endpoint rounds onto the gap's far edge; a WIDENING one
    rounds outward onto its near edge — and both leave the RECORDED box
    inhabited while the DECLARED interval holds nothing. The narrowing half
    landed first, with the widening half pinned as a known blind spot on
    the argument that the parent admitted it too ("parity, not
    endorsement"). Measured on 69ea5d3: every row of this list was ADMITTED
    there, and the two reproducers reached REFUTED at 100% coverage — a
    claim that some input violates the property, over a declaration with no
    inputs at all. Parity with an older build does not make that sound, so
    the re-check now fires on ANY inexact recording rather than on the
    narrowing direction alone.

    ("every row" is a claim about THIS list, and the first version of it was
    false for four of fourteen rows — see the note above the list. The rows
    were replaced, and
    `test_every_widening_empty_row_is_a_control_on_the_widening_direction`
    now checks the property in-tree so the claim cannot rot again.)

    Same cause and same message shape as the narrowing half: the EMPTY-set
    class, judged against the RAW endpoints in exact arithmetic, with the
    neighbours named outside the DECLARED interval.
    """
    with pytest.raises(ValueError) as exc:
        _declare(dtype, lo, hi)
    msg = str(exc.value)
    assert "EMPTY under dtype" in msg, msg
    assert "NARROWS" not in msg, (
        f"nothing here narrows — the recorded box is a SUPERSET of the "
        f"declared interval, and it is the declaration that is empty:\n  {msg}"
    )
    assert repr(lo) in msg and repr(hi) in msg, msg
    sides = _parse_neighbours(msg)
    if sides["below"] is not None:
        assert sides["below"] < lo, (sides, lo)
    if sides["above"] is not None:
        assert sides["above"] > hi, (sides, hi)


@pytest.mark.parametrize("dtype,lo,hi", _WIDENING_EMPTY)
def test_every_widening_empty_row_is_a_control_on_the_widening_direction(
        dtype, lo, hi):
    """A row that the ROUNDED check already refuses, or that has a NARROWING
    endpoint, is refused by the parent's logic too and pins nothing about
    this change. Four of the first fourteen rows in `_WIDENING_EMPTY` were
    exactly that, and the list's own docstring claimed the opposite: an
    inert row is invisible, because it passes.

    So the property is asserted here per row, IN-TREE, needing no parent
    build to check: (1) neither endpoint narrows, and (2) the rounded
    emptiness check ADMITS the recorded box. Together those are the
    definition of "only the widening half of the exact re-check can refuse
    this" — the rounded check has already had its say and said yes, and no
    narrowing endpoint exists for the parent's half to catch.
    """
    dt = str(np.dtype(dtype))
    lo_x, hi_x = declared_bound_value(lo), declared_bound_value(hi)
    rec = (binary64_image(lo_x), binary64_image(hi_x))
    for name, x, stored in (("lo", lo_x, rec[0]), ("hi", hi_x, rec[1])):
        if isinstance(x, Fraction) and stored != x:
            narrows = (stored > x) if name == "lo" else (stored < x)
            assert not narrows, (
                f"{dt} ({lo!r}, {hi!r}): the {name} NARROWS ({x} recorded as "
                f"{stored!r}), so the parent's half of the re-check already "
                f"fires and this row is not a widening control"
            )
    holds, why = JC._dtype_holds_a_value_in(dt, *rec)
    assert holds, (
        f"{dt} ({lo!r}, {hi!r}): the ROUNDED check already refuses the "
        f"recorded box {rec} — {why} — so this row is refused on the "
        f"parent's logic and pins nothing about the widening direction"
    )


# -- an emptiness oracle written from the FORMAT PARAMETERS -------------------
#
# The instrument this replaced claimed to be "computed from numpy alone,
# independently of every helper in `_jax_compat`". It imported nothing from
# the tree, and that is not the same thing: its float branch was
# `_exact_at_or_above`'s wide path with the same primitive, the same
# starting point, and the same load-bearing assumption — step with
# `np.nextafter` from a binary64 cast, decide by comparing `float(c)` —
# which is exactly the assumption the source comment says fails for x86
# longdouble. Measured: on `float128 (2**54+1, 2**54+3)` that walk stops
# after 1025 steps and reports the least value >= 2**54+1 as 2**54+4, so it
# certifies EMPTY an interval with three inhabitants (longdouble holds
# 2**54+1 exactly: `longdouble(2**54+1) - longdouble(2**54) == 1.0`). It
# would have rubber-stamped the one false refusal the binary64-subset screen
# exists to prevent. Three smaller defects came with it: `above = None` on
# an overflowing cast is right above the format max and WRONG below the
# format minimum (it certified `float16 [-10**9, 0]` as empty); the
# assertion passed vacuously whenever `above` was None; and its integer
# branch keyed on `d.kind in "iub"`, while int2/int4/uint2/uint4 are kind
# 'V' and silently took the float branch.
#
# What is below shares no primitive with the code under test. The IEEE
# formats are computed from their PUBLISHED PARAMETERS, written down here,
# in exact rational arithmetic with no `float()` anywhere in the decision
# path; the narrow formats are enumerated from their bit patterns, with the
# conversion's exactness measured per value rather than assumed; integer
# ranges are written down and keyed by NAME, not by numpy's `kind`; and a
# dtype the table does not describe raises instead of answering, so the
# oracle can never rubber-stamp by falling through.
_IEEE_FORMAT = {                 # (precision, emin, emax) — IEEE 754 binary32
    "float32": (24, -126, 127),  # and binary64, and the x87 double-extended
    "float64": (53, -1022, 1023),   # format longdouble uses on x86
    "float128": (64, -16382, 16383),
}
_INT_DOMAIN = {
    "bool": (0, 1),
    **{f"int{n}": (-(2 ** (n - 1)), 2 ** (n - 1) - 1)
       for n in (2, 4, 8, 16, 32, 64)},
    **{f"uint{n}": (0, 2**n - 1) for n in (2, 4, 8, 16, 32, 64)},
}


def _binade(x):
    """The integer ``e`` with ``2**e <= x < 2**(e+1)``, for an exact
    rational ``x > 0``. Seeded from the bit lengths and corrected, so it is
    exact for magnitudes no float can hold."""
    e = x.numerator.bit_length() - x.denominator.bit_length()
    while Fraction(2) ** e > x:
        e -= 1
    while Fraction(2) ** (e + 1) <= x:
        e += 1
    return e


def _least_ieee_at_or_above(p, emin, emax, x):
    """The least value of the binary format ``(p, emin, emax)`` that is
    ``>= x``, as an exact Fraction, or None if the format has none. All
    rational arithmetic: ``//`` on Fractions is an exact floor."""
    tiny = Fraction(1, 2) ** (p - 1 - emin)          # smallest subnormal
    big = (Fraction(2) - Fraction(1, 2) ** (p - 1)) * Fraction(2) ** emax
    if x > big:
        return None
    if x <= -big:
        return -big
    if x == 0:
        return Fraction(0)
    if x < 0:                     # round the MAGNITUDE down, toward zero
        y = -x
        if y < tiny:
            return Fraction(0)
        u = max(tiny, Fraction(1, 2) ** (p - 1 - _binade(y)))
        return -((y // u) * u)
    if x <= tiny:
        return tiny
    u = max(tiny, Fraction(1, 2) ** (p - 1 - _binade(x)))
    v = -((-x) // u) * u          # ceil to a multiple of the spacing
    return None if v > big else v


@functools.lru_cache(maxsize=None)
def _enumerated_values(dtype):
    """Every finite value of a dtype of at most two bytes, as exact
    Fractions, from its bit patterns. ``float()`` appears in the
    CONSTRUCTION but not in the decision, and its exactness is measured
    rather than assumed: each value is cast back into the dtype and its
    VALUE required to survive, which establishes that the Fraction IS the
    dtype's value and not a rounding of it.

    The value, not the bit pattern. A sub-byte format has padding bits, so
    several patterns denote one value: bitwise equality fails for 93 of
    float6_e2m3fn's 256 patterns and 105 of float4_e2m1fn's, while the value
    round-trips for every pattern of every narrow format this jax has."""
    d = np.dtype(dtype)
    assert d.itemsize <= 2, dtype
    raw = np.arange(2 ** (8 * d.itemsize),
                    dtype=np.uint8 if d.itemsize == 1 else np.uint16)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(invalid="ignore"):
            vals = raw.view(d)
    out = set()
    for v in vals:
        f = float(v)
        if not math.isfinite(f):
            continue
        assert float(np.array(f, d)) == f, (
            f"{dtype}: float({v!r}) = {f!r} does not cast back — this "
            f"oracle's construction assumes binary64 holds every value of a "
            f"<=2-byte float format, and this dtype breaks it"
        )
        out.add(Fraction(f))
    return sorted(out)


def _oracle(dtype, lo, hi):
    """Does the closed real interval ``[lo, hi]`` hold a value of ``dtype``?
    Returns ``(answer, witness_or_None)``, the witness exact."""
    dt = str(np.dtype(dtype))
    if lo == math.inf or hi == -math.inf or (
            lo != -math.inf and hi != math.inf and lo > hi):
        return False, None
    if dt == "bool" or dt.startswith(("int", "uint")):
        a, b = _INT_DOMAIN[dt]
        first = a if (lo == -math.inf or lo <= a) else math.ceil(lo)
        if first > b or (hi != math.inf and first > hi):
            return False, None
        return True, first
    if dt in _IEEE_FORMAT:
        p, emin, emax = _IEEE_FORMAT[dt]
        big = (Fraction(2) - Fraction(1, 2) ** (p - 1)) * Fraction(2) ** emax
        v = _least_ieee_at_or_above(
            p, emin, emax, -big if lo == -math.inf else Fraction(lo))
        if v is None or (hi != math.inf and v > hi):
            return False, None
        return True, v
    if np.dtype(dt).itemsize <= 2:
        vals = _enumerated_values(dt)
        first = next((v for v in vals if lo == -math.inf or v >= lo), None)
        if first is None or (hi != math.inf and first > hi):
            return False, None
        return True, first
    raise AssertionError(
        f"the oracle has no written-down description of {dt!r} and will not "
        f"guess — add its parameters or leave the dtype out of the params"
    )


_ORACLE_ROWS = (
    [(dt, lo, hi, "empty") for dt, lo, hi in _WIDENING_EMPTY]
    + [("float64", 2**54 + 1, 2**54 + 4, "inhabited"),
       ("float64", 2**54, 2**54 + 3, "inhabited"),
       ("int64", -(2**64), -(2**63), "inhabited"),
       ("uint64", 0, 2**64 - 1, "inhabited"),
       ("bfloat16", 2**53, 2**53 + 3, "inhabited"),
       ("float64", Fraction(1, 2**1076), Fraction(7, 2**1076), "inhabited"),
       ("uint32", -1.0, Fraction(1, 2**1200), "inhabited"),
       ("float32", int(np.finfo(np.float32).max), float("inf"), "inhabited"),
       # THE ROW THE OLD ORACLE GOT WRONG. x86 longdouble holds 2**54+1
       # exactly; the nextafter-and-float() walk reported the least value at
       # or above it as 2**54+4 and certified this EMPTY. Both the tool and
       # the oracle must say INHABITED.
       (str(np.dtype(np.longdouble)), 2**54 + 1, 2**54 + 3, "inhabited")]
)


@pytest.mark.parametrize("dtype,lo,hi,expected", _ORACLE_ROWS)
def test_the_emptiness_decision_agrees_with_the_format_parameter_oracle(
        dtype, lo, hi, expected):
    """THE ORACLE IS A CONTROL ON BOTH SIDES, which is what the instrument it
    replaced was not.

    That one only ever asserted "this parameter row is empty". It never
    called `any_array`, so it was structurally incapable of failing on any
    source change — its own docstring named a mutant it claimed to kill and
    did not (`return False, ""` at the top of the exact check is killed by
    `test_the_exact_recheck_admits_at_its_boundaries`, not by it).

    This one asserts the TOOL's decision equals the ORACLE's answer, over
    rows in both directions. A wrong parameter row, a broken oracle, and an
    implementation that refuses or admits too much all fail it. The
    inhabited rows carry an exact witness, checked to lie in the declared
    interval AND in the recorded box, so an admit cannot pass vacuously.
    """
    holds, witness = _oracle(dtype, lo, hi)
    assert holds == (expected == "inhabited"), (
        f"the ORACLE says {dtype} [{lo}, {hi}] "
        f"{'holds ' + str(witness) if holds else 'is empty'}, the param says "
        f"{expected} — one of them is wrong, and the oracle is the "
        f"instrument"
    )
    if holds:
        assert lo <= witness <= hi, (witness, lo, hi)
        cj = _declare(dtype, lo, hi)          # must be ADMITTED
        params = dict(cj.eqns[0].params)
        assert params["lo"] <= witness <= params["hi"], (
            f"recorded box [{params['lo']}, {params['hi']}] dropped the "
            f"witness {witness}"
        )
    else:
        with pytest.raises(ValueError) as exc:
            _declare(dtype, lo, hi)
        assert "EMPTY under dtype" in str(exc.value), str(exc.value)


@pytest.mark.parametrize("dtype,lo,hi,witness", [
    # RE-AUDIT RA2a: the declared set's ONLY inhabitant sits exactly AT the
    # declared hi — the admit boundary of the exact re-check. A mutant
    # shrinking `first <= min(hi, f_hi)` to `<` survived the whole suite
    # before this param existed and flips exactly this case.
    ("float64", 2**53 + 3, 2**53 + 4, 2.0**53 + 4),
    # its mirror: the only inhabitant exactly AT the declared lo, with the
    # hi doing the narrowing
    ("float64", 2**53 + 4, 2**53 + 5, 2.0**53 + 4),
    # RE-AUDIT RA1: unbounded-below with a narrowing int hi. The exact
    # check must clamp `lo` into the dtype's range exactly as its rounded
    # twin does — unclamped, the wide-dtype path asked for the smallest
    # value >= -inf, filtered the non-finite answer, and REFUSED an
    # interval containing every float64 through 2**53 (measured, EMPTY
    # cause, positive and negative hi).
    ("float64", float("-inf"), 2**53 + 1, 2.0**53),
    ("float32", float("-inf"), 2**53 + 1, 2.0**53),
    ("float64", float("-inf"), -(2**53) - 3, -(2.0**53) - 4),
    # the symmetric mirror: narrowing int lo, unbounded above
    ("float64", 2**53 + 3, float("inf"), 2.0**53 + 4),
    ("float32", 2**53 + 3, float("inf"), 2.0**53 + 2.0**30),
    # and the enumeration branch's -inf handling, which was already correct
    # (a finite-table bisection has no infinity to choke on), pinned so it
    # stays that way
    ("bfloat16", float("-inf"), 2**54 + 1, 2.0**54),
    # -- THE WIDENING DIRECTION'S OWN ADMIT BOUNDARY ----------------------
    # Each of these is ONE STEP from a `_WIDENING_EMPTY` case above and must
    # stay admitted: the sloppy version of that refusal — "an inexactly
    # recorded endpoint at this magnitude means the dtype cannot hold the
    # interval" — rejects every one of them, and rejecting a program a user
    # legitimately wrote is worse than the hole being closed.
    ("float64", 2**54 + 1, 2**54 + 4, 2.0**54 + 4),   # inhabitant AT the hi
    ("float64", 2**54, 2**54 + 3, 2.0**54),           # inhabitant AT the lo
    ("bfloat16", 2**53, 2**53 + 3, 2.0**53),
    ("float8_e8m0fnu", 2**53, 2**53 + 3, 2.0**53),
    # int64's minimum, sitting exactly at the declared hi, with a WIDENING
    # lo below it (the reproducer's hi moved up by one integer)
    ("int64", -(2**64) - 3072, -(2**63), -(2.0**63)),
    ("uint64", 2**53 + 1, 2**60, 2.0**53 + 2),        # widening lo, uint64
    # the float32-max refusal's paired control: the SAME bounds on a dtype
    # that does hold values up there
    ("float64", int(np.finfo(np.float32).max) + 1, float("inf"), 2.0**128),
    # the subnormal refusal's paired control: [1/4 ulp, 7/4 ulp] straddles
    # the smallest subnormal, so it IS inhabited — by exactly one value
    ("float64", Fraction(1, 2**1076), Fraction(7, 2**1076), 2.0**-1074),
    # the underflow-to-zero refusal's paired control: the same tiny
    # interval, straddling zero instead of sitting below it
    ("uint32", Fraction(-3, 2**1200), Fraction(1, 2**1200), 0.0),
])
def test_the_exact_recheck_admits_at_its_boundaries(dtype, lo, hi, witness):
    """The admit side of the gap-edge re-check, at its edges: an inhabitant
    exactly on a declared endpoint, an infinite endpoint (which means
    "unbounded" and is absorbed by the dtype's own range, exactly as in the
    rounded check), and — since the re-check covers the widening direction —
    the one-step neighbours of every widening refusal. Each case names its
    witness — a value of the dtype inside the DECLARED interval, verified
    exactly — so a refusal here is a proven false refusal, the failure this
    layer must not have."""
    d = np.dtype(dtype)
    assert float(np.array(witness, d)) == witness, "witness must be a dtype value"
    assert lo <= witness <= hi, "witness must sit in the declared interval"
    cj = _declare(dtype, lo, hi)
    assert cj is not None
    params = dict(cj.eqns[0].params)
    # ...and the recorded box, which may round endpoints, still holds it
    assert params["lo"] <= witness <= params["hi"], (
        f"recorded box [{params['lo']}, {params['hi']}] dropped the witness "
        f"{witness}"
    )


@pytest.mark.parametrize("lo,hi", [
    (int(np.finfo(np.float32).max) + 1, float("inf")),
    (int(np.finfo(np.float32).max) + 1, 10**40),
    (float("-inf"), -int(np.finfo(np.float32).max) - 1),
])
def test_beyond_the_format_range_says_so_rather_than_naming_a_gap(lo, hi):
    """The exact check's out-of-finite-range branch decides no DECISION —
    delete it and the walk below still answers "empty" — so it survived
    every mutation the suite could throw at it. What it decides is the
    CAUSE, and with it gone the cause is false: these intervals sit BEYOND
    float32's largest finite value, and the fallthrough calls that "a gap
    between representable values", which is a different and wrong
    explanation of a refusal the caller has to act on.

    Reachable only because rounding pulls the endpoint back INTO range —
    `f32max + 1` records as `f32max` — which is exactly the widening
    direction this branch's re-check now covers, so it is newly worth a
    control rather than newly written.
    """
    with pytest.raises(ValueError) as exc:
        _declare("float32", lo, hi)
    msg = str(exc.value)
    assert "lies entirely outside them" in msg, msg
    assert "gap between representable values" not in msg, (
        f"the interval is beyond float32's finite range, not inside a gap "
        f"in it:\n  {msg}"
    )


def test_the_subnormal_clause_of_the_loss_gate_is_stated_not_exercised():
    """`_narrowing_can_lose_values`'s third clause — that the format's
    smallest subnormal does not underflow binary64 — is DECIDED BY NO DTYPE.
    Measured across every dtype this jax builds arrays in plus x86
    longdouble: no format passes the first two clauses and fails this one,
    so deleting it leaves the whole suite green and no real declaration
    moves.

    That is a coverage gap, not a behaviour gap, and it cannot have a
    control made of real dtypes. So the control is a STUB FORMAT pushed
    through the same lookup: a 52-bit significand inside binary64's
    exponent range whose subnormals fall below binary64's, which is exactly
    the shape the clause exists for. Recorded rather than deleted, because
    the gate is derived from format properties rather than from a name list
    and dropping a property because today's dtypes do not exercise it is
    how a derivation stops being one.
    """
    assert JC._narrowing_can_lose_values("float64") is False    # sanity

    class _Stub:                        # -1023 - 52 = -1075, below -1074
        nmant, maxexp, minexp = 52, 1024, -1023

    def _no(_):
        raise ValueError("stub: not an integer dtype")

    finfo, iinfo = JC.jnp.finfo, JC.jnp.iinfo
    JC._narrowing_can_lose_values.cache_clear()
    try:
        JC.jnp.finfo, JC.jnp.iinfo = (lambda d: _Stub()), _no
        assert JC._narrowing_can_lose_values("float64") is True, (
            "a format whose subnormals underflow binary64 holds values "
            "binary64 does not, so a narrowing bound on it can drop one"
        )
    finally:
        JC.jnp.finfo, JC.jnp.iinfo = finfo, iinfo
        JC._narrowing_can_lose_values.cache_clear()
    assert JC._narrowing_can_lose_values("float64") is False    # restored


def test_zero_extent_keeps_the_storability_decision_of_its_dtype():
    """The storability loop runs BEFORE the zero-size early return, so for the
    lossy dtypes a narrowing bound refuses at shape (0,) exactly as at (1,) —
    pinned unchanged from the dtype-blind guard, message and all.

    For the newly-admitted dtypes the CHOICE, documented: the gate is
    dtype-level and shape-independent, so shape (0,) admits like every other
    shape. A zero-size declaration is vacuously inhabited whatever its bounds
    (see test_a_zero_size_declaration_is_never_refused), and the storability
    question — does the recorded box hold the declared dtype-set — does not
    depend on the extent, so no shape-special case was added.

    The gap-edge refusal (audit F1) joins the EMPTY-SET class, and that
    class exempts zero-size shapes by the same doctrine — so at (0,) the
    gap-point declaration admits. The parent refused it there only through
    the dtype-blind NARROWS cause this change retires; the empty-set cause
    cannot apply to a shape every bounds pair inhabits vacuously.
    """
    with pytest.raises(ValueError) as exc:
        jax.make_jaxpr(lambda: (any_array((0,), "int64", (0, 2**53 + 1)),))()
    assert str(exc.value) == _STORABILITY_REFUSAL.format(
        name="hi", raw=2**53 + 1, stored=float(2**53 + 1)
    )
    assert jax.make_jaxpr(
        lambda: (any_array((0,), "int32", (0, 10**23)),))() is not None
    assert jax.make_jaxpr(
        lambda: (any_array((0,), "float64", (2**53 + 1, 2**53 + 1)),)
    )() is not None


def test_the_loss_condition_is_derived_from_dtype_properties():
    """The gate's truth table, family by family — derived from the same
    iinfo-then-finfo lookup as the empty-set check, not from a name list.
    Of the REAL dtypes jax builds arrays in, only int64 and uint64 have
    value sets leaving binary64; the other real dtypes hold only binary64
    values and cannot lose one to the rounding. Complex answers "can lose"
    BY POLICY, not by that property — its components are binary64 subsets,
    but what a real box means for a complex array is the open item, so the
    gate makes no claim either way and preserves the refusal."""
    can_lose = JC._narrowing_can_lose_values
    assert can_lose("int64") and can_lose("uint64")
    for dt in ("bool", "int8", "int16", "int32", "uint8", "uint16", "uint32",
               "float16", "bfloat16", "float32", "float64"):
        assert not can_lose(dt), dt
    # the ml_dtypes families, through jnp's own lookups: int2/uint2 are absent
    # from the bounds registry (jnp.iinfo covers them), float8_e8m0fnu is the
    # exponent-only type whose 255 finite values are all powers of two
    import ml_dtypes
    for name in ("int2", "uint2", "int4", "uint4",
                 "float8_e4m3fn", "float8_e5m2", "float8_e8m0fnu",
                 "float6_e2m3fn", "float4_e2m1fn"):
        assert not can_lose(str(np.dtype(getattr(ml_dtypes, name)))), name
    # complex keeps the refusal BY POLICY, not by property: what a real-bounded
    # box means for a complex array is an open item, and jnp.finfo(complex128)
    # answers for the float64 COMPONENT — which is why the policy must
    # short-circuit before the lookup rather than fall through to it
    assert can_lose("complex64") and can_lose("complex128")
    # x86 longdouble holds 64-bit significands binary64 does not. Not a dtype
    # jax builds arrays in — the point is that the answer comes from the
    # format's own properties (nmant 63 > 52), not from membership in a list
    if np.dtype(np.longdouble).itemsize > 8:
        assert can_lose(str(np.dtype(np.longdouble)))


# --------------------------------------------------------------------------
# the silent-⊤ rows: a decline that is COUNTED but carries no reason
# --------------------------------------------------------------------------
def test_the_convert_decline_names_the_SOURCE_dtype():
    """`convert_element_type` returning None produced the note
    "no sound rule for params {'new_dtype': 'float64', ...}" — which prints the
    DESTINATION and never the SOURCE, and the source is the load-bearing half.

    Measured: `int64 -> float64` is a terminal in independently-authored
    external code, where a python int literal promotes through it, and a reader
    of the generic note cannot tell which side of the cast is the problem.

    This fails if the reason regresses to the generic form, if it names the
    wrong dtypes, or if the coverage accounting moves — a raised decline and a
    returned None must be counted identically.
    """
    def h():
        x = any_array((3,), "int64", (0, 5))
        return assert_(jnp.sum(x.astype(jnp.float64)) <= 1e30)

    p = P.propagate(trace(h))
    assert p.coverage.unknown == 1
    assert p.coverage.unknown_primitives == (("convert_element_type", 1),)
    note = "".join(n for n in p.notes if "convert_element_type" in n)
    assert "'int64' -> 'float64'" in note, note
    assert "SOURCE" in note, note
    assert "no sound rule" not in note, "the generic note must be gone"


def test_the_float_to_int_decline_prints_the_offending_interval():
    """The other convert path: the truncation rule is modelled, but only while
    the operand fits the target. The refusal must print the operand's span and
    the target's range — name it, explain it, print the numbers."""
    def h():
        x = any_array((2,), "float64", (0.0, 4e9))     # past int32's max
        return assert_(jnp.sum(jnp.asarray(x.astype(jnp.int32), jnp.float64)) <= 1e30)

    p = P.propagate(trace(h))
    note = "".join(n for n in p.notes if "convert_element_type" in n)
    assert "truncates toward zero" in note, note
    assert "4000000000" in note.replace(".0", ""), note
    assert "2147483647" in note, note

# --------------------------------------------------------------------------
# `_finite_values` had ZERO test references. Two green mutations were measured
# against it: dropping negatives (which creates false rejections) and keeping
# infinities (which reproduces the "inf above" defect a test is named for,
# whose own assertions cover float32 only — a WIDE dtype, served by the other
# mechanism entirely).
# --------------------------------------------------------------------------
NARROW = ["float16", "bfloat16", "float8_e4m3fn", "float8_e5m2",
          "float8_e8m0fnu", "float4_e2m1fn", "float6_e3m2fn"]


def _narrow_name(dtype):
    import ml_dtypes
    t = getattr(ml_dtypes, dtype, None)
    return np.dtype(t).name if t is not None else dtype


@pytest.mark.parametrize("dtype", NARROW)
def test_finite_values_is_finite_sorted_unique_and_spans_the_dtype(dtype):
    """The structural contract, which nothing checked."""
    from stelling._jax_compat import _finite_values
    name = _narrow_name(dtype)
    vals = _finite_values(name)
    assert vals.size > 0
    assert np.all(np.isfinite(vals)), "a non-finite value is not a VALUE of the dtype"
    assert np.all(np.diff(vals) > 0), "must be strictly sorted and unique"
    info = jnp.finfo(np.dtype(name))
    assert float(vals[0]) == float(info.min), f"{name}: min"
    assert float(vals[-1]) == float(info.max), f"{name}: max"


@pytest.mark.parametrize("dtype", [d for d in NARROW if d != "float8_e8m0fnu"])
def test_finite_values_keeps_the_negatives(dtype):
    """Dropping them was measured green, and it turns ordinary negative
    envelopes into refusals — the false-rejection failure this layer must not
    have. `float8_e8m0fnu` is excluded because it genuinely has none."""
    from stelling._jax_compat import _finite_values
    name = _narrow_name(dtype)
    vals = _finite_values(name)
    assert (vals < 0).any(), f"{name} has negative values and they must be present"
    # and the behavioural consequence the mutation produced
    lo = float(vals[0]) / 2.0
    hi = float(vals[vals < 0][-1])
    assert _declare(name, lo, hi) is not None, (
        f"{name} ({lo}, {hi}) is inhabited and must be admitted"
    )


@pytest.mark.parametrize("dtype", NARROW)
def test_a_narrow_dtype_refusal_never_names_an_infinity(dtype):
    """The companion test for this property asserted it on `float32` only — a
    WIDE dtype, which never touches `_finite_values`. So the narrow mechanism
    was unpinned, and keeping infinities in the value set was measured green."""
    from stelling._jax_compat import _smallest_at_or_above, _largest_at_or_below
    name = _narrow_name(dtype)
    top = float(jnp.finfo(np.dtype(name)).max)
    assert _smallest_at_or_above(name, top * 4) is None
    assert _largest_at_or_below(name, -abs(top) * 4) is None
    with pytest.raises(ValueError) as exc:
        _declare(name, top * 4, top * 8)
    body = str(exc.value).replace("infinite point", "")
    assert "inf" not in body, body
