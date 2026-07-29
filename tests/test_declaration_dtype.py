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

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
import numpy as np
from jax import lax

from stelling import interval as iv
from stelling import propagate as P
from stelling import _jax_compat as JC
from stelling.harness import any_array, assert_, trace


@pytest.fixture(autouse=True)
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def _declare(dtype, lo, hi):
    return jax.make_jaxpr(lambda: (any_array((1,), dtype, (lo, hi)),))()


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
    below = float(np.nextafter(np.float32(0.1), np.float32(0)))
    above = float(np.float32(0.1))
    assert repr(below) in msg and repr(above) in msg, msg
    assert "nearest float32 values" in msg
    # the integer branch names its neighbours too
    with pytest.raises(ValueError) as exc2:
        _declare("int8", 0.2, 0.8)
    assert "0 below and 1 above" in str(exc2.value), str(exc2.value)


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
    A re-audit measured the words false in 77 of 113 cases while the numbers
    stayed right, and the suite could not catch it because the only pinned
    case was one of the 36 that happened to be correct.
    """
    with pytest.raises(ValueError) as exc:
        _declare(dtype, lo, hi)
    msg = str(exc.value)
    assert expect in msg, msg
    if " below and " in msg:
        b = msg.split("integers are ")[1].split(" below")[0]
        a = msg.split(" and ")[1].split(" above")[0]
        assert b != a, f"a side collapsed onto the other: {msg}"


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
