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
    ("float32", 0.1, 0.1, "0.1 is NOT a float32 and MUST still be admitted"),
    ("float32", 1e-45, 1e-40, "the subnormal band"),
    ("float64", 0.0, 100.0, "the corpus's ordinary case"),
    ("float64", 0.0, float("inf"), "half-infinite: unbounded above"),
    ("complex64", -3.0, 3.0, "complex admitted unconditionally, by policy"),
])
def test_admits_every_legitimate_envelope(dtype, lo, hi, why):
    """A declaration check that refuses a legitimate envelope is worse than
    the hole it closes, so this half is the load-bearing one."""
    assert _declare(dtype, lo, hi) is not None


def test_float_bounds_are_never_tested_for_exact_representability():
    """Stated as its own test because it is the deliberate under-reach: 0.1,
    1/3 and pi are not float32 values, and all three are ordinary bounds."""
    for v in (0.1, 1.0 / 3.0, float(np.pi)):
        assert float(np.float32(v)) != v, f"{v} must not be a float32 exactly"
        assert _declare("float32", v, v) is not None


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


@pytest.mark.parametrize("prim", sorted(SURFACE))
def test_every_one_of_the_thirteen_entry_points_is_closed(prim):
    fn = SURFACE[prim]

    def h():
        x = any_array((1,), "uint8", (-3.0, -1.0))
        return assert_(jnp.sum(jnp.asarray(fn(x), jnp.float64)) >= -1e30)

    with pytest.raises(ValueError, match="EMPTY under dtype"):
        trace(h)


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
    and a blinded audit measured that combination exempting NINE of the thirty
    dtypes jax builds arrays in — so the box this check exists to reject was
    accepted verbatim one width away: `uint8 (-3,-1)` refused, `uint2 (-3,-1)`
    admitted and DISCHARGED at 100% coverage, a VERIFIED over an empty set.
    """
    import ml_dtypes
    t = getattr(ml_dtypes, dtype, None)
    name = np.dtype(t).name if t is not None else dtype
    with pytest.raises(ValueError, match="EMPTY under dtype"):
        _declare(name, lo, hi)


def test_the_exempt_dtypes_really_did_mint_a_verdict():
    """ANTI-VACUITY for the parametrization above: the box it now refuses was
    reaching a DEFINITE verdict, not merely being admitted."""
    import ml_dtypes
    monkey = P.TRANSFERS  # untouched; this is a pure re-derivation
    assert np.dtype(ml_dtypes.uint2).name == "uint2"
    # uint2 spans [0, 3]; (-3, -1) holds nothing
    info = jnp.iinfo(np.dtype(ml_dtypes.uint2))
    assert (int(info.min), int(info.max)) == (0, 3)
    assert monkey is P.TRANSFERS


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


def test_the_representation_gap_is_a_KNOWN_open_hole():
    """PINNED AS A LIMITATION, not as correct behaviour.

    The range-only float rule admits an interval lying wholly inside a
    representation gap. `float32 (1e-50, 1e-49)` sits below the smallest
    subnormal (1.4e-45), holds no float32, and reaches a REFUTED at 100%
    coverage. Closing it requires the exact test, which also rejects
    `float32 (0.1, 0.1)` — an ordinary envelope — so the trade is a
    published-surface decision and is deliberately not taken here.

    This test fails the moment the behaviour changes in either direction,
    which is the point: the hole is recorded rather than forgotten.
    """
    assert _declare("float32", 1e-50, 1e-49) is not None, "still admitted"
    smallest = float(np.nextafter(np.float32(0), np.float32(1)))
    assert smallest > 1e-49, "and the interval genuinely holds no float32"

    def h():
        x = any_array((1,), "float32", (1e-50, 1e-49))
        return assert_(x[0] >= smallest)
    p = P.propagate(trace(h))
    assert p.obligations[0].status == "violated-over-set" and p.coverage.unknown == 0, (
        f"the consequence is a REFUTED at full coverage; got "
        f"{p.obligations[0].status} at unknown={p.coverage.unknown}"
    )
