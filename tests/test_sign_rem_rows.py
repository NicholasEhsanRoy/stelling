# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The `sign` and `rem` rows, and the two facts that shape them.

`sign`: the obvious rule ``lo > 0 -> [1, 1]`` is UNSOUND on the measured
target, because XLA flushes subnormals. `rem`: `lax.rem` is truncated while
`jnp.mod` is floored, and only the truncated form is a primitive.

Every soundness assertion here carries an anti-vacuity control: remove the
guard under test and the false box must come back. Without that these pass
for reasons unrelated to what they claim (docs/norms.md, "A measurement whose result is an ABSENCE needs a positive control").
"""
from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")  # zero-dep CI has no jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from stelling import coverage
from stelling import interval as iv
from stelling import propagate as P
from stelling._jax_compat import transcribe


@pytest.fixture(autouse=True)
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def _eqn(prim, fn, *args):
    cj = transcribe(jax.make_jaxpr(fn)(*args))
    for e in cj.jaxpr.eqns:
        if str(e.primitive) == prim:
            return e
    raise AssertionError(f"fixture produced no {prim!r} equation")


def _box(prim, eqn, ins):
    tf, _tier = P.TRANSFERS[prim]
    return tf(eqn, dict(eqn.params_dict()), list(ins))


def _iv(lo, hi):
    return iv.IntervalArray(shape=(1,), los=(lo,), his=(hi,))


SIGN_EQN = lambda: _eqn("sign", lax.sign, jnp.zeros((1,), jnp.float64))
REM_EQN = lambda: _eqn(
    "rem", lax.rem, jnp.zeros((1,), jnp.float64), jnp.ones((1,), jnp.float64)
)


# --------------------------------------------------------------------------
# sign — the subnormal band
# --------------------------------------------------------------------------
SUBNORMALS = (5e-324, 1e-320, 2.2e-308)


@pytest.mark.parametrize("v", SUBNORMALS)
def test_the_target_really_does_flush_subnormals(v):
    """THE PREMISE, pinned. If this ever stops holding, the MIN_NORMAL floor
    below is over-conservative rather than load-bearing, and the row's
    docstring is making a claim about a target that changed."""
    assert v != 0.0, "the literal must be a genuine subnormal, not a zero"
    assert float(np.asarray(jnp.float64(v))) == v, "the value must round-trip"
    assert float(lax.sign(jnp.asarray(np.array([v], np.float64)))[0]) == 0.0
    assert float(np.sign(np.float64(v))) == 1.0, (
        "numpy must still disagree — this is what makes 'the oracle is jax, "
        "not numpy' a live clause rather than a stale one"
    )


def test_sign_admits_zero_across_the_subnormal_band():
    box = _box("sign", SIGN_EQN(), [_iv(5e-324, 1e-320)])
    assert box is not None, "a bounded real box must be admitted"
    assert box[0].los[0] == 0.0 and box[0].his[0] == 1.0, (
        f"a box inside the subnormal band must admit 0, got "
        f"[{box[0].los[0]}, {box[0].his[0]}]"
    )


def test_the_floor_is_load_bearing(monkeypatch):
    """ANTI-VACUITY. Drop MIN_NORMAL to 0.0 — the naive rule — and the false
    box must reappear, with the executed value outside it."""
    monkeypatch.setattr(iv, "MIN_NORMAL", 0.0)
    box = _box("sign", SIGN_EQN(), [_iv(5e-324, 1e-320)])
    assert box is not None and box[0].los[0] == 1.0, (
        "with the floor removed the naive rule must claim a definite [1, 1]; "
        "if it does not, this file is not testing what it claims to"
    )
    truth = float(lax.sign(jnp.asarray(np.array([1e-320], np.float64)))[0])
    assert truth == 0.0 and not (box[0].los[0] <= truth <= box[0].his[0]), (
        "and the executed value must lie OUTSIDE that box — otherwise the "
        "floor is guarding against nothing"
    )


def test_sign_is_definite_at_and_above_min_normal():
    """The floor must be a threshold, not a blanket widening: the first
    normal is definite, and the last subnormal below it is not."""
    hi = _box("sign", SIGN_EQN(), [_iv(iv.MIN_NORMAL, 1.0)])
    assert (hi[0].los[0], hi[0].his[0]) == (1.0, 1.0)
    lo = _box("sign", SIGN_EQN(), [_iv(1e-320, 1.0)])
    assert (lo[0].los[0], lo[0].his[0]) == (0.0, 1.0)
    truth = float(lax.sign(jnp.asarray(np.array([iv.MIN_NORMAL], np.float64)))[0])
    assert truth == 1.0, "MIN_NORMAL itself must not flush, or the floor is wrong"


@pytest.mark.parametrize(
    "lo,hi,want",
    [
        (-3.0, 3.0, (-1.0, 1.0)),
        (0.5, 3.0, (1.0, 1.0)),
        (-3.0, -0.5, (-1.0, -1.0)),
        (0.0, 0.0, (0.0, 0.0)),      # tight: no operand in [0,0] has sign 1
        (-0.0, 0.0, (0.0, 0.0)),
        (-np.inf, np.inf, (-1.0, 1.0)),
    ],
)
def test_sign_ordinary_boxes(lo, hi, want):
    box = _box("sign", SIGN_EQN(), [_iv(lo, hi)])
    assert (box[0].los[0], box[0].his[0]) == want


def test_sign_refuses_complex():
    """The `square` precedent: complex IS in sign_p's domain and sign(z) is
    z/|z|, which is not a real +/-1."""
    eqn = _eqn("sign", lax.sign, jnp.zeros((1,), jnp.complex128))
    with pytest.raises(iv.IntervalError):
        _box("sign", eqn, [_iv(-3.0, 3.0)])
    z = complex(np.asarray(lax.sign(jnp.asarray(3 + 4j, jnp.complex128))))
    assert abs(z) == pytest.approx(1.0) and z.imag != 0.0


def test_a_nan_endpoint_is_refused_BEFORE_the_transfer_sees_it():
    """`_t_sign` carries no NaN branch, and this is why it does not need one.

    Falling through on a NaN endpoint would give [-1, 1], which does NOT
    contain sign(NaN) = NaN — so the concern is real. It is closed at the
    constructor, not in the transfer, and pinning it here is what keeps the
    missing branch a measured absence rather than an oversight. If this ever
    stops raising, `_t_sign` needs the branch its docstring says it doesn't.
    """
    with pytest.raises(iv.IntervalError, match="NaN endpoint"):
        _iv(float("nan"), 1.0)


@pytest.mark.parametrize("dt,lo,hi,want", [
    ("int8", -128.0, 127.0, (-1.0, 1.0)),
    ("int8", 1.0, 127.0, (1.0, 1.0)),
    ("int8", 0.0, 127.0, (0.0, 1.0)),
    ("uint8", 0.0, 255.0, (0.0, 1.0)),
])
def test_sign_on_integers_uses_a_floor_of_one(dt, lo, hi, want):
    """Integers have no subnormals; the smallest definitely-positive integer
    is 1, so [0, hi] must still admit 0."""
    eqn = _eqn("sign", lax.sign, jnp.zeros((1,), dt))
    box = _box("sign", eqn, [_iv(lo, hi)])
    assert (box[0].los[0], box[0].his[0]) == want


# --------------------------------------------------------------------------
# rem — truncated, and the divisor guard
# --------------------------------------------------------------------------
def test_lax_rem_is_truncated_and_jnp_mod_is_floored():
    """THE PREMISE. They disagree on every mixed-sign case; a transfer built
    for the floored rule would be wrong on all of them."""
    for a, b, trunc, floor in [(-7.0, 3.0, -1.0, 2.0), (7.0, -3.0, 1.0, -2.0),
                               (-5.5, 2.0, -1.5, 0.5)]:
        assert float(lax.rem(jnp.float64(a), jnp.float64(b))) == trunc
        assert float(jnp.mod(jnp.float64(a), jnp.float64(b))) == floor


@pytest.mark.parametrize("alo,ahi,blo,bhi", [
    (1.0, 7.0, 2.0, 3.0), (-7.0, -1.0, 2.0, 3.0), (1.0, 7.0, -3.0, -2.0),
    (-7.0, -1.0, -3.0, -2.0), (-7.0, 7.0, 2.0, 3.0), (0.5, 1.5, 10.0, 20.0),
])
def test_rem_contains_the_executed_value(alo, ahi, blo, bhi):
    box = _box("rem", REM_EQN(), [_iv(alo, ahi), _iv(blo, bhi)])
    assert box is not None
    lo, hi = box[0].los[0], box[0].his[0]
    for a in (alo, ahi, (alo + ahi) / 2):
        for b in (blo, bhi, (blo + bhi) / 2):
            got = float(lax.rem(jnp.float64(a), jnp.float64(b)))
            assert lo <= got <= hi, f"rem({a}, {b}) = {got} outside [{lo}, {hi}]"


def test_rem_carries_the_sign_of_the_dividend():
    """The truncated rule, read off the box: a negative dividend cannot
    produce a positive remainder."""
    box = _box("rem", REM_EQN(), [_iv(-7.0, -1.0), _iv(2.0, 3.0)])
    assert box[0].his[0] == 0.0, "a wholly negative dividend bounds rem above by 0"
    box = _box("rem", REM_EQN(), [_iv(1.0, 7.0), _iv(-3.0, -2.0)])
    assert box[0].los[0] == 0.0, "a wholly positive dividend bounds rem below by 0"


@pytest.mark.parametrize("blo,bhi", [
    (-1.0, 1.0), (0.0, 3.0), (-3.0, 0.0), (0.0, 0.0),
    (5e-324, 1e-320), (-1e-320, -5e-324),
])
def test_rem_refuses_a_divisor_that_can_read_as_zero(blo, bhi):
    with pytest.raises(iv.IntervalError):
        _box("rem", REM_EQN(), [_iv(1.0, 7.0), _iv(blo, bhi)])


def test_rem_admits_a_divisor_at_min_normal():
    """The guard must be a threshold, not a blanket refusal — otherwise it
    would be indistinguishable from 'rem is unimplemented'."""
    box = _box("rem", REM_EQN(), [_iv(1.0, 7.0), _iv(iv.MIN_NORMAL, 1.0)])
    assert box is not None


def test_the_divisor_guard_is_load_bearing(monkeypatch):
    """ANTI-VACUITY, and it is the SUBNORMAL half that needs it: a divisor
    box excluding zero exactly still reads as zero under DAZ."""
    monkeypatch.setattr(iv, "MIN_NORMAL", 0.0)
    box = _box("rem", REM_EQN(), [_iv(1.0, 7.0), _iv(5e-324, 1e-320)])
    assert box is not None, (
        "with the threshold at 0.0 the subnormal divisor must be admitted; "
        "if it still refuses, the MIN_NORMAL threshold is not what excludes it"
    )


def test_rem_refuses_an_unbounded_dividend():
    with pytest.raises(iv.IntervalError):
        _box("rem", REM_EQN(), [_iv(-np.inf, np.inf), _iv(2.0, 3.0)])
    assert bool(np.isnan(np.asarray(lax.rem(jnp.float64(np.inf), jnp.float64(2.0)))))


def test_rem_completes_jnp_mod(monkeypatch):
    """The row's licensed value, measured both ways rather than asserted."""
    cj = transcribe(jax.make_jaxpr(jnp.mod)(jnp.float64(1), jnp.float64(2)))
    after = coverage.measure(cj, frozenset(P.TRANSFERS))
    assert (after.total, after.known, after.transparent, after.unknown) == (
        9, 8, 1, 0
    ), (
        f"with the rem row, jnp.mod must have no unknowns; got {after}. "
        f"Stated as 8 known of 9 equations, NOT '9/9': the ninth is the "
        f"transparent jit wrapper, counted in its own column"
    )
    without = dict(P.TRANSFERS)
    without.pop("rem")
    before = coverage.measure(cj, frozenset(without))
    assert (before.known, before.unknown) == (7, 1)
    assert before.unknown_primitives == (("rem", 1),), (
        f"and rem must be the ONLY thing missing without it; got {before}"
    )


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------
def test_both_rows_are_censused_in_every_registry():
    assert set(P.IEEE_TRANSFERS) == set(P.TRANSFERS)
    for prim in ("sign", "rem"):
        assert prim in P.TRANSFERS and prim in P.IEEE_TRANSFERS
        assert prim in P._INT_COMPUTING, (
            f"{prim} produces values its operands need not contain, so it is "
            f"computing and must be swept by the behavioural integer probe"
        )
    from stelling.obligation import _SUPPORTED
    assert "sign" not in _SUPPORTED and "rem" not in _SUPPORTED, (
        "transfer-only rows: emission is parked, and that is a parking "
        "decision rather than a capability judgment"
    )


def test_the_ieee_rows_are_real_rows_not_declines():
    """Unlike `square`, both of these cover ieee mode — recorded here so a
    later decline is a visible regression rather than a quiet one."""
    for prim, ins in (("sign", [_iv(0.5, 3.0)]),
                      ("rem", [_iv(1.0, 7.0), _iv(2.0, 3.0)])):
        eqn = SIGN_EQN() if prim == "sign" else REM_EQN()
        fn, tier = P.IEEE_TRANSFERS[prim]
        out = fn(eqn, dict(eqn.params_dict()), ins, [False] * len(ins))
        assert out is not None, f"{prim}'s ieee row must not decline"
        # SOUND, not exact: _ieee_rem IS _t_rem plus a flag, so it cannot carry
        # a stronger tier than the real row; and sign is the achievable hull
        # only under gradual underflow. Both corrected after a blinded audit.
        assert tier == "sound", f"{prim}: {tier}"
        assert P.TRANSFERS[prim][1] == "sound"


# --------------------------------------------------------------------------
# regressions from the blinded audits — each was a live defect in a green tree
# --------------------------------------------------------------------------
F32_SUBNORMAL = 1e-40   # normal as a binary64, subnormal as a float32


@pytest.mark.parametrize("dt", ["float32", "bfloat16", "float16"])
def test_non_f64_floats_decline_because_the_floor_is_a_BINARY64_floor(dt):
    """`MIN_NORMAL` is 2**-1022; float32 flushes below 2**-126. An f32 box
    clearing the f64 floor is still flushed by the target, so `sign` returned
    a definite [1,1] where jax returns 0.0 — end to end a DISCHARGED
    obligation at 4/4 known coverage that execution refutes.

    f16 is measured NOT to flush here and declines anyway, following the
    adjudication in tests/test_ieee_f32_band.py: pinned as a decline, not as
    target behaviour.
    """
    zero = jnp.zeros((1,), dt)
    with pytest.raises(iv.IntervalError, match="BINARY64"):
        _box("sign", _eqn("sign", lax.sign, zero), [_iv(F32_SUBNORMAL, 1e-30)])
    with pytest.raises(iv.IntervalError, match="BINARY64"):
        _box("rem", _eqn("rem", lax.rem, zero, jnp.ones((1,), dt)),
             [_iv(1.0, 1.0), _iv(1e-39, 1e-38)])


def test_the_f32_hole_really_was_a_false_box():
    """ANTI-VACUITY for the guard above: the value it excludes is real."""
    assert float(np.asarray(jnp.float32(F32_SUBNORMAL))) != 0.0, "stored nonzero"
    assert float(lax.sign(jnp.float32(F32_SUBNORMAL))) == 0.0, "but flushed"
    assert bool(np.isnan(np.asarray(
        lax.rem(jnp.float32(1.0), jnp.float32(5e-39))))), "and rem is NaN"


def test_sign_on_integers_reaches_the_overflow_guard():
    """`any_array` validates shape and ordering but NOT bounds against the
    dtype, so a uint8 box of (-3, -1) is declarable. `sign` returned
    [-1, -1] — outside uint8 — at 100% coverage, minting a REFUTED. The
    unsigned-is-unreachable invariant is now enforced, not asserted."""
    eqn = _eqn("sign", lax.sign, jnp.zeros((1,), "uint8"))
    with pytest.raises(iv.IntervalError):
        _box("sign", eqn, [_iv(-3.0, -1.0)])
    ok = _box("sign", eqn, [_iv(0.0, 255.0)])
    assert (ok[0].los[0], ok[0].his[0]) == (0.0, 1.0), "the valid box still admits"


def test_rem_handles_the_array_versus_scalar_literal_form():
    """jaxprs carry scalar literals as RANK-0 operands of elementwise
    equations — `jnp.mod(x, 2.0)` and jax-md's `space.periodic` shift both do.
    An equal-shapes-only check declined that form outright while the static
    coverage census still read 0 unknown."""
    cj = transcribe(jax.make_jaxpr(lambda x: lax.rem(x, 2.0))(jnp.zeros((3,))))
    eqn = [e for e in cj.jaxpr.eqns if str(e.primitive) == "rem"][0]
    assert [tuple(v.aval.shape) for v in eqn.invars] == [(3,), ()], "the form"
    a = iv.IntervalArray(shape=(3,), los=(1.0,) * 3, his=(7.0,) * 3)
    b = iv.IntervalArray(shape=(), los=(2.0,), his=(2.0,))
    out = _box("rem", eqn, [a, b])
    assert out is not None and out[0].shape == (3,)
    for v in (1.0, 4.0, 7.0):
        got = float(lax.rem(jnp.float64(v), jnp.float64(2.0)))
        assert out[0].los[0] <= got <= out[0].his[0]


def test_sign_contains_the_executed_value():
    """`rem` had an executed-value containment test and `sign` did not."""
    eqn = _eqn("sign", lax.sign, jnp.zeros((1,), jnp.float64))
    for lo, hi in [(-3.0, 3.0), (0.5, 3.0), (-3.0, -0.5), (0.0, 0.0),
                   (5e-324, 1e-320), (iv.MIN_NORMAL, 1.0), (-1e-320, 1e-320)]:
        box = _box("sign", eqn, [_iv(lo, hi)])
        pts = {lo, hi, 5e-324, 1e-320, 0.0, -0.0, (lo + hi) / 2 if lo > -np.inf else 0.0}
        for v in (p for p in pts if lo <= p <= hi):
            got = float(lax.sign(jnp.asarray(np.array([v], np.float64)))[0])
            assert box[0].los[0] <= got <= box[0].his[0], (
                f"sign({v}) = {got} outside [{box[0].los[0]}, {box[0].his[0]}] "
                f"for box [{lo}, {hi}]"
            )


def test_the_decline_messages_name_the_right_reason():
    """Each refusal must state ITS cause, not a neighbour's."""
    with pytest.raises(iv.IntervalError, match="unit circle"):
        _box("sign", _eqn("sign", lax.sign, jnp.zeros((1,), jnp.complex128)),
             [_iv(-3.0, 3.0)])
    # integer rem-by-zero returns the DIVIDEND, not NaN — measured
    assert int(lax.rem(jnp.int32(7), jnp.int32(0))) == 7
    eqn = _eqn("rem", lax.rem, jnp.zeros((1,), "int32"), jnp.ones((1,), "int32"))
    with pytest.raises(iv.IntervalError, match="returns the DIVIDEND"):
        _box("rem", eqn, [_iv(1.0, 7.0), _iv(0.0, 3.0)])
    feqn = REM_EQN()
    with pytest.raises(iv.IntervalError, match="no interval contains NaN"):
        _box("rem", feqn, [_iv(1.0, 7.0), _iv(0.0, 3.0)])
