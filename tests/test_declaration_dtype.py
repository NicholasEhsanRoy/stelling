# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""A declared box the dtype cannot hold is EMPTY, and empty declarations are
refused at declaration time — the fourth instance of a posture the harness
already held for a negative extent, `lo > hi`, and the infinite point.

Found because `sign` returned `[-1, -1]` on a `uint8` box of `(-3, -1)` at
100% coverage and minted a REFUTED. Routing `sign` through the overflow guard
fixed one row; **13 of 21 integer-accepting transfers admitted that box**, six
of them returning a definite boolean straight into an assert. The hole is at
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


@pytest.mark.parametrize("mode", ["real", "ieee"])
def test_the_refusal_precedes_both_semantics_modes(mode):
    """The check is at trace time, so it fires before any mode is chosen —
    exercised in both rather than argued."""
    with pytest.raises(ValueError, match="EMPTY under dtype"):
        P.propagate(trace(_sign_query()), semantics=mode)
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


@pytest.mark.parametrize("prim", sorted(SURFACE))
def test_a_legitimate_box_never_leaves_the_dtype(prim):
    """The OTHER entry point: a box that arrives by propagation rather than
    declaration. Computing rows reach the overflow guard; the rest cannot
    create a value their operands did not contain."""
    from stelling._jax_compat import transcribe
    fn = SURFACE[prim]
    cj = transcribe(jax.make_jaxpr(fn)(jnp.zeros((1,), "uint8")))
    eqn = [e for e in cj.jaxpr.eqns if str(e.primitive) == prim][0]
    box = iv.IntervalArray(shape=(1,), los=(0.0,), his=(255.0,))
    tf, _tier = P.TRANSFERS[prim]
    try:
        out = tf(eqn, dict(eqn.params_dict()), [box] * len(eqn.invars))
    except iv.IntervalError:
        return  # a loud refusal is sound
    if out is None or eqn.outvars[0].aval.dtype != "uint8":
        return
    assert 0 <= min(out[0].los) and max(out[0].his) <= 255, (
        f"{prim} took an in-range uint8 box to [{min(out[0].los)}, "
        f"{max(out[0].his)}], which leaves the dtype"
    )
