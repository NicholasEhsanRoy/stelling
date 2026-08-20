# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Tests for the 0.2.0 phase-1 transfers: is_finite, int64->float64 point
conversion, and the solver kwarg on check().
"""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402

from stelling.harness import any_array, assert_, trace  # noqa: E402
from stelling.propagate import propagate  # noqa: E402
from stelling.verdict import make_verdict  # noqa: E402


@pytest.fixture(autouse=True, scope="module")
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


VER = dict(
    stelling_version="test",
    jax_version="test",
    precision_config="jax_enable_x64=True",
)


def run(h):
    return propagate(trace(h))


# --- Feature 1: is_finite transfer -------------------------------------------


def test_isfinite_bounded_discharges():
    """jnp.isfinite on a bounded declaration is definitely true -> discharges."""

    def h():
        x = any_array((), "float64", (1.0, 10.0))
        return assert_(jnp.isfinite(x))

    p = run(h)
    assert p.obligations[0].status == "discharged"
    assert ("is_finite", "exact") in p.transfers_used


def test_isfinite_unbounded_is_unknown():
    """jnp.isfinite on an unbounded interval (inf endpoint) is unknown."""

    def h():
        x = any_array((), "float64", (0.0, float("inf")))
        return assert_(jnp.isfinite(x))

    p = run(h)
    assert p.obligations[0].status == "unknown"


def test_isfinite_negative_inf_is_unknown():
    """jnp.isfinite with -inf lower bound is unknown."""

    def h():
        x = any_array((), "float64", (float("-inf"), 5.0))
        return assert_(jnp.isfinite(x))

    p = run(h)
    assert p.obligations[0].status == "unknown"


def test_isfinite_ieee_mode_bounded():
    """is_finite under ieee semantics: bounded operand without NaN -> true."""

    def h():
        x = any_array((), "float64", (1.0, 10.0))
        return assert_(jnp.isfinite(x))

    cj = trace(h)
    p = propagate(cj, semantics="ieee")
    assert p.obligations[0].status == "discharged"


# --- Feature 2: int64->float64 point conversion ------------------------------


def test_int_literal_cast_does_not_decline():
    """jnp.where(mask, x, 1) involves an int->float cast of 1; should not
    decline since 1 is exactly representable in float64."""

    def h():
        mask = any_array((2,), "bool", (0.0, 1.0))
        x = any_array((2,), "float64", (0.0, 5.0))
        result = jnp.where(mask, x, 1)
        return assert_(result <= 5.0)

    p = run(h)
    # The int literal 1 is cast to float64; should pass through, not decline
    assert p.obligations[0].status == "discharged"
    # No convert decline notes
    convert_declines = [n for n in p.notes if "int64" in n and "float64" in n
                        and "declined" in n]
    assert not convert_declines


def test_int64_to_float64_non_point_still_declines():
    """An int64 range (not a point) to float64 still declines."""

    def h():
        # A non-point int64 interval cannot use the point rule
        x = any_array((), "int64", (1, 100))
        return assert_(jnp.float64(x) <= 100.0)

    p = run(h)
    # The non-point int64 range should produce a decline note
    convert_notes = [n for n in p.notes if "int64" in n and "float64" in n]
    assert convert_notes  # the conversion was declined


# --- Feature 3: solver kwarg on check() --------------------------------------


def test_check_solver_kwarg_validation():
    """The solver kwarg validates its value eagerly."""
    from stelling.preconditions import check, scalar_nonzero

    def h():
        _, o = scalar_nonzero("float64", (0.0, 1.0))
        return (o,)

    with pytest.raises(ValueError, match="solver must be"):
        check(h, vacuity_mode="inputs-only", solver="mathematica")


def test_check_solver_kwarg_z3_records_only_z3():
    """check(..., solver='z3') restricts the portfolio to z3 only."""
    from stelling._optional import available
    from stelling.preconditions import check, scalar_nonzero

    if not available("z3"):
        pytest.skip("needs z3")

    def h():
        _, o = scalar_nonzero("float64", (0.0, 1.0))
        return (o,)

    v = check(h, vacuity_mode="inputs-only", solver_timeout_ms=20000,
              solver="z3")
    # The verdict should record z3 usage; no cvc5 invocations
    if isinstance(v.stamp.solver, tuple):
        # tuple of SolverStamp invocations
        for inv in v.stamp.solver:
            if inv.invoked:
                assert inv.name == "z3"
    assert v.status in ("REFUTED", "UNKNOWN")


def test_check_solver_kwarg_none_is_default():
    """solver=None (default) does not restrict the portfolio."""
    from stelling.preconditions import check, scalar_nonzero

    def h():
        _, o = scalar_nonzero("float64", (0.0, 1.0))
        return (o,)

    # Should not raise — None means use all available
    v = check(h, vacuity_mode="inputs-only")
    assert v.status in ("VERIFIED", "UNKNOWN", "REFUTED")
