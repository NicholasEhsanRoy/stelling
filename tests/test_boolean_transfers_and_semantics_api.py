# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Tests for Fix 1 (semantics parameter on check()) and Fix 2 (boolean
logic transfers: and, or, not) — the 0.2.0 integration gap fixes.

AND and OR were already registered; NOT is new. The integration tests
confirm that real jax traces through ``&``, ``|``, ``~`` on boolean
arrays propagate correctly and produce decidable selectors in
``jnp.where``.
"""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402

from stelling.harness import any_array, assert_, trace  # noqa: E402
from stelling.preconditions import check  # noqa: E402
from stelling.propagate import propagate  # noqa: E402


@pytest.fixture(autouse=True, scope="module")
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


# ---------------------------------------------------------------------------
# Fix 2: boolean logic transfers (AND, OR, NOT) through jnp.where
# ---------------------------------------------------------------------------


def test_and_selector_decidable():
    """jnp.where(cond1 & cond2, a, b) where both conditions are decidable
    -> the selector is decidable -> VERIFIED."""

    def h():
        x = any_array((), "float64", (1.0, 5.0))
        cond1 = x >= 1.0   # definitely true over [1, 5]
        cond2 = x <= 5.0   # definitely true over [1, 5]
        result = jnp.where(cond1 & cond2, x, jnp.float64(0.0))
        return assert_(result >= 1.0)

    v = check(h, vacuity_mode="inputs-only")
    assert v.status == "VERIFIED"


def test_or_selector_decidable():
    """jnp.where(cond | fallback_cond, a, b) -> decidable."""

    def h():
        x = any_array((), "float64", (2.0, 4.0))
        cond = x >= 3.0          # unknown over [2, 4]
        fallback = x >= 1.0      # definitely true over [2, 4]
        result = jnp.where(cond | fallback, x, jnp.float64(0.0))
        return assert_(result >= 2.0)

    v = check(h, vacuity_mode="inputs-only")
    assert v.status == "VERIFIED"


def test_not_transfer_traced():
    """~cond on a boolean array traces to 'not' and propagates correctly."""

    def h():
        x = any_array((), "float64", (1.0, 3.0))
        cond = x >= 5.0          # definitely false over [1, 3]
        negated = ~cond           # NOT(false) = definitely true
        result = jnp.where(negated, x, jnp.float64(0.0))
        return assert_(result >= 1.0)

    p = propagate(trace(h))
    assert p.obligations[0].status == "discharged"
    assert ("not", "exact") in p.transfers_used


def test_not_unknown_stays_unknown():
    """NOT of an unknown boolean is unknown."""

    def h():
        x = any_array((), "float64", (0.0, 2.0))
        cond = x >= 1.0          # unknown over [0, 2]
        negated = ~cond           # NOT(unknown) = unknown
        # If negated is unknown, jnp.where cannot decide the branch
        result = jnp.where(negated, x, jnp.float64(-1.0))
        return assert_(result >= 0.0)

    p = propagate(trace(h))
    # The obligation should be unknown (selector is undecided)
    assert p.obligations[0].status == "unknown"


# ---------------------------------------------------------------------------
# Fix 1: semantics parameter on check()
# ---------------------------------------------------------------------------


def test_check_semantics_ieee_stamp():
    """check(..., semantics='ieee') produces a verdict with ieee stamp."""

    def h():
        x = any_array((), "float64", (1.0, 2.0))
        return assert_(x >= 1.0)

    v = check(h, vacuity_mode="inputs-only", semantics="ieee")
    assert v.status == "VERIFIED"
    assert "ieee" in v.stamp.semantics.lower() or "IEEE" in v.stamp.semantics


def test_check_semantics_ieee_with_solver_raises():
    """check(..., semantics='ieee', solver_timeout_ms=5000) raises ValueError."""

    def h():
        x = any_array((), "float64", (1.0, 2.0))
        return assert_(x >= 1.0)

    with pytest.raises(ValueError, match="contradictory"):
        check(h, vacuity_mode="inputs-only", semantics="ieee",
              solver_timeout_ms=5000)


def test_check_semantics_invalid_raises():
    """Invalid semantics string raises ValueError."""

    def h():
        x = any_array((), "float64", (1.0, 2.0))
        return assert_(x >= 1.0)

    with pytest.raises(ValueError, match="semantics must be"):
        check(h, vacuity_mode="inputs-only", semantics="complex")
