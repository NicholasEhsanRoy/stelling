# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The nested-declaration guard: recursive, raising, and suppressing.

Three defects, fixed together because they compound:

1. the walk was ONE LEVEL deep, so a declaration two calls down escaped both
   the guard and the widening;
2. it was an `assert`, so `python -O` stripped it — and stripped, the widened
   query silently kept the nested declaration's ORIGINAL bounds, producing a
   verdict that carried "envelope not load-bearing" about an envelope that IS
   load-bearing (measured: the same claim with the envelope genuinely widened
   is REFUTED);
3. when it did fire it raised a bare `AssertionError` out of `check()`, which
   R7(b) settled: a query stelling cannot fully analyse degrades, it does not
   raise.

The disposition is SUPPRESSION — the vacuity instrument makes no claim in
either direction — reusing the inert path the identical-query case already
uses. Suppression is sound; a false qualification is not.
"""
from __future__ import annotations

import math

import pytest

jax = pytest.importorskip("jax")  # zero-dep CI has no jax
import jax.numpy as jnp  # noqa: E402

from stelling.harness import any_array, assert_  # noqa: E402
from stelling.preconditions import check  # noqa: E402
from stelling.vacuity import NestedDeclaration  # noqa: E402
from _solver_gate import need_solver  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


@jax.jit
def _d1(_):
    return any_array((), "float64", (1.0, 2.0))


@jax.jit
def _d2_inner(_):
    return any_array((), "float64", (1.0, 2.0))


@jax.jit
def _d2(t):
    return _d2_inner(t)


def _query(f):
    def q():
        x = any_array((), "float64", (-1.0, 1.0))
        # TRUE only because the nested envelope caps the second term at 2.0.
        # With it genuinely widened this is REFUTED — which is what makes
        # "envelope not load-bearing" a false claim rather than a harmless one.
        return (assert_((x + f(x)) - x < 2.5),)
    return q


@need_solver
@pytest.mark.parametrize("f,depth", [(_d1, 1), (_d2, 2)])
def test_no_false_load_bearing_note_at_any_depth(f, depth):
    v = check(_query(f), vacuity_mode="inputs-only", solver_timeout_ms=20000)
    assert not any("envelope not load-bearing" in n for n in v.notes), (
        f"depth {depth}: the verdict claims the envelope is not load-bearing, "
        f"but widening never reached the declaration"
    )
    # the reason rides in the STAMP's assumptions, where every other vacuity
    # disposition is recorded — not in notes
    assert any("transparent call" in a for a in v.stamp.assumptions), (
        f"depth {depth}: suppressed silently — the reason must be stamped, or "
        f"a reader cannot tell 'not measured' from 'measured and inert'"
    )


@need_solver
def test_the_envelope_really_is_load_bearing():
    """ANTI-VACUITY. If the claim were true without the nested envelope, the
    suppression above would be protecting nothing and these tests would pass
    for the wrong reason."""
    def q():
        x = any_array((), "float64", (-1.0, 1.0))
        z = any_array((), "float64", (-1e9, 1e9))   # genuinely unbounded
        return (assert_((x + z) - x < 2.5),)

    v = check(q, vacuity_mode="inputs-only", solver_timeout_ms=20000)
    assert v.status == "REFUTED", (
        f"expected REFUTED with the envelope removed, got {v.status} — the "
        f"nested envelope is not load-bearing and these tests prove nothing"
    )


@need_solver
def test_top_level_declarations_still_get_a_real_measurement():
    """ANTI-VACUITY, the other direction: the fix must not suppress
    everything. A top-level declaration is still widened and still measured."""
    def q():
        x = any_array((), "float64", (-1.0, 1.0))
        z = any_array((), "float64", (1.0, 2.0))
        return (assert_((x + z) - x < 2.5),)

    v = check(q, vacuity_mode="inputs-only", solver_timeout_ms=20000)
    assert v.status == "VERIFIED"
    assert not any("transparent call" in a for a in v.stamp.assumptions), (
        "a top-level declaration was suppressed as nested"
    )


def test_the_guard_raises_rather_than_asserts():
    """`-O` strips asserts. This one was missed by the sweep that converted
    the seven module-level census asserts for exactly that reason."""
    assert not issubclass(NestedDeclaration, AssertionError)


# -- the FOURTH defect: mixed point / non-point declarations ----------------
#
# Found by a blinded audit AFTER the three depth fixes, on an axis none of them
# touched. `widen` holds POINT declarations still under "inputs-only" by design
# -- a declared point is a stated constant, not a degenerate range -- but the
# verdict still claimed every bound had been widened.
#
# The fix is one predicate covering both axes: emit the note only if widen
# actually moved EVERY declared bound.


@need_solver
def test_mixed_point_and_nonpoint_suppresses_rather_than_claiming():
    """The audit's sharpest instance, with its executed control."""
    def q(s_lo, s_hi):
        def build():
            s = any_array((), "float64", (s_lo, s_hi))
            x = any_array((), "float64", (0.0, 1.0))
            return (assert_(s != 0.0), assert_(s * x * x <= 0.0))
        return build

    v = check(q(-1.0, -1.0), vacuity_mode="inputs-only",
              solver_timeout_ms=20000)
    assert v.status == "VERIFIED"
    assert not any("load-bearing" in n for n in v.notes), (
        "claimed the envelope is not load-bearing while the point declaration "
        "was never widened"
    )
    assert any("point interval" in a for a in v.stamp.assumptions)

    # EXECUTED CONTROL: the envelope really is load-bearing, so the suppressed
    # claim would have been false rather than merely unsupported.
    for lo, hi in ((-1.0, 1.0), (-1e30, 1e30)):
        w = check(q(lo, hi), vacuity_mode="inputs-only",
                  solver_timeout_ms=20000)
        assert w.status == "REFUTED", (
            f"s over ({lo},{hi}) gave {w.status}; if this is not REFUTED the "
            f"point envelope is not load-bearing and this test proves nothing"
        )


def test_the_one_rule_covers_both_axes():
    """`unwidened` is the single predicate. Depth and point-ness are the same
    fact -- a declared bound that did not move -- so both must produce a
    reason from it rather than from two special-case guards."""
    from stelling.vacuity import declaration_bounds, unwidened

    before = [(1, 0.0, 1.0), (2, -1.0, -1.0)]
    after_mixed = [(1, -math.inf, math.inf), (2, -1.0, -1.0)]
    after_all = [(1, -math.inf, math.inf), (2, -math.inf, math.inf)]
    assert unwidened(before, after_mixed) is not None
    assert unwidened(before, after_all) is None
    # vacuously-true over an empty set must NOT license the claim
    assert unwidened([], []) is not None

