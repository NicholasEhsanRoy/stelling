# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""A verdict is a function of the query and the envelope — not of an
escalation preference that had no effect.

    THE VERDICT IS IDENTICAL WITH AND WITHOUT ``solver_timeout_ms``
    WHENEVER NO OBLIGATION WAS SOLVER-DECIDED.

Found through the scatter VERIFIED bar, which fired on ``escalation is not
None`` — true whenever the caller passed ``solver_timeout_ms``, including when
interval arithmetic had already discharged every obligation and escalation did
nothing at all. Measured on a MIME surface-contact contract: the same query
returned VERIFIED without the argument and UNKNOWN with it, with
``obl-solves = 0`` and the obligation ``discharged`` in both runs.

The bar was the instance. The property is wider, and it is load-bearing for
things that have nothing to do with scatter: stamp comparison, the W6 soak's
per-line records, and the no-flip gate's premise all assume a verdict does not
move when an unused capability is switched on.
"""
from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")  # zero-dep CI has no jax
import jax.numpy as jnp  # noqa: E402

from stelling.harness import any_array, assert_  # noqa: E402
from stelling.preconditions import check  # noqa: E402
from stelling.verdict import VERIFIED_BARRED_PRIMITIVES  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def _scatter_query():
    """Interval-decidable, and its jaxpr contains `scatter` — the shape that
    exposed the defect. The bound is loose on purpose so intervals settle it
    without a solver."""
    def q():
        x = any_array((3,), "float64", (0.0, 1.0))
        return (assert_(x.at[0].set(0.5) <= 2.0),)
    return q


def _plain_query():
    def q():
        x = any_array((3,), "float64", (0.0, 1.0))
        return (assert_(x + 1.0 <= 3.0),)
    return q


def _obl_solves(v) -> int:
    sol = v.stamp.solver if v.stamp else None
    sols = sol if isinstance(sol, tuple) else (sol,)
    return len([s for s in sols if s and s.invoked and "widen" not in s.reason])


@pytest.mark.parametrize("build,label", [
    (_scatter_query(), "scatter on the slice"),
    (_plain_query(), "no barred primitive"),
])
def test_verdict_does_not_move_when_escalation_decides_nothing(build, label):
    without = check(build, vacuity_mode="inputs-only")
    with_ = check(build, vacuity_mode="inputs-only", solver_timeout_ms=20000)

    assert _obl_solves(with_) == 0, (
        f"{label}: an obligation WAS solver-decided, so this pair does not "
        f"exercise the invariant — pick a query intervals can settle"
    )
    assert without.status == with_.status, (
        f"{label}: {without.status} without solver_timeout_ms, "
        f"{with_.status} with it, while the solver decided NOTHING "
        f"(obl-solves=0). A verdict must not depend on an unused preference."
    )
    assert [o.status for o in without.obligations] == \
           [o.status for o in with_.obligations], (
        f"{label}: per-obligation statuses differ across an unused preference"
    )


def test_the_invariant_is_actually_exercised():
    """ANTI-VACUITY (Norm C). If no query in this module put a BARRED
    primitive on an interval-decided slice, the test above would pass on any
    build — including one where the bar fires on everything.

    So: prove the scatter query really does carry a barred primitive, and
    really is decided without a solver."""
    from stelling._jax_compat import transcribe

    build = _scatter_query()
    cj = transcribe(jax.make_jaxpr(build)())
    prims = {str(e.primitive) for e in cj.jaxpr.eqns}
    assert prims & VERIFIED_BARRED_PRIMITIVES, (
        f"the query carries no barred primitive ({sorted(prims)}), so the "
        f"invariant test above cannot detect a bar that over-fires"
    )
    v = check(build, vacuity_mode="inputs-only", solver_timeout_ms=20000)
    assert _obl_solves(v) == 0, "the query escalated; it cannot exercise this"
    assert v.status == "VERIFIED", (
        "an interval-decided, scatter-bearing query is not VERIFIED under "
        "escalation — the bar is still firing on a verdict no emission row "
        "touched"
    )


def test_the_bar_is_currently_unreachable_and_that_is_structural():
    """The other direction, and it cannot be tested here for a reason worth
    pinning rather than skipping.

    The bar withholds a SOLVER-DECIDED VERIFIED. A barred primitive can only
    be solver-decided if it is in the EMISSION set — and `scatter` is not,
    because that row is parked and unmerged. So on this build the bar's
    protective branch is unreachable by construction, and the only behaviour
    it had was the over-firing this module's other tests now forbid.

    This is not "the bar is useless". It is "the bar's function is contingent
    on the parked row landing", which is exactly when it will be needed. The
    assertion is written so that **merging the scatter emission row breaks
    this test**, which is the moment someone must add the positive-direction
    coverage that cannot exist today.
    """
    from stelling.obligation import _SUPPORTED

    assert not (VERIFIED_BARRED_PRIMITIVES & _SUPPORTED), (
        f"a barred primitive is now EMITTABLE "
        f"({sorted(VERIFIED_BARRED_PRIMITIVES & _SUPPORTED)}), so the bar's "
        f"protective branch is reachable for the first time. Add a test that "
        f"a solver-decided VERIFIED on that primitive's slice IS withheld — "
        f"until now no such query could exist, so the repair narrowing the "
        f"bar was never covered in the firing direction."
    )
