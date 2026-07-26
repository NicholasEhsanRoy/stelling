# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The scatter VERIFIED bar: it fires, and it fires only where intended.

The bar closes the one direction a wrong SMT emission row could not
self-check. A spurious witness is caught by exact-rational replay; a MISSED
violation would mint a false VERIFIED with nothing downstream to catch it.
So solver-path VERIFIED is withheld on queries containing a barred
primitive, and the refutation path — which is self-checking — stays open.

Both halves are pinned here. A bar nobody has seen fire is not a mechanism,
and a bar that fires too widely would silently withhold the interval-only
verdicts it was never meant to touch.
"""
from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")  # zero-dep CI has no jax
import jax.numpy as jnp

import stelling.verdict as V
from stelling.harness import any_array, assert_
from stelling.preconditions import check


@pytest.fixture(autouse=True)
def _x64():
    """Scope x64 to this module. Setting it at import leaks float64 into every
    later-run module in the same process — measured here the same way the
    suite already documents: it flipped test_transcribe's cross-process
    hash-stability test, which passes alone and fails in-suite."""
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def _query_with_scatter_and_an_smt_obligation():
    """A query that CONTAINS scatter and reaches the solver.

    Obligation 0 is undecidable by intervals (`y - y` propagates to a box
    straddling zero, since the abstraction is correlation-blind) and trivial
    for SMT. Obligation 1 touches the scatter and is settled by intervals.
    Both discharge, so the verdict would be VERIFIED but for the bar.
    """
    x = any_array((3,), "float64", (0.0, 1.0))
    s = x.at[0].set(0.5)
    y = any_array((), "float64", (1.0, 2.0))
    return (assert_(y - y >= 0.0), assert_(s >= 0.0))


def test_the_bar_withholds_a_solver_path_verified_on_a_scatter_query():
    # Deliberately NOT skipif-guarded on the bar being non-empty. A skip is
    # not a failure, so a guarded test cannot be mutation-proved: emptying
    # VERIFIED_BARRED_PRIMITIVES would silently skip this rather than fail it,
    # which is exactly the false negative the mutation norm exists to catch.
    # When the principal lifts the bar this test fails loudly and is updated
    # deliberately — the same one-identifiable-place discipline as the bar.
    assert V.VERIFIED_BARRED_PRIMITIVES, "the bar has been lifted"
    v = check(_query_with_scatter_and_an_smt_obligation,
              vacuity_mode="inputs-only", solver_timeout_ms=20000)
    assert all(o.status == "discharged" for o in v.obligations), (
        "the scenario must reach the bar with every obligation discharged — "
        "otherwise UNKNOWN would prove nothing about the bar"
    )
    assert v.status == "UNKNOWN"
    assert any("VERIFIED withheld" in n for n in v.notes)


def test_the_bar_does_not_touch_interval_only_verdicts():
    """The negative half, and the reason the bar is scoped to the solver path.

    HeatNode's Dirichlet writeback puts `scatter` in the jaxpr, and this
    obligation is discharged by intervals alone. A whole-verdict bar would
    withhold the Richardson flagship for a reason having nothing to do with
    the emission row under audit — the interval transfer is long-standing and
    unchanged by that work.
    """
    pytest.importorskip("maddening")  # jax CI job does not install it
    from maddening.nodes.heat import HeatNode

    node = HeatNode("h", timestep=0.001, n_cells=5, length=1.0,
                    thermal_diffusivity=0.01)

    def row7():
        T = any_array((5,), "float64", (10.0, 100.0))
        st = {"temperature": T}
        full = node.update(st, {}, 0.01)
        half = node.update(node.update(st, {}, 0.005), {}, 0.005)
        scale = 0.0 + 1e-6 * jnp.maximum(jnp.abs(half["temperature"]),
                                         jnp.abs(full["temperature"]))
        return (assert_(scale > 0.0),)

    cj = jax.make_jaxpr(row7)()
    assert any(str(e.primitive) == "scatter" for e in cj.jaxpr.eqns), (
        "this test is vacuous unless the query really does contain scatter"
    )
    v = check(row7, vacuity_mode="inputs-only")
    assert v.status == "VERIFIED"
    assert not any("VERIFIED withheld" in n for n in v.notes)
