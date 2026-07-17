# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The harness primitives land in the trace, and the hash covers them.

Skipped wholesale when jax is absent. x64 is enabled for this module and
restored afterwards (the harness contract puts precision in the stamp, and
these tests trace f64 declarations).
"""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402

from stelling import ir  # noqa: E402
from stelling.harness import any_array, assert_, assume, trace  # noqa: E402
from stelling.propagate import propagate  # noqa: E402
from stelling.verdict import make_verdict  # noqa: E402


@pytest.fixture(autouse=True, scope="module")
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def exp_harness(bound):
    def h():
        x = any_array((), "float64", (1.0, 2.0))
        return assert_(jnp.exp(x) < bound)

    return h


def primitives(cj: ir.ClosedJaxpr):
    out = []
    stack = [cj.jaxpr]
    while stack:
        j = stack.pop()
        for eqn in j.eqns:
            out.append(eqn.primitive)
    return out


def test_declarations_land_in_the_traced_query():
    cj = trace(exp_harness(8.0))
    prims = primitives(cj)
    assert "stelling_any" in prims and "stelling_assert" in prims
    any_eqn = next(e for e in cj.jaxpr.eqns if e.primitive == "stelling_any")
    params = any_eqn.params_dict()
    assert (params["lo"], params["hi"]) == (1.0, 2.0)


def test_content_hash_covers_the_bounds():
    def h_with(bounds):
        def h():
            x = any_array((), "float64", bounds)
            return assert_(jnp.exp(x) < 8.0)

        return h

    assert trace(h_with((1.0, 2.0))).content_hash() != trace(
        h_with((1.0, 3.0))
    ).content_hash()
    assert trace(h_with((1.0, 2.0))).content_hash() == trace(
        h_with((1.0, 2.0))
    ).content_hash()


def test_assume_lands_and_is_inert():
    def h():
        x = any_array((), "float64", (1.0, 2.0))
        assume(x > 1.5)  # MVP: recorded, not refining
        return assert_(jnp.exp(x) < 8.0)

    cj = trace(h)
    assert "stelling_assume" in primitives(cj)
    assert propagate(cj).all_discharged


def test_end_to_end_verified():
    cj = trace(exp_harness(8.0))
    v = make_verdict(
        cj,
        propagate(cj),
        stelling_version="test",
        jax_version=jax.__version__,
        precision_config="jax_enable_x64=True",
    )
    assert v.status == "VERIFIED"


def test_positive_control_mutation_comes_back_red():
    """§10.8: a harness that cannot fail proves nothing — this one must."""
    cj = trace(exp_harness(2.0))  # exp([1,2]) ⊆ [2.71, 7.39]: never < 2
    v = make_verdict(
        cj,
        propagate(cj),
        stelling_version="test",
        jax_version=jax.__version__,
        precision_config="jax_enable_x64=True",
    )
    assert v.status == "UNKNOWN"
    assert v.obligations[0].status == "violated-over-set"


def test_eager_any_array_refuses():
    with pytest.raises(RuntimeError):
        any_array((), "float64", (0.0, 1.0))
