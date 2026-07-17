# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""cond / select_n branch transfers, traced from real jax control flow.

Skipped without jax. These confirm the transfers descend into branch
sub-jaxprs and join soundly (design/control-flow-hypothesis.md).
"""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402

from stelling.harness import any_array, assert_, trace  # noqa: E402
from stelling.propagate import propagate  # noqa: E402


@pytest.fixture(autouse=True, scope="module")
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def prims(cj):
    from stelling.coverage import sub_jaxprs

    out, stack = [], [cj.jaxpr]
    while stack:
        j = stack.pop()
        for e in j.eqns:
            out.append(e.primitive)
            stack.extend(sub_jaxprs(e))  # descend so nested select_n is seen
    return out


def test_where_select_n_traces_and_join_is_sound():
    # a shrink-or-keep branch over a bounded step; both cases land in [0, 8]
    def h():
        nf = any_array((), "bool", (0.0, 1.0))  # non-finite flag, unknown
        s = any_array((), "float64", (0.0, 10.0))
        new = jnp.where(nf > 0.5, s * 0.8, s)  # select_n under the hood
        return assert_(new <= 10.0)  # join of [0,8] and [0,10] is [0,10]

    cj = trace(h)
    prim = prims(cj)
    assert "select_n" in prim
    assert propagate(cj).all_discharged


def test_where_ratchet_is_dependency_shaped():
    # `new <= s` where new is computed from s: the join loses the correlation
    # -> dependency-shaped UNKNOWN, not a false discharge (recorded finding)
    def h():
        nf = any_array((), "bool", (0.0, 1.0))
        s = any_array((), "float64", (0.1, 10.0))
        new = jnp.where(nf > 0.5, s * 0.8, s)
        return assert_(new <= s)

    assert not propagate(trace(h)).all_discharged


def test_cond_definite_branch_is_exact():
    def h():
        x = any_array((), "float64", (1.0, 2.0))
        # a real lax.cond with a compile-time-constant index -> one branch
        y = jax.lax.cond(True, lambda v: v * 2.0, lambda v: v * 100.0, x)
        return assert_(y <= 4.5)  # only the *2 branch is taken (v*100 -> [100,200])

    cj = trace(h)
    assert "cond" in prims(cj)
    assert propagate(cj).all_discharged  # definite index -> exact single branch


def test_cond_straddling_predicate_joins_and_may_be_unknown():
    def h():
        p = any_array((), "float64", (0.0, 1.0))  # unknown predicate
        x = any_array((), "float64", (1.0, 2.0))
        y = jax.lax.cond(p > 0.5, lambda v: v, lambda v: v + 100.0, x)
        return assert_(y <= 4.0)  # the +100 branch violates -> join straddles

    cj = trace(h)
    assert "cond" in prims(cj)
    # both branches joined: y in [1, 102] -> assert cannot hold -> not discharged
    assert not propagate(cj).all_discharged


def test_cond_straddle_join_is_sound_when_both_branches_satisfy():
    def h():
        p = any_array((), "float64", (0.0, 1.0))
        x = any_array((), "float64", (1.0, 2.0))
        y = jax.lax.cond(p > 0.5, lambda v: v + 1.0, lambda v: v + 2.0, x)
        return assert_(y <= 4.5)  # both branches: [2,3] and [3,4] -> join [2,4] <= 4.5

    assert propagate(trace(h)).all_discharged
