# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Transparent-call descent gives every inner binding a FRESH variable id.

The transcriber numbers variables PER SCOPE, so two inner jaxprs both contain
id 3. Flattening them into one namespace collided, and `_Slicer` poisoned the
whole query rather than alias two different values — sound, but a ceiling: any
query whose flattened scopes happened to number alike could not escalate at all.

Uniqueness is now enforced at the descent instead of assumed of the producer.

NON-TRIVIAL ASSERTIONS THROUGHOUT (CONTRIBUTING.md: a probe reading a final
verdict must assert something non-trivial). The bounds below sit within 5% of the
executed maximum, so no interval discharges them and every test must actually
reach the slice — which is the stage being tested. A `>= -1e30` bound here would
be decided by interval arithmetic and would exercise nothing, which is exactly
how this mechanism's first investigation produced three false "clean" readings.
"""
from __future__ import annotations

import jax
import numpy as np
import pytest

from stelling._jax_compat import transcribe
from stelling.harness import any_array, assert_
from stelling.obligation import _Slicer
from stelling.preconditions import check
from stelling.propagate import interval_env

N = 3


@pytest.fixture(autouse=True)
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


@jax.jit
def _f(x):
    return x * 2.0 + 1.0


def _true_max(body):
    return float(np.max(np.asarray(body(jax.numpy.full((N,), 1.0)))))


def _query(body, bound):
    def q():
        x = any_array((N,), "float64", (0.0, 1.0))
        return (assert_(body(x) <= bound),)
    return q


def test_same_callee_twice_is_no_longer_poisoned():
    """The ceiling itself. Reads `_Slicer.poisoned` DIRECTLY rather than
    inferring it from a verdict, because a verdict cannot distinguish "not
    poisoned" from "poisoned but decided earlier by intervals"."""
    body = lambda x: _f(x) + _f(x)  # noqa: E731
    cj = transcribe(jax.make_jaxpr(_query(body, 99.0))())
    sl = _Slicer(cj, interval_env(cj))
    assert sl.poisoned is None, sl.poisoned


def test_the_renumbering_path_is_actually_exercised():
    """ANTI-VACUITY. A test that passes because the descent never happened
    would look identical to one that passes because renumbering works.

    So: assert that fresh ids were allocated (the counter moved past the
    top-level maximum) AND that the inner bindings are recorded under ids the
    original jaxpr never contained.
    """
    body = lambda x: _f(x) + _f(x)  # noqa: E731
    cj = transcribe(jax.make_jaxpr(_query(body, 99.0))())
    original = {v.id for v in cj.jaxpr.constvars}
    original |= {v.id for e in cj.jaxpr.eqns for v in e.outvars}
    top_max = max(original)

    sl = _Slicer(cj, interval_env(cj))
    assert sl._next_id > top_max + 1, (
        "no fresh id was allocated — the descent did not run, so this test "
        "would pass whether or not renumbering works"
    )
    fresh = {vid for vid in sl.defined if vid > top_max}
    assert fresh, "no binding recorded under a fresh id"
    # and those fresh ids must carry real bindings, not empty slots
    assert any(vid in sl.aliases for vid in fresh), (
        "fresh ids were allocated but nothing was bound to them"
    )


@pytest.mark.parametrize("n_calls", [2, 3])
def test_repeated_callee_now_decides_and_agrees_with_executed_jax(n_calls):
    """The decision must be RIGHT, not merely present. The whole risk of this
    change is converting a safe decline into a wrong answer."""
    def body(x):
        acc = _f(x)
        for _ in range(n_calls - 1):
            acc = acc + _f(x)
        return acc

    m = _true_max(body)
    span = max(abs(m), 1.0)
    v_true = check(_query(body, m + 0.05 * span), vacuity_mode="inputs-only",
                   solver_timeout_ms=20000)
    v_false = check(_query(body, m - 0.05 * span), vacuity_mode="inputs-only",
                    solver_timeout_ms=20000)
    assert v_true.status == "VERIFIED", (
        f"{n_calls} calls: a TRUE bound was not verified ({v_true.status})"
    )
    assert v_false.status == "REFUTED", (
        f"{n_calls} calls: a FALSE bound was not refuted ({v_false.status}) — "
        f"executed jax reaches {m}"
    )


def test_distinct_shallow_callees_are_unchanged():
    """The no-change side. These escaped the ceiling already, so renumbering
    must leave them exactly as they were."""
    @jax.jit
    def s1(x):
        return x + 0.25

    @jax.jit
    def s2(x):
        return x * 1.5

    body = lambda x: s1(x) + s2(x)  # noqa: E731
    m = _true_max(body)
    span = max(abs(m), 1.0)
    assert check(_query(body, m + 0.05 * span), vacuity_mode="inputs-only",
                 solver_timeout_ms=20000).status == "VERIFIED"
    assert check(_query(body, m - 0.05 * span), vacuity_mode="inputs-only",
                 solver_timeout_ms=20000).status == "REFUTED"


def test_the_poison_survives_for_same_level_duplicates():
    """Renumbering must not paper over a duplicate that arrives at the SAME
    level — hand-built or deserialized IR that binds one id twice at top
    level is still a scope defect, and the backstop must still fire."""
    sl = _Slicer.__new__(_Slicer)
    sl.poisoned = None
    sl.defined = set()
    sl._define(7, "constvar 7")
    sl._define(7, "outvar of 'add'")
    assert sl.poisoned is not None
    assert "bound more than once" in sl.poisoned
