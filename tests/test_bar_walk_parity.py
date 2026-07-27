# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The VERIFIED bar's traversal, and its firing direction.

TWO THINGS ARE TESTED HERE THAT COULD NOT BE TESTED BEFORE.

**Parity.** `_barred_primitives` used to descend via
`getattr(v, "jaxpr", None)`, which finds a param that IS a ClosedJaxpr but not
one holding a COLLECTION of them — and `cond` stores its branches as a tuple.
Measured: a `scatter` inside a `cond` branch was not barred at all, while the
same primitive at top level, inside `jit`, and inside `scan` all barred
correctly. That is the UNDER-firing direction, which the function's own
docstring calls the worse one. The fix routes descent through
`coverage.sub_jaxprs`; this module asserts mechanically that the two agree, in
the spirit of the EMISSION == REPLAY census — a review question nobody has to
remember to ask.

**The firing direction, via a SYNTHETIC barred primitive.** The bar withholds a
SOLVER-DECIDED VERIFIED. On this build the real barred set is `{"scatter"}` and
`scatter` is absent from the emission set, so no scatter obligation can ever be
solver-decided and the bar's protective branch is unreachable — its entire
behavioural history is over-firing. Monkeypatching the barred set to contain
something *emittable* (`add`) makes that branch reachable, so the direction the
bar exists for is exercised for the first time. Without this fixture, both the
bar and the traversal fix would be "correct by argument only" until the scatter
row registers, which is exactly when they become load-bearing.
"""
from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")  # zero-dep CI has no jax
import jax.numpy as jnp  # noqa: E402

from stelling import verdict as _verdict  # noqa: E402
from stelling._jax_compat import transcribe  # noqa: E402
from stelling.coverage import sub_jaxprs  # noqa: E402
from stelling.harness import any_array, assert_  # noqa: E402
from stelling.preconditions import check  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


# ---- the query shapes, one per nesting construct --------------------------

@jax.jit
def _jitted(x):
    return x.at[0].set(0.5)


def _top():
    x = any_array((3,), "float64", (0.0, 1.0))
    return (assert_(x.at[0].set(0.5) <= 2.0),)


def _in_jit():
    x = any_array((3,), "float64", (0.0, 1.0))
    return (assert_(_jitted(x) <= 2.0),)


def _in_cond():
    x = any_array((3,), "float64", (0.0, 1.0))
    y = jax.lax.cond(x[0] > 0.5, lambda a: a.at[0].set(0.5), lambda a: a, x)
    return (assert_(y <= 2.0),)


def _in_scan():
    x = any_array((3,), "float64", (0.0, 1.0))

    def step(c, _):
        return c.at[0].set(0.5), 0.0

    y, _ = jax.lax.scan(step, x, jnp.arange(2.0))
    return (assert_(y <= 2.0),)


def _in_cond_in_jit():
    x = any_array((3,), "float64", (0.0, 1.0))

    @jax.jit
    def outer(a):
        return jax.lax.cond(a[0] > 0.5, lambda b: b.at[0].set(0.5),
                            lambda b: b, a)

    return (assert_(outer(x) <= 2.0),)


SHAPES = [
    (_top, "top level"),
    (_in_jit, "inside jit"),
    (_in_cond, "inside cond"),
    (_in_scan, "inside scan"),
    (_in_cond_in_jit, "cond inside jit"),
]


def _all_primitives_via_canonical(closed):
    found, seen = set(), []

    def walk(j):
        for e in j.eqns:
            found.add(str(e.primitive))
            for sub in sub_jaxprs(e):
                if id(sub) not in seen:
                    seen.append(id(sub))
                    walk(sub)

    walk(closed.jaxpr)
    return found


@pytest.mark.parametrize("build,label", SHAPES, ids=[s[1] for s in SHAPES])
def test_bar_walk_has_parity_with_the_canonical_accessor(build, label):
    """The bar must see every primitive `sub_jaxprs` sees. Checked over the
    barred set's own membership, which is what the bar acts on."""
    closed = transcribe(jax.make_jaxpr(build)())
    canonical = _all_primitives_via_canonical(closed)
    barred = set(_verdict._barred_primitives(closed))
    expected = canonical & _verdict.VERIFIED_BARRED_PRIMITIVES
    assert barred == expected, (
        f"{label}: the bar found {sorted(barred)} where the canonical walk "
        f"implies {sorted(expected)} — the two traversals have diverged"
    )


def test_the_parity_test_catches_the_old_accessor():
    """ANTI-VACUITY (Norm C). A parity test that passes under the BROKEN walk
    would prove nothing. Re-implement the old descent and confirm it fails
    exactly where it did in the field: a barred primitive inside `cond`."""
    closed = transcribe(jax.make_jaxpr(_in_cond)())

    def old_walk(closed):
        found, seen = set(), []

        def walk(jaxpr):
            for eqn in getattr(jaxpr, "eqns", ()):
                if str(eqn.primitive) in _verdict.VERIFIED_BARRED_PRIMITIVES:
                    found.add(str(eqn.primitive))
                for v in eqn.params_dict().values():
                    inner = getattr(v, "jaxpr", None)   # the defect
                    if inner is not None and id(inner) not in seen:
                        seen.append(id(inner))
                        walk(inner)

        walk(closed.jaxpr)
        return found

    canonical = _all_primitives_via_canonical(closed)
    expected = canonical & _verdict.VERIFIED_BARRED_PRIMITIVES
    assert expected, "the fixture carries no barred primitive; test is vacuous"
    assert old_walk(closed) != expected, (
        "the OLD accessor agrees with the canonical one on this query, so the "
        "parity test above cannot distinguish them and proves nothing"
    )
    assert set(_verdict._barred_primitives(closed)) == expected


# ---- the firing direction, via a SYNTHETIC barred primitive ---------------

@pytest.fixture
def _bar_add(monkeypatch):
    """Bar `add` — which IS emittable, unlike `scatter`. That is the whole
    point: it makes the bar's protective branch reachable."""
    monkeypatch.setattr(_verdict, "VERIFIED_BARRED_PRIMITIVES",
                        frozenset({"add"}))
    yield


def _solver_decided_query():
    """Correlation-sensitive, so intervals cannot settle it and the solver
    must — which is the precondition for the bar to apply at all."""
    def q():
        x = any_array((3,), "float64", (0.0, 1.0))
        return (assert_((x + 1.0) - x <= 1.5),)
    return q


def _obl_solves(v):
    sol = v.stamp.solver if v.stamp else None
    sols = sol if isinstance(sol, tuple) else (sol,)
    return len([s for s in sols if s and s.invoked and "widen" not in s.reason])


def test_the_bar_withholds_a_solver_decided_verified(_bar_add):
    """THE POSITIVE DIRECTION, exercised for the first time."""
    build = _solver_decided_query()
    v = check(build, vacuity_mode="inputs-only", solver_timeout_ms=20000)
    assert _obl_solves(v) > 0, (
        "intervals settled it, so the bar never applies and this test does "
        "not exercise the firing direction"
    )
    assert v.status != "VERIFIED", "a solver-decided VERIFIED was not withheld"
    assert any("VERIFIED withheld" in n for n in v.notes)


def test_the_bar_finds_a_barred_primitive_nested_in_cond(_bar_add):
    """The traversal fix, at the level where it is decidable.

    The end-to-end version cannot be written: reaching the bar requires a
    SOLVER-DECIDED verdict, and a query containing `cond` can never have one,
    because `cond` is in NEITHER `TRANSFERS` NOR the emission set. So the
    under-fire this fixes was unreachable for a SECOND, independent reason
    beyond `scatter`'s absence — the enclosing construct blocks escalation
    whatever the barred set contains.

    That is worth stating precisely rather than overclaiming: the defect was
    latent, not live, and its reachability needed a collection-valued jaxpr
    param on a primitive that is emittable or transparent. `cond`/`scan`/
    `while` are collection-valued but not emittable; `jit`/`remat` are
    transparent and ARE reached, and the walk handled those correctly all
    along.

    What remains testable, and is what the fix is actually about, is that
    `_barred_primitives` SEES the primitive. Asserted directly.
    """
    def q():
        x = any_array((3,), "float64", (0.0, 1.0))
        y = jax.lax.cond(x[0] > 0.5, lambda a: (a + 1.0) - a,
                         lambda a: (a + 1.0) - a, x)
        return (assert_(y <= 1.5),)

    closed = transcribe(jax.make_jaxpr(q)())
    assert "add" in _verdict._barred_primitives(closed), (
        "a barred primitive inside a cond branch was not found — the "
        "traversal is not reaching cond's branches"
    )


def test_the_synthetic_bar_does_not_leak(_bar_add):
    """The fixture must not make everything UNKNOWN. A query with no `add`
    on it is unaffected, or the two tests above pass for the wrong reason."""
    def q():
        x = any_array((3,), "float64", (0.0, 1.0))
        return (assert_(x * 1.0 <= 2.0),)

    v = check(q, vacuity_mode="inputs-only", solver_timeout_ms=20000)
    assert not any("VERIFIED withheld" in n for n in v.notes)
