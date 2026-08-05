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
SOLVER-DECIDED VERIFIED, and a fixture reaching that branch needs a barred
primitive that can be EMITTED. `scatter` is one: measured on this build,
`"scatter" in obligation._SUPPORTED` is True, six census idioms reach the row
(`tests/test_scatter_emission_reach.py`) and `tests/test_verified_bar.py`
drives a solver-decided scatter obligation end to end. This module said the
opposite for a while — that `scatter` was absent from the emission set, so the
protective branch was unreachable and the bar's whole behavioural history was
over-firing — and that was false when it was written; the correction is worth
keeping visible, because it is the same defect this file exists to catch, one
level up: a claim of non-occurrence that nothing checked.

Monkeypatching the barred set to `{"add"}` still earns its place, for two
reasons that survive the correction. `add` appears INSIDE a sub-jaxpr on a
real slice (`scatter-add` carries `update_jaxpr`), which is what makes the
descent observable at all; and it is emittable under enclosing constructs
where the real barred set cannot be driven — see
`test_the_bar_finds_a_barred_primitive_nested_in_cond` for the one that is
genuinely unreachable end-to-end, and why.

**A THIRD, since the bar became slice-scoped: THE SLICE ROOT.** The bar now
derives the barred set of the obligation the solver actually decided, by the
SAME walk rooted at that obligation's emitted slice equations rather than at
the query. Rooting a walk somewhere new is where a hand-rolled flat version
looks adequate and is not: measured, a `scatter-add` slice equation carries
`update_jaxpr = ['add']`, so under a synthetic barred set of `{"add"}` the flat
`{str(e.primitive) for e in sl.eqns}` finds NOTHING where the canonical walk
finds `add`. That measured disagreement is this module's anti-vacuity control
for the slice-root parity test — a parity test that cannot fail proves nothing.
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

def test_the_registry_facts_this_module_argues_from():
    """THE DOCSTRING'S PREMISES, ASSERTED.

    This module's docstring used to say `scatter` was absent from the emission
    set, so the bar's protective branch was unreachable. That was false when it
    was written and nothing checked it — a claim of non-occurrence resting on a
    registry membership, in the file whose whole subject is claims that stopped
    matching their mechanism. The four memberships the corrected text argues
    from are now read out of the registries themselves.
    """
    from stelling import obligation, propagate

    assert "scatter" in obligation._SUPPORTED, (
        "`scatter` is no longer emittable. The docstring above says the bar's "
        "protective branch is reachable with the REAL barred set, and that is "
        "why the synthetic `add` fixture is a convenience rather than the only "
        "way in; re-derive before re-wording"
    )
    assert "scatter" in propagate.TRANSFERS
    assert "cond" not in obligation._SUPPORTED, (
        "`cond` became emittable, so the cond-nested under-fire may now be "
        "reachable end-to-end — see "
        "test_the_bar_finds_a_barred_primitive_nested_in_cond, whose "
        "'cannot be written' is derived from exactly this"
    )
    assert "cond" not in propagate.TRANSFERS


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
    because `cond` is in NEITHER `TRANSFERS` NOR the emission set (measured on
    this build; `scatter` is in both). The enclosing construct blocks
    escalation whatever the barred set contains, so this under-fire was
    latent even though the barred primitive itself is emittable.

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


# ---- the SLICE root: the same walk, rooted at an obligation's slice -------

def _all_primitives_in_eqns(eqns):
    """The canonical walk, rooted at an arbitrary equation iterable rather
    than at a jaxpr — `sub_jaxprs` and nothing hand-rolled."""
    found, seen = set(), []

    def walk(items):
        for e in items:
            found.add(str(e.primitive))
            for sub in sub_jaxprs(e):
                if id(sub) not in seen:
                    seen.append(id(sub))
                    walk(sub.eqns)

    walk(eqns)
    return found


def _flat_primitives_in_eqns(eqns):
    """THE NAIVE VERSION, re-implemented here so the parity test below has
    something it can actually fail against: the one-line comprehension a
    slice-rooted bar invites, which never opens a sub-jaxpr."""
    return {str(e.primitive) for e in eqns}


def _slices_of(build):
    from stelling.obligation import (
        DeclinedObligation,
        slice_unknown_obligations,
    )
    from stelling.propagate import interval_env, propagate

    closed = transcribe(jax.make_jaxpr(build)())
    p = propagate(closed)
    return [
        s for s in slice_unknown_obligations(closed, p, interval_env(closed))
        if not isinstance(s, DeclinedObligation)
    ]


def _segment_sum_relational():
    """Escalates, and its slice's `scatter-add` equation carries the
    recorded combiner sub-jaxpr — the case the flat version misses."""
    import numpy as np

    d = any_array((2,), "float64", (0.0, 1.0))
    s = jax.ops.segment_sum(
        d, jnp.asarray(np.array([0, 0], dtype=np.int32)), num_segments=1
    )
    return (assert_(s[0] >= d[0]),)


def _set_relational():
    x = any_array((3,), "float64", (0.0, 1.0))
    return (assert_(x.at[0].set(0.5)[1] - x[1] <= 0.0),)


def _plain_relational():
    x = any_array((3,), "float64", (0.0, 1.0))
    return (assert_((x + 1.0) - x <= 1.5),)


SLICE_SHAPES = [
    (_segment_sum_relational, "scatter-add slice (carries a sub-jaxpr)"),
    (_set_relational, "scatter set slice"),
    (_plain_relational, "flat arithmetic slice"),
]


@pytest.mark.parametrize("build,label", SLICE_SHAPES,
                         ids=[s[1] for s in SLICE_SHAPES])
@pytest.mark.parametrize("barred", [frozenset({"add"}), frozenset({"scatter"})],
                         ids=["synthetic add", "real scatter"])
def test_slice_root_walk_has_parity_with_the_canonical_accessor(
    monkeypatch, build, label, barred
):
    """The slice-rooted root of the bar's walk must see every primitive
    `sub_jaxprs` sees from the SAME equations — checked over the barred set's
    own membership, which is what the bar acts on, and under both the real
    barred set and a synthetic one that reaches inside a sub-jaxpr."""
    monkeypatch.setattr(_verdict, "VERIFIED_BARRED_PRIMITIVES", barred)
    slices = _slices_of(build)
    assert slices, f"{label}: nothing escalated, so no slice exists to root at"
    for sl in slices:
        canonical = _all_primitives_in_eqns(sl.eqns)
        got = set(_verdict._barred_in_eqns(sl.eqns))
        assert got == canonical & barred, (
            f"{label}: the slice-rooted walk found {sorted(got)} where the "
            f"canonical walk implies {sorted(canonical & barred)} — the two "
            f"traversals have diverged"
        )


def test_the_slice_parity_test_catches_the_naive_flat_version(monkeypatch):
    """ANTI-VACUITY (Norm C) for the slice root, on the MEASURED case.

    The flat comprehension is the implementation a slice-scoped bar invites,
    and on a `scatter-add` slice under a barred set of `{"add"}` it is wrong:
    the combiner lives in the equation's `update_jaxpr`, not in the slice's
    own equation list. If it AGREED here, the parity test above could not
    distinguish the two and would prove nothing.
    """
    barred = frozenset({"add"})
    monkeypatch.setattr(_verdict, "VERIFIED_BARRED_PRIMITIVES", barred)
    slices = _slices_of(_segment_sum_relational)
    assert slices, "nothing escalated; the control has no slice to run on"
    sl = slices[0]

    canonical = _all_primitives_in_eqns(sl.eqns) & barred
    assert canonical, (
        "the fixture's slice carries no barred primitive at any depth; the "
        "control is vacuous"
    )
    flat = _flat_primitives_in_eqns(sl.eqns) & barred
    assert flat != canonical, (
        f"the NAIVE flat version agrees with the canonical walk on this "
        f"slice ({sorted(flat)}), so the parity test above cannot "
        f"distinguish them and proves nothing"
    )
    assert set(_verdict._barred_in_eqns(sl.eqns)) == canonical


@pytest.mark.parametrize("build,label", SLICE_SHAPES,
                         ids=[s[1] for s in SLICE_SHAPES])
def test_the_bar_re_derives_the_slice_that_was_actually_emitted(
    monkeypatch, build, label
):
    """THE SEAM THE DERIVED SCOPE RESTS ON, asserted rather than argued.

    `verdict._bar_scope` does not read a scope off the escalation — a record
    cannot certify its own cleanliness, and nothing binds an escalation to the
    query it is stamped against. It re-slices the decided obligations out of
    `closed` instead. That is only sound if the re-derivation is the SAME
    slice `escalate` emitted, and it is by construction:
    `slice_unknown_obligations` calls `slice_obligation(closed, index,
    interval_env(closed))`, and its one further argument is documented
    "message wording only, never admission".

    "By construction" is exactly the kind of claim that stops being true
    quietly, so it is measured here on every slice shape, under a synthetic
    barred set that reaches into a sub-jaxpr as well as the real one.
    """
    from stelling.obligation import (
        DeclinedObligation,
        slice_obligation,
        slice_unknown_obligations,
    )
    from stelling.propagate import interval_env, propagate

    monkeypatch.setattr(_verdict, "VERIFIED_BARRED_PRIMITIVES",
                        frozenset({"add", "scatter"}))
    closed = transcribe(jax.make_jaxpr(build)())
    p = propagate(closed)
    env = interval_env(closed)
    emitted = [s for s in slice_unknown_obligations(closed, p, env)
               if not isinstance(s, DeclinedObligation)]
    assert emitted, f"{label}: nothing escalated, so nothing is being pinned"
    for sl in emitted:
        again = slice_obligation(closed, sl.index, interval_env(closed))
        assert not isinstance(again, DeclinedObligation), (
            f"{label}: assert #{sl.index} sliced for escalation but DECLINES "
            f"on re-derivation ({again.reason}) — the bar would fall back to "
            f"the whole query on a verdict that has a slice"
        )
        assert _verdict._barred_in_eqns(again.eqns) == (
            _verdict._barred_in_eqns(sl.eqns)
        ), (
            f"{label}: the bar's re-derived slice for assert #{sl.index} "
            f"carries a different barred set than the one that was emitted — "
            f"the scope is no longer measuring the slice the solver saw"
        )


def test_the_re_derivation_is_not_vacuous(monkeypatch):
    """ANTI-VACUITY (Norm C) for the agreement above: at least one of those
    slices must actually CARRY a barred primitive, or the parity is between
    two empty sets and would hold under any re-derivation at all."""
    from stelling.obligation import (
        DeclinedObligation,
        slice_unknown_obligations,
    )
    from stelling.propagate import interval_env, propagate

    monkeypatch.setattr(_verdict, "VERIFIED_BARRED_PRIMITIVES",
                        frozenset({"add", "scatter"}))
    found = set()
    for build, _label in SLICE_SHAPES:
        closed = transcribe(jax.make_jaxpr(build)())
        p = propagate(closed)
        for sl in slice_unknown_obligations(closed, p, interval_env(closed)):
            if not isinstance(sl, DeclinedObligation):
                found.update(_verdict._barred_in_eqns(sl.eqns))
    assert found, (
        "no slice in SLICE_SHAPES carries a barred primitive under the "
        "synthetic set, so the agreement test compares empty against empty"
    )
