# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""AUDIT 0.2.0 M17 — one `assert_` nested in a `jit` may not decline solver
escalation for EVERY obligation in the query.

`slice_unknown_obligations` compared the number of TOP-LEVEL
`stelling_assert` equations against `len(propagation.obligations)` and, on
a mismatch, declined every unknown obligation. An `assert_` inside any
transparent call is counted by the propagation and not by the top-level
scan, so writing one obligation inside a `@jax.jit` helper silently cost
solver escalation for all the others.

Measured on `main` (`dee8bc2`), jax 0.11.0, `JAX_ENABLE_X64=1`, z3 + cvc5
wheels:

    == a_alone          -> VERIFIED   #0 discharged
    == b_alone          -> VERIFIED   #0 discharged
    == both_top_level   -> VERIFIED   #0 discharged  #1 discharged
    == one_nested       -> UNKNOWN
       #0 unknown  escalation declined: 2 obligation(s) but 1 top-level
                   stelling_assert equation(s): asserts nested in
                   sub-jaxprs cannot be mapped to slices
       #1 unknown  (same)

`one_nested` states exactly what `a_alone` and `b_alone` state; only the
second is written inside a jit. Obligation #0 is an ordinary top-level
assert with an ordinary slice and was being thrown away with its sibling.

WHAT THIS BATCH DID NOT DO, stated so the tests below are read for what
they are: an `assert_` inside a sub-jaxpr is STILL not sliceable. Escalation
slices top-level asserts, `obligation.py`'s module docstring scopes it that
way, and lifting that is a capability change (and, for a `cond` branch, a
soundness question — a branch assert is conditional). So `one_nested` still
reaches UNKNOWN as a QUERY; what changed is that its top-level obligation is
now decided and the decline is the nested obligation's alone, with a reason
that names the actual cause.
"""
from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402

from stelling import _optional  # noqa: E402
from stelling.harness import any_array, assert_, trace  # noqa: E402
from stelling.obligation import (  # noqa: E402
    DeclinedObligation,
    slice_unknown_obligations,
)
from stelling.preconditions import check  # noqa: E402
from stelling.propagate import interval_env, propagate  # noqa: E402

HAVE_Z3 = _optional.available("z3")
HAVE_CVC5 = _optional.available("cvc5") or _optional.cvc5_binary() is not None

need_both = pytest.mark.skipif(
    not (HAVE_Z3 and HAVE_CVC5), reason="needs both z3 and cvc5"
)


@pytest.fixture(autouse=True, scope="module")
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


# ── the audit's five harnesses, verbatim in substance ────────────────────


def a_alone():
    c = any_array((2,), jnp.float64, (100.0, 101.0))
    return assert_(c * c - c > 9899.0)          # true; interval-undecided


def b_alone():
    a = any_array((2,), jnp.float64, (0.0, 1.0))
    return assert_(a - a * a >= 0.0)            # true; interval-undecided


def both_top_level():
    a = any_array((2,), jnp.float64, (0.0, 1.0))
    c = any_array((2,), jnp.float64, (100.0, 101.0))
    return (assert_(c * c - c > 9899.0), assert_(a - a * a >= 0.0))


@jax.jit
def _inner(x):
    return assert_(x - x * x >= 0.0)


def one_nested():
    """Identical obligations; the SECOND one is inside a jit."""
    a = any_array((2,), jnp.float64, (0.0, 1.0))
    c = any_array((2,), jnp.float64, (100.0, 101.0))
    return (assert_(c * c - c > 9899.0), _inner(a))


def nested_first():
    """The nested assert comes FIRST, so the top-level one is obligation #1
    while being top-level assert #0. Under the old count check this whole
    query declined; under an index-is-the-position reading it would slice
    the wrong equation. The carried position is what makes it right."""
    a = any_array((2,), jnp.float64, (0.0, 1.0))
    c = any_array((2,), jnp.float64, (100.0, 101.0))
    return (_inner(a), assert_(c * c - c > 9899.0))


def budget_is_per_obligation():
    big = any_array((600,), jnp.float64, (0.1, 2.0))
    a = any_array((2,), jnp.float64, (0.0, 1.0))
    c = any_array((2,), jnp.float64, (100.0, 101.0))
    return (
        assert_(jnp.sum(big) - 60.0 > 0.0),     # over ELEMENT_BUDGET
        assert_(c * c - c > 9899.0),
        assert_(a - a * a >= 0.0),
    )


def _verdict(h):
    return check(h, vacuity_mode="inputs-only", solver_timeout_ms=20_000)


# ── the four rows the audit tabulated ────────────────────────────────────


@need_both
@pytest.mark.parametrize("h", [a_alone, b_alone], ids=["a_alone", "b_alone"])
def test_each_obligation_passes_on_its_own(h):
    """The control the audit's table rests on: both obligations are
    individually VERIFIED by escalation, so any UNKNOWN below is about the
    mapping and not about the obligations."""
    v = _verdict(h)
    assert v.status == "VERIFIED", [
        (o.index, o.status, o.detail) for o in v.obligations
    ]
    assert all(o.status == "discharged" for o in v.obligations)


@need_both
def test_both_at_the_top_level_are_both_discharged():
    v = _verdict(both_top_level)
    assert v.status == "VERIFIED"
    assert [o.status for o in v.obligations] == ["discharged", "discharged"]


@need_both
@pytest.mark.parametrize(
    "h,top_level_index", [(one_nested, 0), (nested_first, 1)],
    ids=["nested second", "nested first"],
)
def test_a_nested_assert_no_longer_declines_its_SIBLINGS(h, top_level_index):
    """THE DEFECT. Before: both obligations `unknown`, both quoting the
    count mismatch. After: the top-level obligation is discharged by
    escalation exactly as it is when written alone, and only the nested one
    declines.

    `nested_first` is the ordering that matters most: the top-level assert
    is obligation #1 but top-level assert #0, so a fix that kept indexing
    `asserts[o.index]` would slice out of range or slice the wrong equation.
    """
    v = _verdict(h)
    statuses = {o.index: o.status for o in v.obligations}
    assert len(statuses) == 2, statuses
    assert statuses[top_level_index] == "discharged", (
        "the top-level obligation was declined along with its nested "
        f"sibling: {[(o.index, o.detail) for o in v.obligations]}"
    )
    nested_index = 1 - top_level_index
    assert statuses[nested_index] == "unknown"
    # ...and NOT for the old reason
    (nested,) = [o for o in v.obligations if o.index == nested_index]
    assert "top-level stelling_assert equation(s)" not in nested.detail
    assert "not a top-level equation of the query" in nested.detail


@need_both
def test_the_element_budget_is_still_strictly_per_obligation():
    """The audit's control row, unchanged: an over-budget assert leaves its
    siblings escalating normally. It was green before this fix and must stay
    green — it is what establishes that the budget was never the cause of
    the reported cliff."""
    v = _verdict(budget_is_per_obligation)
    statuses = [o.status for o in v.obligations]
    assert statuses == ["unknown", "discharged", "discharged"], [
        (o.index, o.status, o.detail[:90]) for o in v.obligations
    ]
    assert "element terms" in v.obligations[0].detail


# ── the association itself, without a solver ─────────────────────────────


def test_the_walk_records_where_it_saw_each_assert():
    """The carried association, read directly. A top-level assert records
    its position in the top-level `eqns`; a nested one records `None`."""
    closed = trace(one_nested)
    p = propagate(closed)
    positions = [o.top_level_eqn_pos for o in p.obligations]
    assert positions[1] is None, "the jit-nested assert claimed a position"
    pos = positions[0]
    assert pos is not None
    assert closed.jaxpr.eqns[pos].primitive == "stelling_assert"
    assert tuple(closed.jaxpr.eqns[pos].source_info) == tuple(
        p.obligations[0].source_info
    )


def test_an_assert_inside_a_scan_cannot_be_re_associated_and_declines():
    """The other genuinely-unmappable shape, and a different mechanism from
    the jit one: propagation does not descend `scan` at all, so this
    obligation is recorded by `_record_unexamined` rather than by the walk.
    It carries no position either, and declines individually."""
    def with_scan():
        x = any_array((2,), jnp.float64, (0.0, 1.0))

        def body(carry, _):
            assert_(carry >= -1.0)
            return carry, 0.0

        jax.lax.scan(body, x, jnp.zeros((3,)))
        return assert_(x - x * x >= 0.0)

    closed = trace(with_scan)
    p = propagate(closed)
    assert len(p.obligations) == 2
    by_pos = {o.index: o.top_level_eqn_pos for o in p.obligations}
    assert sorted(by_pos.values(), key=lambda v: (v is None, v))[-1] is None, (
        "no obligation came back unmappable; the fixture measures nothing"
    )
    items = slice_unknown_obligations(closed, p, interval_env(closed))
    declined = [i for i in items if isinstance(i, DeclinedObligation)]
    sliced = [i for i in items if not isinstance(i, DeclinedObligation)]
    assert declined, "the unexamined scan obligation was sliced"
    assert sliced, "the top-level obligation declined with it (the M17 cliff)"


def two_other_asserts():
    """A second two-obligation query, deliberately the SAME SHAPE as
    `both_top_level` — same count of top-level asserts, so the whole-query
    count check this replaced would have waved it through."""
    a = any_array((2,), jnp.float64, (0.0, 1.0))
    c = any_array((2,), jnp.float64, (100.0, 101.0))
    return (assert_(a * a - a <= 0.0), assert_(c - c * c < -9899.0))


def test_an_obligation_whose_association_cannot_be_trusted_still_declines():
    """THE SAFETY PROPERTY, attacked where the replaced check was blind.

    The carried position is a claim about a PARTICULAR query. Hand
    `slice_unknown_obligations` a different one with the same number of
    top-level asserts at the same positions and every obligation must
    decline rather than be sliced out of a jaxpr it was never judged
    against.

    The count check could not see this: 2 == 2, so it proceeded and sliced
    `asserts[o.index]` out of the wrong query. The per-obligation mapping is
    therefore not merely finer than what it replaced, it is strictly
    stronger — which is the claim this test exists to make good.
    """
    a = trace(both_top_level)
    b = trace(two_other_asserts)
    p_a = propagate(a)
    assert len(
        [e for e in a.jaxpr.eqns if e.primitive == "stelling_assert"]
    ) == len([e for e in b.jaxpr.eqns if e.primitive == "stelling_assert"]), (
        "the two fixtures differ in assert count, so the old count check "
        "would have caught this and the test measures nothing"
    )
    # THE COUNTERFACTUAL, MEASURED: `slice_obligation(b, index, env_b)` is
    # verbatim what the count check went on to do once the totals matched,
    # and it produces a real slice out of the wrong query.
    from stelling.obligation import slice_obligation

    would_have = slice_obligation(b, 0, interval_env(b))
    assert not isinstance(would_have, DeclinedObligation), (
        "the wrong-query slice declines on its own, so this fixture does "
        "not demonstrate the gap the carried association closes"
    )

    items = slice_unknown_obligations(b, p_a, interval_env(b))
    assert items, "no unknown obligations; the fixture measures nothing"
    assert all(isinstance(i, DeclinedObligation) for i in items), [
        type(i).__name__ for i in items
    ]
    assert all("disagree" in i.reason for i in items), [
        i.reason for i in items
    ]
