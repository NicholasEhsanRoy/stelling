# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""A zero-element assumed predicate constrains nothing — the wrong VERIFIED.

Measured on `main` at `c20f38e`, through the public API with two
`any_array` calls::

    k = any_array((),   "float64", (-1.0, 1.0))
    z = any_array((0,), "float64", (-1.0, 1.0))      # SIZE-0
    assume((k >= 0.5) & (z >= 2.0))
    return (assert_(k > 0.0),)                       # -> VERIFIED

`z >= 2.0` is `bool[0]`, so jax broadcasts the whole `&` to `bool[0]`: the
assumed predicate is a universal over no elements, true at every point,
and it admits the ENTIRE declared box. But the `and` recursion classified
`k >= 0.5` as if it stood alone and narrowed `k` to [0.5, 1.0] — a
SUBSET of its declared [-1, 1] — then discharged `k > 0` over it. An
independent dense sampling of the declared box found 100 348 of 200 000
admitted points violating the assert; `k = -1.0` is admitted by the
declaration, admitted by the assume, and violates.

The one-sidedness the design rests on — VERIFIED over a superset is a
VERIFIED over the subset — does not hold when the region went the other
way, and the run's own note said "a superset" while it had.

The rule these tests pin: `all(A & B)` implies `all(A)` only if every
element of `A` survives the broadcast into the output, which over
numpy/jax broadcasting fails exactly when the output has zero elements
and `A` does not. So a predicate node with zero elements licenses no
narrowing, no satisfiability claim, and no unsatisfiable-precondition
refusal — at the root, at every `and` node, and at every leaf.

Every test here carries a positive control in the same body: the same
shape with a NON-size-0 sibling must still narrow and still reach
VERIFIED, so an inert propagation cannot pass this file.
"""
from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")  # zero-dep CI has no jax

from stelling import propagate as P  # noqa: E402
from stelling.harness import any_array, assert_, assume, trace  # noqa: E402
from stelling.preconditions import check  # noqa: E402

BOX = (-1.0, 1.0)


@pytest.fixture(autouse=True)
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def _reported():
    """The measured construction, verbatim."""
    k = any_array((), "float64", BOX)
    z = any_array((0,), "float64", BOX)
    assume((k >= 0.5) & (z >= 2.0))
    return (assert_(k > 0.0),)


def _positive_control():
    """The SAME shape with a non-size-0 sibling: this one must stay
    VERIFIED. Without it every assertion below would also pass on a
    propagation that had stopped constraining altogether."""
    k = any_array((), "float64", BOX)
    v = any_array((3,), "float64", BOX)
    assume((k >= 0.5) & (v >= -1.0))
    return (assert_(k > 0.0),)


# --- the class ---------------------------------------------------------------


def _size0_right():
    k = any_array((), "float64", BOX)
    z = any_array((0,), "float64", BOX)
    assume((k >= 0.5) & (z >= 2.0))
    return (assert_(k > 0.0),)


def _size0_left():
    k = any_array((), "float64", BOX)
    z = any_array((0,), "float64", BOX)
    assume((z >= 2.0) & (k >= 0.5))
    return (assert_(k > 0.0),)


def _nested_left():
    k = any_array((), "float64", BOX)
    m = any_array((), "float64", BOX)
    z = any_array((0,), "float64", BOX)
    assume(((k >= 0.5) & (m >= 0.5)) & (z >= 2.0))
    return (assert_(k > 0.0), assert_(m > 0.0))


def _nested_right():
    k = any_array((), "float64", BOX)
    m = any_array((), "float64", BOX)
    z = any_array((0,), "float64", BOX)
    assume((k >= 0.5) & ((m >= 0.5) & (z >= 2.0)))
    return (assert_(k > 0.0), assert_(m > 0.0))


def _mixed_with_or():
    k = any_array((), "float64", BOX)
    z = any_array((0,), "float64", BOX)
    assume((k >= 0.5) & ((z >= 2.0) | (z <= 0.0)))
    return (assert_(k > 0.0),)


def _shape1_against_size0():
    w = any_array((1,), "float64", BOX)
    z = any_array((0,), "float64", BOX)
    assume((w >= 0.5) & (z >= 2.0))
    return (assert_(w > 0.0),)


def _rank2_unit_axis_against_zero_axis():
    a = any_array((2, 1), "float64", BOX)
    z = any_array((2, 0), "float64", BOX)
    assume((a >= 0.5) & (z >= 2.0))
    return (assert_(a > 0.0),)


def _size0_from_a_comparison_of_nonzero_operands():
    # the size-0 conjunct is not a size-0 DECLARATION: `k >= z` compares a
    # rank-0 against a size-0, and the comparison itself is bool[0]
    k = any_array((), "float64", BOX)
    m = any_array((), "float64", BOX)
    z = any_array((0,), "float64", BOX)
    assume((m >= 0.5) & (k >= z))
    return (assert_(m > 0.0),)


def _eq_narrowing():
    k = any_array((), "float64", BOX)
    z = any_array((0,), "float64", BOX)
    assume((k == 0.5) & (z >= 2.0))
    return (assert_(k > 0.0),)


def _le_narrowing():
    k = any_array((), "float64", BOX)
    z = any_array((0,), "float64", BOX)
    assume((k <= -0.5) & (z >= 2.0))
    return (assert_(k < 0.0),)


def _inside_a_cond_branch():
    # the branch input truly ranges over (-0.5, 1]; the bool[0] assume
    # narrowed it to [0.5, 1] and `y > 0` then discharged. Independent
    # sampling: 5001 of 20005 admitted points violate — every k in
    # (-0.5, 0], which the taken branch returns unchanged.
    k = any_array((), "float64", BOX)
    z = any_array((0,), "float64", BOX)

    def taken(a):
        assume((a >= 0.5) & (z >= 2.0))
        return a

    y = jax.lax.cond(k > -0.5, taken, lambda a: a * 0.0 + 5.0, k)
    return (assert_(y > 0.0),)


# Each entry is a harness whose assume admits the WHOLE declared box (its
# predicate is bool[0]) while a conjunct, read alone, would cut it. Every
# one of them returned VERIFIED before the gate.
SUBSET_NARROWINGS = [
    _size0_right,
    _size0_left,
    _nested_left,
    _nested_right,
    _mixed_with_or,
    _shape1_against_size0,
    _rank2_unit_axis_against_zero_axis,
    _size0_from_a_comparison_of_nonzero_operands,
    _eq_narrowing,
    _le_narrowing,
    _inside_a_cond_branch,
]


@pytest.mark.parametrize("h", SUBSET_NARROWINGS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("mode", ["inputs-only", "all"])
@pytest.mark.parametrize("refine", [None, "affine"])
def test_a_vacuous_assume_never_reaches_VERIFIED(h, mode, refine):
    assert check(h, vacuity_mode=mode, refine=refine).status != "VERIFIED"
    # the positive control, in the same body: an assume that DOES
    # constrain still gets there, on the same mode and the same depth
    assert check(
        _positive_control, vacuity_mode=mode, refine=refine
    ).status == "VERIFIED"


@pytest.mark.parametrize("h", SUBSET_NARROWINGS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("semantics", ["real", "ieee"])
def test_no_obligation_discharges_under_a_vacuous_assume(h, semantics):
    p = P.propagate(trace(h), semantics=semantics)
    assert [o.status for o in p.obligations] != []
    assert all(o.status != "discharged" for o in p.obligations)
    ctrl = P.propagate(trace(_positive_control), semantics=semantics)
    assert [o.status for o in ctrl.obligations] == ["discharged"]


def test_the_declared_box_is_never_narrowed_to_a_subset():
    """The soundness statement itself, read off the environment: after a
    bool[0] assume every declared variable still spans its declaration."""
    env = P.interval_env(trace(_reported), assume_mode="constrain")
    k = env[1]
    assert (k.los[0], k.his[0]) == BOX
    # control: the same read on a constraining assume DOES move
    cenv = P.interval_env(trace(_positive_control), assume_mode="constrain")
    assert (cenv[1].los[0], cenv[1].his[0]) == (0.5, 1.0)


def test_no_narrowing_note_and_the_reason_names_the_zero_elements():
    p = P.propagate(trace(_reported))
    assert not any(n.startswith("assume CONSTRAINED") for n in p.notes)
    assert p.coverage.constrained == 0 and p.coverage.inert == 1
    assert any("zero elements" in n for n in p.notes)
    # and the note no longer claims a superset it did not take
    assert not any("a superset (" in n for n in p.notes)
    ctrl = P.propagate(trace(_positive_control))
    assert any(n.startswith("assume CONSTRAINED") for n in ctrl.notes)


# --- the opposite-direction face: a false harness-defect refusal --------------


def _empty_meet_sibling():
    # k >= 2.0 is impossible on [-1, 1] — but the assume it sits in is
    # bool[0] and therefore true at every point of the declared box, so
    # the precondition is SATISFIABLE and the loud refusal would be false
    k = any_array((), "float64", BOX)
    z = any_array((0,), "float64", BOX)
    assume((k >= 2.0) & (z >= 2.0))
    return (assert_(k > 0.0),)


def _strict_collapse_sibling():
    k = any_array((), "float64", BOX)
    z = any_array((0,), "float64", BOX)
    assume((k > 1.0) & (z >= 2.0))       # (1, 1] — the collapse refusal
    return (assert_(k > 0.0),)


@pytest.mark.parametrize(
    "h", [_empty_meet_sibling, _strict_collapse_sibling],
    ids=["empty_meet", "strict_collapse"],
)
def test_a_satisfiable_vacuous_assume_is_not_called_a_harness_defect(h):
    assert check(h, vacuity_mode="inputs-only").status == "UNKNOWN"
    # control: the same conjunct WITHOUT the size-0 sibling is a genuine
    # unsatisfiable precondition and still raises
    def alone():
        k = any_array((), "float64", BOX)
        assume(k >= 2.0)
        return (assert_(k > 0.0),)

    with pytest.raises(P.UnsatisfiableAssumptionError):
        check(alone, vacuity_mode="inputs-only")
