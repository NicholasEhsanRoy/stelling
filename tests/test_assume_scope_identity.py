# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Scope-correct identity for forwarded relational assumes (audit 0.2.0 S5).

THE DEFECT. A relational ``assume`` traced inside a ``jit`` / ``custom_jvp``
body was recorded by its producing comparison equation, whose operands carry
THAT BODY'S var ids. ``smt.emit`` resolved them with ``names.get(atom.id)`` —
a bare integer lookup against the SLICE's renumbered table, with no scope
check — and the slicer allocates its fresh ids from ``max(top-level ids) + 1``,
a maximum that never sees a sub-jaxpr's ids. When the two ranges met, a
foreign id resolved to an unrelated term and the axiom was emitted about the
wrong values: measured as the CONVERSE of the user's own precondition,
discharging an obligation false at every admitted point.

WHAT THESE TESTS PIN, and why they are shaped this way. The id window is ONE
STATEMENT WIDE — the audit's reachability sweep shows a single extra top-level
statement closes it — so a test that pins one brittle harness's verdict rots
the moment anything shifts. Three shapes are used instead:

1. DENOTATION, checked against DECLARATION ORDER rather than against a number
   (``test_a_carried_assume_names_the_declarations_it_was_written_about``).
   The assume is written with its operands SWAPPED relative to the assert, so
   a resolution that lands on the right values by luck is not available: the
   pair must come back in the written order.
2. CONTAINMENT: every operand of every carried assume must be something this
   slice actually holds a term for — an input, a constant, or one of its own
   equations' outputs. An id from another scope cannot satisfy this.
3. STABILITY ACROSS THE WINDOW: the same query with 0..3 trailing statements
   must return the same verdict. The defect's signature is exactly that it
   does not.

Plus the two arms of audit M6 (the arity crash and the silent truncation) and
the branch-scoping fix (audit S5-B1).
"""

from __future__ import annotations

from fractions import Fraction as F

import pytest

from stelling import ir

pytest.importorskip("jax", reason="needs jax")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402

from stelling import smt  # noqa: E402
from stelling.harness import any_array, assert_, assume, trace  # noqa: E402
from stelling.obligation import (  # noqa: E402
    ObligationSlice,
    SliceAssume,
    slice_unknown_obligations,
)
from stelling.propagate import interval_env, propagate  # noqa: E402

try:
    from stelling import _optional
    HAVE_SOLVER = (
        _optional.available("z3")
        or _optional.available("cvc5")
        or _optional.cvc5_binary() is not None
    )
except Exception:  # pragma: no cover - environment probe only
    HAVE_SOLVER = False

need_solver = pytest.mark.skipif(not HAVE_SOLVER, reason="needs an SMT solver")

LO, HI = -10.0, 10.0


# ---------------------------------------------------------------------------
# harnesses: the same precondition, carried in four different scopes
# ---------------------------------------------------------------------------


def _swapped_harness(carrier: str, tail: int = 0):
    """assert(x - y <= 0) with the precondition `y < x` stated in `carrier`.

    Under `y < x` the assert is false at EVERY admitted point, so the only
    sound verdicts are REFUTED (with an admitted witness) or UNKNOWN. The
    operands are written SWAPPED — the assume's left operand is declaration
    #1 and its right is declaration #0 — which is what makes a wrong
    resolution visible rather than merely wrong.
    """

    def harness():
        x = any_array((), "float64", (LO, HI))
        y = any_array((), "float64", (LO, HI))
        r = assert_(x - y <= 0.0)

        if carrier == "top":
            assume(y < x)
        elif carrier == "jit":
            @jax.jit
            def side(p, q):
                assume(p < q)
                return p + q
            side(y, x)
        elif carrier == "custom_jvp":
            @jax.custom_jvp
            def side(p, q):
                assume(p < q)
                return p + q

            @side.defjvp
            def side_jvp(primals, tangents):
                a, b = primals
                da, db = tangents
                return side(a, b), da + db
            side(y, x)
        elif carrier == "cond":
            lax.cond(
                x > 0.0,
                lambda p, q: (assume(p < q), p + q)[1],
                lambda p, q: p - q,
                y, x,
            )
        else:  # pragma: no cover
            raise AssertionError(carrier)

        for k in range(tail):
            _ = x * (1.0 + k)
        return r

    return harness


def _inner_intermediate_harness(carrier: str, tail: int = 0):
    """The DISCRIMINATING shape: the assume's operands are INTERMEDIATES of
    the body, not its arguments.

    An argument-operand assume can be resolved correctly by accident — a
    body's invars are numbered first, so the slicer's fresh ids often land on
    them one-for-one and a bare integer lookup reaches the right alias.
    Intermediates are numbered after the body's own outvar, so the identity
    coincidence is gone: only a rename that knows which scope the id came from
    lands on the right term.

    The body is the one that produces the assert's value, so its equations ARE
    in the backward cone. Precondition ``3y < 2x``; assert ``2x - 3y <= 0``,
    which is false at every admitted point.
    """

    def harness():
        x = any_array((), "float64", (LO, HI))
        y = any_array((), "float64", (LO, HI))

        def body(a, b):
            t = a * 2.0
            u = b * 3.0
            assume(u < t)
            return t - u

        if carrier == "jit":
            step = jax.jit(body)
        elif carrier == "custom_jvp":
            step = jax.custom_jvp(body)

            @step.defjvp
            def step_jvp(primals, tangents):
                a, b = primals
                da, db = tangents
                return step(a, b), 2.0 * da - 3.0 * db
        else:  # pragma: no cover
            raise AssertionError(carrier)

        r = assert_(step(x, y) <= 0.0)
        for k in range(tail):
            _ = x * (1.0 + k)
        return r

    return harness


def _mul_outvar_by_literal(sl, k: float):
    """The outvar of the slice's ``mul`` whose literal operand is ``k``.

    Identifies a term by what the PROGRAM says about it, never by an id, so
    a renumbering cannot make the expectation move with the answer."""
    from stelling.obligation import _decode_elements

    found = [
        e.outvars[0].id
        for e in sl.eqns
        if e.primitive == "mul"
        and any(
            isinstance(a, ir.Literal) and _decode_elements(a.val) == (k,)
            for a in e.invars
        )
    ]
    assert len(found) == 1, (k, found, [e.primitive for e in sl.eqns])
    return found[0]


def _one_slice(harness):
    closed = trace(harness)
    p = propagate(closed)
    env = interval_env(closed)
    items = slice_unknown_obligations(closed, p, env)
    assert len(items) == 1, items
    return p, items[0]


def _axiom_lines(text: str) -> list[str]:
    """The RELATIONAL axiom lines: the assert lines that are neither a
    declared bound nor the negated obligation."""
    return [
        line for line in text.splitlines()
        if line.startswith("(assert (")
        and "(not " not in line
        and not line.startswith("(assert (<= (- 10.0) x")
        and not line.startswith("(assert (<= x")
    ]


# ---------------------------------------------------------------------------
# 1. denotation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("carrier", ["top", "jit", "custom_jvp"])
def test_a_carried_assume_names_the_declarations_it_was_written_about(carrier):
    """THE INVARIANT, checked against declaration order, not against ids.

    ``sl.inputs`` is in DECLARATION order, so ``inputs[0].var_id`` is x and
    ``inputs[1].var_id`` is y whatever the transcriber numbered them. The
    precondition is written ``y < x``, so the carried assume's operands must
    be (y, x) — in that order. A resolution that landed on any other pair, or
    on the right pair in the wrong order, fails here; and because the check is
    stated in declaration indices, a renumbering that shifts every id cannot
    make it pass by accident.
    """
    p, sl = _one_slice(_swapped_harness(carrier))
    assert isinstance(sl, ObligationSlice), sl
    assert len(p.relational_assumes) == 1
    assert len(sl.assumes) == 1, sl.assumes_skipped
    assert sl.assumes_skipped == ()
    sa = sl.assumes[0]
    assert sa.primitive == "lt"
    by_decl = [inp.var_id for inp in sl.inputs]
    assert len(by_decl) == 2
    lhs, rhs = sa.invars
    assert isinstance(lhs, ir.Var) and isinstance(rhs, ir.Var)
    assert lhs.id == by_decl[1], (
        f"the assume's LEFT operand must denote declaration #1 (y); got a "
        f"var that is not it — {lhs.id} vs {by_decl}"
    )
    assert rhs.id == by_decl[0], (
        f"the assume's RIGHT operand must denote declaration #0 (x); got a "
        f"var that is not it — {rhs.id} vs {by_decl}"
    )
    # and the script says the same thing, in the emitted names
    script = smt.emit(sl, "z3", 5000)
    assert script.relational_assumes_emitted == 1
    assert _axiom_lines(script.text) == ["(assert (< x1 x0))"], script.text


@pytest.mark.parametrize("carrier", ["jit", "custom_jvp"])
@pytest.mark.parametrize("tail", [0, 1, 2, 3])
def test_a_carried_assume_names_the_inner_intermediates_it_denotes(carrier, tail):
    """THE INVARIANT on the shape where a bare integer lookup CANNOT be right
    by accident (see :func:`_inner_intermediate_harness`).

    The precondition is ``u < t`` with ``t = 2x`` and ``u = 3y``; the carried
    assume's operands must therefore be the slice terms for those two
    products, identified by the literal each ``mul`` carries rather than by
    any id. Walked across the whole one-statement id window, because the
    invariant is not a fact about one alignment."""
    p, sl = _one_slice(_inner_intermediate_harness(carrier, tail))
    assert isinstance(sl, ObligationSlice), sl
    assert len(p.relational_assumes) == 1
    assert len(sl.assumes) == 1, sl.assumes_skipped
    sa = sl.assumes[0]
    assert sa.primitive == "lt"
    lhs, rhs = sa.invars
    assert isinstance(lhs, ir.Var) and isinstance(rhs, ir.Var)
    assert lhs.id == _mul_outvar_by_literal(sl, 3.0), (
        "the assume's LEFT operand is `u = 3*y`; the axiom names something "
        "else"
    )
    assert rhs.id == _mul_outvar_by_literal(sl, 2.0), (
        "the assume's RIGHT operand is `t = 2*x`; the axiom names something "
        "else"
    )
    text = smt.emit(sl, "z3", 5000).text
    assert _axiom_lines(text) == [
        f"(assert (< t{lhs.id} t{rhs.id}))"
    ], text


# ---------------------------------------------------------------------------
# 2. containment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("carrier", ["top", "jit", "custom_jvp", "cond"])
@pytest.mark.parametrize("tail", [0, 1, 2, 3])
def test_no_carried_assume_names_anything_this_slice_does_not_hold(carrier, tail):
    """Every operand of every carried assume is a term this slice HAS.

    The general form of the invariant: a slice's named ids are exactly its
    declared inputs, its constants, and its own equations' outputs. An id from
    a scope the slice did not inline is in none of those sets — which is the
    whole of the defect, stated as a property rather than as a verdict."""
    p, sl = _one_slice(_swapped_harness(carrier, tail))
    assert isinstance(sl, ObligationSlice), (
        f"this shape must slice; it declined with: "
        f"{getattr(sl, 'reason', sl)!r}"
    )
    named = {inp.var_id for inp in sl.inputs}
    named |= {vid for vid, _ in sl.consts}
    named |= {out.id for e in sl.eqns for out in e.outvars}
    for sa in sl.assumes:
        for atom in sa.invars:
            if isinstance(atom, ir.Var):
                assert atom.id in named, (
                    f"a carried assume names var {atom.id}, which this slice "
                    f"holds no term for — the forged-axiom shape"
                )
    # the partition is exact, so "emitted vs requested" is derivable from the
    # slice alone — the contract the un-withholding rule reads
    assert (
        len(sl.assumes) + len(sl.assumes_skipped) == len(p.relational_assumes)
    )
    assert smt.emit(sl, "z3", 5000).relational_assumes_emitted == len(sl.assumes)


def test_one_callee_invoked_TWICE_gets_two_DISTINCT_correct_axioms():
    """THE CASE THAT SEPARATES THE TWO CANDIDATE DESIGNS, and the reason the
    identity is a scope PATH rather than the assume equation's object.

    jax caches a jitted callee's jaxpr, so two call sites of one callee embed
    the SAME sub-jaxpr object and its vars receive the SAME ir ids at both
    sites (measured here: both assumes are recorded over the same two ids).
    Two consequences:

    * a bare integer lookup has one answer for two different pairs of values,
      so at most one of the two axioms could be right;
    * matching the forwarded assume to a scope by ``id(eqn)`` — the obvious
      way to "rewrite the assume through the slicer" without a qualified key
      — is ambiguous here for the same reason: it is one equation object
      bound in two scopes.

    A positional scope path is not ambiguous, and the slicer renumbers each
    descent separately, so the two must come out as two axioms naming two
    different declaration pairs."""
    def harness():
        x = any_array((), "float64", (LO, HI))
        y = any_array((), "float64", (LO, HI))
        z = any_array((), "float64", (LO, HI))
        w = any_array((), "float64", (LO, HI))

        @jax.jit
        def gap(a, b):
            assume(a < b)
            return a - b

        return assert_(gap(x, y) + gap(z, w) <= 0.0)

    p, sl = _one_slice(harness)
    assert isinstance(sl, ObligationSlice), sl
    assert len(p.relational_assumes) == 2
    # the premise of the test: one callee, one jaxpr, the SAME ids twice
    ids = [
        tuple(a.id for a in r.eqn.invars) for r in p.relational_assumes
    ]
    assert ids[0] == ids[1], (
        f"this shape no longer shares a cached jaxpr between the two call "
        f"sites ({ids}), so it no longer discriminates — pick another"
    )
    assert p.relational_assumes[0].scope != p.relational_assumes[1].scope
    # ... and two DIFFERENT declaration pairs come out
    by_decl = [inp.var_id for inp in sl.inputs]
    assert len(by_decl) == 4
    got = [tuple(a.id for a in sa.invars) for sa in sl.assumes]
    assert got == [
        (by_decl[0], by_decl[1]),  # gap(x, y): the precondition is x < y
        (by_decl[2], by_decl[3]),  # gap(z, w): the precondition is z < w
    ], f"{got} vs declarations {by_decl}"
    script = smt.emit(sl, "z3", 5000)
    assert script.relational_assumes_emitted == 2
    assert _axiom_lines(script.text) == [
        "(assert (< x0 x1))",
        "(assert (< x2 x3))",
    ], script.text


def test_the_two_walkers_address_a_scope_the_same_way():
    """The coupling the design rests on, pinned directly.

    The propagator records the scope it was in when it forwarded; the slicer
    records the scope it renumbered. They are two independent walks over one
    IR tree, and the only thing that makes a key from one usable in the other
    is that both compute the address the same way. If they ever drift the
    cost is a disclosed skip rather than a wrong axiom — which is the point of
    the design — but it is silent COVERAGE loss, so it is pinned here where a
    drift is a red test rather than a quiet UNKNOWN.

    Walked over nesting (jit inside jit) as well as siblings, because the path
    is built by concatenation and a one-level test cannot see a composition
    error."""
    from stelling.obligation import _Slicer

    def harness():
        x = any_array((), "float64", (LO, HI))
        y = any_array((), "float64", (LO, HI))

        @jax.jit
        def inner(a, b):
            assume(a < b)
            return a - b

        @jax.jit
        def outer(a, b):
            return inner(a, b) * 2.0

        return assert_(outer(x, y) <= 0.0)

    closed = trace(harness)
    p = propagate(closed)
    assert len(p.relational_assumes) == 1
    scope = p.relational_assumes[0].scope
    assert len(scope) == 2, f"the assume is two scopes deep, not {scope}"
    slicer = _Slicer(closed, interval_env(closed))
    assert scope in slicer._scope_remaps, (
        f"the propagator addressed the assume's scope as {scope}, which the "
        f"slicer does not know; it knows "
        f"{sorted(slicer._scope_remaps)} — the two walkers have drifted"
    )
    # and the whole way through: the axiom is about the declarations
    _p, sl = _one_slice(harness)
    assert isinstance(sl, ObligationSlice), sl
    assert len(sl.assumes) == 1, sl.assumes_skipped
    by_decl = [inp.var_id for inp in sl.inputs]
    assert tuple(a.id for a in sl.assumes[0].invars) == (
        by_decl[0], by_decl[1],
    )


def _wrappers():
    """Every member of ``coverage.DEFAULT_TRANSPARENT``, as a decorator.

    `docs/reading-a-verdict.md` names all four as scopes whose assumes ARE
    forwarded; two of them appear nowhere else in this file, so the claim
    would otherwise be a sentence with no test behind it."""
    def custom_jvp(body):
        f = jax.custom_jvp(body)
        f.defjvp(lambda p, t: (f(*p), t[1] - t[0]))
        return f

    def custom_vjp(body):
        f = jax.custom_vjp(body)
        f.defvjp(lambda a, b: (f(a, b), None), lambda res, g: (g, g))
        return f

    return {
        "jit": jax.jit,
        "custom_jvp_call": custom_jvp,
        "custom_vjp_call": custom_vjp,
        "remat2": jax.checkpoint,
    }


def test_the_wrapper_list_above_is_the_registry():
    """`_wrappers` is keyed by primitive name and must BE
    ``DEFAULT_TRANSPARENT``, so a wrapper added to the registry cannot
    acquire assume-carrying behaviour with no test of it."""
    from stelling.coverage import DEFAULT_TRANSPARENT

    assert set(_wrappers()) == set(DEFAULT_TRANSPARENT), (
        f"untested: {set(DEFAULT_TRANSPARENT) - set(_wrappers())}; "
        f"not in the registry: {set(_wrappers()) - set(DEFAULT_TRANSPARENT)}"
    )


@pytest.mark.parametrize("wrapper_name", sorted(_wrappers()))
def test_every_transparent_wrapper_carries_its_assume(wrapper_name):
    """One axiom, naming the two declarations, from inside each wrapper the
    slicer inlines."""
    wrapper = _wrappers()[wrapper_name]

    def harness():
        x = any_array((), "float64", (LO, HI))
        y = any_array((), "float64", (LO, HI))

        def body(a, b):
            assume(a < b)
            return b - a

        return assert_(wrapper(body)(x, y) >= 0.0)

    p, sl = _one_slice(harness)
    assert isinstance(sl, ObligationSlice), sl
    assert len(p.relational_assumes) == 1
    assert len(sl.assumes) == 1, sl.assumes_skipped
    by_decl = [inp.var_id for inp in sl.inputs]
    assert tuple(a.id for a in sl.assumes[0].invars) == (
        by_decl[0], by_decl[1],
    )
    script = smt.emit(sl, "z3", 5000)
    assert script.relational_assumes_emitted == 1
    assert _axiom_lines(script.text) == ["(assert (< x0 x1))"], script.text


def test_the_emission_has_a_rule_for_every_comparison_that_can_be_forwarded():
    """``ASSUME_CMP_SYM`` and ``propagate._ASSUME_CMPS`` are two registries
    naming one set. A comparison in the second and not the first is forwarded
    by the propagator and then skipped by the slicer for want of a spelling —
    disclosed, so not unsound, but a constraint the user wrote lost to a
    registry that nobody updated twice."""
    from stelling.obligation import ASSUME_CMP_SYM
    from stelling.propagate import _ASSUME_CMPS

    assert set(ASSUME_CMP_SYM) == set(_ASSUME_CMPS), (
        f"only in the emission: {set(ASSUME_CMP_SYM) - set(_ASSUME_CMPS)}; "
        f"only in the propagation: {set(_ASSUME_CMPS) - set(ASSUME_CMP_SYM)}"
    )


# ---------------------------------------------------------------------------
# 3. stability across the one-statement id window
# ---------------------------------------------------------------------------


@need_solver
@pytest.mark.parametrize("carrier", ["jit", "custom_jvp"])
def test_the_verdict_does_not_move_when_an_unrelated_statement_is_added(carrier):
    """Adding a trailing statement that touches nothing must not change the
    answer. The defect's signature is exactly that it does: measured on
    ``main``, the ``jit`` shape at tail=0 returned VERIFIED and at tail=1
    returned UNKNOWN, with nothing in either verdict distinguishing them."""
    from stelling.preconditions import check

    seen = {
        tail: check(
            _swapped_harness(carrier, tail),
            vacuity_mode="inputs-only",
            solver_timeout_ms=5000,
        ).status
        for tail in (0, 1, 2, 3)
    }
    assert len(set(seen.values())) == 1, seen
    # and the answer is a SOUND one: under `y < x` the assert is false at
    # every admitted point, so VERIFIED is the one status that cannot be right
    assert seen[0] != "VERIFIED", seen


@need_solver
@pytest.mark.parametrize("carrier", ["jit", "custom_jvp"])
def test_a_witness_for_the_swapped_precondition_satisfies_it(carrier):
    """A REFUTED released on this query must report a witness that is in the
    box AND satisfies the precondition — checked in exact Fraction
    arithmetic, independently of the solver and of the replay.

    NOT SKIPPED WHEN THE STATUS IS NOT REFUTED: an UNKNOWN here is a real
    answer to check (it must not be VERIFIED, which is the one status this
    query cannot honestly carry), and calling that "needs an SMT solver"
    would be a false reason excused by a true rule."""
    from stelling.preconditions import check

    v = check(
        _swapped_harness(carrier),
        vacuity_mode="inputs-only",
        solver_timeout_ms=5000,
    )
    assert v.status != "VERIFIED", (
        "the assert is false at every point the precondition admits"
    )
    if v.status != "REFUTED":
        return
    assert v.witnesses
    for w in v.witnesses:
        vals = {name: F(text) for name, text in w.values}
        x, y = vals["x0"], vals["x1"]
        assert F(LO) <= x <= F(HI) and F(LO) <= y <= F(HI)
        assert y < x, f"witness {vals} violates the precondition y < x"
        assert not (x - y <= 0), f"witness {vals} satisfies the assert"


# ---------------------------------------------------------------------------
# 4. S5-B1: a branch-scoped assume is not a query-global axiom
# ---------------------------------------------------------------------------


def test_a_cond_scoped_relational_assume_is_not_forwarded():
    """A precondition that holds only when a branch is taken is not a fact
    about the query, so it is never asserted globally — and the drop says
    which of the two facts about it stopped the forwarding."""
    closed = trace(_swapped_harness("cond"))
    p = propagate(closed)
    assert p.relational_assumes == (), (
        "a cond-scoped relational assume reached the solver-forwarding list"
    )
    assert p.assume_dropped is True
    reasons = [n for n in p.notes if "branch-scoped precondition" in n]
    assert reasons, p.notes
    assert any("not forwarded to the solver" in n for n in reasons)


def test_a_jit_INSIDE_a_cond_branch_is_still_branch_scoped():
    """The composition, which a guard written on the scope path's LAST step
    would get wrong.

    The assume sits in a `jit` body — an unconditionally executed scope, and
    exactly the shape the fix forwards — but that `jit` is called from inside
    a `cond` branch, so the precondition still holds only when the branch is
    taken. The guard is on `branch_depth`, which the descent does not reset,
    so the nesting is gated at any depth."""
    def harness():
        x = any_array((), "float64", (LO, HI))
        y = any_array((), "float64", (LO, HI))
        r = assert_(x - y <= 0.0)

        @jax.jit
        def inner(p, q):
            assume(p < q)
            return p + q

        lax.cond(x > 0.0, lambda a, b: inner(a, b), lambda a, b: a - b, y, x)
        return r

    p = propagate(trace(harness))
    assert p.relational_assumes == (), (
        "a jit-carried assume inside a cond branch reached the solver-"
        "forwarding list; it is still branch-scoped"
    )
    assert p.assume_dropped is True
    assert any("branch-scoped precondition" in n for n in p.notes), p.notes


def test_the_same_assume_outside_a_cond_is_forwarded():
    """The positive control for the test above: without the ``cond``, the
    identical predicate IS forwarded. Without this, the branch check could
    be a blanket refusal and the test above would not notice."""
    closed = trace(_swapped_harness("jit"))
    p = propagate(closed)
    assert len(p.relational_assumes) == 1
    assert p.relational_assumes[0].eqn.primitive == "lt"
    assert p.relational_assumes[0].scope != (), (
        "the assume is inside a jit body, so its scope path cannot be the "
        "top level"
    )


# ---------------------------------------------------------------------------
# 5. audit M6: the arity crash and the silent truncation
# ---------------------------------------------------------------------------


def _m6_crash_harness():
    """An ARRAY-shaped assume whose colliding ids resolved to SCALAR terms.
    Measured on ``main``: ``IndexError: tuple index out of range`` out of
    ``smt.emit``, degraded by ``_escalate``'s broad guard to an UNKNOWN
    quoting the internal error."""
    def harness():
        x = any_array((), "float64", (LO, HI))
        y = any_array((), "float64", (LO, HI))
        v = any_array((4,), "float64", (LO, HI))
        w = any_array((4,), "float64", (LO, HI))

        @jax.jit
        def g(a, b):
            return b * (-1.0) - a * (-1.0)
        r = assert_(g(x, y) <= 0.0)

        @jax.jit
        def side(p, q):
            a = p * 1.0
            b = q * 1.0
            assume(a < b)
            return a + b
        side(v, w)
        return r
    return harness


def _m6_truncation_harness():
    """A SCALAR assume whose colliding ids resolved to ARRAY term tuples.
    Measured on ``main``: one ``(assert (< t16_0 t17_0))`` — element 0 of an
    unrelated 4-element array — with ``relational_assumes_emitted == 1`` and
    no note of any kind, and the un-withholding rule released a REFUTED on
    it."""
    def harness():
        x = any_array((4,), "float64", (LO, HI))
        y = any_array((4,), "float64", (LO, HI))
        u1 = any_array((), "float64", (LO, HI))
        u2 = any_array((), "float64", (LO, HI))

        @jax.jit
        def g(a, b):
            return b * (-1.0) - a * (-1.0)
        r = assert_(g(x, y) <= 0.0)

        @jax.jit
        def side(p, q):
            a = p * 1.0
            b = q * 1.0
            assume(a < b)
            return a + b
        side(u1, u2)
        return r
    return harness


@need_solver
def test_the_arity_crash_arm_neither_raises_nor_reaches_the_verdict():
    """M6, crash direction. The check is on the NOTE, not on the status:
    ``main`` also answers UNKNOWN here — it just does so by quoting an
    ``IndexError`` from inside the emission."""
    from stelling.preconditions import check

    v = check(
        _m6_crash_harness(), vacuity_mode="inputs-only", solver_timeout_ms=5000
    )
    offenders = [n for n in v.notes if "internal error" in n]
    assert not offenders, offenders
    assert v.status in ("UNKNOWN", "REFUTED"), v.status


@need_solver
def test_the_truncation_arm_emits_no_partial_axiom():
    """M6, silent direction: a wrong-arity assume must not contribute a
    constraint over element 0 of something else, and must not count toward
    the emitted total."""
    from stelling.preconditions import check

    _p, sl = _one_slice(_m6_truncation_harness())
    assert isinstance(sl, ObligationSlice), sl
    script = smt.emit(sl, "z3", 5000)
    assert script.relational_assumes_emitted == 0
    assert _axiom_lines(script.text) == [], script.text
    v = check(
        _m6_truncation_harness(),
        vacuity_mode="inputs-only",
        solver_timeout_ms=5000,
    )
    assert v.status != "REFUTED", (
        "a REFUTED released on a truncated forged axiom (audit M6 / "
        "id-collision finding 4)"
    )


def test_a_wrong_arity_slice_assume_is_a_message_not_an_IndexError():
    """The structural backstop. The production path cannot build one of
    these, so this is the only way to reach the check — and reaching it must
    produce a sentence, which is what a bare ``lhs_terms[ia[i]]`` did not."""
    _p, sl = _one_slice(_swapped_harness("jit"))
    assert isinstance(sl, ObligationSlice)
    good = sl.assumes[0]
    broken = SliceAssume(
        primitive=good.primitive,
        invars=good.invars,
        pairs=good.pairs + ((7, 9),),  # indices no term tuple holds
        source_info=good.source_info,
    )
    hacked = ObligationSlice(
        index=sl.index,
        fragment=sl.fragment,
        inputs=sl.inputs,
        consts=sl.consts,
        eqns=sl.eqns,
        root=sl.root,
        source_info=sl.source_info,
        element_terms=sl.element_terms,
        assumes=(broken,),
        assumes_skipped=sl.assumes_skipped,
    )
    with pytest.raises(ValueError, match="slice validation should have"):
        smt.emit(hacked, "z3", 5000)


# ---------------------------------------------------------------------------
# 6. the emission has no channel for a foreign-scope id
# ---------------------------------------------------------------------------


def test_emit_takes_no_relational_assumes_parameter():
    """The type IS the fix. ``emit`` used to accept raw comparison equations,
    which is how an assume stated in one scope's ids reached a lookup keyed on
    another's. There is no such parameter now: the axioms come off the slice,
    whose ``SliceAssume`` values cannot express a foreign name."""
    import inspect

    params = set(inspect.signature(smt.emit).parameters)
    assert "relational_assumes" not in params, params
    assert params == {"sl", "solver", "timeout_ms"}, params


def test_a_skipped_assume_is_disclosed_in_the_verdict_notes():
    """An assume this obligation's slice cannot state costs the user a
    constraint they wrote, so it is said out loud. Before the fix the
    emission had five bare ``continue``s and nothing in the verdict
    distinguished a run that applied a precondition from one that dropped
    it."""
    def harness():
        x = any_array((), "float64", (LO, HI))
        y = any_array((), "float64", (LO, HI))
        z = any_array((), "float64", (LO, HI))

        @jax.jit
        def side(p, q):
            assume(p < q)
            return p + q
        side(x, z)  # z is not in the assert's cone
        return assert_(x - y <= 0.0)

    p, sl = _one_slice(harness)
    assert len(p.relational_assumes) == 1
    assert sl.assumes == ()
    assert len(sl.assumes_skipped) == 1
    reason = sl.assumes_skipped[0]
    assert "not forwarded to the solver" in reason
    assert "backward cone" in reason
