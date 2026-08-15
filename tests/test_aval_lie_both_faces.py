# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""AUDIT 0.2.0 S12′ — THE ORACLE IS SHARED; ITS ARGUMENTS ARE NOT.

S12's repair gave ``dot_general`` a shape oracle
(``interval.dot_general_geometry``) that both faces call, and claimed "the
two faces cannot hold different opinions about whether an equation is
admissible". That was false as written. ``interval.dot_general`` asks the
oracle about the shapes of the PROPAGATED BOXES; ``obligation.
_dot_general_plan`` asks the same function about the shapes recorded on the
equation's INVAR AVALS. Move the lie off the declaration and onto those
avals — which ``from_dict`` accepts, ``ir.py`` having scoped per-primitive
shape inference out of ``_validate_loaded`` in writing — and the two faces
disagree again, in the asserting direction.

**Worse than S12's own presentation.** There the interval leg refused the
equation and propagated ⊤. Here it AGREES the contraction has four terms
and prints the box; the emission plans two anyway. Measured on ``4d793cf``:

    propagation: [(0, 'unknown')]  "the operand spans [4.0, 8.0] ...
                                    misses the bound by 3.5"
    ESCALATION 0 -> discharged
    STATUS: VERIFIED    coverage: 4 eqns: 4 known (100%)

**IT IS A CLASS, NOT A ROW.** ``reduce_sum`` carries the identical defect
and reaches VERIFIED the same way, and both rows also mint a false REFUTED
whose witness the verdict calls "confirmed by independent exact-rational
replay" — a sentence true about the arithmetic and false about the plan,
because replay re-derives the same truncated plan.

So the repair is not a third shape rule. It is
``_Slicer._one_shape_per_value``: no equation may be modelled at a shape
that disagrees with the shape the value actually has, checked once for
every primitive at once. Read that method's docstring for its two
witnesses and where each is blind; the ``jit``-nested cases below are the
ones only the binding-site witness can see, because an inner id has no
propagated box at all.
"""
from __future__ import annotations

import copy
from fractions import Fraction

import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402

from stelling import _optional  # noqa: E402
from stelling import ir  # noqa: E402
from stelling.harness import any_array, assert_, trace  # noqa: E402
from stelling.obligation import (  # noqa: E402
    DeclinedObligation,
    slice_unknown_obligations,
)
from stelling.propagate import interval_env, propagate  # noqa: E402
from stelling.solvers import (  # noqa: E402
    OB_DISCHARGED,
    OB_UNKNOWN,
    OB_VIOLATED_WITNESS,
    SolverConfig,
    escalate,
)

HAVE_Z3 = _optional.available("z3")
HAVE_CVC5 = _optional.available("cvc5") or _optional.cvc5_binary() is not None
needs_solvers = pytest.mark.skipif(
    not (HAVE_Z3 and HAVE_CVC5), reason="needs both z3 and cvc5"
)


@pytest.fixture(autouse=True, scope="module")
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


LO, HI = Fraction(1), Fraction(2)
N = 4
CEILING = Fraction(9, 2)     # 4.5
SLACK = Fraction(-1, 2)      # -0.5


# -- the harnesses ------------------------------------------------------------
#
# Each pair is (true reading, truncated reading). The declaration is (4,)
# and the constant operand is four ones in every one of them; only the
# dot_general / reduce_sum INVAR AVALS are edited, to (2,).


def _dot_ceiling():
    a = any_array((N,), jnp.float64, (float(LO), float(HI)))
    c = jnp.array([1.0, 1.0, 1.0, 1.0])
    return assert_(jnp.dot(a, c) <= float(CEILING))


def _sum_ceiling():
    a = any_array((N,), jnp.float64, (float(LO), float(HI)))
    return assert_(jnp.sum(a) <= float(CEILING))


def _dot_identity():
    # `s - t` is IDENTICALLY ZERO: the same four addends, spelled two ways.
    # Interval arithmetic loses the correlation ([4,8] - [4,8] = [-4,4]) so
    # the obligation is undecided and escalates, which is what puts the
    # truncating plan in front of a solver. Truncated, `s` drops two
    # addends and `s - t` becomes -(a2+a3) in [-4,-2] — refutable, and the
    # refutation is false.
    a = any_array((N,), jnp.float64, (float(LO), float(HI)))
    c = jnp.array([1.0, 1.0, 1.0, 1.0])
    s = jnp.dot(a, c)
    t = a[0] + a[1] + a[2] + a[3]
    return assert_(s - t >= float(SLACK))


def _sum_identity():
    a = any_array((N,), jnp.float64, (float(LO), float(HI)))
    s = jnp.sum(a)
    t = a[0] + a[1] + a[2] + a[3]
    return assert_(s - t >= float(SLACK))


@jax.jit
def _jitted_dot(a):
    # `b` is produced INSIDE the body, so the slicer mints a FRESH id for it
    # and no interval environment ever holds a box for that id. The
    # propagated-box witness is blind here by construction; the binding-site
    # witness is not.
    b = a * 1.0
    return jnp.dot(b, jnp.array([1.0, 1.0, 1.0, 1.0]))


@jax.jit
def _jitted_sum(a):
    return jnp.sum(a * 1.0)


def _jit_dot_ceiling():
    a = any_array((N,), jnp.float64, (float(LO), float(HI)))
    return assert_(_jitted_dot(a) <= float(CEILING))


def _jit_sum_ceiling():
    a = any_array((N,), jnp.float64, (float(LO), float(HI)))
    return assert_(_jitted_sum(a) <= float(CEILING))


def _lie(prim, extent):
    """Edit ONLY the ``prim`` equation's invar avals to shape ``(extent,)``.

    The declaration keeps its four elements and the constant operand keeps
    its four elements: this is the lie S12's own repair does not reach,
    because both faces still call one oracle — with different arguments.
    """
    def edit(d):
        edits = 0

        def walk(o):
            nonlocal edits
            if isinstance(o, dict):
                if o.get("primitive") == prim:
                    for a in o["invars"]:
                        if "aval" in a:
                            a["aval"]["shape"] = [extent]
                    edits += 1
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)

        walk(d)
        return edits

    return edit


def _load(harness, prim, extent=2):
    d = copy.deepcopy(trace(harness).to_dict())
    edits = _lie(prim, extent)(d)
    assert edits == 1, (
        f"the serialization no longer has exactly one {prim!r} node this "
        f"edit targets ({edits}); the test would measure nothing"
    )
    return ir.ClosedJaxpr.from_dict(d)


# -- the truth, before the tool is asked --------------------------------------


def test_the_truth_about_these_queries_needs_no_verifier():
    """Exact rationals, stated here rather than asked of the thing on
    trial."""
    four = N * HI
    two = 2 * HI
    assert four == Fraction(8) and two == Fraction(4)
    assert not (four <= CEILING), (
        "the four-term ceiling claim is FALSE, so no VERIFIED about it is "
        "sound"
    )
    assert two <= CEILING, (
        "the truncated two-term claim is TRUE — which is the VERIFIED the "
        "solver honestly returned when it was handed the smaller question"
    )
    # and the identity harnesses: s - t is identically zero in ℝ
    assert Fraction(0) >= SLACK, (
        "the identity claim is TRUE everywhere, so no REFUTED about it is "
        "sound"
    )
    # truncated, s - t is -(a2 + a3), which is at most -2
    assert not (-2 * LO >= SLACK)


def test_concrete_jax_agrees_with_the_exact_arithmetic():
    """The same truth, executed rather than reasoned — the leg the verdict
    never runs."""
    top = jnp.array([2.0, 2.0, 2.0, 2.0])
    ones = jnp.array([1.0, 1.0, 1.0, 1.0])
    assert float(jnp.dot(top, ones)) == 8.0
    assert float(jnp.sum(top)) == 8.0
    assert not (float(jnp.dot(top, ones)) <= float(CEILING))
    for a in (top, ones, jnp.array([1.0, 2.0, 1.5, 1.25])):
        t = a[0] + a[1] + a[2] + a[3]
        assert float(jnp.dot(a, ones) - t) == 0.0
        assert float(jnp.sum(a) - t) == 0.0


# -- the door is UNCHANGED, deliberately --------------------------------------


@pytest.mark.parametrize("prim", ["dot_general", "reduce_sum"])
def test_the_door_still_accepts_the_lying_document(prim):
    """The repair is not a door check, for the reason
    `test_dot_general_from_dict_door.py` already records: `ir.py` scopes
    per-primitive shape inference out of `_validate_loaded` in writing, and
    `ClosedJaxpr` is a public dataclass anyway. Pinning that the door is
    unchanged is what stops a later reader "fixing" this there and
    concluding the in-pipeline cross-check is redundant."""
    q = _load(_dot_ceiling if prim == "dot_general" else _sum_ceiling, prim)
    eqn = next(e for e in q.jaxpr.eqns if e.primitive == prim)
    assert all(tuple(a.aval.shape) == (2,) for a in eqn.invars)
    # the DECLARATION was not touched: four elements, still
    decl = next(e for e in q.jaxpr.eqns if e.primitive == "stelling_any")
    assert tuple(decl.params_dict()["shape"]) == (N,)


# -- the interval leg AGREES, which is what makes this worse than S12 ---------


@pytest.mark.parametrize(
    "harness,prim",
    [(_dot_ceiling, "dot_general"), (_sum_ceiling, "reduce_sum")],
)
def test_the_propagation_leg_still_sees_all_four_terms(harness, prim):
    """S12 had the transfer REFUSE and propagate ⊤. Here it computes the
    true box, prints it, and leaves the obligation undecided because the
    box straddles — so the truncating plan is what escalation then runs,
    with nothing anywhere in the coverage record to warn a reader."""
    q = _load(harness, prim)
    p = propagate(q)
    assert [o.status for o in p.obligations] == ["unknown"]
    assert "[4.0, 8.0]" in p.obligations[0].detail, p.obligations[0].detail
    assert p.coverage.unknown_primitives == (), (
        "the coverage record names a ⊤ primitive, so S12's own recognition "
        "screen would have caught this one"
    )


# -- the false VERIFIED, closed ----------------------------------------------


@needs_solvers
@pytest.mark.parametrize(
    "harness,prim",
    [
        (_dot_ceiling, "dot_general"),
        (_sum_ceiling, "reduce_sum"),
        (_jit_dot_ceiling, "dot_general"),
        (_jit_sum_ceiling, "reduce_sum"),
    ],
    ids=["dot", "sum", "jit-dot", "jit-sum"],
)
def test_the_truncated_reading_is_no_longer_discharged(harness, prim):
    """On `4d793cf` every one of these four returned `discharged`, and the
    verdict read VERIFIED at 100% coverage on a claim whose truth is
    `8 <= 4.5`. They now DECLINE, naming the value whose two recorded
    shapes disagree."""
    q = _load(harness, prim)
    p = propagate(q)
    assert [o.status for o in p.obligations] == ["unknown"]
    esc = escalate(q, p, SolverConfig(timeout_ms=20_000))
    (record,) = esc.records
    assert record.outcome != OB_DISCHARGED, (
        f"the truncated {prim} was discharged again: {record.detail}"
    )
    assert record.outcome == OB_UNKNOWN, record.detail
    assert "one shape" in record.detail, record.detail
    assert "BOUND at shape (4,)" in record.detail, record.detail


# -- the false REFUTED, closed ------------------------------------------------


@needs_solvers
@pytest.mark.parametrize(
    "harness,prim",
    [(_dot_identity, "dot_general"), (_sum_identity, "reduce_sum")],
)
def test_the_truncated_reading_no_longer_mints_a_witness(harness, prim):
    """THE HALF THAT TAKES A CLAIM DOWN WITH IT. On `4d793cf` both of these
    returned `violated-witness` at `x = (1, 1, 1, 1)` — a point where the
    predicate is TRUE — and the verdict said the witness was "confirmed by
    independent exact-rational replay". The arithmetic in that sentence was
    honest; the PLAN it replayed was the truncated one, because replay
    drives the same `_dot_general_plan` / `_group_reduce_sum` the emission
    does. A witness is only independent of the solver, never of the plan."""
    q = _load(harness, prim)
    p = propagate(q)
    assert [o.status for o in p.obligations] == ["unknown"]
    esc = escalate(q, p, SolverConfig(timeout_ms=20_000))
    (record,) = esc.records
    assert record.witness is None, (
        f"a witness was minted from the truncated plan: {record.witness}"
    )
    assert record.outcome == OB_UNKNOWN, record.detail
    assert "one shape" in record.detail, record.detail


# -- the jit-nested pair: where the propagated-box witness is BLIND -----------


@pytest.mark.parametrize(
    "harness,prim",
    [(_jit_dot_ceiling, "dot_general"), (_jit_sum_ceiling, "reduce_sum")],
)
def test_the_lying_operand_inside_a_jit_has_no_box_at_all(harness, prim):
    """The fixture answer for the second witness's blind spot, measured
    rather than argued: `interval_env` returns the TOP-LEVEL environment,
    the propagator runs each transparent call body in an isolated env it
    discards on the way out, and the slicer mints fresh ids for every inner
    binding. So the operand these cases lie about is in no environment at
    all — and they are still closed, by the binding-site witness alone."""
    q = _load(harness, prim)
    assert "jit" in [e.primitive for e in q.jaxpr.eqns], (
        "the wrapper was inlined at trace time; this fixture no longer has "
        "a sub-jaxpr and measures nothing"
    )
    assert prim not in [e.primitive for e in q.jaxpr.eqns], (
        f"the lying {prim!r} is at the TOP level here, where the box "
        f"witness can see it; this fixture measures nothing"
    )
    env = interval_env(q)
    p = propagate(q)
    (item,) = slice_unknown_obligations(q, p, env)
    assert isinstance(item, DeclinedObligation), item
    assert "one shape" in item.reason, item.reason
    # the operand named in the decline is NOT one env has a box for: that is
    # the whole point of this pair
    named = int(item.reason.split("variable ")[1].split(" ")[0])
    assert named not in env, (
        f"variable {named} IS in the environment, so this case does not "
        f"exercise the blind spot it was written for"
    )


# -- THE COST: a check that declines legitimate work is a coverage defect -----


@needs_solvers
@pytest.mark.parametrize(
    "harness,expected",
    [
        (_dot_ceiling, OB_VIOLATED_WITNESS),
        (_sum_ceiling, OB_VIOLATED_WITNESS),
        (_dot_identity, OB_DISCHARGED),
        (_sum_identity, OB_DISCHARGED),
        (_jit_dot_ceiling, OB_VIOLATED_WITNESS),
        (_jit_sum_ceiling, OB_VIOLATED_WITNESS),
    ],
    ids=["dot", "sum", "dot-id", "sum-id", "jit-dot", "jit-sum"],
)
def test_the_same_harnesses_UNEDITED_are_untouched(harness, expected):
    """The control, and it is the acceptance criterion for the fix: every
    one of these six is the SAME program with no aval edited, and none may
    decline — a check that declines legitimate work is a coverage defect,
    not a repair.

    The OUTCOME is asserted, not merely the absence of the decline. The
    identity pair is the interesting one: it is exactly the shape the false
    REFUTED rode in on, and with truthful avals the emission plans all four
    addends and DISCHARGES it — which is the right answer, `s - t` being
    identically zero. The four ceiling harnesses are genuinely refutable
    (their sums reach 8) and produce a witness, which is also the right
    answer. So the cross-check costs nothing on either side of the verdict."""
    q = trace(harness)
    p = propagate(q)
    for item in slice_unknown_obligations(q, p, interval_env(q)):
        assert not isinstance(item, DeclinedObligation), (
            f"a well-formed query declined: {item.reason}"
        )
    esc = escalate(q, p, SolverConfig(timeout_ms=20_000))
    for r in esc.records:
        assert "one shape" not in (r.detail or ""), r.detail
        assert r.outcome == expected, r.detail


# -- the SECOND witness, driven directly, and what its status honestly is -----


def test_the_propagated_box_witness_declines_when_it_is_the_one_that_sees():
    """THE BOX WITNESS, exercised — and this test exists because nothing
    else exercises it.

    In all six reproducers above the BINDING-SITE witness answers first, so
    the box leg never fires; over the whole test suite, instrumented, it saw
    23,072 atoms and disagreed with none of them. That is a fact worth
    stating rather than papering over: **no IR document has been
    constructed on which the box witness is the only thing that sees the
    lie.** The lie it is there for is one applied CONSISTENTLY at a value's
    binding and at every reference to it, and every route to such a lie
    found so far is refused earlier by something else — a `stelling_any`
    whose `shape` param contradicts its outvar aval is refused by
    `ir.JaxprEqn.__post_init__` at construction, a constvar whose aval
    contradicts its value is refused in `slice`'s pass 2, and a computed
    outvar whose aval contradicts its operands is refused by the row's own
    shape rule (`_route_structural`, `_pair_elementwise`, `_group_reduce_
    sum`, `_dot_general_plan`).

    So it is DEFENCE IN DEPTH, and it is kept for two reasons that do not
    depend on a live route. It is the inter-leg agreement the S12 commit
    claimed and did not have — the property, stated where it can be
    checked. And `env` is a caller-supplied argument of the public
    `slice_obligation`, which is exactly how it is driven here: an
    environment whose box disagrees with the query is a caller-visible
    input, not an invented one.
    """
    from stelling import interval as _iv
    from stelling.obligation import slice_obligation

    q = trace(_sum_ceiling)
    env = dict(interval_env(q))
    rebound = [k for k, v in env.items() if v.shape == (N,)]
    assert len(rebound) == 1, (
        f"expected exactly one (4,)-shaped box to rebind, got {rebound}"
    )
    (vid,) = rebound
    env[vid] = _iv.from_bounds((2,), float(LO), float(HI))

    item = slice_obligation(q, 0, env)
    assert isinstance(item, DeclinedObligation), item
    assert "interval propagation computed a box of shape (2,)" in item.reason
    assert "modelling different arrays" in item.reason
    # and the binding-site witness is NOT what fired: every aval in this
    # query is truthful, which is the whole point of driving it this way
    assert "BOUND at shape" not in item.reason, item.reason

    # the same query with the true environment does not decline
    assert not isinstance(
        slice_obligation(q, 0, interval_env(q)), DeclinedObligation
    )
