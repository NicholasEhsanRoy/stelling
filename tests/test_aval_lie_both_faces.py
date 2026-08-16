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

import array as _arraymod
import copy
import operator
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


# -- THE DECLARATION'S OWN TWO SELF-DESCRIPTIONS (B6 re-audit, UNSOUND-1) ----
#
# Everything above moves the lie onto a COMPUTED equation's invar avals. The
# re-audit moved it one step earlier, onto the `stelling_any` itself, and the
# binding-site witness could not see it: `_binding_shape` answered from the
# declaration's OUTVAR AVAL, and `slice` mints one SMT constant per element
# of the declaration's `shape` PARAM. For exactly that one binding class the
# check compared a quantity nothing emits. Measured on `96ab47a`: a `jit`
# body declaring four elements in its param and two in its aval minted four
# symbols, summed two, and returned `discharged` on `8 <= 4.5`.


def _jit_decl_body_eqns(harness):
    """The (jit eqn, its ClosedJaxpr body) of a traced harness."""
    from stelling.coverage import call_body

    q = trace(harness)
    jit_eqn = next(e for e in q.jaxpr.eqns if e.primitive == "jit")
    return q, jit_eqn, call_body(jit_eqn)


def _rebuild_with_body(q, jit_eqn, body_eqns, consts):
    body = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=jit_eqn.params_dict()["jaxpr"].jaxpr.constvars,
            invars=jit_eqn.params_dict()["jaxpr"].jaxpr.invars,
            outvars=jit_eqn.params_dict()["jaxpr"].jaxpr.outvars,
            eqns=tuple(body_eqns),
            effects=jit_eqn.params_dict()["jaxpr"].jaxpr.effects,
            debug_info=jit_eqn.params_dict()["jaxpr"].jaxpr.debug_info,
        ),
        consts=consts,
    )
    new_jit = ir.JaxprEqn(
        primitive=jit_eqn.primitive,
        invars=jit_eqn.invars,
        outvars=jit_eqn.outvars,
        params=tuple((k, (body if k == "jaxpr" else v))
                     for k, v in jit_eqn.params),
        effects=jit_eqn.effects,
        source_info=jit_eqn.source_info,
    )
    return ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=q.jaxpr.constvars,
            invars=q.jaxpr.invars,
            outvars=q.jaxpr.outvars,
            eqns=tuple(new_jit if e is jit_eqn else e for e in q.jaxpr.eqns),
            effects=q.jaxpr.effects,
            debug_info=q.jaxpr.debug_info,
        ),
        consts=q.consts,
    )


@jax.jit
def _jitted_declared_sum():
    # the DECLARATION is inside the body, so no environment ever holds a box
    # for it: `interval_env` returns the top level and the propagator runs
    # each transparent body in an isolated env it discards
    a = any_array((N,), jnp.float64, (float(LO), float(HI)))
    return jnp.sum(a)


def _jit_declared_ceiling():
    return assert_(_jitted_declared_sum() <= float(CEILING))


def test_a_declaration_with_NO_shape_param_binds_at_the_scalar_it_emits():
    """THE FORM THE DOOR BLESSES, AND THE SLICER MUST COVER ALONE.

    `ir._validate_decl_eqn` cross-checks a declaration's `shape` param
    against its outvar aval, and deliberately does not fire when the param
    is ABSENT — hand-built IR legitimately omits params, which is the form
    `ir._validate_required_params`' docstring blesses in writing. `slice`
    reads `params.get("shape", ())`, so an absent param means ONE scalar
    symbol; here every aval in the document says four elements and agrees
    with every other, so no reference disagrees with any other reference and
    the value is inside a `jit` body where no box exists.

    On `96ab47a` this sliced, and `smt.emit` then raised `IndexError: tuple
    index out of range` — reaching the verdict as "escalation attempted;
    internal error", an obligation quoting a stelling defect where the
    honest answer is a named refusal. It is the same wrong-quantity read as
    the list-param reproducer, in the direction that crashes rather than the
    direction that discharges, and no door check can reach it."""
    q, jit_eqn, inner = _jit_decl_body_eqns(_jit_declared_ceiling)
    stripped = tuple(
        ir.JaxprEqn(
            primitive=e.primitive, invars=e.invars, outvars=e.outvars,
            params=tuple((k, v) for k, v in e.params if k != "shape"),
            effects=e.effects, source_info=e.source_info,
        )
        if e.primitive == "stelling_any" else e
        for e in inner.jaxpr.eqns
    )
    assert any(
        e.primitive == "stelling_any" and "shape" not in e.params_dict()
        for e in stripped
    ), "the fixture did not strip the shape param"
    bad = _rebuild_with_body(q, jit_eqn, stripped, inner.consts)

    p = propagate(bad)
    assert [o.status for o in p.obligations] == ["unknown"]
    env = interval_env(bad)
    (item,) = slice_unknown_obligations(bad, p, env)
    assert isinstance(item, DeclinedObligation), item
    assert "BOUND at shape ()" in item.reason, item.reason
    assert "one shape" in item.reason, item.reason
    # the lying value is in NO environment: the box witness cannot see it
    named = int(item.reason.split("variable ")[1].split(" ")[0])
    assert named not in env, (
        f"variable {named} IS in the environment, so this fixture does not "
        f"exercise the blind spot it was written for"
    )
    # and the truth, in exact rationals: the claim is false
    assert not (N * HI <= CEILING)


def test_the_slicer_closes_the_declaration_lie_ON_ITS_OWN():
    """THE SLICER IS CORRECT WITHOUT THE DOOR, and this test is what says so.

    `ir._validate_decl_eqn` now compares a declaration's two
    self-descriptions whatever holds the extents, so the re-audit's own
    reproducer (`shape=[4]`, outvar aval `(2,)`) is refused at construction
    and cannot reach a slicer at all —
    `tests/test_array_emission.py::test_the_declaration_check_reads_the_
    EXTENTS_not_the_param_type` pins that. That refusal must not be the only
    thing standing here: `SOUNDNESS.md` puts hand-built `ir.ClosedJaxpr` in
    scope, `ir.py` scopes per-primitive shape inference out of the load
    validation in writing, and a slicer that relies on a door check is one
    door change away from the false VERIFIED again.

    So the disagreement is installed with `object.__setattr__`, PAST the
    frozen dataclass's `__post_init__` — "suppose the door were bypassed" —
    and the slicer is asked directly. It declines, naming the shape the
    EMISSION would mint terms from as the binding: the `shape` param, four
    elements, against a document every one of whose avals says two.

    The document is the re-audit's `p1_decl_listshape.py` in every other
    respect. The `jit`-nested form of the same lie is
    `test_a_declaration_with_NO_shape_param_binds_at_the_scalar_it_emits`
    above and `..._is_refused_when_the_descent_re_transcribes_it` below —
    the descent rebuilds each equation, so the constructor's check runs a
    second time there and answers first."""
    q = trace(_sum_ceiling)
    decl_id = next(e.outvars[0].id for e in q.jaxpr.eqns
                   if e.primitive == "stelling_any")

    def relabel(a, shape):
        return ir.Aval(kind=a.kind, shape=tuple(shape), dtype=a.dtype,
                       weak_type=a.weak_type)

    lied = []
    for e in q.jaxpr.eqns:
        invars = tuple(
            ir.Var(id=a.id, aval=relabel(a.aval, (2,)))
            if isinstance(a, ir.Var) and a.id == decl_id else a
            for a in e.invars
        )
        outvars = tuple(
            ir.Var(id=v.id, aval=relabel(v.aval, (2,)))
            if v.id == decl_id else v for v in e.outvars
        )
        params = e.params
        if e.primitive == "stelling_any":
            # constructed CONSISTENT (both self-descriptions say (2,)) so
            # that the door lets it through at all
            params = tuple((k, ((2,) if k == "shape" else v))
                           for k, v in e.params)
        new = ir.JaxprEqn(
            primitive=e.primitive, invars=invars, outvars=outvars,
            params=params, effects=e.effects, source_info=e.source_info,
        )
        if e.primitive == "stelling_any":
            # ... and then moved PAST the constructor's check, on purpose:
            # `shape` says four elements, every aval in the document says
            # two, and every reference agrees with every other reference.
            object.__setattr__(
                new, "params",
                tuple((k, ((N,) if k == "shape" else v)) for k, v in e.params),
            )
            assert tuple(new.params_dict()["shape"]) == (N,)
            assert tuple(new.outvars[0].aval.shape) == (2,)
        lied.append(new)
    bad = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(constvars=q.jaxpr.constvars, invars=q.jaxpr.invars,
                       outvars=q.jaxpr.outvars, eqns=tuple(lied),
                       effects=q.jaxpr.effects,
                       debug_info=q.jaxpr.debug_info),
        consts=q.consts)

    p = propagate(bad)
    assert [o.status for o in p.obligations] == ["unknown"], (
        "the interval leg no longer leaves this undecided, so nothing "
        "escalates and the fixture measures nothing"
    )
    env = interval_env(bad)
    (item,) = slice_unknown_obligations(bad, p, env)
    assert isinstance(item, DeclinedObligation), item
    assert f"BOUND at shape ({N},)" in item.reason, item.reason
    # the BINDING leg is what answered, and that is the point: it is checked
    # before the box leg and it read the `shape` param rather than the aval
    assert "interval propagation computed a box" not in item.reason, item.reason
    # the truth: four elements of [1, 2] sum to at most 8, and 8 > 4.5
    assert not (N * HI <= CEILING)


def test_the_declaration_lie_is_refused_when_the_descent_re_transcribes_it():
    """AND THE SAME LIE INSIDE A `jit`, where the slicer's own inlining
    rebuilds every equation through `ir.JaxprEqn` and the constructor check
    therefore runs a SECOND time, over the descended scope.

    Pinned because it is the reason the top-level fixture above is at the
    top level: not because the binding witness cannot see the nested form —
    `test_a_declaration_with_NO_shape_param_binds_at_the_scalar_it_emits`
    shows it seeing exactly that, in the same place, on the form the door
    blesses — but because here something answers first.

    What is asserted is the WORDING as much as the outcome. A
    `TranscriptionError` out of `_renumber_eqn` means the DOCUMENT is
    malformed; quoting it as "internal error" would tell a reader the tool
    broke, which is the M17′ misreading this repository has already paid
    for once."""
    q, jit_eqn, inner = _jit_decl_body_eqns(_jit_declared_ceiling)
    decl_id = next(e.outvars[0].id for e in inner.jaxpr.eqns
                   if e.primitive == "stelling_any")

    def relabel(a, shape):
        return ir.Aval(kind=a.kind, shape=tuple(shape), dtype=a.dtype,
                       weak_type=a.weak_type)

    lied = []
    for e in inner.jaxpr.eqns:
        invars = tuple(
            ir.Var(id=a.id, aval=relabel(a.aval, (2,)))
            if isinstance(a, ir.Var) and a.id == decl_id else a
            for a in e.invars
        )
        outvars = tuple(
            ir.Var(id=v.id, aval=relabel(v.aval, (2,)))
            if v.id == decl_id else v for v in e.outvars
        )
        params = e.params
        if e.primitive == "stelling_any":
            params = tuple((k, ((2,) if k == "shape" else v))
                           for k, v in e.params)
        new = ir.JaxprEqn(
            primitive=e.primitive, invars=invars, outvars=outvars,
            params=params, effects=e.effects, source_info=e.source_info,
        )
        if e.primitive == "stelling_any":
            object.__setattr__(
                new, "params",
                tuple((k, ((N,) if k == "shape" else v)) for k, v in e.params),
            )
        lied.append(new)
    bad = _rebuild_with_body(q, jit_eqn, lied, inner.consts)

    p = propagate(bad)
    assert [o.status for o in p.obligations] == ["unknown"]
    (item,) = slice_unknown_obligations(bad, p, interval_env(bad))
    assert isinstance(item, DeclinedObligation), item
    assert "could not be re-transcribed" in item.reason, item.reason
    assert "contradicts the outvar aval shape" in item.reason, item.reason
    assert "internal error" not in item.reason, item.reason


@needs_solvers
def test_the_declaration_lie_no_longer_reaches_a_discharge():
    """The verdict-bearing leg for the two documents above. On `96ab47a`
    the `object.__setattr__` document returned `discharged` — "the box with
    the negated predicate is unsat per z3 (wheel) and cvc5 (wheel)" — for a
    script whose sum had TWO of the four declared addends."""
    from stelling.smt import emit

    q, jit_eqn, inner = _jit_decl_body_eqns(_jit_declared_ceiling)
    stripped = tuple(
        ir.JaxprEqn(
            primitive=e.primitive, invars=e.invars, outvars=e.outvars,
            params=tuple((k, v) for k, v in e.params if k != "shape"),
            effects=e.effects, source_info=e.source_info,
        )
        if e.primitive == "stelling_any" else e
        for e in inner.jaxpr.eqns
    )
    bad = _rebuild_with_body(q, jit_eqn, stripped, inner.consts)
    p = propagate(bad)
    esc = escalate(bad, p, SolverConfig(timeout_ms=20_000))
    (record,) = esc.records
    assert record.outcome == OB_UNKNOWN, record.detail
    assert record.witness is None
    assert "one shape" in record.detail, record.detail
    # and the sentence is a REFUSAL, not a quoted stelling defect
    assert "internal error" not in record.detail, record.detail

    # the UNEDITED harness is untouched: a check that declines legitimate
    # work is a coverage defect, not a repair
    good = trace(_jit_declared_ceiling)
    gp = propagate(good)
    for item in slice_unknown_obligations(good, gp, interval_env(good)):
        assert not isinstance(item, DeclinedObligation), item.reason
        assert len(item.inputs) == N, (
            f"the emission minted {len(item.inputs)} symbols for a "
            f"{N}-element declaration"
        )
        assert "x0_3" in emit(item, "z3", 20_000).text
    for r in escalate(good, gp, SolverConfig(timeout_ms=20_000)).records:
        assert r.outcome == OB_VIOLATED_WITNESS, r.detail


def test_the_BINDING_witness_alone_closes_the_declaration_lie(monkeypatch):
    """THE MUTATION CONTROL FOR THE STRUCK CLAIM (audit 0.2.0 B6 re-audit,
    UNSOUND-2). On `96ab47a` the box leg was the SOLE detector of the
    top-level declaration lie: delete it and the same query came back
    `discharged`. The entry beside it said no such document existed.

    So the replacement claim — *the binding witness must be total in its
    own right* — is checked here the way the old one was not: the box leg
    is deleted and the binding leg is required to answer alone, on the
    document that defeated it before.
    """
    import stelling.obligation as OB

    real = OB._Slicer._one_shape_per_value

    def binding_leg_only(self, eqn):
        for atom in eqn.invars:
            if not isinstance(atom, ir.Var):
                continue
            here = tuple(atom.aval.shape)
            bound = self._binding_shape(atom)
            if bound is None:
                raise OB._Decline("unbindable")
            if bound != here:
                raise OB._Decline(
                    f"{eqn.primitive!r} refers to variable {atom.id} at "
                    f"shape {here} but it is BOUND at shape {bound}"
                )
        # THE BOX LEG IS DELETED

    q = trace(_sum_ceiling)
    decl_id = next(e.outvars[0].id for e in q.jaxpr.eqns
                   if e.primitive == "stelling_any")

    def relabel(a, shape):
        return ir.Aval(kind=a.kind, shape=tuple(shape), dtype=a.dtype,
                       weak_type=a.weak_type)

    lied = []
    for e in q.jaxpr.eqns:
        invars = tuple(
            ir.Var(id=a.id, aval=relabel(a.aval, (2,)))
            if isinstance(a, ir.Var) and a.id == decl_id else a
            for a in e.invars
        )
        outvars = tuple(
            ir.Var(id=v.id, aval=relabel(v.aval, (2,)))
            if v.id == decl_id else v for v in e.outvars
        )
        params = e.params
        if e.primitive == "stelling_any":
            params = tuple((k, ((2,) if k == "shape" else v))
                           for k, v in e.params)
        new = ir.JaxprEqn(
            primitive=e.primitive, invars=invars, outvars=outvars,
            params=params, effects=e.effects, source_info=e.source_info,
        )
        if e.primitive == "stelling_any":
            object.__setattr__(
                new, "params",
                tuple((k, ((N,) if k == "shape" else v)) for k, v in e.params),
            )
        lied.append(new)
    bad = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(constvars=q.jaxpr.constvars, invars=q.jaxpr.invars,
                       outvars=q.jaxpr.outvars, eqns=tuple(lied),
                       effects=q.jaxpr.effects,
                       debug_info=q.jaxpr.debug_info),
        consts=q.consts)
    p = propagate(bad)
    env = interval_env(bad)
    # the box leg COULD see this one — that is what made it load-bearing
    lying = [k for k, v in env.items() if tuple(v.shape) == (N,)]
    assert lying, "no box for the lying value; this is the wrong fixture"

    monkeypatch.setattr(OB._Slicer, "_one_shape_per_value", binding_leg_only)
    (item,) = slice_unknown_obligations(bad, p, env)
    assert isinstance(item, DeclinedObligation), (
        "with the box leg deleted nothing sees the lie, so the box leg is "
        "still the sole detector and the binding witness is not total"
    )
    assert f"BOUND at shape ({N},)" in item.reason, item.reason
    assert real is not binding_leg_only  # the mutation really replaced it


class _WillNotIterate(list):
    """A `list` SUBCLASS — `isinstance(x, ir._SHAPE_PARAM_CONTAINERS)` is
    true of it — whose `__iter__` refuses. `isinstance` is a claim about
    the TYPE and not about the object, so the container rule cannot be the
    whole guard."""

    def __iter__(self):
        raise RuntimeError("will not iterate")


@pytest.mark.parametrize(
    "bad_param,expect",
    [
        # THE CONTAINER ROWS quote the rule's own sentence. It used to read
        # "not a sequence of extents", which is a claim about the object
        # and was false of most of these — `np.array([4])` and
        # `array.array("i", [4])` ARE sequences of extents, and were
        # refused for their TYPE (audit 0.2.0 B6 audit 4, F1). The marker
        # is the fixed part of the sentence `ir._SHAPE_PARAM_RULE` is
        # interpolated into, so it cannot drift from the rule it applied.
        (b"\x04", "records its extents in"),
        ("4", "records its extents in"),
        (("4",), "non-integer extent"),
        ((-4,), "negative extent"),
        (object(), "records its extents in"),
        # AND THE TWO THE ENUMERATION MISSED — audit 0.2.0 B6 audit 3.
        # `tuple(memoryview(b"44"))` is `(52, 52)` and so is
        # `tuple(array.array("b", b"44"))`: the identical misread the
        # `bytes` row above exists to refuse, in a container the
        # `(str, bytes, bytearray)` test did not name — and the slicer
        # SLICED a four-element declaration off one, measured. The rule is
        # now stated positively (`ir._SHAPE_PARAM_CONTAINERS`), so these
        # fall out of it rather than needing to be named by it, and so
        # does whichever sequence type is noticed next.
        (memoryview(b"\x02\x02"), "records its extents in"),
        (_arraymod.array("b", b"\x02\x02"), "records its extents in"),
        # ... and the arm `isinstance` cannot see: the RIGHT container,
        # whose iteration refuses. A separate fact and a separate sentence.
        (_WillNotIterate(), "whose iteration RAISES"),
    ],
    ids=["bytes", "str", "string-extent", "negative", "not-iterable",
         "memoryview", "array.array", "list-subclass-that-will-not-iterate"],
)
def test_a_declaration_shape_param_that_cannot_be_read_DECLINES(
    bad_param, expect
):
    """`_declared_shape` fails closed, and the two string cases are the ones
    worth spelling out: `tuple(b"\x04")` is `(4,)` and `tuple("4")` is
    `("4",)` — a bytes param would be read as a perfectly plausible extent
    list the declaration never stated, which is a silent misread rather than
    a refusal. Both are refused, and so is a param that will not iterate at
    all.

    Installed past `ir.JaxprEqn.__post_init__`, which refuses all five at
    construction: the point is that the SLICER refuses them too, since it
    may not rest on a door `SOUNDNESS.md` scopes hand-built IR around. Driven
    through `slice_obligation` with a caller-supplied environment, because
    the TRANSFER face reaches `tuple(shape)` on the not-iterable one first —
    see `test_the_transfer_face_still_raises_raw_on_an_uniterable_shape_param`
    below, which records that as a finding rather than fixing it here."""
    q = trace(_sum_ceiling)
    eqns = []
    for e in q.jaxpr.eqns:
        if e.primitive != "stelling_any":
            eqns.append(e)
            continue
        new = ir.JaxprEqn(
            primitive=e.primitive, invars=e.invars, outvars=e.outvars,
            params=e.params, effects=e.effects, source_info=e.source_info,
        )
        object.__setattr__(
            new, "params",
            tuple((k, (bad_param if k == "shape" else v))
                  for k, v in e.params),
        )
        eqns.append(new)
    bad = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(constvars=q.jaxpr.constvars, invars=q.jaxpr.invars,
                       outvars=q.jaxpr.outvars, eqns=tuple(eqns),
                       effects=q.jaxpr.effects,
                       debug_info=q.jaxpr.debug_info),
        consts=q.consts)
    from stelling.obligation import slice_obligation

    item = slice_obligation(bad, 0, {})
    assert isinstance(item, DeclinedObligation), item
    assert expect in item.reason, item.reason
    assert "internal error" not in item.reason, item.reason


def test_the_transfer_face_still_raises_raw_on_an_uniterable_shape_param():
    """REPORTED, NOT FIXED, and pinned so that the report cannot rot.

    `propagate._t_stelling_any` calls `tuple(params["shape"])` with no
    guard, so a `shape` param that will not iterate raises
    `TypeError: 'object' object is not iterable` out of the public
    `propagate()` — the S12" family again, on the transfer face, while the
    emission face (above) declines. It is out of this batch's scope
    (`stelling.propagate`, and a different face from the one under repair),
    and the constructible route to it is now shut: `ir._validate_decl_eqn`
    refuses a non-sequence `shape` param at construction, so only an
    `object.__setattr__` past the frozen dataclass reaches it.

    **AND IT IS NOT CATCHABLE THROUGH THE DOCUMENTED HANDLERS** — audit
    0.2.0 B6 audit 3, Q6, which is what makes this a residue worth
    carrying rather than a wording nit. "The transfer raises where the
    emission declines" reads like a difference in phrasing; it is a
    difference in what a caller can catch. The library's two malformed-IR
    exception classes are `interval.IntervalError` and
    `ir.TranscriptionError`, and NEITHER covers this: what escapes is a
    bare `TypeError`, and `TranscriptionError` *subclasses* `TypeError`,
    so the relationship runs the wrong way. Only `except TypeError` — a
    handler no caller should be asked to write, since it swallows their
    own bugs too — sees it. A caller handling malformed IR exactly as this
    library documents therefore gets a crash.

    IF THIS TEST FAILS because `propagate()` no longer raises, the finding
    has been fixed: delete this test and say so wherever it is recorded."""
    q = trace(_sum_ceiling)
    eqns = []
    for e in q.jaxpr.eqns:
        if e.primitive != "stelling_any":
            eqns.append(e)
            continue
        new = ir.JaxprEqn(
            primitive=e.primitive, invars=e.invars, outvars=e.outvars,
            params=e.params, effects=e.effects, source_info=e.source_info,
        )
        object.__setattr__(
            new, "params",
            tuple((k, (object() if k == "shape" else v))
                  for k, v in e.params),
        )
        eqns.append(new)
    bad = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(constvars=q.jaxpr.constvars, invars=q.jaxpr.invars,
                       outvars=q.jaxpr.outvars, eqns=tuple(eqns),
                       effects=q.jaxpr.effects,
                       debug_info=q.jaxpr.debug_info),
        consts=q.consts)
    with pytest.raises(TypeError, match="not iterable"):
        propagate(bad)
    # ... and the documented handlers do not see it
    from stelling import interval as _iv

    for cls in (_iv.IntervalError, ir.TranscriptionError):
        try:
            propagate(bad)
        except cls:  # pragma: no cover — the finding is that this is dead
            raise AssertionError(
                f"{cls.__name__} now catches the transfer face's raise; the "
                f"Q6 disclosure has been overtaken and must be rewritten"
            )
        except TypeError as ex:
            assert not isinstance(ex, cls), (cls, type(ex))
            assert type(ex) is TypeError, type(ex)


# -- the SECOND witness, driven directly, and what its status honestly is -----


def test_the_propagated_box_witness_declines_when_it_is_the_one_that_sees():
    """THE BOX WITNESS, exercised — and it is LOAD-BEARING, not a second
    opinion.

    **WHAT THIS TEST USED TO SAY, AND WHY IT WAS STRUCK** (audit 0.2.0 B6
    re-audit, UNSOUND-2). It used to call the box leg "defence in depth" and
    support that with an enumeration: every route to a consistently-applied
    lie "found so far is refused earlier by something else", and "no IR
    document has been constructed on which the box witness is the only thing
    that sees the lie". The re-audit constructed one in an afternoon — a
    `stelling_any` whose `shape` param and outvar aval disagreed, which the
    enumeration listed as refused at construction and which was NOT, because
    `ir._validate_decl_eqn` ran only `if isinstance(shape, tuple)`. On
    `96ab47a` the box leg was the SOLE detector of that document at top
    level, proved by deleting the box leg and watching the same query come
    back `discharged`.

    That is the same failure mode twice in one batch, and it is what S12
    itself did: **asserting completeness for an enumeration nobody drove.**
    So no enumeration stands here. What stands is three checkable facts:

    1. The box leg is the only witness a CONSISTENTLY-APPLIED lie cannot
       forge — it is computed from the values flowing in, not from what the
       IR says about them.
    2. It is load-bearing wherever a box exists, and blind inside
       transparent call bodies by construction (`interval_env` returns the
       top-level environment; the propagator discards each body's isolated
       env on the way out).
    3. Which is why the BINDING witness must be total in its own right —
       and is what `test_the_slicer_closes_the_declaration_lie_ON_ITS_OWN`
       above measures, on a document with no box for the lying value at all.

    Neither leg is here as insurance for the other. `env` is a
    caller-supplied argument of the public `slice_obligation`, which is
    exactly how this test drives it: an environment whose box disagrees with
    the query is a caller-visible input, not an invented one.
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


# -- audit 0.2.0 B6 audit 3: the reader's own totality, and its claims ------


class _TwoFacedExtent:
    """`__index__` answers `first` on the first call and `then` after."""

    def __init__(self, first, then):
        self._answers = (first, then)
        self.reads = 0

    def __index__(self):
        self.reads += 1
        return self._answers[0] if self.reads == 1 else self._answers[1]

    def __repr__(self):
        return f"_TwoFacedExtent(reads={self.reads})"


def _decl_eqn(shape_param, aval_shape=(N,)):
    """A `stelling_any` carrying `shape_param`, installed PAST
    `ir.JaxprEqn.__post_init__` — the slicer may not rest on the door."""
    e = ir.JaxprEqn(
        primitive="stelling_any", invars=(),
        outvars=(ir.Var(id=0, aval=ir.Aval(
            kind="ShapedArray", shape=tuple(aval_shape), dtype="float64")),),
        params=(("dtype", "float64"), ("hi", float(HI)), ("lo", float(LO)),
                ("shape", tuple(aval_shape))),
    )
    object.__setattr__(e, "params", tuple(sorted(
        (("dtype", "float64"), ("hi", float(HI)), ("lo", float(LO)),
         ("shape", shape_param)), key=lambda kv: kv[0])))
    return e


def test_declared_shape_RETURNS_the_extents_it_validated():
    """AUDIT 0.2.0 B6 AUDIT 3, F1 — `_declared_shape` returned a shape it
    had just rejected.

    It called `_shape_problem(shape)`, which bound `k = _op_index(d)`,
    tested `k` and DISCARDED it, and then returned
    `tuple(_op_index(d) for d in shape)` — a SECOND read per extent, and
    the one the emission got. An extent answering 4 and then -1 was
    therefore validated at 4 and EMITTED as `(-1,)`, where `_size` takes a
    negative contribution to the element budget and `range(-1)` mints no
    symbols at all. Measured on `d6b6d0b`:

        _declared_shape returned (-1,)   <-- validated 4, EMITS (-1,)

    This is the identical defect this same batch fixed one module over in
    `interval.dot_general_geometry`, and described in its own CHANGELOG:
    "the first spelling called it and discarded the result, so the dims
    were validated and never normalised". A guard that tests a value nobody
    keeps has not guarded the value the emission uses.
    """
    import stelling.obligation as OB

    sl = object.__new__(OB._Slicer)
    d = _TwoFacedExtent(4, -1)
    got = OB._Slicer._declared_shape(sl, _decl_eqn((d,)), 0)
    assert got == (4,), (
        f"the guard validated 4 and the method returned {got}: the returned "
        f"value is a second, unvalidated read"
    )
    assert d.reads == 1, f"{d.reads} reads per extent; one read is the fix"
    # a negative extent on the FIRST read is still refused, and the refusal
    # is the one thing the second read used to be able to smuggle past
    with pytest.raises(OB._Decline, match="negative extent"):
        OB._Slicer._declared_shape(sl, _decl_eqn((_TwoFacedExtent(-1, 4),)), 0)


class _IndexTwoMulOne:
    """A THIRD PROTOCOL. `__index__` answers 2 — every guard in
    `obligation` validates through that and is satisfied — and `__eq__`
    agrees with 2, so every shape comparison passes as well. Only
    `__mul__`/`__rmul__` disagree, and an element COUNT computed as
    `n *= d` over the raw objects is the one reader that asks them."""

    def __index__(self):
        return 2

    def __eq__(self, o):
        return 2 == o

    def __hash__(self):
        return hash(2)

    def __mul__(self, o):
        return 1

    __rmul__ = __mul__

    def __repr__(self):
        return "<index=2, mul=1>"


def test_an_element_COUNT_comes_from___index___and_not_from___mul__():
    """AUDIT 0.2.0 B6 AUDIT 4, F3 — `_size` multiplied the RAW objects.

    `n = 1; for d in shape: n *= d` reaches `__mul__`/`__rmul__`, a third
    protocol beside the `__index__` every guard validates with and the
    `__eq__` the shape comparisons use. Measured on `30d4b04`:

        operator.index(d)               = 2      <- what the guard validates
        obligation._shape_problem((d,)) = None   <- 'no problem'
        obligation._size((d,))          = 1      <- what every COUNT reader got

    Six readers took the predicate face of `_extents` and then counted with
    such a raw second read, and the previous audit could not drive any of
    them to a false verdict: the constvar route is closed earlier by the
    `ir` door and the `_decode_elements` route by the byte-length check.
    **That containment was accidental**, which is why the repair is not a
    list of call sites — `_size` itself now reads through `__index__`, so
    no caller anywhere can obtain a count from a third protocol, including
    one written tomorrow.
    """
    import stelling.obligation as OB

    d = _IndexTwoMulOne()
    # the premise: this object satisfies every OTHER reader
    assert operator.index(d) == 2 and OB._shape_problem((d,)) is None
    assert tuple((d,)) == (2,)  # __eq__ agrees too

    assert OB._size((d,)) == 2, (
        "the element count came from `__mul__` and not from the `__index__` "
        "the guard validated with"
    )
    assert OB._size((d, d)) == 4, OB._size((d, d))

    # ... and a count that CANNOT be read is a decline, not a number: an
    # unreadable extent used to make the product silent garbage
    class _NoIndex:
        def __index__(self):
            raise ValueError("no")

    with pytest.raises(OB._Decline, match="no element count"):
        OB._size((_NoIndex(),))


def test_the_named_count_readers_bind_what_the_guard_VALIDATED():
    """AUDIT 0.2.0 B6 AUDIT 4, F3, second half — one READ, not two.

    `_size` reading through `__index__` closes the third protocol. It does
    not make a reader that screens a shape and then counts it again into a
    single read, and an object that answers `__index__` differently between
    calls is checked at one value and counted at another. The four readers
    the audit named now bind what `_extents` returned:
    `_decode_scalar`, `_decode_elements`, `_Slicer.slice`'s assert root and
    its constvar pass.

    Driven on `_decode_elements`, with the drifting extent installed PAST
    `ir.Array.__post_init__` so the decoder gets the object's FIRST read —
    the door is a separate reader with its own lifetime, and the claim
    here is about one function, not about the whole tree. The extent
    answers 2 and then 8 against a two-element payload. Before this,
    `_shape_problem` validated the first read (2) and `_size` took the
    second (8), so the byte-length check compared the payload against a
    count nothing had validated and the decoder declined a payload that
    matched what it had just approved.
    """
    import struct

    import stelling.obligation as OB

    d = _TwoFacedExtent(2, 8)
    arr = ir.Array(dtype="<f8", shape=(2,), data=struct.pack("<2d", 1.0, 2.0))
    object.__setattr__(arr, "shape", (d,))

    got = OB._decode_elements(arr)
    assert got == (1.0, 2.0), (
        f"the decoder screened the extent at 2 and then counted it again "
        f"at 8: {got!r}"
    )
    assert d.reads == 1, (
        f"the decoder read the extent {d.reads} time(s); one read is the "
        f"fix, and the second read is the one nothing validated"
    )


@pytest.mark.parametrize(
    "exc",
    [ValueError("index says no"), OverflowError("too big"),
     RuntimeError("some other refusal")],
    ids=["ValueError", "OverflowError", "RuntimeError"],
)
def test_the_declaration_reader_refuses_whatever___index___raises(exc):
    """AUDIT 0.2.0 B6 AUDIT 3, F2 — `_shape_problem` caught only
    `TypeError`, and `operator.index` raises whatever `__index__` raises.

    A `ValueError` or an `OverflowError` from one extent therefore escaped
    the reader and reached the caller through `slice_obligation`'s generic
    net as *"slice attempted; internal error: ValueError: index says no"* —
    a malformed document reported as a stelling defect, which is the exact
    sentence M17′ was already named for. The exception type an extent
    chooses is not a fact about whether the document can be read."""
    import stelling.obligation as OB

    class _IndexRaises:
        def __index__(self):
            raise exc

        def __repr__(self):
            return "_IndexRaises"

    sl = object.__new__(OB._Slicer)
    with pytest.raises(OB._Decline, match="non-integer extent"):
        OB._Slicer._declared_shape(sl, _decl_eqn((_IndexRaises(),)), 0)


def test_a_declaration_refusal_cannot_be_stopped_by_a_hostile___repr__():
    """AUDIT 0.2.0 B6 AUDIT 3, F3 — two composers in the refusal path
    interpolated an unguarded `{!r}`.

    `_declared_shape`'s str/bytes branch quoted `{raw!r}` and its
    extent branch quoted `{shape!r}`, while the sibling branch between them
    already used `_safely` for exactly this hazard — the same commit
    introduced `_safely` and then did not reach for it twice. A `str`
    subclass with a refusing `__repr__` turned a clean decline into
    *"internal error: RuntimeError: repr refuses"*: the decline the guard
    decided on never reached the caller, and what did reach them blamed the
    tool for the document.

    A message about an object already known to be malformed may not itself
    be able to raise. The placeholder is visible, so a reader is told
    something could not be read rather than shown a plausible value."""
    import stelling.obligation as OB

    class _HostileStr(str):
        def __repr__(self):
            raise RuntimeError("repr refuses")

    class _HostileExtent:
        def __index__(self):
            raise ValueError("no")

        def __repr__(self):
            raise RuntimeError("repr refuses")

    sl = object.__new__(OB._Slicer)
    with pytest.raises(OB._Decline) as ei:
        OB._Slicer._declared_shape(sl, _decl_eqn(_HostileStr("34")), 0)
    assert "records its extents in" in ei.value.reason
    assert "<unreadable>" in ei.value.reason, ei.value.reason

    with pytest.raises(OB._Decline) as ei:
        OB._Slicer._declared_shape(sl, _decl_eqn((_HostileExtent(),)), 0)
    assert "non-integer extent" in ei.value.reason
    assert "<unreadable>" in ei.value.reason, ei.value.reason

    # and through the public face: a decline, never an "internal error"
    from stelling.obligation import slice_obligation

    q = trace(_sum_ceiling)
    eqns = tuple(
        _decl_eqn(_HostileStr("34")) if e.primitive == "stelling_any" else e
        for e in q.jaxpr.eqns
    )
    # keep the declaration's own var id, which _decl_eqn does not know
    decl = next(e for e in q.jaxpr.eqns if e.primitive == "stelling_any")
    fixed = []
    for e in eqns:
        if e.primitive == "stelling_any":
            object.__setattr__(e, "outvars", decl.outvars)
        fixed.append(e)
    bad = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(constvars=q.jaxpr.constvars, invars=q.jaxpr.invars,
                       outvars=q.jaxpr.outvars, eqns=tuple(fixed),
                       effects=q.jaxpr.effects,
                       debug_info=q.jaxpr.debug_info),
        consts=q.consts)
    item = slice_obligation(bad, 0, {})
    assert isinstance(item, DeclinedObligation), item
    assert "internal error" not in item.reason, item.reason
    assert "records its extents in" in item.reason, item.reason


def test_declared_shape_is_NOT_the_librarys_only_reader_of_an_element_count():
    """AUDIT 0.2.0 B6 AUDIT 3, F4 — the docstring said "THE ONE READER",
    and a `grep` refutes it.

    `stelling.propagate._declared_element_count` is a genuine second reader
    of "how many elements does this declaration have", and it reads the
    OTHER quantity: the outvar AVAL, where `_declared_shape` reads the
    `shape` PARAM. One document makes them answer `()` and `4`.

    IT IS NOT DRIVEABLE TO A WRONG VERDICT, and this test pins both halves
    of that. The count gates only `_region_witness`'s cap — whether the
    non-emptiness search RUNS — whose direction is toward REFUTED, and the
    search re-derives its witness by re-running the honest propagator, so
    nothing downstream is derived from the miscount. But a false claim of
    sole readership is worse than no claim: it tells the next reader to
    stop looking, and this batch has already been wrong twice about
    enumerations nobody drove. `SOUNDNESS.md`'s narrower wording — the
    budget, the input-term construction and `_binding_shape` — is true and
    stays."""
    import stelling.obligation as OB
    import stelling.propagate as P

    # door-legal: NO shape param at all. The aval says 4 elements; the
    # param, absent, reads as ().
    x = ir.Var(id=0, aval=ir.Aval(
        kind="ShapedArray", shape=(N,), dtype="float64"))
    d = ir.JaxprEqn(
        primitive="stelling_any", invars=(), outvars=(x,),
        params=(("dtype", "float64"), ("hi", float(HI)), ("lo", float(LO))),
    )
    j = ir.Jaxpr(constvars=(), invars=(), outvars=(x,), eqns=(d,))

    sl = object.__new__(OB._Slicer)
    assert OB._Slicer._declared_shape(sl, d, 0) == ()
    assert P._declared_element_count(j) == N
    # ... which is a real disagreement, on one document, between two live
    # readers of the same quantity
    assert OB._size(OB._Slicer._declared_shape(sl, d, 0)) != (
        P._declared_element_count(j)
    )

    # the second reader has exactly one call site, and it is the cap
    import inspect

    src = inspect.getsource(P)
    calls = [
        ln.strip() for ln in src.splitlines()
        if "_declared_element_count(" in ln and not ln.strip().startswith("def")
    ]
    assert calls == ["elements = _declared_element_count(closed.jaxpr)"], calls
    assert "_region_witness" in inspect.getsource(P._region_witness)

    # and the docstring no longer claims what the measurement refutes
    doc = OB._Slicer._declared_shape.__doc__
    assert "THE ONE READER" not in doc, doc
    assert "cannot read different quantities by drifting apart" not in doc
    assert "_declared_element_count" in doc, (
        "the second reader must be NAMED where the false claim used to be"
    )


def test_the_declaration_reader_is_a_FUNCTION_and_not_a_single_READ():
    """AUDIT 0.2.0 B6 AUDIT 3, F4, second half — *"cannot drift apart"* was
    false, and what contains the drift is not this method.

    `_declared_shape` is the one reader in the sense that the budget,
    `_binding_shape` and the input-term construction all call it, so none
    can implement a different rule. It is NOT one read: each call re-reads
    the param, and `slice` alone reads it three times. A `list` SUBCLASS
    whose `__iter__` answers differently between calls — `isinstance(raw,
    (tuple, list))` is true of it, so both faces accept it — therefore does
    make the check and the emission differ. Swept over the read at which it
    flips, measured identically on `d6b6d0b` and on this tree:

        flip=1,2   DECLINE, by the binding witness
        flip=3     SLICED with 1 input term for a FOUR-element reference
        flip>=4    SLICED with 4

    What stops the flip=3 document reaching a verdict is
    `ClosedJaxpr.content_hash()`: a param that can answer differently
    between iterations cannot be an `ir._encode`-able value, so hashing
    RAISES, `solvers._query_sha256` swallows that to `""`, and the pairing
    gate refuses an empty hash. Naming the containment where it is matters
    — "cannot drift apart" tells the next reader to stop looking."""
    from stelling.obligation import slice_obligation

    class Drifting(list):
        n = 0

        def __iter__(self):
            type(self).n += 1
            return iter((N,) if type(self).n <= 3 else ())

    x = ir.Var(id=0, aval=ir.Aval(
        kind="ShapedArray", shape=(N,), dtype="float64"))
    s = ir.Var(id=1, aval=ir.Aval(kind="ShapedArray", shape=(), dtype="float64"))
    pr = ir.Var(id=2, aval=ir.Aval(kind="ShapedArray", shape=(), dtype="bool"))
    o = ir.Var(id=3, aval=ir.Aval(kind="ShapedArray", shape=(), dtype="bool"))
    tail = (
        ir.JaxprEqn(primitive="reduce_sum", invars=(x,), outvars=(s,),
                    params=(("axes", (0,)), ("out_sharding", None))),
        ir.JaxprEqn(
            primitive="le",
            invars=(s, ir.Literal(val=float(CEILING), aval=s.aval)),
            outvars=(pr,)),
        ir.JaxprEqn(primitive="stelling_assert", invars=(pr,), outvars=(o,)),
    )
    d = ir.JaxprEqn(
        primitive="stelling_any", invars=(), outvars=(x,),
        params=(("dtype", "float64"), ("hi", float(HI)), ("lo", float(LO)),
                ("shape", Drifting())))
    q = ir.ClosedJaxpr(jaxpr=ir.Jaxpr(constvars=(), invars=(), outvars=(o,),
                                      eqns=(d,) + tail))
    item = slice_obligation(q, 0, {})
    assert not isinstance(item, DeclinedObligation), item
    assert len(item.inputs) == 1, (
        f"the drifting param minted {len(item.inputs)} term(s); the finding "
        f"is that ONE is minted for a {N}-element reference"
    )

    # ... and the containment is the hash, not the reader
    with pytest.raises(TypeError, match="cannot encode"):
        q.content_hash()
