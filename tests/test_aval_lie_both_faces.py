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


@pytest.mark.parametrize(
    "bad_param,expect",
    [
        (b"\x04", "not a sequence of extents"),
        ("4", "not a sequence of extents"),
        (("4",), "non-integer extent"),
        ((-4,), "negative extent"),
        (object(), "not a sequence of extents"),
    ],
    ids=["bytes", "str", "string-extent", "negative", "not-iterable"],
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
