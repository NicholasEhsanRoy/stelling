# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The array-emission acceptance cases, end to end, with agreement checks.

Two known-answer families fixed before the build (reproduced, not tuned),
run under ``jax_enable_x64=True`` float64:

* **FACE** — per-face quantities on a ring (``jnp.roll`` traces to a
  transparent jit wrapping slice+concatenate, so this exercises the call
  descent): benign bounds VERIFY by interval alone and escalation changes
  nothing; a sign-straddling range escalates to REFUTED with a replayed
  per-element witness naming the violating element. **Agreement check**:
  the scalar-reduced form over the same range must agree.
* **DOT** — scalars into arrays into a reduction and a division:
  ``g(0.71)`` escalates VERIFIED, ``g(0.11)`` escalates REFUTED with a
  replayed in-region witness. **Agreement check**: the hand-reduced
  scalar form ``p/(q*r) <= 8`` must agree on both regions.

Both agreement checks are permanent: if array and scalar forms disagree
where both are evaluable, the emission changed an answer rather than
reached one — stop and report.

Plus the measured-behaviour battery: the structural-op index routing is
compared against REAL jax execution (arange values make routing readable
as values), and the zero-size / empty-sum / scalar-selector semantics the
zero-dep tests pin are measured here at their source. Skipped without
jax; the end-to-end cases additionally need a solver.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

jax = pytest.importorskip("jax")

import numpy as np  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402

from stelling import _optional  # noqa: E402
from stelling._jax_compat import transcribe  # noqa: E402
from stelling.harness import any_array, assert_, trace  # noqa: E402
from stelling.obligation import (  # noqa: E402
    ObligationSlice,
    _group_reduce_sum,
    _route_structural,
    slice_unknown_obligations,
)
from stelling.propagate import interval_env, propagate  # noqa: E402
from stelling.smt import emit  # noqa: E402
from stelling.solvers import (  # noqa: E402
    SolverConfig,
    escalate,
    make_solver_verdict,
)

HAVE_SOLVER = (
    _optional.available("z3")
    or _optional.available("cvc5")
    or _optional.cvc5_binary() is not None
)
need_solver = pytest.mark.skipif(not HAVE_SOLVER, reason="needs an SMT solver")

VERSIONS = dict(
    stelling_version="test",
    jax_version=jax.__version__,
    precision_config="jax_enable_x64=True",
)
CONFIG = SolverConfig(timeout_ms=30_000)


@pytest.fixture(autouse=True, scope="module")
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


# --- Case FACE ---------------------------------------------------------------

SIDE = 4
INV_H2 = 1.0 / 0.25**2


def faces(theta):
    a = theta
    a_plus = 0.5 * (a + jnp.roll(a, -1)) * INV_H2
    a_minus = 0.5 * (a + jnp.roll(a, 1)) * INV_H2
    return a_plus, a_minus


def face_harness(lo, hi):
    def h():
        th = any_array((SIDE,), "float64", (lo, hi))
        ap, am = faces(th)
        return assert_(ap > 0.0), assert_(am > 0.0)

    return h


def face_scalar_harness(lo, hi):
    """The scalar-reduced agreement form: one cell's positivity."""

    def h():
        th = any_array((), "float64", (lo, hi))
        return (assert_(th > 0.0),)

    return h


def test_face_benign_bounds_verify_by_interval_and_escalation_is_a_noop():
    cj = trace(face_harness(1e-6, 1e2))
    p = propagate(cj)
    assert [o.status for o in p.obligations] == ["discharged", "discharged"]
    v0 = make_solver_verdict(
        cj, p, escalate(cj, p, CONFIG), **VERSIONS
    )
    assert v0.status == "VERIFIED"
    # escalating changed nothing: zero records, zero spawns, absence stamped
    esc = escalate(cj, p, CONFIG)
    assert esc.records == () and esc.ledger.spawns == 0
    assert v0.stamp.solver.invoked is False


def test_face_slice_descends_the_roll_jit_and_stays_linear():
    cj = trace(face_harness(-2.0, 1e2))
    p = propagate(cj)
    assert [o.status for o in p.obligations] == ["unknown", "unknown"]
    items = slice_unknown_obligations(cj, p, interval_env(cj))
    assert len(items) == 2
    for sl in items:
        assert isinstance(sl, ObligationSlice), getattr(sl, "reason", sl)
        assert sl.fragment == "QF_LRA"  # constant-coefficient face chain
        assert [i.name for i in sl.inputs] == [
            "x0_0", "x0_1", "x0_2", "x0_3"
        ]
    # the roll is pure routing: element i of the assert-0 face pairs cell i
    # with cell (i+1) % 4 — visible in the emitted bodies, with the four
    # declared constants SHARED between both slices' scripts
    text = emit(items[0], "z3", 1000).text
    assert text.count("(declare-const") == 4
    assert "(+ x0_0 x0_1)" in text
    assert "(+ x0_3 x0_0)" in text  # the wrap-around face
    assert "(assert (not (and " in text  # the universal elementwise claim


@need_solver
def test_face_straddling_bounds_escalate_to_refuted_with_element_witness():
    cj = trace(face_harness(-2.0, 1e2))
    p = propagate(cj)
    esc = escalate(cj, p, CONFIG)
    v = make_solver_verdict(cj, p, esc, **VERSIONS)
    assert v.status == "REFUTED"
    assert v.witnesses  # replayed witness(es), not a set-level refutation
    for w in v.witnesses:
        values = dict(w.values)
        assert set(values) == {"x0_0", "x0_1", "x0_2", "x0_3"}
        # every element inside the declared box [-2, 100]
        vals = [Fraction(values[f"x0_{i}"]) for i in range(4)]
        assert all(Fraction(-2) <= x <= Fraction(100) for x in vals)
        # at least one face element non-positive, recomputed here
        # independently of stelling: face_i = 8 * (a_i + a_{i+1 mod 4})
        # for assert #0, 8 * (a_i + a_{i-1 mod 4}) for assert #1
        shift = -1 if w.obligation_index == 0 else 1
        face = [
            Fraction(8) * (vals[i] + vals[(i + (1 if shift == -1 else -1)) % 4])
            for i in range(4)
        ]
        violating = [i for i, f in enumerate(face) if not f > 0]
        assert violating, (w.obligation_index, [float(f) for f in face])
        # the witness NAMES the violating element(s), and correctly
        assert w.violating_elements
        assert set(w.violating_elements) <= set(violating)
    rendered = v.render()
    assert "violating element(s) of the assert operand:" in rendered


@need_solver
def test_face_agreement_check_scalar_form_agrees_on_both_regions():
    """MANDATORY: the array form's verdicts must agree with the
    scalar-reduced form wherever both are evaluable. Disagreement here
    means the emission changed an answer rather than reached one — stop
    and report."""
    # benign region: both VERIFIED
    cj_a = trace(face_harness(1e-6, 1e2))
    p_a = propagate(cj_a)
    v_a = make_solver_verdict(cj_a, p_a, escalate(cj_a, p_a, CONFIG), **VERSIONS)
    cj_as = trace(face_scalar_harness(1e-6, 1e2))
    p_as = propagate(cj_as)
    v_as = make_solver_verdict(
        cj_as, p_as, escalate(cj_as, p_as, CONFIG), **VERSIONS
    )
    assert v_as.status == "VERIFIED"
    assert v_a.status == v_as.status
    # straddling region: scalar form escalates REFUTED (witness 0 exists);
    # the array form must agree
    cj_b = trace(face_harness(-2.0, 1e2))
    p_b = propagate(cj_b)
    v_b = make_solver_verdict(cj_b, p_b, escalate(cj_b, p_b, CONFIG), **VERSIONS)
    cj_bs = trace(face_scalar_harness(-2.0, 1e2))
    p_bs = propagate(cj_bs)
    v_bs = make_solver_verdict(
        cj_bs, p_bs, escalate(cj_bs, p_bs, CONFIG), **VERSIONS
    )
    assert v_bs.status == "REFUTED"
    (ws,) = v_bs.witnesses
    x0 = Fraction(dict(ws.values)["x0"])
    assert Fraction(-2) <= x0 <= Fraction(100) and not x0 > 0
    assert v_b.status == v_bs.status


# --- Case DOT ----------------------------------------------------------------


def dot_harness(r_lo):
    def g():
        p = any_array((), "float64", (0.5, 2.0))
        q = any_array((), "float64", (0.5, 2.0))
        r = any_array((), "float64", (r_lo, 1.0))
        u = jnp.array([p, 0.0 * p, 0.0 * p])
        v = jnp.array([q * r, 0.0 * q, 0.0 * q])
        s = jnp.sum(u * v)
        return (assert_(p**2 / s <= 8.0),)

    return g


def dot_scalar_harness(r_lo):
    def g():
        p = any_array((), "float64", (0.5, 2.0))
        q = any_array((), "float64", (0.5, 2.0))
        r = any_array((), "float64", (r_lo, 1.0))
        return (assert_(p / (q * r) <= 8.0),)

    return g


@need_solver
def test_dot_safe_region_escalates_verified():
    cj = trace(dot_harness(0.71))
    p = propagate(cj)
    assert [o.status for o in p.obligations] == ["unknown"]
    esc = escalate(cj, p, CONFIG)
    (record,) = esc.records
    assert record.outcome == "discharged", record.detail
    v = make_solver_verdict(cj, p, esc, **VERSIONS)
    assert v.status == "VERIFIED"


@need_solver
def test_dot_low_r_region_escalates_refuted_with_in_region_witness():
    cj = trace(dot_harness(0.11))
    p = propagate(cj)
    assert [o.status for o in p.obligations] == ["unknown"]
    esc = escalate(cj, p, CONFIG)
    (record,) = esc.records
    assert record.outcome == "violated-witness", record.detail
    values = dict(record.witness.values)
    pv, qv, rv = (Fraction(values[k]) for k in ("x0", "x1", "x2"))
    # in-region
    assert Fraction(1, 2) <= pv <= 2
    assert Fraction(1, 2) <= qv <= 2
    assert Fraction(11, 100) <= rv <= 1
    # violating, recomputed here without stelling arithmetic:
    # s = p*q*r, so the obligation is p^2/(p*q*r) = p/(q*r) <= 8
    assert (pv * pv) / (pv * qv * rv) > 8
    v = make_solver_verdict(cj, p, esc, **VERSIONS)
    assert v.status == "REFUTED"


@need_solver
def test_dot_agreement_check_scalar_form_agrees_on_both_regions():
    """MANDATORY: the hand-reduced scalar form produces VERIFIED at 0.71
    and REFUTED at 0.11; the array form must agree on both. Disagreement
    = stop and report."""
    for r_lo, want in ((0.71, "VERIFIED"), (0.11, "REFUTED")):
        cj_arr = trace(dot_harness(r_lo))
        p_arr = propagate(cj_arr)
        v_arr = make_solver_verdict(
            cj_arr, p_arr, escalate(cj_arr, p_arr, CONFIG), **VERSIONS
        )
        cj_s = trace(dot_scalar_harness(r_lo))
        p_s = propagate(cj_s)
        v_s = make_solver_verdict(
            cj_s, p_s, escalate(cj_s, p_s, CONFIG), **VERSIONS
        )
        assert v_s.status == want
        assert v_arr.status == v_s.status, (
            f"array/scalar DISAGREEMENT at r_lo={r_lo}: array "
            f"{v_arr.status} vs scalar {v_s.status} — the emission changed "
            f"an answer rather than reached one"
        )


# --- the routing measured against real jax execution --------------------------
#
# arange element values make routing readable as values: if the routing
# says output element j reads source element i, the real jax output's
# element j must BE i. One battery, every structural op, real jax.


def routed_eqn(cj, prim):
    """The one `prim` equation of a transcribed jaxpr, descending
    transparent wrappers (jnp.roll hides its slices inside a jit)."""
    found = []

    def walk(jaxpr):
        for e in jaxpr.eqns:
            if e.primitive == prim:
                found.append(e)
            for _, v in e.params:
                if hasattr(v, "jaxpr"):
                    walk(v.jaxpr)

    walk(cj.jaxpr)
    return found


@pytest.mark.parametrize(
    "fn,proto,prim",
    [
        (lambda x: lax.transpose(x, (1, 0)), np.arange(6.0).reshape(2, 3), "transpose"),
        (
            lambda x: lax.transpose(x, (2, 0, 1)),
            np.arange(24.0).reshape(2, 3, 4),
            "transpose",
        ),
        (
            lambda x: lax.broadcast_in_dim(x, (3, 2), (1,)),
            np.arange(2.0),
            "broadcast_in_dim",
        ),
        (
            lambda x: lax.broadcast_in_dim(x, (2, 3), (0, 1)),
            np.arange(2.0).reshape(2, 1),
            "broadcast_in_dim",
        ),
        (lambda x: lax.slice(x, (1,), (8,), (3,)), np.arange(10.0), "slice"),
        (
            lambda x: lax.slice(x, (0, 1), (2, 3), None),
            np.arange(12.0).reshape(3, 4),
            "slice",
        ),
        (lambda x: jnp.reshape(x, (3, 2)), np.arange(6.0).reshape(2, 3), "reshape"),
        (lambda x: jnp.squeeze(x, 0), np.arange(3.0).reshape(1, 3), "squeeze"),
        (lambda x: jnp.roll(x, -1), np.arange(4.0), "concatenate"),
        (lambda x: jnp.roll(x, 2), np.arange(5.0), "concatenate"),
    ],
)
def test_structural_routing_matches_real_jax_execution(fn, proto, prim):
    cj = transcribe(jax.make_jaxpr(fn)(proto))
    eqns = routed_eqn(cj, prim)
    assert eqns, f"trace has no {prim!r} equation"
    got = np.asarray(fn(proto)).ravel()
    for e in eqns:
        routes = _route_structural(e)
        n_in = [int(np.prod(a.aval.shape or (1,))) for a in e.invars]
        # rebuild each operand's flat values by running the PREFIX of the
        # trace is overkill here: every battery case routes the traced
        # function's own input (or roll's intermediate slices of it), so
        # operand values are recovered from the input by the same measured
        # jax ops — for the single-op cases operand 0 IS the input
        if prim != "concatenate":
            src_vals = [np.asarray(proto, dtype=float).ravel()]
            out_vals = np.array([src_vals[op][i] for op, i in routes])
            assert np.array_equal(out_vals, got), (prim, routes)


def test_roll_routing_end_to_end_matches_real_jax_values():
    # the concatenate case end to end: emission-level routing composed
    # through the jit descent equals jax's own roll
    for shift in (-1, 1, 2):
        def h(shift=shift):
            def hh():
                a = any_array((5,), "float64", (-10.0, 10.0))
                return (assert_(jnp.roll(a, shift) > -100.0),)

            return hh

        cj = trace(h())
        p = propagate(cj)
        # the assert is definitely true (roll of [-10,10] > -100): nothing
        # unknown, so pose the slice directly for the routing check
        from stelling.obligation import slice_obligation

        item = slice_obligation(cj, 0, interval_env(cj))
        assert isinstance(item, ObligationSlice), getattr(item, "reason", item)
        text = emit(item, "z3", 1000).text
        expect = np.roll(np.arange(5), shift)
        for out_el, src_el in enumerate(expect):
            assert f"(define-fun t{item.eqns[-1].outvars[0].id}_{out_el} () Bool (> x0_{src_el} (- 100.0)))" in text


def test_reduce_sum_grouping_matches_real_jax_sums():
    for shape, axes in (((2, 3, 4), (0, 2)), ((2, 3), (1,)), ((4,), (0,)), ((2, 2), (0, 1))):
        proto = np.arange(float(np.prod(shape))).reshape(shape)
        fn = lambda x, axes=axes: jnp.sum(x, axis=axes)
        cj = transcribe(jax.make_jaxpr(fn)(proto))
        (e,) = routed_eqn(cj, "reduce_sum")
        groups = _group_reduce_sum(e)
        got = np.asarray(fn(proto)).ravel()
        mine = np.array([sum(proto.ravel()[i] for i in g) for g in groups])
        assert np.array_equal(mine, got), (shape, axes)


def test_measured_zero_size_and_scalar_selector_semantics():
    # the sources of the zero-dep pins, measured here: the empty universal
    # claim is True; the empty sum is 0.0; jnp.where(scalar, arr, arr)
    # traces to select_n with a SCALAR selector inside a transparent jit
    assert bool(jnp.all(jnp.array([]) > 0.0)) is True
    assert float(jnp.sum(jnp.array([]))) == 0.0
    cj = transcribe(
        jax.make_jaxpr(lambda p, x, y: jnp.where(p, x, y))(
            True, np.arange(3.0), np.arange(3.0)
        )
    )
    (sel,) = routed_eqn(cj, "select_n")
    assert tuple(sel.invars[0].aval.shape) == ()
    assert tuple(sel.invars[1].aval.shape) == (3,)


def test_where_scalar_selector_emits_shared_through_the_descent():
    def h():
        c = any_array((), "float64", (0.0, 1.0))
        x = any_array((2,), "float64", (0.0, 1.0))
        picked = jnp.where(c > 0.5, x, -x)
        return (assert_(picked > -2.0),)

    cj = trace(h)
    from stelling.obligation import slice_obligation

    item = slice_obligation(cj, 0, interval_env(cj))
    assert isinstance(item, ObligationSlice), getattr(item, "reason", item)
    text = emit(item, "z3", 1000).text
    # ONE selector definition, referenced by both element ites
    sel_defs = [
        ln for ln in text.splitlines() if "() Bool (> " in ln and "x0" in ln
    ]
    assert len(sel_defs) == 1
    assert text.count("(ite t") == 2


# --- fix round 2 (re-attack R1): declaration-time refusal ---------------------


def test_negative_dim_declaration_is_refused_at_declaration_time():
    # measured on this jax: no concrete context constructs the shape, so
    # the declared set is empty — the (inf,inf)-sibling refusal posture
    with pytest.raises(Exception, match="nonnegative"):
        jnp.zeros((-2, -2))  # the measured ground truth, pinned
    with pytest.raises(ValueError, match="negative extent"):
        any_array((-2, -2), "float64", (0.25, 1.0))
    with pytest.raises(ValueError, match="negative extent"):
        trace(lambda: (assert_(jnp.sum(any_array((-2, -2), "float64", (0.25, 1.0))) <= 0.5),))
    # zero-size declarations stay LEGAL (jax constructs them; the vacuous
    # discharge convention for (0,) is measured and pinned elsewhere)
    cj = trace(lambda: (assert_(any_array((0,), "float64", (0.0, 1.0)) > 0.0),))
    p = propagate(cj)
    assert [o.status for o in p.obligations] == ["discharged"]


def test_negative_dim_pytree_leaf_is_refused_through_the_same_gate():
    # any_pytree routes every array leaf through any_array: same refusal
    from stelling.harness import any_pytree

    class Proto:
        shape = (-2, -2)
        dtype = np.dtype("float64")

    def h():
        t = any_pytree({"a": Proto()}, (0.25, 1.0))
        return (assert_(jnp.sum(t["a"]) <= 0.5),)

    with pytest.raises(ValueError, match="negative extent"):
        trace(h)
